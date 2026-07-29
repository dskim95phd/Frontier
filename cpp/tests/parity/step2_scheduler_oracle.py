"""Normalized Step 2 oracle backed by Frontier's production vLLM V1 scheduler.

The event driver is intentionally small and fixed-latency. Scheduling,
Request, and Batch transitions are delegated to the production Python
classes; this module only normalizes their semantic milestones to the C++
schema v2 contract.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import csv
from dataclasses import dataclass, field
import heapq
import io
import json
import logging
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from frontier.entities import Batch, Request
from frontier.scheduler.replica_scheduler import (
    vllm_v1_engine_replica_scheduler as scheduler_module,
)
from frontier.scheduler.replica_scheduler.vllm_v1_engine_replica_scheduler import (
    VLLMv1EngineReplicaScheduler,
)
from frontier.types import ClusterType


class Step2OracleError(RuntimeError):
    pass


@dataclass(order=True)
class _Event:
    time_s: float
    sequence: int
    event_type: str = field(compare=False)
    request_id: int | None = field(default=None, compare=False)
    batch_id: int | None = field(default=None, compare=False)
    generation: int | None = field(default=None, compare=False)


def _read_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 2:
        raise Step2OracleError("Step 2 oracle requires schema_version=2")
    if config.get("system_architecture") != "co-location":
        raise Step2OracleError("Step 2 oracle supports only co-location")
    if config.get("enable_parallel_clusters") is not False:
        raise Step2OracleError("Step 2 oracle requires sequential execution")
    if config.get("prefix_cache") != {
        "enabled": False,
        "key_mode": "session",
    }:
        raise Step2OracleError("Step 2 oracle requires prefix cache disabled")
    scheduler = config.get("scheduler", {})
    if scheduler.get("type") != "vllm_v1":
        raise Step2OracleError("Step 2 oracle requires vllm_v1")
    if scheduler.get("scheduling_policy") != "fcfs":
        raise Step2OracleError("Step 2 oracle requires FCFS")
    execution = config.get("execution_model", {})
    if execution.get("type") != "fixed":
        raise Step2OracleError(
            "scheduler probe supports fixed execution only"
        )
    latency = execution.get("batch_latency_ms")
    if (
        isinstance(latency, bool)
        or not isinstance(latency, (int, float))
        or not math.isfinite(float(latency))
        or float(latency) < 0.0
    ):
        raise Step2OracleError("fixed batch latency is invalid")
    return config


def _read_workload(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "arrived_at",
            "num_prefill_tokens",
            "num_decode_tokens",
        }
        if reader.fieldnames is None or not required.issubset(
            reader.fieldnames
        ):
            raise Step2OracleError("normalized workload columns are missing")
        rows: list[dict[str, Any]] = []
        for request_id, row in enumerate(reader):
            arrived_at = float(row["arrived_at"])
            prefill = int(row["num_prefill_tokens"])
            decode = int(row["num_decode_tokens"])
            if (
                not math.isfinite(arrived_at)
                or arrived_at < 0.0
                or prefill <= 0
                or decode <= 0
            ):
                raise Step2OracleError("invalid normalized workload row")
            session_text = row.get("session_id", "")
            rows.append(
                {
                    "request_id": request_id,
                    "arrived_at": arrived_at,
                    "num_prefill_tokens": prefill,
                    "num_decode_tokens": decode,
                    "session_id": (
                        int(session_text) if session_text else None
                    ),
                }
            )
    return rows


def _minimal_scheduler(
    config: dict[str, Any],
    decision_sink: list[dict[str, Any]],
) -> VLLMv1EngineReplicaScheduler:
    """Construct the production scheduler without unrelated cluster plumbing."""

    scheduler_config = config["scheduler"]
    scheduler = VLLMv1EngineReplicaScheduler.__new__(
        VLLMv1EngineReplicaScheduler
    )
    scheduler._config = SimpleNamespace(
        **scheduler_config,
        enable_prefix_caching=False,
        decode_cuda_graph_mode="none",
        enable_phase_aware_thinking_profile=False,
        enable_final_round_priority_boost=False,
        final_round_priority_value=-1,
        final_prefill_reserved_slots=0,
        final_prefill_reserved_tokens=0,
        final_decode_reserved_slots=0,
        enable_final_running_request_reclaim=False,
    )
    scheduler._replica_config = SimpleNamespace(
        num_pipeline_stages=1,
        speculative_decoding_config=None,
    )
    scheduler._request_generator_config = SimpleNamespace(max_tokens=1 << 30)
    scheduler._replica_id = 0
    scheduler._dp_id = 0
    scheduler._replica_is_moe = False
    scheduler._num_stages = 1
    scheduler._cluster_type = ClusterType.MONOLITHIC
    scheduler._cluster_scheduler = None
    scheduler._cpu_kv_cache_config = None

    scheduler._request_queue = []
    scheduler._running_requests = []
    scheduler._preempted_requests = []
    scheduler._waiting_requests = []
    scheduler._pending_kv_transfer_requests = set()
    scheduler._pending_prefill_export_kinds = {}
    scheduler._pending_cpu_offload_operations = {}
    scheduler._pending_cpu_restore_operations = {}
    scheduler._cpu_offload_generation_by_session = {}
    scheduler._scheduled_num_computed_tokens_by_request = {}
    scheduler._allocation_map = {}
    scheduler._num_allocated_blocks = 0
    scheduler._batch_creation_counter = 0
    scheduler._decode_sync_batch_creation_counter = 0
    scheduler._num_running_batches = 0
    scheduler._current_schedule_time = 0.0
    scheduler._continuation_request_ids = set()

    scheduler._max_num_running_reqs = int(
        scheduler_config["batch_size_cap"]
    )
    scheduler._max_num_scheduled_tokens = int(
        scheduler_config["max_tokens_in_batch"]
    )
    scheduler._scheduling_policy = "fcfs"
    scheduler._enable_preemption = bool(
        scheduler_config["enable_preemption"]
    )
    scheduler._enable_chunked_prefill = bool(
        scheduler_config["enable_chunked_prefill"]
    )
    scheduler._enable_phase_aware_thinking_profile = False
    scheduler._enable_final_round_priority_boost = False
    scheduler._final_round_priority_value = -1
    scheduler._final_prefill_reserved_slots = 0
    scheduler._final_prefill_reserved_tokens = 0
    scheduler._final_decode_reserved_slots = 0
    scheduler._enable_final_running_request_reclaim = False
    scheduler._active_iteration_round_class = None
    scheduler._long_prefill_token_threshold = int(
        scheduler_config["long_prefill_token_threshold"]
    )
    scheduler._watermark_blocks = int(
        float(scheduler_config["watermark_blocks_fraction"])
        * int(scheduler_config["num_blocks"])
    )
    scheduler._max_model_len = 1 << 30

    scheduler._spec_decode_config = None
    scheduler._spec_decode_enabled = False
    scheduler._spec_method_uses_lookahead_slots = False
    scheduler._kv_cache_manager = None
    scheduler._cpu_kv_cache_manager = None
    scheduler._cpu_kv_cache_transfer_engine = None
    scheduler._cpu_restore_waiting_requests = {}
    scheduler._staged_cpu_restore_payloads = {}
    scheduler._pending_auxiliary_events = []

    scheduler._schedule_iteration_id = 0
    scheduler._active_schedule_iteration_id = -1
    scheduler._current_iteration_token_budget = 0
    scheduler._prefill_iteration_reserved_slots_remaining = 0
    scheduler._prefill_iteration_reserved_tokens_remaining = 0
    scheduler._decode_iteration_reserved_slots_remaining = 0

    scheduler._monolithic_pp_pending_terminal_release_iters = {}
    scheduler._monolithic_pp_waiting_sensitive_release_extensions = set()
    scheduler._monolithic_pp_terminal_release_followup_poll_pending = False
    scheduler._monolithic_pp_mtp_output_wait_request_ids = set()
    scheduler._monolithic_pp_mtp_output_wait_remaining_iters = {}
    scheduler._monolithic_pp_mtp_near_full_prefill_request_ids = set()
    scheduler._monolithic_pp_mtp_single_output_wait_request_ids = set()
    scheduler._monolithic_pp_mtp_fractional_output_wait_counts = {}
    scheduler._monolithic_pp_mtp_output_wait_followup_poll_pending = False
    scheduler._monolithic_pp_waiting_admission_delay_iters = {}
    scheduler._active_batch_request_counts = {}

    def create_batch(
        requests: list[Request],
        num_tokens: list[int],
    ) -> Batch:
        batch = Batch(0, requests, num_tokens, is_moe=False)
        batch.set_global_id(scheduler._batch_creation_counter)
        scheduler._batch_creation_counter += 1
        return batch

    scheduler._create_batch = create_batch
    scheduler_module._frontier_vllm_v1_sched_decision_logger = object()
    scheduler_module._log_frontier_vllm_v1_schedule_decision = (
        lambda payload: decision_sink.append(dict(payload))
    )
    return scheduler


def _group_scheduler_trace(
    raw_decisions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    order: list[int] = []
    for raw in raw_decisions:
        iteration_id = int(raw["iteration_id"])
        if iteration_id not in grouped:
            grouped[iteration_id] = []
            order.append(iteration_id)
        grouped[iteration_id].append(raw)

    traces: list[dict[str, Any]] = []
    for iteration_id in order:
        rows = grouped[iteration_id]
        start = rows[0]
        end = rows[-1]
        decisions = [
            {
                "decision_result": row["decision_result"],
                "request_id": int(row["request_id"]),
                "num_tokens": int(row["num_tokens"]),
                "token_budget_after": int(row["token_budget"]),
                "available_blocks_after": int(row["available_blocks"]),
            }
            for row in rows
            if row["event"] == "decision"
        ]
        traces.append(
            {
                "iteration_id": iteration_id,
                "simulation_time_s": float(start["simulation_time"]),
                "decisions": decisions,
                "token_budget_before": int(
                    start["max_num_scheduled_tokens"]
                ),
                "token_budget_after": int(end["token_budget"]),
                "available_blocks_before": int(
                    start["available_blocks"]
                ),
                "available_blocks_after": int(end["available_blocks"]),
                "waiting_count_before": int(start["num_waiting_reqs"]),
                "waiting_count_after": int(end["num_waiting_reqs"]),
                "running_count_before": int(start["num_running_reqs"]),
                "running_count_after": int(end["num_running_reqs"]),
                "preempted_count": sum(
                    row.get("decision_result") == "PREEMPTED"
                    for row in rows
                ),
                "batch_request_ids": [
                    int(value) for value in end["batch_request_ids"]
                ],
                "request_num_tokens": [
                    int(value) for value in end["request_num_tokens"]
                ],
            }
        )
    return traces


def run_scheduler_oracle(
    config: dict[str, Any],
    workload: list[dict[str, Any]],
) -> dict[str, Any]:
    Request._id = -1
    Batch._id = -1
    raw_decisions: list[dict[str, Any]] = []
    scheduler = _minimal_scheduler(config, raw_decisions)
    requests = [
        Request(
            arrived_at=row["arrived_at"],
            num_prefill_tokens=row["num_prefill_tokens"],
            num_decode_tokens=row["num_decode_tokens"],
            session_id=row["session_id"],
        )
        for row in workload
    ]
    if [request.id for request in requests] != list(range(len(requests))):
        raise Step2OracleError("production Request IDs are not normalized")

    queue: list[_Event] = []
    next_sequence = 1

    def push(
        time_s: float,
        event_type: str,
        *,
        request_id: int | None = None,
        batch_id: int | None = None,
        generation: int | None = None,
    ) -> None:
        nonlocal next_sequence
        heapq.heappush(
            queue,
            _Event(
                time_s,
                next_sequence,
                event_type,
                request_id,
                batch_id,
                generation,
            ),
        )
        next_sequence += 1

    for row in workload:
        push(
            row["arrived_at"],
            "request_arrival",
            request_id=row["request_id"],
        )

    latency_ms = float(
        config["execution_model"]["batch_latency_ms"]
    )
    batches: list[Batch] = []
    batch_snapshots: list[dict[str, Any]] = []
    event_trace: list[dict[str, Any]] = []
    completed_order: list[int] = []
    completed_seen: set[int] = set()
    in_flight = False

    while queue:
        event = heapq.heappop(queue)
        normalized_event: dict[str, Any] = {
            "time_s": event.time_s,
            "sequence": event.sequence,
            "type": event.event_type,
        }
        if event.request_id is not None:
            normalized_event["request_id"] = event.request_id
        if event.batch_id is not None:
            normalized_event["batch_id"] = event.batch_id
        if event.generation is not None:
            normalized_event["generation"] = event.generation
        event_trace.append(normalized_event)

        if event.event_type == "request_arrival":
            request = requests[event.request_id]
            request.on_arrival(event.time_s, ClusterType.MONOLITHIC)
            scheduler.add_request(request)
            push(event.time_s, "scheduler_poll")
            continue

        if event.event_type == "scheduler_poll":
            if in_flight or (
                not scheduler._request_queue
                and not scheduler._preempted_requests
                and not scheduler._running_requests
            ):
                continue
            scheduler._current_schedule_time = event.time_s
            batch = scheduler._get_next_batch()
            if batch is None:
                continue
            batch.on_schedule(event.time_s, ClusterType.MONOLITHIC)
            scheduler._num_running_batches += 1
            in_flight = True
            batch_id = len(batches)
            batches.append(batch)
            snapshots = [
                {
                    "request_id": request.id,
                    "scheduled_tokens": int(tokens),
                    "processed_tokens": int(request.num_processed_tokens),
                    "prefill_tokens": int(request.num_prefill_tokens),
                }
                for request, tokens in zip(
                    batch.requests,
                    batch.num_tokens,
                )
            ]
            batch_snapshots.append(
                {
                    "scheduled_at_s": event.time_s,
                    "requests": snapshots,
                }
            )
            push(
                event.time_s + latency_ms / 1e3,
                "batch_completion",
                batch_id=batch_id,
                generation=batch_id + 1,
            )
            continue

        if event.event_type != "batch_completion":
            raise Step2OracleError(f"unexpected event {event.event_type}")
        batch = batches[event.batch_id]
        batch.on_batch_end(event.time_s, ClusterType.MONOLITHIC)
        scheduler.on_batch_end(batch)
        in_flight = False
        batch_snapshots[event.batch_id]["completed_at_s"] = event.time_s
        for request in batch.requests:
            if request.completed and request.id not in completed_seen:
                completed_seen.add(request.id)
                completed_order.append(request.id)
        push(event.time_s, "scheduler_poll")

    if len(completed_order) != len(requests):
        raise Step2OracleError(
            "scheduler quiesced with incomplete requests: "
            f"{len(completed_order)}/{len(requests)}"
        )

    request_rows = []
    for request_id in completed_order:
        request = requests[request_id]
        scheduled_at = request._scheduled_at[ClusterType.MONOLITHIC][0]
        request_rows.append(
            {
                "request_id": request.id,
                "arrived_at_s": request.arrived_at,
                "first_scheduled_at_s": scheduled_at,
                "prefill_completed_at_s": request.prefill_completed_at,
                "first_token_completed_at_s": (
                    request.first_decode_token_completed_at
                ),
                "completed_at_s": request.completed_at,
                "scheduling_delay_ms": (
                    scheduled_at - request.arrived_at
                )
                * 1e3,
                "ttft_ms": (
                    request.prefill_completed_at - request.arrived_at
                )
                * 1e3,
                "e2e_ms": (
                    request.completed_at - request.arrived_at
                )
                * 1e3,
                "num_processed_tokens": request.num_processed_tokens,
                "preemption_count": request.get_total_preemption_count(),
                "tokens_at_preemption": [],
            }
        )

    batch_rows = []
    for batch_id, snapshot in enumerate(batch_snapshots):
        request_ids = [
            row["request_id"] for row in snapshot["requests"]
        ]
        scheduled_tokens = [
            row["scheduled_tokens"] for row in snapshot["requests"]
        ]
        prefill_tokens = sum(
            row["scheduled_tokens"]
            for row in snapshot["requests"]
            if row["processed_tokens"] < row["prefill_tokens"]
        )
        batch_rows.append(
            {
                "batch_id": batch_id,
                "iteration_id": batch_id,
                "scheduled_at_s": snapshot["scheduled_at_s"],
                "completed_at_s": snapshot["completed_at_s"],
                "request_ids": request_ids,
                "scheduled_tokens": scheduled_tokens,
                "total_scheduled_tokens": sum(scheduled_tokens),
                "num_prefill_tokens": prefill_tokens,
                "num_decode_tokens": sum(scheduled_tokens)
                - prefill_tokens,
                "predicted_execution_ms": latency_ms,
            }
        )

    return {
        "schema_version": 2,
        "run": {
            "run_id": config["run_id"],
            "simulation_mode": config["simulation_mode"],
            "system_architecture": config["system_architecture"],
            "timestamp_unit": "seconds",
            "latency_unit": "milliseconds",
            "metrics_semantics": "canonical",
        },
        "completed_request_ids": completed_order,
        "requests": request_rows,
        "batches": batch_rows,
        "scheduler_trace": _group_scheduler_trace(raw_decisions),
        "event_trace": event_trace,
        "analytical_diagnostics": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--workload", required=True, type=Path)
    arguments = parser.parse_args()
    logging.disable(logging.CRITICAL)
    with redirect_stdout(io.StringIO()):
        output = run_scheduler_oracle(
            _read_config(arguments.config),
            _read_workload(arguments.workload),
        )
    print(json.dumps(output, indent=2, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
