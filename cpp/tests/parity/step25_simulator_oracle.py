"""Co-location oracle backed by Frontier's production Python Simulator.

The adapter translates the normalized C++ cluster schema into Python dataclasses and
normalizes the production event, routing, batch-stage, and request results.
Routing, batching, pipeline execution, and request mutation are never
reimplemented here.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
import json
import logging
import math
from pathlib import Path
import sys
import tempfile
from types import MethodType
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.disable(logging.CRITICAL)

from frontier.cc_backend.cc_backend_config import AnalyticalCCBackendConfig
from frontier.config import (
    AnalyticalRooflineExecutionTimePredictorConfig,
    ClusterConfig,
    MetricsConfig,
    ReplicaConfig,
    SimulationConfig,
    TraceRequestGeneratorConfig,
    VllmV1SchedulerConfig,
)
from frontier.entities import Batch, BatchStage, ExecutionTime, Request
from frontier.events import BaseEvent
from frontier.simulator import Simulator
from frontier.scheduler.replica_scheduler import (
    vllm_v1_engine_replica_scheduler as scheduler_module,
)
from frontier.types import ClusterType


class Step25OracleError(RuntimeError):
    pass


def _read_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise Step25OracleError("schema_version=1 is required")
    if config.get("system_architecture") != "co-location":
        raise Step25OracleError("co-location is required")
    if config.get("enable_parallel_clusters") is not False:
        raise Step25OracleError("parallel clusters must remain disabled")
    if config.get("cluster_scheduler") != {"type": "round_robin"}:
        raise Step25OracleError("round_robin cluster scheduling is required")
    return config


def _build_python_config(
    config: dict[str, Any],
    workload_path: Path,
    output_dir: str,
) -> SimulationConfig:
    cluster_config = config["clusters"]["monolithic"]
    topology = cluster_config["parallelism"]
    scheduler = cluster_config["scheduler"]
    execution = cluster_config["execution_model"]
    routing = cluster_config.get(
        "moe_routing",
        {
            "mode": "simulation",
            "distribution": "balanced",
            "seed": 42,
        },
    )
    model_name = str(
        cluster_config.get("model_name", "meta-llama/Llama-2-7b-hf")
    )
    replica_scheduler = VllmV1SchedulerConfig(
        batch_size_cap=int(scheduler["batch_size_cap"]),
        block_size=int(scheduler["block_size"]),
        watermark_blocks_fraction=float(
            scheduler["watermark_blocks_fraction"]
        ),
        max_tokens_in_batch=int(scheduler["max_tokens_in_batch"]),
        scheduling_policy=str(scheduler["scheduling_policy"]),
        enable_preemption=bool(scheduler["enable_preemption"]),
        enable_chunked_prefill=bool(
            scheduler["enable_chunked_prefill"]
        ),
        enable_prefix_caching=False,
        prefix_caching_key_mode="session",
        num_preallocate_tokens=int(
            scheduler["num_preallocate_tokens"]
        ),
        long_prefill_token_threshold=int(
            scheduler["long_prefill_token_threshold"]
        ),
        num_blocks=int(scheduler["num_blocks"]),
        num_blocks_mode="explicit",
    )
    replica = ReplicaConfig(
        model_name=model_name,
        device=str(execution.get("device", "rubin")),
        network_device="vera_rubin_nvl72_domain",
        attn_tensor_parallel_size=int(
            topology["tensor_parallel_size"]
        ),
        attn_data_parallel_size=int(
            topology["data_parallel_size"]
        ),
        num_pipeline_stages=int(
            topology["pipeline_parallel_size"]
        ),
        data_parallel_size=int(topology["data_parallel_size"]),
        moe_tensor_parallel_size=int(
            topology.get("moe_tensor_parallel_size", 1)
        ),
        moe_expert_parallel_size=int(
            topology.get("moe_expert_parallel_size", 1)
        ),
        total_expert_num=int(
            cluster_config.get("total_expert_num", 1)
        ),
        router_topk=int(cluster_config.get("router_topk", 1)),
        moe_routing_mode=str(routing.get("mode", "simulation")),
        moe_routing_distribution_type=str(
            routing.get("distribution", "balanced")
        ),
        moe_routing_seed=int(routing.get("seed", 42)),
    )
    predictor = AnalyticalRooflineExecutionTimePredictorConfig()
    cluster = ClusterConfig(
        replica_scheduler_config=replica_scheduler,
        execution_time_predictor_config=predictor,
        cc_backend_config=AnalyticalCCBackendConfig(
            network_bandwidth_gbps=float(
                execution.get("network_bandwidth_gbps", 400.0)
            ),
            network_latency_us=float(
                execution.get("network_latency_us", 1.0)
            ),
            intra_node_bandwidth_gbps=float(
                execution.get("intra_node_bandwidth_gbps", 14400.0)
            ),
        ),
        num_replicas=int(topology["num_replicas"]),
        replica_config=replica,
    )
    request_generator = TraceRequestGeneratorConfig(
        trace_file=str(workload_path),
        prefill_scale_factor=1.0,
        decode_scale_factor=1.0,
        time_scale_factor=1.0,
        max_tokens=1 << 30,
    )
    metrics = MetricsConfig(
        write_metrics=True,
        write_json_trace=True,
        enable_chrome_trace=False,
        store_plots=False,
        output_dir=output_dir,
        run_id="step25-production-oracle",
    )
    return SimulationConfig(
        simulation_mode=str(config["simulation_mode"]),
        sys_arch="co-location",
        decode_cuda_graph_mode="none",
        cluster_config=cluster,
        request_generator_config=request_generator,
        metrics_config=metrics,
        enable_parallel_clusters=False,
        log_level="critical",
    )


def _fixed_execution_time(
    latency_ms: float,
    num_layers: int,
) -> ExecutionTime:
    if not math.isfinite(latency_ms) or latency_ms < 0.0:
        raise Step25OracleError("fixed stage latency is invalid")
    return ExecutionTime(
        num_layers_per_pipeline_stage=num_layers,
        attention_rope_execution_time=0.0,
        attention_kv_cache_save_execution_time=0.0,
        attention_decode_execution_time=0.0,
        attention_prefill_execution_time=0.0,
        attention_layer_pre_proj_execution_time=0.0,
        attention_layer_post_proj_execution_time=0.0,
        attn_norm_time=0.0,
        mlp_norm_time=0.0,
        add_time=0.0,
        tensor_parallel_communication_time=0.0,
        pipeline_parallel_communication_time=latency_ms,
        expert_parallel_communication_time=0.0,
        moe_gating_time=0.0,
        moe_shuffling_time=0.0,
        schedule_time=0.0,
        sampler_e2e_time=0.0,
        prepare_inputs_e2e_time=0.0,
        process_model_outputs_time=0.0,
        ray_comm_time=0.0,
        is_moe=False,
        mlp_layer_up_proj_execution_time=0.0,
        mlp_layer_down_proj_execution_time=0.0,
        mlp_layer_act_execution_time=0.0,
        add_attn_residual_time=0.0,
        add_ffn_residual_time=0.0,
        attn_tensor_parallel_allreduce_time=0.0,
        moe_tensor_parallel_allreduce_time=0.0,
    )


def _install_fixed_predictor(
    simulator: Simulator,
    stage_latencies_ms: list[float],
) -> None:
    predictor = simulator._predictors[ClusterType.MONOLITHIC]
    expected_stages = int(
        predictor._replica_config.num_pipeline_stages
    )
    if len(stage_latencies_ms) != expected_stages:
        raise Step25OracleError(
            "stage_latencies_ms length does not match Python PP"
        )

    def predict_stage_execution_time(
        self: Any,
        batch: Batch,
        pipeline_stage: int,
        **_: Any,
    ) -> ExecutionTime:
        del batch
        return _fixed_execution_time(
            float(stage_latencies_ms[pipeline_stage]),
            int(self._num_layers_per_pipeline_stage),
        )

    predictor.predict_stage_execution_time = MethodType(
        predict_stage_execution_time,
        predictor,
    )


def _capture_predictions(
    simulator: Simulator,
) -> dict[tuple[int, int], ExecutionTime]:
    predictor = simulator._predictors[ClusterType.MONOLITHIC]
    original = predictor.predict_stage_execution_time
    captured: dict[tuple[int, int], ExecutionTime] = {}

    def predict_stage_execution_time(
        self: Any,
        batch: Batch,
        pipeline_stage: int,
        *args: Any,
        **kwargs: Any,
    ) -> ExecutionTime:
        execution_time = original(
            batch,
            pipeline_stage,
            *args,
            **kwargs,
        )
        key = (int(batch.id), int(pipeline_stage))
        if key in captured and not self._model_config.is_moe:
            raise Step25OracleError(
                f"stage prediction repeated for {key}"
            )
        captured[key] = execution_time
        return execution_time

    predictor.predict_stage_execution_time = MethodType(
        predict_stage_execution_time,
        predictor,
    )
    return captured


_EVENT_NAMES = {
    "REQUEST_ARRIVAL": "request_arrival",
    "GLOBAL_SCHEDULE": "global_schedule",
    "CLUSTER_SCHEDULE": "cluster_schedule",
    "REPLICA_SCHEDULE": "replica_schedule",
    "BATCH_STAGE_ARRIVAL": "batch_stage_arrival",
    "REPLICA_STAGE_SCHEDULE": "replica_stage_schedule",
    "BATCH_STAGE_END": "batch_stage_end",
    "CLUSTER_BATCH_END": "cluster_batch_end",
    "GLOBAL_BATCH_END": "global_batch_end",
    "PREFILL_SYNC": "prefill_sync",
    "PREFILL_SYNC_COLLECTIVE": "prefill_sync_collective",
    "DECODE_SYNC": "decode_sync",
    "DECODE_SYNC_COLLECTIVE": "decode_sync_collective",
}


def _event_name(value: Any) -> str:
    text = str(value)
    text = text.rsplit(".", 1)[-1]
    if text in _EVENT_NAMES.values():
        return text
    text = text.upper()
    if text not in _EVENT_NAMES:
        raise Step25OracleError(f"unexpected production event: {text}")
    return _EVENT_NAMES[text]


def _normalize_events(
    raw_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for sequence, raw in enumerate(raw_events, start=1):
        event: dict[str, Any] = {
            "sequence": sequence,
            "time_s": float(raw["time"]),
            "type": _event_name(raw["event_type"]),
        }
        event_type = event["type"]
        if event_type == "request_arrival":
            event["request_id"] = int(raw["request_id"])
        if event_type in {
            "replica_schedule",
            "batch_stage_arrival",
            "replica_stage_schedule",
            "batch_stage_end",
            "prefill_sync",
            "prefill_sync_collective",
            "decode_sync",
            "decode_sync_collective",
        }:
            event["replica_id"] = int(raw["replica_id"])
            if "dp_id" in raw:
                event["dp_id"] = int(raw["dp_id"])
        if event_type in {
            "batch_stage_arrival",
            "batch_stage_end",
        }:
            event["batch_id"] = int(raw["batch_id"])
        if event_type in {
            "batch_stage_arrival",
            "replica_stage_schedule",
            "batch_stage_end",
            "prefill_sync",
            "prefill_sync_collective",
            "decode_sync",
            "decode_sync_collective",
        }:
            event["stage_id"] = int(raw["stage_id"])
        if event_type in {
            "prefill_sync",
            "prefill_sync_collective",
            "decode_sync",
            "decode_sync_collective",
        }:
            event["sync_stage"] = str(raw["sync_stage"])
            event["layer_id"] = int(raw["layer_id"])
        if event_type in {
            "prefill_sync",
            "decode_sync",
        }:
            event["batch_id"] = int(raw["batch_id"])
        if event_type in {
            "prefill_sync_collective",
            "decode_sync_collective",
        }:
            event["sync_group_id"] = int(raw["batch_global_id"])
        result.append(event)
    return result


def _request_owners(
    raw_events: list[dict[str, Any]],
) -> list[dict[str, int]]:
    owners: dict[int, tuple[int, int]] = {}
    for event in raw_events:
        if _event_name(event["event_type"]) != "cluster_schedule":
            continue
        for replica_id, dp_id, request_id in event["request_mapping"]:
            if request_id is None:
                continue
            owner = (int(replica_id), int(dp_id))
            old = owners.setdefault(int(request_id), owner)
            if old != owner:
                raise Step25OracleError(
                    "production scheduler routed a request twice"
                )
    return [
        {
            "request_id": request_id,
            "replica_id": owner[0],
            "dp_id": owner[1],
        }
        for request_id, owner in sorted(owners.items())
    ]


def _normalize_stages(
    simulator: Simulator,
    predictions: dict[tuple[int, int], ExecutionTime],
    *,
    include_components: bool,
) -> list[dict[str, Any]]:
    rows = simulator._metric_store._frontier_stage_batch_ledger_rows
    result: list[dict[str, Any]] = []
    for row in rows:
        key = (int(row["batch_id"]), int(row["stage_id"]))
        prediction = predictions[key]
        stage = {
            "batch_id": int(row["batch_id"]),
            "replica_id": int(row["replica_id"]),
            "dp_id": int(row["dp_id"]),
            "stage_id": int(row["stage_id"]),
            "request_ids": [int(value) for value in row["request_ids"]],
            "request_num_tokens": [
                int(value) for value in row["request_num_tokens"]
            ],
            "started_at_s": float(row["stage_start_ts"]),
            "completed_at_s": float(
                row.get(
                    "stage_completion_observed_ts",
                    row["stage_end_ts"],
                )
            ),
            "duration_ms": float(
                row.get(
                    "observed_stage_duration_ms",
                    (
                        float(row["stage_end_ts"])
                        - float(row["stage_start_ts"])
                    )
                    * 1e3,
                )
            ),
        }
        if include_components:
            tp_ms = float(
                prediction.attention_all_reduce_time
                + prediction.mlp_all_reduce_time
            )
            pp_ms = float(
                prediction.pipeline_parallel_communication_time
            )
            stage.update(
                {
                    "dense_compute_ms": float(
                        prediction.total_time * 1e3
                        - tp_ms
                        - pp_ms
                    ),
                    "tp_communication_ms": tp_ms,
                    "pp_communication_ms": pp_ms,
                }
            )
        result.append(stage)
    return result


def _normalize_requests(
    simulator: Simulator,
    owners: list[dict[str, int]],
) -> list[dict[str, Any]]:
    owner_by_request = {
        row["request_id"]: row for row in owners
    }
    result: list[dict[str, Any]] = []
    for request in simulator._all_requests:
        owner = owner_by_request[int(request.id)]
        result.append(
            {
                "request_id": int(request.id),
                "replica_id": owner["replica_id"],
                "dp_id": owner["dp_id"],
                "arrived_at_s": float(request.arrived_at),
                "first_scheduled_at_s": float(request.scheduled_at),
                "prefill_completed_at_s": float(
                    request.prefill_completed_at
                ),
                "completed_at_s": float(request.completed_at),
                "num_processed_tokens": int(
                    request.num_processed_tokens
                ),
                "preemption_count": int(
                    request.get_total_preemption_count()
                ),
            }
        )
    return sorted(result, key=lambda row: row["request_id"])


def _normalize_scheduler_trace(
    raw_decisions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    for row in raw_decisions:
        if row["event"] == "iteration_start":
            if current:
                raise Step25OracleError(
                    "scheduler decision iterations overlap"
                )
            current = [row]
            continue
        if not current:
            raise Step25OracleError(
                "scheduler decision lacks iteration_start"
            )
        current.append(row)
        if row["event"] != "iteration_end":
            continue
        start = current[0]
        end = current[-1]
        for item in current:
            if (
                item["replica_id"] != start["replica_id"]
                or item["dp_id"] != start["dp_id"]
                or item["iteration_id"] != start["iteration_id"]
            ):
                raise Step25OracleError(
                    "scheduler decision target/iteration changed mid-trace"
                )
        traces.append(
            {
                "iteration_id": int(start["iteration_id"]),
                "simulation_time_s": float(
                    start["simulation_time"]
                ),
                "decisions": [
                    {
                        "decision_result": str(
                            item["decision_result"]
                        ),
                        "request_id": int(item["request_id"]),
                        "num_tokens": int(item["num_tokens"]),
                        "token_budget_after": int(
                            item["token_budget"]
                        ),
                        "available_blocks_after": int(
                            item["available_blocks"]
                        ),
                    }
                    for item in current
                    if item["event"] == "decision"
                ],
                "token_budget_before": int(
                    start["max_num_scheduled_tokens"]
                ),
                "token_budget_after": int(end["token_budget"]),
                "available_blocks_before": int(
                    start["available_blocks"]
                ),
                "available_blocks_after": int(
                    end["available_blocks"]
                ),
                "waiting_count_before": int(
                    start["num_waiting_reqs"]
                ),
                "waiting_count_after": int(
                    end["num_waiting_reqs"]
                ),
                "running_count_before": int(
                    start["num_running_reqs"]
                ),
                "running_count_after": int(
                    end["num_running_reqs"]
                ),
                "preempted_count": sum(
                    item.get("decision_result") == "PREEMPTED"
                    for item in current
                ),
                "batch_request_ids": [
                    int(value)
                    for value in end["batch_request_ids"]
                ],
                "request_num_tokens": [
                    int(value)
                    for value in end["request_num_tokens"]
                ],
                "replica_id": int(start["replica_id"]),
                "dp_id": int(start["dp_id"]),
            }
        )
        current = []
    if current:
        raise Step25OracleError(
            "scheduler decision trace lacks iteration_end"
        )
    return traces


def run_oracle(
    config_path: Path,
    workload_path: Path,
) -> dict[str, Any]:
    config = _read_config(config_path)
    execution = config["clusters"]["monolithic"]["execution_model"]
    Request._id = -1
    Batch._id = -1
    BatchStage._id = -1
    BaseEvent._id = 0
    raw_scheduler_decisions: list[dict[str, Any]] = []
    scheduler_module._frontier_vllm_v1_sched_decision_logger = object()
    scheduler_module._log_frontier_vllm_v1_schedule_decision = (
        lambda payload: raw_scheduler_decisions.append(dict(payload))
    )
    with tempfile.TemporaryDirectory(
        prefix="frontier-step25-oracle-"
    ) as output_dir:
        with redirect_stdout(io.StringIO()):
            simulator = Simulator(
                _build_python_config(
                    config,
                    workload_path,
                    output_dir,
                )
            )
            if execution["type"] == "fixed":
                _install_fixed_predictor(
                    simulator,
                    [
                        float(value)
                        for value in execution[
                            "stage_latencies_ms"
                        ]
                    ],
                )
            predictions = _capture_predictions(simulator)
            simulator.run()
    owners = _request_owners(simulator._event_trace)
    return {
        "oracle": "frontier.simulator.Simulator",
        "request_owners": owners,
        "requests": _normalize_requests(simulator, owners),
        "event_trace": _normalize_events(simulator._event_trace),
        "scheduler_trace": _normalize_scheduler_trace(
            raw_scheduler_decisions
        ),
        "batch_stages": _normalize_stages(
            simulator,
            predictions,
            include_components=(
                execution["type"] == "analytical"
            ),
        ),
        "simulation_completed_at_s": float(simulator._time),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run_oracle(arguments.config, arguments.workload),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
