"""Step 3 oracle backed by Frontier's production sequential PDD Simulator."""

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
    AnalyticalKVCacheTransferConfig,
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
from frontier.scheduler.replica_scheduler import (
    vllm_v1_engine_replica_scheduler as scheduler_module,
)
from frontier.simulator import Simulator
from frontier.types import ClusterType


class Step3OracleError(RuntimeError):
    pass


def _read_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise Step3OracleError("schema_version=1 is required")
    if config.get("system_architecture") != "pd-disaggregation":
        raise Step3OracleError("pd-disaggregation is required")
    if config.get("enable_parallel_clusters") is not False:
        raise Step3OracleError("parallel clusters must remain disabled")
    return config


def _scheduler_config(raw: dict[str, Any]) -> VllmV1SchedulerConfig:
    return VllmV1SchedulerConfig(
        batch_size_cap=int(raw["batch_size_cap"]),
        block_size=int(raw["block_size"]),
        watermark_blocks_fraction=float(
            raw["watermark_blocks_fraction"]
        ),
        max_tokens_in_batch=int(raw["max_tokens_in_batch"]),
        scheduling_policy=str(raw["scheduling_policy"]),
        enable_preemption=bool(raw["enable_preemption"]),
        enable_chunked_prefill=bool(
            raw["enable_chunked_prefill"]
        ),
        enable_prefix_caching=False,
        prefix_caching_key_mode="session",
        num_preallocate_tokens=int(raw["num_preallocate_tokens"]),
        long_prefill_token_threshold=int(
            raw["long_prefill_token_threshold"]
        ),
        num_blocks=int(raw["num_blocks"]),
        num_blocks_mode="explicit",
    )


def _replica_config(
    cluster: dict[str, Any],
    cluster_prefix: str,
) -> ReplicaConfig:
    topology = cluster["parallelism"]
    execution = cluster["execution_model"]
    model = cluster.get(
        "model",
        {
            "name": "llama2-7b",
            "runtime_total_experts": 1,
            "router_topk": 1,
        },
    )
    routing = cluster.get(
        "moe_routing",
        {
            "mode": "simulation",
            "distribution": "balanced",
            "seed": 42,
        },
    )
    model_name = (
        "meta-llama/Llama-2-7b-hf"
        if model["name"] == "llama2-7b"
        else str(model["name"])
    )
    return ReplicaConfig(
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
        data_parallel_size=int(
            topology["data_parallel_size"]
        ),
        moe_tensor_parallel_size=int(
            topology.get("moe_tensor_parallel_size", 1)
        ),
        moe_expert_parallel_size=int(
            topology.get("moe_expert_parallel_size", 1)
        ),
        total_expert_num=int(
            model.get("runtime_total_experts", 1)
        ),
        router_topk=int(model.get("router_topk", 1)),
        moe_routing_mode=str(routing.get("mode", "simulation")),
        moe_routing_distribution_type=str(
            routing.get("distribution", "balanced")
        ),
        moe_routing_seed=int(routing.get("seed", 42)),
        cluster_prefix=cluster_prefix,
    )


def _build_python_config(
    config: dict[str, Any],
    workload_path: Path,
    output_dir: str,
) -> SimulationConfig:
    prefill = config["clusters"]["prefill"]
    decode = config["clusters"]["decode"]
    prefill_scheduler = prefill["scheduler"]
    decode_scheduler = decode["scheduler"]
    prefill_replica = _replica_config(
        prefill,
        "prefill",
    )
    decode_replica = _replica_config(
        decode,
        "decode",
    )
    cluster = ClusterConfig(
        replica_scheduler_config=_scheduler_config(prefill_scheduler),
        execution_time_predictor_config=(
            AnalyticalRooflineExecutionTimePredictorConfig()
        ),
        cc_backend_config=AnalyticalCCBackendConfig(
            network_bandwidth_gbps=float(
                prefill["execution_model"].get(
                    "network_bandwidth_gbps", 400.0
                )
            ),
            network_latency_us=float(
                prefill["execution_model"].get(
                    "network_latency_us", 1.0
                )
            ),
            intra_node_bandwidth_gbps=float(
                prefill["execution_model"].get(
                    "intra_node_bandwidth_gbps", 14400.0
                )
            ),
        ),
        replica_config=prefill_replica,
        prefill_cluster_num_replicas=int(
            prefill["parallelism"]["num_replicas"]
        ),
        prefill_replica_config_num_pipeline_stages=int(
            prefill["parallelism"]["pipeline_parallel_size"]
        ),
        # The public constructor validates before the normalized private
        # replica configs below are installed.  Bootstrap through DP1 with
        # a TP size that preserves the shared attention/MoE domain, then
        # replace it with the requested topology in get_normalized_clusters.
        prefill_replica_config_attn_tensor_parallel_size=int(
            prefill["parallelism"]["moe_tensor_parallel_size"]
            * prefill["parallelism"]["moe_expert_parallel_size"]
        ),
        # Production Python currently rejects dense PDD DP > 1 while its
        # runtime schedulers support DP lanes.  Construct the public config
        # through DP1, then install the normalized cluster configs below.
        prefill_replica_config_attn_data_parallel_size=1,
        prefill_replica_config_moe_tensor_parallel_size=int(
            prefill["parallelism"]["moe_tensor_parallel_size"]
        ),
        prefill_replica_config_moe_expert_parallel_size=int(
            prefill["parallelism"]["moe_expert_parallel_size"]
        ),
        prefill_replica_config_total_expert_num=int(
            prefill["model"]["runtime_total_experts"]
        ),
        prefill_replica_config_device=str(
            prefill["execution_model"].get("device", "rubin")
        ),
        prefill_replica_config_network_device=(
            "vera_rubin_nvl72_domain"
        ),
        decode_cluster_num_replicas=int(
            decode["parallelism"]["num_replicas"]
        ),
        decode_replica_config_num_pipeline_stages=int(
            decode["parallelism"]["pipeline_parallel_size"]
        ),
        decode_replica_config_attn_tensor_parallel_size=int(
            decode["parallelism"]["moe_tensor_parallel_size"]
            * decode["parallelism"]["moe_expert_parallel_size"]
        ),
        decode_replica_config_attn_data_parallel_size=1,
        decode_replica_config_moe_tensor_parallel_size=int(
            decode["parallelism"]["moe_tensor_parallel_size"]
        ),
        decode_replica_config_moe_expert_parallel_size=int(
            decode["parallelism"]["moe_expert_parallel_size"]
        ),
        decode_replica_config_total_expert_num=int(
            decode["model"]["runtime_total_experts"]
        ),
        decode_replica_config_device=str(
            decode["execution_model"].get("device", "rubin")
        ),
        decode_replica_config_network_device=(
            "vera_rubin_nvl72_domain"
        ),
    )
    transfer = config["kv_cache_transfer"]
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
        run_id="step3-production-oracle",
    )
    simulation_config = SimulationConfig(
        simulation_mode=str(config["simulation_mode"]),
        sys_arch="pd-disaggregation",
        decode_cuda_graph_mode="none",
        cluster_config=cluster,
        request_generator_config=request_generator,
        metrics_config=metrics,
        kv_cache_transfer_config=AnalyticalKVCacheTransferConfig(
            network_bandwidth_gbps=float(
                transfer["network_bandwidth_gbps"]
            ),
            network_latency_ms=float(
                transfer["network_latency_ms"]
            ),
            kv_cache_dtype_size_bytes=int(
                transfer["kv_cache_dtype_size_bytes"]
            ),
            enable_compression=False,
        ),
        enable_parallel_clusters=False,
        log_level="critical",
    )
    original_get_clusters = simulation_config.get_clusters

    def get_normalized_clusters(
        self: SimulationConfig,
    ) -> dict[ClusterType, ClusterConfig]:
        clusters = original_get_clusters()
        normalized = {
            ClusterType.PREFILL: (
                prefill_replica,
                _scheduler_config(prefill_scheduler),
            ),
            ClusterType.DECODE: (
                decode_replica,
                _scheduler_config(decode_scheduler),
            ),
        }
        for cluster_type, (
            replica,
            scheduler,
        ) in normalized.items():
            cluster_config = clusters[cluster_type]
            cluster_config.replica_config = replica
            cluster_config.replica_scheduler_config = scheduler
            cluster_config.world_size = (
                int(cluster_config.num_replicas)
                * int(replica.world_size)
            )
        return clusters

    simulation_config.get_clusters = MethodType(
        get_normalized_clusters,
        simulation_config,
    )
    return simulation_config


def _fixed_execution_time(
    latency_ms: float,
    num_layers: int,
) -> ExecutionTime:
    if not math.isfinite(latency_ms) or latency_ms < 0.0:
        raise Step3OracleError("fixed stage latency is invalid")
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


def _install_fixed_predictors(
    simulator: Simulator,
    config: dict[str, Any],
) -> None:
    for cluster_type, name in (
        (ClusterType.PREFILL, "prefill"),
        (ClusterType.DECODE, "decode"),
    ):
        raw = config["clusters"][name]["execution_model"]
        if raw["type"] != "fixed":
            continue
        stage_latencies = [
            float(value) for value in raw["stage_latencies_ms"]
        ]
        predictor = simulator._predictors[cluster_type]
        if len(stage_latencies) != int(
            predictor._replica_config.num_pipeline_stages
        ):
            raise Step3OracleError(
                f"{name} fixed latency count does not match PP"
            )

        def predict_stage_execution_time(
            self: Any,
            batch: Batch,
            pipeline_stage: int,
            *,
            _latencies: list[float] = stage_latencies,
            **_: Any,
        ) -> ExecutionTime:
            del batch
            return _fixed_execution_time(
                _latencies[pipeline_stage],
                int(self._num_layers_per_pipeline_stage),
            )

        predictor.predict_stage_execution_time = MethodType(
            predict_stage_execution_time,
            predictor,
        )


def _event_name(value: Any) -> str:
    text = str(value).rsplit(".", 1)[-1].lower()
    return text


def _cluster_name(value: Any) -> str:
    return str(value).rsplit(".", 1)[-1].upper()


def _replica_bases(
    raw_events: list[dict[str, Any]],
) -> dict[str, int]:
    replicas: dict[str, list[int]] = {
        "PREFILL": [],
        "DECODE": [],
    }
    for raw in raw_events:
        if raw.get("replica_id") is None:
            continue
        cluster = _cluster_name(raw.get("cluster_type", ""))
        if cluster in replicas:
            replicas[cluster].append(int(raw["replica_id"]))
    return {
        cluster: min(values, default=0)
        for cluster, values in replicas.items()
    }


def _normalize_events(
    raw_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    replica_bases = _replica_bases(raw_events)
    result: list[dict[str, Any]] = []
    for sequence, raw in enumerate(raw_events, start=1):
        event_type = _event_name(raw["event_type"])
        event: dict[str, Any] = {
            "sequence": sequence,
            "time_s": float(raw["time"]),
            "type": event_type,
        }
        cluster: str | None = None
        if event_type in {
            "request_arrival",
            "cluster_schedule",
            "replica_schedule",
            "batch_stage_arrival",
            "replica_stage_schedule",
            "batch_stage_end",
            "prefill_sync",
            "prefill_sync_collective",
            "decode_sync",
            "decode_sync_collective",
        } and raw.get("cluster_type") is not None:
            cluster = _cluster_name(raw["cluster_type"])
        elif event_type in {
            "kv_cache_transfer_start",
            "kv_cache_transfer_end",
        }:
            cluster = _cluster_name(
                raw["target_cluster_type"]
            )
        if cluster is not None:
            event["cluster_type"] = cluster
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
            assert cluster is not None
            event["replica_id"] = (
                int(raw["replica_id"])
                - replica_bases[cluster]
            )
            if "dp_id" in raw:
                event["dp_id"] = int(raw["dp_id"])
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
        if event_type in {"prefill_sync", "decode_sync"}:
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
) -> dict[str, list[dict[str, int]]]:
    owners: dict[str, dict[int, tuple[int, int]]] = {
        "PREFILL": {},
        "DECODE": {},
    }
    for event in raw_events:
        if _event_name(event["event_type"]) != "cluster_schedule":
            continue
        cluster_name = _cluster_name(
            event.get("cluster_type", "")
        )
        if cluster_name not in owners:
            continue
        for replica_id, dp_id, request_id in event["request_mapping"]:
            if request_id is None:
                continue
            owners[cluster_name][int(request_id)] = (
                int(replica_id),
                int(dp_id),
            )
    replica_bases = {
        cluster: min(
            (target[0] for target in mapping.values()),
            default=0,
        )
        for cluster, mapping in owners.items()
    }
    return {
        cluster: [
            {
                "request_id": request_id,
                "replica_id": (
                    target[0] - replica_bases[cluster]
                ),
                "dp_id": target[1],
            }
            for request_id, target in sorted(mapping.items())
        ]
        for cluster, mapping in owners.items()
    }


def _normalize_requests(
    simulator: Simulator,
    owners: dict[str, list[dict[str, int]]],
) -> list[dict[str, Any]]:
    owner_maps = {
        cluster: {
            row["request_id"]: row for row in rows
        }
        for cluster, rows in owners.items()
    }
    result: list[dict[str, Any]] = []
    for request in simulator._all_requests:
        request_id = int(request.id)
        prefill = owner_maps["PREFILL"][request_id]
        decode = owner_maps["DECODE"][request_id]
        result.append(
            {
                "request_id": request_id,
                "arrived_at_s": float(request.arrived_at),
                "first_scheduled_at_s": float(request.scheduled_at),
                "prefill_completed_at_s": float(
                    request.prefill_completed_at
                ),
                "first_token_completed_at_s": float(
                    request.first_decode_token_completed_at
                ),
                "completed_at_s": float(request.completed_at),
                "num_processed_tokens": int(
                    request.num_processed_tokens
                ),
                "preemption_count": int(
                    request.get_total_preemption_count()
                ),
                "prefill_replica_id": prefill["replica_id"],
                "prefill_dp_id": prefill["dp_id"],
                "decode_replica_id": decode["replica_id"],
                "decode_dp_id": decode["dp_id"],
                "kv_cache_transfer_start_time_s": float(
                    request.kv_cache_transfer_start_time
                ),
                "kv_cache_transfer_end_time_s": float(
                    request.kv_cache_transfer_end_time
                ),
            }
        )
    return sorted(result, key=lambda row: row["request_id"])


def _normalize_transfers(records: list[Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for info in records:
        result.append(
            {
                "request_id": int(info.batch.requests[0].id),
                "source_cluster_type": info.source_cluster_type.name,
                "target_cluster_type": info.target_cluster_type.name,
                "source_replica_id": int(info.source_replica_id),
                "source_dp_id": int(info.source_dp_id),
                "size_bytes": int(info.kv_cache_size_bytes),
                "predicted_time_ms": float(info.transfer_time_ms),
                "started_at_s": float(info.transfer_start_time),
                "completed_at_s": float(info.transfer_end_time),
            }
        )
    return sorted(result, key=lambda row: row["request_id"])


def _normalize_scheduler_trace(
    raw_decisions: list[dict[str, Any]],
    replica_bases: dict[str, int],
) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    for row in raw_decisions:
        if row["event"] == "iteration_start":
            if current:
                raise Step3OracleError(
                    "scheduler decision iterations overlap"
                )
            current = [row]
            continue
        if not current:
            raise Step3OracleError(
                "scheduler decision lacks iteration_start"
            )
        current.append(row)
        if row["event"] != "iteration_end":
            continue
        start = current[0]
        end = current[-1]
        cluster = _cluster_name(start["cluster_type"])
        for item in current:
            if (
                item["cluster_type"] != start["cluster_type"]
                or item["replica_id"] != start["replica_id"]
                or item["dp_id"] != start["dp_id"]
                or item["iteration_id"] != start["iteration_id"]
            ):
                raise Step3OracleError(
                    "scheduler target changed mid-iteration"
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
                "replica_id": (
                    int(start["replica_id"])
                    - replica_bases[cluster]
                ),
                "dp_id": int(start["dp_id"]),
                "cluster_type": cluster,
            }
        )
        current = []
    if current:
        raise Step3OracleError(
            "scheduler decision trace lacks iteration_end"
        )
    return traces


def _normalize_stages(
    simulator: Simulator,
    replica_bases: dict[str, int],
    *,
    include_components: bool,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in (
        simulator._metric_store
        ._frontier_stage_batch_ledger_rows
    ):
        cluster = _cluster_name(row["cluster_type"])
        execution_time = row["execution_time"]
        components = execution_time["component_ledger_ms"]
        stage: dict[str, Any] = {
            "cluster_type": cluster,
            "replica_id": (
                int(row["replica_id"])
                - replica_bases[cluster]
            ),
            "dp_id": int(row["dp_id"]),
            "stage_id": int(row["stage_id"]),
            "request_ids": [
                int(value) for value in row["request_ids"]
            ],
            "request_num_tokens": [
                int(value)
                for value in row["request_num_tokens"]
            ],
            "started_at_s": float(row["stage_start_ts"]),
            "completed_at_s": float(row["stage_end_ts"]),
            "duration_ms": (
                float(row["stage_end_ts"])
                - float(row["stage_start_ts"])
            )
            * 1e3,
        }
        if include_components:
            tp_ms = float(
                components["attention_all_reduce_time"]
                + components["mlp_all_reduce_time"]
            )
            pp_ms = float(
                components[
                    "pipeline_parallel_communication_time"
                ]
            )
            stage.update(
                {
                    "dense_compute_ms": (
                        float(execution_time["total_time_ms"])
                        - tp_ms
                        - pp_ms
                    ),
                    "tp_communication_ms": tp_ms,
                    "pp_communication_ms": pp_ms,
                }
            )
        result.append(stage)
    return result


def run_oracle(
    config_path: Path,
    workload_path: Path,
) -> dict[str, Any]:
    config = _read_config(config_path)
    Request._id = -1
    Batch._id = -1
    BatchStage._id = -1
    BaseEvent._id = 0
    raw_scheduler_decisions: list[dict[str, Any]] = []
    completed_transfers: list[Any] = []
    scheduler_module._frontier_vllm_v1_sched_decision_logger = object()
    scheduler_module._log_frontier_vllm_v1_schedule_decision = (
        lambda payload: raw_scheduler_decisions.append(dict(payload))
    )
    with tempfile.TemporaryDirectory(
        prefix="frontier-step3-oracle-"
    ) as output_dir:
        with redirect_stdout(io.StringIO()):
            simulator = Simulator(
                _build_python_config(
                    config,
                    workload_path,
                    output_dir,
                )
            )
            original_transfer_end = (
                simulator._metric_store.on_kv_cache_transfer_end
            )

            def observe_transfer_end(
                self: Any,
                time: float,
                duration: float,
                size_bytes: int,
                target_cluster_type: ClusterType,
                transfer_info: Any,
            ) -> None:
                completed_transfers.append(transfer_info)
                original_transfer_end(
                    time,
                    duration,
                    size_bytes,
                    target_cluster_type,
                    transfer_info,
                )

            simulator._metric_store.on_kv_cache_transfer_end = (
                MethodType(
                    observe_transfer_end,
                    simulator._metric_store,
                )
            )
            _install_fixed_predictors(simulator, config)
            simulator.run()
    replica_bases = _replica_bases(simulator._event_trace)
    owners = _request_owners(simulator._event_trace)
    include_components = any(
        config["clusters"][name]["execution_model"]["type"]
        == "analytical"
        for name in ("prefill", "decode")
    )
    return {
        "oracle": "frontier.simulator.Simulator",
        "request_owners": owners,
        "requests": _normalize_requests(simulator, owners),
        "kv_cache_transfers": _normalize_transfers(
            completed_transfers
        ),
        "event_trace": _normalize_events(simulator._event_trace),
        "scheduler_trace": _normalize_scheduler_trace(
            raw_scheduler_decisions,
            replica_bases,
        ),
        "batch_stages": _normalize_stages(
            simulator,
            replica_bases,
            include_components=include_components,
        ),
        "simulation_completed_at_s": max(
            float(request.completed_at)
            for request in simulator._all_requests
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run_oracle(args.config, args.workload),
            indent=2,
            sort_keys=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
