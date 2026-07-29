"""Generate the Step 1D analytical parity fixture from the Python oracle."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

# The fixture generator is a machine-readable oracle. Frontier configures its
# logger during imports, so disable logging before importing project modules to
# keep stdout as JSON only.
logging.disable(logging.CRITICAL)

from frontier.cc_backend.backends.analytical_cc_backend import (
    AnalyticalCCBackend,
)
from frontier.cc_backend.cc_backend_config import AnalyticalCCBackendConfig
from frontier.config import (
    AnalyticalRooflineExecutionTimePredictorConfig,
    MetricsConfig,
    PrecisionType,
    ReplicaConfig,
    VllmV1SchedulerConfig,
)
from frontier.config.kv_cache_transfer_config import (
    AnalyticalKVCacheTransferConfig,
)
from frontier.execution_time_predictor.analytical_roofline_execution_time_predictor import (
    AnalyticalRooflineExecutionTimePredictor,
)
from frontier.kv_cache_transfer.analytical_kv_cache_transfer_predictor import (
    AnalyticalKVCacheTransferPredictor,
)
from frontier.types import ClusterType


class _NoCommunicationBackend:
    def predict_allreduce(self, **_kwargs):
        return 0.0

    def predict_allgather(self, **_kwargs):
        return 0.0

    def predict_all_to_all(self, **_kwargs):
        return 0.0

    def predict_send_recv(self, **_kwargs):
        return 0.0


class _Batch:
    def __init__(
        self,
        *,
        scheduled_tokens: int,
        processed_tokens: int,
        prefill: bool,
    ) -> None:
        self.requests = [
            SimpleNamespace(
                is_prefill_complete=not prefill,
                num_processed_tokens=processed_tokens,
            )
        ]
        self.num_tokens = [scheduled_tokens]
        self.total_num_tokens = scheduled_tokens
        self.num_prefill_tokens = scheduled_tokens if prefill else 0
        self.num_decode_tokens = 0 if prefill else scheduled_tokens
        self.is_moe = False

    def get_effective_total_tokens_for_compute(self, _cluster_type=None):
        return self.total_num_tokens


def _dense_times(predictor, *, tokens: int, past: int, prefill: bool) -> dict:
    batch = _Batch(
        scheduled_tokens=tokens,
        processed_tokens=past,
        prefill=prefill,
    )
    attention = predictor.predict_attention_layer_time(
        batch,
        layer_id=0,
        cluster_type=ClusterType.MONOLITHIC,
    )
    mlp = predictor.predict_mlp_layer_time(
        batch,
        layer_id=0,
        cluster_type=ClusterType.MONOLITHIC,
    )
    return {
        "attention_pre_projection_ms": (
            attention.attention_layer_pre_proj_execution_time
        ),
        "attention_post_projection_ms": (
            attention.attention_layer_post_proj_execution_time
        ),
        "rope_ms": attention.attention_rope_execution_time,
        "kv_cache_save_ms": (
            attention.attention_kv_cache_save_execution_time
        ),
        "attention_norm_ms": attention.attn_norm_time,
        "prefill_attention_ms": (
            attention.attention_prefill_execution_time
        ),
        "decode_attention_ms": attention.attention_decode_execution_time,
        "mlp_up_projection_ms": mlp.mlp_layer_up_proj_execution_time,
        "mlp_activation_ms": mlp.mlp_layer_act_execution_time,
        "mlp_down_projection_ms": mlp.mlp_layer_down_proj_execution_time,
        "mlp_norm_ms": mlp.mlp_norm_time,
        "residual_add_ms": 0.0,
    }


def main() -> None:
    replica = ReplicaConfig(
        device="rubin",
        network_device="vera_rubin_nvl72_domain",
        attn_tensor_parallel_size=8,
    )
    predictor = AnalyticalRooflineExecutionTimePredictor(
        AnalyticalRooflineExecutionTimePredictorConfig(),
        replica,
        VllmV1SchedulerConfig(),
        MetricsConfig(),
        cluster_type=ClusterType.MONOLITHIC,
        cc_backend=_NoCommunicationBackend(),
    )

    custom_predictor = AnalyticalRooflineExecutionTimePredictor(
        AnalyticalRooflineExecutionTimePredictorConfig(
            kernel_launch_latency_us=1.0
        ),
        replica,
        VllmV1SchedulerConfig(),
        MetricsConfig(),
        cluster_type=ClusterType.MONOLITHIC,
        cc_backend=_NoCommunicationBackend(),
    )
    roofline_flops = 4_000e12 * 0.5 * 0.002 / 1e3
    roofline_bytes = 22e12 * 0.5 * 0.003 / 1e3
    roofline_time = custom_predictor.predict_roofline_kernel(
        operator_name="golden_roofline",
        phase="decode",
        local_shape="fixture",
        flops=roofline_flops,
        hbm_bytes=roofline_bytes,
        precision=PrecisionType.FP16,
        compute_efficiency=0.5,
        memory_efficiency=0.5,
        overlap_penalty=0.25,
    )
    roofline_diagnostic = custom_predictor.get_diagnostics()[-1]

    communication = AnalyticalCCBackend(
        AnalyticalCCBackendConfig(
            network_bandwidth_gbps=400.0,
            network_latency_us=1.0,
            intra_node_bandwidth_gbps=14_400.0,
        ),
        ClusterType.MONOLITHIC,
        "rubin",
        "vera_rubin_nvl72_domain",
        72,
    )
    communication_bytes = 1_000_000
    communication_devices = 72

    kv_predictor = AnalyticalKVCacheTransferPredictor(
        AnalyticalKVCacheTransferConfig(
            network_bandwidth_gbps=100.0,
            network_latency_ms=0.1,
            override_num_layers=32,
            override_num_heads=32,
            override_head_dim=128,
            kv_cache_dtype_size_bytes=2,
        )
    )
    kv_tokens = 128
    kv_size = kv_predictor.get_kv_cache_size_for_request(
        SimpleNamespace(num_prefill_tokens=kv_tokens),
        replica,
    )

    result = {
        "schema_version": 1,
        "roofline": {
            "input": {
                "flops": roofline_flops,
                "hbm_bytes": roofline_bytes,
                "compute_efficiency": 0.5,
                "memory_efficiency": 0.5,
                "overlap_penalty": 0.25,
                "kernel_launch_latency_us": 1.0,
            },
            "expected": {
                "compute_time_ms": roofline_diagnostic["compute_time_ms"],
                "memory_time_ms": roofline_diagnostic["memory_time_ms"],
                "launch_time_ms": roofline_diagnostic["launch_time_ms"],
                "predicted_time_ms": roofline_time,
                "bottleneck": roofline_diagnostic["bottleneck"],
            },
        },
        "dense_cases": [
            {
                "name": "prefill_128",
                "input": {
                    "scheduled_tokens": 128,
                    "past_context": 0,
                    "prefill": True,
                },
                "expected": _dense_times(
                    predictor,
                    tokens=128,
                    past=0,
                    prefill=True,
                ),
            },
            {
                "name": "decode_1_past_4096",
                "input": {
                    "scheduled_tokens": 1,
                    "past_context": 4096,
                    "prefill": False,
                },
                "expected": _dense_times(
                    predictor,
                    tokens=1,
                    past=4096,
                    prefill=False,
                ),
            },
        ],
        "communication": {
            "input": {
                "network_bandwidth_gbps": 400.0,
                "latency_us": 1.0,
                "intra_node_bandwidth_gbps": 14_400.0,
                "data_size_bytes": communication_bytes,
                "num_devices": communication_devices,
            },
            "expected": {
                "point_to_point_ms": communication.predict_send_recv(
                    communication_bytes
                ),
                "allreduce_ms": communication.predict_allreduce(
                    communication_bytes,
                    communication_devices,
                ),
                "allgather_ms": communication.predict_allgather(
                    communication_bytes,
                    communication_devices,
                ),
                "broadcast_ms": communication.predict_broadcast(
                    communication_bytes,
                    communication_devices,
                ),
                "reduce_scatter_ms": communication.predict_reduce_scatter(
                    communication_bytes,
                    communication_devices,
                ),
                "all_to_all_ms": communication.predict_all_to_all(
                    communication_bytes,
                    communication_devices,
                ),
            },
        },
        "kv_transfer": {
            "input": {
                "num_tokens": kv_tokens,
                "num_layers": 32,
                "num_kv_heads_per_worker": 32,
                "head_dim": 128,
                "kv_factor": 2,
                "dtype_size_bytes": 2.0,
                "network_bandwidth_gbps": 100.0,
                "network_latency_ms": 0.1,
            },
            "expected": {
                "size_bytes": kv_size,
                "transfer_time_ms": kv_predictor.get_transfer_time(
                    ClusterType.PREFILL,
                    ClusterType.DECODE,
                    SimpleNamespace(),
                    kv_size,
                ),
            },
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
