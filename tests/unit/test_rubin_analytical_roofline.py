from types import SimpleNamespace

import pytest

from frontier.cc_backend.backends.astra_sim_analytical_cc_backend import (
    AstraSimAnalyticalCCBackend,
)
from frontier.cc_backend.cc_backend_config import (
    AstraSimAnalyticalCCBackendConfig,
)
from frontier.config import (
    AnalyticalRooflineExecutionTimePredictorConfig,
    MetricsConfig,
    PrecisionType,
    ReplicaConfig,
    VllmV1SchedulerConfig,
)
from frontier.config.cpu_kv_cache_config import CPUKVCacheConfig
from frontier.config.device_sku_config import BaseDeviceSKUConfig
from frontier.config.node_sku_config import BaseNodeSKUConfig
from frontier.entities.cluster import Cluster
from frontier.execution_time_predictor.analytical_roofline_execution_time_predictor import (
    AnalyticalRooflineExecutionTimePredictor,
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
    def __init__(self, *, scheduled_tokens: int, processed_tokens: int, prefill: bool):
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


def _predictor(**config_overrides):
    replica_config = ReplicaConfig(
        device="rubin",
        network_device="vera_rubin_nvl72_domain",
        attn_tensor_parallel_size=8,
    )
    return AnalyticalRooflineExecutionTimePredictor(
        AnalyticalRooflineExecutionTimePredictorConfig(**config_overrides),
        replica_config,
        VllmV1SchedulerConfig(),
        MetricsConfig(),
        cluster_type=ClusterType.MONOLITHIC,
        cc_backend=_NoCommunicationBackend(),
    )


def test_rubin_and_nvl72_logical_skus_publish_analytical_ceilings():
    device = BaseDeviceSKUConfig.create_from_type_string("rubin")
    node = BaseNodeSKUConfig.create_from_type_string(
        "vera_rubin_nvl72_domain"
    )

    assert device.total_memory_gb == 288
    assert device.hbm_bandwidth_tbps == pytest.approx(22.0)
    assert device.fp16_tflops == 4_000
    assert device.fp8_tflops == pytest.approx(17_500.0)
    assert device.nvfp4_tflops == pytest.approx(50_000.0)
    assert node.num_devices_per_node == 72


def test_roofline_formula_preserves_partial_overlap_and_diagnostics():
    predictor = _predictor(kernel_launch_latency_us=1.0)
    result = predictor.predict_roofline_kernel(
        operator_name="unit_test",
        phase="decode",
        local_shape="M=1,K=1,N=1",
        flops=4_000e12 * 0.5 * 0.002 / 1e3,
        hbm_bytes=22e12 * 0.5 * 0.003 / 1e3,
        precision=PrecisionType.BF16,
        compute_efficiency=0.5,
        memory_efficiency=0.5,
        overlap_penalty=0.25,
    )

    assert result == pytest.approx(0.001 + 0.003 + 0.25 * 0.002)
    diagnostic = predictor.get_diagnostics()[-1]
    assert diagnostic["compute_time_ms"] == pytest.approx(0.002)
    assert diagnostic["memory_time_ms"] == pytest.approx(0.003)
    assert diagnostic["bottleneck"] == "HBM"


def test_long_context_decode_attention_increases_runtime_and_is_auditable():
    predictor = _predictor()
    short = predictor.predict_attention_layer_time(
        _Batch(scheduled_tokens=1, processed_tokens=128, prefill=False),
        layer_id=0,
        cluster_type=ClusterType.MONOLITHIC,
    )
    long = predictor.predict_attention_layer_time(
        _Batch(scheduled_tokens=1, processed_tokens=4096, prefill=False),
        layer_id=0,
        cluster_type=ClusterType.MONOLITHIC,
    )

    assert (
        long.attention_decode_execution_time
        > short.attention_decode_execution_time
    )
    decode_records = [
        item
        for item in predictor.get_diagnostics()
        if item["operator_name"] == "attn_decode"
    ]
    assert decode_records[-1]["hbm_bytes"] > decode_records[-2]["hbm_bytes"]


def test_stage_prediction_is_profile_free_and_scales_layers_once():
    predictor = _predictor()
    batch = _Batch(scheduled_tokens=128, processed_tokens=0, prefill=True)
    one_layer = predictor.predict_stage_execution_time(
        batch, 0, ClusterType.MONOLITHIC, num_layers=1
    )
    four_layers = predictor.predict_stage_execution_time(
        batch, 0, ClusterType.MONOLITHIC, num_layers=4
    )

    assert one_layer.model_time_ms > 0.0
    assert four_layers.model_time_ms == pytest.approx(
        one_layer.model_time_ms * 4
    )


def test_cpu_static_slice_aggregates_over_attention_tp_target():
    config = CPUKVCacheConfig(enable=True, static_slice_per_gpu=True)

    one_gpu = config.resolve_for_target(1)
    eight_gpus = config.resolve_for_target(8)

    assert one_gpu.capacity_bytes == 750_000_000_000
    assert one_gpu.read_bandwidth_gbps == pytest.approx(4_800.0)
    assert one_gpu.write_bandwidth_gbps == pytest.approx(4_800.0)
    assert eight_gpus.capacity_bytes == 6_000_000_000_000
    assert eight_gpus.read_bandwidth_gbps == pytest.approx(38_400.0)
    assert eight_gpus.write_bandwidth_gbps == pytest.approx(38_400.0)


def test_nvl72_materializes_as_one_astra_switch_domain():
    replica_config = ReplicaConfig(
        device="rubin",
        network_device="vera_rubin_nvl72_domain",
        attn_tensor_parallel_size=72,
    )
    config = AstraSimAnalyticalCCBackendConfig(
        intra_server_topology="Switch",
        intra_server_bandwidth_gbps=14_400.0,
        intra_server_latency_us=1.0,
    )
    cluster = Cluster.__new__(Cluster)
    cluster._config = SimpleNamespace(num_replicas=1)
    cluster._cluster_type = ClusterType.MONOLITHIC

    resolved = cluster._materialize_astra_sim_analytical_cc_config(
        config, replica_config, num_devices=replica_config.world_size
    )
    backend = AstraSimAnalyticalCCBackend(
        resolved,
        ClusterType.MONOLITHIC,
        "rubin",
        "vera_rubin_nvl72_domain",
        72,
    )

    assert resolved.cluster_servers == 1
    assert resolved.cluster_gpus_per_server == 72
    assert resolved.intra_server_topology == "Switch"
    assert resolved.intra_server_bandwidth_gbps == pytest.approx(14_400.0)
    assert (
        backend.predict_allreduce(
            1_000_000, 72, comm_domain="ATTN_TP"
        )
        > 0.0
    )
