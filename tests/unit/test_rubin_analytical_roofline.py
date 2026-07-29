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
    ClusterConfig,
    MetricsConfig,
    PrecisionType,
    ReplicaConfig,
    SyntheticRequestGeneratorConfig,
    VllmV1SchedulerConfig,
)
from frontier.config.cpu_kv_cache_config import CPUKVCacheConfig
from frontier.config.device_sku_config import BaseDeviceSKUConfig
from frontier.config.node_sku_config import BaseNodeSKUConfig
from frontier.config.parallel_semantics import (
    build_rack_local_replica_placement,
)
from frontier.entities.cluster import Cluster
from frontier.execution_time_predictor.analytical_roofline_execution_time_predictor import (
    AnalyticalRooflineExecutionTimePredictor,
)
from frontier.scheduler.cluster_scheduler.base_cluster_scheduler import (
    BaseClusterScheduler,
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


def _moe_predictor(
    *,
    routing_mode="simulation",
    distribution_type="balanced",
    routing_seed=42,
    **config_overrides,
):
    replica_config = ReplicaConfig(
        model_name="Phi-tiny-MoE-instruct",
        device="rubin",
        network_device="vera_rubin_nvl72_domain",
        attn_tensor_parallel_size=2,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=2,
        total_expert_num=8,
        router_topk=2,
        moe_routing_mode=routing_mode,
        moe_routing_seed=routing_seed,
        moe_routing_distribution_type=distribution_type,
    )
    return AnalyticalRooflineExecutionTimePredictor(
        AnalyticalRooflineExecutionTimePredictorConfig(**config_overrides),
        replica_config,
        VllmV1SchedulerConfig(),
        MetricsConfig(),
        cluster_type=ClusterType.PREFILL,
        cc_backend=_NoCommunicationBackend(),
    )


def test_rubin_and_nvl72_logical_skus_publish_analytical_ceilings():
    device = BaseDeviceSKUConfig.create_from_type_string("rubin")
    node = BaseNodeSKUConfig.create_from_type_string(
        "vera_rubin_nvl72_domain"
    )

    assert device.total_memory_gb == 288
    assert device.hbm_bandwidth_tbps == pytest.approx(22.0)
    assert device.fp32_tflops == pytest.approx(400.0)
    assert device.fp16_tflops == 4_000
    assert device.fp8_tflops == pytest.approx(17_500.0)
    assert device.fp4_tflops == pytest.approx(35_000.0)
    assert node.num_devices_per_node == 72


def test_rubin_nvl12_partition_is_a_twelve_gpu_logical_domain():
    node = BaseNodeSKUConfig.create_from_type_string(
        "vera_rubin_nvl12_partition"
    )

    assert node.device_sku_type.name == "RUBIN"
    assert node.num_devices_per_node == 12


def test_rubin_nvl12_partition_packs_one_twelve_gpu_replica():
    cluster = Cluster(
        ClusterConfig(
            num_replicas=1,
            num_racks=1,
            replica_config=ReplicaConfig(
                device="rubin",
                network_device="vera_rubin_nvl12_partition",
                attn_tensor_parallel_size=4,
                attn_data_parallel_size=3,
                moe_tensor_parallel_size=1,
                moe_expert_parallel_size=12,
                model_name="moonshotai/Kimi-K2-Instruct",
                total_expert_num=384,
                router_topk=8,
            ),
        ),
        MetricsConfig(write_json_trace=False),
        SyntheticRequestGeneratorConfig(),
    )

    assert cluster.rack_placement.gpus_per_rack == 12
    assert cluster.rack_placement.active_gpus == 12
    assert cluster.rack_placement.idle_gpus == 0


@pytest.mark.parametrize(
    ("precision", "expected_tflops"),
    [
        (PrecisionType.FP4, 35_000.0),
        (PrecisionType.INT4, 35_000.0),
        (PrecisionType.FP8, 17_500.0),
        (PrecisionType.INT8, 17_500.0),
        (PrecisionType.FP16, 4_000.0),
        (PrecisionType.BF16, 4_000.0),
        (PrecisionType.FP32, 400.0),
    ],
)
def test_rubin_analytical_ceiling_is_selected_by_bit_width(
    precision,
    expected_tflops,
):
    assert _predictor()._peak_tflops(precision) == pytest.approx(
        expected_tflops
    )


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


def test_cpu_static_slice_uses_c2c_bottleneck_and_aggregates_over_gpu_target():
    config = CPUKVCacheConfig(enable=True, static_slice_per_gpu=True)

    one_gpu = config.resolve_for_target(1)
    eight_gpus = config.resolve_for_target(8)

    assert one_gpu.capacity_bytes == 750_000_000_000
    assert config.dram_bandwidth_gbps_per_gpu == pytest.approx(4_800.0)
    assert config.c2c_bandwidth_gbps_per_gpu == pytest.approx(3_600.0)
    assert one_gpu.read_bandwidth_gbps == pytest.approx(3_600.0)
    assert one_gpu.write_bandwidth_gbps == pytest.approx(3_600.0)
    assert eight_gpus.capacity_bytes == 6_000_000_000_000
    assert eight_gpus.read_bandwidth_gbps == pytest.approx(28_800.0)
    assert eight_gpus.write_bandwidth_gbps == pytest.approx(28_800.0)


def test_cpu_static_slice_uses_dram_when_it_is_the_slower_path():
    config = CPUKVCacheConfig(
        enable=True,
        static_slice_per_gpu=True,
        dram_bandwidth_gbps_per_gpu=2_400.0,
        c2c_bandwidth_gbps_per_gpu=3_600.0,
    )

    target = config.resolve_for_target(2)

    assert target.read_bandwidth_gbps == pytest.approx(4_800.0)
    assert target.write_bandwidth_gbps == pytest.approx(4_800.0)


def test_cpu_static_slice_counts_attention_tp_across_pipeline_stages():
    config = CPUKVCacheConfig(enable=True, static_slice_per_gpu=True)

    target = config.resolve_for_replica_target(
        attn_tensor_parallel_size=8,
        num_pipeline_stages=2,
    )

    assert target.capacity_bytes == 12_000_000_000_000
    assert target.read_bandwidth_gbps == pytest.approx(57_600.0)
    assert target.write_bandwidth_gbps == pytest.approx(57_600.0)


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
    cluster._rack_placement = build_rack_local_replica_placement(
        num_replicas=1,
        replica_gpus=72,
        gpus_per_rack=72,
        configured_num_racks=1,
    )

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
    assert resolved.runtime_num_racks == 1
    assert resolved.runtime_num_replicas == 1
    assert resolved.intra_server_topology == "Switch"
    assert resolved.intra_server_bandwidth_gbps == pytest.approx(14_400.0)
    assert (
        backend.predict_allreduce(
            1_000_000, 72, comm_domain="ATTN_TP"
        )
        > 0.0
    )


def test_nvl72_multi_rack_keeps_collectives_replica_local():
    replica_config = ReplicaConfig(
        device="rubin",
        network_device="vera_rubin_nvl72_domain",
        attn_tensor_parallel_size=2,
    )
    config = AstraSimAnalyticalCCBackendConfig(
        intra_server_topology="Switch",
        intra_server_bandwidth_gbps=14_400.0,
        intra_server_latency_us=1.0,
    )
    cluster = Cluster.__new__(Cluster)
    cluster._config = SimpleNamespace(num_replicas=72)
    cluster._cluster_type = ClusterType.MONOLITHIC
    cluster._rack_placement = build_rack_local_replica_placement(
        num_replicas=72,
        replica_gpus=2,
        gpus_per_rack=72,
        configured_num_racks=2,
    )

    resolved = cluster._materialize_astra_sim_analytical_cc_config(
        config,
        replica_config,
        num_devices=replica_config.world_size,
    )
    backend = AstraSimAnalyticalCCBackend(
        resolved,
        ClusterType.MONOLITHIC,
        "rubin",
        "vera_rubin_nvl72_domain",
        2,
    )

    assert resolved.runtime_num_racks == 2
    assert resolved.runtime_gpus_per_rack == 72
    assert resolved.runtime_replicas_per_rack == 36
    assert resolved.runtime_idle_gpus == 0
    assert resolved.runtime_num_replicas == 1
    assert resolved.cluster_servers == 1
    assert resolved.cluster_gpus_per_server == 2
    assert backend.predict_allreduce(
        1_000_000,
        2,
        comm_domain="ATTN_TP",
    ) > 0.0
    with pytest.raises(ValueError, match="exceeds domain_size"):
        backend.predict_allreduce(
            1_000_000,
            4,
            comm_domain="ATTN_TP",
        )


def test_nvl72_rack_packing_keeps_fragmented_capacity_idle():
    placement = build_rack_local_replica_placement(
        num_replicas=3,
        replica_gpus=40,
        gpus_per_rack=72,
    )

    assert placement.num_racks == 3
    assert placement.replicas_per_rack == 1
    assert placement.active_gpus == 120
    assert placement.idle_gpus == 96


def test_nvl72_rack_packing_rejects_cross_rack_replica_or_insufficient_racks():
    with pytest.raises(ValueError, match="one replica to fit in one rack"):
        build_rack_local_replica_placement(
            num_replicas=1,
            replica_gpus=73,
            gpus_per_rack=72,
        )

    with pytest.raises(ValueError, match="cannot hold all rack-local replicas"):
        build_rack_local_replica_placement(
            num_replicas=72,
            replica_gpus=2,
            gpus_per_rack=72,
            configured_num_racks=1,
        )


def test_nvl72_cluster_config_exposes_explicit_multi_rack_capacity():
    cluster_config = ClusterConfig(
        num_replicas=72,
        num_racks=2,
        replica_config=ReplicaConfig(
            device="rubin",
            network_device="vera_rubin_nvl72_domain",
            attn_tensor_parallel_size=2,
        ),
    )

    assert cluster_config.get_server_count_metadata("co-location") == {
        "server_count": 2
    }


def test_nvl72_cluster_assigns_each_whole_replica_to_one_rack():
    cluster = Cluster(
        ClusterConfig(
            num_replicas=10,
            num_racks=2,
            replica_config=ReplicaConfig(
                device="rubin",
                network_device="vera_rubin_nvl72_domain",
                attn_tensor_parallel_size=8,
            ),
        ),
        MetricsConfig(write_json_trace=False),
        SyntheticRequestGeneratorConfig(),
    )
    rack_ids = [
        cluster.get_replica_rack_id(replica_id)
        for replica_id in cluster.replicas
    ]

    assert rack_ids == ([0] * 9) + [1]
    assert cluster.rack_placement.num_racks == 2
    assert cluster.rack_placement.replicas_per_rack == 9
    assert cluster.rack_placement.idle_gpus == 64


def test_nvl72_pdd_propagates_cluster_exclusive_rack_counts():
    root_config = ClusterConfig(
        num_replicas=None,
        prefill_cluster_num_replicas=18,
        decode_cluster_num_replicas=18,
        prefill_cluster_num_racks=1,
        decode_cluster_num_racks=1,
        prefill_replica_config_device="rubin",
        decode_replica_config_device="rubin",
        prefill_replica_config_network_device="vera_rubin_nvl72_domain",
        decode_replica_config_network_device="vera_rubin_nvl72_domain",
        prefill_replica_config_attn_tensor_parallel_size=2,
        decode_replica_config_attn_tensor_parallel_size=2,
    )

    clusters = root_config.get_cluster_configs_for_disaggregation()

    assert clusters[ClusterType.PREFILL].num_racks == 1
    assert clusters[ClusterType.DECODE].num_racks == 1
    assert root_config.get_server_count_metadata("pd-disaggregation") == {
        "prefill_server_count": 1,
        "decode_server_count": 1,
    }


def test_analytical_moe_routing_is_selectable_and_conserves_tokens():
    batch = _Batch(scheduled_tokens=32, processed_tokens=0, prefill=True)

    balanced = _moe_predictor(distribution_type="balanced")
    balanced_lanes = balanced._get_ep_lane_per_expert_tokens(
        batch,
        layer_id=0,
        cluster_type=ClusterType.PREFILL,
    )
    assert [sum(tokens.values()) for tokens in balanced_lanes.values()] == [
        32,
        32,
    ]

    zipf = _moe_predictor(distribution_type="zipf")
    zipf_lanes = zipf._get_ep_lane_per_expert_tokens(
        batch,
        layer_id=0,
        cluster_type=ClusterType.PREFILL,
    )
    assert sum(
        token_count
        for allocation in zipf_lanes.values()
        for token_count in allocation.values()
    ) == 64
    assert sum(zipf_lanes[0].values()) > sum(zipf_lanes[1].values())

    random_a = _moe_predictor(
        distribution_type="random",
        routing_seed=7,
    )
    random_b = _moe_predictor(
        distribution_type="random",
        routing_seed=7,
    )
    assert random_a._get_ep_lane_per_expert_tokens(
        batch,
        layer_id=3,
        cluster_type=ClusterType.PREFILL,
    ) == random_b._get_ep_lane_per_expert_tokens(
        batch,
        layer_id=3,
        cluster_type=ClusterType.PREFILL,
    )

    uniform_random = _moe_predictor(
        routing_mode="uniform_random",
        distribution_type="zipf",
    )
    uniform_random_lanes = uniform_random._get_ep_lane_per_expert_tokens(
        batch,
        layer_id=0,
        cluster_type=ClusterType.PREFILL,
    )
    assert sum(
        token_count
        for allocation in uniform_random_lanes.values()
        for token_count in allocation.values()
    ) == 64


def test_analytical_moe_predicts_each_ep_lane_from_local_expert_tokens():
    predictor = _moe_predictor(distribution_type="zipf")
    batch = _Batch(scheduled_tokens=32, processed_tokens=0, prefill=True)

    lane_allocations = predictor._get_ep_lane_per_expert_tokens(
        batch,
        layer_id=0,
        cluster_type=ClusterType.PREFILL,
    )
    lane_times_ms = predictor.predict_ep_lane_moe_times_ms(
        batch,
        layer_id=0,
        cluster_type=ClusterType.PREFILL,
    )

    assert set(lane_times_ms) == {0, 1}
    assert sum(lane_allocations[0].values()) > sum(
        lane_allocations[1].values()
    )
    assert lane_times_ms[0] > lane_times_ms[1]
    assert (
        predictor.predict_monolithic_decode_shared_domain_lane_moe_times_ms(
            batch,
            layer_id=0,
        )
        == predictor.predict_ep_lane_moe_times_ms(
            batch,
            layer_id=0,
            cluster_type=ClusterType.MONOLITHIC,
        )
    )


def test_analytical_moe_models_grouped_projection_once_per_gpu():
    predictor = _moe_predictor()
    batch = _Batch(scheduled_tokens=16, processed_tokens=0, prefill=True)
    predictor.clear_diagnostics()

    predictor.predict_moe_layer_time(
        batch,
        layer_id=0,
        cluster_type=ClusterType.PREFILL,
        per_expert_tokens={0: 24, 1: 8, 2: 0, 3: 0},
    )

    grouped_records = [
        record
        for record in predictor.get_diagnostics()
        if record["operator_name"] == "moe_grouped_gemm"
    ]
    assert len(grouped_records) == 2
    assert "projection=up,groups=2" in grouped_records[0]["local_shape"]
    assert "projection=down,groups=2" in grouped_records[1]["local_shape"]


def test_analytical_mixed_moe_uses_dense_ffn_for_dense_layer():
    predictor = _moe_predictor()
    predictor._model_config.moe_layers_enum = "1"
    predictor._model_config._moe_layer_ids_cache = None
    batch = _Batch(scheduled_tokens=16, processed_tokens=0, prefill=True)

    dense = predictor.predict_stage_execution_time(
        batch,
        stage_id=0,
        cluster_type=ClusterType.PREFILL,
        num_layers=1,
        layer_id=0,
    )
    moe = predictor.predict_stage_execution_time(
        batch,
        stage_id=0,
        cluster_type=ClusterType.PREFILL,
        num_layers=1,
        layer_id=1,
    )

    assert dense._is_moe is False
    assert dense._mlp_layer_up_proj_execution_time > 0.0
    assert dense._moe_grouped_gemm_time == 0.0
    assert dense._expert_parallel_communication_time == 0.0
    assert moe._is_moe is True
    assert moe._moe_grouped_gemm_time > 0.0
    assert moe._mlp_layer_up_proj_execution_time == 0.0


def test_analytical_roofline_supports_kimi_k2_latent_mla():
    replica_config = ReplicaConfig(
        model_name="moonshotai/Kimi-K2-Instruct",
        device="rubin",
        network_device="vera_rubin_nvl12_partition",
        attn_tensor_parallel_size=4,
        attn_data_parallel_size=3,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=12,
        total_expert_num=384,
        router_topk=8,
    )
    predictor = AnalyticalRooflineExecutionTimePredictor(
        AnalyticalRooflineExecutionTimePredictorConfig(),
        replica_config,
        VllmV1SchedulerConfig(),
        MetricsConfig(),
        cluster_type=ClusterType.DECODE,
        cc_backend=_NoCommunicationBackend(),
    )

    short = predictor.predict_attention_layer_time(
        _Batch(scheduled_tokens=1, processed_tokens=1_024, prefill=False),
        layer_id=1,
        cluster_type=ClusterType.DECODE,
    )
    long = predictor.predict_attention_layer_time(
        _Batch(scheduled_tokens=1, processed_tokens=65_535, prefill=False),
        layer_id=1,
        cluster_type=ClusterType.DECODE,
    )

    assert predictor._model_config.get_runtime_num_kv_heads() == 1
    assert predictor._model_config.get_runtime_head_size() == 576
    assert (
        long.attention_decode_execution_time
        > short.attention_decode_execution_time
    )


def test_cluster_scheduler_validates_predictor_ep_lane_times():
    class _LanePredictor:
        def predict_ep_lane_moe_times_ms(
            self,
            _batch,
            _layer_id,
            _cluster_type,
        ):
            return {0: 1.25, 1: 2.5}

    lane_times = BaseClusterScheduler._predict_ep_lane_moe_times_ms(
        _LanePredictor(),
        object(),
        0,
        ClusterType.DECODE,
    )
    assert lane_times == {0: 1.25, 1: 2.5}
    assert max(lane_times.values()) == pytest.approx(2.5)
