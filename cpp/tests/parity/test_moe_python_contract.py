"""Frozen production-Python contracts consumed by C++ Step 3.5."""

from __future__ import annotations

from types import SimpleNamespace

from frontier.config.model_config import BaseModelConfig
from frontier.execution_time_predictor.analytical_roofline_execution_time_predictor import (
    AnalyticalRooflineExecutionTimePredictor,
)
from frontier.execution_time_predictor.sklearn_moe_execution_time_predictor import (
    SklearnMoEExecutionTimePredictor,
)
from frontier.scheduler.cluster_scheduler.base_cluster_scheduler import (
    BaseClusterScheduler,
)
from frontier.types import ClusterType


def _routing_predictor(
    *,
    mode: str,
    distribution: str,
    seed: int,
    experts: int = 16,
) -> AnalyticalRooflineExecutionTimePredictor:
    predictor = object.__new__(AnalyticalRooflineExecutionTimePredictor)
    predictor._replica_config = SimpleNamespace(total_expert_num=experts)
    predictor._moe_routing_mode = mode
    predictor._moe_routing_distribution_type = distribution
    predictor._moe_routing_seed = seed
    return predictor


def test_phi_model_shape_is_the_step35_reference() -> None:
    model = BaseModelConfig.create_from_name("Phi-tiny-MoE-instruct")
    assert model.is_moe
    assert (
        model.num_layers,
        model.embedding_dim,
        model.num_q_heads,
        model.num_kv_heads,
        model.head_dim,
        model.num_experts,
        model.num_experts_per_tok,
    ) == (32, 4096, 16, 4, 128, 16, 2)


def test_largest_remainder_and_random_vectors_are_frozen() -> None:
    discretize = (
        AnalyticalRooflineExecutionTimePredictor._discretize_expert_ratios
    )
    assert list(discretize(38, {index: 1.0 for index in range(8)}).values()) == [
        5,
        5,
        5,
        5,
        5,
        5,
        4,
        4,
    ]

    simulation_random = _routing_predictor(
        mode="simulation", distribution="random", seed=42
    )
    assert list(
        simulation_random._get_global_per_expert_tokens(37, 0).values()
    ) == [3, 2, 3, 3, 1, 4, 3, 3, 1, 2, 1, 3, 2, 3, 2, 1]

    uniform_random = _routing_predictor(
        mode="uniform_random", distribution="balanced", seed=42
    )
    assert list(
        uniform_random._get_global_per_expert_tokens(37, 0).values()
    ) == [0, 5, 2, 2, 0, 1, 2, 4, 5, 0, 2, 3, 4, 4, 2, 1]

    second_layer = _routing_predictor(
        mode="uniform_random",
        distribution="balanced",
        seed=123456789,
    )
    assert list(
        second_layer._get_global_per_expert_tokens(37, 31).values()
    ) == [1, 3, 4, 0, 2, 2, 1, 0, 3, 4, 3, 3, 4, 1, 1, 5]


def test_monolithic_decode_sync_id_is_lane_scoped() -> None:
    scheduler = SimpleNamespace(_replica_ep_size=4)
    assert [
        BaseClusterScheduler.make_decode_sync_global_id(
            scheduler, 0, lane, counter
        )
        for counter in range(2)
        for lane in range(4)
    ] == list(range(8))


def test_dummy_moe_communication_skip_rules_are_frozen() -> None:
    class _ConcreteDummyPredictor(SklearnMoEExecutionTimePredictor):
        def _get_estimator(self):
            raise AssertionError("not used by the dummy contract test")

        def _get_grid_search_params(self):
            raise AssertionError("not used by the dummy contract test")

    predictor = object.__new__(_ConcreteDummyPredictor)
    predictor._dummy_execution_time = 0.25
    predictor._model_config = BaseModelConfig.create_from_name(
        "Phi-tiny-MoE-instruct"
    )
    predictor._replica_config = SimpleNamespace(
        attn_tensor_parallel_size=2,
        moe_tensor_parallel_size=2,
        moe_expert_parallel_size=4,
        data_parallel_size=2,
        num_pipeline_stages=2,
    )
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._num_layers_per_pipeline_stage = 16
    moe_batch = SimpleNamespace(is_moe=True)
    distributed = predictor._get_dummy_execution_time(moe_batch, 0)
    per_stage_component = 0.25 * 16
    assert distributed.attention_all_reduce_time == per_stage_component
    assert distributed.mlp_all_reduce_time == per_stage_component
    assert distributed.expert_parallel_communication_time == per_stage_component
    assert distributed.dp_input_allreduce_time == per_stage_component
    assert distributed.dp_output_allreduce_time == per_stage_component
    assert distributed.pp_stage_boundary_handoff_time == 0.25

    predictor._replica_config = SimpleNamespace(
        attn_tensor_parallel_size=1,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=1,
        data_parallel_size=1,
        num_pipeline_stages=2,
    )
    local = predictor._get_dummy_execution_time(moe_batch, 1)
    assert local.attention_all_reduce_time == 0.0
    assert local.mlp_all_reduce_time == 0.0
    assert local.expert_parallel_communication_time == 0.0
    assert local.dp_input_allreduce_time == 0.0
    assert local.dp_output_allreduce_time == 0.0
    assert local.pp_stage_boundary_handoff_time == 0.0
