from __future__ import annotations

from types import SimpleNamespace

import pytest

from frontier.attention.families import (
    DENSE_ATTENTION_FAMILY,
    DSA_ATTENTION_FAMILY,
    LATENT_MLA_ATTENTION_FAMILY,
)
from frontier.attention.memory import (
    AttentionRuntimeKVLayout,
    get_attention_runtime_kv_layout,
)
from frontier.scheduler.utils.memory_planner import MemoryPlanner
from frontier.config.precision_type import PrecisionType
from frontier.types import ClusterType


def test_dense_kv_layout_matches_vllm_page_bytes_for_gqa_mha_mqa() -> None:
    layout = get_attention_runtime_kv_layout(
        DENSE_ATTENTION_FAMILY,
        runtime_num_kv_heads_per_worker=4,
        runtime_head_size=96,
    )

    assert layout == AttentionRuntimeKVLayout(
        family_id="dense_attention",
        kv_factor=2,
        runtime_num_kv_heads_per_worker=4,
        runtime_head_size=96,
        bytes_per_element=2,
    )
    assert layout.elements_per_token_per_worker == 2 * 4 * 96
    assert layout.page_bytes(block_size=16) == 24576


def test_latent_mla_layout_matches_vllm_runtime_groundtruth_page_bytes() -> None:
    layout = get_attention_runtime_kv_layout(
        LATENT_MLA_ATTENTION_FAMILY,
        runtime_num_kv_heads_per_worker=1,
        runtime_head_size=576,
    )

    assert layout.kv_factor == 1
    assert layout.elements_per_token_per_worker == 1 * 1 * 576
    assert layout.page_bytes(block_size=64) == 73728


def test_attention_runtime_layout_rejects_invalid_dimensions() -> None:
    with pytest.raises(ValueError, match="runtime_head_size"):
        get_attention_runtime_kv_layout(
            DENSE_ATTENTION_FAMILY,
            runtime_num_kv_heads_per_worker=4,
            runtime_head_size=0,
        )
    with pytest.raises(ValueError, match="runtime_num_kv_heads_per_worker"):
        get_attention_runtime_kv_layout(
            DENSE_ATTENTION_FAMILY,
            runtime_num_kv_heads_per_worker=0,
            runtime_head_size=96,
        )


def test_frozen_dsa_layout_fails_fast() -> None:
    with pytest.raises(NotImplementedError, match="DSA attention is frozen"):
        get_attention_runtime_kv_layout(
            DSA_ATTENTION_FAMILY,
            runtime_num_kv_heads_per_worker=1,
            runtime_head_size=128,
        )


def test_decode_ffn_memory_planner_keeps_zero_kv_elements_without_binding() -> None:
    planner = object.__new__(MemoryPlanner)
    planner._cluster_type = ClusterType.DECODE_FFN
    planner._replica_config = object()
    planner._replica = object()

    assert planner._get_kv_cache_elements_per_token_per_worker() == 0


def test_public_kv_block_sizing_matches_gpu_memory_planner_layer_semantics() -> None:
    planner = object.__new__(MemoryPlanner)
    planner._replica = SimpleNamespace(num_layers=24)
    planner._get_kv_cache_memory_per_layer_per_block = lambda block_size: (
        4096 * block_size
    )

    assert planner.get_kv_cache_memory_per_block_bytes(16) == 24 * 4096 * 16


def test_cpu_kv_block_sizing_aggregates_attention_tp_workers_only() -> None:
    planner = object.__new__(MemoryPlanner)
    planner._replica = SimpleNamespace(
        num_layers=24,
        num_attn_tensor_parallel_workers=4,
    )
    planner._get_kv_cache_memory_per_layer_per_block = lambda block_size: (
        4096 * block_size
    )

    per_worker = 24 * 4096 * 16
    assert planner.get_kv_cache_memory_per_block_bytes(16) == per_worker
    assert (
        planner.get_cpu_kv_cache_memory_per_block_bytes(16)
        == per_worker * 4
    )


@pytest.mark.parametrize(
    ("precision", "expected_bytes"),
    [
        (PrecisionType.FP32, 4 * 16 * 3),
        (PrecisionType.FP16, 2 * 16 * 3),
        (PrecisionType.FP8, 16 * 3),
        (PrecisionType.FP4, 24),
    ],
)
def test_memory_planner_kv_block_sizing_uses_runtime_precision(
    precision: PrecisionType,
    expected_bytes: int,
) -> None:
    planner = object.__new__(MemoryPlanner)
    planner._cluster_type = ClusterType.PREFILL
    planner._kv_cache_precision = precision
    planner._get_kv_cache_elements_per_token_per_worker = lambda: 3

    assert (
        planner._get_kv_cache_memory_per_layer_per_block(16)
        == expected_bytes
    )


def test_attention_layout_supports_packed_subbyte_kv_precision() -> None:
    layout = get_attention_runtime_kv_layout(
        DENSE_ATTENTION_FAMILY,
        runtime_num_kv_heads_per_worker=1,
        runtime_head_size=3,
        bytes_per_element=0.5,
    )

    assert layout.page_bytes(block_size=3) == 9
