#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from frontier.config.config import (
    FixedRequestLengthGeneratorConfig,
    TraceRequestLengthGeneratorConfig,
    RoundRobinClusterSchedulerConfig,
    StickyLORClusterSchedulerConfig,
    StickyRoundRobinClusterSchedulerConfig,
    SyntheticRequestGeneratorConfig,
    TraceRequestGeneratorConfig,
    VllmV1SchedulerConfig,
)
from frontier.scheduler.cluster_scheduler.base_cluster_scheduler import (
    BaseClusterScheduler,
)
from frontier.types import ClusterType


class _DummyClusterScheduler(BaseClusterScheduler):
    def schedule(self):
        raise NotImplementedError


def _build_scheduler(
    *,
    num_replicas: int,
    replica_dp_size: int = 1,
    cluster_type: ClusterType,
    cluster_scheduler_config,
    replica_scheduler_config,
    request_generator_config=None,
):
    scheduler = object.__new__(_DummyClusterScheduler)
    scheduler._cluster_type = cluster_type
    scheduler._num_replicas = num_replicas
    scheduler._replica_dp_size = replica_dp_size
    scheduler._config = SimpleNamespace(
        cluster_scheduler_config=cluster_scheduler_config,
    )
    scheduler._request_generator_config = request_generator_config
    return scheduler


def test_prefix_cache_requires_sticky_scheduler_for_multi_replica_clusters() -> None:
    scheduler = _build_scheduler(
        num_replicas=2,
        cluster_type=ClusterType.MONOLITHIC,
        cluster_scheduler_config=RoundRobinClusterSchedulerConfig(),
        replica_scheduler_config=VllmV1SchedulerConfig(enable_prefix_caching=True),
    )

    with pytest.raises(ValueError, match="sticky"):
        scheduler._validate_prefix_cache_cluster_config(
            VllmV1SchedulerConfig(enable_prefix_caching=True)
        )


def test_prefix_cache_allows_sticky_scheduler_for_multi_replica_clusters() -> None:
    scheduler = _build_scheduler(
        num_replicas=2,
        cluster_type=ClusterType.PREFILL,
        cluster_scheduler_config=StickyRoundRobinClusterSchedulerConfig(),
        replica_scheduler_config=VllmV1SchedulerConfig(enable_prefix_caching=True),
    )

    scheduler._validate_prefix_cache_cluster_config(
        VllmV1SchedulerConfig(enable_prefix_caching=True)
    )


def test_prefix_cache_requires_sticky_scheduler_for_multiple_dp_lanes() -> None:
    scheduler = _build_scheduler(
        num_replicas=1,
        replica_dp_size=2,
        cluster_type=ClusterType.MONOLITHIC,
        cluster_scheduler_config=RoundRobinClusterSchedulerConfig(),
        replica_scheduler_config=VllmV1SchedulerConfig(enable_prefix_caching=True),
    )

    with pytest.raises(
        ValueError,
        match=r"replica/DP cache targets.*num_cache_targets=2",
    ):
        scheduler._validate_prefix_cache_cluster_config(
            VllmV1SchedulerConfig(enable_prefix_caching=True)
        )


def test_prefix_cache_allows_sticky_scheduler_for_multiple_dp_lanes() -> None:
    scheduler = _build_scheduler(
        num_replicas=1,
        replica_dp_size=2,
        cluster_type=ClusterType.MONOLITHIC,
        cluster_scheduler_config=StickyRoundRobinClusterSchedulerConfig(),
        replica_scheduler_config=VllmV1SchedulerConfig(enable_prefix_caching=True),
    )

    scheduler._validate_prefix_cache_cluster_config(
        VllmV1SchedulerConfig(enable_prefix_caching=True)
    )


def test_prefix_cache_allows_nonsticky_scheduler_for_single_cache_target() -> None:
    scheduler = _build_scheduler(
        num_replicas=1,
        replica_dp_size=1,
        cluster_type=ClusterType.MONOLITHIC,
        cluster_scheduler_config=RoundRobinClusterSchedulerConfig(),
        replica_scheduler_config=VllmV1SchedulerConfig(enable_prefix_caching=True),
    )

    scheduler._validate_prefix_cache_cluster_config(
        VllmV1SchedulerConfig(enable_prefix_caching=True)
    )


def test_pdd_prefix_cache_rejects_sticky_lor_before_runtime() -> None:
    scheduler = _build_scheduler(
        num_replicas=1,
        replica_dp_size=1,
        cluster_type=ClusterType.PREFILL,
        cluster_scheduler_config=StickyLORClusterSchedulerConfig(),
        replica_scheduler_config=VllmV1SchedulerConfig(enable_prefix_caching=True),
    )

    with pytest.raises(
        ValueError,
        match=r"only supported for MONOLITHIC.*use sticky_round_robin",
    ):
        scheduler._validate_prefix_cache_cluster_config(
            VllmV1SchedulerConfig(enable_prefix_caching=True)
        )


def test_monolithic_prefix_cache_allows_sticky_lor() -> None:
    scheduler = _build_scheduler(
        num_replicas=2,
        replica_dp_size=1,
        cluster_type=ClusterType.MONOLITHIC,
        cluster_scheduler_config=StickyLORClusterSchedulerConfig(),
        replica_scheduler_config=VllmV1SchedulerConfig(enable_prefix_caching=True),
    )

    scheduler._validate_prefix_cache_cluster_config(
        VllmV1SchedulerConfig(enable_prefix_caching=True)
    )


def test_prefix_cache_rejects_plain_synthetic_request_source_before_scheduling() -> None:
    scheduler = _build_scheduler(
        num_replicas=1,
        cluster_type=ClusterType.PREFILL,
        cluster_scheduler_config=StickyRoundRobinClusterSchedulerConfig(),
        replica_scheduler_config=VllmV1SchedulerConfig(enable_prefix_caching=True),
        request_generator_config=SyntheticRequestGeneratorConfig(
            length_generator_config=FixedRequestLengthGeneratorConfig(
                prefill_tokens=32,
                decode_tokens=8,
            )
        ),
    )

    with pytest.raises(ValueError, match="session_id.*block_hash_ids"):
        scheduler._validate_prefix_cache_cluster_config(
            VllmV1SchedulerConfig(enable_prefix_caching=True)
        )


def test_prefix_cache_rejects_synthetic_trace_length_source_before_scheduling(
    tmp_path: Path,
) -> None:
    trace_file = tmp_path / "synthetic_length_trace.csv"
    trace_file.write_text(
        "num_prefill_tokens,num_decode_tokens,session_id,block_hash_ids\n"
        "32,8,7,11|22\n",
        encoding="utf-8",
    )
    scheduler = _build_scheduler(
        num_replicas=1,
        cluster_type=ClusterType.PREFILL,
        cluster_scheduler_config=StickyRoundRobinClusterSchedulerConfig(),
        replica_scheduler_config=VllmV1SchedulerConfig(enable_prefix_caching=True),
        request_generator_config=SyntheticRequestGeneratorConfig(
            length_generator_config=TraceRequestLengthGeneratorConfig(
                trace_file=str(trace_file),
            )
        ),
    )

    with pytest.raises(ValueError, match="trace request source"):
        scheduler._validate_prefix_cache_cluster_config(
            VllmV1SchedulerConfig(enable_prefix_caching=True)
        )


def test_prefix_cache_allows_trace_request_source_with_prefix_metadata(
    tmp_path: Path,
) -> None:
    trace_file = tmp_path / "prefix_trace.csv"
    trace_file.write_text(
        "arrived_at,num_prefill_tokens,num_decode_tokens,session_id,block_hash_ids\n"
        "0.0,32,8,7,11|22\n",
        encoding="utf-8",
    )
    scheduler = _build_scheduler(
        num_replicas=1,
        cluster_type=ClusterType.PREFILL,
        cluster_scheduler_config=StickyRoundRobinClusterSchedulerConfig(),
        replica_scheduler_config=VllmV1SchedulerConfig(enable_prefix_caching=True),
        request_generator_config=TraceRequestGeneratorConfig(
            trace_file=str(trace_file)
        ),
    )

    scheduler._validate_prefix_cache_cluster_config(
        VllmV1SchedulerConfig(enable_prefix_caching=True)
    )


def test_prefix_cache_rejects_trace_request_source_without_block_hash_ids(
    tmp_path: Path,
) -> None:
    trace_file = tmp_path / "missing_prefix_trace.csv"
    trace_file.write_text(
        "arrived_at,num_prefill_tokens,num_decode_tokens,session_id\n"
        "0.0,32,8,7\n",
        encoding="utf-8",
    )
    scheduler = _build_scheduler(
        num_replicas=1,
        cluster_type=ClusterType.PREFILL,
        cluster_scheduler_config=StickyRoundRobinClusterSchedulerConfig(),
        replica_scheduler_config=VllmV1SchedulerConfig(enable_prefix_caching=True),
        request_generator_config=TraceRequestGeneratorConfig(
            trace_file=str(trace_file)
        ),
    )

    with pytest.raises(ValueError, match="block_hash_ids"):
        scheduler._validate_prefix_cache_cluster_config(
            VllmV1SchedulerConfig(enable_prefix_caching=True)
        )


@pytest.mark.parametrize(
    ("row", "missing_column"),
    [
        ("0.0,32,8,,11|22\n", "session_id"),
        ("0.0,32,8,7,\n", "block_hash_ids"),
    ],
)
def test_prefix_cache_rejects_trace_request_source_with_empty_metadata_values(
    tmp_path: Path,
    row: str,
    missing_column: str,
) -> None:
    trace_file = tmp_path / "empty_prefix_metadata_trace.csv"
    trace_file.write_text(
        "arrived_at,num_prefill_tokens,num_decode_tokens,session_id,block_hash_ids\n"
        + row,
        encoding="utf-8",
    )
    scheduler = _build_scheduler(
        num_replicas=1,
        cluster_type=ClusterType.PREFILL,
        cluster_scheduler_config=StickyRoundRobinClusterSchedulerConfig(),
        replica_scheduler_config=VllmV1SchedulerConfig(enable_prefix_caching=True),
        request_generator_config=TraceRequestGeneratorConfig(
            trace_file=str(trace_file)
        ),
    )

    with pytest.raises(ValueError, match=fr"row 2.*{missing_column}"):
        scheduler._validate_prefix_cache_cluster_config(
            VllmV1SchedulerConfig(enable_prefix_caching=True)
        )


def test_session_prefix_cache_allows_trace_without_block_hash_ids(
    tmp_path: Path,
) -> None:
    trace_file = tmp_path / "session_prefix_trace.csv"
    trace_file.write_text(
        "arrived_at,num_prefill_tokens,num_decode_tokens,session_id\n"
        "0.0,32,16,7\n"
        "10.0,8,8,7\n",
        encoding="utf-8",
    )
    scheduler = _build_scheduler(
        num_replicas=1,
        cluster_type=ClusterType.MONOLITHIC,
        cluster_scheduler_config=StickyRoundRobinClusterSchedulerConfig(),
        replica_scheduler_config=VllmV1SchedulerConfig(
            enable_prefix_caching=True,
            prefix_caching_key_mode="session",
        ),
        request_generator_config=TraceRequestGeneratorConfig(
            trace_file=str(trace_file)
        ),
    )

    scheduler._validate_prefix_cache_cluster_config(
        VllmV1SchedulerConfig(
            enable_prefix_caching=True,
            prefix_caching_key_mode="session",
        )
    )


@pytest.mark.parametrize(
    ("rows", "error_pattern"),
    [
        (
            "10.0,32,16,7\n"
            "5.0,8,8,7\n",
            "nondecreasing arrived_at",
        ),
    ],
)
def test_session_prefix_cache_rejects_invalid_turn_sequence(
    tmp_path: Path,
    rows: str,
    error_pattern: str,
) -> None:
    trace_file = tmp_path / "invalid_session_prefix_trace.csv"
    trace_file.write_text(
        "arrived_at,num_prefill_tokens,num_decode_tokens,session_id\n" + rows,
        encoding="utf-8",
    )
    scheduler = _build_scheduler(
        num_replicas=1,
        cluster_type=ClusterType.MONOLITHIC,
        cluster_scheduler_config=StickyRoundRobinClusterSchedulerConfig(),
        replica_scheduler_config=VllmV1SchedulerConfig(
            enable_prefix_caching=True,
            prefix_caching_key_mode="session",
        ),
        request_generator_config=TraceRequestGeneratorConfig(
            trace_file=str(trace_file)
        ),
    )

    with pytest.raises(ValueError, match=error_pattern):
        scheduler._validate_prefix_cache_cluster_config(
            VllmV1SchedulerConfig(
                enable_prefix_caching=True,
                prefix_caching_key_mode="session",
            )
        )


def test_session_prefix_cache_rejects_expanded_context_overflow(
    tmp_path: Path,
) -> None:
    trace_file = tmp_path / "clipped_session_prefix_trace.csv"
    trace_file.write_text(
        "arrived_at,num_prefill_tokens,num_decode_tokens,session_id\n"
        "0.0,32,16,7\n"
        "10.0,16,8,7\n",
        encoding="utf-8",
    )
    scheduler = _build_scheduler(
        num_replicas=1,
        cluster_type=ClusterType.MONOLITHIC,
        cluster_scheduler_config=StickyRoundRobinClusterSchedulerConfig(),
        replica_scheduler_config=VllmV1SchedulerConfig(
            enable_prefix_caching=True,
            prefix_caching_key_mode="session",
        ),
        request_generator_config=TraceRequestGeneratorConfig(
            trace_file=str(trace_file),
            max_tokens=64,
        ),
    )

    with pytest.raises(ValueError, match="incremental ISL expands beyond max_tokens"):
        scheduler._validate_prefix_cache_cluster_config(
            VllmV1SchedulerConfig(
                enable_prefix_caching=True,
                prefix_caching_key_mode="session",
            )
        )


def test_session_prefix_cache_accepts_incremental_thinking_round_isls(
    tmp_path: Path,
) -> None:
    trace_file = tmp_path / "thinking_session_prefix_trace.csv"
    trace_file.write_text(
        "arrived_at,num_prefill_tokens,num_decode_tokens,session_id,"
        "thinking_depth,thinking_round_plans_json\n"
        '0.0,32,8,7,2,"[{""num_prefill_tokens"":48,'
        '""num_decode_tokens"":16},{""num_prefill_tokens"":32,'
        '""num_decode_tokens"":8}]"\n',
        encoding="utf-8",
    )
    scheduler = _build_scheduler(
        num_replicas=1,
        cluster_type=ClusterType.MONOLITHIC,
        cluster_scheduler_config=StickyRoundRobinClusterSchedulerConfig(),
        replica_scheduler_config=VllmV1SchedulerConfig(
            enable_prefix_caching=True,
            prefix_caching_key_mode="session",
        ),
        request_generator_config=TraceRequestGeneratorConfig(
            trace_file=str(trace_file),
            max_tokens=128,
        ),
    )

    scheduler._validate_prefix_cache_cluster_config(
        VllmV1SchedulerConfig(
            enable_prefix_caching=True,
            prefix_caching_key_mode="session",
        )
    )


def test_session_prefix_cache_scales_explicit_thinking_round_plans(
    tmp_path: Path,
) -> None:
    trace_file = tmp_path / "scaled_thinking_session_prefix_trace.csv"
    trace_file.write_text(
        "arrived_at,num_prefill_tokens,num_decode_tokens,session_id,"
        "thinking_depth,thinking_round_plans_json\n"
        '2.0,32,8,7,2,"[{""num_prefill_tokens"":48,'
        '""num_decode_tokens"":16},{""num_prefill_tokens"":32,'
        '""num_decode_tokens"":8}]"\n',
        encoding="utf-8",
    )
    request_generator_config = TraceRequestGeneratorConfig(
        trace_file=str(trace_file),
        max_tokens=128,
        prefill_scale_factor=0.5,
        decode_scale_factor=0.5,
        time_scale_factor=2.0,
    )
    scheduler = _build_scheduler(
        num_replicas=1,
        cluster_type=ClusterType.MONOLITHIC,
        cluster_scheduler_config=StickyRoundRobinClusterSchedulerConfig(),
        replica_scheduler_config=VllmV1SchedulerConfig(
            enable_prefix_caching=True,
            prefix_caching_key_mode="session",
        ),
        request_generator_config=request_generator_config,
    )

    scheduler._validate_prefix_cache_cluster_config(
        VllmV1SchedulerConfig(
            enable_prefix_caching=True,
            prefix_caching_key_mode="session",
        )
    )


@pytest.mark.parametrize(
    ("arrived_at", "prefill_tokens", "decode_tokens", "error_pattern"),
    [
        ("nan", 32, 8, "arrived_at must be finite"),
        ("inf", 32, 8, "arrived_at must be finite"),
        (-1, 32, 8, "arrived_at must be nonnegative"),
        (0, 0, 8, "num_prefill_tokens must be positive"),
        (0, -1, 8, "num_prefill_tokens must be positive"),
        (0, 32, 0, "num_decode_tokens must be positive"),
        (0, 32, "-inf", "num_decode_tokens must be finite"),
    ],
)
def test_session_prefix_cache_rejects_nonfinite_or_nonpositive_trace_values(
    tmp_path: Path,
    arrived_at: object,
    prefill_tokens: object,
    decode_tokens: object,
    error_pattern: str,
) -> None:
    trace_file = tmp_path / "invalid_strict_session_prefix_trace.csv"
    trace_file.write_text(
        "arrived_at,num_prefill_tokens,num_decode_tokens,session_id\n"
        f"{arrived_at},{prefill_tokens},{decode_tokens},7\n",
        encoding="utf-8",
    )
    request_generator_config = TraceRequestGeneratorConfig(
        trace_file=str(trace_file),
        max_tokens=128,
    )
    scheduler = _build_scheduler(
        num_replicas=1,
        cluster_type=ClusterType.MONOLITHIC,
        cluster_scheduler_config=StickyRoundRobinClusterSchedulerConfig(),
        replica_scheduler_config=VllmV1SchedulerConfig(
            enable_prefix_caching=True,
            prefix_caching_key_mode="session",
        ),
        request_generator_config=request_generator_config,
    )

    with pytest.raises(ValueError, match=error_pattern):
        scheduler._validate_prefix_cache_cluster_config(
            VllmV1SchedulerConfig(
                enable_prefix_caching=True,
                prefix_caching_key_mode="session",
            )
        )


def test_prefix_cache_key_mode_validation() -> None:
    with pytest.raises(ValueError, match="prefix_caching_key_mode"):
        VllmV1SchedulerConfig(prefix_caching_key_mode="unknown")


@pytest.mark.parametrize(
    "kv_cache_dtype",
    ["auto", "fp32", "fp16", "bf16", "fp8", "int8", "fp4", "int4"],
)
def test_vllm_v1_accepts_supported_kv_cache_dtypes(
    kv_cache_dtype: str,
) -> None:
    config = VllmV1SchedulerConfig(kv_cache_dtype=kv_cache_dtype)
    assert config.kv_cache_dtype == kv_cache_dtype


def test_vllm_v1_rejects_unknown_kv_cache_dtype() -> None:
    with pytest.raises(ValueError, match="Unsupported precision type"):
        VllmV1SchedulerConfig(kv_cache_dtype="fp3")
