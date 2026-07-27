from __future__ import annotations

from types import SimpleNamespace

import pytest

from frontier.scheduler.replica_scheduler import (
    sglang_style_replica_scheduler as sglang_scheduler_module,
)
from frontier.scheduler.replica_scheduler import (
    vllm_v1_engine_replica_scheduler as vllm_scheduler_module,
)
from frontier.entities import Request, RequestRoundPlan
from frontier.kv_cache.replica_kv_cache_manager import ReplicaKVCacheManager
from frontier.scheduler.replica_scheduler.sglang_style_replica_scheduler import (
    SGLangStyleReplicaScheduler,
)
from frontier.scheduler.replica_scheduler.sj2q_bounded_carryover_replica_scheduler import (
    SJ2QBoundedCarryoverReplicaScheduler,
)
from frontier.scheduler.replica_scheduler.sj2q_fastserve_lite_replica_scheduler import (
    SJ2QFastServeLiteReplicaScheduler,
)
from frontier.scheduler.replica_scheduler.sj2q_penalty_only_replica_scheduler import (
    SJ2QPenaltyOnlyReplicaScheduler,
)
from frontier.scheduler.replica_scheduler.vllm_v1_engine_replica_scheduler import (
    VLLMv1EngineReplicaScheduler,
)
from frontier.types import ClusterType


def _manager(*, key_mode: str = "session", num_blocks: int = 16):
    return ReplicaKVCacheManager(
        block_size=16,
        num_gpu_blocks=num_blocks,
        enable_caching=True,
        caching_hash_algo="builtin",
        caching_key_mode=key_mode,
        num_preallocate_tokens=0,
    )


def _request(
    *,
    session_id: int | None,
    prefill_tokens: int,
    decode_tokens: int,
    block_hash_ids: list[int] | None = None,
) -> Request:
    return Request(
        arrived_at=0.0,
        num_prefill_tokens=prefill_tokens,
        num_decode_tokens=decode_tokens,
        session_id=session_id,
        block_hash_ids=block_hash_ids,
    )


def _complete_tokens(
    manager: ReplicaKVCacheManager,
    request: Request,
    num_processed_tokens: int,
) -> None:
    request._num_processed_tokens = num_processed_tokens
    manager.mark_blocks_computed(request)


def test_session_mode_reuses_only_full_prompt_blocks() -> None:
    manager = _manager()
    first = _request(session_id=7, prefill_tokens=34, decode_tokens=8)

    assert manager.allocate_slots(first, 34) is not None
    _complete_tokens(manager, first, 34)
    manager.free(first)

    second = _request(session_id=7, prefill_tokens=42, decode_tokens=4)
    blocks, cached_tokens, query_blocks = manager.get_computed_blocks(second)

    assert query_blocks == 2
    assert len(blocks) == 2
    assert cached_tokens == 32


def test_session_mode_never_reuses_across_sessions() -> None:
    manager = _manager()
    first = _request(session_id=7, prefill_tokens=32, decode_tokens=8)
    assert manager.allocate_slots(first, 32) is not None
    _complete_tokens(manager, first, 32)
    manager.free(first)

    other_session = _request(session_id=8, prefill_tokens=32, decode_tokens=8)
    blocks, cached_tokens, query_blocks = manager.get_computed_blocks(other_session)

    assert query_blocks == 2
    assert blocks == []
    assert cached_tokens == 0


def test_monolithic_decode_full_blocks_extend_session_cache() -> None:
    manager = _manager()
    first = _request(session_id=7, prefill_tokens=32, decode_tokens=16)

    assert manager.allocate_slots(first, 32) is not None
    _complete_tokens(manager, first, 32)
    assert manager.allocate_slots(first, 16) is not None
    _complete_tokens(manager, first, 48)
    manager.free(first)

    second = _request(session_id=7, prefill_tokens=56, decode_tokens=8)
    blocks, cached_tokens, query_blocks = manager.get_computed_blocks(second)

    assert query_blocks == 3
    assert len(blocks) == 3
    assert cached_tokens == 48


def test_prefill_only_processing_does_not_invent_decode_cache_blocks() -> None:
    manager = _manager()
    first = _request(session_id=7, prefill_tokens=32, decode_tokens=16)

    assert manager.allocate_slots(first, 32) is not None
    _complete_tokens(manager, first, 32)
    manager.free(first)

    second = _request(session_id=7, prefill_tokens=56, decode_tokens=8)
    blocks, cached_tokens, query_blocks = manager.get_computed_blocks(second)

    assert query_blocks == 3
    assert len(blocks) == 2
    assert cached_tokens == 32


def test_session_mode_requires_session_id() -> None:
    manager = _manager()
    request = _request(session_id=None, prefill_tokens=32, decode_tokens=8)

    with pytest.raises(ValueError, match="session_id"):
        manager.get_computed_blocks(request)


def test_prefix_lookup_stops_at_first_missing_block() -> None:
    manager = _manager(key_mode="block_hash")
    first = _request(
        session_id=7,
        prefill_tokens=32,
        decode_tokens=8,
        block_hash_ids=[11, 22],
    )
    assert manager.allocate_slots(first, 32) is not None
    _complete_tokens(manager, first, 32)
    manager.free(first)

    second = _request(
        session_id=7,
        prefill_tokens=48,
        decode_tokens=8,
        block_hash_ids=[11, 99, 22],
    )
    blocks, cached_tokens, query_blocks = manager.get_computed_blocks(second)

    assert query_blocks == 3
    assert len(blocks) == 1
    assert cached_tokens == 16


def test_evicted_session_blocks_no_longer_hit() -> None:
    manager = _manager(num_blocks=2)
    first = _request(session_id=7, prefill_tokens=32, decode_tokens=8)
    assert manager.allocate_slots(first, 32) is not None
    _complete_tokens(manager, first, 32)
    manager.free(first)

    replacement = _request(session_id=8, prefill_tokens=32, decode_tokens=8)
    assert manager.allocate_slots(replacement, 32) is not None
    _complete_tokens(manager, replacement, 32)
    manager.free(replacement)

    retry = _request(session_id=7, prefill_tokens=32, decode_tokens=8)
    blocks, cached_tokens, query_blocks = manager.get_computed_blocks(retry)

    assert query_blocks == 2
    assert blocks == []
    assert cached_tokens == 0


def test_block_hash_mode_keeps_explicit_hash_behavior() -> None:
    manager = _manager(key_mode="block_hash")
    first = _request(
        session_id=7,
        prefill_tokens=32,
        decode_tokens=8,
        block_hash_ids=[11, 22],
    )
    assert manager.allocate_slots(first, 32) is not None
    _complete_tokens(manager, first, 32)
    manager.free(first)

    second = _request(
        session_id=99,
        prefill_tokens=32,
        decode_tokens=8,
        block_hash_ids=[11, 22],
    )
    blocks, cached_tokens, query_blocks = manager.get_computed_blocks(second)

    assert query_blocks == 2
    assert len(blocks) == 2
    assert cached_tokens == 32


def test_request_records_zero_hit_session_lookup() -> None:
    request = _request(session_id=7, prefill_tokens=32, decode_tokens=8)

    request.on_prefix_cache_lookup(
        query_blocks=2,
        hit_blocks=0,
        num_tokens_cached=0,
        key_mode="session",
    )

    assert request.prefix_cache_query_blocks == 2
    assert request.prefix_cache_hit_blocks == 0
    assert request.prefix_cache_key_mode == "session"
    assert request.num_prefill_tokens_cached == 0


def test_thinking_round_prefix_cache_metrics_accumulate_without_preemption_double_count(
) -> None:
    request = Request(
        arrived_at=0.0,
        num_prefill_tokens=96,
        num_decode_tokens=8,
        session_id=7,
        thinking_depth=2,
        thinking_round_plans=[
            RequestRoundPlan(48, 16),
            RequestRoundPlan(96, 8),
        ],
    )

    request.on_prefix_cache_lookup(
        query_blocks=3,
        hit_blocks=0,
        num_tokens_cached=0,
        key_mode="session",
    )
    request._reset_runtime_state_for_next_thinking_round()
    request.advance_thinking_round()
    request.on_prefix_cache_lookup(
        query_blocks=6,
        hit_blocks=3,
        num_tokens_cached=48,
        key_mode="session",
    )

    assert request.prefix_cache_query_blocks == 6
    assert request.prefix_cache_hit_blocks == 3
    assert request.num_prefill_tokens_cached == 48
    assert request.total_prefix_cache_query_blocks == 9
    assert request.total_prefix_cache_hit_blocks == 3
    assert request.total_num_prefill_tokens_cached == 48

    request._num_processed_tokens = 0
    request.on_prefix_cache_lookup(
        query_blocks=6,
        hit_blocks=3,
        num_tokens_cached=48,
        key_mode="session",
    )

    assert request.num_processed_tokens == 48
    assert request.total_prefix_cache_query_blocks == 9
    assert request.total_prefix_cache_hit_blocks == 3
    assert request.total_num_prefill_tokens_cached == 48


def test_scheduled_blocks_are_not_visible_until_batch_completion() -> None:
    manager = _manager()
    source = _request(session_id=7, prefill_tokens=32, decode_tokens=8)
    follower = _request(session_id=7, prefill_tokens=32, decode_tokens=8)

    assert manager.allocate_slots(source, 16) is not None
    blocks, cached_tokens, _ = manager.get_computed_blocks(follower)
    assert blocks == []
    assert cached_tokens == 0

    _complete_tokens(manager, source, 16)
    blocks, cached_tokens, _ = manager.get_computed_blocks(follower)
    assert len(blocks) == 1
    assert cached_tokens == 16


def test_chunked_prefill_publishes_only_completed_frontier() -> None:
    manager = _manager()
    source = _request(session_id=7, prefill_tokens=48, decode_tokens=8)
    follower = _request(session_id=7, prefill_tokens=48, decode_tokens=8)

    assert manager.allocate_slots(source, 16) is not None
    _complete_tokens(manager, source, 16)
    assert manager.get_computed_blocks(follower)[1] == 16

    assert manager.allocate_slots(source, 16) is not None
    assert manager.get_computed_blocks(follower)[1] == 16

    _complete_tokens(manager, source, 32)
    assert manager.get_computed_blocks(follower)[1] == 32


def test_mark_blocks_computed_generates_only_new_session_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager(num_blocks=2)
    request = _request(
        session_id=7,
        prefill_tokens=128 * 1024,
        decode_tokens=1,
    )
    assert manager.allocate_slots(request, 16) is not None

    generated_ranges: list[tuple[int, int]] = []
    original_get_session_block_keys = manager._get_session_block_keys

    def _tracking_get_session_block_keys(
        tracked_request: Request,
        num_blocks: int,
        start_block_index: int = 0,
    ) -> list[object]:
        generated_ranges.append((num_blocks, start_block_index))
        return original_get_session_block_keys(
            tracked_request,
            num_blocks,
            start_block_index=start_block_index,
        )

    monkeypatch.setattr(
        manager,
        "_get_session_block_keys",
        _tracking_get_session_block_keys,
    )

    for _ in range(1_000):
        manager.mark_blocks_computed(request)
    assert generated_ranges == []

    _complete_tokens(manager, request, 16)
    assert generated_ranges == [(1, 0)]

    for _ in range(1_000):
        manager.mark_blocks_computed(request)
    assert generated_ranges == [(1, 0)]


def test_prefix_cache_diagnostics_count_only_successful_admissions() -> None:
    manager = _manager()
    request = _request(session_id=7, prefill_tokens=32, decode_tokens=8)

    manager.get_computed_blocks(request)
    manager.get_computed_blocks(request)

    stats = manager.prefix_cache_stats
    assert stats.admissions == 0
    assert stats.queries == 0
    assert stats.hits == 0

    manager.record_prefix_cache_admission(query_blocks=2, hit_blocks=0)

    assert stats.admissions == 1
    assert stats.requests == 1
    assert stats.queries == 2
    assert stats.hits == 0


@pytest.mark.parametrize(
    ("scheduler_class", "scheduler_module"),
    [
        (VLLMv1EngineReplicaScheduler, vllm_scheduler_module),
        (SGLangStyleReplicaScheduler, sglang_scheduler_module),
    ],
)
def test_scheduler_diagnostics_label_admission_scoped_prefix_cache_stats(
    monkeypatch: pytest.MonkeyPatch,
    scheduler_class: type[VLLMv1EngineReplicaScheduler],
    scheduler_module: object,
) -> None:
    manager = _manager()
    manager.record_prefix_cache_admission(query_blocks=2, hit_blocks=1)

    scheduler = object.__new__(scheduler_class)
    scheduler._cluster_type = ClusterType.MONOLITHIC
    scheduler._config = SimpleNamespace(num_blocks=16, block_size=16)
    scheduler._num_allocated_blocks = 0
    scheduler._active_schedule_iteration_id = 1
    scheduler._running_requests = []
    scheduler._request_queue = []
    scheduler._preempted_requests = []
    scheduler._max_num_running_reqs = 8
    scheduler._max_num_scheduled_tokens = 128
    scheduler._current_schedule_time = 1.0
    scheduler._kv_cache_manager = manager

    payloads: list[dict[str, object]] = []
    monkeypatch.setattr(
        scheduler_module,
        "_log_frontier_vllm_v1_schedule_decision",
        payloads.append,
    )
    if scheduler_module is vllm_scheduler_module:
        monkeypatch.setattr(
            scheduler_module,
            "_frontier_vllm_v1_sched_decision_logger",
            object(),
        )

    scheduler._emit_schedule_decision_event(
        event="decision",
        decision_result="ADMISSION",
        request_id=1,
        token_budget=126,
        num_tokens=2,
    )

    assert len(payloads) == 1
    payload = payloads[0]
    assert payload["prefix_cache_metric_semantics"] == (
        "successful_admission_block_level"
    )
    assert payload["prefix_cache_admissions"] == 1
    assert payload["prefix_cache_queries"] == 2
    assert payload["prefix_cache_hits"] == 1
    assert "prefix_cache_requests" not in payload


def test_ref_count_zero_cached_block_is_touched_on_admission() -> None:
    manager = _manager()
    source = _request(session_id=7, prefill_tokens=16, decode_tokens=8)
    assert manager.allocate_slots(source, 16) is not None
    _complete_tokens(manager, source, 16)
    manager.free(source)

    follower = _request(session_id=7, prefill_tokens=32, decode_tokens=8)
    computed_blocks, _, _ = manager.get_computed_blocks(follower)
    assert computed_blocks[0].ref_cnt == 0

    assert (
        manager.allocate_slots(
            follower,
            16,
            new_computed_blocks=computed_blocks,
        )
        is not None
    )
    assert computed_blocks[0].ref_cnt == 1


def test_preemption_reentry_restores_runtime_cache_without_rewriting_metrics() -> None:
    manager = _manager()
    request = _request(session_id=7, prefill_tokens=48, decode_tokens=8)

    computed_blocks, cached_tokens, query_blocks = manager.get_computed_blocks(
        request
    )
    request.on_prefix_cache_lookup(
        query_blocks=query_blocks,
        hit_blocks=len(computed_blocks),
        num_tokens_cached=cached_tokens,
        key_mode="session",
    )
    assert manager.allocate_slots(request, 32) is not None
    _complete_tokens(manager, request, 32)

    scheduler = object.__new__(VLLMv1EngineReplicaScheduler)
    scheduler._cluster_type = ClusterType.MONOLITHIC
    scheduler._allocation_map = {
        request.id: manager.get_num_blocks_for_request(request)
    }
    scheduler._running_requests = [request]
    scheduler._scheduled_num_computed_tokens_by_request = {request.id: 32}
    scheduler._current_schedule_time = 1.0
    scheduler._request_queue = []
    scheduler._scheduling_policy = "fcfs"
    scheduler._config = SimpleNamespace(num_blocks=16)
    scheduler._num_allocated_blocks = 2
    scheduler._current_iteration_token_budget = 128
    scheduler._emit_schedule_decision_event = lambda **_: None

    def _free_request_resources(victim: Request) -> None:
        manager.free(victim)
        scheduler._allocation_map.pop(victim.id, None)
        scheduler._num_allocated_blocks = 0

    scheduler._free_request_resources = _free_request_resources
    preempted_requests: list[Request] = []
    scheduler._preempt_request(request, preempted_requests)

    assert preempted_requests == [request]
    assert request.num_processed_tokens == 0

    computed_blocks, cached_tokens, query_blocks = manager.get_computed_blocks(
        request
    )
    assert cached_tokens == 32
    request.on_prefix_cache_lookup(
        query_blocks=query_blocks,
        hit_blocks=len(computed_blocks),
        num_tokens_cached=cached_tokens,
        key_mode="session",
    )

    assert request.num_processed_tokens == 32
    assert request.num_prefill_tokens_cached == 0
    assert request.prefix_cache_query_blocks == 3
    assert request.prefix_cache_hit_blocks == 0


@pytest.mark.parametrize(
    "scheduler_class",
    [
        VLLMv1EngineReplicaScheduler,
        SGLangStyleReplicaScheduler,
        SJ2QFastServeLiteReplicaScheduler,
        SJ2QPenaltyOnlyReplicaScheduler,
        SJ2QBoundedCarryoverReplicaScheduler,
    ],
)
def test_fully_cached_prompt_recomputes_one_block_on_inherited_scheduler_paths(
    scheduler_class: type[VLLMv1EngineReplicaScheduler],
) -> None:
    manager = _manager()
    source = _request(session_id=7, prefill_tokens=32, decode_tokens=8)
    assert manager.allocate_slots(source, 32) is not None
    _complete_tokens(manager, source, 32)
    manager.free(source)

    scheduler = object.__new__(scheduler_class)
    scheduler._kv_cache_manager = manager
    scheduler._config = SimpleNamespace(block_size=16)
    target = _request(session_id=7, prefill_tokens=32, decode_tokens=8)

    computed, cached_tokens, new_tokens, query_blocks, hit_blocks = (
        scheduler._prepare_prefix_cache_admission(target)
    )

    assert len(computed) == 1
    assert cached_tokens == 16
    assert new_tokens == 16
    assert query_blocks == 2
    assert hit_blocks == 1
