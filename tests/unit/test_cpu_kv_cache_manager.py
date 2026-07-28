from __future__ import annotations

import pytest

from frontier.config.cpu_kv_cache_config import CPUKVCacheConfig
from frontier.cpu_kv_cache_transfer import AnalyticalCPUKVCacheTransferEngine
from frontier.kv_cache import CPUKVCacheManager


def _manager(
    *,
    num_blocks: int = 8,
    bytes_per_block: int = 1024,
    policy: str = "prefix_fit",
) -> CPUKVCacheManager:
    return CPUKVCacheManager(
        capacity_bytes=num_blocks * bytes_per_block,
        bytes_per_block=bytes_per_block,
        capacity_pressure_policy=policy,
    )


def _commit(
    manager: CPUKVCacheManager,
    *,
    session_id: int,
    frontier: int,
    generation: int,
    time: float,
):
    reservation = manager.reserve_offload(
        session_id=session_id,
        desired_frontier_blocks=frontier,
        generation=generation,
        time=time,
    )
    manager.commit_offload(reservation, time=time + 0.1)
    return reservation


def test_cpu_store_commits_only_full_contiguous_frontier() -> None:
    manager = _manager()
    reservation = manager.reserve_offload(
        session_id=7,
        desired_frontier_blocks=3,
        generation=1,
        time=0.0,
    )

    assert reservation.block_indices == [0, 1, 2]
    assert manager.get_committed_frontier(7) == 0
    assert not manager.has_committed_block(7, 0)
    assert manager.reserved_blocks == 3

    manager.commit_offload(reservation, time=1.0)

    assert manager.get_committed_frontier(7) == 3
    assert [manager.has_committed_block(7, i) for i in range(4)] == [
        True,
        True,
        True,
        False,
    ]
    assert manager.resident_blocks == 3
    assert manager.reserved_blocks == 0


def test_incremental_offload_reserves_only_missing_suffix() -> None:
    manager = _manager()
    _commit(manager, session_id=7, frontier=2, generation=1, time=0.0)

    reservation = manager.reserve_offload(
        session_id=7,
        desired_frontier_blocks=5,
        generation=2,
        time=2.0,
    )

    assert reservation.block_indices == [2, 3, 4]
    assert reservation.num_blocks == 3
    manager.commit_offload(reservation, time=3.0)
    assert manager.get_committed_frontier(7) == 5


def test_aborted_reservation_is_invisible_and_returns_capacity() -> None:
    manager = _manager(num_blocks=3)
    reservation = manager.reserve_offload(
        session_id=7,
        desired_frontier_blocks=3,
        generation=1,
        time=0.0,
    )

    manager.abort_offload(reservation)

    assert manager.get_committed_frontier(7) == 0
    assert manager.free_blocks == 3
    assert manager.reserved_blocks == 0


def test_session_lru_evicts_suffix_first() -> None:
    manager = _manager(num_blocks=5)
    _commit(manager, session_id=7, frontier=3, generation=1, time=0.0)
    _commit(manager, session_id=8, frontier=2, generation=1, time=10.0)

    reservation = manager.reserve_offload(
        session_id=9,
        desired_frontier_blocks=2,
        generation=1,
        time=20.0,
    )

    assert reservation.num_blocks == 2
    assert manager.get_committed_frontier(7) == 1
    assert manager.has_committed_block(7, 0)
    assert not manager.has_committed_block(7, 1)
    assert manager.get_committed_frontier(8) == 2
    assert manager.evicted_blocks == 2


def test_restore_pin_prevents_session_eviction() -> None:
    manager = _manager(num_blocks=4)
    _commit(manager, session_id=7, frontier=2, generation=1, time=0.0)
    _commit(manager, session_id=8, frontier=2, generation=1, time=10.0)
    lease = manager.pin_restore_blocks(
        session_id=7,
        block_indices=[0, 1],
        time=11.0,
    )

    reservation = manager.reserve_offload(
        session_id=9,
        desired_frontier_blocks=2,
        generation=1,
        time=12.0,
    )

    assert reservation.num_blocks == 2
    assert manager.get_committed_frontier(7) == 2
    assert manager.get_committed_frontier(8) == 0
    assert manager._sessions[7].restore_pin_count == 2
    manager.release_restore_lease(lease, time=13.0)
    assert manager._sessions[7].restore_pin_count == 0


def test_prefix_fit_truncates_session_larger_than_capacity() -> None:
    manager = _manager(num_blocks=3)
    reservation = manager.reserve_offload(
        session_id=7,
        desired_frontier_blocks=5,
        generation=1,
        time=0.0,
    )

    assert reservation.block_indices == [0, 1, 2]
    assert reservation.truncated
    manager.commit_offload(reservation, time=1.0)
    assert manager.get_committed_frontier(7) == 3
    assert manager.truncated_offloads == 1


def test_skip_policy_does_not_admit_partial_delta() -> None:
    manager = _manager(num_blocks=3, policy="skip_offload")
    _commit(manager, session_id=7, frontier=2, generation=1, time=0.0)
    lease = manager.pin_restore_blocks(
        session_id=7,
        block_indices=[0],
        time=1.0,
    )

    reservation = manager.reserve_offload(
        session_id=8,
        desired_frontier_blocks=2,
        generation=1,
        time=2.0,
    )

    assert reservation.skipped
    assert reservation.terminal
    assert reservation.num_blocks == 0
    assert manager.get_committed_frontier(7) == 2
    manager.release_restore_lease(lease, time=3.0)


def test_out_of_order_commits_never_shrink_frontier() -> None:
    manager = _manager()
    older = manager.reserve_offload(
        session_id=7,
        desired_frontier_blocks=2,
        generation=1,
        time=0.0,
    )
    newer = manager.reserve_offload(
        session_id=7,
        desired_frontier_blocks=4,
        generation=2,
        time=0.1,
    )

    manager.commit_offload(newer, time=1.0)
    assert manager.get_committed_frontier(7) == 0
    following = manager.reserve_offload(
        session_id=7,
        desired_frontier_blocks=6,
        generation=3,
        time=1.1,
    )
    assert following.block_indices == [4, 5]

    manager.commit_offload(older, time=2.0)
    assert manager.get_committed_frontier(7) == 4
    manager.commit_offload(following, time=3.0)
    assert manager.get_committed_frontier(7) == 6


def test_cpu_transfer_engine_serializes_each_direction_but_is_full_duplex() -> None:
    config = CPUKVCacheConfig(
        enable=True,
        capacity_bytes=4096,
        write_bandwidth_gbps=8.0,
        write_latency_ms=1.0,
        read_bandwidth_gbps=8.0,
        read_latency_ms=2.0,
    )
    engine = AnalyticalCPUKVCacheTransferEngine(config)

    first_write = engine.schedule(
        direction="d2h",
        size_bytes=1_000_000,
        submitted_at=0.0,
    )
    second_write = engine.schedule(
        direction="d2h",
        size_bytes=1_000_000,
        submitted_at=0.0,
    )
    first_read = engine.schedule(
        direction="h2d",
        size_bytes=1_000_000,
        submitted_at=0.0,
    )

    assert first_write.service_time_ms == pytest.approx(2.0)
    assert second_write.start_time == pytest.approx(first_write.end_time)
    assert second_write.queue_time_ms == pytest.approx(2.0)
    assert first_read.start_time == 0.0
    assert first_read.service_time_ms == pytest.approx(3.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"enable": True, "capacity_bytes": 0}, "capacity_bytes"),
        ({"write_bandwidth_gbps": 0}, "write_bandwidth"),
        ({"read_bandwidth_gbps": 0}, "read_bandwidth"),
        ({"write_latency_ms": -1}, "write_latency"),
        ({"read_latency_ms": -1}, "read_latency"),
        ({"write_bandwidth_gbps": float("nan")}, "write_bandwidth"),
        ({"read_bandwidth_gbps": float("inf")}, "read_bandwidth"),
        ({"write_latency_ms": float("-inf")}, "write_latency"),
        ({"read_latency_ms": float("nan")}, "read_latency"),
        ({"eviction_policy": "fifo"}, "eviction_policy"),
        ({"capacity_pressure_policy": "invalid"}, "capacity_pressure_policy"),
        ({"transfer_concurrency": "independent"}, "transfer_concurrency"),
    ],
)
def test_cpu_config_rejects_invalid_values(kwargs, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        CPUKVCacheConfig(**kwargs)


def test_large_capacity_uses_lazy_block_materialization() -> None:
    manager = CPUKVCacheManager(
        capacity_bytes=128 * 1024**3,
        bytes_per_block=1024,
    )

    assert manager.num_blocks == 128 * 1024**2
    assert manager._blocks == {}

    reservation = manager.reserve_offload(
        session_id=7,
        desired_frontier_blocks=2,
        generation=1,
        time=0.0,
    )

    assert len(manager._blocks) == 2
    assert manager.reserved_blocks == 2


def test_abort_earlier_reservation_reclaims_dependent_suffix() -> None:
    manager = _manager(num_blocks=4)
    earlier = manager.reserve_offload(
        session_id=7,
        desired_frontier_blocks=2,
        generation=1,
        time=0.0,
    )
    later = manager.reserve_offload(
        session_id=7,
        desired_frontier_blocks=4,
        generation=2,
        time=0.1,
    )

    manager.commit_offload(later, time=1.0)
    manager.abort_offload(earlier)

    assert manager.get_committed_frontier(7) == 0
    assert manager.resident_blocks == 0
    assert manager.reserved_blocks == 0
    assert manager.free_blocks == 4
    replacement = manager.reserve_offload(
        session_id=8,
        desired_frontier_blocks=4,
        generation=1,
        time=2.0,
    )
    assert replacement.num_blocks == 4


def test_abort_earlier_reservation_cancels_active_dependent_suffix() -> None:
    manager = _manager(num_blocks=4)
    earlier = manager.reserve_offload(
        session_id=7,
        desired_frontier_blocks=2,
        generation=1,
        time=0.0,
    )
    later = manager.reserve_offload(
        session_id=7,
        desired_frontier_blocks=4,
        generation=2,
        time=0.1,
    )

    manager.abort_offload(earlier)

    assert later.terminal
    assert manager.resident_blocks == 0
    assert manager.reserved_blocks == 0
    manager.commit_offload(later, time=1.0)


def test_large_reservation_counters_match_materialized_state() -> None:
    manager = _manager(num_blocks=20_000, bytes_per_block=1)
    reservation = manager.reserve_offload(
        session_id=7,
        desired_frontier_blocks=20_000,
        generation=1,
        time=0.0,
    )
    assert manager.reserved_blocks == 20_000
    assert manager.resident_blocks == 0
    assert manager.free_blocks == 0

    manager.commit_offload(reservation, time=1.0)

    assert manager.reserved_blocks == 0
    assert manager.resident_blocks == 20_000
    assert sum(
        block.state.value == "committed"
        for block in manager._blocks.values()
    ) == manager.resident_blocks


def test_noop_offload_does_not_refresh_cpu_session_lru() -> None:
    manager = _manager(num_blocks=2)
    _commit(manager, session_id=1, frontier=1, generation=1, time=1.0)
    _commit(manager, session_id=2, frontier=1, generation=1, time=2.0)

    noop = manager.reserve_offload(
        session_id=1,
        desired_frontier_blocks=1,
        generation=2,
        time=100.0,
    )
    assert noop.terminal
    _commit(manager, session_id=3, frontier=1, generation=1, time=101.0)

    assert manager.get_committed_frontier(1) == 0
    assert manager.get_committed_frontier(2) == 1
    assert manager.get_committed_frontier(3) == 1


def test_cpu_lookup_metrics_deduplicate_logical_request_retries() -> None:
    manager = _manager()
    manager.record_lookup(
        session_id=7,
        query_blocks=3,
        hit_blocks=2,
        lookup_id=99,
    )
    manager.record_lookup(
        session_id=7,
        query_blocks=3,
        hit_blocks=2,
        lookup_id=99,
    )

    assert manager.cpu_query_blocks == 3
    assert manager.cpu_hit_blocks == 2


def test_empty_session_metadata_does_not_accumulate() -> None:
    manager = _manager(num_blocks=1)
    for session_id in range(10_000):
        reservation = manager.reserve_offload(
            session_id=session_id,
            desired_frontier_blocks=0,
            generation=1,
            time=float(session_id),
        )
        assert reservation.terminal

    assert manager._sessions == {}


def test_large_suffix_eviction_selects_lru_session_once() -> None:
    manager = _manager(num_blocks=10_000, bytes_per_block=1)
    _commit(
        manager,
        session_id=1,
        frontier=10_000,
        generation=1,
        time=1.0,
    )
    select_calls = 0
    original_select = manager._select_lru_evictable_session

    def counted_select(*, exclude_session_id: int):
        nonlocal select_calls
        select_calls += 1
        return original_select(exclude_session_id=exclude_session_id)

    manager._select_lru_evictable_session = counted_select
    reservation = manager.reserve_offload(
        session_id=2,
        desired_frontier_blocks=10_000,
        generation=1,
        time=2.0,
    )

    assert reservation.num_blocks == 10_000
    assert manager.get_committed_frontier(1) == 0
    assert manager.evicted_blocks == 10_000
    assert select_calls == 1


def test_zero_block_prefix_fit_is_terminal_and_does_not_pin_session() -> None:
    manager = _manager(num_blocks=1)
    _commit(manager, session_id=1, frontier=1, generation=1, time=1.0)
    lease = manager.pin_restore_blocks(
        session_id=1,
        block_indices=[0],
        time=2.0,
    )

    reservation = manager.reserve_offload(
        session_id=2,
        desired_frontier_blocks=1,
        generation=1,
        time=3.0,
    )

    assert reservation.terminal
    assert reservation.truncated
    assert reservation.num_blocks == 0
    assert manager._reservations == {}
    assert 2 not in manager._sessions
    manager.release_restore_lease(lease, time=4.0, used=False)
