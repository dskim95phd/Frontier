from __future__ import annotations

import math
import random

import pytest

from frontier.config.cpu_kv_cache_config import CPUKVCacheConfig
from frontier.cpu_kv_cache_transfer import AnalyticalCPUKVCacheTransferEngine
from frontier.kv_cache import CPUKVCacheManager
from frontier.kv_cache.cpu_kv_cache_manager import CPUBlockState


def _assert_manager_invariants(manager: CPUKVCacheManager) -> None:
    blocks = list(manager._blocks.values())
    actual_reserved = sum(
        block.state == CPUBlockState.RESERVED for block in blocks
    )
    actual_resident = sum(
        block.state == CPUBlockState.COMMITTED for block in blocks
    )
    assert manager.reserved_blocks == actual_reserved
    assert manager.resident_blocks == actual_resident
    assert (
        manager.free_blocks + manager.reserved_blocks + manager.resident_blocks
        == manager.num_blocks
    )
    assert 0 <= manager.free_blocks <= manager.num_blocks

    owned_block_ids: set[int] = set()
    for entry in manager._sessions.values():
        assert 0 <= entry.committed_frontier_blocks <= entry.reserved_frontier_blocks
        for index in range(entry.committed_frontier_blocks):
            assert entry.blocks[index].state == CPUBlockState.COMMITTED
        for index, block in entry.blocks.items():
            assert index < entry.reserved_frontier_blocks
            assert block.session_id == entry.session_id
            assert block.block_index == index
            assert block.pin_count >= 0
            assert block.block_id not in owned_block_ids
            owned_block_ids.add(block.block_id)
        assert entry.active_reservation_ids <= manager._reservations.keys()
        assert entry.restore_pin_count == sum(
            block.pin_count for block in entry.blocks.values()
        )

    for reservation_id, reservation in manager._reservations.items():
        assert not reservation.terminal
        assert reservation_id in manager._sessions[
            reservation.session_id
        ].active_reservation_ids
        for block in reservation.blocks:
            assert block.state == CPUBlockState.RESERVED
            assert block.reservation_id == reservation_id
            assert block.block_id not in owned_block_ids
            owned_block_ids.add(block.block_id)

    recycled = list(manager._recycled_free_block_ids)
    assert len(recycled) == len(set(recycled))
    for block_id in recycled:
        assert manager._blocks[block_id].state == CPUBlockState.FREE
        assert block_id not in owned_block_ids
        owned_block_ids.add(block_id)
    assert owned_block_ids == set(manager._blocks)


@pytest.mark.parametrize("policy", ["prefix_fit", "skip_offload"])
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_randomized_cpu_store_state_machine_stress(
    policy: str,
    seed: int,
) -> None:
    rng = random.Random(seed)
    manager = CPUKVCacheManager(
        capacity_bytes=32 * 1024,
        bytes_per_block=1024,
        capacity_pressure_policy=policy,
    )
    reservations = []
    leases = []
    generations = {session_id: 0 for session_id in range(10)}

    for step in range(2_000):
        operation = rng.random()
        if operation < 0.42:
            session_id = rng.randrange(10)
            generations[session_id] += 1
            reservation = manager.reserve_offload(
                session_id=session_id,
                desired_frontier_blocks=rng.randrange(13),
                generation=generations[session_id],
                time=float(step),
            )
            if not reservation.terminal:
                reservations.append(reservation)
        elif operation < 0.60 and reservations:
            reservation = rng.choice(reservations)
            manager.commit_offload(reservation, time=float(step))
        elif operation < 0.72 and reservations:
            reservation = rng.choice(reservations)
            manager.abort_offload(reservation)
        elif operation < 0.86:
            candidates = [
                (session_id, entry.committed_frontier_blocks)
                for session_id, entry in manager._sessions.items()
                if entry.committed_frontier_blocks > 0
            ]
            if candidates:
                session_id, frontier = rng.choice(candidates)
                count = rng.randint(1, frontier)
                lease = manager.pin_restore_blocks(
                    session_id=session_id,
                    block_indices=range(count),
                    time=float(step),
                )
                leases.append(lease)
        elif leases:
            lease = rng.choice(leases)
            manager.release_restore_lease(
                lease,
                time=float(step),
                used=bool(rng.randrange(2)),
            )

        reservations = [
            reservation
            for reservation in reservations
            if not reservation.terminal
        ]
        leases = [lease for lease in leases if not lease.released]
        _assert_manager_invariants(manager)

    for lease in leases:
        manager.release_restore_lease(lease, time=2_001.0, used=False)
    for reservation in list(reservations):
        manager.abort_offload(reservation)
    _assert_manager_invariants(manager)


def test_randomized_full_duplex_transfer_queue_stress() -> None:
    rng = random.Random(20260727)
    engine = AnalyticalCPUKVCacheTransferEngine(
        CPUKVCacheConfig(
            enable=True,
            capacity_bytes=1024,
            write_bandwidth_gbps=31.5,
            write_latency_ms=0.017,
            read_bandwidth_gbps=47.25,
            read_latency_ms=0.023,
        )
    )
    previous_end = {"d2h": 0.0, "h2d": 0.0}

    for _ in range(5_000):
        direction = rng.choice(["d2h", "h2d"])
        submitted_at = rng.random() * 10.0
        timing = engine.schedule(
            direction=direction,
            size_bytes=rng.randrange(0, 64 * 1024**2),
            submitted_at=submitted_at,
        )
        assert math.isfinite(timing.end_time)
        assert timing.start_time >= submitted_at
        assert timing.start_time >= previous_end[direction]
        assert timing.end_time >= timing.start_time
        assert timing.queue_time_ms >= 0
        assert timing.service_time_ms >= 0
        previous_end[direction] = timing.end_time
