from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Hashable, Iterable


class CPUBlockState(str, Enum):
    FREE = "free"
    RESERVED = "reserved"
    COMMITTED = "committed"


@dataclass
class CPUKVCacheBlock:
    block_id: int
    state: CPUBlockState = CPUBlockState.FREE
    key: Hashable | None = None
    session_id: int | None = None
    block_index: int | None = None
    pin_count: int = 0
    reservation_id: int | None = None

    def reset(self) -> None:
        if self.pin_count:
            raise ValueError(
                f"Cannot reset pinned CPU KV block {self.block_id}: pins={self.pin_count}"
            )
        self.state = CPUBlockState.FREE
        self.key = None
        self.session_id = None
        self.block_index = None
        self.reservation_id = None


@dataclass
class CPUSessionCacheEntry:
    session_id: int
    committed_frontier_blocks: int = 0
    reserved_frontier_blocks: int = 0
    last_access_time: float = 0.0
    last_commit_time: float = 0.0
    latest_generation: int = -1
    blocks: dict[int, CPUKVCacheBlock] = field(default_factory=dict)
    active_reservation_ids: set[int] = field(default_factory=set)
    restore_pin_count: int = 0

    @property
    def is_pinned(self) -> bool:
        return bool(self.active_reservation_ids) or self.restore_pin_count > 0


@dataclass
class CPUOffloadReservation:
    reservation_id: int
    session_id: int
    generation: int
    desired_frontier_blocks: int
    admitted_frontier_blocks: int
    block_indices: list[int]
    blocks: list[CPUKVCacheBlock]
    submitted_at: float
    skipped: bool = False
    truncated: bool = False
    terminal: bool = False

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)


@dataclass
class CPURestoreLease:
    lease_id: int
    session_id: int
    block_indices: list[int]
    blocks: list[CPUKVCacheBlock]
    started_at: float
    released: bool = False

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)


class CPUKVCacheManager:
    """Finite, session-scoped CPU KV store for one prefill cache target.

    The manager stores only full blocks and maintains a committed contiguous
    prefix for every session. Reservations account against capacity but are
    invisible to lookups until committed.
    """

    def __init__(
        self,
        *,
        capacity_bytes: int,
        bytes_per_block: int,
        capacity_pressure_policy: str = "prefix_fit",
    ) -> None:
        self.capacity_bytes = int(capacity_bytes)
        self.bytes_per_block = int(bytes_per_block)
        if self.capacity_bytes <= 0:
            raise ValueError("capacity_bytes must be > 0")
        if self.bytes_per_block <= 0:
            raise ValueError("bytes_per_block must be > 0")
        if capacity_pressure_policy not in {"prefix_fit", "skip_offload"}:
            raise ValueError(
                "capacity_pressure_policy must be 'prefix_fit' or 'skip_offload'"
            )
        self.capacity_pressure_policy = capacity_pressure_policy
        self.num_blocks = self.capacity_bytes // self.bytes_per_block
        if self.num_blocks <= 0:
            raise ValueError(
                "CPU KV-cache capacity is smaller than one KV block: "
                f"capacity_bytes={self.capacity_bytes}, "
                f"bytes_per_block={self.bytes_per_block}"
            )

        # Materialize block objects lazily. A realistic DRAM capacity can map
        # to hundreds of thousands of KV blocks; constructing one Python
        # object and one free-list entry per block would make simulator startup
        # proportional to configured capacity even when the cache is unused.
        self._blocks: dict[int, CPUKVCacheBlock] = {}
        self._recycled_free_block_ids: deque[int] = deque()
        self._next_unused_block_id = 0
        self._resident_block_count = 0
        self._reserved_block_count = 0
        self._sessions: dict[int, CPUSessionCacheEntry] = {}
        self._reservations: dict[int, CPUOffloadReservation] = {}
        self._restore_leases: dict[int, CPURestoreLease] = {}
        self._next_reservation_id = 0
        self._next_restore_lease_id = 0

        self.evicted_sessions = 0
        self.evicted_blocks = 0
        self.evicted_bytes = 0
        self.skipped_offloads = 0
        self.truncated_offloads = 0
        self.stale_generation_completions = 0
        self.cpu_query_blocks = 0
        self.cpu_hit_blocks = 0
        self._recorded_lookup_ids: set[Hashable] = set()
        self._sessions_with_cpu_hits: set[int] = set()
        self.peak_resident_blocks = 0
        self.peak_reserved_blocks = 0

    @staticmethod
    def make_session_key(session_id: int, block_index: int) -> tuple[str, int, int]:
        return ("session", int(session_id), int(block_index))

    @property
    def resident_blocks(self) -> int:
        return self._resident_block_count

    @property
    def reserved_blocks(self) -> int:
        return self._reserved_block_count

    @property
    def free_blocks(self) -> int:
        return self.num_blocks - self.resident_blocks - self.reserved_blocks

    @property
    def resident_bytes(self) -> int:
        return self.resident_blocks * self.bytes_per_block

    @property
    def reserved_bytes(self) -> int:
        return self.reserved_blocks * self.bytes_per_block

    @property
    def resident_sessions(self) -> int:
        return sum(
            1
            for entry in self._sessions.values()
            if entry.committed_frontier_blocks > 0
        )

    def get_committed_frontier(self, session_id: int) -> int:
        entry = self._sessions.get(int(session_id))
        return int(entry.committed_frontier_blocks) if entry is not None else 0

    def has_committed_block(self, session_id: int, block_index: int) -> bool:
        entry = self._sessions.get(int(session_id))
        if entry is None or int(block_index) >= entry.committed_frontier_blocks:
            return False
        block = entry.blocks.get(int(block_index))
        return block is not None and block.state == CPUBlockState.COMMITTED

    def probe_committed_block(self, session_id: int, block_index: int) -> bool:
        """Probe one CPU block without counting a failed admission retry."""
        return self.has_committed_block(session_id, block_index)

    def record_lookup(
        self,
        *,
        session_id: int,
        query_blocks: int,
        hit_blocks: int,
        lookup_id: Hashable | None = None,
    ) -> None:
        query_blocks = int(query_blocks)
        hit_blocks = int(hit_blocks)
        if query_blocks < 0 or hit_blocks < 0 or hit_blocks > query_blocks:
            raise ValueError(
                "CPU lookup metrics require 0 <= hits <= queries, "
                f"got hits={hit_blocks}, queries={query_blocks}"
            )
        if lookup_id is not None:
            if lookup_id in self._recorded_lookup_ids:
                return
            self._recorded_lookup_ids.add(lookup_id)
        self.cpu_query_blocks += query_blocks
        self.cpu_hit_blocks += hit_blocks
        if hit_blocks:
            self._sessions_with_cpu_hits.add(int(session_id))

    def _get_or_create_entry(self, session_id: int) -> CPUSessionCacheEntry:
        session_id = int(session_id)
        entry = self._sessions.get(session_id)
        if entry is None:
            entry = CPUSessionCacheEntry(session_id=session_id)
            self._sessions[session_id] = entry
        return entry

    def _drop_empty_session_entry(self, entry: CPUSessionCacheEntry) -> None:
        if (
            entry.committed_frontier_blocks == 0
            and entry.reserved_frontier_blocks == 0
            and not entry.blocks
            and not entry.active_reservation_ids
        ):
            self._sessions.pop(entry.session_id, None)

    def _alloc_reserved_block(
        self,
        *,
        session_id: int,
        block_index: int,
        reservation_id: int,
    ) -> CPUKVCacheBlock:
        if self._recycled_free_block_ids:
            block_id = self._recycled_free_block_ids.popleft()
            block = self._blocks[block_id]
        else:
            block_id = self._next_unused_block_id
            if block_id >= self.num_blocks:
                raise AssertionError(
                    "CPU KV-cache lazy allocator exceeded configured capacity"
                )
            self._next_unused_block_id += 1
            block = CPUKVCacheBlock(block_id)
            self._blocks[block_id] = block
        if block.state != CPUBlockState.FREE:
            raise AssertionError("CPU KV free list contains a non-free block")
        block.state = CPUBlockState.RESERVED
        block.key = self.make_session_key(session_id, block_index)
        block.session_id = int(session_id)
        block.block_index = int(block_index)
        block.reservation_id = int(reservation_id)
        self._reserved_block_count += 1
        return block

    def _free_block(self, block: CPUKVCacheBlock) -> None:
        block_id = block.block_id
        if block.state == CPUBlockState.RESERVED:
            self._reserved_block_count -= 1
        elif block.state == CPUBlockState.COMMITTED:
            self._resident_block_count -= 1
        else:
            raise AssertionError("CPU KV block is already free")
        block.reset()
        self._recycled_free_block_ids.append(block_id)

    def _select_lru_evictable_session(
        self, *, exclude_session_id: int
    ) -> CPUSessionCacheEntry | None:
        candidates = [
            entry
            for session_id, entry in self._sessions.items()
            if session_id != int(exclude_session_id)
            and entry.committed_frontier_blocks > 0
            and not entry.is_pinned
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda entry: (
                entry.last_access_time,
                entry.last_commit_time,
                entry.session_id,
            ),
        )

    def _evict_suffix_blocks(
        self,
        entry: CPUSessionCacheEntry,
        num_blocks: int,
    ) -> int:
        if entry.is_pinned or entry.committed_frontier_blocks <= 0:
            raise ValueError("Session is not evictable")
        num_blocks = min(
            max(int(num_blocks), 0),
            entry.committed_frontier_blocks,
        )
        if num_blocks == 0:
            return 0
        new_frontier = entry.committed_frontier_blocks - num_blocks
        for block_index in range(
            entry.committed_frontier_blocks - 1,
            new_frontier - 1,
            -1,
        ):
            block = entry.blocks.pop(block_index)
            if block.pin_count:
                raise AssertionError("Pinned CPU KV block selected for eviction")
            self._free_block(block)
        entry.committed_frontier_blocks = new_frontier
        entry.reserved_frontier_blocks = entry.committed_frontier_blocks
        self.evicted_blocks += num_blocks
        self.evicted_bytes += num_blocks * self.bytes_per_block
        if entry.committed_frontier_blocks == 0:
            self.evicted_sessions += 1
            self._drop_empty_session_entry(entry)
        return num_blocks

    def _evict_until_free(
        self, required_blocks: int, *, exclude_session_id: int
    ) -> None:
        while self.free_blocks < required_blocks:
            entry = self._select_lru_evictable_session(
                exclude_session_id=exclude_session_id
            )
            if entry is None:
                return
            deficit = required_blocks - self.free_blocks
            self._evict_suffix_blocks(entry, deficit)

    def _evictable_blocks(self, *, exclude_session_id: int) -> int:
        return sum(
            entry.committed_frontier_blocks
            for session_id, entry in self._sessions.items()
            if session_id != int(exclude_session_id) and not entry.is_pinned
        )

    def reserve_offload(
        self,
        *,
        session_id: int,
        desired_frontier_blocks: int,
        generation: int,
        time: float,
    ) -> CPUOffloadReservation:
        session_id = int(session_id)
        desired_frontier_blocks = max(int(desired_frontier_blocks), 0)
        entry = self._get_or_create_entry(session_id)
        base_frontier = max(
            entry.committed_frontier_blocks,
            entry.reserved_frontier_blocks,
        )
        missing_blocks = max(desired_frontier_blocks - base_frontier, 0)

        reservation_id = self._next_reservation_id
        self._next_reservation_id += 1

        if missing_blocks == 0:
            entry.latest_generation = max(
                entry.latest_generation, int(generation)
            )
            reservation = CPUOffloadReservation(
                reservation_id=reservation_id,
                session_id=session_id,
                generation=int(generation),
                desired_frontier_blocks=desired_frontier_blocks,
                admitted_frontier_blocks=min(
                    desired_frontier_blocks, base_frontier
                ),
                block_indices=[],
                blocks=[],
                submitted_at=float(time),
                terminal=True,
            )
            self._drop_empty_session_entry(entry)
            return reservation

        if self.capacity_pressure_policy == "skip_offload":
            potential = self.free_blocks + self._evictable_blocks(
                exclude_session_id=session_id
            )
            if potential < missing_blocks:
                self.skipped_offloads += 1
                reservation = CPUOffloadReservation(
                    reservation_id=reservation_id,
                    session_id=session_id,
                    generation=int(generation),
                    desired_frontier_blocks=desired_frontier_blocks,
                    admitted_frontier_blocks=base_frontier,
                    block_indices=[],
                    blocks=[],
                    submitted_at=float(time),
                    skipped=True,
                    terminal=True,
                )
                self._drop_empty_session_entry(entry)
                return reservation

        self._evict_until_free(
            missing_blocks,
            exclude_session_id=session_id,
        )
        admitted_new_blocks = min(missing_blocks, self.free_blocks)
        if (
            self.capacity_pressure_policy == "skip_offload"
            and admitted_new_blocks < missing_blocks
        ):
            self.skipped_offloads += 1
            reservation = CPUOffloadReservation(
                reservation_id=reservation_id,
                session_id=session_id,
                generation=int(generation),
                desired_frontier_blocks=desired_frontier_blocks,
                admitted_frontier_blocks=base_frontier,
                block_indices=[],
                blocks=[],
                submitted_at=float(time),
                skipped=True,
                terminal=True,
            )
            self._drop_empty_session_entry(entry)
            return reservation

        admitted_frontier = base_frontier + admitted_new_blocks
        if admitted_new_blocks == 0:
            self.truncated_offloads += 1
            reservation = CPUOffloadReservation(
                reservation_id=reservation_id,
                session_id=session_id,
                generation=int(generation),
                desired_frontier_blocks=desired_frontier_blocks,
                admitted_frontier_blocks=base_frontier,
                block_indices=[],
                blocks=[],
                submitted_at=float(time),
                truncated=True,
                terminal=True,
            )
            self._drop_empty_session_entry(entry)
            return reservation
        block_indices = list(range(base_frontier, admitted_frontier))
        blocks = [
            self._alloc_reserved_block(
                session_id=session_id,
                block_index=block_index,
                reservation_id=reservation_id,
            )
            for block_index in block_indices
        ]
        truncated = admitted_frontier < desired_frontier_blocks
        if truncated:
            self.truncated_offloads += 1

        reservation = CPUOffloadReservation(
            reservation_id=reservation_id,
            session_id=session_id,
            generation=int(generation),
            desired_frontier_blocks=desired_frontier_blocks,
            admitted_frontier_blocks=admitted_frontier,
            block_indices=block_indices,
            blocks=blocks,
            submitted_at=float(time),
            truncated=truncated,
        )
        self._reservations[reservation_id] = reservation
        entry.active_reservation_ids.add(reservation_id)
        entry.reserved_frontier_blocks = max(
            entry.reserved_frontier_blocks, admitted_frontier
        )
        self.peak_reserved_blocks = max(
            self.peak_reserved_blocks, self.reserved_blocks
        )
        return reservation

    def _recompute_reserved_frontier(self, entry: CPUSessionCacheEntry) -> None:
        frontier = entry.committed_frontier_blocks
        active = sorted(
            (
                self._reservations[reservation_id]
                for reservation_id in entry.active_reservation_ids
                if reservation_id in self._reservations
            ),
            key=lambda reservation: reservation.block_indices[0]
            if reservation.block_indices
            else reservation.admitted_frontier_blocks,
        )
        # A later reservation can complete before an earlier one. Its blocks
        # are then committed but cannot yet advance committed_frontier_blocks
        # across the earlier gap. Treat both those pending committed blocks
        # and active reservations as accounted contiguous ranges; otherwise a
        # third reservation can allocate duplicate logical block indices.
        while True:
            previous_frontier = frontier
            for reservation in active:
                if not reservation.block_indices:
                    continue
                if reservation.block_indices[0] > frontier:
                    break
                frontier = max(frontier, reservation.admitted_frontier_blocks)
            while True:
                block = entry.blocks.get(frontier)
                if block is None or block.state != CPUBlockState.COMMITTED:
                    break
                frontier += 1
            if frontier == previous_frontier:
                break
        entry.reserved_frontier_blocks = frontier

    def commit_offload(
        self, reservation: CPUOffloadReservation, *, time: float
    ) -> None:
        if reservation.terminal:
            return
        current = self._reservations.pop(reservation.reservation_id, None)
        if current is None:
            raise ValueError(
                f"Unknown CPU offload reservation {reservation.reservation_id}"
            )
        entry = self._sessions[reservation.session_id]
        entry.active_reservation_ids.discard(reservation.reservation_id)

        if reservation.generation < entry.latest_generation:
            self.stale_generation_completions += 1

        for block_index, block in zip(
            reservation.block_indices, reservation.blocks
        ):
            existing = entry.blocks.get(block_index)
            if existing is not None and existing.state == CPUBlockState.COMMITTED:
                self._free_block(block)
                continue
            block.state = CPUBlockState.COMMITTED
            block.reservation_id = None
            self._reserved_block_count -= 1
            self._resident_block_count += 1
            entry.blocks[block_index] = block

        while True:
            block = entry.blocks.get(entry.committed_frontier_blocks)
            if block is None or block.state != CPUBlockState.COMMITTED:
                break
            entry.committed_frontier_blocks += 1

        entry.latest_generation = max(
            entry.latest_generation, reservation.generation
        )
        entry.last_access_time = float(time)
        entry.last_commit_time = float(time)
        self._recompute_reserved_frontier(entry)
        reservation.terminal = True
        self._drop_empty_session_entry(entry)
        self.peak_resident_blocks = max(
            self.peak_resident_blocks, self.resident_blocks
        )

    def abort_offload(self, reservation: CPUOffloadReservation) -> None:
        if reservation.terminal:
            return
        current = self._reservations.pop(reservation.reservation_id, None)
        if current is None:
            raise ValueError(
                f"Unknown CPU offload reservation {reservation.reservation_id}"
            )
        entry = self._sessions[reservation.session_id]
        entry.active_reservation_ids.discard(reservation.reservation_id)
        for block in reservation.blocks:
            self._free_block(block)
        # Later reservations are suffixes of this one and therefore depend on
        # its logical range. Abort any still-active dependents. A dependent
        # that completed earlier is committed beyond the new gap but was never
        # lookup-visible; reclaim that orphaned suffix as well.
        aborted_start = (
            reservation.block_indices[0]
            if reservation.block_indices
            else entry.committed_frontier_blocks
        )
        for dependent_id in list(entry.active_reservation_ids):
            dependent = self._reservations.get(dependent_id)
            if (
                dependent is None
                or not dependent.block_indices
                or dependent.block_indices[0] < aborted_start
            ):
                continue
            self._reservations.pop(dependent_id, None)
            entry.active_reservation_ids.discard(dependent_id)
            for block in dependent.blocks:
                self._free_block(block)
            dependent.terminal = True
        for block_index in sorted(list(entry.blocks)):
            if block_index < entry.committed_frontier_blocks:
                continue
            block = entry.blocks.pop(block_index)
            self._free_block(block)
        self._recompute_reserved_frontier(entry)
        reservation.terminal = True

    def pin_restore_blocks(
        self,
        *,
        session_id: int,
        block_indices: Iterable[int],
        time: float,
    ) -> CPURestoreLease:
        session_id = int(session_id)
        indices = [int(index) for index in block_indices]
        entry = self._sessions.get(session_id)
        if entry is None:
            raise ValueError(f"No CPU KV-cache entry for session {session_id}")
        blocks: list[CPUKVCacheBlock] = []
        for block_index in indices:
            block = entry.blocks.get(block_index)
            if (
                block is None
                or block.state != CPUBlockState.COMMITTED
                or block_index >= entry.committed_frontier_blocks
            ):
                for pinned in blocks:
                    pinned.pin_count -= 1
                raise ValueError(
                    "Cannot pin non-committed CPU KV block: "
                    f"session_id={session_id}, block_index={block_index}"
                )
            block.pin_count += 1
            blocks.append(block)
        entry.restore_pin_count += len(blocks)

        lease_id = self._next_restore_lease_id
        self._next_restore_lease_id += 1
        lease = CPURestoreLease(
            lease_id=lease_id,
            session_id=session_id,
            block_indices=indices,
            blocks=blocks,
            started_at=float(time),
        )
        self._restore_leases[lease_id] = lease
        return lease

    def release_restore_lease(
        self,
        lease: CPURestoreLease,
        *,
        time: float,
        used: bool = True,
    ) -> None:
        if lease.released:
            return
        current = self._restore_leases.pop(lease.lease_id, None)
        if current is None:
            raise ValueError(f"Unknown CPU restore lease {lease.lease_id}")
        for block in lease.blocks:
            if block.pin_count <= 0:
                raise AssertionError("CPU KV restore pin underflow")
            block.pin_count -= 1
        entry = self._sessions[lease.session_id]
        if entry.restore_pin_count < len(lease.blocks):
            raise AssertionError("CPU KV session restore pin underflow")
        entry.restore_pin_count -= len(lease.blocks)
        if used:
            entry.last_access_time = float(time)
        lease.released = True

    def get_statistics(self) -> dict[str, int | float]:
        return {
            "capacity_bytes": self.capacity_bytes,
            "capacity_blocks": self.num_blocks,
            "bytes_per_block": self.bytes_per_block,
            "resident_bytes": self.resident_bytes,
            "resident_blocks": self.resident_blocks,
            "reserved_bytes": self.reserved_bytes,
            "reserved_blocks": self.reserved_blocks,
            "free_blocks": self.free_blocks,
            "peak_resident_bytes": self.peak_resident_blocks
            * self.bytes_per_block,
            "peak_resident_blocks": self.peak_resident_blocks,
            "peak_reserved_bytes": self.peak_reserved_blocks
            * self.bytes_per_block,
            "peak_reserved_blocks": self.peak_reserved_blocks,
            "resident_sessions": self.resident_sessions,
            "evicted_sessions": self.evicted_sessions,
            "evicted_blocks": self.evicted_blocks,
            "evicted_bytes": self.evicted_bytes,
            "skipped_offloads": self.skipped_offloads,
            "truncated_offloads": self.truncated_offloads,
            "stale_generation_completions": self.stale_generation_completions,
            "cpu_query_blocks": self.cpu_query_blocks,
            "cpu_hit_blocks": self.cpu_hit_blocks,
            "sessions_with_cpu_hits": len(self._sessions_with_cpu_hits),
        }
