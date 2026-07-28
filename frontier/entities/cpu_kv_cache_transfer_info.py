from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Hashable

from frontier.cpu_kv_cache_transfer import CPUKVCacheTransferTiming

if TYPE_CHECKING:
    from frontier.entities.request import Request
    from frontier.kv_cache import (
        CPUOffloadReservation,
        CPURestoreLease,
        TieredPrefixPlan,
    )


@dataclass
class CPUKVCacheRestoreInfo:
    request: "Request"
    replica_id: int
    dp_id: int
    plan: "TieredPrefixPlan"
    cpu_lease: "CPURestoreLease"
    timing: CPUKVCacheTransferTiming

    @property
    def size_bytes(self) -> int:
        return self.timing.size_bytes


@dataclass(frozen=True)
class StagedCPUKVCacheRestore:
    """CPU restore payload that has arrived but does not own GPU KV pages."""

    request: "Request"
    replica_id: int
    dp_id: int
    session_id: int
    block_keys: tuple[Hashable, ...]
    transferred_cpu_block_indices: tuple[int, ...]
    query_blocks: int
    cpu_query_blocks: int
    lookup_gpu_hit_blocks: int
    lookup_hit_frontier_blocks: int
    block_size: int
    prompt_tokens: int
    timing: CPUKVCacheTransferTiming

    @classmethod
    def from_restore_info(
        cls, restore_info: CPUKVCacheRestoreInfo
    ) -> "StagedCPUKVCacheRestore":
        request = restore_info.request
        if request.session_id is None:
            raise ValueError("A staged CPU restore requires request.session_id")
        plan = restore_info.plan
        return cls(
            request=request,
            replica_id=int(restore_info.replica_id),
            dp_id=int(restore_info.dp_id),
            session_id=int(request.session_id),
            block_keys=tuple(plan.block_keys),
            transferred_cpu_block_indices=tuple(plan.cpu_block_indices),
            query_blocks=int(plan.query_blocks),
            cpu_query_blocks=int(plan.cpu_query_blocks),
            lookup_gpu_hit_blocks=int(plan.gpu_hit_blocks),
            lookup_hit_frontier_blocks=int(plan.hit_frontier_blocks),
            block_size=int(plan.block_size),
            prompt_tokens=int(plan.prompt_tokens),
            timing=restore_info.timing,
        )

    @property
    def transferred_cpu_blocks(self) -> int:
        return len(self.transferred_cpu_block_indices)


@dataclass
class CPUKVCacheOffloadInfo:
    request: "Request"
    replica_id: int
    dp_id: int
    reservation: "CPUOffloadReservation"
    timing: CPUKVCacheTransferTiming
    desired_frontier_blocks: int
    decode_transfer_completed_at: float | None = None

    @property
    def size_bytes(self) -> int:
        return self.timing.size_bytes

    @property
    def source_gpu_hold_time_ms(self) -> float:
        """Extra source-GPU hold caused specifically by the CPU copy."""
        if self.decode_transfer_completed_at is None:
            return 0.0
        return max(
            0.0,
            (self.timing.end_time - self.decode_transfer_completed_at) * 1e3,
        )
