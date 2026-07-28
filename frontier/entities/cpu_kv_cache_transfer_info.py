from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from frontier.cpu_kv_cache_transfer import CPUKVCacheTransferTiming

if TYPE_CHECKING:
    from frontier.entities.request import Request
    from frontier.kv_cache import (
        CPUOffloadReservation,
        CPURestoreLease,
        TieredPrefixPlan,
    )
    from frontier.kv_cache.kv_cache_block import KVCacheBlock


@dataclass
class CPUKVCacheRestoreInfo:
    request: "Request"
    replica_id: int
    dp_id: int
    plan: "TieredPrefixPlan"
    restore_blocks_by_index: dict[int, "KVCacheBlock"]
    cpu_lease: "CPURestoreLease"
    timing: CPUKVCacheTransferTiming

    @property
    def size_bytes(self) -> int:
        return self.timing.size_bytes


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
