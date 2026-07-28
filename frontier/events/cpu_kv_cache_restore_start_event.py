from __future__ import annotations

from typing import TYPE_CHECKING, List

from frontier.events.base_event import BaseEvent
from frontier.types import ClusterType, EventType

if TYPE_CHECKING:
    from frontier.entities.cpu_kv_cache_transfer_info import CPUKVCacheRestoreInfo
    from frontier.metrics import MetricsStore
    from frontier.scheduler import BaseGlobalScheduler


class CPUKVCacheRestoreStartEvent(BaseEvent):
    def __init__(self, restore_info: "CPUKVCacheRestoreInfo") -> None:
        super().__init__(
            restore_info.timing.start_time,
            EventType.CPU_KV_CACHE_RESTORE_START,
        )
        self._restore_info = restore_info

    def handle_event(
        self,
        scheduler: "BaseGlobalScheduler",
        metrics_store: "MetricsStore",
    ) -> List[BaseEvent]:
        del scheduler
        from frontier.events.cpu_kv_cache_restore_end_event import (
            CPUKVCacheRestoreEndEvent,
        )

        metrics_store.on_cpu_kv_cache_restore_start(
            self.time, self._restore_info
        )
        return [CPUKVCacheRestoreEndEvent(self._restore_info)]

    def get_target_cluster(self) -> ClusterType:
        return ClusterType.PREFILL
