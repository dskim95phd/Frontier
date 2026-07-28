from __future__ import annotations

from typing import TYPE_CHECKING, List

from frontier.events.base_event import BaseEvent
from frontier.types import ClusterType, EventType

if TYPE_CHECKING:
    from frontier.entities.cpu_kv_cache_transfer_info import CPUKVCacheOffloadInfo
    from frontier.metrics import MetricsStore
    from frontier.scheduler import BaseGlobalScheduler


class CPUKVCacheOffloadStartEvent(BaseEvent):
    def __init__(self, offload_info: "CPUKVCacheOffloadInfo") -> None:
        super().__init__(
            offload_info.timing.start_time,
            EventType.CPU_KV_CACHE_OFFLOAD_START,
        )
        self._offload_info = offload_info

    def handle_event(
        self,
        scheduler: "BaseGlobalScheduler",
        metrics_store: "MetricsStore",
    ) -> List[BaseEvent]:
        del scheduler
        from frontier.events.cpu_kv_cache_offload_end_event import (
            CPUKVCacheOffloadEndEvent,
        )

        metrics_store.on_cpu_kv_cache_offload_start(
            self.time, self._offload_info
        )
        return [CPUKVCacheOffloadEndEvent(self._offload_info)]

    def get_target_cluster(self) -> ClusterType:
        return ClusterType.PREFILL
