from __future__ import annotations

from typing import TYPE_CHECKING, List

from frontier.events.base_event import BaseEvent
from frontier.types import ClusterType, EventType

if TYPE_CHECKING:
    from frontier.entities.cpu_kv_cache_transfer_info import CPUKVCacheOffloadInfo
    from frontier.metrics import MetricsStore
    from frontier.scheduler import BaseGlobalScheduler


class CPUKVCacheOffloadEndEvent(BaseEvent):
    def __init__(self, offload_info: "CPUKVCacheOffloadInfo") -> None:
        super().__init__(
            offload_info.timing.end_time,
            EventType.CPU_KV_CACHE_OFFLOAD_END,
        )
        self._offload_info = offload_info

    def handle_event(
        self,
        scheduler: "BaseGlobalScheduler",
        metrics_store: "MetricsStore",
    ) -> List[BaseEvent]:
        from frontier.events.replica_schedule_event import ReplicaScheduleEvent

        cluster_scheduler = scheduler.get_cluster_scheduler(ClusterType.PREFILL)
        replica_scheduler = cluster_scheduler.get_dp_replica_scheduler(
            self._offload_info.replica_id,
            self._offload_info.dp_id,
        )
        try:
            replica_scheduler.complete_cpu_kv_cache_offload(
                self.time, self._offload_info
            )
        except Exception:
            replica_scheduler.abort_cpu_kv_cache_offload(self._offload_info)
            raise
        metrics_store.on_cpu_kv_cache_offload_end(
            self.time, self._offload_info
        )
        return [
            ReplicaScheduleEvent(
                self.time,
                self._offload_info.replica_id,
                ClusterType.PREFILL,
                self._offload_info.dp_id,
            )
        ]

    def get_target_cluster(self) -> ClusterType:
        return ClusterType.PREFILL
