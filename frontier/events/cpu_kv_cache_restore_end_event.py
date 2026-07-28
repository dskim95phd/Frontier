from __future__ import annotations

from typing import TYPE_CHECKING, List

from frontier.events.base_event import BaseEvent
from frontier.types import ClusterType, EventType

if TYPE_CHECKING:
    from frontier.entities.cpu_kv_cache_transfer_info import CPUKVCacheRestoreInfo
    from frontier.metrics import MetricsStore
    from frontier.scheduler import BaseGlobalScheduler


class CPUKVCacheRestoreEndEvent(BaseEvent):
    def __init__(self, restore_info: "CPUKVCacheRestoreInfo") -> None:
        super().__init__(
            restore_info.timing.end_time,
            EventType.CPU_KV_CACHE_RESTORE_END,
        )
        self._restore_info = restore_info

    def handle_event(
        self,
        scheduler: "BaseGlobalScheduler",
        metrics_store: "MetricsStore",
    ) -> List[BaseEvent]:
        from frontier.events.replica_schedule_event import ReplicaScheduleEvent

        cluster_scheduler = scheduler.get_cluster_scheduler(ClusterType.PREFILL)
        replica_scheduler = cluster_scheduler.get_dp_replica_scheduler(
            self._restore_info.replica_id,
            self._restore_info.dp_id,
        )
        try:
            replica_scheduler.complete_cpu_kv_cache_restore(
                self.time, self._restore_info
            )
        except Exception:
            replica_scheduler.cancel_cpu_kv_cache_restore(
                self._restore_info.request.id,
                time=self.time,
            )
            raise
        metrics_store.on_cpu_kv_cache_restore_end(
            self.time, self._restore_info
        )
        return [
            ReplicaScheduleEvent(
                self.time,
                self._restore_info.replica_id,
                ClusterType.PREFILL,
                self._restore_info.dp_id,
            )
        ]

    def get_target_cluster(self) -> ClusterType:
        return ClusterType.PREFILL
