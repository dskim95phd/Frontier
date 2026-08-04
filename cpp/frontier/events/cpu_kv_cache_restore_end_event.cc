#include "frontier/events/event_handlers.h"

#include "frontier/simulator/simulator.h"

namespace frontier::events {

void handle_event(const CpuKVCacheRestoreEndPayload &payload, SimTime time,
                  simulator::Simulator &simulator) {
    scheduler::BaseReplicaScheduler &scheduler =
        simulator.cluster(payload.cluster_type)
            .get_replica_scheduler(payload.replica_id, payload.dp_id);
    if (scheduler.on_cpu_kv_cache_restore_end(
            payload.transfer_id, payload.generation, time)) {
        simulator.event_queue().push(time, [&]() {
            ReplicaSchedulePayload value{};
            value.replica_id = payload.replica_id;
            value.dp_id = payload.dp_id;
            value.cluster_type = payload.cluster_type;
            return value;
        }());
    }
}

} // namespace frontier::events
