#include "frontier/events/event_handlers.h"

#include "frontier/simulator/simulator.h"

namespace frontier::events {

void handle_event(const CpuKVCacheOffloadEndPayload &payload, SimTime time,
                  simulator::Simulator &simulator) {
    scheduler::BaseReplicaScheduler &scheduler =
        simulator.cluster(payload.cluster_type)
            .get_replica_scheduler(payload.replica_id, payload.dp_id);
    const bool completed = scheduler.on_cpu_kv_cache_offload_end(
        payload.transfer_id, payload.cpu_generation, time);
    if (!completed) {
        return;
    }

    // Same-session requests can be held above the replica while this D2H
    // export owns the session.  Re-run cluster routing as soon as the export
    // closes so those requests do not wait for an unrelated future arrival.
    if (!simulator.cluster(payload.cluster_type).empty()) {
        simulator.event_queue().push(time, [&]() {
            ClusterSchedulePayload value{};
            value.cluster_type = payload.cluster_type;
            return value;
        }());
    }

    if (scheduler.waiting_count() > 0 &&
        scheduler.in_flight_batch_count() == 0) {
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
