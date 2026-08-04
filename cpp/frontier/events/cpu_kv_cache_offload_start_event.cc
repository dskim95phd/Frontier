#include "frontier/events/event_handlers.h"

#include <utility>

#include "frontier/simulator/simulator.h"

namespace frontier::events {

void handle_event(const CpuKVCacheOffloadStartPayload &payload, SimTime time,
                  simulator::Simulator &simulator) {
    scheduler::BaseReplicaScheduler &scheduler =
        simulator.cluster(payload.cluster_type)
            .get_replica_scheduler(payload.replica_id, payload.dp_id);
    scheduler.on_cpu_kv_cache_offload_start(
        payload.transfer_id, payload.cpu_generation, time);
    for (scheduler::ScheduledAuxiliaryEvent &event :
         scheduler.drain_auxiliary_events()) {
        simulator.event_queue().push(event.time, std::move(event.payload));
    }
}

} // namespace frontier::events
