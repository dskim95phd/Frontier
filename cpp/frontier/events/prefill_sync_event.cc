#include "frontier/events/event_handlers.h"

#include "frontier/scheduler/cluster_scheduler/base_cluster_scheduler.h"
#include "frontier/simulator/simulator.h"

namespace frontier::events {

void handle_event(const PrefillSyncPayload &payload, SimTime time,
                  simulator::Simulator &simulator) {
    simulator.cluster(payload.cluster_type)
        .on_prefill_sync(payload, time, simulator);
}

} // namespace frontier::events
