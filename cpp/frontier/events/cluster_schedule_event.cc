#include "frontier/events/event_handlers.h"

#include <algorithm>
#include <vector>

#include "frontier/simulator/simulator.h"

namespace frontier::events {

void handle_event(const ClusterSchedulePayload &payload, SimTime time,
                  simulator::Simulator &simulator) {
    const auto assignments = simulator.cluster(payload.cluster_type).schedule();
    std::vector<scheduler::ReplicaTarget> affected;
    for (const auto &assignment : assignments) {
        const scheduler::ReplicaTarget target = [&]() {
            scheduler::ReplicaTarget value{};
            value.replica_id = assignment.replica_id;
            value.dp_id = assignment.dp_id;
            return value;
        }();
        simulator.assign_request_target(assignment.request_id, target,
                                        payload.cluster_type);
        if (std::find(affected.begin(), affected.end(), target) ==
            affected.end()) {
            affected.push_back(target);
        }
    }
    for (const scheduler::ReplicaTarget target : affected) {
        simulator.event_queue().push(time, [&]() {
            ReplicaSchedulePayload value{};
            value.replica_id = target.replica_id;
            value.dp_id = target.dp_id;
            value.cluster_type = payload.cluster_type;
            return value;
        }());
    }
}

} // namespace frontier::events
