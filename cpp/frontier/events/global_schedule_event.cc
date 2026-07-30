#include "frontier/events/event_handlers.h"

#include <algorithm>
#include <vector>

#include "frontier/simulator/simulator.h"

namespace frontier::events {

void handle_event(const GlobalSchedulePayload &, SimTime time,
                  simulator::Simulator &simulator) {
    const auto assignments = simulator.global_scheduler().schedule();
    if (assignments.empty()) {
        return;
    }
    std::vector<ClusterType> affected;
    for (const auto &assignment : assignments) {
        simulator.global_scheduler()
            .get_cluster_scheduler(assignment.cluster_type)
            .add_request(assignment.request_id,
                         simulator.request(assignment.request_id).arrived_at());
        if (std::find(affected.begin(), affected.end(),
                      assignment.cluster_type) == affected.end()) {
            affected.push_back(assignment.cluster_type);
        }
    }
    for (const ClusterType cluster_type : affected) {
        simulator.event_queue().push(time, [&]() {
            ClusterSchedulePayload value{};
            value.cluster_type = cluster_type;
            return value;
        }());
    }
}

} // namespace frontier::events
