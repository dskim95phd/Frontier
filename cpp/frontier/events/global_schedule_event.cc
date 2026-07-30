#include "frontier/events/event_handlers.h"

#include <algorithm>
#include <vector>

#include "frontier/simulator/simulation_context.h"

namespace frontier::events {

void handle_event(
    const GlobalSchedulePayload&,
    SimTime time,
    simulator::SimulationContext& context) {
  const auto assignments =
      context.global_scheduler().schedule();
  if (assignments.empty()) {
    return;
  }
  std::vector<ClusterType> affected;
  for (const auto& assignment : assignments) {
    context.global_scheduler()
        .get_cluster_scheduler(assignment.cluster_type)
        .add_request(
            assignment.request_id,
            context.request(assignment.request_id).arrived_at());
    if (std::find(
            affected.begin(),
            affected.end(),
            assignment.cluster_type) == affected.end()) {
      affected.push_back(assignment.cluster_type);
    }
  }
  for (const ClusterType cluster_type : affected) {
    context.event_queue().push(
        time,
        ClusterSchedulePayload{
            .cluster_type = cluster_type,
        });
  }
}

}  // namespace frontier::events
