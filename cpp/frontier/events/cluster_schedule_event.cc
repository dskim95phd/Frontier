#include "frontier/events/event_handlers.h"

#include <algorithm>
#include <vector>

#include "frontier/simulator/simulation_context.h"

namespace frontier::events {

void handle_event(
    const ClusterSchedulePayload& payload,
    SimTime time,
    simulator::SimulationContext& context) {
  const auto assignments =
      context.cluster(payload.cluster_type).schedule();
  std::vector<scheduler::ReplicaTarget> affected;
  for (const auto& assignment : assignments) {
    const scheduler::ReplicaTarget target{
        .replica_id = assignment.replica_id,
        .dp_id = assignment.dp_id,
    };
    context.assign_request_target(
        assignment.request_id, target, payload.cluster_type);
    if (std::find(affected.begin(), affected.end(), target) ==
        affected.end()) {
      affected.push_back(target);
    }
  }
  for (const scheduler::ReplicaTarget target : affected) {
    context.event_queue().push(
        time,
        ReplicaSchedulePayload{
            .replica_id = target.replica_id,
            .dp_id = target.dp_id,
            .cluster_type = payload.cluster_type,
        });
  }
}

}  // namespace frontier::events
