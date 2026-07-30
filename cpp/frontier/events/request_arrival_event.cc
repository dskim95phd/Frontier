#include "frontier/events/event_handlers.h"

#include "frontier/simulator/simulation_context.h"

namespace frontier::events {

void handle_event(
    const RequestArrivalPayload& payload,
    SimTime time,
    simulator::SimulationContext& context) {
  entities::Request& request = context.request(payload.request_id);
  request.on_arrival(time, payload.cluster_type);
  context.global_scheduler().add_request(
      payload.request_id, payload.cluster_type);
  context.event_queue().push(
      time,
      GlobalSchedulePayload{
          .cluster_type = payload.cluster_type,
      });
}

}  // namespace frontier::events
