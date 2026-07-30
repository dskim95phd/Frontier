#include "frontier/events/event_handlers.h"

#include "frontier/simulator/simulator.h"

namespace frontier::events {

void handle_event(const RequestArrivalPayload &payload, SimTime time,
                  simulator::Simulator &simulator) {
    entities::Request &request = simulator.request(payload.request_id);
    request.on_arrival(time, payload.cluster_type);
    simulator.global_scheduler().add_request(payload.request_id,
                                             payload.cluster_type);
    simulator.event_queue().push(time, [&]() {
        GlobalSchedulePayload value{};
        value.cluster_type = payload.cluster_type;
        return value;
    }());
}

} // namespace frontier::events
