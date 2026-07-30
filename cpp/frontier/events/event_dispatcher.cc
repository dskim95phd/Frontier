#include "frontier/events/event_dispatcher.h"

#include "frontier/events/event_handlers.h"
#include "frontier/simulator/simulator.h"

namespace frontier::events {

void EventDispatcher::dispatch(const Event &event,
                               simulator::Simulator &simulator) const {
    std::visit(
        [&](const auto &payload) {
            handle_event(payload, event.time, simulator);
        },
        event.payload);
}

} // namespace frontier::events
