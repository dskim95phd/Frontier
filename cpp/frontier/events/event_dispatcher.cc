#include "frontier/events/event_dispatcher.h"

#include "frontier/events/event_handlers.h"
#include "frontier/simulator/simulation_context.h"

namespace frontier::events {

void EventDispatcher::dispatch(
    const Event& event,
    simulator::SimulationContext& context) const {
  std::visit(
      [&](const auto& payload) {
        handle_event(payload, event.time, context);
      },
      event.payload);
}

}  // namespace frontier::events
