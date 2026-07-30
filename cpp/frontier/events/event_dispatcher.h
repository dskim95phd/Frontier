#pragma once

#include "frontier/core/event.h"

namespace frontier::simulator {
class SimulationContext;
}

namespace frontier::events {

class EventDispatcher {
 public:
  void dispatch(
      const Event& event,
      simulator::SimulationContext& context) const;
};

}  // namespace frontier::events
