#pragma once

#include "frontier/core/event.h"

namespace frontier::simulator {
class SimulationContext;
}

namespace frontier::events {

class BaseEventHandler {
 public:
  virtual ~BaseEventHandler() = default;
  [[nodiscard]] virtual EventType type() const noexcept = 0;
  virtual void handle(
      const Event& event,
      simulator::SimulationContext& context) const = 0;
};

}  // namespace frontier::events
