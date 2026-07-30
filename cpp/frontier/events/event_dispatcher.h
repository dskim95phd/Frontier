#pragma once

#include "frontier/core/event.h"

namespace frontier::simulator {
class Simulator;
}

namespace frontier::events {

class EventDispatcher {
  public:
    void dispatch(const Event &event, simulator::Simulator &simulator) const;
};

} // namespace frontier::events
