#pragma once

#include <cstddef>
#include <cstdint>
#include <queue>
#include <vector>

#include "frontier/core/event.h"

namespace frontier {

class EventQueue {
 public:
  EventQueue() = default;

  EventSequence push(
      SimTime time,
      EventType type,
      EventPayload payload = {});

  [[nodiscard]] bool empty() const noexcept { return events_.empty(); }
  [[nodiscard]] std::size_t size() const noexcept { return events_.size(); }
  [[nodiscard]] const Event& top() const;
  Event pop();

 private:
  struct LaterEvent {
    [[nodiscard]] bool operator()(
        const Event& left,
        const Event& right) const noexcept;
  };

  std::priority_queue<Event, std::vector<Event>, LaterEvent> events_;
  std::uint64_t next_sequence_ = 1;
};

}  // namespace frontier
