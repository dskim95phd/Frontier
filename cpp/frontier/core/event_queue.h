#pragma once

#include <cstddef>
#include <cstdint>
#include <queue>
#include <utility>
#include <vector>

#include "frontier/core/event.h"

namespace frontier {

class EventQueue {
 public:
  EventQueue() = default;

  template <typename Payload>
  EventSequence push(SimTime time, Payload payload) {
    return push_payload(
        time, EventPayload{std::move(payload)});
  }

  [[nodiscard]] bool empty() const noexcept { return events_.empty(); }
  [[nodiscard]] std::size_t size() const noexcept { return events_.size(); }
  [[nodiscard]] const Event& top() const;
  Event pop();

 private:
  EventSequence push_payload(
      SimTime time,
      EventPayload payload);

  struct LaterEvent {
    [[nodiscard]] bool operator()(
        const Event& left,
        const Event& right) const noexcept;
  };

  std::priority_queue<Event, std::vector<Event>, LaterEvent> events_;
  std::uint64_t next_sequence_ = 1;
};

}  // namespace frontier
