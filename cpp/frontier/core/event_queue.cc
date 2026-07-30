#include "frontier/core/event_queue.h"

#include <limits>
#include <stdexcept>
#include <utility>

namespace frontier {

EventSequence EventQueue::push_payload(
    SimTime time,
    EventPayload payload) {
  if (!time.valid()) {
    throw std::invalid_argument(
        "event time must be finite and nonnegative");
  }
  if (next_sequence_ == std::numeric_limits<std::uint64_t>::max()) {
    throw std::overflow_error("event sequence exhausted");
  }

  const EventSequence sequence{next_sequence_};
  ++next_sequence_;
  events_.push(Event{
      time,
      sequence,
      std::move(payload),
  });
  return sequence;
}

const Event& EventQueue::top() const {
  if (events_.empty()) {
    throw std::out_of_range("cannot inspect an empty event queue");
  }
  return events_.top();
}

Event EventQueue::pop() {
  if (events_.empty()) {
    throw std::out_of_range("cannot pop an empty event queue");
  }

  Event event = events_.top();
  events_.pop();
  return event;
}

bool EventQueue::LaterEvent::operator()(
    const Event& left,
    const Event& right) const noexcept {
  if (left.time.seconds() != right.time.seconds()) {
    return left.time.seconds() > right.time.seconds();
  }
  return left.sequence > right.sequence;
}

}  // namespace frontier
