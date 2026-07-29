#include "frontier/core/event_queue.h"
#include "tests/test_support.h"

#include <cmath>
#include <limits>
#include <tuple>
#include <vector>

namespace {

using frontier::Event;
using frontier::EventPayload;
using frontier::EventQueue;
using frontier::EventSequence;
using frontier::EventType;
using frontier::Generation;
using frontier::RequestId;
using frontier::SimTime;
using frontier::test::expect;
using frontier::test::expect_throws;

void test_earlier_time_precedes_later_time() {
  EventQueue queue;
  const EventSequence later = queue.push(
      SimTime::from_seconds(2.0),
      EventType::kFoundationCompletion);
  const EventSequence earlier = queue.push(
      SimTime::from_seconds(1.0),
      EventType::kRequestArrival);

  expect(queue.size() == 2, "queue size must reflect inserted events");
  expect(queue.pop().sequence == earlier, "earlier time must pop first");
  expect(queue.pop().sequence == later, "later time must pop second");
  expect(queue.empty(), "queue must be empty after both events pop");
}

void test_equal_times_preserve_creation_sequence() {
  EventQueue queue;
  const EventSequence first = queue.push(
      SimTime::from_seconds(3.0),
      EventType::kFoundationCompletion);
  const EventSequence second = queue.push(
      SimTime::from_seconds(3.0),
      EventType::kRequestArrival);

  expect(
      queue.pop().sequence == first,
      "event type must not override equal-time creation sequence");
  expect(
      queue.pop().sequence == second,
      "second equal-time event must remain second");
}

void test_nonfinite_times_are_rejected() {
  EventQueue queue;
  expect_throws<std::invalid_argument>(
      [&queue] {
        queue.push(
            SimTime::from_seconds(
                std::numeric_limits<double>::quiet_NaN()),
            EventType::kRequestArrival);
      },
      "NaN event time must be rejected");
  expect_throws<std::invalid_argument>(
      [&queue] {
        queue.push(
            SimTime::from_seconds(
                std::numeric_limits<double>::infinity()),
            EventType::kRequestArrival);
      },
      "infinite event time must be rejected");
  expect(queue.empty(), "invalid events must not enter the queue");
}

void test_empty_queue_access_is_explicit() {
  EventQueue queue;
  expect_throws<std::out_of_range>(
      [&queue] { static_cast<void>(queue.top()); },
      "top on an empty queue must throw");
  expect_throws<std::out_of_range>(
      [&queue] { static_cast<void>(queue.pop()); },
      "pop on an empty queue must throw");
}

void test_generation_staleness_uses_ids() {
  EventQueue queue;
  EventPayload payload;
  payload.request_id = RequestId{7};
  payload.generation = Generation{3};
  queue.push(
      SimTime::from_seconds(0.0),
      EventType::kRequestArrival,
      payload);

  const Event event = queue.pop();
  expect(
      !event.is_stale(Generation{3}),
      "matching generation must remain current");
  expect(
      event.is_stale(Generation{4}),
      "different generation must be stale");
}

using TraceEntry = std::tuple<double, std::uint64_t, EventType>;

std::vector<TraceEntry> make_deterministic_trace() {
  EventQueue queue;
  queue.push(
      SimTime::from_seconds(2.0),
      EventType::kFoundationCompletion);
  queue.push(
      SimTime::from_seconds(1.0),
      EventType::kFoundationCompletion);
  queue.push(
      SimTime::from_seconds(1.0),
      EventType::kRequestArrival);
  queue.push(
      SimTime::from_seconds(4.0),
      EventType::kRequestArrival);

  std::vector<TraceEntry> trace;
  while (!queue.empty()) {
    const Event event = queue.pop();
    trace.emplace_back(
        event.time.seconds(),
        event.sequence.value(),
        event.type);
  }
  return trace;
}

void test_repeated_fixture_is_deterministic() {
  const auto expected = make_deterministic_trace();
  for (int iteration = 0; iteration < 100; ++iteration) {
    expect(
        make_deterministic_trace() == expected,
        "repeated event traces must be identical");
  }
}

}  // namespace

int main() {
  int failures = 0;
  failures += frontier::test::run(
      "earlier time precedes later time",
      test_earlier_time_precedes_later_time);
  failures += frontier::test::run(
      "equal times preserve creation sequence",
      test_equal_times_preserve_creation_sequence);
  failures += frontier::test::run(
      "nonfinite times are rejected",
      test_nonfinite_times_are_rejected);
  failures += frontier::test::run(
      "empty queue access is explicit",
      test_empty_queue_access_is_explicit);
  failures += frontier::test::run(
      "generation staleness uses IDs",
      test_generation_staleness_uses_ids);
  failures += frontier::test::run(
      "repeated fixture is deterministic",
      test_repeated_fixture_is_deterministic);
  return failures == 0 ? 0 : 1;
}
