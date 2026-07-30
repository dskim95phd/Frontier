#include "frontier/core/event_queue.h"
#include "tests/test_support.h"

#include <cmath>
#include <limits>
#include <tuple>
#include <vector>

namespace {

using frontier::BatchId;
using frontier::ClusterBatchEndPayload;
using frontier::ClusterType;
using frontier::DataParallelId;
using frontier::Event;
using frontier::EventQueue;
using frontier::EventSequence;
using frontier::EventType;
using frontier::Generation;
using frontier::GlobalSchedulePayload;
using frontier::ReplicaId;
using frontier::RequestArrivalPayload;
using frontier::RequestId;
using frontier::SimTime;
using frontier::test::expect;
using frontier::test::expect_throws;

void test_default_ids_and_times_are_invalid() {
    const RequestId request_id;
    const SimTime time;
    expect(request_id.value() == -1 && !request_id.valid(),
           "default StrongId must use the -1 invalid sentinel");
    expect(time.seconds() == -1.0 && !time.valid(),
           "default SimTime must use the -1 second invalid sentinel");
    expect(RequestId{0}.valid() && SimTime::from_seconds(0.0).valid(),
           "zero IDs and times must remain valid");

    EventQueue queue;
    expect_throws<std::invalid_argument>(
        [&queue] {
            queue.push(SimTime{}, [&]() {
                RequestArrivalPayload value{};
                value.request_id = RequestId{0};
                value.cluster_type = ClusterType::kMonolithic;
                return value;
            }());
        },
        "invalid sentinel time must not enter the event queue");
}

void test_earlier_time_precedes_later_time() {
    EventQueue queue;
    const EventSequence later = queue.push(SimTime::from_seconds(2.0), [&]() {
        GlobalSchedulePayload value{};
        value.cluster_type = ClusterType::kMonolithic;
        return value;
    }());
    const EventSequence earlier = queue.push(SimTime::from_seconds(1.0), [&]() {
        RequestArrivalPayload value{};
        value.request_id = RequestId{0};
        value.cluster_type = ClusterType::kMonolithic;
        return value;
    }());

    expect(queue.size() == 2, "queue size must reflect inserted events");
    expect(queue.pop().sequence == earlier, "earlier time must pop first");
    expect(queue.pop().sequence == later, "later time must pop second");
    expect(queue.empty(), "queue must be empty after both events pop");
}

void test_equal_times_preserve_creation_sequence() {
    EventQueue queue;
    const EventSequence first = queue.push(SimTime::from_seconds(3.0), [&]() {
        GlobalSchedulePayload value{};
        value.cluster_type = ClusterType::kMonolithic;
        return value;
    }());
    const EventSequence second = queue.push(SimTime::from_seconds(3.0), [&]() {
        RequestArrivalPayload value{};
        value.request_id = RequestId{0};
        value.cluster_type = ClusterType::kMonolithic;
        return value;
    }());

    expect(queue.pop().sequence == first,
           "event type must not override equal-time creation sequence");
    expect(queue.pop().sequence == second,
           "second equal-time event must remain second");
}

void test_nonfinite_times_are_rejected() {
    EventQueue queue;
    expect_throws<std::invalid_argument>(
        [&queue] {
            queue.push(
                SimTime::from_seconds(std::numeric_limits<double>::quiet_NaN()),
                [&]() {
                    RequestArrivalPayload value{};
                    value.request_id = RequestId{0};
                    value.cluster_type = ClusterType::kMonolithic;
                    return value;
                }());
        },
        "NaN event time must be rejected");
    expect_throws<std::invalid_argument>(
        [&queue] {
            queue.push(
                SimTime::from_seconds(std::numeric_limits<double>::infinity()),
                [&]() {
                    RequestArrivalPayload value{};
                    value.request_id = RequestId{0};
                    value.cluster_type = ClusterType::kMonolithic;
                    return value;
                }());
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
    queue.push(SimTime::from_seconds(0.0), [&]() {
        ClusterBatchEndPayload value{};
        value.batch_id = BatchId{7};
        value.replica_id = ReplicaId{0};
        value.dp_id = DataParallelId{0};
        value.generation = Generation{3};
        value.cluster_type = ClusterType::kMonolithic;
        return value;
    }());

    const Event event = queue.pop();
    expect(!event.is_stale(Generation{3}),
           "matching generation must remain current");
    expect(event.is_stale(Generation{4}), "different generation must be stale");
}

using TraceEntry = std::tuple<double, std::uint64_t, EventType>;

std::vector<TraceEntry> make_deterministic_trace() {
    EventQueue queue;
    queue.push(SimTime::from_seconds(2.0), [&]() {
        GlobalSchedulePayload value{};
        value.cluster_type = ClusterType::kMonolithic;
        return value;
    }());
    queue.push(SimTime::from_seconds(1.0), [&]() {
        GlobalSchedulePayload value{};
        value.cluster_type = ClusterType::kMonolithic;
        return value;
    }());
    queue.push(SimTime::from_seconds(1.0), [&]() {
        RequestArrivalPayload value{};
        value.request_id = RequestId{2};
        value.cluster_type = ClusterType::kMonolithic;
        return value;
    }());
    queue.push(SimTime::from_seconds(4.0), [&]() {
        RequestArrivalPayload value{};
        value.request_id = RequestId{3};
        value.cluster_type = ClusterType::kMonolithic;
        return value;
    }());

    std::vector<TraceEntry> trace;
    while (!queue.empty()) {
        const Event event = queue.pop();
        trace.emplace_back(event.time.seconds(), event.sequence.value(),
                           event.type());
    }
    return trace;
}

void test_repeated_fixture_is_deterministic() {
    const auto expected = make_deterministic_trace();
    for (int iteration = 0; iteration < 100; ++iteration) {
        expect(make_deterministic_trace() == expected,
               "repeated event traces must be identical");
    }
}

} // namespace

int main() {
    int failures = 0;
    failures += frontier::test::run("default IDs and times are invalid",
                                    test_default_ids_and_times_are_invalid);
    failures += frontier::test::run("earlier time precedes later time",
                                    test_earlier_time_precedes_later_time);
    failures +=
        frontier::test::run("equal times preserve creation sequence",
                            test_equal_times_preserve_creation_sequence);
    failures += frontier::test::run("nonfinite times are rejected",
                                    test_nonfinite_times_are_rejected);
    failures += frontier::test::run("empty queue access is explicit",
                                    test_empty_queue_access_is_explicit);
    failures += frontier::test::run("generation staleness uses IDs",
                                    test_generation_staleness_uses_ids);
    failures += frontier::test::run("repeated fixture is deterministic",
                                    test_repeated_fixture_is_deterministic);
    return failures == 0 ? 0 : 1;
}
