#include "frontier/entities/batch.h"
#include "tests/test_support.h"

#include <vector>

namespace {

using frontier::BatchId;
using frontier::Generation;
using frontier::IterationId;
using frontier::RequestId;
using frontier::SimTime;
using frontier::entities::Batch;
using frontier::entities::BatchError;
using frontier::entities::RequestBatchSnapshot;
using frontier::test::expect;
using frontier::test::expect_throws;

RequestBatchSnapshot snapshot(RequestId id, std::uint64_t tokens) {
    return [&]() {
        RequestBatchSnapshot value{};
        value.request_id = id;
        value.scheduled_tokens = tokens;
        value.runtime_epoch = 0;
        value.execution_epoch = 0;
        value.processed_tokens = 0;
        value.scheduler_frontier = tokens;
        return value;
    }();
}

void test_batch_is_immutable_arena_value() {
    Batch batch{BatchId{3},
                IterationId{7},
                {snapshot(RequestId{0}, 4), snapshot(RequestId{1}, 2)},
                SimTime::from_seconds(1.0),
                Generation{9}};
    expect(batch.total_scheduled_tokens() == 6,
           "batch must sum scheduled tokens");
    expect(batch.schedule_epoch() == Generation{9},
           "batch generation must be retained");
    batch.mark_completed(SimTime::from_seconds(1.5));
    expect(batch.completed(), "batch completion must be explicit");
}

void test_invalid_batch_shape_is_rejected() {
    expect_throws<BatchError>(
        [] {
            static_cast<void>(Batch{BatchId{0},
                                    IterationId{0},
                                    {},
                                    SimTime::from_seconds(0.0),
                                    Generation{1}});
        },
        "empty batch must be rejected");
    expect_throws<BatchError>(
        [] {
            static_cast<void>(Batch{BatchId{0},
                                    IterationId{0},
                                    {
                                        snapshot(RequestId{0}, 1),
                                        snapshot(RequestId{0}, 1),
                                    },
                                    SimTime::from_seconds(0.0),
                                    Generation{1}});
        },
        "duplicate request IDs must be rejected");
}

} // namespace

int main() {
    int failures = 0;
    failures += frontier::test::run("batch is immutable arena value",
                                    test_batch_is_immutable_arena_value);
    failures += frontier::test::run("invalid batch shape is rejected",
                                    test_invalid_batch_shape_is_rejected);
    return failures == 0 ? 0 : 1;
}
