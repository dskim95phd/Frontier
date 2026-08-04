#include "frontier/entities/request.h"
#include "tests/test_support.h"

#include <vector>

namespace {

using frontier::RequestId;
using frontier::SimTime;
using frontier::entities::Request;
using frontier::entities::RequestError;
using frontier::entities::RequestState;
using frontier::request_generator::WorkloadRequest;
using frontier::test::expect;
using frontier::test::expect_throws;

Request make_request(std::uint64_t prefill = 4, std::uint64_t decode = 3) {
    return Request{[&]() {
        WorkloadRequest value{};
        value.request_id = RequestId{0};
        value.session_start_at = SimTime::from_seconds(0.0);
        value.num_prefill_tokens = prefill;
        value.num_decode_tokens = decode;
        value.session_id = frontier::SessionId{};
        value.session_turn_index = std::nullopt;
        return value;
    }()};
}

void test_monolithic_prefill_boundary_grants_first_token() {
    Request request = make_request(4, 1);
    request.on_arrival(SimTime::from_seconds(0.0));
    request.on_admitted(SimTime::from_seconds(0.25));
    request.advance_scheduler_frontier(4);
    request.on_batch_completion(SimTime::from_seconds(0.5), 4);

    expect(request.completed(), "decode-length-one request must complete");
    expect(request.num_processed_tokens() == 5,
           "prefill boundary must include the first decode token");
    expect(request.prefill_completed_at() == SimTime::from_seconds(0.5),
           "prefill completion timestamp must be recorded");
    expect(request.first_token_completed_at() == request.prefill_completed_at(),
           "first token must complete at monolithic prefill boundary");
    expect(request.cumulative_waiting_time_s() == 0.25,
           "initial queue waiting time must accumulate");
    expect(request.scheduled_prefill_tokens() == 4 &&
               request.preemption_recomputed_prefill_tokens() == 0,
           "cached-free prefill completion must count scheduled prompt work");
}

void test_chunked_prefill_and_decode_progress() {
    Request request = make_request(10, 3);
    request.on_arrival(SimTime::from_seconds(0.0));
    request.on_admitted(SimTime::from_seconds(0.0));

    for (const auto &[tokens, time] :
         std::vector<std::pair<std::uint64_t, double>>{
             {4, 0.1}, {4, 0.2}, {2, 0.3}}) {
        request.advance_scheduler_frontier(tokens);
        request.on_batch_completion(SimTime::from_seconds(time), tokens);
    }
    expect(request.is_prefill_complete(), "final chunk completes prefill");
    expect(request.num_processed_tokens() == 11,
           "final prefill chunk grants first decode token");
    expect(!request.completed(), "two decode tokens must remain");

    request.advance_scheduler_frontier(1);
    request.on_batch_completion(SimTime::from_seconds(0.4), 1);
    request.advance_scheduler_frontier(1);
    request.on_batch_completion(SimTime::from_seconds(0.5), 1);
    expect(request.completed(), "decode iterations must finish request");
    expect(request.completed_at() == SimTime::from_seconds(0.5),
           "final decode timestamp must be retained");
}

void test_preemption_resets_recompute_progress_and_epochs() {
    Request request = make_request(4, 2);
    request.on_arrival(SimTime::from_seconds(0.0));
    request.on_admitted(SimTime::from_seconds(0.0));
    request.advance_scheduler_frontier(4);
    request.on_batch_completion(SimTime::from_seconds(0.1), 4);
    const auto original_ttft = request.prefill_completed_at();

    request.on_preempted(SimTime::from_seconds(0.2));
    expect(request.state() == RequestState::kWaiting,
           "preempted request must re-enter waiting state");
    expect(request.num_processed_tokens() == 0 &&
               request.scheduler_num_computed_tokens() == 0,
           "preemption must reset recompute frontiers");
    expect(!request.is_prefill_complete(),
           "preemption must restore prefill work");
    expect(request.num_prefill_tokens() == 5 &&
               request.num_decode_tokens() == 1 &&
               request.initial_num_prefill_tokens() == 4 &&
               request.initial_num_decode_tokens() == 2,
           "committed decode progress must become replay-prefill without "
           "changing the initial token split");
    expect(request.runtime_epoch() == 1 && request.execution_epoch() == 1,
           "preemption must invalidate stale execution snapshots");
    expect(request.preemption_count() == 1 &&
               request.tokens_at_preemption() == std::vector<std::uint64_t>{4},
           "preemption metrics must retain prior scheduler frontier");

    request.on_admitted(SimTime::from_seconds(0.3));
    request.advance_scheduler_frontier(5);
    request.on_batch_completion(SimTime::from_seconds(0.4), 5);
    expect(request.completed(),
           "replay-prefill plus the remaining decode token must complete");
    expect(request.prefill_completed_at() == original_ttft,
           "canonical first prefill timestamp must be write-once");
    expect(request.scheduled_prefill_tokens() == 9 &&
               request.preemption_recomputed_prefill_tokens() == 5,
           "replayed prefill work must be counted exactly once");
}

void test_partial_and_repeated_prefill_replay_boundaries() {
    Request partial = make_request(10, 3);
    partial.on_arrival(SimTime::from_seconds(0.0));
    partial.on_admitted(SimTime::from_seconds(0.0));
    partial.advance_scheduler_frontier(6);
    partial.on_batch_completion(SimTime::from_seconds(0.1), 6);
    partial.on_preempted(SimTime::from_seconds(0.2));
    expect(partial.num_prefill_tokens() == 10 &&
               partial.num_decode_tokens() == 3,
           "partial-prefill preemption must preserve the original boundary");

    Request repeated = make_request(5, 5);
    repeated.on_arrival(SimTime::from_seconds(0.0));
    repeated.on_admitted(SimTime::from_seconds(0.0));
    repeated.advance_scheduler_frontier(5);
    repeated.on_batch_completion(SimTime::from_seconds(0.1), 5);
    for (std::uint64_t index = 0; index < 3; ++index) {
        repeated.advance_scheduler_frontier(1);
        repeated.on_batch_completion(
            SimTime::from_seconds(0.2 + static_cast<double>(index) * 0.1), 1);
    }
    expect(repeated.num_processed_tokens() == 9,
           "fixture must commit four decode tokens before preemption");
    repeated.on_preempted(SimTime::from_seconds(0.5));
    expect(repeated.num_prefill_tokens() == 9 &&
               repeated.num_decode_tokens() == 1,
           "decode progress must extend the replay-prefill boundary");

    repeated.on_admitted(SimTime::from_seconds(0.6));
    repeated.advance_scheduler_frontier(4);
    repeated.on_batch_completion(SimTime::from_seconds(0.7), 4);
    repeated.on_preempted(SimTime::from_seconds(0.8));
    expect(repeated.num_prefill_tokens() == 9 &&
               repeated.num_decode_tokens() == 1,
           "preempting an incomplete replay must not shrink its boundary");
}

void test_cached_prefill_is_not_scheduled_work() {
    Request request = make_request(8, 1);
    request.on_arrival(SimTime::from_seconds(0.0));
    request.restore_prefix_cache_lookup(2, 2, 4);
    request.on_admitted(SimTime::from_seconds(0.0));
    request.advance_scheduler_frontier(4);
    request.on_batch_completion(SimTime::from_seconds(0.1), 4);
    expect(request.scheduled_prefill_tokens() == 4,
           "cached prompt tokens must not be counted as scheduled PREFILL");
    expect(request.preemption_recomputed_prefill_tokens() == 0,
           "cache restoration without preemption is not replay work");
}

void test_invalid_transitions_are_rejected() {
    Request request = make_request();
    expect_throws<RequestError>(
        [&request] { request.on_admitted(SimTime::from_seconds(0.0)); },
        "pending request cannot be admitted");
    request.on_arrival(SimTime::from_seconds(0.0));
    expect_throws<RequestError>(
        [&request] { request.advance_scheduler_frontier(0); },
        "zero-token schedule must be rejected");
}

} // namespace

int main() {
    int failures = 0;
    failures += frontier::test::run(
        "monolithic prefill boundary grants first token",
        test_monolithic_prefill_boundary_grants_first_token);
    failures += frontier::test::run("chunked prefill and decode progress",
                                    test_chunked_prefill_and_decode_progress);
    failures += frontier::test::run(
        "preemption resets recompute progress and epochs",
        test_preemption_resets_recompute_progress_and_epochs);
    failures += frontier::test::run(
        "partial and repeated replay boundaries",
        test_partial_and_repeated_prefill_replay_boundaries);
    failures += frontier::test::run(
        "cached prefill is excluded from scheduled work",
        test_cached_prefill_is_not_scheduled_work);
    failures += frontier::test::run("invalid request transitions are rejected",
                                    test_invalid_transitions_are_rejected);
    return failures == 0 ? 0 : 1;
}
