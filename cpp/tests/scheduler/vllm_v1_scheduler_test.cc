#include "frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.h"
#include "tests/test_support.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace {

using frontier::BatchId;
using frontier::Generation;
using frontier::RequestId;
using frontier::SimTime;
using frontier::config::SchedulerConfig;
using frontier::entities::Batch;
using frontier::entities::Request;
using frontier::entities::RequestBatchSnapshot;
using frontier::request_generator::WorkloadRequest;
using frontier::scheduler::ScheduleResult;
using frontier::scheduler::VllmV1Scheduler;
using frontier::test::expect;

SchedulerConfig scheduler_config() {
  SchedulerConfig config;
  config.batch_size_cap = 8;
  config.max_tokens_in_batch = 128;
  config.block_size = 16;
  config.num_blocks = 128;
  return config;
}

std::vector<Request> make_requests(
    const std::vector<std::pair<std::uint64_t, std::uint64_t>>& tokens) {
  std::vector<Request> requests;
  requests.reserve(tokens.size());
  for (std::size_t index = 0; index < tokens.size(); ++index) {
    requests.emplace_back(WorkloadRequest{
        .request_id = RequestId{
            static_cast<RequestId::ValueType>(index)},
        .arrived_at = SimTime::from_seconds(0.0),
        .num_prefill_tokens = tokens[index].first,
        .num_decode_tokens = tokens[index].second,
        .session_id = frontier::SessionId{},
        .session_turn_index = std::nullopt,
    });
  }
  return requests;
}

void arrive_all(
    VllmV1Scheduler& scheduler,
    std::vector<Request>& requests) {
  for (Request& request : requests) {
    request.on_arrival(request.arrived_at());
    scheduler.add_request(request.id());
  }
}

Batch complete_schedule(
    VllmV1Scheduler& scheduler,
    std::vector<Request>& requests,
    const ScheduleResult& schedule,
    std::uint64_t batch_id,
    double completion_time) {
  std::vector<RequestBatchSnapshot> snapshots;
  for (const auto& scheduled : schedule.scheduled_requests) {
    const Request& request =
        requests.at(static_cast<std::size_t>(
            scheduled.request_id.value()));
    snapshots.push_back(RequestBatchSnapshot{
        .request_id = scheduled.request_id,
        .scheduled_tokens = scheduled.num_tokens,
        .runtime_epoch = request.runtime_epoch(),
        .execution_epoch = request.execution_epoch(),
        .processed_tokens = request.num_processed_tokens(),
        .scheduler_frontier =
            request.scheduler_num_computed_tokens(),
    });
  }
  Batch batch{
      BatchId{batch_id},
      schedule.iteration_id,
      std::move(snapshots),
      schedule.simulation_time,
      Generation{batch_id + 1}};
  scheduler.mark_batch_started(batch);
  expect(
      scheduler.on_batch_completed(
          batch,
          SimTime::from_seconds(completion_time)),
      "scheduled batch must mutate at least one request");
  return batch;
}

void test_fcfs_continuous_batch_admission_and_order() {
  auto requests = make_requests({{2, 1}, {2, 1}, {2, 1}});
  SchedulerConfig config = scheduler_config();
  config.batch_size_cap = 2;
  VllmV1Scheduler scheduler{config, requests};
  arrive_all(scheduler, requests);

  const ScheduleResult first =
      scheduler.schedule(SimTime::from_seconds(0.0));
  expect(
      first.scheduled_requests.size() == 2,
      "batch size cap must limit initial admissions");
  expect(
      first.scheduled_requests[0].request_id == RequestId{0} &&
          first.scheduled_requests[1].request_id == RequestId{1},
      "initial admissions must preserve FCFS order");
  static_cast<void>(
      complete_schedule(scheduler, requests, first, 0, 0.001));

  const ScheduleResult second =
      scheduler.schedule(SimTime::from_seconds(0.001));
  expect(
      second.scheduled_requests.size() == 1 &&
          second.scheduled_requests[0].request_id == RequestId{2},
      "next waiting request must enter after capacity is released");
}

void test_oversized_unchunked_head_is_skipped_for_follower() {
  auto requests = make_requests({{8, 1}, {2, 1}});
  SchedulerConfig config = scheduler_config();
  config.max_tokens_in_batch = 4;
  config.enable_chunked_prefill = false;
  VllmV1Scheduler scheduler{config, requests};
  arrive_all(scheduler, requests);

  const ScheduleResult schedule =
      scheduler.schedule(SimTime::from_seconds(0.0));
  expect(
      schedule.scheduled_requests.size() == 1 &&
          schedule.scheduled_requests[0].request_id == RequestId{1},
      "oversized head must not block smaller FCFS follower this iteration");
  expect(
      scheduler.waiting_queue().size() == 1 &&
          scheduler.waiting_queue().front() == RequestId{0},
      "skipped request must remain waiting");
  expect(
      schedule.decisions.size() == 1 &&
          schedule.decisions.front().request_id == RequestId{1},
      "production decision stream must contain only the admitted follower");
}

void test_chunked_prefill_runs_before_new_waiting_work() {
  auto requests = make_requests({{10, 1}, {2, 1}});
  SchedulerConfig config = scheduler_config();
  config.max_tokens_in_batch = 4;
  config.enable_chunked_prefill = true;
  VllmV1Scheduler scheduler{config, requests};
  arrive_all(scheduler, requests);

  const ScheduleResult first =
      scheduler.schedule(SimTime::from_seconds(0.0));
  expect(
      first.scheduled_requests.size() == 1 &&
          first.scheduled_requests[0].request_id == RequestId{0} &&
          first.scheduled_requests[0].num_tokens == 4,
      "first long prompt chunk must consume the budget");
  static_cast<void>(
      complete_schedule(scheduler, requests, first, 0, 0.001));

  const ScheduleResult second =
      scheduler.schedule(SimTime::from_seconds(0.001));
  expect(
      second.scheduled_requests.size() == 1 &&
          second.scheduled_requests[0].request_id == RequestId{0},
      "running partial prefill must run before waiting request");
  static_cast<void>(
      complete_schedule(scheduler, requests, second, 1, 0.002));

  const ScheduleResult third =
      scheduler.schedule(SimTime::from_seconds(0.002));
  expect(
      third.scheduled_requests.size() == 2 &&
          third.scheduled_requests[0].request_id == RequestId{1} &&
          third.scheduled_requests[1].request_id == RequestId{0},
      "new admissions must precede running continuation in emitted batch");
  expect(
      third.scheduled_requests[0].num_tokens == 2 &&
          third.scheduled_requests[1].num_tokens == 2,
      "remaining budget must mix waiting and running prefill work");
}

void test_decode_frontier_grows_kv_at_boundary() {
  auto requests = make_requests({{4, 2}});
  SchedulerConfig config = scheduler_config();
  config.block_size = 4;
  config.num_blocks = 2;
  VllmV1Scheduler scheduler{config, requests};
  arrive_all(scheduler, requests);

  const ScheduleResult prefill =
      scheduler.schedule(SimTime::from_seconds(0.0));
  static_cast<void>(
      complete_schedule(scheduler, requests, prefill, 0, 0.001));
  expect(
      scheduler.kv_blocks().allocated_blocks(RequestId{0}) == 1,
      "prefill must own one exact-boundary block");

  const ScheduleResult decode =
      scheduler.schedule(SimTime::from_seconds(0.001));
  expect(
      decode.scheduled_requests.size() == 1 &&
          decode.scheduled_requests[0].num_tokens == 1,
      "one subsequent decode token is scheduled per iteration");
  expect(
      scheduler.kv_blocks().allocated_blocks(RequestId{0}) == 2,
      "first scheduled decode step must grow KV beyond boundary");
  static_cast<void>(
      complete_schedule(scheduler, requests, decode, 1, 0.002));
  expect(
      requests[0].completed() && scheduler.kv_blocks().empty(),
      "completion must free all KV blocks");
}

void test_preemption_rolls_back_same_iteration_victim() {
  auto requests = make_requests({{3, 3}, {4, 2}});
  SchedulerConfig config = scheduler_config();
  config.block_size = 4;
  config.num_blocks = 2;
  config.max_tokens_in_batch = 8;
  config.enable_preemption = true;
  VllmV1Scheduler scheduler{config, requests};
  arrive_all(scheduler, requests);

  const ScheduleResult prefill =
      scheduler.schedule(SimTime::from_seconds(0.0));
  static_cast<void>(
      complete_schedule(scheduler, requests, prefill, 0, 0.001));

  Batch stale_batch{
      BatchId{99},
      frontier::IterationId{99},
      {
          RequestBatchSnapshot{
              .request_id = RequestId{0},
              .scheduled_tokens = 1,
              .runtime_epoch = requests[0].runtime_epoch(),
              .execution_epoch = requests[0].execution_epoch(),
              .processed_tokens =
                  requests[0].num_processed_tokens(),
              .scheduler_frontier =
                  requests[0].scheduler_num_computed_tokens(),
          },
      },
      SimTime::from_seconds(0.001),
      Generation{100}};

  const ScheduleResult pressure =
      scheduler.schedule(SimTime::from_seconds(0.001));
  expect(
      pressure.preempted_count == 1,
      "memory pressure must preempt one FCFS tail victim");
  expect(
      pressure.scheduled_requests.size() == 1 &&
          pressure.scheduled_requests[0].request_id == RequestId{1},
      "same-iteration victim schedule must be rolled back");
  expect(
      requests[0].preempted() &&
          requests[0].num_processed_tokens() == 0,
      "victim must reset for recomputation");
  expect(
      pressure.token_budget_after == 7,
      "rolled-back victim token must be refunded before requester schedule");
  static_cast<void>(
      complete_schedule(scheduler, requests, pressure, 1, 0.002));

  const std::uint64_t progress_before_stale =
      requests[0].num_processed_tokens();
  scheduler.mark_batch_started(stale_batch);
  expect(
      !scheduler.on_batch_completed(
          stale_batch,
          SimTime::from_seconds(0.0025)),
      "epoch-mismatched stale completion must be ignored");
  expect(
      requests[0].num_processed_tokens() == progress_before_stale,
      "stale batch completion must not mutate request progress");

  std::uint64_t batch_id = 2;
  double time = 0.002;
  while (!scheduler.idle() && batch_id < 10) {
    const ScheduleResult next =
        scheduler.schedule(SimTime::from_seconds(time));
    expect(
        !next.scheduled_requests.empty(),
        "preempted request must eventually make progress");
    time += 0.001;
    static_cast<void>(
        complete_schedule(
            scheduler,
            requests,
            next,
            batch_id++,
            time));
  }
  expect(
      scheduler.idle() && requests[0].completed() &&
          requests[1].completed(),
      "recompute preemption must recover to successful quiescence");
}

void test_mixed_stale_and_valid_request_snapshots_are_applied_per_request() {
  auto requests = make_requests({{4, 2}, {4, 2}});
  VllmV1Scheduler scheduler{scheduler_config(), requests};
  arrive_all(scheduler, requests);

  const ScheduleResult schedule =
      scheduler.schedule(SimTime::from_seconds(0.0));
  expect(
      schedule.scheduled_requests.size() == 2,
      "fixture must schedule both requests");

  std::vector<RequestBatchSnapshot> snapshots;
  for (const auto& scheduled : schedule.scheduled_requests) {
    const Request& request =
        requests.at(static_cast<std::size_t>(
            scheduled.request_id.value()));
    snapshots.push_back(RequestBatchSnapshot{
        .request_id = scheduled.request_id,
        .scheduled_tokens = scheduled.num_tokens,
        .runtime_epoch = request.runtime_epoch(),
        .execution_epoch = request.execution_epoch(),
        .processed_tokens = request.num_processed_tokens(),
        .scheduler_frontier =
            request.scheduler_num_computed_tokens(),
    });
  }
  Batch batch{
      BatchId{99},
      schedule.iteration_id,
      std::move(snapshots),
      schedule.simulation_time,
      Generation{100}};

  requests[0].on_batch_completion(
      SimTime::from_seconds(0.0005), 1);
  const std::uint64_t stale_request_progress =
      requests[0].num_processed_tokens();

  scheduler.mark_batch_started(batch);
  expect(
      scheduler.on_batch_completed(
          batch, SimTime::from_seconds(0.001)),
      "a valid request snapshot in a mixed batch must still be applied");
  expect(
      requests[0].num_processed_tokens() == stale_request_progress,
      "a stale request snapshot must not mutate that request");
  expect(
      requests[1].num_processed_tokens() ==
          requests[1].num_prefill_tokens() + 1,
      "a valid request snapshot in the same batch must mutate its request");
  expect(
      batch.completed(),
      "a generation-valid mixed batch must complete");
}

void test_requester_self_preemption_and_disabled_pressure() {
  const auto reach_second_decode_boundary =
      [](VllmV1Scheduler& scheduler,
         std::vector<Request>& requests) {
        const ScheduleResult prefill =
            scheduler.schedule(SimTime::from_seconds(0.0));
        static_cast<void>(complete_schedule(
            scheduler, requests, prefill, 0, 0.001));
        const ScheduleResult first_decode =
            scheduler.schedule(SimTime::from_seconds(0.001));
        static_cast<void>(complete_schedule(
            scheduler, requests, first_decode, 1, 0.002));
      };

  auto preempting_requests = make_requests({{3, 3}});
  SchedulerConfig preempting_config = scheduler_config();
  preempting_config.block_size = 4;
  preempting_config.num_blocks = 1;
  preempting_config.enable_preemption = true;
  VllmV1Scheduler preempting{
      preempting_config, preempting_requests};
  arrive_all(preempting, preempting_requests);
  reach_second_decode_boundary(preempting, preempting_requests);

  const ScheduleResult self_pressure =
      preempting.schedule(SimTime::from_seconds(0.002));
  expect(
      self_pressure.preempted_count == 1 &&
          self_pressure.scheduled_requests.empty(),
      "a sole requester must self-preempt when KV cannot grow");
  expect(
      preempting_requests[0].preempted() &&
          preempting_requests[0].num_processed_tokens() == 0 &&
          preempting.waiting_queue().front() == RequestId{0} &&
          preempting.kv_blocks().empty(),
      "self-preemption must reset recompute state and free KV");

  auto disabled_requests = make_requests({{3, 3}});
  SchedulerConfig disabled_config = preempting_config;
  disabled_config.enable_preemption = false;
  VllmV1Scheduler disabled{disabled_config, disabled_requests};
  arrive_all(disabled, disabled_requests);
  reach_second_decode_boundary(disabled, disabled_requests);

  const ScheduleResult disabled_pressure =
      disabled.schedule(SimTime::from_seconds(0.002));
  expect(
      disabled_pressure.preempted_count == 0 &&
          disabled_pressure.scheduled_requests.empty(),
      "disabled preemption must leave the pressure iteration empty");
  expect(
      !disabled_requests[0].preempted() &&
          disabled_requests[0].num_processed_tokens() == 5 &&
          disabled.running_order().front() == RequestId{0} &&
          disabled.kv_blocks().allocated_blocks(RequestId{0}) == 1,
      "disabled pressure must preserve running progress and allocation");
}

}  // namespace

int main() {
  int failures = 0;
  failures += frontier::test::run(
      "FCFS continuous batch admission and order",
      test_fcfs_continuous_batch_admission_and_order);
  failures += frontier::test::run(
      "oversized unchunked head is skipped",
      test_oversized_unchunked_head_is_skipped_for_follower);
  failures += frontier::test::run(
      "chunked prefill runs before waiting work",
      test_chunked_prefill_runs_before_new_waiting_work);
  failures += frontier::test::run(
      "decode frontier grows KV at boundary",
      test_decode_frontier_grows_kv_at_boundary);
  failures += frontier::test::run(
      "preemption rolls back same-iteration victim",
      test_preemption_rolls_back_same_iteration_victim);
  failures += frontier::test::run(
      "mixed stale and valid snapshots apply per request",
      test_mixed_stale_and_valid_request_snapshots_are_applied_per_request);
  failures += frontier::test::run(
      "requester self-preemption and disabled pressure",
      test_requester_self_preemption_and_disabled_pressure);
  return failures == 0 ? 0 : 1;
}
