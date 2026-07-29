#include "frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>

namespace frontier::scheduler {
namespace {

std::uint64_t checked_size(std::size_t value, const char* context) {
  if (value > std::numeric_limits<std::uint64_t>::max()) {
    throw SchedulerError(context);
  }
  return static_cast<std::uint64_t>(value);
}

}  // namespace

VllmV1Scheduler::VllmV1Scheduler(
    config::SchedulerConfig config,
    std::vector<entities::Request>& requests)
    : VllmV1Scheduler(
          std::move(config),
          requests,
          std::make_unique<
              execution_time_predictor::FixedBatchExecutionModel>(
              config::FixedExecutionModelConfig{
                  .batch_latency_ms = 0.0,
                  .stage_latencies_ms = {},
              })) {}

VllmV1Scheduler::VllmV1Scheduler(
    config::SchedulerConfig config,
    std::vector<entities::Request>& requests,
    std::unique_ptr<
        execution_time_predictor::BatchExecutionModel>
        execution_model,
    ReplicaId replica_id,
    DataParallelId dp_id,
    std::uint64_t pipeline_parallel_size)
    : BaseReplicaScheduler(
          replica_id,
          dp_id,
          std::move(execution_model),
          pipeline_parallel_size),
      config_(std::move(config)),
      requests_(&requests),
      kv_blocks_(config_),
      pipeline_parallel_size_(pipeline_parallel_size) {
  if (config_.type != config::SchedulerType::kVllmV1 ||
      config_.scheduling_policy != config::SchedulingPolicy::kFcfs) {
    throw SchedulerError("Step 2 requires the FCFS vLLM V1 scheduler");
  }
  if (config_.batch_size_cap == 0 ||
      config_.max_tokens_in_batch == 0 ||
      pipeline_parallel_size_ == 0) {
    throw SchedulerError("scheduler capacity values must be positive");
  }
}

entities::Request& VllmV1Scheduler::request(RequestId request_id) {
  if (request_id.value() >= requests_->size()) {
    throw SchedulerError("scheduler references an unknown request ID");
  }
  entities::Request& value =
      requests_->at(static_cast<std::size_t>(request_id.value()));
  if (value.id() != request_id) {
    throw SchedulerError("request arena ID/index invariant failed");
  }
  return value;
}

const entities::Request& VllmV1Scheduler::request(
    RequestId request_id) const {
  if (request_id.value() >= requests_->size()) {
    throw SchedulerError("scheduler references an unknown request ID");
  }
  const entities::Request& value =
      requests_->at(static_cast<std::size_t>(request_id.value()));
  if (value.id() != request_id) {
    throw SchedulerError("request arena ID/index invariant failed");
  }
  return value;
}

bool VllmV1Scheduler::contains_request(
    RequestId request_id) const {
  return std::find(
             preempted_.begin(),
             preempted_.end(),
             request_id) != preempted_.end() ||
      std::find(waiting_.begin(), waiting_.end(), request_id) !=
          waiting_.end() ||
      std::find(running_.begin(), running_.end(), request_id) !=
          running_.end();
}

bool VllmV1Scheduler::request_is_active(
    RequestId request_id) const {
  return active_requests_.contains(request_id);
}

void VllmV1Scheduler::add_request(RequestId request_id) {
  entities::Request& value = request(request_id);
  if (value.state() != entities::RequestState::kWaiting) {
    throw SchedulerError("scheduler accepts only waiting requests");
  }
  if (contains_request(request_id)) {
    throw SchedulerError("request is already present in scheduler state");
  }
  waiting_.push_back(request_id);
  validate_state();
}

std::uint64_t VllmV1Scheduler::next_num_tokens(
    const entities::Request& value) const {
  if (value.completed()) {
    throw SchedulerError("completed request cannot be scheduled");
  }
  if (!value.is_prefill_complete()) {
    if (value.scheduler_num_computed_tokens() >
        value.num_prefill_tokens()) {
      throw SchedulerError(
          "prefill scheduler frontier exceeds prompt length");
    }
    return value.num_prefill_tokens() -
        value.scheduler_num_computed_tokens();
  }
  if (value.num_processed_tokens() <
      value.scheduler_num_computed_tokens()) {
    throw SchedulerError(
        "request-visible decode progress trails scheduler frontier");
  }
  return value.num_processed_tokens() -
      value.scheduler_num_computed_tokens();
}

std::uint64_t VllmV1Scheduler::kv_accounted_tokens(
    const entities::Request& value) const {
  if (!value.is_prefill_complete()) {
    return value.scheduler_num_computed_tokens();
  }
  const std::uint64_t boundary_adjusted =
      value.num_processed_tokens() > 0
      ? std::max(
            value.num_prefill_tokens(),
            value.num_processed_tokens() - 1)
      : value.num_prefill_tokens();
  return std::max(
      value.scheduler_num_computed_tokens(),
      boundary_adjusted);
}

std::optional<RequestId>
VllmV1Scheduler::select_preemption_victim(
    RequestId requester) const {
  for (auto candidate = running_.rbegin();
       candidate != running_.rend();
       ++candidate) {
    if (*candidate != requester &&
        !request_is_active(*candidate) &&
        !request(*candidate).completed()) {
      return *candidate;
    }
  }
  return std::nullopt;
}

std::uint64_t
VllmV1Scheduler::extra_terminal_release_iterations() const noexcept {
  if (pipeline_parallel_size_ <= 1) {
    return 0;
  }
  return pipeline_parallel_size_ / 2 - 1;
}

std::uint64_t
VllmV1Scheduler::iteration_start_release_threshold() const noexcept {
  if (pipeline_parallel_size_ <= 4) {
    return 1;
  }
  return std::max<std::uint64_t>(
      pipeline_parallel_size_ / 4, 1);
}

bool VllmV1Scheduler::has_visible_waiting_requests() const noexcept {
  return !preempted_.empty() || !waiting_.empty();
}

void VllmV1Scheduler::free_completed_request(
    RequestId request_id) {
  entities::Request& value = request(request_id);
  if (!value.completed()) {
    throw SchedulerError(
        "terminal release references an incomplete request");
  }
  static_cast<void>(kv_blocks_.free(request_id));
  const auto position =
      std::find(running_.begin(), running_.end(), request_id);
  if (position == running_.end()) {
    throw SchedulerError(
        "terminal release request is missing from running order");
  }
  running_.erase(position);
}

void VllmV1Scheduler::
materialize_terminal_releases_before_iteration() {
  if (pending_terminal_release_iterations_.empty() ||
      has_visible_waiting_requests()) {
    return;
  }
  const std::uint64_t threshold =
      iteration_start_release_threshold();
  std::vector<RequestId> ready;
  for (const auto& [request_id, remaining] :
       pending_terminal_release_iterations_) {
    if (remaining <= threshold) {
      ready.push_back(request_id);
    }
  }
  std::sort(
      ready.begin(),
      ready.end(),
      [](RequestId left, RequestId right) {
        return left.value() < right.value();
      });
  for (const RequestId request_id : ready) {
    pending_terminal_release_iterations_.erase(request_id);
    waiting_sensitive_release_extensions_.erase(request_id);
    free_completed_request(request_id);
  }
}

void VllmV1Scheduler::advance_terminal_release_boundary() {
  if (pending_terminal_release_iterations_.empty()) {
    return;
  }
  const std::uint64_t threshold =
      iteration_start_release_threshold();
  std::vector<RequestId> ready;
  for (auto& [request_id, remaining] :
       pending_terminal_release_iterations_) {
    if (remaining <= threshold &&
        has_visible_waiting_requests() &&
        !waiting_sensitive_release_extensions_.contains(request_id)) {
      waiting_sensitive_release_extensions_.insert(request_id);
      remaining = 1;
      continue;
    }
    if (remaining <= 1) {
      ready.push_back(request_id);
    } else {
      --remaining;
    }
  }
  std::sort(
      ready.begin(),
      ready.end(),
      [](RequestId left, RequestId right) {
        return left.value() < right.value();
      });
  for (const RequestId request_id : ready) {
    pending_terminal_release_iterations_.erase(request_id);
    waiting_sensitive_release_extensions_.erase(request_id);
    free_completed_request(request_id);
  }
  terminal_release_followup_poll_pending_ =
      !pending_terminal_release_iterations_.empty() ||
      has_visible_waiting_requests();
}

void VllmV1Scheduler::preempt_request(
    RequestId victim,
    SimTime time,
    std::vector<RequestId>& newly_preempted,
    ScheduleResult& result) {
  entities::Request& value = request(victim);
  const auto running_position =
      std::find(running_.begin(), running_.end(), victim);
  if (running_position == running_.end()) {
    throw SchedulerError("preemption victim is not running");
  }
  running_.erase(running_position);
  static_cast<void>(kv_blocks_.free(victim));
  value.on_preempted(time);
  waiting_.push_front(victim);
  newly_preempted.push_back(victim);
  ++result.preempted_count;
  result.decisions.push_back(SchedulerDecision{
      .type = SchedulerDecisionType::kPreempted,
      .request_id = victim,
      .num_tokens = 0,
      .token_budget_after = result.token_budget_after,
      .available_blocks_after = kv_blocks_.available_blocks(),
  });
}

void VllmV1Scheduler::rollback_preempted_schedules(
    const std::vector<RequestId>& newly_preempted,
    std::vector<ScheduledRequest>& running_scheduled,
    std::uint64_t& token_budget) {
  if (newly_preempted.empty()) {
    return;
  }
  const std::unordered_set<
      RequestId,
      StrongIdHash<RequestId>>
      ids(newly_preempted.begin(), newly_preempted.end());
  std::vector<ScheduledRequest> kept;
  kept.reserve(running_scheduled.size());
  for (const ScheduledRequest& scheduled : running_scheduled) {
    if (ids.contains(scheduled.request_id)) {
      if (token_budget >
          std::numeric_limits<std::uint64_t>::max() -
              scheduled.num_tokens) {
        throw SchedulerError("token-budget refund overflows uint64");
      }
      token_budget += scheduled.num_tokens;
    } else {
      kept.push_back(scheduled);
    }
  }
  running_scheduled = std::move(kept);
}

bool VllmV1Scheduler::try_reserve_with_preemption(
    RequestId requester,
    std::uint64_t scheduled_tokens,
    SimTime time,
    std::vector<RequestId>& newly_preempted,
    std::vector<ScheduledRequest>& running_scheduled,
    std::uint64_t& token_budget,
    ScheduleResult& result) {
  while (true) {
    entities::Request& value = request(requester);
    const std::uint64_t accounted = kv_accounted_tokens(value);
    if (kv_blocks_.can_reserve(
            requester,
            accounted,
            scheduled_tokens)) {
      kv_blocks_.reserve(requester, accounted, scheduled_tokens);
      return true;
    }
    if (!config_.enable_preemption) {
      return false;
    }

    const std::size_t preemption_count_before =
        newly_preempted.size();
    const std::optional<RequestId> victim =
        select_preemption_victim(requester);
    preempt_request(
        victim.value_or(requester),
        time,
        newly_preempted,
        result);
    rollback_preempted_schedules(
        std::vector<RequestId>{
            newly_preempted.begin() +
                static_cast<std::ptrdiff_t>(preemption_count_before),
            newly_preempted.end()},
        running_scheduled,
        token_budget);
    result.token_budget_after = token_budget;
    if (!victim.has_value()) {
      return false;
    }
  }
}

ScheduleResult VllmV1Scheduler::schedule(SimTime time) {
  if (!std::isfinite(time.seconds()) || time.seconds() < 0.0) {
    throw SchedulerError("schedule time must be finite and nonnegative");
  }
  if (in_flight_batch_count_ >= pipeline_parallel_size_) {
    throw SchedulerError(
        "cannot schedule beyond pipeline in-flight capacity");
  }
  materialize_terminal_releases_before_iteration();
  validate_state();

  ScheduleResult result{
      .iteration_id = IterationId{next_iteration_id_++},
      .simulation_time = time,
      .token_budget_before = config_.max_tokens_in_batch,
      .token_budget_after = config_.max_tokens_in_batch,
      .available_blocks_before = kv_blocks_.available_blocks(),
      .available_blocks_after = kv_blocks_.available_blocks(),
      .waiting_count_before = checked_size(
          preempted_.size() + waiting_.size(),
          "waiting queue size overflows uint64"),
      .waiting_count_after = 0,
      .running_count_before = checked_size(
          running_.size(),
          "running queue size overflows uint64"),
      .running_count_after = 0,
      .preempted_count = 0,
      .decisions = {},
      .scheduled_requests = {},
  };

  if (terminal_release_followup_poll_pending_) {
    result.waiting_count_after = result.waiting_count_before;
    result.running_count_after = result.running_count_before;
    return result;
  }

  std::uint64_t token_budget = config_.max_tokens_in_batch;
  std::vector<ScheduledRequest> running_scheduled;
  std::vector<RequestId> preempted_requests;

  std::size_t running_index = 0;
  while (running_index < running_.size() && token_budget > 0) {
    const RequestId request_id = running_[running_index];
    entities::Request& value = request(request_id);
    if (value.completed()) {
      ++running_index;
      continue;
    }
    if (request_is_active(request_id)) {
      ++running_index;
      continue;
    }
    std::uint64_t num_tokens = next_num_tokens(value);
    if (!value.is_prefill_complete() &&
        config_.long_prefill_token_threshold > 0) {
      num_tokens = std::min(
          num_tokens,
          config_.long_prefill_token_threshold);
    }
    num_tokens = std::min(num_tokens, token_budget);
    if (num_tokens == 0) {
      ++running_index;
      continue;
    }

    const std::size_t preempted_before =
        preempted_requests.size();
    const bool reserved = try_reserve_with_preemption(
        request_id,
        num_tokens,
        time,
        preempted_requests,
        running_scheduled,
        token_budget,
        result);
    if (!reserved) {
      break;
    }

    value.advance_scheduler_frontier(num_tokens);
    running_scheduled.push_back(ScheduledRequest{
        .request_id = request_id,
        .num_tokens = num_tokens,
        .admitted_from_waiting = false,
    });
    token_budget -= num_tokens;
    result.token_budget_after = token_budget;
    result.decisions.push_back(SchedulerDecision{
        .type = SchedulerDecisionType::kRunningScheduled,
        .request_id = request_id,
        .num_tokens = num_tokens,
        .token_budget_after = token_budget,
        .available_blocks_after = kv_blocks_.available_blocks(),
    });
    ++running_index;

    if (preempted_requests.size() != preempted_before) {
      validate_state();
    }
  }

  std::vector<ScheduledRequest> waiting_scheduled;
  if (preempted_requests.empty() &&
      pending_terminal_release_iterations_.empty()) {
    std::deque<RequestId> queue = std::move(preempted_);
    queue.insert(
        queue.end(),
        waiting_.begin(),
        waiting_.end());
    preempted_.clear();
    waiting_.clear();
    std::deque<RequestId> skipped;

    while (!queue.empty() && token_budget > 0) {
      if (running_.size() >= config_.batch_size_cap) {
        break;
      }
      const RequestId request_id = queue.front();
      entities::Request& value = request(request_id);
      std::uint64_t num_tokens = next_num_tokens(value);
      if (!value.is_prefill_complete() &&
          config_.long_prefill_token_threshold > 0) {
        num_tokens = std::min(
            num_tokens,
            config_.long_prefill_token_threshold);
      }

      if (!config_.enable_chunked_prefill &&
          !value.is_prefill_complete() &&
          num_tokens > token_budget) {
        queue.pop_front();
        skipped.push_back(request_id);
        continue;
      }

      num_tokens = std::min(num_tokens, token_budget);
      if (num_tokens == 0) {
        queue.pop_front();
        continue;
      }
      const std::uint64_t accounted = kv_accounted_tokens(value);
      if (!kv_blocks_.can_reserve(
              request_id,
              accounted,
              num_tokens)) {
        break;
      }

      queue.pop_front();
      value.on_admitted(time);
      kv_blocks_.reserve(request_id, accounted, num_tokens);
      value.advance_scheduler_frontier(num_tokens);
      running_.push_back(request_id);
      waiting_scheduled.push_back(ScheduledRequest{
          .request_id = request_id,
          .num_tokens = num_tokens,
          .admitted_from_waiting = true,
      });
      token_budget -= num_tokens;
      result.token_budget_after = token_budget;
      result.decisions.push_back(SchedulerDecision{
          .type = SchedulerDecisionType::kAdmission,
          .request_id = request_id,
          .num_tokens = num_tokens,
          .token_budget_after = token_budget,
          .available_blocks_after = kv_blocks_.available_blocks(),
      });
    }

    queue.insert(queue.end(), skipped.begin(), skipped.end());
    for (const RequestId request_id : queue) {
      if (request(request_id).preempted()) {
        preempted_.push_back(request_id);
      } else {
        waiting_.push_back(request_id);
      }
    }
  }

  result.scheduled_requests.reserve(
      waiting_scheduled.size() + running_scheduled.size());
  result.scheduled_requests.insert(
      result.scheduled_requests.end(),
      waiting_scheduled.begin(),
      waiting_scheduled.end());
  result.scheduled_requests.insert(
      result.scheduled_requests.end(),
      running_scheduled.begin(),
      running_scheduled.end());
  result.token_budget_after = token_budget;
  result.available_blocks_after = kv_blocks_.available_blocks();
  result.waiting_count_after = checked_size(
      preempted_.size() + waiting_.size(),
      "waiting queue size overflows uint64");
  result.running_count_after = checked_size(
      running_.size(),
      "running queue size overflows uint64");
  const bool empty_iteration =
      result.scheduled_requests.empty();
  advance_terminal_release_boundary();
  if (empty_iteration) {
    result.available_blocks_after =
        kv_blocks_.available_blocks();
    result.waiting_count_after = checked_size(
        preempted_.size() + waiting_.size(),
        "waiting queue size overflows uint64");
    result.running_count_after = checked_size(
        running_.size(),
        "running queue size overflows uint64");
  }
  validate_state();
  return result;
}

void VllmV1Scheduler::mark_batch_started(
    const entities::Batch& batch) {
  if (in_flight_batch_count_ >= pipeline_parallel_size_) {
    throw SchedulerError(
        "a scheduler batch exceeds pipeline capacity");
  }
  std::vector<RequestId> request_ids;
  request_ids.reserve(batch.requests().size());
  for (const entities::RequestBatchSnapshot& snapshot :
       batch.requests()) {
    if (!active_requests_.insert(snapshot.request_id).second) {
      throw SchedulerError(
          "request is already active in another pipeline batch");
    }
    request_ids.push_back(snapshot.request_id);
  }
  if (!in_flight_batches_
           .emplace(batch.id(), std::move(request_ids))
           .second) {
    throw SchedulerError("batch is already in flight");
  }
  ++in_flight_batch_count_;
}

bool VllmV1Scheduler::on_batch_completed(
    entities::Batch& batch,
    SimTime time) {
  const auto in_flight = in_flight_batches_.find(batch.id());
  if (in_flight == in_flight_batches_.end()) {
    return false;
  }
  if (batch.completed()) {
    return false;
  }

  bool has_valid_request = false;
  for (const entities::RequestBatchSnapshot& snapshot :
       batch.requests()) {
    entities::Request& value = request(snapshot.request_id);
    if (value.completed() ||
        value.runtime_epoch() != snapshot.runtime_epoch ||
        value.execution_epoch() != snapshot.execution_epoch ||
        value.num_processed_tokens() != snapshot.processed_tokens ||
        value.scheduler_num_computed_tokens() !=
            snapshot.scheduler_frontier) {
      continue;
    }
    value.on_batch_completion(time, snapshot.scheduled_tokens);
    has_valid_request = true;
    if (value.completed()) {
      const std::uint64_t extra =
          extra_terminal_release_iterations();
      if (extra > 0) {
        auto [position, inserted] =
            pending_terminal_release_iterations_.try_emplace(
                value.id(), extra);
        if (!inserted) {
          position->second = std::max(position->second, extra);
        }
      } else {
        free_completed_request(value.id());
      }
    }
  }
  batch.mark_completed(time);
  for (const RequestId request_id : in_flight->second) {
    active_requests_.erase(request_id);
  }
  in_flight_batches_.erase(in_flight);
  if (in_flight_batch_count_ == 0) {
    throw SchedulerError("in-flight batch count underflow");
  }
  --in_flight_batch_count_;
  validate_state();
  return has_valid_request;
}

bool VllmV1Scheduler::
consume_terminal_release_followup_poll() noexcept {
  const bool pending = terminal_release_followup_poll_pending_;
  terminal_release_followup_poll_pending_ = false;
  return pending;
}

void VllmV1Scheduler::validate_state() const {
  std::unordered_set<RequestId, StrongIdHash<RequestId>> ids;
  for (const RequestId request_id : preempted_) {
    if (!ids.insert(request_id).second) {
      throw SchedulerError("request appears in multiple scheduler queues");
    }
    const entities::Request& value = request(request_id);
    if (value.state() != entities::RequestState::kWaiting ||
        !value.preempted()) {
      throw SchedulerError(
          "preempted queue contains a non-preempted waiting request");
    }
  }
  for (const RequestId request_id : waiting_) {
    if (!ids.insert(request_id).second) {
      throw SchedulerError("request appears in multiple scheduler queues");
    }
    if (request(request_id).state() !=
        entities::RequestState::kWaiting) {
      throw SchedulerError(
          "waiting queue contains a non-waiting request");
    }
  }
  for (const RequestId request_id : running_) {
    if (!ids.insert(request_id).second) {
      throw SchedulerError("request appears in multiple scheduler queues");
    }
    const entities::Request& value = request(request_id);
    const bool pending_completed =
        value.completed() &&
        pending_terminal_release_iterations_.contains(request_id);
    if (value.state() != entities::RequestState::kRunning &&
        !pending_completed) {
      throw SchedulerError(
          "running order contains a non-running request");
    }
  }
  if (running_.size() > config_.batch_size_cap) {
    throw SchedulerError("running request count exceeds batch_size_cap");
  }
  if (in_flight_batch_count_ != in_flight_batches_.size() ||
      in_flight_batch_count_ > pipeline_parallel_size_) {
    throw SchedulerError("in-flight batch registry invariant failed");
  }
  std::unordered_set<RequestId, StrongIdHash<RequestId>>
      expected_active;
  for (const auto& [batch_id, request_ids] :
       in_flight_batches_) {
    static_cast<void>(batch_id);
    for (const RequestId request_id : request_ids) {
      if (!expected_active.insert(request_id).second) {
        throw SchedulerError(
            "request appears in multiple in-flight batches");
      }
    }
  }
  if (expected_active != active_requests_) {
    throw SchedulerError("active request registry invariant failed");
  }
  if (kv_blocks_.total_allocated_blocks() >
      kv_blocks_.capacity_blocks()) {
    throw SchedulerError("KV block accounting exceeds capacity");
  }
  for (const RequestId request_id : running_) {
    if (kv_blocks_.allocated_blocks(request_id) == 0) {
      throw SchedulerError("running request owns no KV blocks");
    }
  }
  for (const auto& [request_id, remaining] :
       pending_terminal_release_iterations_) {
    if (remaining == 0 || !request(request_id).completed() ||
        std::find(
            running_.begin(), running_.end(), request_id) ==
            running_.end()) {
      throw SchedulerError(
          "terminal release registry invariant failed");
    }
  }
}

}  // namespace frontier::scheduler
