#include "frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>
#include <utility>

#include "frontier/execution_time_predictor/fixed_execution_time_predictor.h"

namespace frontier::scheduler {
namespace {

std::uint64_t checked_size(std::size_t value, const char *context) {
    if (value > std::numeric_limits<std::uint64_t>::max()) {
        throw SchedulerError(context);
    }
    return static_cast<std::uint64_t>(value);
}

const entities::Replica &default_replica() {
    static const entities::Replica replica{
        ReplicaId{0}, config::ParallelismConfig{}, config::ModelConfig{}};
    return replica;
}

} // namespace

VllmV1Scheduler::VllmV1Scheduler(config::SchedulerConfig config,
                                 std::vector<entities::Request> &requests)
    : VllmV1Scheduler(
          std::move(config), requests,
          std::make_unique<
              execution_time_predictor::FixedExecutionTimePredictor>([&]() {
              config::FixedExecutionModelConfig value{};
              value.batch_latency_ms = 0.0;
              value.stage_latencies_ms = {};
              return value;
          }())) {}

VllmV1Scheduler::VllmV1Scheduler(config::SchedulerConfig config,
                                 std::vector<entities::Request> &requests,
                                 config::PrefixCacheConfig prefix_cache_config)
    : VllmV1Scheduler(
          std::move(config), requests,
          std::make_shared<
              execution_time_predictor::FixedExecutionTimePredictor>([&]() {
              config::FixedExecutionModelConfig value{};
              value.batch_latency_ms = 0.0;
              value.stage_latencies_ms = {};
              return value;
          }()),
          default_replica(), DataParallelId{0}, ClusterType::kMonolithic,
          prefix_cache_config) {}

VllmV1Scheduler::VllmV1Scheduler(
    config::SchedulerConfig config, std::vector<entities::Request> &requests,
    std::unique_ptr<execution_time_predictor::BaseExecutionTimePredictor>
        predictor)
    : BaseReplicaScheduler(config, requests, default_replica(),
                           DataParallelId{0}, std::move(predictor),
                           ClusterType::kMonolithic) {
    if (config_.type != config::SchedulerType::kVllmV1 ||
        config_.scheduling_policy != config::SchedulingPolicy::kFcfs) {
        throw SchedulerError(
            "the C++ port requires the FCFS vLLM V1 scheduler");
    }
}

VllmV1Scheduler::VllmV1Scheduler(
    config::SchedulerConfig config, std::vector<entities::Request> &requests,
    execution_time_predictor::ExecutionTimePredictorPtr predictor,
    const entities::Replica &replica, DataParallelId dp_id,
    ClusterType cluster_type, config::PrefixCacheConfig prefix_cache_config)
    : BaseReplicaScheduler(std::move(config), requests, replica, dp_id,
                           std::move(predictor), cluster_type,
                           prefix_cache_config) {
    if (config_.type != config::SchedulerType::kVllmV1 ||
        config_.scheduling_policy != config::SchedulingPolicy::kFcfs) {
        throw SchedulerError(
            "the C++ port requires the FCFS vLLM V1 scheduler");
    }
    if (config_.batch_size_cap == 0 || config_.max_tokens_in_batch == 0 ||
        pipeline_parallel_size_ == 0) {
        throw SchedulerError("scheduler capacity values must be positive");
    }
}

bool VllmV1Scheduler::contains_request(RequestId request_id) const {
    return std::find(preempted_.begin(), preempted_.end(), request_id) !=
               preempted_.end() ||
           std::find(waiting_.begin(), waiting_.end(), request_id) !=
               waiting_.end() ||
           std::find(running_.begin(), running_.end(), request_id) !=
               running_.end();
}

std::uint64_t
VllmV1Scheduler::next_num_tokens(const entities::Request &value) const {
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
    if (cluster_type() == ClusterType::kPrefill) {
        throw SchedulerError(
            "PREFILL scheduler retained a completed prefill as runnable");
    }
    if (cluster_type() == ClusterType::kDecode) {
        return 1;
    }
    if (value.num_processed_tokens() < value.scheduler_num_computed_tokens()) {
        throw SchedulerError(
            "request-visible decode progress trails scheduler frontier");
    }
    return value.num_processed_tokens() - value.scheduler_num_computed_tokens();
}

std::uint64_t
VllmV1Scheduler::kv_accounted_tokens(const entities::Request &value) const {
    if (cluster_type() == ClusterType::kDecode) {
        return value.scheduler_num_computed_tokens();
    }
    if (!value.is_prefill_complete()) {
        return value.scheduler_num_computed_tokens();
    }
    const std::uint64_t boundary_adjusted =
        value.num_processed_tokens() > 0
            ? std::max(value.num_prefill_tokens(),
                       value.num_processed_tokens() - 1)
            : value.num_prefill_tokens();
    return std::max(value.scheduler_num_computed_tokens(), boundary_adjusted);
}

std::optional<RequestId>
VllmV1Scheduler::select_preemption_victim(RequestId requester) const {
    for (auto candidate = running_.rbegin(); candidate != running_.rend();
         ++candidate) {
        if (*candidate != requester && !request_is_active(*candidate) &&
            !request(*candidate).completed()) {
            return *candidate;
        }
    }
    return std::nullopt;
}

std::uint64_t
VllmV1Scheduler::extra_terminal_release_iterations() const noexcept {
    if (cluster_type() != ClusterType::kMonolithic ||
        pipeline_parallel_size_ <= 1) {
        return 0;
    }
    return pipeline_parallel_size_ / 2 - 1;
}

std::uint64_t
VllmV1Scheduler::iteration_start_release_threshold() const noexcept {
    if (pipeline_parallel_size_ <= 4) {
        return 1;
    }
    return std::max<std::uint64_t>(pipeline_parallel_size_ / 4, 1);
}

bool VllmV1Scheduler::has_visible_waiting_requests() const noexcept {
    return !preempted_.empty() || !waiting_.empty();
}

void VllmV1Scheduler::free_completed_request(RequestId request_id) {
    entities::Request &value = request(request_id);
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

void VllmV1Scheduler::materialize_terminal_releases_before_iteration() {
    if (pending_terminal_release_iterations_.empty() ||
        has_visible_waiting_requests()) {
        return;
    }
    const std::uint64_t threshold = iteration_start_release_threshold();
    std::vector<RequestId> ready;
    for (const auto &[request_id, remaining] :
         pending_terminal_release_iterations_) {
        if (remaining <= threshold) {
            ready.push_back(request_id);
        }
    }
    std::sort(ready.begin(), ready.end(), [](RequestId left, RequestId right) {
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
    const std::uint64_t threshold = iteration_start_release_threshold();
    std::vector<RequestId> ready;
    for (auto &[request_id, remaining] : pending_terminal_release_iterations_) {
        if (remaining <= threshold && has_visible_waiting_requests() &&
            waiting_sensitive_release_extensions_.find(request_id) ==
                waiting_sensitive_release_extensions_.end()) {
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
    std::sort(ready.begin(), ready.end(), [](RequestId left, RequestId right) {
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

void VllmV1Scheduler::preempt_request(RequestId victim, SimTime time,
                                      std::vector<RequestId> &newly_preempted,
                                      ScheduleResult &result) {
    entities::Request &value = request(victim);
    const auto running_position =
        std::find(running_.begin(), running_.end(), victim);
    if (running_position == running_.end()) {
        throw SchedulerError("preemption victim is not running");
    }
    running_.erase(running_position);
    static_cast<void>(kv_blocks_.free(victim));
    value.on_preempted(time, cluster_type());
    waiting_.push_front(victim);
    newly_preempted.push_back(victim);
    ++result.preempted_count;
    result.decisions.push_back([&]() {
        SchedulerDecision value{};
        value.type = SchedulerDecisionType::kPreempted;
        value.request_id = victim;
        value.num_tokens = 0;
        value.token_budget_after = result.token_budget_after;
        value.available_blocks_after = kv_blocks_.available_blocks();
        return value;
    }());
}

void VllmV1Scheduler::rollback_preempted_schedules(
    const std::vector<RequestId> &newly_preempted,
    std::vector<ScheduledRequest> &running_scheduled,
    std::uint64_t &token_budget) {
    if (newly_preempted.empty()) {
        return;
    }
    const std::unordered_set<RequestId, StrongIdHash<RequestId>> ids(
        newly_preempted.begin(), newly_preempted.end());
    std::vector<ScheduledRequest> kept;
    kept.reserve(running_scheduled.size());
    for (const ScheduledRequest &scheduled : running_scheduled) {
        if (ids.find(scheduled.request_id) != ids.end()) {
            if (token_budget > std::numeric_limits<std::uint64_t>::max() -
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
    RequestId requester, std::uint64_t scheduled_tokens, SimTime time,
    std::vector<RequestId> &newly_preempted,
    std::vector<ScheduledRequest> &running_scheduled,
    std::uint64_t &token_budget, ScheduleResult &result) {
    while (true) {
        entities::Request &value = request(requester);
        const std::uint64_t accounted = kv_accounted_tokens(value);
        if (kv_blocks_.can_reserve(requester, accounted, scheduled_tokens)) {
            kv_blocks_.reserve(requester, accounted, scheduled_tokens);
            return true;
        }
        if (!config_.enable_preemption) {
            return false;
        }

        const std::size_t preemption_count_before = newly_preempted.size();
        const std::optional<RequestId> victim =
            select_preemption_victim(requester);
        preempt_request(victim.value_or(requester), time, newly_preempted,
                        result);
        rollback_preempted_schedules(
            std::vector<RequestId>{
                newly_preempted.begin() +
                    static_cast<std::ptrdiff_t>(preemption_count_before),
                newly_preempted.end()},
            running_scheduled, token_budget);
        result.token_budget_after = token_budget;
        if (!victim.has_value()) {
            return false;
        }
    }
}

std::deque<RequestId> VllmV1Scheduler::take_admission_queue() {
    if (cluster_type() == ClusterType::kDecode) {
        // Python's unified DECODE scheduler has one waiting queue. Newly
        // preempted requests are inserted at its front and keep that order.
        return std::exchange(waiting_, {});
    }

    // MONOLITHIC and PREFILL retain Python's two-list priority:
    // preempted requests before newly arrived requests.
    std::deque<RequestId> queue = std::exchange(preempted_, {});
    queue.insert(queue.end(), waiting_.begin(), waiting_.end());
    waiting_.clear();
    return queue;
}

void VllmV1Scheduler::restore_admission_queue(std::deque<RequestId> queue) {
    for (const RequestId request_id : queue) {
        if (cluster_type() != ClusterType::kDecode &&
            request(request_id).preempted()) {
            preempted_.push_back(request_id);
        } else {
            waiting_.push_back(request_id);
        }
    }
}

ScheduleResult VllmV1Scheduler::schedule_requests(SimTime time) {
    materialize_terminal_releases_before_iteration();
    validate_policy_state();

    ScheduleResult result = [&]() {
        ScheduleResult value{};
        value.iteration_id = IterationId{next_iteration_id_++};
        value.simulation_time = time;
        value.token_budget_before = config_.max_tokens_in_batch;
        value.token_budget_after = config_.max_tokens_in_batch;
        value.available_blocks_before = kv_blocks_.available_blocks();
        value.available_blocks_after = kv_blocks_.available_blocks();
        value.waiting_count_before =
            checked_size(preempted_.size() + waiting_.size(),
                         "waiting queue size overflows uint64");
        value.waiting_count_after = 0;
        value.running_count_before = checked_size(
            running_.size(), "running queue size overflows uint64");
        value.running_count_after = 0;
        value.preempted_count = 0;
        value.decisions = {};
        value.scheduled_requests = {};
        return value;
    }();

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
        entities::Request &value = request(request_id);
        if (value.completed()) {
            ++running_index;
            continue;
        }
        if (request_is_active(request_id) &&
            (cluster_type() != ClusterType::kPrefill ||
             value.is_prefill_complete())) {
            ++running_index;
            continue;
        }
        std::uint64_t num_tokens = next_num_tokens(value);
        if (!value.is_prefill_complete() &&
            config_.long_prefill_token_threshold > 0) {
            num_tokens =
                std::min(num_tokens, config_.long_prefill_token_threshold);
        }
        num_tokens = std::min(num_tokens, token_budget);
        if (num_tokens == 0) {
            ++running_index;
            continue;
        }

        const std::size_t preempted_before = preempted_requests.size();
        const bool reserved = try_reserve_with_preemption(
            request_id, num_tokens, time, preempted_requests, running_scheduled,
            token_budget, result);
        if (!reserved) {
            break;
        }

        value.advance_scheduler_frontier(num_tokens);
        running_scheduled.push_back([&]() {
            ScheduledRequest value{};
            value.request_id = request_id;
            value.num_tokens = num_tokens;
            value.admitted_from_waiting = false;
            return value;
        }());
        token_budget -= num_tokens;
        result.token_budget_after = token_budget;
        result.decisions.push_back([&]() {
            SchedulerDecision value{};
            value.type = SchedulerDecisionType::kRunningScheduled;
            value.request_id = request_id;
            value.num_tokens = num_tokens;
            value.token_budget_after = token_budget;
            value.available_blocks_after = kv_blocks_.available_blocks();
            return value;
        }());
        ++running_index;

        if (preempted_requests.size() != preempted_before) {
            validate_policy_state();
        }
    }

    std::vector<ScheduledRequest> waiting_scheduled;
    if (preempted_requests.empty() &&
        pending_terminal_release_iterations_.empty()) {
        std::deque<RequestId> queue = take_admission_queue();
        std::deque<RequestId> skipped;

        while (!queue.empty() && token_budget > 0) {
            if (running_.size() >= config_.batch_size_cap) {
                break;
            }
            const RequestId request_id = queue.front();
            entities::Request &value = request(request_id);
            kv_cache::PrefixLookupResult prefix_lookup{};
            if (kv_blocks_.prefix_cache_enabled() &&
                !value.is_prefill_complete()) {
                prefix_lookup = kv_blocks_.lookup(value);
                if (prefix_lookup.cached_tokens == value.num_prefill_tokens() &&
                    prefix_lookup.hit_blocks > 0) {
                    --prefix_lookup.hit_blocks;
                    prefix_lookup.cached_tokens -= kv_blocks_.block_size();
                }
            }
            std::uint64_t num_tokens =
                value.is_prefill_complete()
                    ? next_num_tokens(value)
                    : value.num_prefill_tokens() - prefix_lookup.cached_tokens;
            if (!value.is_prefill_complete() &&
                config_.long_prefill_token_threshold > 0) {
                num_tokens =
                    std::min(num_tokens, config_.long_prefill_token_threshold);
            }

            if (!config_.enable_chunked_prefill &&
                !value.is_prefill_complete() && num_tokens > token_budget) {
                queue.pop_front();
                skipped.push_back(request_id);
                continue;
            }

            num_tokens = std::min(num_tokens, token_budget);
            if (num_tokens == 0) {
                queue.pop_front();
                continue;
            }
            const std::uint64_t accounted = prefix_lookup.cached_tokens > 0
                                                ? prefix_lookup.cached_tokens
                                                : kv_accounted_tokens(value);
            const bool can_reserve =
                kv_blocks_.prefix_cache_enabled() &&
                        !value.is_prefill_complete()
                    ? kv_blocks_.can_admit(request_id, value.session_id(),
                                           prefix_lookup.cached_tokens,
                                           num_tokens)
                    : kv_blocks_.can_reserve(request_id, accounted, num_tokens);
            if (!can_reserve) {
                break;
            }

            queue.pop_front();
            if (kv_blocks_.prefix_cache_enabled() &&
                !value.is_prefill_complete()) {
                kv_blocks_.admit(request_id, value.session_id(),
                                 prefix_lookup.cached_tokens, num_tokens);
                value.restore_prefix_cache_lookup(prefix_lookup.query_blocks,
                                                  prefix_lookup.hit_blocks,
                                                  prefix_lookup.cached_tokens);
                kv_blocks_.record_successful_admission(
                    prefix_lookup.query_blocks, prefix_lookup.hit_blocks);
            } else {
                kv_blocks_.reserve(request_id, accounted, num_tokens);
            }
            value.on_admitted(time);
            value.advance_scheduler_frontier(num_tokens);
            running_.push_back(request_id);
            waiting_scheduled.push_back([&]() {
                ScheduledRequest value{};
                value.request_id = request_id;
                value.num_tokens = num_tokens;
                value.admitted_from_waiting = true;
                return value;
            }());
            token_budget -= num_tokens;
            result.token_budget_after = token_budget;
            result.decisions.push_back([&]() {
                SchedulerDecision value{};
                value.type = SchedulerDecisionType::kAdmission;
                value.request_id = request_id;
                value.num_tokens = num_tokens;
                value.token_budget_after = token_budget;
                value.available_blocks_after = kv_blocks_.available_blocks();
                return value;
            }());
        }

        queue.insert(queue.end(), skipped.begin(), skipped.end());
        restore_admission_queue(std::move(queue));
    }

    result.scheduled_requests.reserve(waiting_scheduled.size() +
                                      running_scheduled.size());
    result.scheduled_requests.insert(result.scheduled_requests.end(),
                                     waiting_scheduled.begin(),
                                     waiting_scheduled.end());
    result.scheduled_requests.insert(result.scheduled_requests.end(),
                                     running_scheduled.begin(),
                                     running_scheduled.end());
    result.token_budget_after = token_budget;
    result.available_blocks_after = kv_blocks_.available_blocks();
    result.waiting_count_after =
        checked_size(preempted_.size() + waiting_.size(),
                     "waiting queue size overflows uint64");
    result.running_count_after =
        checked_size(running_.size(), "running queue size overflows uint64");
    const bool empty_iteration = result.scheduled_requests.empty();
    advance_terminal_release_boundary();
    if (empty_iteration) {
        result.available_blocks_after = kv_blocks_.available_blocks();
        result.waiting_count_after =
            checked_size(preempted_.size() + waiting_.size(),
                         "waiting queue size overflows uint64");
        result.running_count_after = checked_size(
            running_.size(), "running queue size overflows uint64");
    }
    validate_policy_state();
    return result;
}

bool VllmV1Scheduler::apply_batch_completion(entities::Batch &batch,
                                             SimTime time) {
    bool has_valid_request = false;
    for (const entities::RequestBatchSnapshot &snapshot : batch.requests()) {
        entities::Request &value = request(snapshot.request_id);
        if (snapshot.scheduler_frontier < snapshot.scheduled_tokens) {
            throw SchedulerError(
                "batch snapshot scheduler frontier underflows");
        }
        const std::uint64_t expected_processed_tokens =
            std::max(snapshot.processed_tokens,
                     snapshot.scheduler_frontier - snapshot.scheduled_tokens);
        const bool frontier_matches =
            cluster_type() == ClusterType::kPrefill
                ? value.num_processed_tokens() == expected_processed_tokens &&
                      value.scheduler_num_computed_tokens() >=
                          snapshot.scheduler_frontier
                : value.num_processed_tokens() == snapshot.processed_tokens &&
                      value.scheduler_num_computed_tokens() ==
                          snapshot.scheduler_frontier;
        if (value.completed() ||
            value.runtime_epoch() != snapshot.runtime_epoch ||
            value.execution_epoch() != snapshot.execution_epoch ||
            !frontier_matches) {
            continue;
        }
        value.on_batch_completion(time, snapshot.scheduled_tokens,
                                  cluster_type());
        kv_blocks_.mark_blocks_computed(value);
        has_valid_request = true;
        if (cluster_type() == ClusterType::kPrefill &&
            value.is_prefill_complete()) {
            const auto position =
                std::find(running_.begin(), running_.end(), value.id());
            if (position == running_.end()) {
                throw SchedulerError(
                    "completed prefill is missing from running order");
            }
            running_.erase(position);
            value.mark_prefill_transfer_pending();
            if (!pending_kv_transfers_.insert(value.id()).second) {
                throw SchedulerError(
                    "request already has a pending KV transfer");
            }
        } else if (value.completed()) {
            const std::uint64_t extra = extra_terminal_release_iterations();
            if (extra > 0) {
                auto [position, inserted] =
                    pending_terminal_release_iterations_.try_emplace(value.id(),
                                                                     extra);
                if (!inserted) {
                    position->second = std::max(position->second, extra);
                }
            } else {
                free_completed_request(value.id());
            }
        }
    }
    return has_valid_request;
}

void VllmV1Scheduler::complete_kv_transfer(RequestId request_id) {
    if (cluster_type() != ClusterType::kPrefill) {
        throw SchedulerError(
            "KV transfer completion must target a PREFILL scheduler");
    }
    if (!pending_kv_transfers_.erase(request_id)) {
        throw SchedulerError(
            "KV transfer completion has no pending source request");
    }
    if (kv_blocks_.allocated_blocks(request_id) == 0) {
        throw SchedulerError("pending KV transfer source owns no blocks");
    }
    static_cast<void>(kv_blocks_.free(request_id));
    validate_policy_state();
}

bool VllmV1Scheduler::consume_terminal_release_followup_poll() noexcept {
    const bool pending = terminal_release_followup_poll_pending_;
    terminal_release_followup_poll_pending_ = false;
    return pending;
}

void VllmV1Scheduler::validate_policy_state() const {
    if (!runtime_validation_enabled()) {
        return;
    }
    if (cluster_type() == ClusterType::kDecode && !preempted_.empty()) {
        throw SchedulerError(
            "DECODE scheduler must retain preempted requests in waiting order");
    }
    std::unordered_set<RequestId, StrongIdHash<RequestId>> ids;
    for (const RequestId request_id : preempted_) {
        if (!ids.insert(request_id).second) {
            throw SchedulerError(
                "request appears in multiple scheduler queues");
        }
        const entities::Request &value = request(request_id);
        if (value.state() != entities::RequestState::kWaiting ||
            !value.preempted()) {
            throw SchedulerError(
                "preempted queue contains a non-preempted waiting request");
        }
    }
    for (const RequestId request_id : waiting_) {
        if (!ids.insert(request_id).second) {
            throw SchedulerError(
                "request appears in multiple scheduler queues");
        }
        if (request(request_id).state() != entities::RequestState::kWaiting) {
            throw SchedulerError(
                "waiting queue contains a non-waiting request");
        }
    }
    for (const RequestId request_id : running_) {
        if (!ids.insert(request_id).second) {
            throw SchedulerError(
                "request appears in multiple scheduler queues");
        }
        const entities::Request &value = request(request_id);
        const bool pending_completed =
            value.completed() &&
            pending_terminal_release_iterations_.find(request_id) !=
                pending_terminal_release_iterations_.end();
        if (value.state() != entities::RequestState::kRunning &&
            !pending_completed) {
            throw SchedulerError(
                "running order contains a non-running request");
        }
    }
    if (running_.size() > config_.batch_size_cap) {
        throw SchedulerError("running request count exceeds batch_size_cap");
    }
    if (kv_blocks_.total_allocated_blocks() > kv_blocks_.capacity_blocks()) {
        throw SchedulerError("KV block accounting exceeds capacity");
    }
    for (const RequestId request_id : running_) {
        if (kv_blocks_.allocated_blocks(request_id) == 0) {
            throw SchedulerError("running request owns no KV blocks");
        }
    }
    for (const auto &[request_id, remaining] :
         pending_terminal_release_iterations_) {
        if (remaining == 0 || !request(request_id).completed() ||
            std::find(running_.begin(), running_.end(), request_id) ==
                running_.end()) {
            throw SchedulerError("terminal release registry invariant failed");
        }
    }
    for (const RequestId request_id : pending_kv_transfers_) {
        const entities::RequestState state = request(request_id).state();
        if ((state != entities::RequestState::kTransferPending &&
             state != entities::RequestState::kTransferInFlight) ||
            kv_blocks_.allocated_blocks(request_id) == 0 ||
            ids.find(request_id) != ids.end()) {
            throw SchedulerError(
                "pending KV transfer registry invariant failed");
        }
    }
}

} // namespace frontier::scheduler
