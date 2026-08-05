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

entities::TieredPrefixPlan build_contiguous_tiered_prefix_plan(
    std::uint64_t query_blocks, std::uint64_t gpu_frontier_blocks,
    std::uint64_t cpu_frontier_blocks, std::uint64_t block_size,
    std::uint64_t prompt_tokens) {
    if (block_size == 0 || query_blocks != prompt_tokens / block_size) {
        throw SchedulerError("invalid contiguous tiered prefix dimensions");
    }
    entities::TieredPrefixPlan plan{};
    plan.query_blocks = query_blocks;
    plan.gpu_hit_frontier_blocks =
        std::min(gpu_frontier_blocks, query_blocks);
    const std::uint64_t cpu_frontier =
        std::min(cpu_frontier_blocks, query_blocks);
    plan.cpu_query_blocks =
        query_blocks > plan.gpu_hit_frontier_blocks
            ? query_blocks - plan.gpu_hit_frontier_blocks
            : 0;
    plan.hit_frontier_blocks =
        std::max(plan.gpu_hit_frontier_blocks, cpu_frontier);
    if (prompt_tokens % block_size == 0 &&
        plan.hit_frontier_blocks == query_blocks && query_blocks > 0) {
        --plan.hit_frontier_blocks;
    }
    plan.cpu_begin_block =
        std::min(plan.gpu_hit_frontier_blocks, plan.hit_frontier_blocks);
    plan.cpu_end_block =
        std::min(cpu_frontier, plan.hit_frontier_blocks);
    if (plan.cpu_end_block < plan.cpu_begin_block) {
        plan.cpu_end_block = plan.cpu_begin_block;
    }
    plan.block_size = block_size;
    plan.prompt_tokens = prompt_tokens;
    return plan;
}

std::uint64_t revalidate_contiguous_tiered_prefix_frontier(
    std::uint64_t current_gpu_frontier_blocks,
    const entities::StagedCpuKVCacheRestore &staged) {
    if (staged.block_size == 0 ||
        staged.query_blocks != staged.prompt_tokens / staged.block_size ||
        staged.cpu_begin_block > staged.cpu_end_block) {
        throw SchedulerError("invalid staged CPU prefix dimensions");
    }
    std::uint64_t reusable =
        std::min(current_gpu_frontier_blocks, staged.query_blocks);
    if (reusable >= staged.cpu_begin_block) {
        reusable = std::max(reusable,
                            std::min(staged.cpu_end_block,
                                     staged.query_blocks));
    }
    if (staged.prompt_tokens % staged.block_size == 0 &&
        reusable == staged.query_blocks && reusable > 0) {
        --reusable;
    }
    return reusable;
}

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
    ClusterType cluster_type, config::PrefixCacheConfig prefix_cache_config,
    config::ResolvedCpuKVCacheTargetConfig cpu_kv_cache_config)
    : BaseReplicaScheduler(std::move(config), requests, replica, dp_id,
                           std::move(predictor), cluster_type,
                           prefix_cache_config),
      cpu_kv_cache_config_(cpu_kv_cache_config) {
    if (config_.type != config::SchedulerType::kVllmV1 ||
        config_.scheduling_policy != config::SchedulingPolicy::kFcfs) {
        throw SchedulerError(
            "the C++ port requires the FCFS vLLM V1 scheduler");
    }
    if (config_.batch_size_cap == 0 || config_.max_tokens_in_batch == 0 ||
        pipeline_parallel_size_ == 0) {
        throw SchedulerError("scheduler capacity values must be positive");
    }
    if (cpu_kv_cache_config_.enabled) {
        if (cluster_type != ClusterType::kPrefill ||
            !prefix_cache_config.enabled ||
            cpu_kv_cache_config_.capacity_blocks == 0 ||
            cpu_kv_cache_config_.bytes_per_block == 0) {
            throw SchedulerError(
                "CPU KV cache requires a configured PREFILL cache owner");
        }
        cpu_kv_cache_ = std::make_unique<kv_cache::CpuKVCacheManager>(
            cpu_kv_cache_config_.capacity_blocks,
            cpu_kv_cache_config_.capacity_pressure_policy);
        cpu_transfer_engine_ = std::make_unique<
            cpu_kv_cache_transfer::AnalyticalCpuKVCacheTransferEngine>(
            cpu_kv_cache_transfer::CpuTransferEngineConfig{
                cpu_kv_cache_config_.d2h_bandwidth_gbps,
                cpu_kv_cache_config_.d2h_latency_ms,
                cpu_kv_cache_config_.h2d_bandwidth_gbps,
                cpu_kv_cache_config_.h2d_latency_ms});
    }
}

bool VllmV1Scheduler::contains_request(RequestId request_id) const {
    return std::find(preempted_.begin(), preempted_.end(), request_id) !=
               preempted_.end() ||
           std::find(waiting_.begin(), waiting_.end(), request_id) !=
               waiting_.end() ||
           std::find(running_.begin(), running_.end(), request_id) !=
               running_.end() ||
           pending_cpu_restores_.find(request_id) !=
               pending_cpu_restores_.end() ||
           staged_cpu_restores_.find(request_id) != staged_cpu_restores_.end();
}

std::uint64_t VllmV1Scheduler::queued_kv_blocks() const noexcept {
    // Do not use the request's declared future decode length here.  Routing
    // can only observe the current scheduler frontier of queued requests.
    // This keeps placement stable for online arrivals whose eventual output
    // length is not known to the serving scheduler.
    const std::uint64_t block_size = kv_blocks_.block_size();
    if (block_size == 0 || requests_ == nullptr) {
        return std::numeric_limits<std::uint64_t>::max();
    }

    std::uint64_t total = 0;
    const auto add = [&](RequestId request_id) {
        if (!request_id.valid() || request_id.index() >= requests_->size()) {
            total = std::numeric_limits<std::uint64_t>::max();
            return;
        }
        try {
            const entities::Request &value = requests_->at(request_id.index());
            const std::uint64_t tokens = kv_accounted_tokens(value);
            std::uint64_t blocks = tokens / block_size;
            if (tokens % block_size != 0) {
                if (blocks == std::numeric_limits<std::uint64_t>::max()) {
                    total = std::numeric_limits<std::uint64_t>::max();
                    return;
                }
                ++blocks;
            }
            if (blocks > std::numeric_limits<std::uint64_t>::max() - total) {
                total = std::numeric_limits<std::uint64_t>::max();
            } else {
                total += blocks;
            }
        } catch (...) {
            // Scheduler state is validated at lifecycle boundaries.  If a
            // caller observes a malformed request while routing, treat it as
            // maximally loaded rather than allowing a noexcept telemetry
            // accessor to terminate the process.
            total = std::numeric_limits<std::uint64_t>::max();
        }
    };
    for (const RequestId request_id : preempted_) {
        add(request_id);
    }
    for (const RequestId request_id : waiting_) {
        add(request_id);
    }
    for (const auto &[request_id, unused] : pending_cpu_restores_) {
        static_cast<void>(unused);
        add(request_id);
    }
    return total;
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

entities::TieredPrefixPlan VllmV1Scheduler::build_tiered_prefix_plan(
    const entities::Request &value) const {
    const std::uint64_t block_size = kv_blocks_.block_size();
    const std::uint64_t prompt_tokens = value.num_prefill_tokens();
    const std::uint64_t query_blocks = prompt_tokens / block_size;
    const kv_cache::PrefixLookupResult gpu = kv_blocks_.lookup(value);
    const kv_cache::CpuPrefixLookupResult cpu =
        cpu_kv_cache_->lookup(value.session_id(), query_blocks);
    return build_contiguous_tiered_prefix_plan(
        query_blocks, gpu.hit_blocks, cpu.hit_blocks, block_size,
        prompt_tokens);
}

void VllmV1Scheduler::suspend_for_cpu_restore(
    RequestId request_id, const entities::TieredPrefixPlan &plan,
    SimTime time) {
    if (cpu_kv_cache_ == nullptr || cpu_transfer_engine_ == nullptr ||
        plan.cpu_begin_block >= plan.cpu_end_block ||
        pending_cpu_restores_.find(request_id) != pending_cpu_restores_.end() ||
        staged_cpu_restores_.find(request_id) != staged_cpu_restores_.end()) {
        throw SchedulerError("invalid CPU restore suspension");
    }
    entities::Request &value = request(request_id);
    const CpuRestoreLeaseId lease = cpu_kv_cache_->pin_restore(
        value.session_id(), plan.cpu_begin_block, plan.cpu_end_block, time);
    const std::uint64_t blocks = plan.cpu_end_block - plan.cpu_begin_block;
    if (blocks > std::numeric_limits<std::uint64_t>::max() /
                     cpu_kv_cache_config_.bytes_per_block) {
        static_cast<void>(cpu_kv_cache_->release_restore(lease, false, time));
        throw SchedulerError("CPU restore transfer size overflows uint64");
    }
    try {
        const auto timing = cpu_transfer_engine_->schedule(
            cpu_kv_cache_transfer::CpuTransferDirection::kH2D,
            blocks * cpu_kv_cache_config_.bytes_per_block, time);
        const CpuKvTransferId transfer_id{next_cpu_transfer_id_++};
        const Generation generation{value.runtime_epoch()};
        cpu_restore_operations_.emplace(
            transfer_id,
            entities::CpuKVCacheRestoreInfo{
                transfer_id, request_id, replica_id(), dp_id(), lease, plan,
                timing, generation});
        pending_cpu_restores_.emplace(request_id, transfer_id);
        pending_auxiliary_events_.push_back(ScheduledAuxiliaryEvent{
            timing.started_at,
            [&]() {
                CpuKVCacheRestoreStartPayload payload{};
                payload.transfer_id = transfer_id;
                payload.request_id = request_id;
                payload.replica_id = replica_id();
                payload.dp_id = dp_id();
                payload.generation = generation;
                payload.cluster_type = cluster_type();
                return EventPayload{payload};
            }()});
    } catch (...) {
        static_cast<void>(cpu_kv_cache_->release_restore(lease, false, time));
        throw;
    }
}

std::uint64_t VllmV1Scheduler::revalidate_staged_frontier(
    const entities::Request &value,
    const entities::StagedCpuKVCacheRestore &staged) const {
    return revalidate_contiguous_tiered_prefix_frontier(
        kv_blocks_.lookup(value).hit_blocks, staged);
}

std::vector<ScheduledAuxiliaryEvent>
VllmV1Scheduler::drain_auxiliary_events() {
    return std::exchange(pending_auxiliary_events_, {});
}

std::vector<entities::CpuKVCacheOffloadInfo>
VllmV1Scheduler::cpu_kv_cache_offload_operations() const {
    std::vector<entities::CpuKVCacheOffloadInfo> result;
    result.reserve(cpu_offload_operations_.size());
    for (const auto &[id, operation] : cpu_offload_operations_) {
        static_cast<void>(id);
        result.push_back(operation);
    }
    return result;
}

std::vector<entities::CpuKVCacheRestoreInfo>
VllmV1Scheduler::cpu_kv_cache_restore_operations() const {
    std::vector<entities::CpuKVCacheRestoreInfo> result;
    result.reserve(cpu_restore_operations_.size());
    for (const auto &[id, operation] : cpu_restore_operations_) {
        static_cast<void>(id);
        result.push_back(operation);
    }
    return result;
}

void VllmV1Scheduler::on_cpu_kv_cache_restore_start(
    CpuKvTransferId transfer_id, Generation generation, SimTime time) {
    auto operation = cpu_restore_operations_.find(transfer_id);
    if (operation == cpu_restore_operations_.end() ||
        operation->second.request_generation() != generation ||
        operation->second.state() !=
            entities::CpuKVCacheTransferState::kPending) {
        return;
    }
    operation->second.mark_started(time);
    const entities::CpuKVCacheRestoreInfo &restore = operation->second;
    pending_auxiliary_events_.push_back(ScheduledAuxiliaryEvent{
        restore.timing().completed_at,
        [&]() {
            CpuKVCacheRestoreEndPayload payload{};
            payload.transfer_id = transfer_id;
            payload.request_id = restore.request_id();
            payload.replica_id = replica_id();
            payload.dp_id = dp_id();
            payload.generation = generation;
            payload.cluster_type = cluster_type();
            return EventPayload{payload};
        }()});
}

bool VllmV1Scheduler::on_cpu_kv_cache_restore_end(
    CpuKvTransferId transfer_id, Generation generation, SimTime time) {
    auto operation = cpu_restore_operations_.find(transfer_id);
    if (operation == cpu_restore_operations_.end() ||
        operation->second.request_generation() != generation ||
        operation->second.state() !=
            entities::CpuKVCacheTransferState::kInFlight) {
        return false;
    }
    entities::CpuKVCacheRestoreInfo &restore = operation->second;
    const auto pending = pending_cpu_restores_.find(restore.request_id());
    if (pending == pending_cpu_restores_.end() ||
        pending->second != transfer_id) {
        return false;
    }
    entities::Request &value = request(restore.request_id());
    const entities::TieredPrefixPlan &plan = restore.plan();
    bool staged_published = false;
    bool runnable_published = false;
    try {
        if (!cpu_kv_cache_->release_restore(restore.lease_id(), true, time)) {
            throw SchedulerError("CPU restore lease was already terminal");
        }
        entities::StagedCpuKVCacheRestore staged{};
        staged.request_id = value.id();
        staged.session_id = value.session_id();
        staged.replica_id = replica_id();
        staged.dp_id = dp_id();
        staged.cpu_begin_block = plan.cpu_begin_block;
        staged.cpu_end_block = plan.cpu_end_block;
        staged.lookup_gpu_frontier_blocks = plan.gpu_hit_frontier_blocks;
        staged.lookup_cpu_frontier_blocks = plan.cpu_end_block;
        staged.query_blocks = plan.query_blocks;
        staged.block_size = plan.block_size;
        staged.prompt_tokens = plan.prompt_tokens;
        staged.timing = restore.timing();
        if (!staged_cpu_restores_.emplace(value.id(), std::move(staged)).second) {
            throw SchedulerError("CPU restore staged payload already exists");
        }
        staged_published = true;
        waiting_.push_back(value.id());
        runnable_published = true;
        restore.mark_completed(time);
        value.record_cpu_restore_transfer(
            plan.cpu_end_block - plan.cpu_begin_block,
            restore.timing().size_bytes, restore.timing().queue_time_ms,
            restore.timing().service_time_ms);
        pending_cpu_restores_.erase(pending);
    } catch (...) {
        if (runnable_published) {
            const auto runnable =
                std::find(waiting_.begin(), waiting_.end(), value.id());
            if (runnable != waiting_.end()) {
                waiting_.erase(runnable);
            }
        }
        if (staged_published) {
            staged_cpu_restores_.erase(value.id());
        }
        if (cpu_kv_cache_->lease_active(restore.lease_id())) {
            static_cast<void>(cpu_kv_cache_->release_restore(
                restore.lease_id(), false, time));
        }
        pending_cpu_restores_.erase(value.id());
        if (restore.state() !=
            entities::CpuKVCacheTransferState::kCompleted) {
            restore.cancel();
        }
        validate_policy_state();
        throw;
    }
    validate_policy_state();
    return true;
}

bool VllmV1Scheduler::cancel_cpu_kv_cache_restore(RequestId request_id,
                                                  SimTime time) {
    if (!time.valid()) {
        throw SchedulerError("CPU restore cancellation time is invalid");
    }
    const auto pending = pending_cpu_restores_.find(request_id);
    if (pending != pending_cpu_restores_.end()) {
        auto operation = cpu_restore_operations_.find(pending->second);
        if (operation == cpu_restore_operations_.end()) {
            throw SchedulerError("pending CPU restore operation disappeared");
        }
        entities::CpuKVCacheRestoreInfo &restore = operation->second;
        const CpuKvTransferId transfer_id = restore.transfer_id();
        restore.cancel();
        static_cast<void>(
            cpu_kv_cache_->release_restore(restore.lease_id(), false, time));
        pending_cpu_restores_.erase(pending);
        pending_auxiliary_events_.erase(
            std::remove_if(
                pending_auxiliary_events_.begin(),
                pending_auxiliary_events_.end(), [&](const auto &event) {
                    const auto *start = std::get_if<
                        CpuKVCacheRestoreStartPayload>(&event.payload);
                    return start != nullptr &&
                           start->transfer_id == transfer_id;
                }),
            pending_auxiliary_events_.end());
        waiting_.push_back(request_id);
        validate_policy_state();
        return true;
    }
    const auto staged = staged_cpu_restores_.find(request_id);
    if (staged != staged_cpu_restores_.end()) {
        staged_cpu_restores_.erase(staged);
        const auto runnable =
            std::find(waiting_.begin(), waiting_.end(), request_id);
        if (runnable == waiting_.end()) {
            throw SchedulerError("staged CPU restore runnable entry disappeared");
        }
        waiting_.erase(runnable);
        validate_policy_state();
        return true;
    }
    return false;
}

void VllmV1Scheduler::on_cpu_kv_cache_offload_start(
    CpuKvTransferId transfer_id, CpuOffloadGeneration generation,
    SimTime time) {
    auto operation = cpu_offload_operations_.find(transfer_id);
    if (operation == cpu_offload_operations_.end() ||
        operation->second.generation() != generation ||
        operation->second.state() !=
            entities::CpuKVCacheTransferState::kPending) {
        return;
    }
    operation->second.mark_started(time);
    const entities::CpuKVCacheOffloadInfo &offload = operation->second;
    pending_auxiliary_events_.push_back(ScheduledAuxiliaryEvent{
        offload.timing().completed_at,
        [&]() {
            CpuKVCacheOffloadEndPayload payload{};
            payload.transfer_id = transfer_id;
            payload.request_id = offload.request_id();
            payload.replica_id = replica_id();
            payload.dp_id = dp_id();
            payload.cpu_generation = generation;
            payload.cluster_type = cluster_type();
            return EventPayload{payload};
        }()});
}

bool VllmV1Scheduler::on_cpu_kv_cache_offload_end(
    CpuKvTransferId transfer_id, CpuOffloadGeneration generation,
    SimTime time) {
    auto operation = cpu_offload_operations_.find(transfer_id);
    if (operation == cpu_offload_operations_.end() ||
        operation->second.generation() != generation ||
        operation->second.state() !=
            entities::CpuKVCacheTransferState::kInFlight) {
        return false;
    }
    entities::CpuKVCacheOffloadInfo &offload = operation->second;
    const auto pending = pending_cpu_offloads_.find(offload.request_id());
    auto export_state = pending_exports_.find(offload.request_id());
    if (pending == pending_cpu_offloads_.end() ||
        pending->second != transfer_id ||
        export_state == pending_exports_.end() ||
        !export_state->second.cpu_offload_pending) {
        return false;
    }
    try {
        if (!cpu_kv_cache_->commit_offload(offload.reservation_id(), time)) {
            throw SchedulerError(
                "CPU offload reservation was already terminal");
        }
        offload.mark_completed(time);
    } catch (...) {
        if (offload.state() !=
            entities::CpuKVCacheTransferState::kCompleted) {
            offload.cancel();
        }
        if (cpu_kv_cache_->reservation_pending(offload.reservation_id())) {
            static_cast<void>(
                cpu_kv_cache_->abort_offload(offload.reservation_id()));
        }
        pending_cpu_offloads_.erase(pending);
        export_state->second.cpu_offload_pending = false;
        if (!export_state->second.decode_pending) {
            static_cast<void>(kv_blocks_.free(offload.request_id()));
            pending_exports_.erase(export_state);
        }
        validate_policy_state();
        throw;
    }
    pending_cpu_offloads_.erase(pending);
    export_state->second.cpu_offload_pending = false;
    if (!export_state->second.decode_pending) {
        static_cast<void>(kv_blocks_.free(offload.request_id()));
        pending_exports_.erase(export_state);
    }
    request(offload.request_id())
        .record_cpu_offload_transfer(offload.timing().size_bytes,
                                     offload.timing().queue_time_ms,
                                     offload.timing().service_time_ms);
    validate_policy_state();
    return true;
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
        value.waiting_count_before = checked_size(
            waiting_count(), "waiting queue size overflows uint64");
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
            const auto staged_position =
                staged_cpu_restores_.find(request_id);
            const bool has_staged_restore =
                staged_position != staged_cpu_restores_.end();
            if (has_staged_restore) {
                const std::uint64_t reusable = revalidate_staged_frontier(
                    value, staged_position->second);
                prefix_lookup.query_blocks =
                    staged_position->second.query_blocks;
                prefix_lookup.hit_blocks = reusable;
                prefix_lookup.cached_tokens =
                    reusable * kv_blocks_.block_size();
            } else if (cpu_kv_cache_ != nullptr &&
                       !value.is_prefill_complete()) {
                const entities::TieredPrefixPlan plan =
                    build_tiered_prefix_plan(value);
                if (plan.cpu_end_block > plan.cpu_begin_block) {
                    try {
                        suspend_for_cpu_restore(request_id, plan, time);
                    } catch (...) {
                        queue.insert(queue.end(), skipped.begin(),
                                     skipped.end());
                        restore_admission_queue(std::move(queue));
                        throw;
                    }
                    queue.pop_front();
                    continue;
                }
                prefix_lookup.query_blocks = plan.query_blocks;
                prefix_lookup.hit_blocks = plan.hit_frontier_blocks;
                prefix_lookup.cached_tokens =
                    plan.hit_frontier_blocks * kv_blocks_.block_size();
            } else if (kv_blocks_.prefix_cache_enabled() &&
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
                has_staged_restore
                    ? kv_blocks_.can_admit_tiered(
                          request_id, value.session_id(),
                          prefix_lookup.hit_blocks, num_tokens)
                    : kv_blocks_.prefix_cache_enabled() &&
                        !value.is_prefill_complete()
                    ? kv_blocks_.can_admit(request_id, value.session_id(),
                                           prefix_lookup.cached_tokens,
                                           num_tokens)
                    : kv_blocks_.can_reserve(request_id, accounted, num_tokens);
            if (!can_reserve) {
                break;
            }

            queue.pop_front();
            if (has_staged_restore) {
                const std::uint64_t current_gpu =
                    kv_blocks_.lookup(value).hit_blocks;
                kv_blocks_.admit_tiered(request_id, value.session_id(),
                                        prefix_lookup.hit_blocks, num_tokens);
                value.restore_prefix_cache_lookup(
                    prefix_lookup.query_blocks, prefix_lookup.hit_blocks,
                    prefix_lookup.cached_tokens);
                kv_blocks_.record_successful_admission(
                    prefix_lookup.query_blocks, prefix_lookup.hit_blocks);
                const entities::StagedCpuKVCacheRestore staged =
                    staged_position->second;
                const std::uint64_t cpu_query =
                    staged.query_blocks > current_gpu
                        ? staged.query_blocks - current_gpu
                        : 0;
                const std::uint64_t cpu_used =
                    prefix_lookup.hit_blocks > current_gpu
                        ? prefix_lookup.hit_blocks - current_gpu
                        : 0;
                const std::uint64_t gpu_used =
                    std::min(current_gpu, prefix_lookup.hit_blocks);
                value.record_cpu_prefix_admission(
                    gpu_used, cpu_query, cpu_used,
                    cpu_used * staged.block_size);
                cpu_kv_cache_->record_successful_lookup(
                    request_id,
                    kv_cache::CpuPrefixLookupResult{cpu_query, cpu_used},
                    value.session_id());
                staged_cpu_restores_.erase(staged_position);
            } else if (kv_blocks_.prefix_cache_enabled() &&
                !value.is_prefill_complete()) {
                const std::uint64_t current_gpu =
                    kv_blocks_.lookup(value).hit_blocks;
                kv_blocks_.admit(request_id, value.session_id(),
                                 prefix_lookup.cached_tokens, num_tokens);
                value.restore_prefix_cache_lookup(prefix_lookup.query_blocks,
                                                  prefix_lookup.hit_blocks,
                                                  prefix_lookup.cached_tokens);
                kv_blocks_.record_successful_admission(
                    prefix_lookup.query_blocks, prefix_lookup.hit_blocks);
                if (cpu_kv_cache_ != nullptr) {
                    const std::uint64_t cpu_query =
                        prefix_lookup.query_blocks > current_gpu
                            ? prefix_lookup.query_blocks - current_gpu
                            : 0;
                    const std::uint64_t gpu_used =
                        std::min(current_gpu, prefix_lookup.hit_blocks);
                    value.record_cpu_prefix_admission(gpu_used, cpu_query, 0,
                                                      0);
                    cpu_kv_cache_->record_successful_lookup(
                        request_id,
                        kv_cache::CpuPrefixLookupResult{cpu_query, 0},
                        value.session_id());
                }
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
    result.waiting_count_after = checked_size(
        waiting_count(), "waiting queue size overflows uint64");
    result.running_count_after =
        checked_size(running_.size(), "running queue size overflows uint64");
    const bool empty_iteration = result.scheduled_requests.empty();
    advance_terminal_release_boundary();
    if (empty_iteration) {
        result.available_blocks_after = kv_blocks_.available_blocks();
        result.waiting_count_after = checked_size(
            waiting_count(), "waiting queue size overflows uint64");
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
            if (!pending_exports_
                     .emplace(value.id(), PrefillExportState{true, false})
                     .second) {
                throw SchedulerError(
                    "request already has pending PREFILL exports");
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
    auto export_state = pending_exports_.find(request_id);
    if (export_state == pending_exports_.end() ||
        !export_state->second.decode_pending) {
        throw SchedulerError(
            "KV transfer completion has no pending source request");
    }
    if (kv_blocks_.allocated_blocks(request_id) == 0) {
        throw SchedulerError("pending KV transfer source owns no blocks");
    }
    export_state->second.decode_pending = false;
    const auto offload = pending_cpu_offloads_.find(request_id);
    if (offload != pending_cpu_offloads_.end()) {
        cpu_offload_operations_.at(offload->second)
            .set_decode_transfer_completed_at(
                request(request_id).kv_cache_transfer_end_time());
    }
    if (!export_state->second.cpu_offload_pending) {
        static_cast<void>(kv_blocks_.free(request_id));
        pending_exports_.erase(export_state);
    }
    validate_policy_state();
}

bool VllmV1Scheduler::prepare_cpu_kv_cache_offload(RequestId request_id,
                                                    SimTime time) {
    if (cpu_kv_cache_ == nullptr) {
        return false;
    }
    auto export_state = pending_exports_.find(request_id);
    entities::Request &value = request(request_id);
    if (export_state == pending_exports_.end() ||
        !export_state->second.decode_pending ||
        export_state->second.cpu_offload_pending ||
        value.state() != entities::RequestState::kTransferPending ||
        !value.session_id().valid()) {
        throw SchedulerError("invalid CPU offload preparation state");
    }
    CpuOffloadGeneration &last = cpu_offload_generations_[value.session_id()];
    if (last.valid() &&
        last.value() ==
            std::numeric_limits<CpuOffloadGeneration::ValueType>::max()) {
        throw SchedulerError("CPU offload generation exhausted");
    }
    const CpuOffloadGeneration::ValueType next_value =
        last.valid() ? last.value() + 1 : 0;
    const CpuOffloadGeneration generation{next_value};
    last = generation;
    const std::uint64_t desired =
        value.num_processed_prefill_tokens() / kv_blocks_.block_size();
    const auto reservation = cpu_kv_cache_->reserve_offload(
        value.session_id(), generation, desired, time);
    if (!reservation.requires_transfer()) {
        return false;
    }
    if (reservation.reserved_blocks >
        std::numeric_limits<std::uint64_t>::max() /
            cpu_kv_cache_config_.bytes_per_block) {
        static_cast<void>(
            cpu_kv_cache_->abort_offload(reservation.reservation_id));
        throw SchedulerError("CPU offload transfer size overflows uint64");
    }
    try {
        const auto timing = cpu_transfer_engine_->schedule(
            cpu_kv_cache_transfer::CpuTransferDirection::kD2H,
            reservation.reserved_blocks *
                cpu_kv_cache_config_.bytes_per_block,
            time);
        const CpuKvTransferId transfer_id{next_cpu_transfer_id_++};
        cpu_offload_operations_.emplace(
            transfer_id,
            entities::CpuKVCacheOffloadInfo{
                transfer_id, request_id, replica_id(), dp_id(),
                reservation.reservation_id, timing, desired, generation});
        pending_cpu_offloads_.emplace(request_id, transfer_id);
        export_state->second.cpu_offload_pending = true;
        pending_auxiliary_events_.push_back(ScheduledAuxiliaryEvent{
            timing.started_at,
            [&]() {
                CpuKVCacheOffloadStartPayload payload{};
                payload.transfer_id = transfer_id;
                payload.request_id = request_id;
                payload.replica_id = replica_id();
                payload.dp_id = dp_id();
                payload.cpu_generation = generation;
                payload.cluster_type = cluster_type();
                return EventPayload{payload};
            }()});
    } catch (...) {
        static_cast<void>(
            cpu_kv_cache_->abort_offload(reservation.reservation_id));
        export_state->second.cpu_offload_pending = false;
        pending_cpu_offloads_.erase(request_id);
        throw;
    }
    validate_policy_state();
    return true;
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
    for (const auto &[request_id, export_state] : pending_exports_) {
        const entities::RequestState state = request(request_id).state();
        const bool transfer_state =
            state == entities::RequestState::kTransferPending ||
            state == entities::RequestState::kTransferInFlight;
        const auto offload = pending_cpu_offloads_.find(request_id);
        if ((!export_state.decode_pending &&
             !export_state.cpu_offload_pending) ||
            (export_state.decode_pending && !transfer_state) ||
            (export_state.cpu_offload_pending !=
             (offload != pending_cpu_offloads_.end())) ||
            kv_blocks_.allocated_blocks(request_id) == 0 ||
            ids.find(request_id) != ids.end()) {
            throw SchedulerError(
                "pending PREFILL export registry invariant failed");
        }
        if (offload != pending_cpu_offloads_.end()) {
            const auto operation = cpu_offload_operations_.find(offload->second);
            if (operation == cpu_offload_operations_.end() ||
                operation->second.request_id() != request_id ||
                !cpu_kv_cache_->reservation_pending(
                    operation->second.reservation_id())) {
                throw SchedulerError("pending CPU offload invariant failed");
            }
        }
    }
    for (const auto &[request_id, transfer_id] : pending_cpu_restores_) {
        const auto operation = cpu_restore_operations_.find(transfer_id);
        if (request(request_id).state() != entities::RequestState::kWaiting ||
            ids.find(request_id) != ids.end() ||
            kv_blocks_.allocated_blocks(request_id) != 0 ||
            operation == cpu_restore_operations_.end() ||
            operation->second.request_id() != request_id ||
            !cpu_kv_cache_->lease_active(operation->second.lease_id())) {
            throw SchedulerError("pending CPU restore invariant failed");
        }
    }
    for (const auto &[request_id, staged] : staged_cpu_restores_) {
        static_cast<void>(staged);
        if (request(request_id).state() != entities::RequestState::kWaiting ||
            ids.find(request_id) == ids.end() ||
            kv_blocks_.allocated_blocks(request_id) != 0 ||
            pending_cpu_restores_.find(request_id) !=
                pending_cpu_restores_.end()) {
            throw SchedulerError("staged CPU restore invariant failed");
        }
    }
    if (cpu_kv_cache_ != nullptr) {
        cpu_kv_cache_->validate_invariants();
    }
}

} // namespace frontier::scheduler
