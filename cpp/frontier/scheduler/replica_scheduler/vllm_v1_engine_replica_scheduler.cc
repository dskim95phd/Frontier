#include "frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>
#include <utility>

#include "frontier/execution_time_predictor/fixed_execution_time_predictor.h"

namespace frontier::scheduler {
namespace {

const entities::Replica &default_replica() {
    static const entities::Replica replica{
        ReplicaId{0}, config::ParallelismConfig{}, config::ModelConfig{}};
    return replica;
}

} // namespace

VllmV1Scheduler::VllmV1Scheduler(config::SchedulerConfig config,
                                 entities::RequestCollection &requests)
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
                                 entities::RequestCollection &requests,
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
    config::SchedulerConfig config, entities::RequestCollection &requests,
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
    config::SchedulerConfig config, entities::RequestCollection &requests,
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
    // Sequential PDD PREFILL must reserve a request's complete prompt before
    // admitting its first chunk.  MONOLITHIC/DECODE retain the legacy
    // chunk-local accounting path.
    kv_blocks_.enable_full_sequence_commitments(cluster_type ==
                                                ClusterType::kPrefill);
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
        cpu_kv_cache_->configure_kda_snapshot(
            cpu_kv_cache_config_.kda_snapshot_blocks,
            cpu_kv_cache_config_.kda_snapshot_bytes);
        if (cpu_kv_cache_config_.kda_snapshot_blocks != 0 &&
            !kv_blocks_.kda_snapshot_enabled()) {
            throw SchedulerError(
                "CPU KDA snapshots require a configured GPU snapshot cache");
        }
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
           std::find(restored_ready_.begin(), restored_ready_.end(),
                     request_id) != restored_ready_.end() ||
           std::find(waiting_.begin(), waiting_.end(), request_id) !=
               waiting_.end() ||
           std::find(running_.begin(), running_.end(), request_id) !=
               running_.end() ||
           pending_cpu_restores_.find(request_id) !=
               pending_cpu_restores_.end() ||
           staged_cpu_restores_.find(request_id) !=
               staged_cpu_restores_.end() ||
           pending_exports_.find(request_id) != pending_exports_.end();
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
        if (!requests_->contains(request_id)) {
            total = std::numeric_limits<std::uint64_t>::max();
            return;
        }
        try {
            const entities::Request &value = requests_->at(request_id.index());
            // A pending/staged PDD restore already owns a full-sequence
            // commitment. Its physical and future suffix blocks are exposed
            // through the target load, so counting the request here would
            // double-count it. Ordinary queued PREFILL requests have no
            // commitment yet; account their materialized full ISL so routing
            // can see the pressure they will impose on admission.
            const bool has_commitment =
                kv_blocks_.request_committed_blocks(request_id) > 0;
            const bool queued_prefill =
                cluster_type() == ClusterType::kPrefill &&
                !value.is_prefill_complete() && !has_commitment;
            const std::uint64_t tokens = queued_prefill
                                             ? value.num_prefill_tokens()
                                             : kv_accounted_tokens(value);
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
    for (const RequestId request_id : restored_ready_) {
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
    return !preempted_.empty() || !restored_ready_.empty() || !waiting_.empty();
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
    if (kv_blocks_.request_virtual_committed_blocks(request_id) != 0) {
        throw SchedulerError("pending KV transfer source still has an "
                             "unmaterialized commitment");
    }
    export_state->second.decode_pending = false;
    const auto offload = pending_cpu_offloads_.find(request_id);
    if (offload != pending_cpu_offloads_.end()) {
        cpu_offload_operations_.at(offload->second)
            .set_decode_transfer_completed_at(
                request(request_id).kv_cache_transfer_end_time());
    }
    if (!export_state->second.cpu_offload_pending) {
        release_request_kv(request_id);
        pending_exports_.erase(export_state);
    }
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
        if (staged_cpu_restores_.find(request_id) !=
            staged_cpu_restores_.end()) {
            throw SchedulerError(
                "staged CPU restore appears in ordinary waiting queue");
        }
        if (request(request_id).state() != entities::RequestState::kWaiting) {
            throw SchedulerError(
                "waiting queue contains a non-waiting request");
        }
    }
    for (const RequestId request_id : restored_ready_) {
        if (!ids.insert(request_id).second) {
            throw SchedulerError(
                "request appears in multiple scheduler queues");
        }
        const entities::Request &value = request(request_id);
        if (value.state() != entities::RequestState::kWaiting ||
            staged_cpu_restores_.find(request_id) ==
                staged_cpu_restores_.end() ||
            pending_cpu_restores_.find(request_id) !=
                pending_cpu_restores_.end() ||
            kv_blocks_.request_committed_blocks(request_id) == 0) {
            throw SchedulerError(
                "restored-ready queue contains a non-runnable request");
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
            kv_blocks_.request_virtual_committed_blocks(request_id) != 0 ||
            ids.find(request_id) != ids.end()) {
            throw SchedulerError(
                "pending PREFILL export registry invariant failed");
        }
        if (offload != pending_cpu_offloads_.end()) {
            const auto operation =
                cpu_offload_operations_.find(offload->second);
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
            kv_blocks_.request_committed_blocks(request_id) == 0 ||
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
            std::find(restored_ready_.begin(), restored_ready_.end(),
                      request_id) == restored_ready_.end() ||
            kv_blocks_.request_committed_blocks(request_id) == 0 ||
            pending_cpu_restores_.find(request_id) !=
                pending_cpu_restores_.end()) {
            throw SchedulerError("staged CPU restore invariant failed");
        }
    }
    for (const RequestId request_id : retired_session_requests_) {
        if (!contains_request(request_id)) {
            throw SchedulerError(
                "retired session request escaped scheduler ownership");
        }
    }
    if (cpu_kv_cache_ != nullptr) {
        cpu_kv_cache_->validate_invariants();
    }
}

} // namespace frontier::scheduler
