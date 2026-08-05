#pragma once

#include <cstdint>
#include <deque>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "frontier/config/config.h"
#include "frontier/core/event.h"
#include "frontier/core/ids.h"
#include "frontier/entities/batch.h"
#include "frontier/entities/cpu_kv_cache_transfer_info.h"
#include "frontier/entities/replica.h"
#include "frontier/entities/request.h"
#include "frontier/execution_time_predictor/base_execution_time_predictor.h"
#include "frontier/kv_cache/replica_kv_cache_manager.h"
#include "frontier/kv_cache/cpu_kv_cache_manager.h"
#include "frontier/scheduler/replica_stage_scheduler/replica_stage_scheduler.h"

namespace frontier::scheduler {

enum class SchedulerDecisionType : std::uint8_t {
    kRunningScheduled,
    kAdmission,
    kPreempted,
};

[[nodiscard]] std::string_view
to_string(SchedulerDecisionType decision) noexcept;

struct SchedulerDecision {
    SchedulerDecisionType type;
    RequestId request_id;
    std::uint64_t num_tokens;
    std::uint64_t token_budget_after;
    std::uint64_t available_blocks_after;
};

struct ScheduledRequest {
    RequestId request_id;
    std::uint64_t num_tokens;
    bool admitted_from_waiting;
};

struct ScheduleResult {
    IterationId iteration_id;
    SimTime simulation_time;
    std::uint64_t token_budget_before;
    std::uint64_t token_budget_after;
    std::uint64_t available_blocks_before;
    std::uint64_t available_blocks_after;
    std::uint64_t waiting_count_before;
    std::uint64_t waiting_count_after;
    std::uint64_t running_count_before;
    std::uint64_t running_count_after;
    std::uint64_t preempted_count;
    std::vector<SchedulerDecision> decisions;
    std::vector<ScheduledRequest> scheduled_requests;
};

struct ScheduledAuxiliaryEvent {
    SimTime time;
    EventPayload payload;
};

class SchedulerError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

class BaseReplicaScheduler {
  public:
    BaseReplicaScheduler(
        config::SchedulerConfig config,
        std::vector<entities::Request> &requests,
        const entities::Replica &replica, DataParallelId dp_id,
        execution_time_predictor::ExecutionTimePredictorPtr predictor,
        ClusterType cluster_type = ClusterType::kMonolithic,
        config::PrefixCacheConfig prefix_cache_config = {});
    virtual ~BaseReplicaScheduler() = default;

    BaseReplicaScheduler(const BaseReplicaScheduler &) = delete;
    BaseReplicaScheduler &operator=(const BaseReplicaScheduler &) = delete;
    BaseReplicaScheduler(BaseReplicaScheduler &&) = delete;
    BaseReplicaScheduler &operator=(BaseReplicaScheduler &&) = delete;

    void add_request(RequestId request_id);
    [[nodiscard]] ScheduleResult schedule(SimTime time);
    void mark_batch_started(const entities::Batch &batch);
    [[nodiscard]] bool on_batch_completed(entities::Batch &batch, SimTime time);
    [[nodiscard]] virtual bool
    consume_terminal_release_followup_poll() noexcept {
        return false;
    }
    virtual void complete_kv_transfer(RequestId request_id) {
        static_cast<void>(request_id);
        throw SchedulerError(
            "replica scheduler has no pending KV transfer support");
    }
    virtual bool prepare_cpu_kv_cache_offload(RequestId request_id,
                                               SimTime time) {
        static_cast<void>(request_id);
        static_cast<void>(time);
        return false;
    }
    [[nodiscard]] virtual std::size_t
    pending_kv_transfer_count() const noexcept {
        return 0;
    }
    [[nodiscard]] virtual std::vector<ScheduledAuxiliaryEvent>
    drain_auxiliary_events() {
        return {};
    }
    virtual void on_cpu_kv_cache_offload_start(CpuKvTransferId transfer_id,
                                                CpuOffloadGeneration generation,
                                                SimTime time);
    virtual bool on_cpu_kv_cache_offload_end(CpuKvTransferId transfer_id,
                                              CpuOffloadGeneration generation,
                                              SimTime time);
    virtual void on_cpu_kv_cache_restore_start(CpuKvTransferId transfer_id,
                                                Generation generation,
                                                SimTime time);
    virtual bool on_cpu_kv_cache_restore_end(CpuKvTransferId transfer_id,
                                              Generation generation,
                                              SimTime time);

    [[nodiscard]] bool has_in_flight_batch() const noexcept {
        return in_flight_batch_count_ > 0;
    }
    [[nodiscard]] std::uint64_t in_flight_batch_count() const noexcept {
        return in_flight_batch_count_;
    }
    [[nodiscard]] std::uint64_t pipeline_parallel_size() const noexcept {
        return pipeline_parallel_size_;
    }
    void set_runtime_validation_enabled(bool enabled) noexcept {
        runtime_validation_enabled_ = enabled;
    }
    [[nodiscard]] virtual bool has_pending_work() const noexcept = 0;
    [[nodiscard]] virtual bool idle() const noexcept = 0;
    [[nodiscard]] virtual std::size_t waiting_count() const noexcept = 0;
    [[nodiscard]] virtual std::size_t running_count() const noexcept = 0;
    [[nodiscard]] virtual std::uint64_t
    allocated_kv_blocks() const noexcept = 0;
    [[nodiscard]] virtual std::uint64_t
    available_kv_blocks() const noexcept {
        return 0;
    }
    [[nodiscard]] virtual std::uint64_t kv_block_size() const noexcept {
        return 0;
    }
    // KV footprint of requests waiting for admission (including
    // preempted/restore-pending requests).  Cluster routing policies may use
    // this observable value to avoid placing work behind a long sequence.
    // The default keeps third-party schedulers source-compatible; the C++
    // vLLM V1 implementation overrides it with block-accurate accounting.
    [[nodiscard]] virtual std::uint64_t queued_kv_blocks() const noexcept {
        return 0;
    }
    [[nodiscard]] kv_cache::PrefixLookupResult
    gpu_prefix_cache_lookup(const entities::Request &request) const {
        return kv_blocks_.lookup(request);
    }
    [[nodiscard]] std::uint64_t
    discard_gpu_prefix_cache_session(SessionId session_id) {
        return kv_blocks_.discard_session(session_id);
    }
    [[nodiscard]] std::uint64_t
    gpu_cache_valid_prefix_blocks(SessionId session_id) const noexcept {
        return kv_blocks_.gpu_cache_valid_prefix_blocks(session_id);
    }
    [[nodiscard]] virtual const kv_cache::PrefixCacheStats &
    prefix_cache_stats() const noexcept = 0;
    [[nodiscard]] virtual kv_cache::PrefixCacheDiagnostics
    prefix_cache_diagnostics() const = 0;
    [[nodiscard]] virtual const kv_cache::CpuKVCacheManager *
    cpu_kv_cache_manager() const noexcept {
        return nullptr;
    }
    [[nodiscard]] virtual const config::ResolvedCpuKVCacheTargetConfig *
    cpu_kv_cache_target_config() const noexcept {
        return nullptr;
    }
    [[nodiscard]] virtual std::size_t
    pending_cpu_restore_count() const noexcept {
        return 0;
    }
    [[nodiscard]] virtual std::size_t
    staged_cpu_restore_count() const noexcept {
        return 0;
    }
    [[nodiscard]] virtual std::vector<entities::CpuKVCacheOffloadInfo>
    cpu_kv_cache_offload_operations() const {
        return {};
    }
    [[nodiscard]] virtual std::vector<entities::CpuKVCacheRestoreInfo>
    cpu_kv_cache_restore_operations() const {
        return {};
    }

    [[nodiscard]] ReplicaId replica_id() const noexcept {
        return replica_->id();
    }
    [[nodiscard]] const entities::Replica &replica_entity() const noexcept {
        return *replica_;
    }
    [[nodiscard]] DataParallelId dp_id() const noexcept { return dp_id_; }
    [[nodiscard]] ClusterType cluster_type() const noexcept {
        return cluster_type_;
    }
    [[nodiscard]] ReplicaStageScheduler &
    get_replica_stage_scheduler(StageId stage_id);
    [[nodiscard]] const ReplicaStageScheduler &
    get_replica_stage_scheduler(StageId stage_id) const;

  protected:
    [[nodiscard]] entities::Request &request(RequestId request_id);
    [[nodiscard]] const entities::Request &request(RequestId request_id) const;
    [[nodiscard]] bool request_is_active(RequestId request_id) const;
    [[nodiscard]] bool runtime_validation_enabled() const noexcept {
        return runtime_validation_enabled_;
    }

    [[nodiscard]] virtual bool contains_request(RequestId request_id) const = 0;
    [[nodiscard]] virtual ScheduleResult schedule_requests(SimTime time) = 0;
    [[nodiscard]] virtual bool apply_batch_completion(entities::Batch &batch,
                                                      SimTime time) = 0;
    virtual void validate_policy_state() const = 0;

    config::SchedulerConfig config_;
    std::vector<entities::Request> *requests_;
    kv_cache::ReplicaKVCacheManager kv_blocks_;
    std::deque<RequestId> waiting_;
    std::uint64_t pipeline_parallel_size_;
    std::uint64_t in_flight_batch_count_ = 0;
    std::unordered_map<BatchId, std::vector<RequestId>, StrongIdHash<BatchId>>
        in_flight_batches_;
    std::unordered_map<RequestId, std::uint64_t, StrongIdHash<RequestId>>
        active_requests_;
    bool runtime_validation_enabled_ = true;

  private:
    void validate_lifecycle_state() const;

    const entities::Replica *replica_;
    DataParallelId dp_id_;
    ClusterType cluster_type_;
    std::vector<ReplicaStageScheduler> stage_schedulers_;
};

} // namespace frontier::scheduler
