#pragma once

#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "frontier/config/config.h"
#include "frontier/core/event.h"
#include "frontier/core/ids.h"
#include "frontier/entities/batch.h"
#include "frontier/entities/request.h"
#include "frontier/entities/cpu_kv_cache_transfer_info.h"
#include "frontier/execution_time_predictor/base_execution_time_predictor.h"
#include "frontier/cpu_kv_cache_transfer/analytical_transfer.h"
#include "frontier/kv_cache/cpu_kv_cache_manager.h"
#include "frontier/scheduler/kv_block_accounting.h"
#include "frontier/scheduler/replica_scheduler/base_replica_scheduler.h"

namespace frontier::scheduler {

[[nodiscard]] entities::TieredPrefixPlan build_contiguous_tiered_prefix_plan(
    std::uint64_t query_blocks, std::uint64_t gpu_frontier_blocks,
    std::uint64_t cpu_frontier_blocks, std::uint64_t block_size,
    std::uint64_t prompt_tokens);

[[nodiscard]] std::uint64_t revalidate_contiguous_tiered_prefix_frontier(
    std::uint64_t current_gpu_frontier_blocks,
    const entities::StagedCpuKVCacheRestore &staged);

class VllmV1Scheduler final : public BaseReplicaScheduler {
  public:
    VllmV1Scheduler(config::SchedulerConfig config,
                    std::vector<entities::Request> &requests);
    VllmV1Scheduler(config::SchedulerConfig config,
                    std::vector<entities::Request> &requests,
                    config::PrefixCacheConfig prefix_cache_config);
    VllmV1Scheduler(
        config::SchedulerConfig config,
        std::vector<entities::Request> &requests,
        std::unique_ptr<execution_time_predictor::BaseExecutionTimePredictor>
            predictor);
    VllmV1Scheduler(
        config::SchedulerConfig config,
        std::vector<entities::Request> &requests,
        execution_time_predictor::ExecutionTimePredictorPtr predictor,
        const entities::Replica &replica, DataParallelId dp_id,
        ClusterType cluster_type,
        config::PrefixCacheConfig prefix_cache_config = {},
        config::ResolvedCpuKVCacheTargetConfig cpu_kv_cache_config = {});

    [[nodiscard]] bool
    consume_terminal_release_followup_poll() noexcept override;
    void complete_kv_transfer(RequestId request_id) override;
    bool prepare_cpu_kv_cache_offload(RequestId request_id,
                                      SimTime time) override;
    [[nodiscard]] std::size_t
    pending_kv_transfer_count() const noexcept override {
        std::size_t count = 0;
        for (const auto &[request_id, state] : pending_exports_) {
            static_cast<void>(request_id);
            count += static_cast<std::size_t>(state.decode_pending);
        }
        return count;
    }
    [[nodiscard]] std::vector<ScheduledAuxiliaryEvent>
    drain_auxiliary_events() override;
    void on_cpu_kv_cache_offload_start(CpuKvTransferId transfer_id,
                                       CpuOffloadGeneration generation,
                                       SimTime time) override;
    bool on_cpu_kv_cache_offload_end(CpuKvTransferId transfer_id,
                                     CpuOffloadGeneration generation,
                                     SimTime time) override;
    void on_cpu_kv_cache_restore_start(CpuKvTransferId transfer_id,
                                       Generation generation,
                                       SimTime time) override;
    bool on_cpu_kv_cache_restore_end(CpuKvTransferId transfer_id,
                                     Generation generation,
                                     SimTime time) override;

    [[nodiscard]] bool has_pending_work() const noexcept override {
        return !preempted_.empty() || !waiting_.empty() || !running_.empty() ||
               !pending_cpu_restores_.empty() || !staged_cpu_restores_.empty() ||
               !pending_exports_.empty();
    }
    [[nodiscard]] bool idle() const noexcept override {
        return !has_pending_work() && in_flight_batch_count_ == 0 &&
               pending_exports_.empty() && kv_blocks_.empty();
    }
    [[nodiscard]] std::size_t waiting_count() const noexcept override {
        return preempted_.size() + waiting_.size() +
               pending_cpu_restores_.size();
    }
    [[nodiscard]] std::size_t running_count() const noexcept override {
        return running_.size();
    }
    [[nodiscard]] std::uint64_t allocated_kv_blocks() const noexcept override {
        return kv_blocks_.total_allocated_blocks();
    }
    [[nodiscard]] std::uint64_t available_kv_blocks() const noexcept override {
        return kv_blocks_.available_blocks();
    }
    [[nodiscard]] std::uint64_t kv_block_size() const noexcept override {
        return kv_blocks_.block_size();
    }
    [[nodiscard]] std::uint64_t queued_kv_blocks() const noexcept override;
    [[nodiscard]] const std::deque<RequestId> &waiting_queue() const noexcept {
        return waiting_;
    }
    [[nodiscard]] const std::deque<RequestId> &
    preempted_queue() const noexcept {
        return preempted_;
    }
    [[nodiscard]] const std::vector<RequestId> &running_order() const noexcept {
        return running_;
    }
    [[nodiscard]] const kv_cache::ReplicaKVCacheManager &
    kv_blocks() const noexcept {
        return kv_blocks_;
    }
    [[nodiscard]] const kv_cache::PrefixCacheStats &
    prefix_cache_stats() const noexcept override {
        return kv_blocks_.stats();
    }
    [[nodiscard]] kv_cache::PrefixCacheDiagnostics
    prefix_cache_diagnostics() const override {
        return kv_blocks_.diagnostics();
    }
    [[nodiscard]] kv_cache::CpuKVCacheManager *
    cpu_kv_cache_manager() noexcept {
        return cpu_kv_cache_.get();
    }
    [[nodiscard]] const kv_cache::CpuKVCacheManager *
    cpu_kv_cache_manager() const noexcept override {
        return cpu_kv_cache_.get();
    }
    [[nodiscard]] const config::ResolvedCpuKVCacheTargetConfig *
    cpu_kv_cache_target_config() const noexcept override {
        return cpu_kv_cache_ == nullptr ? nullptr : &cpu_kv_cache_config_;
    }
    [[nodiscard]] std::size_t pending_cpu_restore_count() const noexcept override {
        return pending_cpu_restores_.size();
    }
    [[nodiscard]] std::size_t staged_cpu_restore_count() const noexcept override {
        return staged_cpu_restores_.size();
    }
    [[nodiscard]] std::vector<entities::CpuKVCacheOffloadInfo>
    cpu_kv_cache_offload_operations() const override;
    [[nodiscard]] std::vector<entities::CpuKVCacheRestoreInfo>
    cpu_kv_cache_restore_operations() const override;
    [[nodiscard]] bool cancel_cpu_kv_cache_restore(RequestId request_id,
                                                    SimTime time);

  private:
    [[nodiscard]] bool contains_request(RequestId request_id) const override;
    [[nodiscard]] ScheduleResult schedule_requests(SimTime time) override;
    [[nodiscard]] bool apply_batch_completion(entities::Batch &batch,
                                              SimTime time) override;
    void validate_policy_state() const override;
    [[nodiscard]] std::uint64_t
    next_num_tokens(const entities::Request &request) const;
    [[nodiscard]] std::uint64_t
    kv_accounted_tokens(const entities::Request &request) const;
    [[nodiscard]] std::optional<RequestId>
    select_preemption_victim(RequestId requester) const;
    [[nodiscard]] std::uint64_t
    extra_terminal_release_iterations() const noexcept;
    [[nodiscard]] std::uint64_t
    iteration_start_release_threshold() const noexcept;
    [[nodiscard]] bool has_visible_waiting_requests() const noexcept;
    void free_completed_request(RequestId request_id);
    void materialize_terminal_releases_before_iteration();
    void advance_terminal_release_boundary();
    void preempt_request(RequestId victim, SimTime time,
                         std::vector<RequestId> &newly_preempted,
                         ScheduleResult &result);
    void rollback_preempted_schedules(
        const std::vector<RequestId> &newly_preempted,
        std::vector<ScheduledRequest> &running_scheduled,
        std::uint64_t &token_budget);
    [[nodiscard]] bool try_reserve_with_preemption(
        RequestId requester, std::uint64_t scheduled_tokens, SimTime time,
        std::vector<RequestId> &newly_preempted,
        std::vector<ScheduledRequest> &running_scheduled,
        std::uint64_t &token_budget, ScheduleResult &result);
    [[nodiscard]] std::deque<RequestId> take_admission_queue();
    void restore_admission_queue(std::deque<RequestId> queue);
    [[nodiscard]] entities::TieredPrefixPlan
    build_tiered_prefix_plan(const entities::Request &request) const;
    void suspend_for_cpu_restore(RequestId request_id,
                                 const entities::TieredPrefixPlan &plan,
                                 SimTime time);
    [[nodiscard]] std::uint64_t revalidate_staged_frontier(
        const entities::Request &request,
        const entities::StagedCpuKVCacheRestore &staged) const;
    std::deque<RequestId> preempted_;
    std::vector<RequestId> running_;
    std::uint64_t next_iteration_id_ = 0;
    std::unordered_map<RequestId, std::uint64_t, StrongIdHash<RequestId>>
        pending_terminal_release_iterations_;
    std::unordered_set<RequestId, StrongIdHash<RequestId>>
        waiting_sensitive_release_extensions_;
    struct PrefillExportState {
        bool decode_pending = true;
        bool cpu_offload_pending = false;
    };
    std::unordered_map<RequestId, PrefillExportState, StrongIdHash<RequestId>>
        pending_exports_;
    config::ResolvedCpuKVCacheTargetConfig cpu_kv_cache_config_;
    std::unique_ptr<kv_cache::CpuKVCacheManager> cpu_kv_cache_;
    std::unique_ptr<
        cpu_kv_cache_transfer::AnalyticalCpuKVCacheTransferEngine>
        cpu_transfer_engine_;
    CpuKvTransferId::ValueType next_cpu_transfer_id_ = 0;
    std::unordered_map<CpuKvTransferId, entities::CpuKVCacheRestoreInfo,
                       StrongIdHash<CpuKvTransferId>>
        cpu_restore_operations_;
    std::unordered_map<CpuKvTransferId, entities::CpuKVCacheOffloadInfo,
                       StrongIdHash<CpuKvTransferId>>
        cpu_offload_operations_;
    std::unordered_map<RequestId, CpuKvTransferId, StrongIdHash<RequestId>>
        pending_cpu_restores_;
    std::unordered_map<RequestId, CpuKvTransferId, StrongIdHash<RequestId>>
        pending_cpu_offloads_;
    std::unordered_map<SessionId, CpuOffloadGeneration,
                       StrongIdHash<SessionId>>
        cpu_offload_generations_;
    std::unordered_map<RequestId, entities::StagedCpuKVCacheRestore,
                       StrongIdHash<RequestId>>
        staged_cpu_restores_;
    std::vector<ScheduledAuxiliaryEvent> pending_auxiliary_events_;
    bool terminal_release_followup_poll_pending_ = false;
};

} // namespace frontier::scheduler
