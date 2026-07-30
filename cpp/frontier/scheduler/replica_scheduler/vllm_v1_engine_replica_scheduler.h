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
#include "frontier/execution_time_predictor/base_execution_time_predictor.h"
#include "frontier/scheduler/kv_block_accounting.h"
#include "frontier/scheduler/replica_scheduler/base_replica_scheduler.h"

namespace frontier::scheduler {

class VllmV1Scheduler final : public BaseReplicaScheduler {
  public:
    VllmV1Scheduler(config::SchedulerConfig config,
                    std::vector<entities::Request> &requests);
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
        ClusterType cluster_type);

    [[nodiscard]] bool
    consume_terminal_release_followup_poll() noexcept override;
    void complete_kv_transfer(RequestId request_id) override;
    [[nodiscard]] std::size_t
    pending_kv_transfer_count() const noexcept override {
        return pending_kv_transfers_.size();
    }

    [[nodiscard]] bool has_pending_work() const noexcept override {
        return !preempted_.empty() || !waiting_.empty() || !running_.empty();
    }
    [[nodiscard]] bool idle() const noexcept override {
        return !has_pending_work() && in_flight_batch_count_ == 0 &&
               pending_kv_transfers_.empty() && kv_blocks_.empty();
    }
    [[nodiscard]] std::size_t waiting_count() const noexcept override {
        return preempted_.size() + waiting_.size();
    }
    [[nodiscard]] std::size_t running_count() const noexcept override {
        return running_.size();
    }
    [[nodiscard]] std::uint64_t allocated_kv_blocks() const noexcept override {
        return kv_blocks_.total_allocated_blocks();
    }
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
    [[nodiscard]] const KvBlockAccounting &kv_blocks() const noexcept {
        return kv_blocks_;
    }

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
    std::deque<RequestId> preempted_;
    std::vector<RequestId> running_;
    std::uint64_t next_iteration_id_ = 0;
    std::unordered_map<RequestId, std::uint64_t, StrongIdHash<RequestId>>
        pending_terminal_release_iterations_;
    std::unordered_set<RequestId, StrongIdHash<RequestId>>
        waiting_sensitive_release_extensions_;
    std::unordered_set<RequestId, StrongIdHash<RequestId>>
        pending_kv_transfers_;
    bool terminal_release_followup_poll_pending_ = false;
};

} // namespace frontier::scheduler
