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
#include "frontier/execution_time_predictor/batch_execution_model.h"
#include "frontier/scheduler/kv_block_accounting.h"
#include "frontier/scheduler/replica_scheduler/base_replica_scheduler.h"

namespace frontier::scheduler {

class VllmV1Scheduler final : public BaseReplicaScheduler {
 public:
  VllmV1Scheduler(
      config::SchedulerConfig config,
      std::vector<entities::Request>& requests);
  VllmV1Scheduler(
      config::SchedulerConfig config,
      std::vector<entities::Request>& requests,
      std::unique_ptr<
          execution_time_predictor::BatchExecutionModel>
          execution_model,
      ReplicaId replica_id = ReplicaId{0},
      DataParallelId dp_id = DataParallelId{0},
      std::uint64_t pipeline_parallel_size = 1);

  void add_request(RequestId request_id) override;
  [[nodiscard]] ScheduleResult schedule(SimTime time) override;
  void mark_batch_started(const entities::Batch& batch) override;
  [[nodiscard]] bool on_batch_completed(
      entities::Batch& batch,
      SimTime time) override;
  [[nodiscard]] bool
  consume_terminal_release_followup_poll() noexcept override;

  [[nodiscard]] bool has_in_flight_batch()
      const noexcept override {
    return in_flight_batch_count_ > 0;
  }
  [[nodiscard]] std::uint64_t in_flight_batch_count()
      const noexcept override {
    return in_flight_batch_count_;
  }
  [[nodiscard]] std::uint64_t pipeline_parallel_size()
      const noexcept override {
    return pipeline_parallel_size_;
  }
  [[nodiscard]] bool has_pending_work() const noexcept override {
    return !preempted_.empty() || !waiting_.empty() ||
        !running_.empty();
  }
  [[nodiscard]] bool idle() const noexcept override {
    return !has_pending_work() && in_flight_batch_count_ == 0 &&
        kv_blocks_.empty();
  }
  [[nodiscard]] std::size_t waiting_count()
      const noexcept override {
    return preempted_.size() + waiting_.size();
  }
  [[nodiscard]] std::size_t running_count()
      const noexcept override {
    return running_.size();
  }
  [[nodiscard]] std::uint64_t allocated_kv_blocks()
      const noexcept override {
    return kv_blocks_.total_allocated_blocks();
  }
  [[nodiscard]] const std::deque<RequestId>& waiting_queue()
      const noexcept {
    return waiting_;
  }
  [[nodiscard]] const std::deque<RequestId>& preempted_queue()
      const noexcept {
    return preempted_;
  }
  [[nodiscard]] const std::vector<RequestId>& running_order()
      const noexcept {
    return running_;
  }
  [[nodiscard]] const KvBlockAccounting& kv_blocks() const noexcept {
    return kv_blocks_;
  }

 private:
  [[nodiscard]] entities::Request& request(RequestId request_id);
  [[nodiscard]] const entities::Request& request(
      RequestId request_id) const;
  [[nodiscard]] bool contains_request(RequestId request_id) const;
  [[nodiscard]] bool request_is_active(RequestId request_id) const;
  [[nodiscard]] std::uint64_t next_num_tokens(
      const entities::Request& request) const;
  [[nodiscard]] std::uint64_t kv_accounted_tokens(
      const entities::Request& request) const;
  [[nodiscard]] std::optional<RequestId> select_preemption_victim(
      RequestId requester) const;
  [[nodiscard]] std::uint64_t
  extra_terminal_release_iterations() const noexcept;
  [[nodiscard]] std::uint64_t
  iteration_start_release_threshold() const noexcept;
  [[nodiscard]] bool has_visible_waiting_requests() const noexcept;
  void free_completed_request(RequestId request_id);
  void materialize_terminal_releases_before_iteration();
  void advance_terminal_release_boundary();
  void preempt_request(
      RequestId victim,
      SimTime time,
      std::vector<RequestId>& newly_preempted,
      ScheduleResult& result);
  void rollback_preempted_schedules(
      const std::vector<RequestId>& newly_preempted,
      std::vector<ScheduledRequest>& running_scheduled,
      std::uint64_t& token_budget);
  [[nodiscard]] bool try_reserve_with_preemption(
      RequestId requester,
      std::uint64_t scheduled_tokens,
      SimTime time,
      std::vector<RequestId>& newly_preempted,
      std::vector<ScheduledRequest>& running_scheduled,
      std::uint64_t& token_budget,
      ScheduleResult& result);
  void validate_state() const;

  config::SchedulerConfig config_;
  std::vector<entities::Request>* requests_;
  KvBlockAccounting kv_blocks_;
  std::deque<RequestId> preempted_;
  std::deque<RequestId> waiting_;
  std::vector<RequestId> running_;
  std::uint64_t next_iteration_id_ = 0;
  std::uint64_t pipeline_parallel_size_ = 1;
  std::uint64_t in_flight_batch_count_ = 0;
  std::unordered_map<
      BatchId,
      std::vector<RequestId>,
      StrongIdHash<BatchId>>
      in_flight_batches_;
  std::unordered_set<RequestId, StrongIdHash<RequestId>>
      active_requests_;
  std::unordered_map<
      RequestId,
      std::uint64_t,
      StrongIdHash<RequestId>>
      pending_terminal_release_iterations_;
  std::unordered_set<RequestId, StrongIdHash<RequestId>>
      waiting_sensitive_release_extensions_;
  bool terminal_release_followup_poll_pending_ = false;
};

}  // namespace frontier::scheduler
