#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "frontier/core/event.h"
#include "frontier/core/ids.h"
#include "frontier/entities/batch.h"
#include "frontier/execution_time_predictor/batch_execution_model.h"
#include "frontier/scheduler/replica_stage_scheduler/replica_stage_scheduler.h"

namespace frontier::scheduler {

enum class SchedulerDecisionType : std::uint8_t {
  kRunningScheduled,
  kAdmission,
  kPreempted,
};

[[nodiscard]] std::string_view to_string(
    SchedulerDecisionType decision) noexcept;

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

class SchedulerError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class BaseReplicaScheduler {
 public:
  BaseReplicaScheduler(
      ReplicaId replica_id,
      DataParallelId dp_id,
      std::unique_ptr<
          execution_time_predictor::BatchExecutionModel>
          execution_model,
      std::uint64_t pipeline_parallel_size = 1);
  virtual ~BaseReplicaScheduler() = default;

  BaseReplicaScheduler(const BaseReplicaScheduler&) = delete;
  BaseReplicaScheduler& operator=(const BaseReplicaScheduler&) = delete;
  BaseReplicaScheduler(BaseReplicaScheduler&&) = delete;
  BaseReplicaScheduler& operator=(BaseReplicaScheduler&&) = delete;

  virtual void add_request(RequestId request_id) = 0;
  [[nodiscard]] virtual ScheduleResult schedule(SimTime time) = 0;
  virtual void mark_batch_started(const entities::Batch& batch) = 0;
  [[nodiscard]] virtual bool on_batch_completed(
      entities::Batch& batch,
      SimTime time) = 0;
  [[nodiscard]] virtual bool
  consume_terminal_release_followup_poll() noexcept {
    return false;
  }

  [[nodiscard]] virtual bool has_in_flight_batch()
      const noexcept = 0;
  [[nodiscard]] virtual std::uint64_t in_flight_batch_count()
      const noexcept = 0;
  [[nodiscard]] virtual std::uint64_t pipeline_parallel_size()
      const noexcept = 0;
  [[nodiscard]] virtual bool has_pending_work() const noexcept = 0;
  [[nodiscard]] virtual bool idle() const noexcept = 0;
  [[nodiscard]] virtual std::size_t waiting_count()
      const noexcept = 0;
  [[nodiscard]] virtual std::size_t running_count()
      const noexcept = 0;
  [[nodiscard]] virtual std::uint64_t allocated_kv_blocks()
      const noexcept = 0;

  [[nodiscard]] ReplicaId replica_id() const noexcept {
    return replica_id_;
  }
  [[nodiscard]] DataParallelId dp_id() const noexcept {
    return dp_id_;
  }
  [[nodiscard]] ReplicaStageScheduler& get_replica_stage_scheduler(
      StageId stage_id);
  [[nodiscard]] const ReplicaStageScheduler&
  get_replica_stage_scheduler(StageId stage_id) const;

 private:
  ReplicaId replica_id_;
  DataParallelId dp_id_;
  std::vector<ReplicaStageScheduler> stage_schedulers_;
};

}  // namespace frontier::scheduler
