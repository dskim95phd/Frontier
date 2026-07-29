#include "frontier/scheduler/replica_scheduler/base_replica_scheduler.h"

#include <utility>

namespace frontier::scheduler {

std::string_view to_string(
    SchedulerDecisionType decision) noexcept {
  switch (decision) {
    case SchedulerDecisionType::kRunningScheduled:
      return "RUNNING_SCHEDULED";
    case SchedulerDecisionType::kAdmission:
      return "ADMISSION";
    case SchedulerDecisionType::kPreempted:
      return "PREEMPTED";
  }
  return "UNKNOWN";
}

BaseReplicaScheduler::BaseReplicaScheduler(
    ReplicaId replica_id,
    DataParallelId dp_id,
    std::unique_ptr<
        execution_time_predictor::BatchExecutionModel>
        execution_model,
    std::uint64_t pipeline_parallel_size)
    : replica_id_(replica_id),
      dp_id_(dp_id) {
  if (execution_model == nullptr ||
      pipeline_parallel_size == 0) {
    throw SchedulerError(
        "replica scheduler requires an execution model and positive PP");
  }
  stage_schedulers_.reserve(
      static_cast<std::size_t>(pipeline_parallel_size));
  for (std::uint64_t stage = 0;
       stage < pipeline_parallel_size;
       ++stage) {
    std::unique_ptr<
        execution_time_predictor::BatchExecutionModel>
        stage_model =
            stage + 1 == pipeline_parallel_size
            ? std::move(execution_model)
            : execution_model->clone();
    stage_schedulers_.emplace_back(
        replica_id,
        dp_id,
        StageId{stage},
        stage + 1 == pipeline_parallel_size,
        std::move(stage_model));
  }
}

ReplicaStageScheduler&
BaseReplicaScheduler::get_replica_stage_scheduler(
    StageId stage_id) {
  if (stage_id.value() >= stage_schedulers_.size()) {
    throw SchedulerError(
        "co-location baseline exposes only pipeline stage zero");
  }
  return stage_schedulers_.at(
      static_cast<std::size_t>(stage_id.value()));
}

const ReplicaStageScheduler&
BaseReplicaScheduler::get_replica_stage_scheduler(
    StageId stage_id) const {
  if (stage_id.value() >= stage_schedulers_.size()) {
    throw SchedulerError(
        "co-location baseline exposes only pipeline stage zero");
  }
  return stage_schedulers_.at(
      static_cast<std::size_t>(stage_id.value()));
}

}  // namespace frontier::scheduler
