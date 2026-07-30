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
    const entities::Replica& replica,
    DataParallelId dp_id,
    std::shared_ptr<
        const execution_time_predictor::BatchExecutionModel>
        execution_model,
    ClusterType cluster_type)
    : replica_(&replica),
      dp_id_(dp_id),
      cluster_type_(cluster_type) {
  const std::uint64_t pipeline_parallel_size =
      replica.pipeline_parallel_size();
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
    stage_schedulers_.emplace_back(
        replica.id(),
        dp_id,
        StageId{stage},
        stage + 1 == pipeline_parallel_size,
        execution_model);
  }
}

ReplicaStageScheduler&
BaseReplicaScheduler::get_replica_stage_scheduler(
    StageId stage_id) {
  if (!stage_id.valid() ||
      stage_id.index() >= stage_schedulers_.size()) {
    throw SchedulerError(
        "single-stage scheduler exposes only pipeline stage zero");
  }
  return stage_schedulers_.at(
      stage_id.index());
}

const ReplicaStageScheduler&
BaseReplicaScheduler::get_replica_stage_scheduler(
    StageId stage_id) const {
  if (!stage_id.valid() ||
      stage_id.index() >= stage_schedulers_.size()) {
    throw SchedulerError(
        "single-stage scheduler exposes only pipeline stage zero");
  }
  return stage_schedulers_.at(
      stage_id.index());
}

}  // namespace frontier::scheduler
