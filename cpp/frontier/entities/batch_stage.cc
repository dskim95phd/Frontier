#include "frontier/entities/batch_stage.h"

#include <cmath>

namespace frontier::entities {
namespace {

void require_time_at_or_after(
    SimTime time,
    SimTime lower_bound,
    const char* context) {
  if (!std::isfinite(time.seconds()) ||
      time.seconds() < lower_bound.seconds()) {
    throw BatchStageError(
        std::string{context} +
        " must be finite and not precede the previous boundary");
  }
}

}  // namespace

BatchStage::BatchStage(
    BatchId batch_id,
    ReplicaId replica_id,
    DataParallelId dp_id,
    StageId stage_id,
    SimTime arrived_at,
    ExecutionTime execution_time)
    : batch_id_(batch_id),
      replica_id_(replica_id),
      dp_id_(dp_id),
      stage_id_(stage_id),
      arrived_at_(arrived_at),
      execution_time_(execution_time) {
  if (!std::isfinite(arrived_at_.seconds()) ||
      arrived_at_.seconds() < 0.0 ||
      !std::isfinite(execution_time_.dense_compute_ms) ||
      !std::isfinite(execution_time_.tp_communication_ms) ||
      !std::isfinite(execution_time_.pp_communication_ms) ||
      execution_time_.dense_compute_ms < 0.0 ||
      execution_time_.tp_communication_ms < 0.0 ||
      execution_time_.pp_communication_ms < 0.0) {
    throw BatchStageError("batch stage contains invalid timing");
  }
}

void BatchStage::mark_started(SimTime time) {
  if (started_at_.valid()) {
    throw BatchStageError("batch stage started more than once");
  }
  require_time_at_or_after(time, arrived_at_, "stage start");
  started_at_ = time;
}

void BatchStage::mark_completed(SimTime time) {
  if (!started_at_.valid()) {
    throw BatchStageError("batch stage completed before start");
  }
  if (completed_at_.valid()) {
    throw BatchStageError("batch stage completed more than once");
  }
  require_time_at_or_after(time, started_at_, "stage completion");
  completed_at_ = time;
}

}  // namespace frontier::entities
