#include "frontier/scheduler/replica_stage_scheduler/replica_stage_scheduler.h"

#include <algorithm>
#include <utility>

namespace frontier::scheduler {

ReplicaStageScheduler::ReplicaStageScheduler(
    ReplicaId replica_id,
    DataParallelId dp_id,
    StageId stage_id,
    bool is_last_stage,
    std::unique_ptr<
        execution_time_predictor::BatchExecutionModel>
        execution_model)
    : replica_id_(replica_id),
      dp_id_(dp_id),
      stage_id_(stage_id),
      is_last_stage_(is_last_stage),
      execution_model_(std::move(execution_model)) {
  if (execution_model_ == nullptr) {
    throw ReplicaStageSchedulerError(
        "replica stage requires an execution model");
  }
}

void ReplicaStageScheduler::add_batch(
    const entities::Batch& batch) {
  const auto has_batch_id =
      [&batch](const StageBatchTicket& ticket) {
        return ticket.batch_id == batch.id();
      };
  if ((active_batch_id_.has_value() &&
       active_batch_id_.value() == batch.id()) ||
      std::find_if(queue_.begin(), queue_.end(), has_batch_id) !=
          queue_.end()) {
    throw ReplicaStageSchedulerError(
        "batch is already present in replica stage state");
  }
  queue_.push_back(StageBatchTicket{
      .batch_id = batch.id(),
      .schedule_epoch = batch.schedule_epoch(),
  });
}

std::optional<StageBatchTicket>
ReplicaStageScheduler::pop_batch_if_not_busy() {
  if (active_batch_id_.has_value() || queue_.empty()) {
    return std::nullopt;
  }
  const StageBatchTicket ticket = queue_.front();
  queue_.pop_front();
  active_batch_id_ = ticket.batch_id;
  return ticket;
}

execution_time_predictor::BatchExecutionPrediction
ReplicaStageScheduler::predict(
    const entities::Batch& batch,
    const std::vector<entities::Request>& requests) const {
  if (!active_batch_id_.has_value() ||
      active_batch_id_.value() != batch.id()) {
    throw ReplicaStageSchedulerError(
        "only the active stage batch can be predicted");
  }
  return execution_model_->predict(batch, requests, stage_id_);
}

void ReplicaStageScheduler::on_stage_end(BatchId batch_id) {
  if (!active_batch_id_.has_value() ||
      active_batch_id_.value() != batch_id) {
    throw ReplicaStageSchedulerError(
        "stage completion does not match the active batch");
  }
  active_batch_id_.reset();
}

}  // namespace frontier::scheduler
