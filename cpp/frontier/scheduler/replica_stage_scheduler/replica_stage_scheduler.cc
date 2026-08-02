#include "frontier/scheduler/replica_stage_scheduler/replica_stage_scheduler.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace frontier::scheduler {

ReplicaStageScheduler::ReplicaStageScheduler(
    ReplicaId replica_id, DataParallelId dp_id, StageId stage_id,
    bool is_last_stage,
    execution_time_predictor::ExecutionTimePredictorPtr predictor)
    : replica_id_(replica_id), dp_id_(dp_id), stage_id_(stage_id),
      is_last_stage_(is_last_stage), predictor_(std::move(predictor)) {
    if (predictor_ == nullptr) {
        throw ReplicaStageSchedulerError(
            "replica stage requires an execution model");
    }
}

void ReplicaStageScheduler::add_batch(const entities::Batch &batch) {
    if (!batch.global_id().valid()) {
        throw ReplicaStageSchedulerError(
            "replica stage batch requires a valid global ID");
    }
    if ((active_batch_id_.valid() && active_batch_id_ == batch.id()) ||
        queued_batch_ids_.find(batch.id()) != queued_batch_ids_.end()) {
        throw ReplicaStageSchedulerError(
            "batch is already present in replica stage state");
    }
    if (next_insertion_order_ == std::numeric_limits<std::uint64_t>::max()) {
        throw ReplicaStageSchedulerError(
            "replica stage insertion order overflow");
    }
    queue_.push([&]() {
        StageBatchTicket value{};
        value.batch_id = batch.id();
        value.batch_global_id = batch.global_id();
        value.insertion_order = next_insertion_order_++;
        value.schedule_epoch = batch.schedule_epoch();
        return value;
    }());
    queued_batch_ids_.insert(batch.id());
}

std::optional<StageBatchTicket> ReplicaStageScheduler::pop_batch_if_not_busy() {
    if (active_batch_id_.valid() || queue_.empty()) {
        return std::nullopt;
    }
    const StageBatchTicket ticket = queue_.top();
    queue_.pop();
    queued_batch_ids_.erase(ticket.batch_id);
    active_batch_id_ = ticket.batch_id;
    return ticket;
}

execution_time_predictor::ExecutionTimePrediction
ReplicaStageScheduler::predict(
    const entities::Batch &batch,
    const std::vector<entities::Request> &requests) const {
    if (!active_batch_id_.valid() || active_batch_id_ != batch.id()) {
        throw ReplicaStageSchedulerError(
            "only the active stage batch can be predicted");
    }
    return predictor_->predict_stage_execution_time(batch, requests, stage_id_);
}

bool ReplicaStageScheduler::supports_lazy_moe_prediction() const noexcept {
    return predictor_->supports_lazy_moe_prediction();
}

execution_time_predictor::ExecutionTimePrediction
ReplicaStageScheduler::prepare_moe_stage(
    const entities::Batch &batch,
    const std::vector<entities::Request> &requests) const {
    if (!active_batch_id_.valid() || active_batch_id_ != batch.id()) {
        throw ReplicaStageSchedulerError(
            "only the active stage batch can prepare MoE execution");
    }
    return predictor_->prepare_moe_stage_execution(batch, requests, stage_id_);
}

execution_time_predictor::ExecutionTimePrediction
ReplicaStageScheduler::predict_moe_layer(
    const entities::Batch &batch,
    const std::vector<entities::Request> &requests,
    std::uint64_t local_moe_layer) const {
    if (!active_batch_id_.valid() || active_batch_id_ != batch.id()) {
        throw ReplicaStageSchedulerError(
            "only the active stage batch can predict a MoE layer");
    }
    return predictor_->predict_moe_layer_execution(
        batch, requests, stage_id_, local_moe_layer);
}

void ReplicaStageScheduler::on_stage_end(BatchId batch_id) {
    if (!active_batch_id_.valid() || active_batch_id_ != batch_id) {
        throw ReplicaStageSchedulerError(
            "stage completion does not match the active batch");
    }
    active_batch_id_ = BatchId{};
}

} // namespace frontier::scheduler
