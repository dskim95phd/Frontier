#include "frontier/scheduler/replica_stage_scheduler/replica_stage_scheduler.h"

#include <algorithm>
#include <cmath>
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

std::optional<StageBatchTicket>
ReplicaStageScheduler::peek_batch_if_not_busy() const {
    if (active_batch_id_.valid() || queue_.empty()) {
        return std::nullopt;
    }
    return queue_.top();
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
    const entities::RequestCollection &requests) const {
    if (!active_batch_id_.valid() || active_batch_id_ != batch.id()) {
        throw ReplicaStageSchedulerError(
            "only the active stage batch can be predicted");
    }
    return predictor_->predict_stage_execution_time(batch, requests, stage_id_);
}

execution_time_predictor::ExecutionTimePrediction
ReplicaStageScheduler::predict_collapsed(
    const entities::Batch &batch,
    const entities::RequestCollection &requests) const {
    return predictor_->predict_stage_execution_time(batch, requests, stage_id_);
}

void ReplicaStageScheduler::prune_collapsed_reservations(SimTime time) {
    if (!time.valid()) {
        throw ReplicaStageSchedulerError(
            "collapsed reservation time must be finite and nonnegative");
    }
    collapsed_reservations_.erase(
        std::remove_if(collapsed_reservations_.begin(),
                       collapsed_reservations_.end(),
                       [time](const CollapsedReservation &reservation) {
                           return reservation.end <= time;
                       }),
        collapsed_reservations_.end());
    if (collapsed_stage_batch_id_.valid() &&
        collapsed_stage_release_at_.valid() &&
        collapsed_stage_release_at_ <= time) {
        if (active_batch_id_ == collapsed_stage_batch_id_) {
            active_batch_id_ = BatchId{};
        }
        collapsed_stage_batch_id_ = BatchId{};
        collapsed_stage_release_at_ = SimTime{};
    }
}

SimTime ReplicaStageScheduler::collapsed_start_time(SimTime requested,
                                                     double duration_ms) {
    if (!requested.valid() || !std::isfinite(duration_ms) ||
        duration_ms < 0.0) {
        throw ReplicaStageSchedulerError(
            "collapsed stage interval has invalid timing");
    }
    const double duration_seconds = duration_ms * 1e-3;
    if (!std::isfinite(duration_seconds)) {
        throw ReplicaStageSchedulerError(
            "collapsed stage interval duration is nonfinite");
    }
    prune_collapsed_reservations(requested);
    // The exact calendar runs each stage first-in-first-out in arrival order,
    // and a batch cannot overtake an earlier one because every stage is
    // serialized.  Reservations are therefore installed in that same order, so
    // this batch starts once every still-live reservation on the stage has
    // ended.  Searching for the earliest free gap instead would let a later
    // batch backfill ahead of an earlier one, which exact mode never does and
    // which measurably understates pipeline latency once three or more batches
    // are in flight.
    double candidate = requested.seconds();
    for (const CollapsedReservation &reservation : collapsed_reservations_) {
        candidate = std::max(candidate, reservation.end.seconds());
    }
    const double candidate_end = candidate + duration_seconds;
    if (!std::isfinite(candidate_end)) {
        throw ReplicaStageSchedulerError(
            "collapsed stage interval end is nonfinite");
    }
    return SimTime::from_seconds(candidate);
}

void ReplicaStageScheduler::reserve_collapsed_interval(BatchId batch_id,
                                                        SimTime start,
                                                        SimTime end) {
    if (!batch_id.valid() || !start.valid() || !end.valid() || end < start) {
        throw ReplicaStageSchedulerError(
            "collapsed stage reservation has invalid identity or timing");
    }
    prune_collapsed_reservations(start);
    const auto duplicate = std::find_if(
        collapsed_reservations_.begin(), collapsed_reservations_.end(),
        [batch_id](const CollapsedReservation &reservation) {
            return reservation.batch_id == batch_id;
        });
    if (duplicate != collapsed_reservations_.end()) {
        throw ReplicaStageSchedulerError(
            "collapsed stage reservation already exists for batch");
    }
    const auto position = std::lower_bound(
        collapsed_reservations_.begin(), collapsed_reservations_.end(), start,
        [](const CollapsedReservation &reservation, SimTime value) {
            return reservation.start < value;
        });
    if (position != collapsed_reservations_.begin()) {
        const auto previous = std::prev(position);
        if (previous->end > start) {
            throw ReplicaStageSchedulerError(
                "collapsed stage reservations overlap");
        }
    }
    if (position != collapsed_reservations_.end() &&
        position->start < end) {
        throw ReplicaStageSchedulerError(
            "collapsed stage reservations overlap");
    }
    collapsed_reservations_.insert(position,
                                   CollapsedReservation{batch_id, start, end});
    // Stage zero is claimed by pop_batch_if_not_busy() immediately before the
    // collapsed reservation is installed.  Keep that claim until its own
    // interval ends so a wake schedule event cannot start work early.
    if (stage_id_.value() == 0) {
        if (collapsed_stage_batch_id_.valid()) {
            throw ReplicaStageSchedulerError(
                "collapsed stage zero already has an active reservation");
        }
        collapsed_stage_batch_id_ = batch_id;
        collapsed_stage_release_at_ = end;
    }
}

void ReplicaStageScheduler::release_collapsed_stage_if_due(SimTime time) {
    prune_collapsed_reservations(time);
}

void ReplicaStageScheduler::release_collapsed_reservation(BatchId batch_id) {
    if (!batch_id.valid()) {
        throw ReplicaStageSchedulerError(
            "collapsed reservation release requires a valid batch");
    }
    collapsed_reservations_.erase(
        std::remove_if(collapsed_reservations_.begin(),
                       collapsed_reservations_.end(),
                       [batch_id](const CollapsedReservation &reservation) {
                           return reservation.batch_id == batch_id;
                       }),
        collapsed_reservations_.end());
    if (collapsed_stage_batch_id_ == batch_id) {
        if (active_batch_id_ == batch_id) {
            active_batch_id_ = BatchId{};
        }
        collapsed_stage_batch_id_ = BatchId{};
        collapsed_stage_release_at_ = SimTime{};
    }
}

bool ReplicaStageScheduler::supports_lazy_moe_prediction() const noexcept {
    return predictor_->supports_lazy_moe_prediction();
}

execution_time_predictor::ExecutionTimePrediction
ReplicaStageScheduler::prepare_moe_stage(
    const entities::Batch &batch,
    const entities::RequestCollection &requests) const {
    if (!active_batch_id_.valid() || active_batch_id_ != batch.id()) {
        throw ReplicaStageSchedulerError(
            "only the active stage batch can prepare MoE execution");
    }
    return predictor_->prepare_moe_stage_execution(batch, requests, stage_id_);
}

execution_time_predictor::ExecutionTimePrediction
ReplicaStageScheduler::predict_moe_layer(
    const entities::Batch &batch,
    const entities::RequestCollection &requests,
    std::uint64_t local_moe_layer) const {
    if (!active_batch_id_.valid() || active_batch_id_ != batch.id()) {
        throw ReplicaStageSchedulerError(
            "only the active stage batch can predict a MoE layer");
    }
    return predictor_->predict_moe_layer_execution(
        batch, requests, stage_id_, local_moe_layer);
}

execution_time_predictor::MoEGroupLayerPrediction
ReplicaStageScheduler::predict_moe_group_layer(
    const execution_time_predictor::MoEGroupLayerInput &input) const {
    if (!active_batch_id_.valid()) {
        throw ReplicaStageSchedulerError(
            "only an active stage can predict a grouped MoE layer");
    }
    return predictor_->predict_moe_group_layer(input);
}

void ReplicaStageScheduler::on_stage_end(BatchId batch_id) {
    if (!active_batch_id_.valid() || active_batch_id_ != batch_id) {
        throw ReplicaStageSchedulerError(
            "stage completion does not match the active batch");
    }
    active_batch_id_ = BatchId{};
}

} // namespace frontier::scheduler
