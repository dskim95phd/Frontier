#include "frontier/entities/batch_stage.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace frontier::entities {
namespace {

void require_time_at_or_after(SimTime time, SimTime lower_bound,
                              const char *context) {
    if (!std::isfinite(time.seconds()) ||
        time.seconds() < lower_bound.seconds()) {
        throw BatchStageError(
            std::string{context} +
            " must be finite and not precede the previous boundary");
    }
}

bool valid_execution_time(const ExecutionTime &execution_time) {
    const std::array values{
        execution_time.dense_compute_ms,
        execution_time.tp_communication_ms,
        execution_time.pp_communication_ms,
        execution_time.moe_gating_linear_ms,
        execution_time.moe_gating_routing_topk_ms,
        execution_time.moe_grouped_gemm_ms,
        execution_time.moe_shuffling_ms,
        execution_time.moe_post_attention_norm_ms,
        execution_time.moe_tp_communication_ms,
        execution_time.ep_dispatch_ms,
        execution_time.ep_combine_ms,
        execution_time.dp_input_communication_ms,
        execution_time.dp_output_communication_ms,
        execution_time.synchronization_wait_ms,
    };
    return std::all_of(values.begin(), values.end(), [](double value) {
        return std::isfinite(value) && value >= 0.0;
    });
}

} // namespace

BatchStage::BatchStage(BatchId batch_id, ReplicaId replica_id,
                       DataParallelId dp_id, StageId stage_id,
                       SimTime arrived_at, ExecutionTime execution_time)
    : batch_id_(batch_id), replica_id_(replica_id), dp_id_(dp_id),
      stage_id_(stage_id), arrived_at_(arrived_at),
      execution_time_(execution_time) {
    if (!std::isfinite(arrived_at_.seconds()) || arrived_at_.seconds() < 0.0 ||
        !valid_execution_time(execution_time_)) {
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

void BatchStage::reconcile_synchronization_wait(SimTime completed_at) {
    if (!started_at_.valid() || !completed_at.valid() ||
        completed_at < started_at_) {
        throw BatchStageError(
            "cannot reconcile synchronization wait without valid boundaries");
    }
    const double wall_ms =
        (completed_at.seconds() - started_at_.seconds()) * 1e3;
    const double modeled_without_wait =
        execution_time_.total_ms() - execution_time_.synchronization_wait_ms;
    execution_time_.synchronization_wait_ms =
        std::max(0.0, wall_ms - modeled_without_wait);
}

} // namespace frontier::entities
