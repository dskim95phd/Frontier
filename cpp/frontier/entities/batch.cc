#include "frontier/entities/batch.h"

#include <cmath>
#include <limits>
#include <unordered_set>

namespace frontier::entities {

Batch::Batch(
    BatchId batch_id,
    IterationId iteration_id,
    std::vector<RequestBatchSnapshot> requests,
    SimTime scheduled_at,
    Generation schedule_epoch)
    : Batch(
          batch_id,
          iteration_id,
          std::move(requests),
          scheduled_at,
          schedule_epoch,
          ReplicaId{0},
          DataParallelId{0},
          1) {}

Batch::Batch(
    BatchId batch_id,
    IterationId iteration_id,
    std::vector<RequestBatchSnapshot> requests,
    SimTime scheduled_at,
    Generation schedule_epoch,
    ReplicaId replica_id,
    DataParallelId dp_id,
    std::uint64_t num_pipeline_stages,
    ClusterType cluster_type,
    BatchKind kind,
    config::ModelKind model_kind,
    MoESyncGroupId moe_sync_group_id,
    MoEParticipantId moe_participant_id)
    : batch_id_(batch_id),
      iteration_id_(iteration_id),
      requests_(std::move(requests)),
      scheduled_at_(scheduled_at),
      schedule_epoch_(schedule_epoch),
      replica_id_(replica_id),
      dp_id_(dp_id),
      num_pipeline_stages_(num_pipeline_stages),
      cluster_type_(cluster_type),
      kind_(kind),
      model_kind_(model_kind),
      moe_sync_group_id_(moe_sync_group_id),
      moe_participant_id_(moe_participant_id) {
  if (requests_.empty() && kind_ != BatchKind::kMoeIdle) {
    throw BatchError("batch must contain at least one request");
  }
  if (kind_ == BatchKind::kMoeIdle &&
      (!requests_.empty() ||
       model_kind_ != config::ModelKind::kMoe ||
       !moe_sync_group_id_.valid() ||
       !moe_participant_id_.valid())) {
    throw BatchError(
        "idle MoE batch requires no requests and valid synchronization IDs");
  }
  if (kind_ == BatchKind::kWork &&
      (moe_participant_id_.valid() != moe_sync_group_id_.valid())) {
    throw BatchError(
        "work batch MoE synchronization IDs must be both valid or invalid");
  }
  if (!std::isfinite(scheduled_at_.seconds()) ||
      scheduled_at_.seconds() < 0.0) {
    throw BatchError("batch schedule time must be finite and nonnegative");
  }
  if (num_pipeline_stages_ == 0) {
    throw BatchError("batch pipeline stage count must be positive");
  }
  std::unordered_set<std::uint64_t> ids;
  std::uint64_t total = 0;
  for (const RequestBatchSnapshot& request : requests_) {
    if (request.scheduled_tokens == 0) {
      throw BatchError("batch request token count must be positive");
    }
    if (!ids.insert(request.request_id.value()).second) {
      throw BatchError("batch contains a duplicate request ID");
    }
    if (total >
        std::numeric_limits<std::uint64_t>::max() -
            request.scheduled_tokens) {
      throw BatchError("batch token count overflows uint64");
    }
    total += request.scheduled_tokens;
  }
}

std::uint64_t Batch::total_scheduled_tokens() const noexcept {
  std::uint64_t total = 0;
  for (const RequestBatchSnapshot& request : requests_) {
    total += request.scheduled_tokens;
  }
  return total;
}

void Batch::mark_completed(SimTime time) {
  if (completed_at_.valid()) {
    throw BatchError("batch completed more than once");
  }
  if (!std::isfinite(time.seconds()) ||
      time.seconds() < scheduled_at_.seconds()) {
    throw BatchError(
        "batch completion must be finite and not precede scheduling");
  }
  completed_at_ = time;
}

void Batch::set_global_id(BatchGlobalId global_id) {
  if (!global_id.valid() || global_id_.valid()) {
    throw BatchError(
        "batch global ID must be valid and assigned exactly once");
  }
  global_id_ = global_id;
}

void Batch::set_stage_layer(LayerId layer_id) {
  if (!layer_id.valid()) {
    throw BatchError("batch stage layer must be valid");
  }
  if (stage_layer_.valid() && layer_id < stage_layer_) {
    throw BatchError("batch stage layer cannot move backwards");
  }
  stage_layer_ = layer_id;
}

void Batch::reset_stage_layer() {
  stage_layer_ = LayerId{0};
}

void Batch::set_moe_synchronization(
    MoESyncGroupId sync_group_id,
    MoEParticipantId participant_id) {
  if (!is_moe() || !sync_group_id.valid() ||
      !participant_id.valid()) {
    throw BatchError(
        "MoE synchronization requires a MoE batch and valid IDs");
  }
  if ((moe_sync_group_id_.valid() &&
       moe_sync_group_id_ != sync_group_id) ||
      (moe_participant_id_.valid() &&
       moe_participant_id_ != participant_id)) {
    throw BatchError("MoE synchronization IDs cannot be reassigned");
  }
  moe_sync_group_id_ = sync_group_id;
  moe_participant_id_ = participant_id;
}

}  // namespace frontier::entities
