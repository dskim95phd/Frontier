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
    std::uint64_t num_pipeline_stages)
    : batch_id_(batch_id),
      iteration_id_(iteration_id),
      requests_(std::move(requests)),
      scheduled_at_(scheduled_at),
      schedule_epoch_(schedule_epoch),
      replica_id_(replica_id),
      dp_id_(dp_id),
      num_pipeline_stages_(num_pipeline_stages) {
  if (requests_.empty()) {
    throw BatchError("batch must contain at least one request");
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
  if (completed_at_.has_value()) {
    throw BatchError("batch completed more than once");
  }
  if (!std::isfinite(time.seconds()) ||
      time.seconds() < scheduled_at_.seconds()) {
    throw BatchError(
        "batch completion must be finite and not precede scheduling");
  }
  completed_at_ = time;
}

}  // namespace frontier::entities
