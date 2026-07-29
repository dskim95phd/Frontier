#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "frontier/core/event.h"
#include "frontier/core/ids.h"
#include "frontier/entities/request.h"

namespace frontier::entities {

struct RequestBatchSnapshot {
  RequestId request_id;
  std::uint64_t scheduled_tokens;
  std::uint64_t runtime_epoch;
  std::uint64_t execution_epoch;
  std::uint64_t processed_tokens;
  std::uint64_t scheduler_frontier;

  friend bool operator==(
      const RequestBatchSnapshot&,
      const RequestBatchSnapshot&) = default;
};

class BatchError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class Batch {
 public:
  Batch(
      BatchId batch_id,
      IterationId iteration_id,
      std::vector<RequestBatchSnapshot> requests,
      SimTime scheduled_at,
      Generation schedule_epoch);
  Batch(
      BatchId batch_id,
      IterationId iteration_id,
      std::vector<RequestBatchSnapshot> requests,
      SimTime scheduled_at,
      Generation schedule_epoch,
      ReplicaId replica_id,
      DataParallelId dp_id,
      std::uint64_t num_pipeline_stages);

  [[nodiscard]] BatchId id() const noexcept { return batch_id_; }
  [[nodiscard]] IterationId iteration_id() const noexcept {
    return iteration_id_;
  }
  [[nodiscard]] const std::vector<RequestBatchSnapshot>& requests()
      const noexcept {
    return requests_;
  }
  [[nodiscard]] SimTime scheduled_at() const noexcept {
    return scheduled_at_;
  }
  [[nodiscard]] const std::optional<SimTime>& completed_at()
      const noexcept {
    return completed_at_;
  }
  [[nodiscard]] Generation schedule_epoch() const noexcept {
    return schedule_epoch_;
  }
  [[nodiscard]] ReplicaId replica_id() const noexcept {
    return replica_id_;
  }
  [[nodiscard]] DataParallelId dp_id() const noexcept {
    return dp_id_;
  }
  [[nodiscard]] std::uint64_t num_pipeline_stages() const noexcept {
    return num_pipeline_stages_;
  }
  [[nodiscard]] std::uint64_t total_scheduled_tokens() const noexcept;
  [[nodiscard]] bool completed() const noexcept {
    return completed_at_.has_value();
  }

  void mark_completed(SimTime time);

 private:
  BatchId batch_id_;
  IterationId iteration_id_;
  std::vector<RequestBatchSnapshot> requests_;
  SimTime scheduled_at_;
  std::optional<SimTime> completed_at_;
  Generation schedule_epoch_;
  ReplicaId replica_id_{0};
  DataParallelId dp_id_{0};
  std::uint64_t num_pipeline_stages_ = 1;
};

}  // namespace frontier::entities
