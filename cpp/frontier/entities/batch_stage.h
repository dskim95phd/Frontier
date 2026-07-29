#pragma once

#include <optional>
#include <stdexcept>

#include "frontier/core/event.h"
#include "frontier/core/ids.h"
#include "frontier/entities/execution_time.h"

namespace frontier::entities {

class BatchStageError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class BatchStage {
 public:
  BatchStage(
      BatchId batch_id,
      ReplicaId replica_id,
      DataParallelId dp_id,
      StageId stage_id,
      SimTime arrived_at,
      ExecutionTime execution_time);

  void mark_started(SimTime time);
  void mark_completed(SimTime time);

  [[nodiscard]] BatchId batch_id() const noexcept { return batch_id_; }
  [[nodiscard]] ReplicaId replica_id() const noexcept {
    return replica_id_;
  }
  [[nodiscard]] DataParallelId dp_id() const noexcept { return dp_id_; }
  [[nodiscard]] StageId stage_id() const noexcept { return stage_id_; }
  [[nodiscard]] SimTime arrived_at() const noexcept { return arrived_at_; }
  [[nodiscard]] const std::optional<SimTime>& started_at() const noexcept {
    return started_at_;
  }
  [[nodiscard]] const std::optional<SimTime>& completed_at() const noexcept {
    return completed_at_;
  }
  [[nodiscard]] const ExecutionTime& execution_time() const noexcept {
    return execution_time_;
  }

 private:
  BatchId batch_id_;
  ReplicaId replica_id_;
  DataParallelId dp_id_;
  StageId stage_id_;
  SimTime arrived_at_;
  ExecutionTime execution_time_;
  std::optional<SimTime> started_at_;
  std::optional<SimTime> completed_at_;
};

}  // namespace frontier::entities
