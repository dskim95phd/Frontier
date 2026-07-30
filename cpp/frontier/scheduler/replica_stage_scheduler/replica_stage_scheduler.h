#pragma once

#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include "frontier/core/ids.h"
#include "frontier/entities/batch.h"
#include "frontier/entities/request.h"
#include "frontier/execution_time_predictor/batch_execution_model.h"

namespace frontier::scheduler {

class ReplicaStageSchedulerError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

struct StageBatchTicket {
  BatchId batch_id;
  Generation schedule_epoch;
};

class ReplicaStageScheduler {
 public:
  ReplicaStageScheduler(
      ReplicaId replica_id,
      DataParallelId dp_id,
      StageId stage_id,
      bool is_last_stage,
      std::unique_ptr<
          execution_time_predictor::BatchExecutionModel>
          execution_model);

  void add_batch(const entities::Batch& batch);
  [[nodiscard]] std::optional<StageBatchTicket>
  pop_batch_if_not_busy();
  [[nodiscard]]
  execution_time_predictor::BatchExecutionPrediction predict(
      const entities::Batch& batch,
      const std::vector<entities::Request>& requests) const;
  void on_stage_end(BatchId batch_id);

  [[nodiscard]] ReplicaId replica_id() const noexcept {
    return replica_id_;
  }
  [[nodiscard]] DataParallelId dp_id() const noexcept {
    return dp_id_;
  }
  [[nodiscard]] StageId stage_id() const noexcept {
    return stage_id_;
  }
  [[nodiscard]] bool is_last_stage() const noexcept {
    return is_last_stage_;
  }
  [[nodiscard]] bool is_busy() const noexcept {
    return active_batch_id_.valid();
  }
  [[nodiscard]] bool empty() const noexcept {
    return queue_.empty();
  }

 private:
  ReplicaId replica_id_;
  DataParallelId dp_id_;
  StageId stage_id_;
  bool is_last_stage_;
  std::unique_ptr<
      execution_time_predictor::BatchExecutionModel>
      execution_model_;
  std::deque<StageBatchTicket> queue_;
  BatchId active_batch_id_;
};

}  // namespace frontier::scheduler
