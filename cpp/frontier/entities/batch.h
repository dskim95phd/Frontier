#pragma once

#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "frontier/config/config.h"
#include "frontier/core/cluster_type.h"
#include "frontier/core/event.h"
#include "frontier/core/ids.h"
#include "frontier/entities/request.h"

namespace frontier::entities {

enum class BatchKind : std::uint8_t {
  kWork,
  kMoeIdle,
};

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
      std::uint64_t num_pipeline_stages,
      ClusterType cluster_type = ClusterType::kMonolithic,
      BatchKind kind = BatchKind::kWork,
      config::ModelKind model_kind = config::ModelKind::kDense,
      MoESyncGroupId moe_sync_group_id = MoESyncGroupId{},
      MoEParticipantId moe_participant_id = MoEParticipantId{});

  [[nodiscard]] BatchId id() const noexcept { return batch_id_; }
  [[nodiscard]] IterationId iteration_id() const noexcept {
    return iteration_id_;
  }
  [[nodiscard]] BatchGlobalId global_id() const noexcept {
    return global_id_;
  }
  [[nodiscard]] const std::vector<RequestBatchSnapshot>& requests()
      const noexcept {
    return requests_;
  }
  [[nodiscard]] SimTime scheduled_at() const noexcept {
    return scheduled_at_;
  }
  [[nodiscard]] SimTime completed_at() const noexcept {
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
  [[nodiscard]] ClusterType cluster_type() const noexcept {
    return cluster_type_;
  }
  [[nodiscard]] BatchKind kind() const noexcept { return kind_; }
  [[nodiscard]] bool is_idle() const noexcept {
    return kind_ == BatchKind::kMoeIdle;
  }
  [[nodiscard]] config::ModelKind model_kind() const noexcept {
    return model_kind_;
  }
  [[nodiscard]] bool is_moe() const noexcept {
    return model_kind_ == config::ModelKind::kMoe;
  }
  [[nodiscard]] MoESyncGroupId moe_sync_group_id() const noexcept {
    return moe_sync_group_id_;
  }
  [[nodiscard]] MoEParticipantId moe_participant_id() const noexcept {
    return moe_participant_id_;
  }
  [[nodiscard]] LayerId stage_layer() const noexcept {
    return stage_layer_;
  }
  [[nodiscard]] std::uint64_t total_scheduled_tokens() const noexcept;
  [[nodiscard]] bool completed() const noexcept {
    return completed_at_.valid();
  }

  void mark_completed(SimTime time);
  void set_global_id(BatchGlobalId global_id);
  void set_stage_layer(LayerId layer_id);
  void reset_stage_layer();
  void set_moe_synchronization(
      MoESyncGroupId sync_group_id,
      MoEParticipantId participant_id);

 private:
  BatchId batch_id_;
  IterationId iteration_id_;
  BatchGlobalId global_id_;
  std::vector<RequestBatchSnapshot> requests_;
  SimTime scheduled_at_;
  SimTime completed_at_;
  Generation schedule_epoch_;
  ReplicaId replica_id_{0};
  DataParallelId dp_id_{0};
  std::uint64_t num_pipeline_stages_ = 1;
  ClusterType cluster_type_ = ClusterType::kMonolithic;
  BatchKind kind_ = BatchKind::kWork;
  config::ModelKind model_kind_ = config::ModelKind::kDense;
  MoESyncGroupId moe_sync_group_id_;
  MoEParticipantId moe_participant_id_;
  LayerId stage_layer_;
};

}  // namespace frontier::entities
