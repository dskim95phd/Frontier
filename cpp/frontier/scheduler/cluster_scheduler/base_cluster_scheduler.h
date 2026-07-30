#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "frontier/core/event.h"
#include "frontier/core/ids.h"
#include "frontier/entities/cluster.h"
#include "frontier/moe/barrier_coordinator.h"
#include "frontier/scheduler/replica_scheduler/base_replica_scheduler.h"
#include "frontier/scheduler/scheduler_types.h"

namespace frontier::entities {
class Batch;
}

namespace frontier::execution_time_predictor {
struct BatchExecutionPrediction;
}

namespace frontier::simulator {
class SimulationContext;
}

namespace frontier::scheduler {

class ClusterSchedulerError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

struct ClusterRequestAssignment {
  ReplicaId replica_id;
  DataParallelId dp_id;
  RequestId request_id;
};

// Owns the state and behavior common to every scheduler for one cluster.
// Concrete subclasses choose only the request-to-replica routing policy.
class BaseClusterScheduler {
 public:
  virtual ~BaseClusterScheduler() = default;

  BaseClusterScheduler(const BaseClusterScheduler&) = delete;
  BaseClusterScheduler& operator=(const BaseClusterScheduler&) = delete;
  BaseClusterScheduler(BaseClusterScheduler&&) = delete;
  BaseClusterScheduler& operator=(BaseClusterScheduler&&) = delete;

  [[nodiscard]] ClusterType cluster_type() const noexcept {
    return cluster_->type();
  }
  [[nodiscard]] const entities::Cluster& cluster_entity()
      const noexcept {
    return *cluster_;
  }
  void add_request(RequestId request_id, SimTime arrived_at);
  [[nodiscard]] virtual std::vector<ClusterRequestAssignment>
  schedule() = 0;
  [[nodiscard]] bool empty() const noexcept {
    return request_queue_.empty();
  }
  [[nodiscard]] std::vector<
      std::pair<ReplicaId, DataParallelId>>
  targets() const;
  [[nodiscard]] BaseReplicaScheduler& get_replica_scheduler(
      ReplicaId replica_id,
      DataParallelId dp_id);
  [[nodiscard]] const BaseReplicaScheduler&
  get_replica_scheduler(
      ReplicaId replica_id,
      DataParallelId dp_id) const;

  [[nodiscard]] std::uint64_t next_batch_global_id(
      ReplicaId replica_id,
      DataParallelId dp_id);

  [[nodiscard]] bool requires_moe_synchronization(
      const entities::Batch& batch,
      const simulator::SimulationContext& context) const;
  void begin_moe_stage(
      entities::Batch& batch,
      StageId stage_id,
      SimTime started_at,
      const execution_time_predictor::BatchExecutionPrediction&
          prediction,
      simulator::SimulationContext& context);
  void ensure_moe_group_participants(
      const moe::BarrierKey& key,
      moe::SyncPath path,
      MoEParticipantId arriving_participant,
      SimTime time,
      simulator::SimulationContext& context);
  [[nodiscard]] std::optional<moe::BarrierReady>
  compact_moe_group_participants(
      const moe::BarrierKey& key,
      moe::SyncPath path,
      SimTime time,
      simulator::SimulationContext& context);
  void continue_moe_stage(
      const moe::BarrierKey& key,
      moe::SyncPath path,
      const std::vector<moe::BarrierParticipant>& participants,
      SimTime time,
      simulator::SimulationContext& context);
  [[nodiscard]] moe::BarrierCoordinator& moe_barrier() noexcept {
    return moe_barrier_;
  }
  void require_quiescent() const;

 protected:
  struct QueuedRequest {
    RequestId request_id;
    SimTime arrived_at;
  };

  BaseClusterScheduler(
      std::vector<std::unique_ptr<BaseReplicaScheduler>>
          replica_schedulers,
      const entities::Cluster& cluster);

  [[nodiscard]] std::uint64_t num_replicas() const noexcept {
    return cluster_->parallelism().num_replicas;
  }
  [[nodiscard]] std::uint64_t data_parallel_size() const noexcept {
    return cluster_->parallelism().data_parallel_size;
  }

  std::vector<QueuedRequest> request_queue_;

 private:
  struct MoEStageKey {
    BatchId batch_id;
    StageId stage_id;

    friend bool operator==(
        const MoEStageKey&,
        const MoEStageKey&) = default;
    friend auto operator<=>(const MoEStageKey&, const MoEStageKey&) =
        default;
  };

  struct MoEGroupKey {
    ClusterType cluster_type;
    ReplicaId replica_id;
    StageId stage_id;
    MoESyncGroupId sync_group_id;
    Generation sync_generation;
    moe::SyncPath path;

    friend bool operator==(
        const MoEGroupKey&,
        const MoEGroupKey&) = default;
    friend auto operator<=>(const MoEGroupKey&, const MoEGroupKey&) =
        default;
  };

  struct MoECounterKey {
    ReplicaId replica_id;
    StageId stage_id;
    moe::SyncPath path;
    DataParallelId lane_id;

    friend bool operator==(
        const MoECounterKey&,
        const MoECounterKey&) = default;
    friend auto operator<=>(const MoECounterKey&, const MoECounterKey&) =
        default;
  };

  struct BatchCounterKey {
    ReplicaId replica_id;
    DataParallelId dp_id;

    friend bool operator==(
        const BatchCounterKey&,
        const BatchCounterKey&) = default;
    friend auto operator<=>(const BatchCounterKey&, const BatchCounterKey&) =
        default;
  };

  struct MoEStageState {
    BatchId batch_id;
    ReplicaId replica_id;
    DataParallelId dp_id;
    StageId stage_id;
    ClusterType cluster_type;
    MoESyncGroupId sync_group_id;
    Generation batch_generation;
    Generation sync_generation;
    moe::SyncPath path;
    std::vector<MoEParticipantId> participants;
    std::uint64_t layers_per_stage = 0;
    std::uint64_t current_layer = 0;
    double attention_ms_per_layer = 0.0;
    std::vector<double> prefill_post_attention_ms_by_layer;
    double decode_ep_communication_ms_per_layer = 0.0;
    double decode_dp_communication_ms_per_layer = 0.0;
    std::vector<std::vector<double>> decode_lane_times_ms;
    double pp_ms = 0.0;
  };

  struct MoEGroupState {
    std::uint64_t expected_participants = 0;
    bool participants_initialized = false;
    SimTime initial_pre_arrival;
    std::map<MoEParticipantId, BatchId> participants;
  };

  [[nodiscard]] std::size_t target_index(
      ReplicaId replica_id,
      DataParallelId dp_id) const;
  [[nodiscard]] MoEGroupKey make_moe_group_key(
      const moe::BarrierKey& key,
      moe::SyncPath path) const noexcept;
  [[nodiscard]] MoEStageState& moe_stage_state(
      BatchId batch_id,
      StageId stage_id);
  void enqueue_moe_arrival(
      const MoEStageState& state,
      MoEParticipantId participant_id,
      LayerId layer_id,
      moe::SyncPhase phase,
      SimTime time,
      double elapsed_component_ms,
      simulator::SimulationContext& context);
  void enqueue_idle_moe_arrival(
      const MoEGroupKey& group_key,
      BatchId idle_batch_id,
      MoEParticipantId participant_id,
      DataParallelId dp_id,
      LayerId layer_id,
      moe::SyncPhase phase,
      SimTime time,
      double elapsed_component_ms,
      simulator::SimulationContext& context);

  std::vector<std::unique_ptr<BaseReplicaScheduler>>
      replica_schedulers_;
  const entities::Cluster* cluster_;
  moe::BarrierCoordinator moe_barrier_;
  std::map<MoEStageKey, MoEStageState> moe_stage_states_;
  std::map<MoEGroupKey, MoEGroupState> moe_group_states_;
  std::map<MoECounterKey, std::uint64_t> moe_group_counters_;
  std::map<BatchCounterKey, std::uint64_t> batch_counters_;
  std::uint64_t next_moe_sync_generation_ = 1;
};

}  // namespace frontier::scheduler
