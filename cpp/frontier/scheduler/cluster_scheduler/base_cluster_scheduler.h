#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include "frontier/core/event.h"
#include "frontier/core/ids.h"
#include "frontier/entities/cluster.h"
#include "frontier/kv_cache_transfer/base_kv_cache_transfer_predictor.h"
#include "frontier/scheduler/replica_scheduler/base_replica_scheduler.h"
#include "frontier/scheduler/scheduler_types.h"

namespace frontier::entities {
class Batch;
}

namespace frontier::execution_time_predictor {
struct ExecutionTimePrediction;
}

namespace frontier::simulator {
class Simulator;
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

    BaseClusterScheduler(const BaseClusterScheduler &) = delete;
    BaseClusterScheduler &operator=(const BaseClusterScheduler &) = delete;
    BaseClusterScheduler(BaseClusterScheduler &&) = delete;
    BaseClusterScheduler &operator=(BaseClusterScheduler &&) = delete;

    [[nodiscard]] ClusterType cluster_type() const noexcept {
        return cluster_->type();
    }
    [[nodiscard]] const entities::Cluster &cluster_entity() const noexcept {
        return *cluster_;
    }
    void add_request(RequestId request_id, SimTime arrived_at);
    [[nodiscard]] virtual std::vector<ClusterRequestAssignment> schedule() = 0;
    [[nodiscard]] bool empty() const noexcept { return request_queue_.empty(); }
    [[nodiscard]] std::vector<std::pair<ReplicaId, DataParallelId>>
    targets() const;
    [[nodiscard]] BaseReplicaScheduler &
    get_replica_scheduler(ReplicaId replica_id, DataParallelId dp_id);
    [[nodiscard]] const BaseReplicaScheduler &
    get_replica_scheduler(ReplicaId replica_id, DataParallelId dp_id) const;

    [[nodiscard]] std::uint64_t next_batch_global_id(ReplicaId replica_id,
                                                     DataParallelId dp_id);
    [[nodiscard]] kv_cache_transfer::TransferPrediction
    predict_kv_cache_transfer(std::uint64_t num_tokens) const;

    [[nodiscard]] bool
    requires_moe_synchronization(const entities::Batch &batch,
                                 const simulator::Simulator &simulator) const;
    void begin_moe_stage(
        entities::Batch &batch, StageId stage_id, SimTime started_at,
        const execution_time_predictor::ExecutionTimePrediction &prediction,
        simulator::Simulator &simulator);
    void on_prefill_sync(const PrefillSyncPayload &payload, SimTime time,
                         simulator::Simulator &simulator);
    void on_prefill_sync_collective(const PrefillSyncCollectivePayload &payload,
                                    SimTime time,
                                    simulator::Simulator &simulator);
    void on_decode_sync(const DecodeSyncPayload &payload, SimTime time,
                        simulator::Simulator &simulator);
    void on_decode_sync_collective(const DecodeSyncCollectivePayload &payload,
                                   SimTime time,
                                   simulator::Simulator &simulator);
    void require_quiescent() const;

  protected:
    struct QueuedRequest {
        RequestId request_id;
        SimTime arrived_at;
    };

    BaseClusterScheduler(
        const entities::Cluster &cluster,
        std::vector<entities::Request> &requests,
        execution_time_predictor::ExecutionTimePredictorPtr predictor,
        std::shared_ptr<const kv_cache_transfer::BaseKVCacheTransferPredictor>
            kv_cache_transfer_predictor);

    [[nodiscard]] std::uint64_t num_replicas() const noexcept {
        return cluster_->parallelism().num_replicas;
    }
    [[nodiscard]] std::uint64_t data_parallel_size() const noexcept {
        return cluster_->parallelism().data_parallel_size;
    }

    std::vector<QueuedRequest> request_queue_;

  private:
    enum class MoESyncPath : std::uint8_t {
        kPrefill,
        kDecode,
    };

    class MoEBarrierError : public std::runtime_error {
      public:
        using std::runtime_error::runtime_error;
    };

    struct MoEBarrierKey {
        ClusterType cluster_type;
        ReplicaId replica_id;
        StageId stage_id;
        MoESyncGroupId sync_group_id;
        LayerId layer_id;
        MoESyncPhase phase;
        Generation generation;

        friend bool operator==(const MoEBarrierKey &lhs,
                               const MoEBarrierKey &rhs) {
            return std::tie(lhs.cluster_type, lhs.replica_id, lhs.stage_id,
                            lhs.sync_group_id, lhs.layer_id, lhs.phase,
                            lhs.generation) ==
                   std::tie(rhs.cluster_type, rhs.replica_id, rhs.stage_id,
                            rhs.sync_group_id, rhs.layer_id, rhs.phase,
                            rhs.generation);
        }
        friend bool operator<(const MoEBarrierKey &lhs,
                              const MoEBarrierKey &rhs) {
            return std::tie(lhs.cluster_type, lhs.replica_id, lhs.stage_id,
                            lhs.sync_group_id, lhs.layer_id, lhs.phase,
                            lhs.generation) <
                   std::tie(rhs.cluster_type, rhs.replica_id, rhs.stage_id,
                            rhs.sync_group_id, rhs.layer_id, rhs.phase,
                            rhs.generation);
        }
    };

    struct MoEBarrierParticipant {
        MoEParticipantId participant_id;
        BatchId batch_id;
        SimTime arrival_time;
        double elapsed_component_ms = 0.0;
        bool is_idle = false;

        friend bool operator==(const MoEBarrierParticipant &lhs,
                               const MoEBarrierParticipant &rhs) {
            return std::tie(lhs.participant_id, lhs.batch_id, lhs.arrival_time,
                            lhs.elapsed_component_ms, lhs.is_idle) ==
                   std::tie(rhs.participant_id, rhs.batch_id, rhs.arrival_time,
                            rhs.elapsed_component_ms, rhs.is_idle);
        }
    };

    struct MoEBarrierReady {
        MoEBarrierKey key;
        SimTime collective_time;
    };

    class MoEBarrierCoordinator {
      public:
        [[nodiscard]] std::optional<MoEBarrierReady>
        arrive(const MoEBarrierKey &key, MoEBarrierParticipant participant,
               std::uint64_t expected_participants);
        [[nodiscard]] std::optional<MoEBarrierReady>
        compact_missing_idle(const MoEBarrierKey &key,
                             std::uint64_t expected_participants,
                             SimTime arrival_time);
        [[nodiscard]] std::vector<MoEBarrierParticipant>
        consume(const MoEBarrierKey &key);
        void require_empty() const;

      private:
        struct Entry {
            std::uint64_t expected_participants = 0;
            std::map<MoEParticipantId, MoEBarrierParticipant> participants;
            bool collective_emitted = false;
        };

        [[nodiscard]] std::optional<MoEBarrierReady>
        maybe_ready(const MoEBarrierKey &key, Entry &entry);

        std::map<MoEBarrierKey, Entry> waiting_;
        std::map<MoEBarrierKey, bool> consumed_;
    };

    struct MoEStageKey {
        BatchId batch_id;
        StageId stage_id;

        friend bool operator==(const MoEStageKey &lhs, const MoEStageKey &rhs) {
            return std::tie(lhs.batch_id, lhs.stage_id) ==
                   std::tie(rhs.batch_id, rhs.stage_id);
        }
        friend bool operator<(const MoEStageKey &lhs, const MoEStageKey &rhs) {
            return std::tie(lhs.batch_id, lhs.stage_id) <
                   std::tie(rhs.batch_id, rhs.stage_id);
        }
    };

    struct MoEGroupKey {
        ClusterType cluster_type;
        ReplicaId replica_id;
        StageId stage_id;
        MoESyncGroupId sync_group_id;
        Generation sync_generation;
        MoESyncPath path;

        friend bool operator==(const MoEGroupKey &lhs, const MoEGroupKey &rhs) {
            return std::tie(lhs.cluster_type, lhs.replica_id, lhs.stage_id,
                            lhs.sync_group_id, lhs.sync_generation, lhs.path) ==
                   std::tie(rhs.cluster_type, rhs.replica_id, rhs.stage_id,
                            rhs.sync_group_id, rhs.sync_generation, rhs.path);
        }
        friend bool operator<(const MoEGroupKey &lhs, const MoEGroupKey &rhs) {
            return std::tie(lhs.cluster_type, lhs.replica_id, lhs.stage_id,
                            lhs.sync_group_id, lhs.sync_generation, lhs.path) <
                   std::tie(rhs.cluster_type, rhs.replica_id, rhs.stage_id,
                            rhs.sync_group_id, rhs.sync_generation, rhs.path);
        }
    };

    struct MoECounterKey {
        ReplicaId replica_id;
        StageId stage_id;
        MoESyncPath path;
        DataParallelId lane_id;

        friend bool operator==(const MoECounterKey &lhs,
                               const MoECounterKey &rhs) {
            return std::tie(lhs.replica_id, lhs.stage_id, lhs.path,
                            lhs.lane_id) == std::tie(rhs.replica_id,
                                                     rhs.stage_id, rhs.path,
                                                     rhs.lane_id);
        }
        friend bool operator<(const MoECounterKey &lhs,
                              const MoECounterKey &rhs) {
            return std::tie(lhs.replica_id, lhs.stage_id, lhs.path,
                            lhs.lane_id) < std::tie(rhs.replica_id,
                                                    rhs.stage_id, rhs.path,
                                                    rhs.lane_id);
        }
    };

    struct BatchCounterKey {
        ReplicaId replica_id;
        DataParallelId dp_id;

        friend bool operator==(const BatchCounterKey &lhs,
                               const BatchCounterKey &rhs) {
            return std::tie(lhs.replica_id, lhs.dp_id) ==
                   std::tie(rhs.replica_id, rhs.dp_id);
        }
        friend bool operator<(const BatchCounterKey &lhs,
                              const BatchCounterKey &rhs) {
            return std::tie(lhs.replica_id, lhs.dp_id) <
                   std::tie(rhs.replica_id, rhs.dp_id);
        }
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
        MoESyncPath path;
        std::vector<MoEParticipantId> participants;
        std::uint64_t layers_per_stage = 0;
        std::uint64_t current_layer = 0;
        std::vector<double> pre_moe_compute_ms_by_layer;
        std::vector<double> prefill_post_attention_ms_by_layer;
        double decode_ep_communication_ms_per_layer = 0.0;
        double decode_dp_communication_ms_per_layer = 0.0;
        std::vector<std::vector<double>> decode_lane_times_ms;
        double suffix_compute_ms = 0.0;
        double lm_head_ms = 0.0;
        double pp_ms = 0.0;
        double remaining_moe_layer_wait_ms = 0.0;
        bool lazy_layer_prediction = false;
    };

    struct MoEGroupState {
        std::uint64_t expected_participants = 0;
        bool participants_initialized = false;
        SimTime initial_pre_arrival;
        std::map<MoEParticipantId, BatchId> participants;
    };

    [[nodiscard]] std::size_t target_index(ReplicaId replica_id,
                                           DataParallelId dp_id) const;
    [[nodiscard]] MoEGroupKey
    make_moe_group_key(const MoEBarrierKey &key,
                       MoESyncPath path) const noexcept;
    [[nodiscard]] MoEStageState &moe_stage_state(BatchId batch_id,
                                                 StageId stage_id);
    void enqueue_moe_arrival(const MoEStageState &state,
                             MoEParticipantId participant_id, LayerId layer_id,
                             MoESyncPhase phase, SimTime time,
                             double elapsed_component_ms,
                             simulator::Simulator &simulator);
    void enqueue_idle_moe_arrival(const MoEGroupKey &group_key,
                                  BatchId idle_batch_id,
                                  MoEParticipantId participant_id,
                                  DataParallelId dp_id, LayerId layer_id,
                                  MoESyncPhase phase, SimTime time,
                                  double elapsed_component_ms,
                                  simulator::Simulator &simulator);
    void ensure_moe_group_participants(const MoEBarrierKey &key,
                                       MoESyncPath path,
                                       MoEParticipantId arriving_participant,
                                       SimTime time,
                                       simulator::Simulator &simulator);
    [[nodiscard]] std::optional<MoEBarrierReady>
    compact_moe_group_participants(const MoEBarrierKey &key, MoESyncPath path,
                                   SimTime time,
                                   simulator::Simulator &simulator);
    void
    continue_moe_stage(const MoEBarrierKey &key, MoESyncPath path,
                       const std::vector<MoEBarrierParticipant> &participants,
                       SimTime time, simulator::Simulator &simulator);

    std::vector<std::unique_ptr<BaseReplicaScheduler>> replica_schedulers_;
    const entities::Cluster *cluster_;
    std::shared_ptr<const kv_cache_transfer::BaseKVCacheTransferPredictor>
        kv_cache_transfer_predictor_;
    MoEBarrierCoordinator moe_barrier_;
    std::map<MoEStageKey, MoEStageState> moe_stage_states_;
    std::map<MoEGroupKey, MoEGroupState> moe_group_states_;
    std::map<MoECounterKey, std::uint64_t> moe_group_counters_;
    std::map<BatchCounterKey, std::uint64_t> batch_counters_;
    std::uint64_t next_moe_sync_generation_ = 1;
};

} // namespace frontier::scheduler
