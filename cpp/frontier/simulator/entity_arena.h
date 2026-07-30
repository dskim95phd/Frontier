#pragma once

#include <cstddef>
#include <map>
#include <optional>
#include <vector>

#include "frontier/config/config.h"
#include "frontier/entities/batch.h"
#include "frontier/entities/batch_stage.h"
#include "frontier/entities/kv_cache_transfer_info.h"
#include "frontier/entities/request.h"
#include "frontier/execution_time_predictor/batch_execution_model.h"
#include "frontier/request_generator/workload.h"
#include "frontier/scheduler/replica_scheduler/base_replica_scheduler.h"
#include "frontier/scheduler/scheduler_types.h"

namespace frontier::simulator {

// Owns simulation entities whose stable numeric IDs are their arena indexes.
// Event orchestration stays in SimulationContext; ID validation and entity
// lifecycle bookkeeping live here.
class EntityArena {
 public:
  EntityArena(
      const std::vector<request_generator::WorkloadRequest>& workload,
      config::SimulationMode simulation_mode);

  [[nodiscard]] std::size_t request_count() const noexcept {
    return requests_.size();
  }
  [[nodiscard]] std::vector<entities::Request>& requests() noexcept {
    return requests_;
  }
  [[nodiscard]] const std::vector<entities::Request>& requests()
      const noexcept {
    return requests_;
  }
  [[nodiscard]] entities::Request& request(RequestId request_id);
  [[nodiscard]] const entities::Request& request(
      RequestId request_id) const;
  [[nodiscard]] entities::Batch& batch(BatchId batch_id);
  [[nodiscard]] const entities::Batch& batch(BatchId batch_id) const;

  BatchId create_batch(
      const scheduler::ScheduleResult& schedule,
      scheduler::ReplicaTarget target,
      ClusterType cluster_type,
      std::uint64_t pipeline_parallel_size,
      config::ModelKind model_kind);
  BatchId create_moe_idle_batch(
      MoESyncGroupId sync_group_id,
      MoEParticipantId participant_id,
      scheduler::ReplicaTarget target,
      ClusterType cluster_type,
      std::uint64_t pipeline_parallel_size,
      SimTime created_at,
      Generation generation);
  void record_stage_arrival(
      BatchId batch_id,
      StageId stage_id,
      SimTime time);
  entities::BatchStage& create_batch_stage(
      BatchId batch_id,
      StageId stage_id,
      SimTime started_at,
      const execution_time_predictor::BatchExecutionPrediction&
          prediction);
  [[nodiscard]] entities::BatchStage& batch_stage(
      BatchId batch_id,
      StageId stage_id);
  [[nodiscard]] double predicted_batch_ms(BatchId batch_id) const;

  void add_target_domain(ClusterType cluster_type);
  void assign_request_target(
      RequestId request_id,
      scheduler::ReplicaTarget target,
      ClusterType cluster_type);
  [[nodiscard]] scheduler::ReplicaTarget request_target(
      RequestId request_id,
      ClusterType cluster_type) const;

  TransferId create_kv_cache_transfer(
      RequestId request_id,
      BatchId source_batch_id,
      scheduler::ReplicaTarget source_target,
      std::uint64_t size_bytes,
      double predicted_time_ms);
  [[nodiscard]] entities::KVCacheTransferInfo& kv_cache_transfer(
      TransferId transfer_id);
  [[nodiscard]] const entities::KVCacheTransferInfo&
  kv_cache_transfer(TransferId transfer_id) const;
  [[nodiscard]] TransferId request_transfer_id(
      RequestId request_id) const;
  [[nodiscard]] const std::vector<entities::KVCacheTransferInfo>&
  kv_cache_transfers() const noexcept {
    return kv_cache_transfers_;
  }

  void record_request_completion(RequestId request_id);
  [[nodiscard]] bool request_completion_recorded(
      RequestId request_id) const;
  [[nodiscard]] const std::vector<RequestId>& completion_order()
      const noexcept {
    return completion_order_;
  }

 private:
  std::vector<entities::Request> requests_;
  std::vector<entities::Batch> batches_;
  std::vector<entities::BatchStage> batch_stages_;
  std::vector<std::vector<SimTime>> stage_arrival_times_;
  std::vector<std::vector<std::optional<std::size_t>>>
      stage_record_indices_;
  std::vector<double> predicted_batch_ms_;
  std::map<
      ClusterType,
      std::vector<scheduler::ReplicaTarget>>
      request_targets_;
  std::vector<entities::KVCacheTransferInfo> kv_cache_transfers_;
  std::vector<TransferId> request_transfer_ids_;
  std::vector<bool> completion_recorded_;
  std::vector<RequestId> completion_order_;
};

}  // namespace frontier::simulator
