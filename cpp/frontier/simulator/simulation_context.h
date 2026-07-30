#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <vector>

#include "frontier/config/config.h"
#include "frontier/core/event_queue.h"
#include "frontier/entities/cluster.h"
#include "frontier/execution_time_predictor/batch_execution_model.h"
#include "frontier/kv_cache_transfer/base_kv_cache_transfer_predictor.h"
#include "frontier/metrics/metrics_store.h"
#include "frontier/request_generator/workload.h"
#include "frontier/scheduler/global_scheduler/global_scheduler.h"
#include "frontier/scheduler/scheduler_types.h"
#include "frontier/simulator/entity_arena.h"

namespace frontier::simulator {

class SimulationContext {
 public:
  SimulationContext(
      const config::SimulationConfig& config,
      const std::vector<request_generator::WorkloadRequest>& workload);

  [[nodiscard]] const config::SimulationConfig& config() const noexcept {
    return config_;
  }
  [[nodiscard]] EventQueue& event_queue() noexcept { return event_queue_; }
  [[nodiscard]] metrics::MetricsStore& metrics() noexcept {
    return metrics_;
  }
  [[nodiscard]] scheduler::GlobalScheduler&
  global_scheduler() noexcept {
    return *global_scheduler_;
  }
  [[nodiscard]] scheduler::BaseClusterScheduler&
  cluster(ClusterType cluster_type);
  [[nodiscard]] const scheduler::BaseClusterScheduler&
  cluster(ClusterType cluster_type) const;
  [[nodiscard]] entities::Request& request(RequestId request_id);
  [[nodiscard]] const entities::Request& request(
      RequestId request_id) const;
  [[nodiscard]] entities::Batch& batch(BatchId batch_id);
  [[nodiscard]] const entities::Batch& batch(BatchId batch_id) const;
  [[nodiscard]] std::vector<entities::Request>& requests() noexcept {
    return entities_.requests();
  }
  [[nodiscard]] const std::vector<entities::Request>& requests()
      const noexcept {
    return entities_.requests();
  }
  [[nodiscard]] const config::ParallelismConfig& parallelism(
      ClusterType cluster_type) const;
  [[nodiscard]] const entities::Cluster& cluster_entity(
      ClusterType cluster_type) const;
  [[nodiscard]] const config::ExecutionModelConfig& execution_model(
      ClusterType cluster_type) const;
  [[nodiscard]] const config::ClusterRuntimeConfig& runtime_config(
      ClusterType cluster_type) const;

  BatchId create_batch(
      const scheduler::ScheduleResult& schedule,
      scheduler::ReplicaTarget target,
      ClusterType cluster_type = ClusterType::kMonolithic);
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

  void assign_request_target(
      RequestId request_id,
      scheduler::ReplicaTarget target,
      ClusterType cluster_type = ClusterType::kMonolithic);
  [[nodiscard]] scheduler::ReplicaTarget request_target(
      RequestId request_id,
      ClusterType cluster_type = ClusterType::kMonolithic) const;
  TransferId create_kv_cache_transfer(
      RequestId request_id,
      BatchId source_batch_id,
      scheduler::ReplicaTarget source_target);
  [[nodiscard]] entities::KVCacheTransferInfo& kv_cache_transfer(
      TransferId transfer_id);
  [[nodiscard]] const entities::KVCacheTransferInfo&
  kv_cache_transfer(TransferId transfer_id) const;
  [[nodiscard]] TransferId request_transfer_id(
      RequestId request_id) const;
  [[nodiscard]] bool on_decode_kv_arrival();
  void record_request_completion(RequestId request_id);
  [[nodiscard]] bool request_completion_recorded(
      RequestId request_id) const;
  [[nodiscard]] const std::vector<RequestId>& completion_order()
      const noexcept {
    return entities_.completion_order();
  }

  void finalize();
  [[nodiscard]] metrics::SimulationOutput take_output();

 private:
  config::SimulationConfig config_;
  EntityArena entities_;
  std::map<ClusterType, entities::Cluster> clusters_;
  std::size_t expected_decode_arrivals_ = 0;
  std::size_t decode_arrivals_ = 0;
  std::shared_ptr<
      const kv_cache_transfer::BaseKVCacheTransferPredictor>
      kv_cache_transfer_predictor_;
  EventQueue event_queue_;
  metrics::MetricsStore metrics_;
  std::unique_ptr<scheduler::GlobalScheduler>
      global_scheduler_;
};

}  // namespace frontier::simulator
