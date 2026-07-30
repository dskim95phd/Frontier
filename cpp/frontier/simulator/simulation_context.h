#pragma once

#include <cstddef>
#include <map>
#include <optional>
#include <vector>

#include "frontier/config/config.h"
#include "frontier/core/event_queue.h"
#include "frontier/entities/batch.h"
#include "frontier/entities/batch_stage.h"
#include "frontier/entities/cluster.h"
#include "frontier/entities/kv_cache_transfer_info.h"
#include "frontier/entities/request.h"
#include "frontier/execution_time_predictor/batch_execution_model.h"
#include "frontier/metrics/output_contract.h"
#include "frontier/request_generator/workload.h"
#include "frontier/scheduler/global_scheduler/co_location_global_scheduler.h"
#include "frontier/scheduler/scheduler_types.h"

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
  [[nodiscard]] metrics::SimulationOutput& output() noexcept {
    return output_;
  }
  [[nodiscard]] scheduler::CoLocationGlobalScheduler&
  global_scheduler() noexcept {
    return *global_scheduler_;
  }
  [[nodiscard]] scheduler::BaseClusterScheduler&
  monolithic_cluster();
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
    return requests_;
  }
  [[nodiscard]] const std::vector<entities::Request>& requests()
      const noexcept {
    return requests_;
  }
  [[nodiscard]] const config::ParallelismConfig& parallelism()
      const noexcept {
    return parallelism_;
  }
  [[nodiscard]] const config::ParallelismConfig& parallelism(
      ClusterType cluster_type) const;
  [[nodiscard]] const config::ExecutionModelConfig& execution_model(
      ClusterType cluster_type) const;

  BatchId create_batch(
      const scheduler::ScheduleResult& schedule,
      scheduler::ReplicaTarget target,
      ClusterType cluster_type = ClusterType::kMonolithic);
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
    return completion_order_;
  }

  void finalize();
  [[nodiscard]] metrics::SimulationOutput take_output();

 private:
  config::SimulationConfig config_;
  config::ParallelismConfig parallelism_;
  entities::Cluster cluster_;
  std::map<ClusterType, config::ParallelismConfig>
      cluster_parallelism_;
  std::vector<entities::Request> requests_;
  std::vector<entities::Batch> batches_;
  std::vector<entities::BatchStage> batch_stages_;
  std::vector<std::vector<std::optional<SimTime>>>
      stage_arrival_times_;
  std::vector<std::vector<std::optional<std::size_t>>>
      stage_record_indices_;
  std::vector<double> predicted_batch_ms_;
  std::map<
      ClusterType,
      std::vector<std::optional<scheduler::ReplicaTarget>>>
      request_targets_;
  std::vector<entities::KVCacheTransferInfo> kv_cache_transfers_;
  std::vector<std::optional<TransferId>> request_transfer_ids_;
  std::size_t expected_decode_arrivals_ = 0;
  std::size_t decode_arrivals_ = 0;
  std::vector<bool> completion_recorded_;
  std::vector<RequestId> completion_order_;
  EventQueue event_queue_;
  metrics::SimulationOutput output_;
  std::unique_ptr<scheduler::CoLocationGlobalScheduler>
      global_scheduler_;
};

}  // namespace frontier::simulator
