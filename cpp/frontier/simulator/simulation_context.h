#pragma once

#include <cstddef>
#include <optional>
#include <vector>

#include "frontier/config/config.h"
#include "frontier/core/event_queue.h"
#include "frontier/entities/batch.h"
#include "frontier/entities/batch_stage.h"
#include "frontier/entities/cluster.h"
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

  BatchId create_batch(
      const scheduler::ScheduleResult& schedule,
      scheduler::ReplicaTarget target);
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
      scheduler::ReplicaTarget target);
  [[nodiscard]] scheduler::ReplicaTarget request_target(
      RequestId request_id) const;
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
  std::vector<entities::Request> requests_;
  std::vector<entities::Batch> batches_;
  std::vector<entities::BatchStage> batch_stages_;
  std::vector<std::vector<std::optional<SimTime>>>
      stage_arrival_times_;
  std::vector<std::vector<std::optional<std::size_t>>>
      stage_record_indices_;
  std::vector<double> predicted_batch_ms_;
  std::vector<std::optional<scheduler::ReplicaTarget>>
      request_targets_;
  std::vector<bool> completion_recorded_;
  std::vector<RequestId> completion_order_;
  EventQueue event_queue_;
  metrics::SimulationOutput output_;
  std::unique_ptr<scheduler::CoLocationGlobalScheduler>
      global_scheduler_;
};

}  // namespace frontier::simulator
