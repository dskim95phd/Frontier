#include "frontier/simulator/simulation_context.h"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "frontier/execution_time_predictor/batch_execution_model.h"
#include "frontier/scheduler/cluster_scheduler/co_location_cluster_scheduler.h"
#include "frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.h"

namespace frontier::simulator {
namespace {

config::ParallelismConfig resolve_parallelism(
    const config::SimulationConfig& config) {
  if (config.parallelism.has_value()) {
    return config.parallelism.value();
  }
  config::ParallelismConfig result;
  if (config.execution_model.has_value() &&
      config.execution_model->type ==
          config::ExecutionModelType::kAnalytical) {
    result.tensor_parallel_size =
        config.execution_model->analytical.tensor_parallel_size;
  }
  return result;
}

std::vector<entities::RequestBatchSnapshot> make_snapshots(
    const scheduler::ScheduleResult& schedule,
    const std::vector<entities::Request>& requests) {
  std::vector<entities::RequestBatchSnapshot> result;
  result.reserve(schedule.scheduled_requests.size());
  for (const scheduler::ScheduledRequest& scheduled :
       schedule.scheduled_requests) {
    const entities::Request& request = requests.at(
        static_cast<std::size_t>(scheduled.request_id.value()));
    result.push_back(entities::RequestBatchSnapshot{
        .request_id = scheduled.request_id,
        .scheduled_tokens = scheduled.num_tokens,
        .runtime_epoch = request.runtime_epoch(),
        .execution_epoch = request.execution_epoch(),
        .processed_tokens = request.num_processed_tokens(),
        .scheduler_frontier =
            request.scheduler_num_computed_tokens(),
    });
  }
  return result;
}

}  // namespace

SimulationContext::SimulationContext(
    const config::SimulationConfig& config,
    const std::vector<request_generator::WorkloadRequest>& workload)
    : config_(config),
      parallelism_(resolve_parallelism(config_)),
      cluster_(parallelism_),
      completion_recorded_(workload.size(), false),
      output_{
          .schema_version = config.schema_version,
          .run = metrics::RunMetadata{
              .run_id = config.run_id,
              .simulation_mode = config.simulation_mode,
              .system_architecture = config.system_architecture,
              .metrics_semantics =
                  metrics::MetricsSemantics::kCanonical,
          },
          .requests = {},
          .batches = {},
          .batch_stages = {},
          .scheduler_trace = {},
          .event_trace = {},
          .analytical_diagnostics = {},
      } {
  if (!config_.scheduler.has_value() ||
      !config_.execution_model.has_value()) {
    throw std::invalid_argument(
        "simulation context requires scheduler and execution model");
  }
  requests_.reserve(workload.size());
  for (const auto& request : workload) {
    request_generator::WorkloadRequest normalized = request;
    if (config_.simulation_mode ==
        config::SimulationMode::kOffline) {
      normalized.arrived_at = SimTime::from_seconds(0.0);
    }
    requests_.emplace_back(normalized);
  }
  request_targets_.resize(workload.size());

  std::vector<std::unique_ptr<scheduler::BaseReplicaScheduler>>
      replica_schedulers;
  replica_schedulers.reserve(static_cast<std::size_t>(
      parallelism_.num_replicas *
      parallelism_.data_parallel_size));
  for (std::uint64_t replica = 0;
       replica < parallelism_.num_replicas;
       ++replica) {
    for (std::uint64_t dp = 0;
         dp < parallelism_.data_parallel_size;
         ++dp) {
      replica_schedulers.push_back(
          std::make_unique<scheduler::VllmV1Scheduler>(
              config_.scheduler.value(),
              requests_,
              execution_time_predictor::make_batch_execution_model(
                  config_.execution_model.value(),
                  parallelism_),
              ReplicaId{replica},
              DataParallelId{dp},
              parallelism_.pipeline_parallel_size));
    }
  }
  auto cluster_scheduler =
      std::make_unique<scheduler::CoLocationClusterScheduler>(
          std::move(replica_schedulers),
          parallelism_.num_replicas,
          parallelism_.data_parallel_size);
  global_scheduler_ =
      std::make_unique<scheduler::CoLocationGlobalScheduler>(
          std::move(cluster_scheduler));

  output_.event_trace.reserve(workload.size() * 12);
  if (config_.simulation_mode ==
      config::SimulationMode::kOffline) {
    const SimTime start = SimTime::from_seconds(0.0);
    for (entities::Request& request : requests_) {
      request.on_arrival(start);
      global_scheduler_->add_request(
          request.id(),
          scheduler::ClusterType::kMonolithic);
    }
    event_queue_.push(start, EventType::kGlobalSchedule);
  } else {
    for (const auto& request : workload) {
      EventPayload payload;
      payload.request_id = request.request_id;
      event_queue_.push(
          request.arrived_at,
          EventType::kRequestArrival,
          std::move(payload));
    }
  }
}

scheduler::BaseClusterScheduler&
SimulationContext::monolithic_cluster() {
  return global_scheduler_->get_cluster_scheduler(
      scheduler::ClusterType::kMonolithic);
}

entities::Request& SimulationContext::request(
    RequestId request_id) {
  if (request_id.value() >= requests_.size()) {
    throw std::out_of_range("event references unknown request");
  }
  entities::Request& result = requests_.at(
      static_cast<std::size_t>(request_id.value()));
  if (result.id() != request_id) {
    throw std::logic_error("request arena invariant failed");
  }
  return result;
}

const entities::Request& SimulationContext::request(
    RequestId request_id) const {
  return const_cast<SimulationContext*>(this)->request(request_id);
}

entities::Batch& SimulationContext::batch(BatchId batch_id) {
  if (batch_id.value() >= batches_.size()) {
    throw std::out_of_range("event references unknown batch");
  }
  entities::Batch& result = batches_.at(
      static_cast<std::size_t>(batch_id.value()));
  if (result.id() != batch_id) {
    throw std::logic_error("batch arena invariant failed");
  }
  return result;
}

const entities::Batch& SimulationContext::batch(
    BatchId batch_id) const {
  return const_cast<SimulationContext*>(this)->batch(batch_id);
}

BatchId SimulationContext::create_batch(
    const scheduler::ScheduleResult& schedule,
    scheduler::ReplicaTarget target) {
  const BatchId batch_id{
      static_cast<BatchId::ValueType>(batches_.size())};
  const Generation generation{
      static_cast<Generation::ValueType>(batches_.size() + 1)};
  batches_.emplace_back(
      batch_id,
      schedule.iteration_id,
      make_snapshots(schedule, requests_),
      schedule.simulation_time,
      generation,
      target.replica_id,
      target.dp_id,
      parallelism_.pipeline_parallel_size);
  stage_arrival_times_.emplace_back(
      static_cast<std::size_t>(
          parallelism_.pipeline_parallel_size));
  stage_record_indices_.emplace_back(
      static_cast<std::size_t>(
          parallelism_.pipeline_parallel_size));
  predicted_batch_ms_.push_back(0.0);
  return batch_id;
}

void SimulationContext::record_stage_arrival(
    BatchId batch_id,
    StageId stage_id,
    SimTime time) {
  auto& arrivals = stage_arrival_times_.at(
      static_cast<std::size_t>(batch_id.value()));
  std::optional<SimTime>& arrival = arrivals.at(
      static_cast<std::size_t>(stage_id.value()));
  if (arrival.has_value()) {
    throw std::logic_error("batch stage arrived more than once");
  }
  arrival = time;
}

entities::BatchStage& SimulationContext::create_batch_stage(
    BatchId batch_id,
    StageId stage_id,
    SimTime started_at,
    const execution_time_predictor::BatchExecutionPrediction&
        prediction) {
  const std::size_t batch_index =
      static_cast<std::size_t>(batch_id.value());
  const std::size_t stage_index =
      static_cast<std::size_t>(stage_id.value());
  const std::optional<SimTime>& arrival =
      stage_arrival_times_.at(batch_index).at(stage_index);
  if (!arrival.has_value()) {
    throw std::logic_error("batch stage schedule has no arrival");
  }
  std::optional<std::size_t>& record_index =
      stage_record_indices_.at(batch_index).at(stage_index);
  if (record_index.has_value()) {
    throw std::logic_error("batch stage record already exists");
  }
  batch_stages_.emplace_back(
      batch_id,
      batch(batch_id).replica_id(),
      batch(batch_id).dp_id(),
      stage_id,
      arrival.value(),
      prediction.execution_time);
  record_index = batch_stages_.size() - 1;
  entities::BatchStage& stage =
      batch_stages_.back();
  stage.mark_started(started_at);
  predicted_batch_ms_.at(batch_index) +=
      prediction.duration_ms;
  return stage;
}

entities::BatchStage& SimulationContext::batch_stage(
    BatchId batch_id,
    StageId stage_id) {
  const std::optional<std::size_t>& index =
      stage_record_indices_
          .at(static_cast<std::size_t>(batch_id.value()))
          .at(static_cast<std::size_t>(stage_id.value()));
  if (!index.has_value()) {
    throw std::logic_error("batch stage record does not exist");
  }
  return batch_stages_.at(index.value());
}

double SimulationContext::predicted_batch_ms(
    BatchId batch_id) const {
  return predicted_batch_ms_.at(
      static_cast<std::size_t>(batch_id.value()));
}

void SimulationContext::assign_request_target(
    RequestId request_id,
    scheduler::ReplicaTarget target) {
  std::optional<scheduler::ReplicaTarget>& current =
      request_targets_.at(
          static_cast<std::size_t>(request_id.value()));
  if (current.has_value() && current.value() != target) {
    throw std::logic_error(
        "request was routed to more than one target");
  }
  current = target;
}

scheduler::ReplicaTarget SimulationContext::request_target(
    RequestId request_id) const {
  const auto& target = request_targets_.at(
      static_cast<std::size_t>(request_id.value()));
  if (!target.has_value()) {
    throw std::logic_error("request has no replica target");
  }
  return target.value();
}

void SimulationContext::record_request_completion(
    RequestId request_id) {
  const std::size_t index =
      static_cast<std::size_t>(request_id.value());
  if (!completion_recorded_.at(index)) {
    completion_recorded_[index] = true;
    completion_order_.push_back(request_id);
  }
}

bool SimulationContext::request_completion_recorded(
    RequestId request_id) const {
  return completion_recorded_.at(
      static_cast<std::size_t>(request_id.value()));
}

void SimulationContext::finalize() {
  if (!global_scheduler_->empty() ||
      !monolithic_cluster().empty() ||
      completion_order_.size() != requests_.size()) {
    throw std::runtime_error(
        "simulation quiesced with incomplete global or request state");
  }
  for (const auto& [replica_id, dp_id] :
       monolithic_cluster().targets()) {
    scheduler::BaseReplicaScheduler& scheduler =
        monolithic_cluster().get_replica_scheduler(
            replica_id, dp_id);
    if (!scheduler.idle()) {
      throw std::runtime_error(
          "simulation quiesced with non-idle replica target");
    }
    for (std::uint64_t stage = 0;
         stage < scheduler.pipeline_parallel_size();
         ++stage) {
      const scheduler::ReplicaStageScheduler& stage_scheduler =
          scheduler.get_replica_stage_scheduler(StageId{stage});
      if (stage_scheduler.is_busy() ||
          !stage_scheduler.empty()) {
        throw std::runtime_error(
            "simulation quiesced with nonempty stage scheduler");
      }
    }
  }
  output_.requests.reserve(completion_order_.size());
  for (const RequestId request_id : completion_order_) {
    const entities::Request& value = request(request_id);
    if (!value.completed() ||
        !value.first_scheduled_at().has_value() ||
        !value.prefill_completed_at().has_value() ||
        !value.first_token_completed_at().has_value() ||
        !value.completed_at().has_value()) {
      throw std::runtime_error(
          "completed request is missing canonical metrics");
    }
    const scheduler::ReplicaTarget target =
        request_target(request_id);
    output_.requests.push_back(metrics::RequestMetricsRecord{
        .request_id = request_id,
        .arrived_at = value.arrived_at(),
        .prefill_completed_at =
            value.prefill_completed_at().value(),
        .completed_at = value.completed_at().value(),
        .first_scheduled_at = value.first_scheduled_at(),
        .first_token_completed_at =
            value.first_token_completed_at(),
        .num_processed_tokens =
            value.num_processed_tokens(),
        .preemption_count = value.preemption_count(),
        .tokens_at_preemption = {},
        .replica_id = target.replica_id,
        .dp_id = target.dp_id,
    });
  }
}

metrics::SimulationOutput SimulationContext::take_output() {
  return std::move(output_);
}

}  // namespace frontier::simulator
