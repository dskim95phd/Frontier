#include "frontier/simulator/simulation_context.h"

#include <cmath>
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "frontier/execution_time_predictor/batch_execution_model.h"
#include "frontier/kv_cache_transfer/analytical_transfer.h"
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

std::unique_ptr<scheduler::BaseClusterScheduler>
make_cluster_scheduler(
    const config::ClusterRuntimeConfig& runtime,
    std::vector<entities::Request>& requests,
    ClusterType cluster_type) {
  std::vector<std::unique_ptr<scheduler::BaseReplicaScheduler>>
      replica_schedulers;
  replica_schedulers.reserve(static_cast<std::size_t>(
      runtime.parallelism.num_replicas *
      runtime.parallelism.data_parallel_size));
  for (std::uint64_t replica = 0;
       replica < runtime.parallelism.num_replicas;
       ++replica) {
    for (std::uint64_t dp = 0;
         dp < runtime.parallelism.data_parallel_size;
         ++dp) {
      replica_schedulers.push_back(
          std::make_unique<scheduler::VllmV1Scheduler>(
              runtime.scheduler,
              requests,
              execution_time_predictor::make_batch_execution_model(
                  runtime.execution_model,
                  runtime.parallelism),
              ReplicaId{replica},
              DataParallelId{dp},
              runtime.parallelism.pipeline_parallel_size,
              cluster_type));
    }
  }
  return std::make_unique<
      scheduler::CoLocationClusterScheduler>(
          std::move(replica_schedulers),
          runtime.parallelism.num_replicas,
          runtime.parallelism.data_parallel_size,
          cluster_type);
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
          .kv_cache_transfers = {},
      } {
  const bool is_pdd =
      config_.schema_version == config::kPddSchemaVersion;
  if ((!is_pdd &&
       (!config_.scheduler.has_value() ||
        !config_.execution_model.has_value())) ||
      (is_pdd &&
       (!config_.clusters.has_value() ||
        !config_.kv_cache_transfer.has_value()))) {
    throw std::invalid_argument(
        "simulation context is missing runtime configuration");
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
  request_transfer_ids_.resize(workload.size());

  std::vector<std::unique_ptr<scheduler::BaseClusterScheduler>>
      cluster_schedulers;
  if (is_pdd) {
    const config::PddClustersConfig& clusters =
        config_.clusters.value();
    cluster_parallelism_.emplace(
        ClusterType::kPrefill,
        clusters.prefill.parallelism);
    cluster_parallelism_.emplace(
        ClusterType::kDecode,
        clusters.decode.parallelism);
    request_targets_[ClusterType::kPrefill].resize(workload.size());
    request_targets_[ClusterType::kDecode].resize(workload.size());
    cluster_schedulers.push_back(make_cluster_scheduler(
        clusters.prefill, requests_, ClusterType::kPrefill));
    cluster_schedulers.push_back(make_cluster_scheduler(
        clusters.decode, requests_, ClusterType::kDecode));
    expected_decode_arrivals_ = workload.size();
  } else {
    cluster_parallelism_.emplace(
        ClusterType::kMonolithic, parallelism_);
    request_targets_[ClusterType::kMonolithic].resize(workload.size());
    const config::ClusterRuntimeConfig runtime{
        .parallelism = parallelism_,
        .scheduler = config_.scheduler.value(),
        .execution_model = config_.execution_model.value(),
    };
    cluster_schedulers.push_back(make_cluster_scheduler(
        runtime, requests_, ClusterType::kMonolithic));
  }
  global_scheduler_ =
      std::make_unique<scheduler::CoLocationGlobalScheduler>(
          std::move(cluster_schedulers));

  output_.event_trace.reserve(workload.size() * 12);
  if (config_.simulation_mode ==
      config::SimulationMode::kOffline) {
    const SimTime start = SimTime::from_seconds(0.0);
    for (entities::Request& request : requests_) {
      request.on_arrival(start);
      global_scheduler_->add_request(
          request.id(),
          is_pdd
              ? ClusterType::kPrefill
              : ClusterType::kMonolithic);
    }
    EventPayload payload;
    payload.cluster_type =
        is_pdd ? ClusterType::kPrefill : ClusterType::kMonolithic;
    event_queue_.push(
        start, EventType::kGlobalSchedule, std::move(payload));
  } else {
    for (const auto& request : workload) {
      EventPayload payload;
      payload.request_id = request.request_id;
      payload.cluster_type =
          is_pdd ? ClusterType::kPrefill : ClusterType::kMonolithic;
      event_queue_.push(
          request.arrived_at,
          EventType::kRequestArrival,
          std::move(payload));
    }
  }
}

scheduler::BaseClusterScheduler&
SimulationContext::monolithic_cluster() {
  return cluster(ClusterType::kMonolithic);
}

scheduler::BaseClusterScheduler& SimulationContext::cluster(
    ClusterType cluster_type) {
  return global_scheduler_->get_cluster_scheduler(cluster_type);
}

const scheduler::BaseClusterScheduler& SimulationContext::cluster(
    ClusterType cluster_type) const {
  return global_scheduler_->get_cluster_scheduler(cluster_type);
}

const config::ParallelismConfig& SimulationContext::parallelism(
    ClusterType cluster_type) const {
  const auto position = cluster_parallelism_.find(cluster_type);
  if (position == cluster_parallelism_.end()) {
    throw std::out_of_range(
        "simulation references unknown cluster topology");
  }
  return position->second;
}

const config::ExecutionModelConfig&
SimulationContext::execution_model(
    ClusterType cluster_type) const {
  if (config_.schema_version == config::kPddSchemaVersion) {
    if (cluster_type == ClusterType::kPrefill) {
      return config_.clusters->prefill.execution_model;
    }
    if (cluster_type == ClusterType::kDecode) {
      return config_.clusters->decode.execution_model;
    }
    throw std::out_of_range(
        "PDD execution model references unknown cluster");
  }
  if (cluster_type != ClusterType::kMonolithic ||
      !config_.execution_model.has_value()) {
    throw std::out_of_range(
        "co-location execution model references unknown cluster");
  }
  return config_.execution_model.value();
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
    scheduler::ReplicaTarget target,
    ClusterType cluster_type) {
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
      parallelism(cluster_type).pipeline_parallel_size,
      cluster_type);
  stage_arrival_times_.emplace_back(
      static_cast<std::size_t>(
          parallelism(cluster_type).pipeline_parallel_size));
  stage_record_indices_.emplace_back(
      static_cast<std::size_t>(
          parallelism(cluster_type).pipeline_parallel_size));
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
    scheduler::ReplicaTarget target,
    ClusterType cluster_type) {
  std::optional<scheduler::ReplicaTarget>& current =
      request_targets_.at(cluster_type).at(
          static_cast<std::size_t>(request_id.value()));
  if (current.has_value() && current.value() != target) {
    throw std::logic_error(
        "request was routed to more than one target");
  }
  current = target;
}

scheduler::ReplicaTarget SimulationContext::request_target(
    RequestId request_id,
    ClusterType cluster_type) const {
  const auto& target = request_targets_.at(cluster_type).at(
      static_cast<std::size_t>(request_id.value()));
  if (!target.has_value()) {
    throw std::logic_error("request has no replica target");
  }
  return target.value();
}

TransferId SimulationContext::create_kv_cache_transfer(
    RequestId request_id,
    BatchId source_batch_id,
    scheduler::ReplicaTarget source_target) {
  if (config_.schema_version != config::kPddSchemaVersion ||
      !config_.kv_cache_transfer.has_value()) {
    throw std::logic_error(
        "KV transfers require schema v4");
  }
  const std::size_t request_index =
      static_cast<std::size_t>(request_id.value());
  if (request_transfer_ids_.at(request_index).has_value()) {
    throw std::logic_error(
        "request already owns a KV transfer");
  }
  const config::KvCacheTransferConfig& transfer =
      config_.kv_cache_transfer.value();
  const std::uint64_t size_bytes =
      frontier::kv_cache_transfer::dense_kv_cache_size_bytes(
          request(request_id).num_prefill_tokens(),
          frontier::kv_cache_transfer::DenseKvLayout{
              .num_layers = 32,
              .num_kv_heads_per_worker = 32,
              .head_dim = 128,
              .kv_factor = 2,
              .dtype_size_bytes =
                  transfer.kv_cache_dtype_size_bytes,
          });
  const frontier::kv_cache_transfer::TransferPrediction prediction =
      frontier::kv_cache_transfer::predict_transfer(
          size_bytes,
          frontier::kv_cache_transfer::TransferConfig{
              .network_bandwidth_gbps =
                  transfer.network_bandwidth_gbps,
              .network_latency_ms =
                  transfer.network_latency_ms,
              .enable_compression = false,
              .compression_ratio = 1.0,
          });
  const TransferId transfer_id{
      static_cast<TransferId::ValueType>(
          kv_cache_transfers_.size())};
  const entities::Batch& source_batch = batch(source_batch_id);
  kv_cache_transfers_.emplace_back(
      transfer_id,
      request_id,
      source_batch_id,
      source_target.replica_id,
      source_target.dp_id,
      prediction.size_bytes,
      prediction.transfer_time_ms,
      source_batch.schedule_epoch());
  request_transfer_ids_.at(request_index) = transfer_id;
  return transfer_id;
}

entities::KVCacheTransferInfo&
SimulationContext::kv_cache_transfer(
    TransferId transfer_id) {
  if (transfer_id.value() >= kv_cache_transfers_.size()) {
    throw std::out_of_range(
        "event references unknown KV transfer");
  }
  entities::KVCacheTransferInfo& transfer =
      kv_cache_transfers_.at(
          static_cast<std::size_t>(transfer_id.value()));
  if (transfer.id() != transfer_id) {
    throw std::logic_error(
        "KV transfer arena invariant failed");
  }
  return transfer;
}

const entities::KVCacheTransferInfo&
SimulationContext::kv_cache_transfer(
    TransferId transfer_id) const {
  return const_cast<SimulationContext*>(this)
      ->kv_cache_transfer(transfer_id);
}

TransferId SimulationContext::request_transfer_id(
    RequestId request_id) const {
  const auto& transfer = request_transfer_ids_.at(
      static_cast<std::size_t>(request_id.value()));
  if (!transfer.has_value()) {
    throw std::logic_error(
        "request has no KV transfer");
  }
  return transfer.value();
}

bool SimulationContext::on_decode_kv_arrival() {
  if (decode_arrivals_ >= expected_decode_arrivals_) {
    throw std::logic_error(
        "DECODE received more requests than expected");
  }
  ++decode_arrivals_;
  return config_.simulation_mode == config::SimulationMode::kOnline ||
      decode_arrivals_ == expected_decode_arrivals_;
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
      completion_order_.size() != requests_.size()) {
    throw std::runtime_error(
        "simulation quiesced with incomplete global or request state");
  }
  for (const auto& [cluster_type, topology] :
       cluster_parallelism_) {
    static_cast<void>(topology);
    scheduler::BaseClusterScheduler& cluster_scheduler =
        cluster(cluster_type);
    if (!cluster_scheduler.empty()) {
      throw std::runtime_error(
          "simulation quiesced with nonempty cluster queue");
    }
    for (const auto& [replica_id, dp_id] :
         cluster_scheduler.targets()) {
      scheduler::BaseReplicaScheduler& replica_scheduler =
          cluster_scheduler.get_replica_scheduler(
              replica_id, dp_id);
      if (!replica_scheduler.idle()) {
        throw std::runtime_error(
            "simulation quiesced with non-idle replica target");
      }
      for (std::uint64_t stage = 0;
           stage < replica_scheduler.pipeline_parallel_size();
           ++stage) {
        const scheduler::ReplicaStageScheduler& stage_scheduler =
            replica_scheduler.get_replica_stage_scheduler(
                StageId{stage});
        if (stage_scheduler.is_busy() ||
            !stage_scheduler.empty()) {
          throw std::runtime_error(
              "simulation quiesced with nonempty stage scheduler");
        }
      }
    }
  }
  if (config_.schema_version == config::kPddSchemaVersion) {
    if (decode_arrivals_ != expected_decode_arrivals_ ||
        kv_cache_transfers_.size() != requests_.size() ||
        std::any_of(
            kv_cache_transfers_.begin(),
            kv_cache_transfers_.end(),
            [](const entities::KVCacheTransferInfo& transfer) {
              return transfer.state() !=
                  entities::KVCacheTransferState::kCompleted;
            })) {
      throw std::runtime_error(
          "simulation quiesced with incomplete PDD transfer state");
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
    const bool is_pdd =
        config_.schema_version == config::kPddSchemaVersion;
    const scheduler::ReplicaTarget target = request_target(
        request_id,
        is_pdd
            ? ClusterType::kDecode
            : ClusterType::kMonolithic);
    metrics::RequestMetricsRecord record{
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
        .prefill_replica_id = std::nullopt,
        .prefill_dp_id = std::nullopt,
        .decode_replica_id = std::nullopt,
        .decode_dp_id = std::nullopt,
        .transfer_id = std::nullopt,
        .kv_cache_transfer_start_time = std::nullopt,
        .kv_cache_transfer_end_time = std::nullopt,
        .decode_arrived_at = std::nullopt,
        .kv_cache_transfer_size_bytes = 0,
    };
    if (is_pdd) {
      const scheduler::ReplicaTarget prefill =
          request_target(request_id, ClusterType::kPrefill);
      const TransferId transfer_id =
          request_transfer_id(request_id);
      record.prefill_replica_id = prefill.replica_id;
      record.prefill_dp_id = prefill.dp_id;
      record.decode_replica_id = target.replica_id;
      record.decode_dp_id = target.dp_id;
      record.transfer_id = transfer_id;
      record.kv_cache_transfer_start_time =
          value.kv_cache_transfer_start_time();
      record.kv_cache_transfer_end_time =
          value.kv_cache_transfer_end_time();
      record.decode_arrived_at = value.decode_arrived_at();
      record.kv_cache_transfer_size_bytes =
          value.kv_cache_transfer_size_bytes();
    }
    output_.requests.push_back(std::move(record));
  }
}

metrics::SimulationOutput SimulationContext::take_output() {
  return std::move(output_);
}

}  // namespace frontier::simulator
