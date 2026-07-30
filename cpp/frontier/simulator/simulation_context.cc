#include "frontier/simulator/simulation_context.h"

#include <cmath>
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "frontier/execution_time_predictor/batch_execution_model.h"
#include "frontier/kv_cache_transfer/analytical_transfer.h"
#include "frontier/scheduler/cluster_scheduler/round_robin_cluster_scheduler.h"
#include "frontier/scheduler/global_scheduler/global_scheduler.h"
#include "frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.h"

namespace frontier::simulator {
namespace {

std::unique_ptr<scheduler::BaseClusterScheduler>
make_cluster_scheduler(
    const entities::Cluster& cluster,
    std::vector<entities::Request>& requests) {
  const config::ClusterRuntimeConfig& runtime =
      cluster.runtime_config();
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
              cluster.execution_model(),
              cluster.replica(ReplicaId{replica}),
              DataParallelId{dp},
              cluster.type()));
    }
  }
  return std::make_unique<
      scheduler::RoundRobinClusterScheduler>(
          std::move(replica_schedulers),
          cluster);
}

}  // namespace

SimulationContext::SimulationContext(
    const config::SimulationConfig& config,
    const std::vector<request_generator::WorkloadRequest>& workload)
    : config_(config),
      entities_(workload, config.simulation_mode),
      metrics_(config, workload.size()) {
  const bool is_pdd =
      config_.system_architecture ==
      config::SystemArchitecture::kPdDisaggregation;

  std::vector<std::unique_ptr<scheduler::BaseClusterScheduler>>
      cluster_schedulers;
  if (is_pdd) {
    const config::PddClustersConfig& clusters =
        config_.pdd().clusters;
    kv_cache_transfer_predictor_ =
        kv_cache_transfer::make_kv_cache_transfer_predictor(
            config_.pdd().kv_cache_transfer);
    clusters_.emplace(
        ClusterType::kPrefill,
        entities::Cluster{
            ClusterType::kPrefill, clusters.prefill});
    clusters_.emplace(
        ClusterType::kDecode,
        entities::Cluster{
            ClusterType::kDecode, clusters.decode});
    entities_.add_target_domain(ClusterType::kPrefill);
    entities_.add_target_domain(ClusterType::kDecode);
    cluster_schedulers.push_back(make_cluster_scheduler(
        clusters_.at(ClusterType::kPrefill),
        entities_.requests()));
    cluster_schedulers.push_back(make_cluster_scheduler(
        clusters_.at(ClusterType::kDecode),
        entities_.requests()));
    expected_decode_arrivals_ = workload.size();
  } else {
    const config::ClusterRuntimeConfig& runtime = config_.cluster();
    clusters_.emplace(
        ClusterType::kMonolithic,
        entities::Cluster{
            ClusterType::kMonolithic, runtime});
    entities_.add_target_domain(ClusterType::kMonolithic);
    cluster_schedulers.push_back(make_cluster_scheduler(
        clusters_.at(ClusterType::kMonolithic),
        entities_.requests()));
  }
  global_scheduler_ =
      std::make_unique<scheduler::GlobalScheduler>(
          std::move(cluster_schedulers));

  if (config_.simulation_mode ==
      config::SimulationMode::kOffline) {
    const SimTime start = SimTime::from_seconds(0.0);
    for (entities::Request& request : entities_.requests()) {
      request.on_arrival(start);
      global_scheduler_->add_request(
          request.id(),
          is_pdd
              ? ClusterType::kPrefill
              : ClusterType::kMonolithic);
    }
    event_queue_.push(
        start,
        GlobalSchedulePayload{
            .cluster_type =
                is_pdd
                ? ClusterType::kPrefill
                : ClusterType::kMonolithic,
        });
  } else {
    for (const auto& request : workload) {
      event_queue_.push(
          request.arrived_at,
          RequestArrivalPayload{
              .request_id = request.request_id,
              .cluster_type =
                  is_pdd
                  ? ClusterType::kPrefill
                  : ClusterType::kMonolithic,
          });
    }
  }
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
  return cluster_entity(cluster_type).parallelism();
}

const entities::Cluster& SimulationContext::cluster_entity(
    ClusterType cluster_type) const {
  const auto position = clusters_.find(cluster_type);
  if (position == clusters_.end()) {
    throw std::out_of_range(
        "simulation references unknown cluster entity");
  }
  return position->second;
}

const config::ExecutionModelConfig&
SimulationContext::execution_model(
    ClusterType cluster_type) const {
  if (config_.system_architecture ==
      config::SystemArchitecture::kPdDisaggregation) {
    const config::PddClustersConfig& clusters =
        config_.pdd().clusters;
    if (cluster_type == ClusterType::kPrefill) {
      return clusters.prefill.execution_model;
    }
    if (cluster_type == ClusterType::kDecode) {
      return clusters.decode.execution_model;
    }
    throw std::out_of_range(
        "PDD execution model references unknown cluster");
  }
  if (cluster_type != ClusterType::kMonolithic) {
    throw std::out_of_range(
        "single-cluster execution model references unknown cluster");
  }
  return config_.cluster().execution_model;
}

const config::ClusterRuntimeConfig&
SimulationContext::runtime_config(
    ClusterType cluster_type) const {
  if (config_.system_architecture ==
      config::SystemArchitecture::kPdDisaggregation) {
    const config::PddClustersConfig& clusters =
        config_.pdd().clusters;
    if (cluster_type == ClusterType::kPrefill) {
      return clusters.prefill;
    }
    if (cluster_type == ClusterType::kDecode) {
      return clusters.decode;
    }
    throw std::out_of_range(
        "PDD runtime config references unknown cluster");
  }
  if (cluster_type != ClusterType::kMonolithic) {
    throw std::out_of_range(
        "single-cluster runtime config references unknown cluster");
  }
  return config_.cluster();
}

entities::Request& SimulationContext::request(
    RequestId request_id) {
  return entities_.request(request_id);
}

const entities::Request& SimulationContext::request(
    RequestId request_id) const {
  return entities_.request(request_id);
}

entities::Batch& SimulationContext::batch(BatchId batch_id) {
  return entities_.batch(batch_id);
}

const entities::Batch& SimulationContext::batch(
    BatchId batch_id) const {
  return entities_.batch(batch_id);
}

BatchId SimulationContext::create_batch(
    const scheduler::ScheduleResult& schedule,
    scheduler::ReplicaTarget target,
    ClusterType cluster_type) {
  const BatchId batch_id = entities_.create_batch(
      schedule,
      target,
      cluster_type,
      parallelism(cluster_type).pipeline_parallel_size,
      runtime_config(cluster_type).model.kind);
  const std::uint64_t global_id =
      cluster(cluster_type).next_batch_global_id(
          target.replica_id, target.dp_id);
  batch(batch_id).set_global_id(BatchGlobalId{global_id});
  return batch_id;
}

BatchId SimulationContext::create_moe_idle_batch(
    MoESyncGroupId sync_group_id,
    MoEParticipantId participant_id,
    scheduler::ReplicaTarget target,
    ClusterType cluster_type,
    std::uint64_t pipeline_parallel_size,
    SimTime created_at,
    Generation generation) {
  return entities_.create_moe_idle_batch(
      sync_group_id,
      participant_id,
      target,
      cluster_type,
      pipeline_parallel_size,
      created_at,
      generation);
}

void SimulationContext::record_stage_arrival(
    BatchId batch_id,
    StageId stage_id,
    SimTime time) {
  entities_.record_stage_arrival(batch_id, stage_id, time);
}

entities::BatchStage& SimulationContext::create_batch_stage(
    BatchId batch_id,
    StageId stage_id,
    SimTime started_at,
    const execution_time_predictor::BatchExecutionPrediction&
        prediction) {
  return entities_.create_batch_stage(
      batch_id, stage_id, started_at, prediction);
}

entities::BatchStage& SimulationContext::batch_stage(
    BatchId batch_id,
    StageId stage_id) {
  return entities_.batch_stage(batch_id, stage_id);
}

double SimulationContext::predicted_batch_ms(
    BatchId batch_id) const {
  return entities_.predicted_batch_ms(batch_id);
}

void SimulationContext::assign_request_target(
    RequestId request_id,
    scheduler::ReplicaTarget target,
    ClusterType cluster_type) {
  entities_.assign_request_target(
      request_id, target, cluster_type);
}

scheduler::ReplicaTarget SimulationContext::request_target(
    RequestId request_id,
    ClusterType cluster_type) const {
  return entities_.request_target(request_id, cluster_type);
}

TransferId SimulationContext::create_kv_cache_transfer(
    RequestId request_id,
    BatchId source_batch_id,
    scheduler::ReplicaTarget source_target) {
  if (config_.system_architecture !=
          config::SystemArchitecture::kPdDisaggregation ||
      !std::holds_alternative<config::PddRuntimeConfig>(
          config_.runtime)) {
    throw std::logic_error(
        "KV transfers require pd-disaggregation");
  }
  if (kv_cache_transfer_predictor_ == nullptr) {
    throw std::logic_error(
        "PDD runtime has no KV-cache transfer predictor");
  }
  const kv_cache_transfer::TransferPrediction prediction =
      kv_cache_transfer_predictor_->predict(
          request(request_id).num_prefill_tokens(),
          config_.pdd().clusters.prefill.model);
  return entities_.create_kv_cache_transfer(
      request_id,
      source_batch_id,
      source_target,
      prediction.size_bytes,
      prediction.transfer_time_ms);
}

entities::KVCacheTransferInfo&
SimulationContext::kv_cache_transfer(
    TransferId transfer_id) {
  return entities_.kv_cache_transfer(transfer_id);
}

const entities::KVCacheTransferInfo&
SimulationContext::kv_cache_transfer(
    TransferId transfer_id) const {
  return entities_.kv_cache_transfer(transfer_id);
}

TransferId SimulationContext::request_transfer_id(
    RequestId request_id) const {
  return entities_.request_transfer_id(request_id);
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
  entities_.record_request_completion(request_id);
}

bool SimulationContext::request_completion_recorded(
    RequestId request_id) const {
  return entities_.request_completion_recorded(request_id);
}

void SimulationContext::finalize() {
  if (!global_scheduler_->empty() ||
      entities_.completion_order().size() !=
          entities_.request_count()) {
    throw std::runtime_error(
        "simulation quiesced with incomplete global or request state");
  }
  for (const auto& [cluster_type, cluster_entity] : clusters_) {
    static_cast<void>(cluster_entity);
    scheduler::BaseClusterScheduler& cluster_scheduler =
        cluster(cluster_type);
    cluster_scheduler.require_quiescent();
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
  if (config_.system_architecture ==
      config::SystemArchitecture::kPdDisaggregation) {
    const auto& transfers = entities_.kv_cache_transfers();
    if (decode_arrivals_ != expected_decode_arrivals_ ||
        transfers.size() != entities_.request_count() ||
        std::any_of(
            transfers.begin(),
            transfers.end(),
            [](const entities::KVCacheTransferInfo& transfer) {
              return transfer.state() !=
                  entities::KVCacheTransferState::kCompleted;
            })) {
      throw std::runtime_error(
          "simulation quiesced with incomplete PDD transfer state");
    }
  }
  metrics_.collect_completed_requests(config_, entities_);
}

metrics::SimulationOutput SimulationContext::take_output() {
  return metrics_.take_output();
}

}  // namespace frontier::simulator
