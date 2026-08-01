#include "frontier/simulator/simulator.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "frontier/events/event_dispatcher.h"
#include "frontier/execution_time_predictor/execution_time_predictor_factory.h"
#include "frontier/kv_cache_transfer/analytical_transfer.h"
#include "frontier/scheduler/global_scheduler/global_scheduler.h"

namespace frontier::simulator {
namespace {

void validate_inputs(
    const config::SimulationConfig &config,
    const std::vector<request_generator::WorkloadRequest> &workload) {
    if (config.schema_version != config::kSchemaVersion) {
        throw SimulationError(
            "scheduler requires the current config schema_version");
    }
    const bool is_pdd = config.system_architecture ==
                        config::SystemArchitecture::kPdDisaggregation;
    if (config.enable_parallel_clusters) {
        throw SimulationError(
            "scheduler runtime requires sequential cluster execution");
    }
    if (config.prefix_cache.enabled) {
        throw SimulationError(
            "scheduler runtime requires prefix caching disabled");
    }
    if ((!is_pdd && !std::holds_alternative<config::ClusterRuntimeConfig>(
                        config.runtime)) ||
        (is_pdd &&
         !std::holds_alternative<config::PddRuntimeConfig>(config.runtime))) {
        throw SimulationError(
            "system architecture has an incompatible runtime config");
    }
    for (std::size_t index = 0; index < workload.size(); ++index) {
        const request_generator::WorkloadRequest &request = workload[index];
        if (!request.request_id.valid() ||
            request.request_id.index() != index) {
            throw SimulationError(
                "workload request IDs must be contiguous and start at zero");
        }
        if (!std::isfinite(request.arrived_at.seconds()) ||
            request.arrived_at.seconds() < 0.0 ||
            request.num_prefill_tokens == 0 || request.num_decode_tokens == 0) {
            throw SimulationError("workload contains an invalid request");
        }
    }
}

} // namespace

Simulator::Simulator(
    const config::SimulationConfig &config,
    const std::vector<request_generator::WorkloadRequest> &workload)
    : config_(config), entities_(workload, config.simulation_mode),
      metrics_(config, workload.size()) {
    const bool is_pdd = config_.system_architecture ==
                        config::SystemArchitecture::kPdDisaggregation;

    if (is_pdd) {
        const config::PddClustersConfig &clusters = config_.pdd().clusters;
        kv_cache_transfer_predictor_ =
            kv_cache_transfer::make_kv_cache_transfer_predictor(
                config_.pdd().kv_cache_transfer);
        clusters_.emplace(
            ClusterType::kPrefill,
            entities::Cluster{ClusterType::kPrefill, clusters.prefill});
        clusters_.emplace(
            ClusterType::kDecode,
            entities::Cluster{ClusterType::kDecode, clusters.decode});
        entities_.add_target_domain(ClusterType::kPrefill);
        entities_.add_target_domain(ClusterType::kDecode);
        expected_decode_arrivals_ = workload.size();
    } else {
        const config::ClusterRuntimeConfig &runtime = config_.cluster();
        clusters_.emplace(ClusterType::kMonolithic,
                          entities::Cluster{ClusterType::kMonolithic, runtime});
        entities_.add_target_domain(ClusterType::kMonolithic);
    }

    for (const auto &[cluster_type, cluster] : clusters_) {
        predictors_.emplace(
            cluster_type,
            execution_time_predictor::make_execution_time_predictor(
                cluster.runtime_config().execution_model, cluster.parallelism(),
                cluster.model(), cluster.runtime_config().moe_routing,
                cluster.communication_backend()));
    }
    global_scheduler_ = std::make_unique<scheduler::GlobalScheduler>(
        clusters_, entities_.requests(), predictors_,
        kv_cache_transfer_predictor_, config_.cluster_scheduler);

    if (config_.simulation_mode == config::SimulationMode::kOffline) {
        const SimTime start = SimTime::from_seconds(0.0);
        for (entities::Request &request : entities_.requests()) {
            request.on_arrival(start);
            global_scheduler_->add_request(request.id(),
                                           is_pdd ? ClusterType::kPrefill
                                                  : ClusterType::kMonolithic);
        }
        event_queue_.push(start, [&]() {
            GlobalSchedulePayload value{};
            value.cluster_type =
                is_pdd ? ClusterType::kPrefill : ClusterType::kMonolithic;
            return value;
        }());
    } else {
        for (const auto &request : workload) {
            event_queue_.push(request.arrived_at, [&]() {
                RequestArrivalPayload value{};
                value.request_id = request.request_id;
                value.cluster_type =
                    is_pdd ? ClusterType::kPrefill : ClusterType::kMonolithic;
                return value;
            }());
        }
    }
}

scheduler::BaseClusterScheduler &Simulator::cluster(ClusterType cluster_type) {
    return global_scheduler_->get_cluster_scheduler(cluster_type);
}

const scheduler::BaseClusterScheduler &
Simulator::cluster(ClusterType cluster_type) const {
    return global_scheduler_->get_cluster_scheduler(cluster_type);
}

void Simulator::set_runtime_validation_enabled(bool enabled) {
    for (const auto &[cluster_type, cluster_entity] : clusters_) {
        const config::ParallelismConfig &parallelism =
            cluster_entity.parallelism();
        scheduler::BaseClusterScheduler &cluster_scheduler =
            cluster(cluster_type);
        for (std::uint64_t replica = 0;
             replica < parallelism.num_replicas; ++replica) {
            for (std::uint64_t dp = 0;
                 dp < parallelism.data_parallel_size; ++dp) {
                cluster_scheduler
                    .get_replica_scheduler(ReplicaId{replica},
                                           DataParallelId{dp})
                    .set_runtime_validation_enabled(enabled);
            }
        }
    }
}

const config::ParallelismConfig &
Simulator::parallelism(ClusterType cluster_type) const {
    return cluster_entity(cluster_type).parallelism();
}

const entities::Cluster &
Simulator::cluster_entity(ClusterType cluster_type) const {
    const auto position = clusters_.find(cluster_type);
    if (position == clusters_.end()) {
        throw std::out_of_range("simulation references unknown cluster entity");
    }
    return position->second;
}

const config::ExecutionModelConfig &
Simulator::execution_model(ClusterType cluster_type) const {
    return cluster_entity(cluster_type).runtime_config().execution_model;
}

const config::ClusterRuntimeConfig &
Simulator::runtime_config(ClusterType cluster_type) const {
    return cluster_entity(cluster_type).runtime_config();
}

entities::Request &Simulator::request(RequestId request_id) {
    return entities_.request(request_id);
}

const entities::Request &Simulator::request(RequestId request_id) const {
    return entities_.request(request_id);
}

entities::Batch &Simulator::batch(BatchId batch_id) {
    return entities_.batch(batch_id);
}

const entities::Batch &Simulator::batch(BatchId batch_id) const {
    return entities_.batch(batch_id);
}

BatchId Simulator::create_batch(const scheduler::ScheduleResult &schedule,
                                scheduler::ReplicaTarget target,
                                ClusterType cluster_type) {
    const BatchId batch_id =
        entities_.create_batch(schedule, target, cluster_type,
                               parallelism(cluster_type).pipeline_parallel_size,
                               runtime_config(cluster_type).model.kind);
    const std::uint64_t global_id =
        cluster(cluster_type)
            .next_batch_global_id(target.replica_id, target.dp_id);
    batch(batch_id).set_global_id(BatchGlobalId{global_id});
    return batch_id;
}

BatchId Simulator::create_moe_idle_batch(MoESyncGroupId sync_group_id,
                                         MoEParticipantId participant_id,
                                         scheduler::ReplicaTarget target,
                                         ClusterType cluster_type,
                                         std::uint64_t pipeline_parallel_size,
                                         SimTime created_at,
                                         Generation generation) {
    return entities_.create_moe_idle_batch(
        sync_group_id, participant_id, target, cluster_type,
        pipeline_parallel_size, created_at, generation);
}

void Simulator::record_stage_arrival(BatchId batch_id, StageId stage_id,
                                     SimTime time) {
    entities_.record_stage_arrival(batch_id, stage_id, time);
}

entities::BatchStage &Simulator::create_batch_stage(
    BatchId batch_id, StageId stage_id, SimTime started_at,
    const execution_time_predictor::ExecutionTimePrediction &prediction) {
    return entities_.create_batch_stage(batch_id, stage_id, started_at,
                                        prediction);
}

entities::BatchStage &Simulator::batch_stage(BatchId batch_id,
                                             StageId stage_id) {
    return entities_.batch_stage(batch_id, stage_id);
}

double Simulator::predicted_batch_ms(BatchId batch_id) const {
    return entities_.predicted_batch_ms(batch_id);
}

void Simulator::release_batch(BatchId batch_id) {
    entities_.release_batch(batch_id);
}

void Simulator::assign_request_target(RequestId request_id,
                                      scheduler::ReplicaTarget target,
                                      ClusterType cluster_type) {
    entities_.assign_request_target(request_id, target, cluster_type);
}

scheduler::ReplicaTarget
Simulator::request_target(RequestId request_id,
                          ClusterType cluster_type) const {
    return entities_.request_target(request_id, cluster_type);
}

TransferId
Simulator::create_kv_cache_transfer(RequestId request_id,
                                    BatchId source_batch_id,
                                    scheduler::ReplicaTarget source_target) {
    if (config_.system_architecture !=
            config::SystemArchitecture::kPdDisaggregation ||
        !std::holds_alternative<config::PddRuntimeConfig>(config_.runtime)) {
        throw std::logic_error("KV transfers require pd-disaggregation");
    }
    const kv_cache_transfer::TransferPrediction prediction =
        cluster(ClusterType::kPrefill)
            .predict_kv_cache_transfer(
                request(request_id).num_prefill_tokens());
    return entities_.create_kv_cache_transfer(
        request_id, source_batch_id, source_target, prediction.size_bytes,
        prediction.transfer_time_ms);
}

entities::KVCacheTransferInfo &
Simulator::kv_cache_transfer(TransferId transfer_id) {
    return entities_.kv_cache_transfer(transfer_id);
}

const entities::KVCacheTransferInfo &
Simulator::kv_cache_transfer(TransferId transfer_id) const {
    return entities_.kv_cache_transfer(transfer_id);
}

TransferId Simulator::request_transfer_id(RequestId request_id) const {
    return entities_.request_transfer_id(request_id);
}

bool Simulator::on_decode_kv_arrival() {
    if (decode_arrivals_ >= expected_decode_arrivals_) {
        throw std::logic_error("DECODE received more requests than expected");
    }
    ++decode_arrivals_;
    return config_.simulation_mode == config::SimulationMode::kOnline ||
           decode_arrivals_ == expected_decode_arrivals_;
}

void Simulator::record_request_completion(RequestId request_id) {
    entities_.record_request_completion(request_id);
}

bool Simulator::request_completion_recorded(RequestId request_id) const {
    return entities_.request_completion_recorded(request_id);
}

void Simulator::finalize() {
    if (!global_scheduler_->empty() ||
        entities_.completion_order().size() != entities_.request_count()) {
        throw std::runtime_error(
            "simulation quiesced with incomplete global or request state");
    }
    for (const auto &[cluster_type, cluster_entity] : clusters_) {
        static_cast<void>(cluster_entity);
        scheduler::BaseClusterScheduler &cluster_scheduler =
            cluster(cluster_type);
        cluster_scheduler.require_quiescent();
        if (!cluster_scheduler.empty()) {
            throw std::runtime_error(
                "simulation quiesced with nonempty cluster queue");
        }
        for (const auto &[replica_id, dp_id] : cluster_scheduler.targets()) {
            scheduler::BaseReplicaScheduler &replica_scheduler =
                cluster_scheduler.get_replica_scheduler(replica_id, dp_id);
            if (!replica_scheduler.idle()) {
                throw std::runtime_error(
                    "simulation quiesced with non-idle replica target");
            }
            for (std::uint64_t stage = 0;
                 stage < replica_scheduler.pipeline_parallel_size(); ++stage) {
                const scheduler::ReplicaStageScheduler &stage_scheduler =
                    replica_scheduler.get_replica_stage_scheduler(
                        StageId{stage});
                if (stage_scheduler.is_busy() || !stage_scheduler.empty()) {
                    throw std::runtime_error(
                        "simulation quiesced with nonempty stage scheduler");
                }
            }
        }
    }
    if (config_.system_architecture ==
        config::SystemArchitecture::kPdDisaggregation) {
        const auto &transfers = entities_.kv_cache_transfers();
        if (decode_arrivals_ != expected_decode_arrivals_ ||
            transfers.size() != entities_.request_count() ||
            std::any_of(transfers.begin(), transfers.end(),
                        [](const entities::KVCacheTransferInfo &transfer) {
                            return transfer.state() !=
                                   entities::KVCacheTransferState::kCompleted;
                        })) {
            throw std::runtime_error(
                "simulation quiesced with incomplete PDD transfer state");
        }
    }
    metrics_.collect_completed_requests(config_, entities_);
}

metrics::SimulationOutput Simulator::take_output() {
    return metrics_.take_output();
}

metrics::SimulationOutput Simulator::run() {
    const events::EventDispatcher dispatcher;
    while (!event_queue_.empty()) {
        Event event = event_queue_.pop();
        metrics_.record_event(event);
        dispatcher.dispatch(event, *this);
    }
    finalize();
    return take_output();
}

metrics::SimulationOutput run_simulation(
    const config::SimulationConfig &config,
    const std::vector<request_generator::WorkloadRequest> &workload) {
    validate_inputs(config, workload);
    request_generator::validate_workload_for_config(workload, config);
    return Simulator{config, workload}.run();
}

} // namespace frontier::simulator
