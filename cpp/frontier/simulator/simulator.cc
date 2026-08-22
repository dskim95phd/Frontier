#include "frontier/simulator/simulator.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>

#include "frontier/core/precision.h"
#include "frontier/events/event_dispatcher.h"
#include "frontier/execution_time_predictor/execution_time_predictor_factory.h"
#include "frontier/kv_cache_transfer/analytical_transfer.h"
#include "frontier/scheduler/global_scheduler/global_scheduler.h"
#include "frontier/simulator/simulator_memory_layout.h"

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
    if (config.prefill_only.has_value() && !is_pdd) {
        throw SimulationError(
            "PREFILL-only mode is supported only for pd-disaggregation");
    }
    if (config.prefill_only.has_value() &&
        (!std::isfinite(
             config.prefill_only->decode_tokens_per_second) ||
         config.prefill_only->decode_tokens_per_second <= 0.0)) {
        throw SimulationError(
            "PREFILL-only decode_tokens_per_second must be finite and "
            "positive");
    }
    if ((!is_pdd && !std::holds_alternative<config::ClusterRuntimeConfig>(
                        config.runtime)) ||
        (is_pdd &&
         !std::holds_alternative<config::PddRuntimeConfig>(config.runtime))) {
        throw SimulationError(
            "system architecture has an incompatible runtime config");
    }
    if (is_pdd &&
        config.pdd()
                .clusters.prefill.parallelism.decode_context_parallel_size !=
            1) {
        throw SimulationError(
            "PDD PREFILL decode_context_parallel_size must be 1");
    }
    for (std::size_t index = 0; index < workload.size(); ++index) {
        const request_generator::WorkloadRequest &request = workload[index];
        if (!request.request_id.valid() ||
            request.request_id.index() != index) {
            throw SimulationError(
                "workload request IDs must be contiguous and start at zero");
        }
        if (!request.think_time.valid() || request.num_prefill_tokens == 0 ||
            request.num_decode_tokens == 0) {
            throw SimulationError("workload contains an invalid request");
        }
    }
    if (config.prefix_cache.enabled) {
        const auto target_count =
            [](const config::ClusterRuntimeConfig &runtime) {
                return runtime.parallelism.num_replicas *
                       runtime.parallelism.data_parallel_size;
            };
        const std::uint64_t cache_targets =
            is_pdd ? target_count(config.pdd().clusters.prefill)
                   : target_count(config.cluster());
        const config::ClusterSchedulerType cache_scheduler_type =
            is_pdd ? config.cluster_scheduler.type_for_cluster(
                         ClusterType::kPrefill)
                   : config.cluster_scheduler.type;
        if (cache_targets > 1 &&
            cache_scheduler_type !=
                config::ClusterSchedulerType::kStickyRoundRobin &&
            cache_scheduler_type != config::ClusterSchedulerType::kCacheAware) {
            throw SimulationError(
                "multi-target session prefix caching requires "
                "a cache-affine PREFILL cluster scheduler "
                "(cluster_scheduler.prefill_type='sticky_round_robin' or "
                "'cache_aware')");
        }
    }
}

using simulator_detail::gpu_kv_physical_block_layout;
using simulator_detail::GpuKvPhysicalBlockLayout;
using simulator_detail::total_hbm_bytes_per_gpu;

std::vector<request_generator::WorkloadRequest>
prepare_workload(const config::SimulationConfig &config,
                 std::vector<request_generator::WorkloadRequest> workload,
                 const SimulatorOptions &options) {
    validate_inputs(config, workload);
    request_generator::validate_workload_for_config(workload, config);
    if (!options.observation_end_time.has_value() ||
        config.simulation_mode != config::SimulationMode::kOnline) {
        request_generator::materialize_workload_for_config_in_place(workload,
                                                                    config);
        return workload;
    }
    const SimTime horizon = options.observation_end_time.value();
    if (!horizon.valid() || horizon.seconds() <= 0.0) {
        throw SimulationError(
            "observation end time must be finite and positive");
    }

    std::unordered_set<SessionId, StrongIdHash<SessionId>> excluded_sessions;
    for (const request_generator::WorkloadRequest &request : workload) {
        if (request.session_id.valid() && request.session_start_at.valid() &&
            request.session_start_at > horizon) {
            excluded_sessions.insert(request.session_id);
        }
    }

    workload.erase(
        std::remove_if(workload.begin(), workload.end(),
                       [&](const request_generator::WorkloadRequest &request) {
                           const bool excluded_session =
                               request.session_id.valid() &&
                               excluded_sessions.find(request.session_id) !=
                                   excluded_sessions.end();
                           const bool excluded_standalone =
                               !request.session_id.valid() &&
                               request.session_start_at > horizon;
                           return excluded_session || excluded_standalone;
                       }),
        workload.end());
    request_generator::materialize_workload_for_config_in_place(workload,
                                                                config);
    return workload;
}

} // namespace

Simulator::Simulator(const config::SimulationConfig &config,
                     std::vector<request_generator::WorkloadRequest> workload,
                     SimulatorOptions options)
    : config_(config),
      observation_end_time_(config.simulation_mode ==
                                    config::SimulationMode::kOnline
                                ? options.observation_end_time
                                : std::nullopt),
      entities_(prepare_workload(config, std::move(workload), options)),
      metrics_(config, options.detailed_traces_enabled) {
    const bool is_pdd = config_.system_architecture ==
                        config::SystemArchitecture::kPdDisaggregation;

    if (is_pdd) {
        const config::PddClustersConfig &clusters = config_.pdd().clusters;
        // Config parsing validates the PREFILL/DECODE KDA snapshot contract;
        // resolve the same common dtype here so transfer accounting never
        // silently trusts PREFILL when a caller supplies an in-memory config.
        const double kda_snapshot_dtype_size_bytes =
            config::resolve_pdd_kda_snapshot_dtype_size_bytes(clusters);
        kv_cache_transfer_predictor_ =
            kv_cache_transfer::make_kv_cache_transfer_predictor(
                config_.pdd().kv_cache_transfer,
                clusters.decode.parallelism.tensor_parallel_size,
                clusters.decode.parallelism.decode_context_parallel_size,
                kda_snapshot_dtype_size_bytes);
        clusters_.emplace(
            ClusterType::kPrefill,
            entities::Cluster{ClusterType::kPrefill, clusters.prefill});
        clusters_.emplace(
            ClusterType::kDecode,
            entities::Cluster{ClusterType::kDecode, clusters.decode});
        entities_.add_target_domain(ClusterType::kPrefill);
        entities_.add_target_domain(ClusterType::kDecode);
        expected_decode_arrivals_ = entities_.request_count();
    } else {
        const config::ClusterRuntimeConfig &runtime = config_.cluster();
        clusters_.emplace(ClusterType::kMonolithic,
                          entities::Cluster{ClusterType::kMonolithic, runtime});
        entities_.add_target_domain(ClusterType::kMonolithic);
    }

    for (const auto &[cluster_type, cluster] : clusters_) {
        auto [position, inserted] = predictors_.emplace(
            cluster_type,
            execution_time_predictor::make_execution_time_predictor(
                cluster.runtime_config().execution_model, cluster.parallelism(),
                cluster.model(), cluster.runtime_config().moe_routing,
                cluster.communication_backend()));
        static_cast<void>(inserted);
        position->second->set_detailed_diagnostics_enabled(
            options.detailed_traces_enabled);
    }
    global_scheduler_ = std::make_unique<scheduler::GlobalScheduler>(
        clusters_, entities_.requests(), predictors_,
        kv_cache_transfer_predictor_, config_.cluster_scheduler,
        config_.prefix_cache,
        config_.cpu_kv_cache.enabled
            ? config::resolve_cpu_kv_cache_target(config_)
            : config::ResolvedCpuKVCacheTargetConfig{});

    session_successors_.resize(entities_.request_count());
    std::vector<bool> has_predecessor(entities_.request_count(), false);
    std::unordered_map<SessionId, RequestId, StrongIdHash<SessionId>>
        latest_by_session;
    for (const entities::Request &request : entities_.requests()) {
        if (!request.session_id().valid()) {
            continue;
        }
        const auto previous = latest_by_session.find(request.session_id());
        if (previous != latest_by_session.end()) {
            session_successors_.at(
                entities_.requests().position(previous->second)) = request.id();
            has_predecessor.at(entities_.requests().position(request.id())) =
                true;
            has_session_successors_ = true;
        }
        latest_by_session[request.session_id()] = request.id();
    }

    const SimTime simulation_start = SimTime::from_seconds(0.0);
    bool preloaded_offline_root = false;
    for (const entities::Request &request : entities_.requests()) {
        if (has_predecessor.at(entities_.requests().position(request.id()))) {
            continue;
        }
        if (config_.simulation_mode == config::SimulationMode::kOffline) {
            entities::Request &root = entities_.request(request.id());
            root.set_pending_arrival(simulation_start);
            root.on_arrival(simulation_start);
            global_scheduler_->add_request(root.id(),
                                           is_pdd ? ClusterType::kPrefill
                                                  : ClusterType::kMonolithic);
            preloaded_offline_root = true;
        } else {
            enqueue_request_arrival(request.id(), simulation_start);
        }
    }
    if (preloaded_offline_root) {
        event_queue_.push(simulation_start, [&]() {
            GlobalSchedulePayload value{};
            value.cluster_type =
                is_pdd ? ClusterType::kPrefill : ClusterType::kMonolithic;
            return value;
        }());
    }
    peak_event_queue_size_ = event_queue_.size();
}

void Simulator::enqueue_request_arrival(RequestId request_id,
                                        SimTime ready_at) {
    entities::Request &request = entities_.request(request_id);
    double arrival_seconds = ready_at.seconds();
    if (config_.simulation_mode == config::SimulationMode::kOnline) {
        if (request.session_start_at().valid()) {
            arrival_seconds = request.session_start_at().seconds();
        } else {
            arrival_seconds += request.think_time().seconds();
        }
    }
    if (!std::isfinite(arrival_seconds) || arrival_seconds < 0.0) {
        throw SimulationError("request arrival time overflowed");
    }
    const SimTime arrival = SimTime::from_seconds(arrival_seconds);
    request.set_pending_arrival(arrival);
    const bool is_pdd = config_.system_architecture ==
                        config::SystemArchitecture::kPdDisaggregation;
    event_queue_.push(arrival, [&]() {
        RequestArrivalPayload value{};
        value.request_id = request_id;
        value.cluster_type =
            is_pdd ? ClusterType::kPrefill : ClusterType::kMonolithic;
        return value;
    }());
}

void Simulator::record_gpu_kv_occupancy_for_event(const Event &event) {
    const auto sample = [&](ClusterType cluster_type, ReplicaId replica_id,
                            DataParallelId dp_id) {
        const auto &cluster_scheduler = cluster(cluster_type);
        const scheduler::BaseReplicaScheduler &replica_scheduler =
            cluster_scheduler.get_replica_scheduler(replica_id, dp_id);
        const config::ClusterRuntimeConfig &runtime =
            runtime_config(cluster_type);
        const GpuKvPhysicalBlockLayout bytes =
            gpu_kv_physical_block_layout(runtime, config_);
        metrics_.record_gpu_kv_cache_occupancy(
            event.time, replica_scheduler, bytes.max_rank_bytes,
            bytes.pipeline_bytes, total_hbm_bytes_per_gpu(runtime));
    };

    std::visit(
        [&](const auto &payload) {
            using Payload =
                std::remove_cv_t<std::remove_reference_t<decltype(payload)>>;
            if constexpr (std::is_same_v<Payload, ReplicaSchedulePayload> ||
                          std::is_same_v<Payload, ClusterBatchEndPayload> ||
                          std::is_same_v<Payload, GlobalBatchEndPayload>) {
                sample(payload.cluster_type, payload.replica_id, payload.dp_id);
            } else if constexpr (std::is_same_v<Payload,
                                                KVCacheTransferEndPayload>) {
                // The event payload addresses the decode target.  Source KV
                // is released during dispatch, so sample its actual retained
                // PREFILL owner from the transfer record.
                const entities::KVCacheTransferInfo &transfer =
                    kv_cache_transfer(payload.transfer_id);
                sample(ClusterType::kPrefill, transfer.source_replica_id(),
                       transfer.source_dp_id());
            } else if constexpr (std::is_same_v<Payload,
                                                CpuKVCacheOffloadEndPayload> ||
                                 std::is_same_v<Payload,
                                                CpuKVCacheRestoreEndPayload>) {
                // Offload completion can release the PREFILL source
                // allocation without scheduling another event on an otherwise
                // idle target.  Restore completion is sampled as well so the
                // same-time restore/schedule transition remains explicit.
                sample(payload.cluster_type, payload.replica_id, payload.dp_id);
            }
        },
        event.payload);
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
        for (std::uint64_t replica = 0; replica < parallelism.num_replicas;
             ++replica) {
            for (std::uint64_t dp = 0; dp < parallelism.data_parallel_size;
                 ++dp) {
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
    const BatchGlobalId global_id =
        cluster(cluster_type)
            .next_batch_global_id(target.replica_id, target.dp_id);
    batch(batch_id).set_global_id(global_id);
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
    const ClusterType cluster_type = batch(batch_id).cluster_type();
    predictors_.at(cluster_type)->release_batch_timing_cache(batch_id);
    metrics_.release_batch_diagnostics(batch_id);
    entities_.release_batch(batch_id);
}

void Simulator::set_detailed_traces_enabled(bool enabled) {
    metrics_.set_detailed_traces_enabled(enabled);
    for (const auto &[cluster_type, predictor] : predictors_) {
        static_cast<void>(cluster_type);
        predictor->set_detailed_diagnostics_enabled(enabled);
    }
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
    // Completion-relative session turns can only be injected after terminal
    // DECODE completion. Waiting for every PREFILL transfer in offline PDD
    // would deadlock, so chained workloads schedule each DECODE arrival
    // immediately.
    return config_.simulation_mode == config::SimulationMode::kOnline ||
           has_session_successors_ ||
           decode_arrivals_ == expected_decode_arrivals_;
}

void Simulator::record_request_completion(RequestId request_id, SimTime time) {
    entities_.record_request_completion(request_id);
    const RequestId successor =
        session_successors_.at(entities_.requests().position(request_id));
    if (!successor.valid()) {
        return;
    }
    enqueue_request_arrival(successor, time);
}

bool Simulator::request_completion_recorded(RequestId request_id) const {
    return entities_.request_completion_recorded(request_id);
}

metrics::SimulationOutput run_simulation(
    const config::SimulationConfig &config,
    const std::vector<request_generator::WorkloadRequest> &workload) {
    return Simulator{config, workload}.run();
}

} // namespace frontier::simulator
