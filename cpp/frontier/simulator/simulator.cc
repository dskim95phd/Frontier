#include "frontier/simulator/simulator.h"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>

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
    if ((!is_pdd && !std::holds_alternative<config::ClusterRuntimeConfig>(
                        config.runtime)) ||
        (is_pdd &&
         !std::holds_alternative<config::PddRuntimeConfig>(config.runtime))) {
        throw SimulationError(
            "system architecture has an incompatible runtime config");
    }
    if (is_pdd && config.pdd()
                          .clusters.prefill.parallelism
                          .decode_context_parallel_size != 1) {
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
            is_pdd
                ? config.cluster_scheduler.type_for_cluster(
                      ClusterType::kPrefill)
                : config.cluster_scheduler.type;
        if (cache_targets > 1 &&
            cache_scheduler_type !=
                config::ClusterSchedulerType::kStickyRoundRobin &&
            cache_scheduler_type !=
                config::ClusterSchedulerType::kCacheAware) {
            throw SimulationError(
                "multi-target session prefix caching requires "
                "a cache-affine PREFILL cluster scheduler "
                "(cluster_scheduler.prefill_type='sticky_round_robin' or "
                "'cache_aware')");
        }
    }
}

double kv_cache_dtype_size_bytes(const config::ClusterRuntimeConfig &runtime,
                                 const config::SimulationConfig &config) {
    if (config.system_architecture ==
        config::SystemArchitecture::kPdDisaggregation) {
        return config.pdd().kv_cache_transfer.kv_cache_dtype_size_bytes;
    }
    if (runtime.execution_model.type != config::ExecutionModelType::kAnalytical) {
        // Fixed-latency configs do not carry a precision declaration.  The
        // historical simulator contract uses BF16-sized KV by default.
        return 2.0;
    }
    const std::string &precision =
        runtime.execution_model.analytical.kv_cache_precision();
    if (precision == "fp32") {
        return 4.0;
    }
    if (precision == "fp16" || precision == "bf16") {
        return 2.0;
    }
    if (precision == "fp8" || precision == "int8") {
        return 1.0;
    }
    if (precision == "fp4" || precision == "int4") {
        return 0.5;
    }
    throw SimulationError("unsupported KV-cache precision: " + precision);
}

std::optional<std::uint64_t>
total_hbm_bytes_per_gpu(const config::ClusterRuntimeConfig &runtime) {
    if (runtime.execution_model.type != config::ExecutionModelType::kAnalytical) {
        return std::nullopt;
    }
    std::string device = runtime.execution_model.analytical.device;
    std::transform(device.begin(), device.end(), device.begin(),
                   [](unsigned char value) {
                       return static_cast<char>(std::tolower(value));
                   });
    if (device.find("gb300") != std::string::npos) {
        // NVIDIA GB300 nominal HBM capacity used by the experiment contract.
        return 288'000'000'000ULL;
    }
    return std::nullopt;
}

std::vector<request_generator::WorkloadRequest> prepare_workload(
    const config::SimulationConfig &config,
    const std::vector<request_generator::WorkloadRequest> &workload) {
    validate_inputs(config, workload);
    return request_generator::materialize_workload_for_config(workload, config);
}

} // namespace

Simulator::Simulator(
    const config::SimulationConfig &config,
    const std::vector<request_generator::WorkloadRequest> &workload)
    : config_(config), entities_(prepare_workload(config, workload)),
      metrics_(config, workload.size()) {
    const bool is_pdd = config_.system_architecture ==
                        config::SystemArchitecture::kPdDisaggregation;

    if (is_pdd) {
        const config::PddClustersConfig &clusters = config_.pdd().clusters;
        kv_cache_transfer_predictor_ =
            kv_cache_transfer::make_kv_cache_transfer_predictor(
                config_.pdd().kv_cache_transfer,
                clusters.decode.parallelism.tensor_parallel_size,
                clusters.decode.parallelism.decode_context_parallel_size);
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
            session_successors_.at(previous->second.index()) = request.id();
            has_predecessor.at(request.id().index()) = true;
            has_session_successors_ = true;
        }
        latest_by_session[request.session_id()] = request.id();
    }

    const SimTime simulation_start = SimTime::from_seconds(0.0);
    bool preloaded_offline_root = false;
    for (const entities::Request &request : entities_.requests()) {
        if (has_predecessor.at(request.id().index())) {
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
        const std::uint64_t bytes_per_block =
            kv_cache_transfer::model_kv_cache_size_bytes_rank_local(
                runtime.scheduler.block_size, runtime.model,
                kv_cache_dtype_size_bytes(runtime, config_),
                runtime.parallelism.decode_context_parallel_size);
        metrics_.record_gpu_kv_cache_occupancy(
            event.time, replica_scheduler, bytes_per_block,
            total_hbm_bytes_per_gpu(runtime));
    };

    std::visit(
        [&](const auto &payload) {
            using Payload =
                std::remove_cv_t<std::remove_reference_t<decltype(payload)>>;
            if constexpr (std::is_same_v<Payload, ReplicaSchedulePayload> ||
                          std::is_same_v<Payload, ClusterBatchEndPayload> ||
                          std::is_same_v<Payload, GlobalBatchEndPayload>) {
                sample(payload.cluster_type, payload.replica_id,
                       payload.dp_id);
            } else if constexpr (std::is_same_v<Payload,
                                                KVCacheTransferEndPayload>) {
                // The event payload addresses the decode target.  Source KV
                // is released during dispatch, so sample its actual retained
                // PREFILL owner from the transfer record.
                const entities::KVCacheTransferInfo &transfer =
                    kv_cache_transfer(payload.transfer_id);
                sample(ClusterType::kPrefill,
                       transfer.source_replica_id(),
                       transfer.source_dp_id());
            } else if constexpr (
                std::is_same_v<Payload, CpuKVCacheOffloadEndPayload> ||
                std::is_same_v<Payload, CpuKVCacheRestoreEndPayload>) {
                // Offload completion can release the PREFILL source
                // allocation without scheduling another event on an otherwise
                // idle target.  Restore completion is sampled as well so the
                // same-time restore/schedule transition remains explicit.
                sample(payload.cluster_type, payload.replica_id,
                       payload.dp_id);
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
    const RequestId successor = session_successors_.at(request_id.index());
    if (!successor.valid()) {
        return;
    }
    enqueue_request_arrival(successor, time);
}

bool Simulator::request_completion_recorded(RequestId request_id) const {
    return entities_.request_completion_recorded(request_id);
}

void Simulator::finalize() {
    if (!global_scheduler_->empty() ||
        entities_.completion_order().size() != entities_.request_count()) {
        std::ostringstream detail;
        detail << " incomplete_requests=[";
        bool first = true;
        for (const entities::Request &request : entities_.requests()) {
            if (request_completion_recorded(request.id())) {
                continue;
            }
            if (!first) {
                detail << ',';
            }
            first = false;
            detail << request.id().value()
                   << ":state=" << static_cast<int>(request.state())
                   << ":processed=" << request.num_processed_tokens()
                   << ":scheduled=" << request.scheduler_num_computed_tokens();
        }
        detail << "] targets=[";
        first = true;
        for (const auto &[cluster_type, cluster_entity] : clusters_) {
            scheduler::BaseClusterScheduler &cluster_scheduler =
                cluster(cluster_type);
            for (const auto &[replica_id, dp_id] :
                 cluster_scheduler.targets()) {
                if (!first) {
                    detail << ',';
                }
                first = false;
                const scheduler::BaseReplicaScheduler &replica_scheduler =
                    cluster_scheduler.get_replica_scheduler(replica_id, dp_id);
                detail << static_cast<int>(cluster_type) << ':'
                       << replica_id.value() << ':' << dp_id.value()
                       << ":waiting=" << replica_scheduler.waiting_count()
                       << ":running=" << replica_scheduler.running_count()
                       << ":inflight="
                       << replica_scheduler.in_flight_batch_count()
                       << ":transfers="
                       << replica_scheduler.pending_kv_transfer_count();
            }
        }
        detail << ']';
        throw std::runtime_error(
            "simulation quiesced with incomplete global or request state "
            "(global_empty=" +
            std::string{global_scheduler_->empty() ? "true" : "false"} +
            ", completed=" +
            std::to_string(entities_.completion_order().size()) + "/" +
            std::to_string(entities_.request_count()) + ")" + detail.str());
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
            const config::ClusterRuntimeConfig &runtime =
                cluster_entity.runtime_config();
            const std::uint64_t bytes_per_block =
                kv_cache_transfer::model_kv_cache_size_bytes_rank_local(
                    runtime.scheduler.block_size, runtime.model,
                    kv_cache_dtype_size_bytes(runtime, config_),
                    runtime.parallelism.decode_context_parallel_size);
            metrics_.record_gpu_kv_cache_occupancy(
                last_event_time_, replica_scheduler, bytes_per_block,
                total_hbm_bytes_per_gpu(runtime), true);
            if (config_.prefix_cache.enabled &&
                cluster_type != ClusterType::kDecode) {
                metrics_.record_prefix_cache_target(
                    replica_scheduler.prefix_cache_stats(),
                    replica_scheduler.prefix_cache_diagnostics(),
                    scheduler::ReplicaTarget{replica_id, dp_id}, cluster_type,
                    cluster_entity.runtime_config().scheduler.block_size,
                    config_.prefix_cache.key_mode);
            }
            if (const auto *cpu_manager =
                    replica_scheduler.cpu_kv_cache_manager()) {
                const auto *cpu_config =
                    replica_scheduler.cpu_kv_cache_target_config();
                if (cpu_config == nullptr) {
                    throw std::runtime_error(
                        "CPU KV-cache manager has no resolved target config");
                }
                const auto diagnostics = cpu_manager->diagnostics();
                const auto offloads =
                    replica_scheduler.cpu_kv_cache_offload_operations();
                const auto restores =
                    replica_scheduler.cpu_kv_cache_restore_operations();
                const auto transfer_is_pending = [](const auto &operation) {
                    const auto state = operation.state();
                    return state ==
                               entities::CpuKVCacheTransferState::kPending ||
                           state ==
                               entities::CpuKVCacheTransferState::kInFlight;
                };
                if (diagnostics.active_reservations != 0 ||
                    diagnostics.active_restore_leases != 0 ||
                    replica_scheduler.pending_cpu_restore_count() != 0 ||
                    replica_scheduler.staged_cpu_restore_count() != 0 ||
                    std::any_of(offloads.begin(), offloads.end(),
                                transfer_is_pending) ||
                    std::any_of(restores.begin(), restores.end(),
                                transfer_is_pending)) {
                    throw std::runtime_error(
                        "simulation quiesced with nonterminal CPU KV-cache "
                        "ownership or transfer state");
                }
                metrics_.record_cpu_kv_cache_target(
                    *cpu_config, cpu_manager->stats(), diagnostics,
                    scheduler::ReplicaTarget{replica_id, dp_id}, cluster_type,
                    replica_scheduler.pending_cpu_restore_count(),
                    replica_scheduler.staged_cpu_restore_count());
                for (const auto &operation : offloads) {
                    metrics_.record_cpu_kv_cache_offload(
                        operation, cluster_type, cpu_config->bytes_per_block);
                }
                for (const auto &operation : restores) {
                    metrics_.record_cpu_kv_cache_restore(
                        operation, cluster_type, cpu_config->bytes_per_block);
                }
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
    if (entities_.live_batch_count() != 0) {
        throw std::runtime_error(
            "simulation quiesced with unreleased batch entities");
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
        last_event_time_ = event.time;
        metrics_.record_event(event);
        dispatcher.dispatch(event, *this);
        // Scheduler mutations (admission, preemption, completion, and PDD
        // transfer release) are visible only after dispatch.  Sampling here
        // keeps the occupancy stream event-driven and coalesces same-time
        // transitions in MetricsStore.
        record_gpu_kv_occupancy_for_event(event);
    }
    finalize();
    return take_output();
}

metrics::SimulationOutput Simulator::run_until(SimTime end_time) {
    if (!end_time.valid() || end_time.seconds() <= 0.0) {
        throw SimulationError("simulation end time must be finite and positive");
    }
    const events::EventDispatcher dispatcher;
    while (!event_queue_.empty() && event_queue_.top().time <= end_time) {
        Event event = event_queue_.pop();
        last_event_time_ = event.time;
        metrics_.record_event(event);
        dispatcher.dispatch(event, *this);
        record_gpu_kv_occupancy_for_event(event);
    }
    // A bounded experiment intentionally leaves future session turns and
    // possibly in-flight requests outside the observation horizon.  Export
    // only requests that reached canonical completion; do not run quiescence
    // validation or emit terminal cache diagnostics for this partial state.
    metrics_.collect_completed_requests(config_, entities_);
    return take_output();
}

metrics::SimulationOutput run_simulation(
    const config::SimulationConfig &config,
    const std::vector<request_generator::WorkloadRequest> &workload) {
    return Simulator{config, workload}.run();
}

} // namespace frontier::simulator
