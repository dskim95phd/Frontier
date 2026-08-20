#include "frontier/scheduler/cluster_scheduler/base_cluster_scheduler.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <utility>

#include "frontier/entities/batch.h"
#include "frontier/entities/cluster.h"
#include "frontier/entities/request.h"
#include "frontier/execution_time_predictor/base_execution_time_predictor.h"
#include "frontier/scheduler/replica_scheduler/replica_scheduler_factory.h"
#include "frontier/simulator/simulator.h"

namespace frontier::scheduler {

BaseClusterScheduler::BaseClusterScheduler(
    const entities::Cluster &cluster, entities::RequestCollection &requests,
    execution_time_predictor::ExecutionTimePredictorPtr predictor,
    std::shared_ptr<const kv_cache_transfer::BaseKVCacheTransferPredictor>
        kv_cache_transfer_predictor,
    config::PrefixCacheConfig prefix_cache_config,
    config::ResolvedCpuKVCacheTargetConfig cpu_kv_cache_config)
    : cluster_(&cluster), requests_(&requests),
      kv_cache_transfer_predictor_(std::move(kv_cache_transfer_predictor)) {
    const std::uint64_t num_replicas = cluster.parallelism().num_replicas;
    const std::uint64_t data_parallel_size =
        cluster.parallelism().data_parallel_size;
    if (predictor == nullptr) {
        throw ClusterSchedulerError(
            "cluster scheduler requires an execution predictor");
    }
    if (num_replicas == 0 || data_parallel_size == 0 ||
        num_replicas >
            std::numeric_limits<std::uint64_t>::max() / data_parallel_size) {
        throw ClusterSchedulerError("cluster target matrix is invalid");
    }
    replica_schedulers_.reserve(
        static_cast<std::size_t>(num_replicas * data_parallel_size));
    const config::ClusterRuntimeConfig &runtime = cluster.runtime_config();
    for (std::uint64_t replica = 0; replica < num_replicas; ++replica) {
        for (std::uint64_t dp = 0; dp < data_parallel_size; ++dp) {
            replica_schedulers_.push_back(make_replica_scheduler(
                runtime.scheduler, requests, predictor,
                cluster.replica(ReplicaId{replica}), DataParallelId{dp},
                cluster.type(), prefix_cache_config,
                cluster.type() == ClusterType::kPrefill
                    ? cpu_kv_cache_config
                    : config::ResolvedCpuKVCacheTargetConfig{}));
        }
    }
    for (std::uint64_t replica = 0; replica < num_replicas; ++replica) {
        for (std::uint64_t dp = 0; dp < data_parallel_size; ++dp) {
            BaseReplicaScheduler *scheduler =
                replica_schedulers_
                    .at(target_index(ReplicaId{replica}, DataParallelId{dp}))
                    .get();
            if (scheduler == nullptr ||
                scheduler->replica_id() != ReplicaId{replica} ||
                scheduler->dp_id() != DataParallelId{dp} ||
                &scheduler->replica_entity() !=
                    &cluster.replica(ReplicaId{replica})) {
                throw ClusterSchedulerError(
                    "cluster scheduler target identity mismatch");
            }
        }
    }
}

SessionId BaseClusterScheduler::request_session_id(RequestId request_id) const {
    if (!requests_->contains(request_id)) {
        throw ClusterSchedulerError(
            "cluster scheduler references an unknown request");
    }
    return requests_->at(request_id.index()).session_id();
}

const entities::Request &
BaseClusterScheduler::request(RequestId request_id) const {
    if (!requests_->contains(request_id)) {
        throw ClusterSchedulerError(
            "cluster scheduler references an unknown request");
    }
    const entities::Request &value = requests_->at(request_id.index());
    if (value.id() != request_id) {
        throw ClusterSchedulerError(
            "cluster scheduler request arena ID/index invariant failed");
    }
    return value;
}

kv_cache_transfer::TransferPrediction
BaseClusterScheduler::predict_kv_cache_transfer(
    std::uint64_t num_tokens) const {
    if (kv_cache_transfer_predictor_ == nullptr) {
        throw ClusterSchedulerError(
            "cluster scheduler has no KV-cache transfer predictor");
    }
    return kv_cache_transfer_predictor_->predict(num_tokens, cluster_->model());
}

void BaseClusterScheduler::add_request(RequestId request_id,
                                       SimTime arrived_at) {
    request_queue_.push_back([&]() {
        QueuedRequest value{};
        value.request_id = request_id;
        value.arrived_at = arrived_at;
        return value;
    }());
}

std::vector<std::pair<ReplicaId, DataParallelId>>
BaseClusterScheduler::targets() const {
    std::vector<std::pair<ReplicaId, DataParallelId>> result;
    result.reserve(replica_schedulers_.size());
    for (const auto &scheduler : replica_schedulers_) {
        result.emplace_back(scheduler->replica_id(), scheduler->dp_id());
    }
    return result;
}

std::size_t BaseClusterScheduler::target_index(ReplicaId replica_id,
                                               DataParallelId dp_id) const {
    if (!replica_id.valid() || !dp_id.valid() ||
        replica_id.index() >= num_replicas() ||
        dp_id.index() >= data_parallel_size()) {
        throw ClusterSchedulerError(
            "cluster references an unknown replica target");
    }
    return static_cast<std::size_t>(replica_id.index() * data_parallel_size() +
                                    dp_id.index());
}

BaseReplicaScheduler &
BaseClusterScheduler::get_replica_scheduler(ReplicaId replica_id,
                                            DataParallelId dp_id) {
    return *replica_schedulers_.at(target_index(replica_id, dp_id));
}

const BaseReplicaScheduler &
BaseClusterScheduler::get_replica_scheduler(ReplicaId replica_id,
                                            DataParallelId dp_id) const {
    return *replica_schedulers_.at(target_index(replica_id, dp_id));
}

BatchGlobalId BaseClusterScheduler::next_batch_global_id(ReplicaId replica_id,
                                                         DataParallelId dp_id) {
    const BatchCounterKey key = [&]() {
        BatchCounterKey value{};
        value.replica_id = replica_id;
        value.dp_id = dp_id;
        return value;
    }();
    return batch_counters_[key].next("batch global ID space exhausted");
}

bool BaseClusterScheduler::requires_moe_synchronization(
    const entities::Batch &batch, const simulator::Simulator &simulator) const {
    if (batch.cluster_type() != cluster_type()) {
        throw ClusterSchedulerError(
            "batch was sent to the wrong cluster scheduler");
    }
    const config::ClusterRuntimeConfig &runtime =
        simulator.runtime_config(cluster_type());
    if (!batch.is_moe() || !runtime.model.is_moe()) {
        return false;
    }
    return runtime.parallelism.data_parallel_size > 1 ||
           runtime.parallelism.moe_expert_parallel_size > 1;
}

BaseClusterScheduler::MoEDomainKey
BaseClusterScheduler::make_moe_domain_key(ReplicaId replica_id,
                                          StageId stage_id) const noexcept {
    return MoEDomainKey{cluster_type(), replica_id, stage_id};
}

bool BaseClusterScheduler::can_schedule_moe_stage(const entities::Batch &batch,
                                                  StageId stage_id,
                                                  SimTime time) {
    if (!time.valid()) {
        throw ClusterSchedulerError(
            "MoE stage scheduling time must be finite and nonnegative");
    }
    const auto position = moe_domain_reservations_.find(
        make_moe_domain_key(batch.replica_id(), stage_id));
    if (position == moe_domain_reservations_.end()) {
        return true;
    }
    const MoEDomainReservation &reservation = position->second;
    if (reservation.release_at.valid() && time >= reservation.release_at) {
        moe_domain_reservations_.erase(position);
        return true;
    }
    // Before the first pre-MoE arrival the group is still collecting real
    // stage registrations.  A lane that has not registered yet may join that
    // group even though the shared domain is already reserved.
    const auto group = moe_group_states_.find(reservation.group_key);
    if (group != moe_group_states_.end() &&
        !group->second.participants_initialized && batch.is_moe()) {
        const MoEParticipantId participant{batch.dp_id().value()};
        if (group->second.participants.find(participant) ==
            group->second.participants.end()) {
            return true;
        }
    }
    return false;
}

void BaseClusterScheduler::release_moe_domain_if_ready(ReplicaId replica_id,
                                                       StageId stage_id,
                                                       SimTime time) {
    if (!time.valid()) {
        throw ClusterSchedulerError(
            "MoE domain release time must be finite and nonnegative");
    }
    const auto position = moe_domain_reservations_.find(
        make_moe_domain_key(replica_id, stage_id));
    if (position != moe_domain_reservations_.end() &&
        position->second.release_at.valid() &&
        time >= position->second.release_at) {
        moe_domain_reservations_.erase(position);
    }
}

BaseClusterScheduler::MoEGroupKey
BaseClusterScheduler::make_moe_group_key(const MoEBarrierKey &key,
                                         MoESyncPath path) const noexcept {
    return [&]() {
        MoEGroupKey value{};
        value.cluster_type = key.cluster_type;
        value.replica_id = key.replica_id;
        value.stage_id = key.stage_id;
        value.sync_group_id = key.sync_group_id;
        value.sync_generation = key.generation;
        value.path = path;
        return value;
    }();
}

BaseClusterScheduler::MoEStageState &
BaseClusterScheduler::moe_stage_state(BatchId batch_id, StageId stage_id) {
    const auto position = moe_stage_states_.find([&]() {
        MoEStageKey value{};
        value.batch_id = batch_id;
        value.stage_id = stage_id;
        return value;
    }());
    if (position == moe_stage_states_.end()) {
        throw std::logic_error(
            "MoE synchronization references unknown stage state");
    }
    return position->second;
}

void BaseClusterScheduler::enqueue_moe_arrival(
    const MoEStageState &state, MoEParticipantId participant_id,
    LayerId layer_id, MoESyncPhase phase, SimTime time,
    double elapsed_component_ms, simulator::Simulator &simulator) {
    if (state.path == MoESyncPath::kPrefill) {
        simulator.event_queue().push(time, [&]() {
            PrefillSyncPayload value{};
            value.batch_id = state.batch_id;
            value.replica_id = state.replica_id;
            value.dp_id = state.dp_id;
            value.participant_id = participant_id;
            value.stage_id = state.stage_id;
            value.sync_group_id = state.sync_group_id;
            value.layer_id = layer_id;
            value.sync_phase = phase;
            value.elapsed_component_ms = elapsed_component_ms;
            value.is_idle = false;
            value.generation = state.batch_generation;
            value.sync_generation = state.sync_generation;
            value.cluster_type = state.cluster_type;
            return value;
        }());
    } else {
        simulator.event_queue().push(time, [&]() {
            DecodeSyncPayload value{};
            value.batch_id = state.batch_id;
            value.replica_id = state.replica_id;
            value.dp_id = state.dp_id;
            value.participant_id = participant_id;
            value.stage_id = state.stage_id;
            value.sync_group_id = state.sync_group_id;
            value.layer_id = layer_id;
            value.sync_phase = phase;
            value.elapsed_component_ms = elapsed_component_ms;
            value.is_idle = false;
            value.generation = state.batch_generation;
            value.sync_generation = state.sync_generation;
            value.cluster_type = state.cluster_type;
            return value;
        }());
    }
}

void BaseClusterScheduler::enqueue_idle_moe_arrival(
    const MoEGroupKey &group_key, BatchId idle_batch_id,
    MoEParticipantId participant_id, DataParallelId dp_id, LayerId layer_id,
    MoESyncPhase phase, SimTime time, double elapsed_component_ms,
    simulator::Simulator &simulator) {
    const entities::Batch &idle = simulator.batch(idle_batch_id);
    if (group_key.path == MoESyncPath::kPrefill) {
        simulator.event_queue().push(time, [&]() {
            PrefillSyncPayload value{};
            value.batch_id = idle_batch_id;
            value.replica_id = group_key.replica_id;
            value.dp_id = dp_id;
            value.participant_id = participant_id;
            value.stage_id = group_key.stage_id;
            value.sync_group_id = group_key.sync_group_id;
            value.layer_id = layer_id;
            value.sync_phase = phase;
            value.elapsed_component_ms = elapsed_component_ms;
            value.is_idle = true;
            value.generation = idle.schedule_epoch();
            value.sync_generation = group_key.sync_generation;
            value.cluster_type = group_key.cluster_type;
            return value;
        }());
    } else {
        simulator.event_queue().push(time, [&]() {
            DecodeSyncPayload value{};
            value.batch_id = idle_batch_id;
            value.replica_id = group_key.replica_id;
            value.dp_id = dp_id;
            value.participant_id = participant_id;
            value.stage_id = group_key.stage_id;
            value.sync_group_id = group_key.sync_group_id;
            value.layer_id = layer_id;
            value.sync_phase = phase;
            value.elapsed_component_ms = elapsed_component_ms;
            value.is_idle = true;
            value.generation = idle.schedule_epoch();
            value.sync_generation = group_key.sync_generation;
            value.cluster_type = group_key.cluster_type;
            return value;
        }());
    }
}

void BaseClusterScheduler::begin_moe_stage(
    entities::Batch &batch, StageId stage_id, SimTime started_at,
    const execution_time_predictor::ExecutionTimePrediction &prediction,
    simulator::Simulator &simulator) {
    if (!requires_moe_synchronization(batch, simulator)) {
        throw std::logic_error(
            "local or dense batch cannot enter MoE synchronization");
    }
    const config::ClusterRuntimeConfig &runtime =
        simulator.runtime_config(cluster_type());
    bool has_prefill = cluster_type() == ClusterType::kPrefill;
    if (cluster_type() == ClusterType::kMonolithic) {
        has_prefill = std::any_of(
            batch.requests().begin(), batch.requests().end(),
            [&simulator](const entities::RequestBatchSnapshot &snapshot) {
                return snapshot.processed_tokens <
                       simulator.request(snapshot.request_id)
                           .num_prefill_tokens();
            });
    }
    const MoESyncPath path =
        has_prefill ? MoESyncPath::kPrefill : MoESyncPath::kDecode;

    const std::vector<MoEParticipantId> participants{
        MoEParticipantId{batch.dp_id().value()}};
    const bool monolithic_decode = cluster_type() == ClusterType::kMonolithic &&
                                   path == MoESyncPath::kDecode;
    const bool aligned_decode = path == MoESyncPath::kDecode &&
                                (cluster_type() == ClusterType::kMonolithic ||
                                 cluster_type() == ClusterType::kDecode);
    // Barrier participants represent attention-DP lanes. EP lanes remain a
    // predictor output and must not be materialized as synthetic DP batches.
    const std::uint64_t expected = runtime.parallelism.data_parallel_size;
    const std::uint64_t predicted_layers = prediction.moe_routing.size();
    if (predicted_layers == 0) {
        throw std::logic_error("MoE synchronization stage has no MoE layers");
    }
    const std::uint64_t logical_moe_layers =
        prediction.logical_moe_layer_count == 0
            ? predicted_layers
            : prediction.logical_moe_layer_count;
    const bool lazy_layer_prediction = prediction.lazy_moe_layer_prediction;
    const bool scaled_layer_prediction = prediction.scaled_moe_layer_prediction;
    if (lazy_layer_prediction && scaled_layer_prediction) {
        throw std::logic_error(
            "MoE prediction cannot be lazy and scaled simultaneously");
    }
    if (logical_moe_layers < predicted_layers ||
        ((lazy_layer_prediction || scaled_layer_prediction) &&
         predicted_layers != 1) ||
        (!lazy_layer_prediction && !scaled_layer_prediction &&
         logical_moe_layers != predicted_layers)) {
        throw std::logic_error("invalid compressed MoE layer prediction");
    }
    const std::uint64_t layers_per_stage =
        lazy_layer_prediction ? logical_moe_layers : predicted_layers;
    const std::uint64_t component_layer_count =
        scaled_layer_prediction ? logical_moe_layers : predicted_layers;
    const entities::ExecutionTime &execution = prediction.execution_time;
    const auto &initial_routing = prediction.moe_routing.front();
    const double initial_pre_moe_ms =
        initial_routing.pre_moe_compute_ms +
        (path == MoESyncPath::kDecode
             ? initial_routing.pre_moe_tp_communication_ms
             : 0.0);
    const SimTime initial_pre_arrival =
        SimTime::from_seconds(started_at.seconds() + initial_pre_moe_ms * 1e-3);
    const std::uint64_t maximum_id =
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
    std::uint64_t group_value = 0;
    if (aligned_decode) {
        // DECODE groups are allocated by this scheduler, not by a
        // lane-local BatchGlobalId.  The latter is intentionally independent
        // on each DP target and therefore cannot identify one aligned
        // forward across lanes in either co-location or PDD.
        group_value = next_moe_sync_generation_;
    } else {
        group_value = static_cast<std::uint64_t>(batch.global_id().value());
    }
    if (group_value > maximum_id || next_moe_sync_generation_ > maximum_id) {
        throw std::overflow_error("MoE synchronization group ID overflows");
    }
    MoEGroupKey selected_group_key = [&]() {
        MoEGroupKey value{};
        value.cluster_type = cluster_type();
        value.replica_id = batch.replica_id();
        value.stage_id = stage_id;
        value.sync_group_id = MoESyncGroupId{group_value};
        value.sync_generation = Generation{next_moe_sync_generation_};
        value.path = path;
        return value;
    }();
    bool joined_open_group = false;
    for (auto &[candidate_key, candidate] : moe_group_states_) {
        if (candidate_key.cluster_type != selected_group_key.cluster_type ||
            candidate_key.replica_id != selected_group_key.replica_id ||
            candidate_key.stage_id != selected_group_key.stage_id ||
            candidate_key.path != selected_group_key.path ||
            candidate.participants_initialized ||
            candidate.expected_participants != expected ||
            (!aligned_decode &&
             (candidate_key.sync_group_id != selected_group_key.sync_group_id ||
              candidate.initial_pre_arrival != initial_pre_arrival)) ||
            candidate.participants.find(participants.front()) !=
                candidate.participants.end()) {
            continue;
        }
        selected_group_key = candidate_key;
        joined_open_group = true;
        break;
    }
    if (!joined_open_group) {
        moe_group_states_.emplace(selected_group_key, [&]() {
            MoEGroupState value{};
            value.expected_participants = expected;
            value.participants_initialized = false;
            value.initial_pre_arrival = initial_pre_arrival;
            value.participants = {};
            return value;
        }());
        if (aligned_decode) {
            const MoEDomainKey domain_key =
                make_moe_domain_key(batch.replica_id(), stage_id);
            const auto reservation_position =
                moe_domain_reservations_.find(domain_key);
            if (reservation_position != moe_domain_reservations_.end()) {
                if (reservation_position->second.release_at.valid() &&
                    started_at >= reservation_position->second.release_at) {
                    moe_domain_reservations_.erase(reservation_position);
                } else {
                    throw std::logic_error(
                        "DECODE MoE domain is reserved by another group");
                }
            }
            moe_domain_reservations_.emplace(
                domain_key,
                MoEDomainReservation{selected_group_key, SimTime{}});
        }
        ++next_moe_sync_generation_;
    }
    const MoESyncGroupId sync_group_id = selected_group_key.sync_group_id;
    const Generation sync_generation = selected_group_key.sync_generation;
    if (!batch.moe_sync_group_id().valid()) {
        batch.set_moe_synchronization(sync_group_id, participants.front());
    }

    batch.reset_stage_layer();
    const double post_attention_ms =
        execution.tp_communication_ms + execution.moe_gating_linear_ms +
        execution.moe_gating_routing_topk_ms + execution.moe_grouped_gemm_ms +
        execution.moe_shuffling_ms + execution.moe_post_attention_norm_ms +
        execution.moe_tp_communication_ms + execution.ep_dispatch_ms +
        execution.ep_combine_ms + execution.dp_input_communication_ms +
        execution.dp_output_communication_ms;
    std::vector<std::vector<double>> decode_lane_times_ms;
    decode_lane_times_ms.reserve(prediction.moe_routing.size());
    for (const auto &diagnostic : prediction.moe_routing) {
        if (diagnostic.lane_times_ms.size() !=
            runtime.parallelism.moe_expert_parallel_size) {
            throw std::logic_error(
                "MoE prediction lane count does not match EP domain");
        }
        decode_lane_times_ms.push_back(diagnostic.lane_times_ms);
    }
    if (decode_lane_times_ms.size() != predicted_layers) {
        throw std::logic_error(
            "MoE prediction layer count does not match pipeline stage");
    }
    double critical_lane_sum_ms = 0.0;
    for (const auto &lane_times : decode_lane_times_ms) {
        critical_lane_sum_ms +=
            *std::max_element(lane_times.begin(), lane_times.end());
    }
    if (scaled_layer_prediction) {
        critical_lane_sum_ms *= static_cast<double>(logical_moe_layers);
    }
    const double shared_post_attention_ms =
        post_attention_ms - critical_lane_sum_ms;
    if (shared_post_attention_ms < -1e-12) {
        throw std::logic_error(
            "MoE post-attention prediction is smaller than lane work");
    }
    const double shared_post_attention_ms_per_layer =
        std::max(0.0, shared_post_attention_ms) /
        static_cast<double>(component_layer_count);
    std::vector<double> prefill_post_attention_ms_by_layer;
    prefill_post_attention_ms_by_layer.reserve(
        static_cast<std::size_t>(layers_per_stage));
    for (const auto &lane_times : decode_lane_times_ms) {
        prefill_post_attention_ms_by_layer.push_back(
            shared_post_attention_ms_per_layer +
            *std::max_element(lane_times.begin(), lane_times.end()));
    }

    MoEStageState state = [&]() {
        MoEStageState value{};
        value.batch_id = batch.id();
        value.replica_id = batch.replica_id();
        value.dp_id = batch.dp_id();
        value.stage_id = stage_id;
        value.cluster_type = cluster_type();
        value.sync_group_id = sync_group_id;
        value.batch_generation = batch.schedule_epoch();
        value.sync_generation = sync_generation;
        value.path = path;
        value.participants = participants;
        value.layers_per_stage = layers_per_stage;
        value.current_layer = 0;
        value.pre_moe_compute_ms_by_layer.reserve(
            prediction.moe_routing.size());
        value.pre_moe_tp_communication_ms_by_layer.reserve(
            prediction.moe_routing.size());
        value.decode_local_ep_communication_ms_by_layer.reserve(
            prediction.moe_routing.size());
        for (const auto &diagnostic : prediction.moe_routing) {
            value.pre_moe_compute_ms_by_layer.push_back(
                diagnostic.pre_moe_compute_ms);
            value.pre_moe_tp_communication_ms_by_layer.push_back(
                diagnostic.pre_moe_tp_communication_ms);
            value.decode_local_ep_communication_ms_by_layer.push_back(
                diagnostic.exposed_ep_dispatch_ms +
                diagnostic.exposed_ep_combine_ms);
        }
        value.prefill_post_attention_ms_by_layer =
            std::move(prefill_post_attention_ms_by_layer);
        value.decode_ep_communication_ms_per_layer =
            (execution.ep_dispatch_ms + execution.ep_combine_ms) /
            static_cast<double>(component_layer_count);
        value.decode_dp_communication_ms_per_layer =
            (execution.dp_input_communication_ms +
             execution.dp_output_communication_ms) /
            static_cast<double>(component_layer_count);
        value.decode_lane_times_ms = std::move(decode_lane_times_ms);
        value.moe_routing_by_layer = prediction.moe_routing;
        value.suffix_compute_ms = prediction.moe_suffix_compute_ms;
        value.suffix_tp_communication_ms =
            prediction.moe_suffix_tp_communication_ms;
        value.lm_head_ms = execution.lm_head_ms;
        value.pp_ms = execution.pp_communication_ms;
        value.lazy_layer_prediction = lazy_layer_prediction;
        if (scaled_layer_prediction) {
            value.remaining_scaled_moe_layers =
                logical_moe_layers - predicted_layers;
            value.scaled_moe_attention_groups =
                prediction.scaled_moe_attention_groups;
            value.repeated_moe_layer_pre_compute_ms =
                prediction.repeated_moe_layer_pre_compute_ms;
            std::uint64_t grouped_attention_layers = 0;
            double remaining_attention_ms = 0.0;
            for (const auto &attention_group :
                 value.scaled_moe_attention_groups) {
                grouped_attention_layers += attention_group.layer_count;
                remaining_attention_ms +=
                    static_cast<double>(attention_group.layer_count) *
                    (attention_group.pre_moe_compute_ms_per_layer +
                     (path == MoESyncPath::kDecode
                          ? attention_group
                                .pre_moe_tp_communication_ms_per_layer
                          : 0.0));
            }
            if (grouped_attention_layers != value.remaining_scaled_moe_layers) {
                throw std::logic_error(
                    "scaled MoE attention groups do not cover the remaining "
                    "logical layers");
            }
            const double critical_lane_ms =
                *std::max_element(value.decode_lane_times_ms.front().begin(),
                                  value.decode_lane_times_ms.front().end());
            const double repeated_post_ms =
                path == MoESyncPath::kPrefill
                    ? value.prefill_post_attention_ms_by_layer.front()
                    : critical_lane_ms +
                          (monolithic_decode
                               ? 0.0
                               : value.decode_dp_communication_ms_per_layer);
            const double repeated_transition_ms =
                path == MoESyncPath::kDecode
                    ? value.decode_ep_communication_ms_per_layer
                    : 0.0;
            // Aligned decode must use the group-level repeated critical path,
            // which is known only after all real DP lanes reach the barrier.
            // Non-decode paths retain the predictor's batch-local scaling.
            if (!aligned_decode) {
                value.remaining_moe_layer_wait_ms =
                    remaining_attention_ms +
                    static_cast<double>(value.remaining_scaled_moe_layers) *
                        (repeated_post_ms + repeated_transition_ms);
            }
        }
        return value;
    }();
    const MoEStageKey stage_key = [&]() {
        MoEStageKey value{};
        value.batch_id = batch.id();
        value.stage_id = stage_id;
        return value;
    }();
    const auto [state_position, inserted] =
        moe_stage_states_.emplace(stage_key, std::move(state));
    if (!inserted) {
        throw std::logic_error("MoE stage state already exists");
    }
    MoEStageState &stored = state_position->second;
    MoEGroupState &group = moe_group_states_.at(selected_group_key);
    for (const MoEParticipantId participant : stored.participants) {
        const auto [unused, participant_inserted] =
            group.participants.emplace(participant, stored.batch_id);
        static_cast<void>(unused);
        if (!participant_inserted) {
            throw std::logic_error(
                "two real MoE batches claim one synchronization participant");
        }
        enqueue_moe_arrival(stored, participant, LayerId{0},
                            MoESyncPhase::kPreMoe,
                            SimTime::from_seconds(started_at.seconds() +
                                                  initial_pre_moe_ms * 1e-3),
                            initial_pre_moe_ms, simulator);
    }
}

void BaseClusterScheduler::ensure_moe_group_participants(
    const MoEBarrierKey &key, MoESyncPath path,
    MoEParticipantId arriving_participant, SimTime time,
    simulator::Simulator &simulator) {
    const MoEGroupKey group_key = make_moe_group_key(key, path);
    MoEGroupState &group = moe_group_states_.at(group_key);
    if (group.participants_initialized) {
        return;
    }
    group.participants_initialized = true;
    const config::ClusterRuntimeConfig &runtime =
        simulator.runtime_config(cluster_type());
    const bool compact_decode = cluster_type() == ClusterType::kMonolithic &&
                                path == MoESyncPath::kDecode;
    for (std::uint64_t value = 0; value < group.expected_participants;
         ++value) {
        const MoEParticipantId participant{value};
        if (participant == arriving_participant) {
            continue;
        }
        if (group.participants.find(participant) != group.participants.end()) {
            // A real batch may already have joined this synchronization group
            // before the first barrier arrival.  Creating an idle stand-in in
            // that case leaves an unowned batch entity behind and can also
            // enqueue a redundant barrier arrival for the same participant.
            continue;
        }
        const DataParallelId dp_id{value};
        const BatchId idle_batch_id = simulator.create_moe_idle_batch(
            group_key.sync_group_id, participant,
            [&]() {
                ReplicaTarget value{};
                value.replica_id = group_key.replica_id;
                value.dp_id = dp_id;
                return value;
            }(),
            cluster_type(), runtime.parallelism.pipeline_parallel_size, time,
            group_key.sync_generation);
        if (!group.participants.emplace(participant, idle_batch_id).second) {
            throw std::logic_error(
                "MoE idle participant was concurrently materialized");
        }
        if (!compact_decode) {
            enqueue_idle_moe_arrival(group_key, idle_batch_id, participant,
                                     dp_id, key.layer_id, key.phase, time, 0.0,
                                     simulator);
        }
    }
}

std::optional<MoEBarrierReady>
BaseClusterScheduler::compact_moe_group_participants(
    const MoEBarrierKey &key, MoESyncPath path, SimTime time,
    simulator::Simulator &simulator) {
    const MoEGroupKey group_key = make_moe_group_key(key, path);
    const MoEGroupState &group = moe_group_states_.at(group_key);
    if (cluster_type() != ClusterType::kMonolithic ||
        path != MoESyncPath::kDecode) {
        return std::nullopt;
    }
    std::optional<MoEBarrierReady> ready;
    for (const auto &[participant, batch_id] : group.participants) {
        const entities::Batch &owner = simulator.batch(batch_id);
        if (!owner.is_idle()) {
            continue;
        }
        const auto result = moe_barrier_.arrive(
            key,
            [&]() {
                MoEBarrierParticipant value{};
                value.participant_id = participant;
                value.batch_id = batch_id;
                value.arrival_time = time;
                value.elapsed_component_ms = 0.0;
                value.is_idle = true;
                return value;
            }(),
            group.expected_participants);
        if (result.has_value()) {
            ready = result;
        }
    }
    return ready;
}

void BaseClusterScheduler::continue_moe_stage(
    const MoEBarrierKey &key, MoESyncPath path,
    const std::vector<MoEBarrierParticipant> &participants, SimTime time,
    simulator::Simulator &simulator) {
    if (participants.empty()) {
        return;
    }
    const MoEGroupKey group_key = make_moe_group_key(key, path);
    const auto group_position = moe_group_states_.find(group_key);
    if (group_position == moe_group_states_.end()) {
        return;
    }
    MoEGroupState &group = group_position->second;
    const bool monolithic_decode = cluster_type() == ClusterType::kMonolithic &&
                                   path == MoESyncPath::kDecode;
    const bool aligned_decode = path == MoESyncPath::kDecode &&
                                (cluster_type() == ClusterType::kMonolithic ||
                                 cluster_type() == ClusterType::kDecode);

    if (key.phase == MoESyncPhase::kPreMoe) {
        const MoEStageState *sample_state = nullptr;
        for (const auto &[unused, batch_id] : group.participants) {
            static_cast<void>(unused);
            if (!simulator.batch(batch_id).is_idle()) {
                sample_state = &moe_stage_state(batch_id, key.stage_id);
                break;
            }
        }
        if (sample_state == nullptr) {
            throw std::logic_error(
                "MoE collective has no real stage participant");
        }
        const std::size_t layer_index =
            static_cast<std::size_t>(sample_state->current_layer);
        if (layer_index >= sample_state->decode_lane_times_ms.size()) {
            throw std::logic_error(
                "MoE collective layer index is out of range");
        }
        std::vector<double> aggregate_lane_times_ms =
            sample_state->decode_lane_times_ms.at(layer_index);
        if (aligned_decode) {
            execution_time_predictor::MoEGroupLayerInput group_input{};
            group_input.layer_id = key.layer_id;
            group_input.fallback_lane_times_ms.assign(
                aggregate_lane_times_ms.size(), 0.0);
            bool routing_initialized = false;
            std::size_t source_local_breakdown_sources = 0;
            std::vector<double> maximum_source_local_by_lane(
                aggregate_lane_times_ms.size(), 0.0);
            const auto add_checked = [](std::uint64_t &target,
                                        std::uint64_t value) {
                if (value >
                    std::numeric_limits<std::uint64_t>::max() - target) {
                    throw std::overflow_error(
                        "aligned DECODE MoE token aggregation overflows");
                }
                target += value;
            };
            for (const auto &[unused, batch_id] : group.participants) {
                static_cast<void>(unused);
                const entities::Batch &owner = simulator.batch(batch_id);
                if (owner.is_idle()) {
                    continue;
                }
                const MoEStageState &state =
                    moe_stage_state(batch_id, key.stage_id);
                if (layer_index >= state.moe_routing_by_layer.size()) {
                    throw std::logic_error(
                        "aligned DECODE MoE routing layer is missing");
                }
                const auto &routing =
                    state.moe_routing_by_layer.at(layer_index);
                if (routing.layer_id != key.layer_id ||
                    routing.lane_times_ms.size() !=
                        aggregate_lane_times_ms.size()) {
                    throw std::logic_error(
                        "aligned DECODE MoE lane domains do not match");
                }
                if (!routing_initialized) {
                    group_input.model_layer_id = routing.model_layer_id;
                    group_input.global_expert_tokens.assign(
                        routing.global_expert_tokens.size(), 0);
                    routing_initialized = true;
                } else if (group_input.model_layer_id !=
                               routing.model_layer_id ||
                           group_input.global_expert_tokens.size() !=
                               routing.global_expert_tokens.size()) {
                    throw std::logic_error(
                        "aligned DECODE MoE expert domains do not match");
                }
                add_checked(group_input.input_tokens, routing.input_tokens);
                add_checked(group_input.routed_tokens, routing.routed_tokens);
                group_input.source_input_tokens.push_back(routing.input_tokens);
                group_input.source_shared_expert_path_ms.push_back(
                    routing.shared_expert_path_ms);
                for (std::size_t expert = 0;
                     expert < routing.global_expert_tokens.size(); ++expert) {
                    add_checked(group_input.global_expert_tokens.at(expert),
                                routing.global_expert_tokens.at(expert));
                }
                for (std::size_t lane = 0;
                     lane < aggregate_lane_times_ms.size(); ++lane) {
                    group_input.fallback_lane_times_ms.at(lane) +=
                        routing.lane_times_ms.at(lane);
                }
                if (!routing.source_local_lane_times_ms.empty()) {
                    if (routing.source_local_lane_times_ms.size() !=
                        aggregate_lane_times_ms.size()) {
                        throw std::logic_error(
                            "aligned DECODE MoE source-local lane domain "
                            "does not "
                            "match");
                    }
                    for (std::size_t lane = 0;
                         lane < aggregate_lane_times_ms.size(); ++lane) {
                        maximum_source_local_by_lane.at(lane) = std::max(
                            maximum_source_local_by_lane.at(lane),
                            routing.source_local_lane_times_ms.at(lane));
                    }
                    ++source_local_breakdown_sources;
                }
                if (routing.lane_routed_tokens.size() !=
                        aggregate_lane_times_ms.size() ||
                    routing.lane_unique_tokens.size() !=
                        aggregate_lane_times_ms.size()) {
                    throw std::logic_error(
                        "aligned DECODE MoE source traffic lane domains do "
                        "not match");
                }
                group_input.source_lane_routed_tokens.push_back(
                    routing.lane_routed_tokens);
                group_input.source_lane_unique_tokens.push_back(
                    routing.lane_unique_tokens);
            }
            if (!routing_initialized) {
                throw std::logic_error(
                    "aligned DECODE MoE group has no routing allocation");
            }
            const auto group_prediction =
                get_replica_scheduler(sample_state->replica_id,
                                      sample_state->dp_id)
                    .get_replica_stage_scheduler(sample_state->stage_id)
                    .predict_moe_group_layer(group_input);
            if (group_prediction.lane_times_ms.size() !=
                aggregate_lane_times_ms.size()) {
                throw std::logic_error(
                    "group MoE prediction changed the EP lane domain");
            }
            aggregate_lane_times_ms = group_prediction.lane_times_ms;
            for (const auto &[unused, batch_id] : group.participants) {
                static_cast<void>(unused);
                if (!simulator.batch(batch_id).is_idle()) {
                    MoEStageState &state =
                        moe_stage_state(batch_id, key.stage_id);
                    if (layer_index >= state.moe_routing_by_layer.size()) {
                        throw std::logic_error(
                            "group MoE geometry layer is out of range");
                    }
                    auto &diagnostic =
                        state.moe_routing_by_layer.at(layer_index);
                    diagnostic.grouped_gemm_geometry =
                        group_prediction.grouped_gemm_geometry;
                    simulator.metrics().update_moe_grouped_gemm_geometry(
                        batch_id, key.stage_id, diagnostic.layer_id,
                        group_prediction.grouped_gemm_geometry);
                }
            }
            if (group_prediction.lane_times_are_routed_only) {
                if (source_local_breakdown_sources !=
                    group_input.source_input_tokens.size()) {
                    throw std::logic_error(
                        "routed-only MoE group prediction requires a local "
                        "source-local lane breakdown for every source");
                }
                for (std::size_t lane = 0;
                     lane < aggregate_lane_times_ms.size(); ++lane) {
                    // Preserve lane pairing before the final max reduction.
                    // max_l(R_l + max_s NR_s,l) is tighter than the former
                    // max_l(R_l) + max_s,l(NR_s,l) upper bound.
                    aggregate_lane_times_ms.at(lane) +=
                        maximum_source_local_by_lane.at(lane);
                }
            }
            if (group_prediction.has_source_aware_ep_communication) {
                if (group_prediction.destination_lane_routed_tokens.size() !=
                        aggregate_lane_times_ms.size() ||
                    group_prediction.destination_lane_unique_tokens.size() !=
                        aggregate_lane_times_ms.size()) {
                    throw std::logic_error(
                        "group MoE source-aware communication changed the EP "
                        "lane domain");
                }
                const double group_ep_communication_ms =
                    group_prediction.ep_dispatch_ms +
                    group_prediction.ep_combine_ms;
                for (const auto &[unused, batch_id] : group.participants) {
                    static_cast<void>(unused);
                    if (!simulator.batch(batch_id).is_idle()) {
                        moe_stage_state(batch_id, key.stage_id)
                            .decode_ep_communication_ms_per_layer =
                            group_ep_communication_ms;
                    }
                }
            }
        }
        const double critical_lane_ms = *std::max_element(
            aggregate_lane_times_ms.begin(), aggregate_lane_times_ms.end());
        if (aligned_decode) {
            std::optional<std::uint64_t> remaining_scaled_layers;
            std::vector<execution_time_predictor::ScaledMoEAttentionGroup>
                attention_group_layout;
            std::vector<double> maximum_pre_transition_ms_by_family;
            for (const auto &[unused, batch_id] : group.participants) {
                static_cast<void>(unused);
                if (simulator.batch(batch_id).is_idle()) {
                    continue;
                }
                MoEStageState &state = moe_stage_state(batch_id, key.stage_id);
                if (!remaining_scaled_layers.has_value()) {
                    remaining_scaled_layers = state.remaining_scaled_moe_layers;
                } else if (*remaining_scaled_layers !=
                           state.remaining_scaled_moe_layers) {
                    throw std::logic_error(
                        "aligned DECODE scaled layer counts do not match");
                }
                if (state.remaining_scaled_moe_layers == 0) {
                    continue;
                }
                if (attention_group_layout.empty()) {
                    attention_group_layout = state.scaled_moe_attention_groups;
                    maximum_pre_transition_ms_by_family.assign(
                        attention_group_layout.size(), 0.0);
                } else if (attention_group_layout.size() !=
                           state.scaled_moe_attention_groups.size()) {
                    throw std::logic_error(
                        "aligned DECODE scaled attention family counts do "
                        "not match");
                }
                for (std::size_t family = 0;
                     family < attention_group_layout.size(); ++family) {
                    const auto &expected = attention_group_layout.at(family);
                    const auto &local =
                        state.scaled_moe_attention_groups.at(family);
                    if (expected.family != local.family ||
                        expected.layer_count != local.layer_count) {
                        throw std::logic_error(
                            "aligned DECODE scaled attention family layout "
                            "does not match");
                    }
                    maximum_pre_transition_ms_by_family.at(family) = std::max(
                        maximum_pre_transition_ms_by_family.at(family),
                        local.pre_moe_compute_ms_per_layer +
                            local.pre_moe_tp_communication_ms_per_layer +
                            state.decode_ep_communication_ms_per_layer);
                }
            }

            double maximum_pre_transition_ms = 0.0;
            for (const MoEBarrierParticipant &participant : participants) {
                if (!participant.is_idle) {
                    maximum_pre_transition_ms =
                        std::max(maximum_pre_transition_ms,
                                 participant.elapsed_component_ms);
                }
            }
            for (const MoEBarrierParticipant &participant : participants) {
                if (participant.is_idle) {
                    continue;
                }
                MoEStageState &state =
                    moe_stage_state(participant.batch_id, key.stage_id);
                const auto &local_lane_times =
                    state.decode_lane_times_ms.at(layer_index);
                const double local_critical_lane_ms = *std::max_element(
                    local_lane_times.begin(), local_lane_times.end());
                entities::ExecutionTime synchronization_breakdown{};
                synchronization_breakdown.moe_pre_barrier_wait_ms =
                    std::max(0.0, maximum_pre_transition_ms -
                                      participant.elapsed_component_ms);
                synchronization_breakdown.moe_ep_aggregation_extra_ms =
                    std::max(0.0, critical_lane_ms - local_critical_lane_ms);
                const double local_ep_communication_ms =
                    state.decode_local_ep_communication_ms_by_layer.at(
                        layer_index);
                const double source_aware_ep_extra_ms =
                    std::max(0.0, state.decode_ep_communication_ms_per_layer -
                                      local_ep_communication_ms);
                synchronization_breakdown.moe_ep_aggregation_extra_ms +=
                    source_aware_ep_extra_ms;
                if (state.remaining_scaled_moe_layers > 0) {
                    for (std::size_t family = 0;
                         family < attention_group_layout.size(); ++family) {
                        const auto &local =
                            state.scaled_moe_attention_groups.at(family);
                        const double local_pre_transition_ms =
                            local.pre_moe_compute_ms_per_layer +
                            local.pre_moe_tp_communication_ms_per_layer +
                            state.decode_ep_communication_ms_per_layer;
                        synchronization_breakdown.moe_pre_barrier_wait_ms +=
                            static_cast<double>(local.layer_count) *
                            std::max(
                                0.0,
                                maximum_pre_transition_ms_by_family.at(family) -
                                    local_pre_transition_ms);
                    }
                    synchronization_breakdown.moe_ep_aggregation_extra_ms +=
                        static_cast<double>(state.remaining_scaled_moe_layers) *
                        std::max(0.0,
                                 critical_lane_ms - local_critical_lane_ms);
                    synchronization_breakdown.moe_ep_aggregation_extra_ms +=
                        static_cast<double>(state.remaining_scaled_moe_layers) *
                        source_aware_ep_extra_ms;
                }
                simulator.batch_stage(state.batch_id, state.stage_id)
                    .accumulate_execution_time(synchronization_breakdown);
            }
            if (remaining_scaled_layers.value_or(0) > 0) {
                const double repeated_post_ms =
                    critical_lane_ms +
                    (monolithic_decode
                         ? 0.0
                         : sample_state->decode_dp_communication_ms_per_layer);
                double group_remaining_ms =
                    static_cast<double>(*remaining_scaled_layers) *
                    repeated_post_ms;
                for (std::size_t family = 0;
                     family < attention_group_layout.size(); ++family) {
                    group_remaining_ms +=
                        static_cast<double>(
                            attention_group_layout.at(family).layer_count) *
                        maximum_pre_transition_ms_by_family.at(family);
                }
                for (const auto &[unused, batch_id] : group.participants) {
                    static_cast<void>(unused);
                    if (!simulator.batch(batch_id).is_idle()) {
                        moe_stage_state(batch_id, key.stage_id)
                            .remaining_moe_layer_wait_ms = group_remaining_ms;
                    }
                }
            }
        }
        for (const bool idle_pass : {false, true}) {
            for (const auto &[participant, batch_id] : group.participants) {
                const entities::Batch &owner = simulator.batch(batch_id);
                if (owner.is_idle() != idle_pass) {
                    continue;
                }
                double component_ms = 0.0;
                if (path == MoESyncPath::kPrefill) {
                    component_ms =
                        sample_state->prefill_post_attention_ms_by_layer.at(
                            layer_index);
                } else if (monolithic_decode) {
                    // Participants are DP lanes, not EP lanes. Every DP lane
                    // observes the shared expert critical path.
                    component_ms = critical_lane_ms;
                } else {
                    component_ms =
                        critical_lane_ms +
                        sample_state->decode_dp_communication_ms_per_layer;
                }
                const SimTime arrival =
                    SimTime::from_seconds(time.seconds() + component_ms * 1e-3);
                if (owner.is_idle()) {
                    enqueue_idle_moe_arrival(group_key, batch_id, participant,
                                             owner.dp_id(), key.layer_id,
                                             MoESyncPhase::kPostMoe, arrival,
                                             component_ms, simulator);
                } else {
                    const MoEStageState &state =
                        moe_stage_state(batch_id, key.stage_id);
                    enqueue_moe_arrival(state, participant, key.layer_id,
                                        MoESyncPhase::kPostMoe, arrival,
                                        component_ms, simulator);
                }
            }
        }
        return;
    }

    std::set<MoEStageKey> real_states;
    for (const auto &[unused, batch_id] : group.participants) {
        static_cast<void>(unused);
        if (!simulator.batch(batch_id).is_idle()) {
            real_states.emplace([&]() {
                MoEStageKey value{};
                value.batch_id = batch_id;
                value.stage_id = key.stage_id;
                return value;
            }());
        }
    }
    bool has_next_layer = false;
    for (const MoEStageKey &stage_key : real_states) {
        MoEStageState &state = moe_stage_states_.at(stage_key);
        ++state.current_layer;
        has_next_layer =
            has_next_layer || state.current_layer < state.layers_per_stage;
        if (state.current_layer < state.layers_per_stage) {
            simulator.batch(state.batch_id)
                .set_stage_layer(LayerId{state.current_layer});
        }
    }

    if (has_next_layer) {
        const LayerId next_layer{key.layer_id.value() + 1};
        const bool persistent_decode_idle = aligned_decode;
        if (!persistent_decode_idle) {
            for (auto position = group.participants.begin();
                 position != group.participants.end();) {
                if (simulator.batch(position->second).is_idle()) {
                    simulator.release_batch(position->second);
                    position = group.participants.erase(position);
                } else {
                    ++position;
                }
            }
        }
        // Missing decode lanes are represented by one persistent dummy batch
        // for the entire aligned forward. Releasing/recreating it here would
        // allow a late real batch to replace the lane between MoE layers.
        group.participants_initialized = false;
        for (const auto &[participant, batch_id] : group.participants) {
            const entities::Batch &owner = simulator.batch(batch_id);
            if (owner.is_idle() && persistent_decode_idle) {
                // Monolithic decode keeps its existing compact trace contract:
                // the persistent dummy is inserted directly into the barrier
                // after the first real arrival. PDD represents the same dummy
                // with an explicit DecodeSync event.
                if (!monolithic_decode) {
                    enqueue_idle_moe_arrival(group_key, batch_id, participant,
                                             owner.dp_id(), next_layer,
                                             MoESyncPhase::kPreMoe, time, 0.0,
                                             simulator);
                }
                continue;
            }
            MoEStageState &state = moe_stage_state(batch_id, key.stage_id);
            double refreshed_pre_moe_ms = 0.0;
            double refreshed_pre_moe_tp_communication_ms = 0.0;
            if (state.lazy_layer_prediction) {
                const auto next_prediction =
                    get_replica_scheduler(state.replica_id, state.dp_id)
                        .get_replica_stage_scheduler(state.stage_id)
                        .predict_moe_layer(simulator.batch(batch_id),
                                           simulator.requests(),
                                           state.current_layer);
                if (!next_prediction.lazy_moe_layer_prediction ||
                    next_prediction.moe_routing.size() != 1 ||
                    next_prediction.logical_moe_layer_count !=
                        state.layers_per_stage ||
                    next_prediction.moe_routing.front().layer_id.index() !=
                        state.current_layer) {
                    throw std::logic_error(
                        "lazy MoE prediction changed the stage contract");
                }
                const auto &diagnostic = next_prediction.moe_routing.front();
                if (diagnostic.lane_times_ms.size() !=
                    simulator.runtime_config(cluster_type())
                        .parallelism.moe_expert_parallel_size) {
                    throw std::logic_error(
                        "lazy MoE prediction changed the EP lane count");
                }
                refreshed_pre_moe_ms = diagnostic.pre_moe_compute_ms;
                refreshed_pre_moe_tp_communication_ms =
                    diagnostic.pre_moe_tp_communication_ms;
                state.pre_moe_compute_ms_by_layer.push_back(
                    refreshed_pre_moe_ms);
                state.pre_moe_tp_communication_ms_by_layer.push_back(
                    refreshed_pre_moe_tp_communication_ms);
                state.decode_lane_times_ms.push_back(diagnostic.lane_times_ms);
                state.moe_routing_by_layer.push_back(diagnostic);
                state.decode_local_ep_communication_ms_by_layer.push_back(
                    diagnostic.exposed_ep_dispatch_ms +
                    diagnostic.exposed_ep_combine_ms);
                const entities::ExecutionTime &layer_execution =
                    next_prediction.execution_time;
                const double critical_lane_ms =
                    *std::max_element(diagnostic.lane_times_ms.begin(),
                                      diagnostic.lane_times_ms.end());
                const double post_attention_ms =
                    layer_execution.tp_communication_ms +
                    layer_execution.moe_gating_linear_ms +
                    layer_execution.moe_gating_routing_topk_ms +
                    layer_execution.moe_grouped_gemm_ms +
                    layer_execution.moe_shuffling_ms +
                    layer_execution.moe_post_attention_norm_ms +
                    layer_execution.moe_tp_communication_ms +
                    layer_execution.ep_dispatch_ms +
                    layer_execution.ep_combine_ms +
                    layer_execution.dp_input_communication_ms +
                    layer_execution.dp_output_communication_ms;
                const double shared_post_attention_ms =
                    post_attention_ms - critical_lane_ms;
                if (shared_post_attention_ms < -1e-12) {
                    throw std::logic_error(
                        "lazy MoE post-attention work is smaller than lane "
                        "work");
                }
                state.prefill_post_attention_ms_by_layer.push_back(
                    std::max(0.0, shared_post_attention_ms) + critical_lane_ms);
                simulator.batch_stage(batch_id, state.stage_id)
                    .accumulate_execution_time(layer_execution);
                const config::ClusterRuntimeConfig &runtime =
                    simulator.runtime_config(cluster_type());
                simulator.metrics().record_moe_routing(
                    simulator.batch(batch_id), state.stage_id, diagnostic,
                    runtime);
            } else {
                const std::size_t current_layer =
                    static_cast<std::size_t>(state.current_layer);
                refreshed_pre_moe_ms =
                    state.pre_moe_compute_ms_by_layer.at(current_layer);
                refreshed_pre_moe_tp_communication_ms =
                    state.pre_moe_tp_communication_ms_by_layer.at(
                        current_layer);
            }
            const double transition_ms =
                refreshed_pre_moe_ms +
                (path == MoESyncPath::kDecode
                     ? refreshed_pre_moe_tp_communication_ms +
                           state.decode_ep_communication_ms_per_layer
                     : 0.0);
            enqueue_moe_arrival(
                state, participant, next_layer, MoESyncPhase::kPreMoe,
                SimTime::from_seconds(time.seconds() + transition_ms * 1e-3),
                transition_ms, simulator);
        }
        return;
    }

    SimTime domain_release_at;
    for (const MoEStageKey &stage_key : real_states) {
        const MoEStageState &state = moe_stage_states_.at(stage_key);
        const double final_transition_ms =
            state.remaining_moe_layer_wait_ms + state.suffix_compute_ms +
            state.lm_head_ms + state.pp_ms +
            (path == MoESyncPath::kDecode
                 ? state.decode_ep_communication_ms_per_layer +
                       state.suffix_tp_communication_ms
                 : 0.0);
        const SimTime completion_time =
            SimTime::from_seconds(time.seconds() + final_transition_ms * 1e-3);
        if (!domain_release_at.valid() || completion_time > domain_release_at) {
            domain_release_at = completion_time;
        }
        simulator.event_queue().push(completion_time, [&]() {
            BatchStageEndPayload value{};
            value.batch_id = state.batch_id;
            value.replica_id = state.replica_id;
            value.dp_id = state.dp_id;
            value.stage_id = state.stage_id;
            value.generation = state.batch_generation;
            value.cluster_type = state.cluster_type;
            return value;
        }());
    }
    if (aligned_decode) {
        if (!domain_release_at.valid()) {
            throw std::logic_error(
                "DECODE MoE collective has no release boundary");
        }
        const MoEDomainKey domain_key =
            make_moe_domain_key(key.replica_id, key.stage_id);
        const auto reservation_position =
            moe_domain_reservations_.find(domain_key);
        if (reservation_position == moe_domain_reservations_.end()) {
            throw std::logic_error(
                "DECODE MoE collective lost its domain reservation");
        }
        reservation_position->second.release_at = domain_release_at;
        // A lane-local stage-end event may have observed the still-reserved
        // domain before the latest real lane completed.  Re-emit the existing
        // schedule event for every DP lane at the release boundary so queued
        // work is woken without introducing a new DES event type.
        for (std::uint64_t dp = 0; dp < group.expected_participants; ++dp) {
            simulator.event_queue().push(
                domain_release_at,
                ReplicaStageSchedulePayload{key.replica_id, DataParallelId{dp},
                                            key.stage_id, key.cluster_type});
        }
    }
    for (const MoEStageKey &stage_key : real_states) {
        moe_stage_states_.erase(stage_key);
    }
    for (const auto &[unused, batch_id] : group.participants) {
        static_cast<void>(unused);
        if (simulator.batch(batch_id).is_idle()) {
            simulator.release_batch(batch_id);
        }
    }
    moe_group_states_.erase(group_position);
}

void BaseClusterScheduler::on_prefill_sync(const PrefillSyncPayload &payload,
                                           SimTime time,
                                           simulator::Simulator &simulator) {
    const entities::Batch &batch = simulator.batch(payload.batch_id);
    if (batch.schedule_epoch() != payload.generation) {
        return;
    }
    if (batch.is_idle() != payload.is_idle) {
        throw MoEBarrierError(
            "prefill synchronization idle marker disagrees with batch");
    }
    const MoEBarrierKey key{payload.cluster_type,   payload.replica_id,
                            payload.stage_id,       payload.sync_group_id,
                            payload.layer_id,       payload.sync_phase,
                            payload.sync_generation};
    ensure_moe_group_participants(key, MoESyncPath::kPrefill,
                                  payload.participant_id, time, simulator);
    const auto ready = moe_barrier_.arrive(
        key,
        MoEBarrierParticipant{payload.participant_id, payload.batch_id, time,
                              payload.elapsed_component_ms, payload.is_idle},
        simulator.parallelism(payload.cluster_type).data_parallel_size);
    if (ready.has_value()) {
        simulator.event_queue().push(
            ready->collective_time,
            PrefillSyncCollectivePayload{
                payload.replica_id, payload.stage_id, payload.sync_group_id,
                payload.layer_id, payload.sync_phase, payload.sync_generation,
                payload.cluster_type});
    }
}

void BaseClusterScheduler::on_prefill_sync_collective(
    const PrefillSyncCollectivePayload &payload, SimTime time,
    simulator::Simulator &simulator) {
    const MoEBarrierKey key{payload.cluster_type,   payload.replica_id,
                            payload.stage_id,       payload.sync_group_id,
                            payload.layer_id,       payload.sync_phase,
                            payload.sync_generation};
    const auto participants = moe_barrier_.consume(key);
    continue_moe_stage(key, MoESyncPath::kPrefill, participants, time,
                       simulator);
}

void BaseClusterScheduler::on_decode_sync(const DecodeSyncPayload &payload,
                                          SimTime time,
                                          simulator::Simulator &simulator) {
    const entities::Batch &batch = simulator.batch(payload.batch_id);
    if (batch.schedule_epoch() != payload.generation) {
        return;
    }
    if (batch.is_idle() != payload.is_idle) {
        throw MoEBarrierError(
            "decode synchronization idle marker disagrees with batch");
    }
    const MoEBarrierKey key{payload.cluster_type,   payload.replica_id,
                            payload.stage_id,       payload.sync_group_id,
                            payload.layer_id,       payload.sync_phase,
                            payload.sync_generation};
    ensure_moe_group_participants(key, MoESyncPath::kDecode,
                                  payload.participant_id, time, simulator);
    const std::uint64_t expected_participants =
        simulator.parallelism(payload.cluster_type).data_parallel_size;
    auto ready = moe_barrier_.arrive(
        key,
        MoEBarrierParticipant{payload.participant_id, payload.batch_id, time,
                              payload.elapsed_component_ms, payload.is_idle},
        expected_participants);
    if (!ready.has_value() && payload.sync_phase == MoESyncPhase::kPreMoe) {
        ready = compact_moe_group_participants(key, MoESyncPath::kDecode, time,
                                               simulator);
    }
    if (ready.has_value()) {
        simulator.event_queue().push(
            ready->collective_time,
            DecodeSyncCollectivePayload{
                payload.replica_id, payload.stage_id, payload.sync_group_id,
                payload.layer_id, payload.sync_phase, payload.sync_generation,
                payload.cluster_type});
    }
}

void BaseClusterScheduler::on_decode_sync_collective(
    const DecodeSyncCollectivePayload &payload, SimTime time,
    simulator::Simulator &simulator) {
    const MoEBarrierKey key{payload.cluster_type,   payload.replica_id,
                            payload.stage_id,       payload.sync_group_id,
                            payload.layer_id,       payload.sync_phase,
                            payload.sync_generation};
    const auto participants = moe_barrier_.consume(key);
    continue_moe_stage(key, MoESyncPath::kDecode, participants, time,
                       simulator);
}

void BaseClusterScheduler::require_quiescent() const {
    moe_barrier_.require_empty();
    if (!moe_stage_states_.empty() || !moe_group_states_.empty() ||
        !moe_domain_reservations_.empty()) {
        throw std::runtime_error(
            "simulation quiesced with live cluster MoE state");
    }
}

} // namespace frontier::scheduler
