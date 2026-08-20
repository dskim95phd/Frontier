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
#include "frontier/simulator/simulator.h"

namespace frontier::scheduler {

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

} // namespace frontier::scheduler
