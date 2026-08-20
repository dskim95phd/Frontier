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

} // namespace frontier::scheduler
