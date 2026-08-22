#include "frontier/execution_time_predictor/analytical_moe_stage.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>

namespace frontier::execution_time_predictor::internal {

detail::MoEModel make_moe_model(const config::ModelConfig &model,
                                const config::ParallelismConfig &parallelism) {
    detail::MoEModel result{};
    result.hidden_size = model.hidden_size;
    result.intermediate_size = model.moe_intermediate_size;
    result.model_num_experts = model.num_experts;
    result.num_shared_experts = model.num_shared_experts;
    result.moe_tensor_parallel_size = parallelism.moe_tensor_parallel_size;
    result.expert_parallel_size = parallelism.moe_expert_parallel_size;
    result.routed_expert_hidden_size = model.routed_expert_hidden_size;
    result.latent_moe_use_norm = model.latent_moe_use_norm;
    result.attn_res_block_size = model.attn_res_block_size;
    result.gated_mlp = model.gated_mlp;
    result.fused_add_norm = model.fused_add_norm;
    return result;
}

detail::MoEOperatorPrecisions make_moe_operator_precisions(
    const config::AnalyticalExecutionModelConfig &config) {
    // Router storage and compute dtypes are intentionally independent.  K3
    // reads BF16 weights/inputs and performs the router GEMM, logits, and
    // top-k work in FP32; conversion is assumed fused into those kernels.
    detail::MoEOperatorPrecisions result{};
    result.expert =
        detail::precision_from_string(config.moe_expert_precision());
    result.router =
        detail::precision_from_string(config.moe_router_precision());
    result.dense = detail::precision_from_string(config.dense_precision());
    result.shared_expert =
        detail::precision_from_string(config.shared_expert_weight_precision());
    result.expert_weight =
        detail::precision_from_string(config.routed_expert_weight_precision());
    result.expert_activation = detail::precision_from_string(
        config.routed_expert_activation_precision());
    result.latent_moe_projection_weight = detail::precision_from_string(
        config.latent_moe_projection_weight_precision());
    result.latent_moe_projection_activation = detail::precision_from_string(
        config.latent_moe_projection_activation_precision());
    result.shared_expert_weight =
        detail::precision_from_string(config.shared_expert_weight_precision());
    result.shared_expert_activation = detail::precision_from_string(
        config.shared_expert_activation_precision());
    result.router_weight =
        detail::precision_from_string(config.router_weight_storage_precision());
    result.router_activation =
        detail::precision_from_string(config.moe_router_activation_precision());
    result.dense_weight =
        detail::precision_from_string(config.dense_weight_precision());
    result.dense_activation =
        detail::precision_from_string(config.dense_activation_precision());
    result.router_compute =
        detail::precision_from_string(config.router_compute_precision());
    return result;
}

double total_attention_layer_compute_ms(const detail::DenseLayerTimes &layer) {
    return layer.attention_pre_projection_ms +
           layer.attention_post_projection_ms + layer.rope_ms +
           layer.kv_cache_save_ms + layer.attention_norm_ms +
           layer.attention_inter_norm_ms + layer.attention_wq_projection_ms +
           layer.prefill_attention_ms + layer.decode_attention_ms +
           layer.attn_res_ms;
}

std::uint64_t activation_payload_bytes(std::uint64_t tokens,
                                       std::uint64_t hidden_size,
                                       double element_bytes) {
    const long double bytes = static_cast<long double>(tokens) *
                              static_cast<long double>(hidden_size) *
                              static_cast<long double>(element_bytes);
    if (bytes >
        static_cast<long double>(std::numeric_limits<std::uint64_t>::max())) {
        throw ExecutionTimePredictorError(
            "analytical communication byte count overflows uint64");
    }
    return static_cast<std::uint64_t>(std::ceil(bytes));
}

std::vector<double> lane_times_ms(const detail::MoELanePrediction &prediction) {
    std::vector<double> result;
    result.reserve(prediction.lane_times.size());
    for (const detail::MoELayerTime &lane : prediction.lane_times) {
        result.push_back(lane.total_ms());
    }
    return result;
}

MoERoutingDiagnostic make_moe_routing_diagnostic(
    std::uint64_t moe_layer_index, std::uint64_t model_layer,
    double pre_moe_compute_ms, double pre_moe_tp_communication_ms,
    const detail::RoutingAllocation &allocation,
    const detail::MoELanePrediction &lane_prediction) {
    MoERoutingDiagnostic result{};
    result.layer_id = LayerId{moe_layer_index};
    result.model_layer_id = model_layer;
    result.pre_moe_compute_ms = pre_moe_compute_ms;
    result.pre_moe_tp_communication_ms = pre_moe_tp_communication_ms;
    result.input_tokens = allocation.input_tokens;
    result.routed_tokens = allocation.routed_tokens;
    result.global_expert_tokens = allocation.global_expert_tokens;
    result.lane_expert_tokens = allocation.lane_expert_tokens;
    result.lane_routed_tokens = allocation.lane_routed_tokens;
    result.lane_active_experts = allocation.lane_active_experts;
    result.lane_unique_tokens = allocation.lane_unique_tokens;
    result.lane_times_ms = lane_times_ms(lane_prediction);
    result.routed_lane_times_ms = lane_prediction.routed_lane_times_ms;
    result.source_local_lane_times_ms =
        lane_prediction.source_local_lane_times_ms;
    result.shared_expert_path_ms = lane_prediction.shared_expert_path_ms;
    result.critical_lane = lane_prediction.critical_lane;
    result.critical_lane_time_ms = lane_prediction.critical_lane_time_ms;
    return result;
}

void add_critical_moe_layer_time(entities::ExecutionTime &execution_time,
                                 const detail::MoELayerTime &critical_layer,
                                 double residual_add_ms) {
    execution_time.moe_gating_linear_ms += critical_layer.gating_linear_ms;
    execution_time.moe_gating_routing_topk_ms +=
        critical_layer.gating_routing_topk_ms;
    execution_time.moe_grouped_gemm_ms +=
        critical_layer.grouped_up_projection_ms +
        critical_layer.grouped_down_projection_ms +
        critical_layer.latent_projection_ms;
    execution_time.moe_shuffling_ms += critical_layer.shuffling_ms;
    execution_time.moe_post_attention_norm_ms +=
        critical_layer.post_attention_norm_ms + critical_layer.latent_norm_ms +
        critical_layer.attn_res_ms + 2.0 * residual_add_ms;
}

ScaledMoEAttentionFamily
scaled_attention_family(const config::ModelConfig &model,
                        std::uint64_t model_layer) noexcept {
    if (model.has_kda() && model.is_kda_layer(model_layer)) {
        return ScaledMoEAttentionFamily::kKda;
    }
    if (model.is_mla_layer(model_layer) || (model.use_mla && !model.has_kda() &&
                                            model.mla_layer_indices.empty())) {
        return ScaledMoEAttentionFamily::kMla;
    }
    return ScaledMoEAttentionFamily::kStandard;
}

// Stage-group compression is exact only when the configured layer scope makes
// the routing allocation independent of the logical model layer.
bool routing_is_layer_invariant(
    const config::MoeRoutingConfig &routing) noexcept {
    return routing.layer_scope == config::MoeRoutingLayerScope::kShared;
}

bool routing_is_batch_shared(
    const config::AnalyticalExecutionModelConfig &config,
    const config::MoeRoutingConfig &routing) noexcept {
    // first_layer_scaled is defined by one representative expert path for the
    // whole batch. It therefore always uses one batch-shared draw, even when a
    // caller leaves layer_scope set to per_layer.
    return config.moe_layer_event_mode == "first_layer_scaled" ||
           routing_is_layer_invariant(routing);
}

// A shared assignment must not depend on which layer asked for it: two
// pipeline stages hold different layers of the same batch, and if the seed
// moved with the layer they would charge different routing for one batch.
// Seeding from a fixed layer makes every layer and every stage agree, which
// is the property the stage timing templates rely on.
std::uint64_t routing_seed_layer(bool batch_shared,
                                 std::uint64_t model_layer) noexcept {
    return batch_shared ? 0 : model_layer;
}

bool has_contiguous_moe_suffix(const config::ModelConfig &model,
                               const config::PipelineStageLayerRange &layers) {
    bool saw_moe = false;
    for (std::uint64_t layer = layers.begin; layer < layers.end; ++layer) {
        const bool is_moe = model.is_moe_layer(layer);
        if (saw_moe && !is_moe) {
            return false;
        }
        saw_moe = saw_moe || is_moe;
    }
    return true;
}

void add_scaled_attention_layer(std::vector<ScaledMoEAttentionGroup> &groups,
                                ScaledMoEAttentionFamily family,
                                double pre_moe_compute_ms,
                                double pre_moe_tp_communication_ms) {
    const auto position =
        std::find_if(groups.begin(), groups.end(), [family](const auto &group) {
            return group.family == family;
        });
    if (position == groups.end()) {
        groups.push_back(ScaledMoEAttentionGroup{family, 1, pre_moe_compute_ms,
                                                 pre_moe_tp_communication_ms});
        return;
    }
    ++position->layer_count;
    // The analytical roofline uses one representative prediction per
    // attention family.  Fail fast if a future layer-dependent operator is
    // accidentally folded into this compression mode.
    const double tolerance =
        1e-12 * std::max({1.0, std::abs(position->pre_moe_compute_ms_per_layer),
                          std::abs(pre_moe_compute_ms)});
    if (std::abs(position->pre_moe_compute_ms_per_layer - pre_moe_compute_ms) >
        tolerance) {
        throw ExecutionTimePredictorError(
            "scaled MoE prediction requires identical attention time within "
            "each attention family");
    }
    const double communication_tolerance =
        1e-12 *
        std::max({1.0,
                  std::abs(position->pre_moe_tp_communication_ms_per_layer),
                  std::abs(pre_moe_tp_communication_ms)});
    if (std::abs(position->pre_moe_tp_communication_ms_per_layer -
                 pre_moe_tp_communication_ms) > communication_tolerance) {
        throw ExecutionTimePredictorError(
            "scaled MoE prediction requires identical attention communication "
            "within each attention family");
    }
}

// Both MoE stage assemblers charge the same per-layer collective set; each
// decides for itself whether to multiply it by the layer count.
detail::MoECommunicationTime
moe_stage_communication(const MoEStageContext &context,
                        const detail::RoutingAllocation &allocation,
                        const detail::MoELanePrediction &lane_prediction) {
    if (context.config.moe_communication_backend == "generic" &&
        context.reusable_moe_communication != nullptr) {
        return *context.reusable_moe_communication;
    }
    const std::uint64_t input_tokens = context.dense_batch.total_tokens;
    const detail::MoELayerTime &critical = lane_prediction.lane_times.at(
        static_cast<std::size_t>(lane_prediction.critical_lane));
    const double fused_expert_compute_ms = critical.grouped_up_projection_ms +
                                           critical.grouped_down_projection_ms +
                                           critical.shuffling_ms;
    return detail::predict_moe_communication(
        context.communication_backend, input_tokens, context.model.hidden_size,
        input_tokens * context.model.router_topk,
        context.parallelism.tensor_parallel_size,
        context.parallelism.moe_tensor_parallel_size,
        context.parallelism.moe_expert_parallel_size,
        context.parallelism.data_parallel_size, false,
        context.communication_element_bytes,
        context.model.routed_expert_hidden_size,
        context.config.moe_communication_backend, &allocation,
        fused_expert_compute_ms, context.analytical.moe_a2a_overlap_residual,
        context.analytical.mega_moe_a2a_bandwidth_scale,
        context.analytical.mega_moe_a2a_startup_scale,
        detail::bytes_per_element(detail::precision_from_string(
            context.config.routed_expert_activation_precision())));
}

void reset_moe_compute(entities::ExecutionTime &execution_time) noexcept {
    execution_time.dense_compute_ms = 0.0;
    execution_time.tp_communication_ms = 0.0;
    execution_time.moe_gating_linear_ms = 0.0;
    execution_time.moe_gating_routing_topk_ms = 0.0;
    execution_time.moe_grouped_gemm_ms = 0.0;
    execution_time.moe_shuffling_ms = 0.0;
    execution_time.moe_post_attention_norm_ms = 0.0;
}

void reset_moe_communication(entities::ExecutionTime &execution_time) noexcept {
    execution_time.moe_tp_communication_ms = 0.0;
    execution_time.ep_dispatch_ms = 0.0;
    execution_time.ep_combine_ms = 0.0;
    execution_time.dp_input_communication_ms = 0.0;
    execution_time.dp_output_communication_ms = 0.0;
}

void record_moe_layer(MoEStagePrediction &result,
                      const MoEStageContext &context,
                      std::uint64_t local_moe_layer, std::uint64_t model_layer,
                      double pre_moe_compute_ms,
                      double pre_moe_tp_communication_ms,
                      const detail::RoutingAllocation &allocation,
                      const detail::MoELanePrediction &lane_prediction) {
    result.routing_diagnostics.push_back(make_moe_routing_diagnostic(
        local_moe_layer, model_layer, pre_moe_compute_ms,
        pre_moe_tp_communication_ms, allocation, lane_prediction));
    const detail::MoELayerTime &critical = lane_prediction.lane_times.at(
        static_cast<std::size_t>(lane_prediction.critical_lane));
    add_critical_moe_layer_time(
        result.execution_time, critical,
        context.layer_time(model_layer).residual_add_ms);
    const detail::MoECommunicationTime communication =
        moe_stage_communication(context, allocation, lane_prediction);
    MoERoutingDiagnostic &diagnostic = result.routing_diagnostics.back();
    diagnostic.raw_ep_dispatch_ms = communication.raw_ep_dispatch_ms;
    diagnostic.raw_ep_combine_ms = communication.raw_ep_combine_ms;
    diagnostic.exposed_ep_dispatch_ms = communication.ep_dispatch_ms;
    diagnostic.exposed_ep_combine_ms = communication.ep_combine_ms;
    result.execution_time.moe_tp_communication_ms += communication.moe_tp_ms;
    result.execution_time.ep_dispatch_ms += communication.ep_dispatch_ms;
    result.execution_time.ep_combine_ms += communication.ep_combine_ms;
    if (context.cluster_type == ClusterType::kDecode) {
        result.execution_time.dp_input_communication_ms +=
            communication.dp_input_ms;
        result.execution_time.dp_output_communication_ms +=
            communication.dp_output_ms;
    }
    if (!context.detailed_diagnostics_enabled) {
        return;
    }
    result.diagnostics.emplace_back(
        "layer_" + std::to_string(model_layer) + "_critical_lane",
        static_cast<double>(lane_prediction.critical_lane));
    result.diagnostics.emplace_back("layer_" + std::to_string(model_layer) +
                                        "_critical_lane_ms",
                                    lane_prediction.critical_lane_time_ms);
}

MoEStagePrediction
predict_selected_moe_layer_execution(const MoEStageContext &context,
                                     std::uint64_t selected_moe_layer) {
    const config::ParallelismConfig &parallelism = context.parallelism;
    const config::ModelConfig &model = context.model;
    MoEStagePrediction result{};
    result.execution_time = context.base_execution_time;
    reset_moe_compute(result.execution_time);
    reset_moe_communication(result.execution_time);

    detail::MoEModel moe_model = make_moe_model(model, parallelism);
    moe_model.decode_only = context.dense_batch.prefill_requests.empty() &&
                            !context.dense_batch.decode_requests.empty();
    const detail::MoEOperatorPrecisions moe_precisions =
        make_moe_operator_precisions(context.config);
    double pending_dense_compute_ms = 0.0;
    double pending_tp_communication_ms = 0.0;
    std::uint64_t local_moe_layer = 0;
    for (std::uint64_t model_layer = context.stage_layers.begin;
         model_layer < context.stage_layers.end; ++model_layer) {
        if (!model.is_moe_layer(model_layer)) {
            pending_dense_compute_ms +=
                context.dense_layer_compute_ms(model_layer);
            pending_tp_communication_ms += context.tp_layer_ms(model_layer);
            continue;
        }
        ++result.logical_moe_layer_count;
        if (local_moe_layer == selected_moe_layer) {
            const double pre_moe_compute_ms =
                pending_dense_compute_ms +
                context.attention_compute_ms(model_layer);
            const double pre_moe_tp_communication_ms =
                pending_tp_communication_ms +
                context.attention_communication_ms(model_layer);
            result.execution_time.dense_compute_ms += pre_moe_compute_ms;
            result.execution_time.tp_communication_ms +=
                pre_moe_tp_communication_ms;
            std::optional<detail::RoutingAllocation> owned_allocation;
            const detail::RoutingAllocation *allocation =
                context.reusable_routing_allocation;
            if (allocation == nullptr) {
                owned_allocation.emplace(detail::route_tokens(
                    context.dense_batch.total_tokens, model.router_topk,
                    model.total_expert_num,
                    parallelism.moe_expert_parallel_size, context.routing,
                    routing_seed_layer(routing_is_batch_shared(context.config,
                                                               context.routing),
                                       model_layer),
                    context.routing_sample_id));
                allocation = &*owned_allocation;
            }
            std::optional<detail::MoELanePrediction> owned_lane_prediction;
            const detail::MoELanePrediction *lane_prediction =
                context.reusable_moe_lane_prediction;
            if (lane_prediction == nullptr) {
                owned_lane_prediction.emplace(detail::predict_moe_lanes(
                    context.device, context.analytical, moe_model, *allocation,
                    model.router_topk, moe_precisions));
                lane_prediction = &*owned_lane_prediction;
            }
            record_moe_layer(result, context, local_moe_layer, model_layer,
                             pre_moe_compute_ms, pre_moe_tp_communication_ms,
                             *allocation, *lane_prediction);
        }
        pending_dense_compute_ms = 0.0;
        pending_tp_communication_ms = 0.0;
        ++local_moe_layer;
    }
    if (selected_moe_layer >= result.logical_moe_layer_count ||
        result.routing_diagnostics.size() != 1) {
        throw ExecutionTimePredictorError(
            "lazy MoE layer index is outside the pipeline stage");
    }
    if (selected_moe_layer == 0) {
        result.suffix_compute_ms = pending_dense_compute_ms;
        result.suffix_tp_communication_ms = pending_tp_communication_ms;
        result.execution_time.dense_compute_ms += pending_dense_compute_ms;
        result.execution_time.tp_communication_ms +=
            pending_tp_communication_ms;
    }

    return result;
}

MoEStagePrediction predict_moe_stage_execution(const MoEStageContext &context) {
    const config::AnalyticalExecutionModelConfig &config = context.config;
    const config::ParallelismConfig &parallelism = context.parallelism;
    const config::ModelConfig &model = context.model;
    const config::MoeRoutingConfig &routing = context.routing;
    const config::PipelineStageLayerRange &stage_layers = context.stage_layers;
    MoEStagePrediction result{};
    result.execution_time = context.base_execution_time;
    reset_moe_compute(result.execution_time);
    reset_moe_communication(result.execution_time);
    detail::MoEModel moe_model = make_moe_model(model, parallelism);
    moe_model.decode_only = context.dense_batch.prefill_requests.empty() &&
                            !context.dense_batch.decode_requests.empty();
    const detail::MoEOperatorPrecisions moe_precisions =
        make_moe_operator_precisions(config);
    double pending_pre_moe_compute_ms = 0.0;
    double pending_pre_moe_tp_communication_ms = 0.0;
    std::uint64_t moe_layer_index = 0;
    const detail::MoELanePrediction *repeated_lane_prediction = nullptr;
    const detail::RoutingAllocation *repeated_allocation = nullptr;
    const bool first_layer_scaled =
        config.moe_layer_event_mode == "first_layer_scaled";
    const bool stage_group_requested =
        config.moe_layer_event_mode == "stage_group_scaled";
    // The compressed scheduler contract can represent a contiguous MoE
    // suffix.  If a stage interleaves dense and MoE layers, or routing is
    // layer dependent, retain detailed per-layer predictions instead of
    // silently charging a representative lane to a different logical layer.
    const bool stage_group_scaled =
        stage_group_requested && routing_is_layer_invariant(routing) &&
        has_contiguous_moe_suffix(model, stage_layers);
    // Shared routing gives every MoE layer the same assignment, so the
    // allocation and the expert roofline it feeds are computed once and reused
    // for the rest of the stage. first_layer_scaled has the same batch-shared
    // contract even when the raw layer_scope is per_layer.
    const bool layer_shared = routing_is_batch_shared(config, routing);
    std::optional<detail::RoutingAllocation> owned_allocation;
    std::optional<detail::MoELanePrediction> owned_lane_prediction;
    const detail::RoutingAllocation *shared_allocation =
        context.reusable_routing_allocation;
    const detail::MoELanePrediction *shared_lane_prediction =
        context.reusable_moe_lane_prediction;
    for (std::uint64_t model_layer = stage_layers.begin;
         model_layer < stage_layers.end; ++model_layer) {
        if (!model.is_moe_layer(model_layer)) {
            if (first_layer_scaled && repeated_lane_prediction != nullptr) {
                throw ExecutionTimePredictorError(
                    "first_layer_scaled requires a contiguous MoE suffix "
                    "within each pipeline stage");
            }
            result.execution_time.dense_compute_ms +=
                context.dense_layer_compute_ms(model_layer);
            result.execution_time.tp_communication_ms +=
                context.tp_layer_ms(model_layer);
            pending_pre_moe_compute_ms +=
                context.dense_layer_compute_ms(model_layer);
            pending_pre_moe_tp_communication_ms +=
                context.tp_layer_ms(model_layer);
            continue;
        }
        const double attention_compute_ms =
            context.attention_compute_ms(model_layer);
        result.execution_time.dense_compute_ms += attention_compute_ms;
        result.execution_time.tp_communication_ms +=
            context.attention_communication_ms(model_layer);
        ++result.logical_moe_layer_count;
        if ((first_layer_scaled || stage_group_scaled) &&
            repeated_lane_prediction != nullptr) {
            add_scaled_attention_layer(
                result.scaled_attention_groups,
                scaled_attention_family(model, model_layer),
                attention_compute_ms,
                context.attention_communication_ms(model_layer));
            const detail::MoELayerTime &critical =
                repeated_lane_prediction->lane_times.at(
                    static_cast<std::size_t>(
                        repeated_lane_prediction->critical_lane));
            add_critical_moe_layer_time(
                result.execution_time, critical,
                context.layer_time(model_layer).residual_add_ms);
            const detail::MoECommunicationTime communication =
                moe_stage_communication(context, *repeated_allocation,
                                        *repeated_lane_prediction);
            result.execution_time.moe_tp_communication_ms +=
                communication.moe_tp_ms;
            result.execution_time.ep_dispatch_ms +=
                communication.ep_dispatch_ms;
            result.execution_time.ep_combine_ms += communication.ep_combine_ms;
            if (context.cluster_type == ClusterType::kDecode) {
                result.execution_time.dp_input_communication_ms +=
                    communication.dp_input_ms;
                result.execution_time.dp_output_communication_ms +=
                    communication.dp_output_ms;
            }
            pending_pre_moe_compute_ms = 0.0;
            pending_pre_moe_tp_communication_ms = 0.0;
            ++moe_layer_index;
            continue;
        }
        pending_pre_moe_compute_ms += attention_compute_ms;
        const detail::RoutingAllocation *allocation = shared_allocation;
        if (allocation == nullptr || !layer_shared) {
            owned_allocation.emplace(detail::route_tokens(
                context.dense_batch.total_tokens, model.router_topk,
                model.total_expert_num, parallelism.moe_expert_parallel_size,
                routing, routing_seed_layer(layer_shared, model_layer),
                context.routing_sample_id));
            allocation = &*owned_allocation;
            if (layer_shared) {
                shared_allocation = allocation;
            }
        }
        const detail::MoELanePrediction *lane_prediction =
            shared_lane_prediction;
        if (lane_prediction == nullptr || !layer_shared) {
            owned_lane_prediction.emplace(detail::predict_moe_lanes(
                context.device, context.analytical, moe_model, *allocation,
                model.router_topk, moe_precisions));
            lane_prediction = &*owned_lane_prediction;
            if (layer_shared) {
                shared_lane_prediction = lane_prediction;
            }
        }
        record_moe_layer(result, context, moe_layer_index, model_layer,
                         pending_pre_moe_compute_ms,
                         pending_pre_moe_tp_communication_ms +
                             context.attention_communication_ms(model_layer),
                         *allocation, *lane_prediction);
        pending_pre_moe_compute_ms = 0.0;
        pending_pre_moe_tp_communication_ms = 0.0;
        ++moe_layer_index;
        if (first_layer_scaled || stage_group_scaled) {
            repeated_lane_prediction = lane_prediction;
            repeated_allocation = allocation;
            result.repeated_moe_layer_pre_compute_ms = attention_compute_ms;
        }
    }
    result.suffix_compute_ms = pending_pre_moe_compute_ms;
    result.suffix_tp_communication_ms = pending_pre_moe_tp_communication_ms;
    return result;
}

const detail::DenseLayerTimes &
MoEStageContext::layer_time(std::uint64_t model_layer) const noexcept {
    return layer_times[static_cast<std::size_t>(model_layer -
                                                stage_layers.begin)];
}

double MoEStageContext::dense_layer_compute_ms(
    std::uint64_t model_layer) const noexcept {
    return layer_time(model_layer).total_ms();
}

double MoEStageContext::attention_compute_ms(
    std::uint64_t model_layer) const noexcept {
    return total_attention_layer_compute_ms(layer_time(model_layer));
}

bool MoEStageContext::is_mla_layer(std::uint64_t model_layer) const noexcept {
    return !(model.has_kda() && model.is_kda_layer(model_layer));
}

double MoEStageContext::tp_layer_ms(std::uint64_t model_layer) const noexcept {
    return 2.0 * allreduce_ms +
           (is_mla_layer(model_layer) ? dcp_attention_communication_ms : 0.0);
}

double MoEStageContext::attention_communication_ms(
    std::uint64_t model_layer) const noexcept {
    return allreduce_ms +
           (is_mla_layer(model_layer) ? dcp_attention_communication_ms : 0.0);
}

} // namespace frontier::execution_time_predictor::internal
