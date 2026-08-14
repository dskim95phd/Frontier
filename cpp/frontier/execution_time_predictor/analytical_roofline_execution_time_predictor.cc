#include "frontier/execution_time_predictor/analytical_roofline_execution_time_predictor.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <utility>

#include "frontier/attention/mla.h"
#include "frontier/cc_backend/analytical_model.h"

namespace frontier::execution_time_predictor {
namespace {

const entities::Request &
get_request(const std::vector<entities::Request> &requests,
            RequestId request_id) {
    if (!request_id.valid() || request_id.index() >= requests.size()) {
        throw ExecutionTimePredictorError(
            "batch execution references an unknown request");
    }
    const entities::Request &request = requests.at(request_id.index());
    if (request.id() != request_id) {
        throw ExecutionTimePredictorError(
            "batch execution request arena invariant failed");
    }
    return request;
}

struct StageBatchInfo {
    detail::DenseBatch dense_batch;
    std::uint64_t lm_head_tokens = 0;
};

detail::AttentionRequestSlice
make_attention_request_slice(const entities::RequestBatchSnapshot &snapshot,
                             const entities::Request &request) {
    return detail::AttentionRequestSlice{
        snapshot.scheduled_tokens,
        request.num_processed_tokens(),
    };
}

StageBatchInfo
build_stage_batch_info(const entities::Batch &batch,
                       const std::vector<entities::Request> &requests) {
    StageBatchInfo result{};
    result.dense_batch.total_tokens = batch.total_scheduled_tokens();
    for (const entities::RequestBatchSnapshot &snapshot : batch.requests()) {
        const entities::Request &request =
            get_request(requests, snapshot.request_id);
        // Production Python evaluates analytical attention from the request's
        // mutable state when each PP stage starts, rather than from the batch
        // creation snapshot. Earlier overlapping stages may have made
        // additional progress visible by then.
        const detail::AttentionRequestSlice slice =
            make_attention_request_slice(snapshot, request);
        if (!request.is_prefill_complete()) {
            result.dense_batch.prefill_requests.push_back(slice);
            if (snapshot.processed_tokens + snapshot.scheduled_tokens >=
                request.num_prefill_tokens()) {
                ++result.lm_head_tokens;
            }
        } else {
            result.dense_batch.decode_requests.push_back(slice);
            ++result.lm_head_tokens;
        }
    }
    return result;
}

detail::DenseModel
make_dense_model(const config::ModelConfig &model,
                 const config::ParallelismConfig &parallelism) {
    detail::DenseModel result{};
    result.hidden_size = model.hidden_size;
    result.intermediate_size = model.dense_intermediate_size;
    result.num_query_heads = model.num_query_heads;
    result.num_kv_heads = model.num_kv_heads;
    result.head_dim = model.head_dim;
    result.tensor_parallel_size = parallelism.tensor_parallel_size;
    result.gated_mlp = model.gated_mlp;
    result.fused_add_norm = model.fused_add_norm;
    result.use_mla = model.use_mla;
    result.mla_use_output_gate = model.mla_use_output_gate;
    result.mla_use_nope = model.mla_use_nope;
    result.use_mfa = model.use_mfa;
    result.q_lora_rank = model.q_lora_rank;
    result.kv_lora_rank = model.kv_lora_rank;
    result.qk_nope_head_dim = model.qk_nope_head_dim;
    result.qk_rope_head_dim = model.qk_rope_head_dim;
    result.qk_head_dim = model.qk_head_dim;
    result.v_head_dim = model.v_head_dim;
    result.share_q_dim = model.share_q_dim;
    result.decode_context_parallel_size =
        parallelism.decode_context_parallel_size;
    // KDA is selected per logical layer in predict_execution.  Keep the
    // dimensions in the compact detail model even when this particular
    // invocation represents an MLA layer so a stage can cheaply switch
    // variants without rebuilding model metadata.
    result.kda_num_heads = model.kda_num_heads;
    result.kda_num_k_heads = model.kda_num_k_heads;
    result.kda_num_v_heads = model.kda_num_v_heads;
    result.kda_key_head_dim = model.kda_key_head_dim;
    result.kda_value_head_dim = model.kda_value_head_dim;
    result.kda_head_dim = model.kda_head_dim;
    result.kda_short_conv_kernel_size = model.kda_short_conv_kernel_size;
    result.kda_conv_state_dim = model.kda_conv_state_dim;
    result.attn_res_block_size = model.attn_res_block_size;
    return result;
}

detail::DenseOperatorPrecisions make_dense_operator_precisions(
    const config::AnalyticalExecutionModelConfig &config) {
    return detail::DenseOperatorPrecisions{
        detail::precision_from_string(config.attention_precision()),
        detail::precision_from_string(config.dense_precision()),
        detail::precision_from_string(config.kv_cache_precision()),
        detail::precision_from_string(config.attention_weight_precision()),
        detail::precision_from_string(config.attention_activation_precision()),
        detail::precision_from_string(config.dense_mlp_weight_precision()),
        detail::precision_from_string(config.dense_mlp_activation_precision()),
        detail::precision_from_string(config.kda_snapshot_precision()),
    };
}

detail::MoEModel make_moe_model(const config::ModelConfig &model,
                                const config::ParallelismConfig &parallelism) {
    detail::MoEModel result{};
    result.hidden_size = model.hidden_size;
    result.intermediate_size = model.moe_intermediate_size;
    result.model_num_experts = model.num_experts;
    result.num_shared_experts = model.num_shared_experts;
    result.moe_tensor_parallel_size = parallelism.moe_tensor_parallel_size;
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
    result.shared_expert = detail::precision_from_string(
        config.shared_expert_weight_precision());
    result.expert_weight = detail::precision_from_string(
        config.routed_expert_weight_precision());
    result.expert_activation = detail::precision_from_string(
        config.routed_expert_activation_precision());
    result.latent_moe_projection_weight = detail::precision_from_string(
        config.latent_moe_projection_weight_precision());
    result.latent_moe_projection_activation = detail::precision_from_string(
        config.latent_moe_projection_activation_precision());
    result.shared_expert_weight = detail::precision_from_string(
        config.shared_expert_weight_precision());
    result.shared_expert_activation = detail::precision_from_string(
        config.shared_expert_activation_precision());
    result.router_weight = detail::precision_from_string(
        config.router_weight_storage_precision());
    result.router_activation = detail::precision_from_string(
        config.moe_router_activation_precision());
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
    result.lane_times_ms = lane_times_ms(lane_prediction);
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

struct MoEStagePrediction {
    entities::ExecutionTime execution_time;
    std::vector<std::pair<std::string, double>> diagnostics;
    std::vector<MoERoutingDiagnostic> routing_diagnostics;
    std::uint64_t logical_moe_layer_count = 0;
    std::vector<ScaledMoEAttentionGroup> scaled_attention_groups;
    double repeated_moe_layer_pre_compute_ms = 0.0;
    double suffix_compute_ms = 0.0;
    double suffix_tp_communication_ms = 0.0;
};

ScaledMoEAttentionFamily
scaled_attention_family(const config::ModelConfig &model,
                        std::uint64_t model_layer) noexcept {
    if (model.has_kda() && model.is_kda_layer(model_layer)) {
        return ScaledMoEAttentionFamily::kKda;
    }
    if (model.is_mla_layer(model_layer) ||
        (model.use_mla && !model.has_kda() && model.mla_layer_indices.empty())) {
        return ScaledMoEAttentionFamily::kMla;
    }
    return ScaledMoEAttentionFamily::kStandard;
}

// Routing allocations generated by these two modes are independent of the
// logical model layer.  Every other mode/distribution intentionally includes
// the layer ID in its seed or weight calculation and must be recomputed by
// stage_group_scaled rather than reusing a representative lane allocation.
bool routing_is_layer_invariant(
    const config::MoeRoutingConfig &routing) noexcept {
    if (routing.mode == config::MoeRoutingMode::kUniformLegacy) {
        return true;
    }
    return routing.mode == config::MoeRoutingMode::kSimulation &&
           routing.distribution == config::MoeRoutingDistribution::kBalanced;
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

void add_scaled_attention_layer(
    std::vector<ScaledMoEAttentionGroup> &groups,
    ScaledMoEAttentionFamily family, double pre_moe_compute_ms,
    double pre_moe_tp_communication_ms) {
    const auto position = std::find_if(
        groups.begin(), groups.end(), [family](const auto &group) {
            return group.family == family;
        });
    if (position == groups.end()) {
        groups.push_back(ScaledMoEAttentionGroup{
            family, 1, pre_moe_compute_ms, pre_moe_tp_communication_ms});
        return;
    }
    ++position->layer_count;
    // The analytical roofline uses one representative prediction per
    // attention family.  Fail fast if a future layer-dependent operator is
    // accidentally folded into this compression mode.
    const double tolerance =
        1e-12 * std::max({1.0, std::abs(position->pre_moe_compute_ms_per_layer),
                          std::abs(pre_moe_compute_ms)});
    if (std::abs(position->pre_moe_compute_ms_per_layer -
                 pre_moe_compute_ms) > tolerance) {
        throw ExecutionTimePredictorError(
            "scaled MoE prediction requires identical attention time within "
            "each attention family");
    }
    const double communication_tolerance =
        1e-12 * std::max(
                    {1.0,
                     std::abs(
                         position->pre_moe_tp_communication_ms_per_layer),
                     std::abs(pre_moe_tp_communication_ms)});
    if (std::abs(position->pre_moe_tp_communication_ms_per_layer -
                 pre_moe_tp_communication_ms) > communication_tolerance) {
        throw ExecutionTimePredictorError(
            "scaled MoE prediction requires identical attention communication "
            "within each attention family");
    }
}

struct MoEStageContext {
    ClusterType cluster_type;
    const StageBatchInfo &batch_info;
    const config::AnalyticalExecutionModelConfig &config;
    const detail::DeviceCeilings &device;
    const config::ParallelismConfig &parallelism;
    const config::ModelConfig &model;
    const config::MoeRoutingConfig &routing;
    const cc_backend::BaseCCBackend &communication_backend;
    config::PipelineStageLayerRange stage_layers;
    const std::vector<detail::DenseLayerTimes> &layer_times;
    double allreduce_ms;
    double dcp_attention_communication_ms;
    double communication_element_bytes;
    entities::ExecutionTime base_execution_time;
    // stage_group_scaled may provide a lane/communication template computed
    // for an equivalent stage group.  The pointer is valid for the duration
    // of the enclosing predictor call; null preserves detailed routing.
    const detail::MoELanePrediction *reusable_moe_lane_prediction = nullptr;
    const detail::MoECommunicationTime *reusable_moe_communication = nullptr;

    [[nodiscard]] const detail::DenseLayerTimes &
    layer_time(std::uint64_t model_layer) const noexcept {
        return layer_times[static_cast<std::size_t>(model_layer -
                                                    stage_layers.begin)];
    }

    [[nodiscard]] double
    dense_layer_compute_ms(std::uint64_t model_layer) const noexcept {
        return layer_time(model_layer).total_ms();
    }

    [[nodiscard]] double
    attention_compute_ms(std::uint64_t model_layer) const noexcept {
        return total_attention_layer_compute_ms(layer_time(model_layer));
    }

    [[nodiscard]] bool is_mla_layer(std::uint64_t model_layer) const noexcept {
        return !(model.has_kda() && model.is_kda_layer(model_layer));
    }

    [[nodiscard]] double tp_layer_ms(std::uint64_t model_layer) const noexcept {
        return 2.0 * allreduce_ms + (is_mla_layer(model_layer)
                                         ? dcp_attention_communication_ms
                                         : 0.0);
    }

    [[nodiscard]] double
    attention_communication_ms(std::uint64_t model_layer) const noexcept {
        return allreduce_ms + (is_mla_layer(model_layer)
                                   ? dcp_attention_communication_ms
                                   : 0.0);
    }
};

MoEStagePrediction
predict_selected_moe_layer_execution(const MoEStageContext &context,
                                     std::uint64_t selected_moe_layer) {
    const StageBatchInfo &batch_info = context.batch_info;
    const config::ParallelismConfig &parallelism = context.parallelism;
    const config::ModelConfig &model = context.model;
    MoEStagePrediction result{};
    result.execution_time = context.base_execution_time;
    result.execution_time.dense_compute_ms = 0.0;
    result.execution_time.tp_communication_ms = 0.0;
    result.execution_time.moe_gating_linear_ms = 0.0;
    result.execution_time.moe_gating_routing_topk_ms = 0.0;
    result.execution_time.moe_grouped_gemm_ms = 0.0;
    result.execution_time.moe_shuffling_ms = 0.0;
    result.execution_time.moe_post_attention_norm_ms = 0.0;
    result.execution_time.moe_tp_communication_ms = 0.0;
    result.execution_time.ep_dispatch_ms = 0.0;
    result.execution_time.ep_combine_ms = 0.0;
    result.execution_time.dp_input_communication_ms = 0.0;
    result.execution_time.dp_output_communication_ms = 0.0;

    const detail::MoEModel moe_model = make_moe_model(model, parallelism);
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
            const detail::RoutingAllocation allocation = detail::route_tokens(
                batch_info.dense_batch.total_tokens, model.router_topk,
                model.total_expert_num, parallelism.moe_expert_parallel_size,
                context.routing, model_layer);
            const detail::MoELanePrediction lane_prediction =
                context.reusable_moe_lane_prediction != nullptr
                    ? *context.reusable_moe_lane_prediction
                    : detail::predict_moe_lanes(
                          context.device, detail::AnalyticalConfig{},
                          moe_model, allocation, model.router_topk,
                          moe_precisions);
            result.routing_diagnostics.push_back(make_moe_routing_diagnostic(
                local_moe_layer, model_layer, pre_moe_compute_ms,
                pre_moe_tp_communication_ms, allocation, lane_prediction));
            const detail::MoELayerTime &critical =
                lane_prediction.lane_times.at(
                    static_cast<std::size_t>(lane_prediction.critical_lane));
            add_critical_moe_layer_time(
                result.execution_time, critical,
                context.layer_time(model_layer).residual_add_ms);
            result.diagnostics.emplace_back(
                "layer_" + std::to_string(model_layer) + "_critical_lane",
                static_cast<double>(lane_prediction.critical_lane));
            result.diagnostics.emplace_back(
                "layer_" + std::to_string(model_layer) + "_critical_lane_ms",
                lane_prediction.critical_lane_time_ms);
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

    const detail::MoECommunicationTime communication_time =
        context.reusable_moe_communication != nullptr
            ? *context.reusable_moe_communication
            : detail::predict_moe_communication(
                  context.communication_backend,
                  batch_info.dense_batch.total_tokens, model.hidden_size,
                  batch_info.dense_batch.total_tokens * model.router_topk,
                  parallelism.tensor_parallel_size,
                  parallelism.moe_tensor_parallel_size,
                  parallelism.moe_expert_parallel_size,
                  parallelism.data_parallel_size, false,
                  context.communication_element_bytes,
                  model.routed_expert_hidden_size);
    result.execution_time.moe_tp_communication_ms =
        communication_time.moe_tp_ms;
    result.execution_time.ep_dispatch_ms = communication_time.ep_dispatch_ms;
    result.execution_time.ep_combine_ms = communication_time.ep_combine_ms;
    if (context.cluster_type == ClusterType::kDecode) {
        result.execution_time.dp_input_communication_ms =
            communication_time.dp_input_ms;
        result.execution_time.dp_output_communication_ms =
            communication_time.dp_output_ms;
    }
    return result;
}

MoEStagePrediction predict_moe_stage_execution(const MoEStageContext &context) {
    const StageBatchInfo &batch_info = context.batch_info;
    const config::AnalyticalExecutionModelConfig &config = context.config;
    const config::ParallelismConfig &parallelism = context.parallelism;
    const config::ModelConfig &model = context.model;
    const config::MoeRoutingConfig &routing = context.routing;
    const config::PipelineStageLayerRange &stage_layers = context.stage_layers;
    MoEStagePrediction result{};
    result.execution_time = context.base_execution_time;
    result.execution_time.dense_compute_ms = 0.0;
    result.execution_time.tp_communication_ms = 0.0;
    result.execution_time.moe_gating_linear_ms = 0.0;
    result.execution_time.moe_gating_routing_topk_ms = 0.0;
    result.execution_time.moe_grouped_gemm_ms = 0.0;
    result.execution_time.moe_shuffling_ms = 0.0;
    result.execution_time.moe_post_attention_norm_ms = 0.0;
    const detail::MoEModel moe_model = make_moe_model(model, parallelism);
    const detail::MoEOperatorPrecisions moe_precisions =
        make_moe_operator_precisions(config);
    double pending_pre_moe_compute_ms = 0.0;
    double pending_pre_moe_tp_communication_ms = 0.0;
    std::uint64_t moe_layer_index = 0;
    std::optional<detail::MoELanePrediction> repeated_lane_prediction;
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
    for (std::uint64_t model_layer = stage_layers.begin;
         model_layer < stage_layers.end; ++model_layer) {
        if (!model.is_moe_layer(model_layer)) {
            if (first_layer_scaled && repeated_lane_prediction.has_value()) {
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
            repeated_lane_prediction.has_value()) {
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
            pending_pre_moe_compute_ms = 0.0;
            pending_pre_moe_tp_communication_ms = 0.0;
            ++moe_layer_index;
            continue;
        }
        pending_pre_moe_compute_ms += attention_compute_ms;
        const detail::RoutingAllocation allocation = detail::route_tokens(
            batch_info.dense_batch.total_tokens, model.router_topk,
            model.total_expert_num, parallelism.moe_expert_parallel_size,
            routing, model_layer);
        const detail::MoELanePrediction lane_prediction =
            context.reusable_moe_lane_prediction != nullptr
                ? *context.reusable_moe_lane_prediction
                : detail::predict_moe_lanes(
                      context.device, detail::AnalyticalConfig{}, moe_model,
                      allocation, model.router_topk, moe_precisions);
        result.routing_diagnostics.push_back(make_moe_routing_diagnostic(
            moe_layer_index, model_layer, pending_pre_moe_compute_ms,
            pending_pre_moe_tp_communication_ms +
                context.attention_communication_ms(model_layer),
            allocation, lane_prediction));
        const detail::MoELayerTime &critical = lane_prediction.lane_times.at(
            static_cast<std::size_t>(lane_prediction.critical_lane));
        add_critical_moe_layer_time(
            result.execution_time, critical,
            context.layer_time(model_layer).residual_add_ms);
        result.diagnostics.emplace_back(
            "layer_" + std::to_string(model_layer) + "_critical_lane",
            static_cast<double>(lane_prediction.critical_lane));
        result.diagnostics.emplace_back("layer_" + std::to_string(model_layer) +
                                            "_critical_lane_ms",
                                        lane_prediction.critical_lane_time_ms);
        pending_pre_moe_compute_ms = 0.0;
        pending_pre_moe_tp_communication_ms = 0.0;
        ++moe_layer_index;
        if (first_layer_scaled || stage_group_scaled) {
            repeated_lane_prediction = lane_prediction;
            result.repeated_moe_layer_pre_compute_ms = attention_compute_ms;
        }
    }
    result.suffix_compute_ms = pending_pre_moe_compute_ms;
    result.suffix_tp_communication_ms =
        pending_pre_moe_tp_communication_ms;
    const detail::MoECommunicationTime communication_time =
        context.reusable_moe_communication != nullptr
            ? *context.reusable_moe_communication
            : detail::predict_moe_communication(
                  context.communication_backend,
                  batch_info.dense_batch.total_tokens, model.hidden_size,
                  batch_info.dense_batch.total_tokens * model.router_topk,
                  parallelism.tensor_parallel_size,
                  parallelism.moe_tensor_parallel_size,
                  parallelism.moe_expert_parallel_size,
                  parallelism.data_parallel_size, false,
                  context.communication_element_bytes,
                  model.routed_expert_hidden_size);
    const double moe_layers =
        static_cast<double>(result.logical_moe_layer_count);
    result.execution_time.moe_tp_communication_ms =
        moe_layers * communication_time.moe_tp_ms;
    result.execution_time.ep_dispatch_ms =
        moe_layers * communication_time.ep_dispatch_ms;
    result.execution_time.ep_combine_ms =
        moe_layers * communication_time.ep_combine_ms;
    if (context.cluster_type == ClusterType::kDecode) {
        result.execution_time.dp_input_communication_ms =
            moe_layers * communication_time.dp_input_ms;
        result.execution_time.dp_output_communication_ms =
            moe_layers * communication_time.dp_output_ms;
    }
    return result;
}

double
kv_cache_bytes_per_token_per_layer(const config::ModelConfig &model,
                                   detail::Precision kv_cache_precision) {
    if (model.use_mla) {
        const double bytes =
            attention::mla_kv_cache_bytes_per_token(attention::MlaKvCacheLayout{
                model.kv_lora_rank,
                model.qk_rope_head_dim,
                detail::bytes_per_element(kv_cache_precision),
                2.0,
            });
        // Hybrid KDA/MLA models only materialize latent KV entries on MLA
        // layers.  Report an average per logical layer for diagnostics and
        // transfer accounting; legacy flat MLA models retain the exact value.
        if (model.has_kda() && model.num_layers != 0) {
            return bytes * static_cast<double>(model.num_mla_layers) /
                   static_cast<double>(model.num_layers);
        }
        return bytes;
    }
    return static_cast<double>(model.runtime_num_kv_heads()) *
           static_cast<double>(model.runtime_head_size()) *
           static_cast<double>(model.kv_factor()) *
           detail::bytes_per_element(kv_cache_precision);
}

} // namespace

config::StageTimingSignature
AnalyticalRooflineExecutionTimePredictor::make_stage_timing_signature(
    std::uint64_t stage) const {
    return config::build_pipeline_stage_timing_signature(
        model_, parallelism_, stage);
}

void AnalyticalRooflineExecutionTimePredictor::build_stage_timing_groups() {
    timing_catalogue_.timing_groups.clear();
    timing_catalogue_.stage_to_timing_group.clear();
    timing_catalogue_.timing_group_multiplicity.clear();
    timing_catalogue_.stage_to_timing_group.reserve(
        static_cast<std::size_t>(parallelism_.pipeline_parallel_size));
    for (std::uint64_t stage = 0;
         stage < parallelism_.pipeline_parallel_size; ++stage) {
        config::StageTimingSignature signature =
            make_stage_timing_signature(stage);
        const auto existing = std::find(timing_catalogue_.timing_groups.begin(),
                                         timing_catalogue_.timing_groups.end(),
                                         signature);
        std::uint32_t group_id = 0;
        if (existing == timing_catalogue_.timing_groups.end()) {
            group_id = static_cast<std::uint32_t>(
                timing_catalogue_.timing_groups.size());
            timing_catalogue_.timing_groups.push_back(std::move(signature));
            timing_catalogue_.timing_group_multiplicity.push_back(0);
        } else {
            group_id = static_cast<std::uint32_t>(
                std::distance(timing_catalogue_.timing_groups.begin(),
                              existing));
        }
        timing_catalogue_.stage_to_timing_group.push_back(group_id);
        ++timing_catalogue_.timing_group_multiplicity.at(group_id);
    }
}

std::size_t AnalyticalRooflineExecutionTimePredictor::
    StageTimingCacheKeyHash::operator()(
        const StageTimingCacheKey &key) const noexcept {
    std::size_t seed = std::hash<std::uint64_t>{}(key.timing_group_id);
    const auto combine = [&seed](std::uint64_t value) {
        constexpr std::size_t kGoldenRatio =
            static_cast<std::size_t>(0x9e3779b97f4a7c15ULL);
        seed ^= std::hash<std::uint64_t>{}(value) + kGoldenRatio +
                (seed << 6U) + (seed >> 2U);
    };
    combine(key.cluster_type);
    combine(key.selected_moe_layer_valid ? 1U : 0U);
    combine(key.selected_moe_layer);
    combine(key.total_tokens);
    const auto combine_requests = [&combine](
                                      const auto &requests) {
        combine(static_cast<std::uint64_t>(requests.size()));
        for (const auto &[query_tokens, past_context] : requests) {
            combine(query_tokens);
            combine(past_context);
        }
    };
    combine_requests(key.prefill_requests);
    combine_requests(key.decode_requests);
    return seed;
}

AnalyticalRooflineExecutionTimePredictor::StageTimingCacheLookup
AnalyticalRooflineExecutionTimePredictor::lookup_stage_timing_template(
    std::uint32_t timing_group_id, const detail::DenseBatch &dense_batch,
    ClusterType cluster_type,
    std::optional<std::uint64_t> selected_moe_layer,
    const config::PipelineStageLayerRange &stage_layers) const {
    StageTimingCacheKey key{};
    key.timing_group_id = timing_group_id;
    key.cluster_type = static_cast<std::uint8_t>(cluster_type);
    key.selected_moe_layer_valid = selected_moe_layer.has_value();
    key.selected_moe_layer = selected_moe_layer.value_or(0);
    key.total_tokens = dense_batch.total_tokens;
    key.prefill_requests.reserve(dense_batch.prefill_requests.size());
    for (const detail::AttentionRequestSlice &slice :
         dense_batch.prefill_requests) {
        key.prefill_requests.emplace_back(slice.query_tokens,
                                           slice.past_context);
    }
    key.decode_requests.reserve(dense_batch.decode_requests.size());
    for (const detail::AttentionRequestSlice &slice :
         dense_batch.decode_requests) {
        key.decode_requests.emplace_back(slice.query_tokens,
                                          slice.past_context);
    }

    // This bound is intentionally small: the useful reuse window is the set
    // of equivalent PP stages for a batch.  Clearing the table at the bound
    // prevents an online workload with ever-changing batch shapes from
    // retaining unbounded request metadata.
    constexpr std::size_t kMaxTimingCacheEntries = 256;
    std::lock_guard<std::mutex> lock(timing_cache_mutex_);
    const auto existing = timing_cache_.find(key);
    if (existing != timing_cache_.end()) {
        ++timing_cache_hits_total_;
        return StageTimingCacheLookup{
            existing->second, true, timing_cache_hits_total_,
            timing_cache_misses_total_, timing_cache_unique_templates_total_,
            static_cast<std::uint64_t>(timing_cache_.size())};
    }

    auto value = std::make_shared<StageTimingCacheValue>();
    const detail::DenseOperatorPrecisions dense_precisions =
        make_dense_operator_precisions(config_);
    detail::DenseModel dense_model = make_dense_model(model_, parallelism_);
    std::optional<detail::DenseLayerTimes> standard_layer_times;
    std::optional<detail::DenseLayerTimes> mla_layer_times;
    std::optional<detail::DenseLayerTimes> kda_layer_times;
    value->layer_times.reserve(static_cast<std::size_t>(stage_layers.size()));
    for (std::uint64_t model_layer = stage_layers.begin;
         model_layer < stage_layers.end; ++model_layer) {
        dense_model.use_kda =
            model_.has_kda() && model_.is_kda_layer(model_layer);
        dense_model.use_mla = !dense_model.use_kda && model_.use_mla;
        std::optional<detail::DenseLayerTimes> *representative =
            dense_model.use_kda
                ? &kda_layer_times
                : (dense_model.use_mla ? &mla_layer_times
                                       : &standard_layer_times);
        if (!representative->has_value()) {
            *representative = detail::predict_dense_layer(
                device_, detail::AnalyticalConfig{}, dense_model, dense_batch,
                dense_precisions);
        }
        value->layer_times.push_back(representative->value());
    }

    const detail::Precision communication_precision =
        detail::precision_from_string(config_.communication_precision());
    const double communication_element_bytes =
        detail::bytes_per_element(communication_precision);
    const std::uint64_t activation_bytes = activation_payload_bytes(
        dense_batch.total_tokens, model_.hidden_size,
        communication_element_bytes);
    value->allreduce_ms =
        parallelism_.tensor_parallel_size > 1
            ? communication_backend_->allreduce_ms(
                  activation_bytes, parallelism_.tensor_parallel_size, true)
            : 0.0;

    if (model_.use_mla && parallelism_.decode_context_parallel_size > 1 &&
        !dense_batch.decode_requests.empty() &&
        std::any_of(value->layer_times.begin(), value->layer_times.end(),
                    [](const detail::DenseLayerTimes &times) {
                        return times.kda_projection_ms == 0.0 &&
                               times.kda_recurrent_ms == 0.0;
                    })) {
        const std::uint64_t local_query_heads =
            model_.num_query_heads / parallelism_.tensor_parallel_size;
        std::uint64_t decode_tokens = 0;
        for (const detail::AttentionRequestSlice &request :
             dense_batch.decode_requests) {
            decode_tokens += request.query_tokens;
        }
        const std::uint64_t query_bytes = activation_payload_bytes(
            decode_tokens,
            local_query_heads *
                (model_.kv_lora_rank + model_.qk_rope_head_dim),
            communication_element_bytes);
        const std::uint64_t gathered_output_bytes = activation_payload_bytes(
            decode_tokens,
            local_query_heads * parallelism_.decode_context_parallel_size *
                model_.v_head_dim,
            communication_element_bytes);
        value->dcp_attention_communication_ms =
            communication_backend_->allgather_ms(
                query_bytes, parallelism_.decode_context_parallel_size, true) +
            communication_backend_->reduce_scatter_ms(
                gathered_output_bytes,
                parallelism_.decode_context_parallel_size, true);
    }

    // stage_group_scaled is only allowed to reuse a representative lane for
    // layer-invariant routing and a contiguous MoE suffix.  The caller only
    // requests this template on that guarded path, so the first MoE layer is
    // a stable canonical route seed for every stage in the timing group.
    if (model_.is_moe() &&
        config_.moe_layer_event_mode == "stage_group_scaled" &&
        routing_is_layer_invariant(routing_) &&
        has_contiguous_moe_suffix(model_, stage_layers)) {
        std::optional<std::uint64_t> first_moe;
        for (std::uint64_t layer = stage_layers.begin;
             layer < stage_layers.end; ++layer) {
            if (model_.is_moe_layer(layer)) {
                first_moe = layer;
                break;
            }
        }
        if (first_moe.has_value()) {
            const detail::MoEModel moe_model =
                make_moe_model(model_, parallelism_);
            const detail::MoEOperatorPrecisions moe_precisions =
                make_moe_operator_precisions(config_);
            const detail::RoutingAllocation allocation = detail::route_tokens(
                dense_batch.total_tokens, model_.router_topk,
                model_.total_expert_num,
                parallelism_.moe_expert_parallel_size, routing_,
                first_moe.value());
            value->representative_moe_lane = detail::predict_moe_lanes(
                device_, detail::AnalyticalConfig{}, moe_model, allocation,
                model_.router_topk, moe_precisions);
            value->has_representative_moe_lane = true;
            value->moe_communication = detail::predict_moe_communication(
                *communication_backend_, dense_batch.total_tokens,
                model_.hidden_size,
                dense_batch.total_tokens * model_.router_topk,
                parallelism_.tensor_parallel_size,
                parallelism_.moe_tensor_parallel_size,
                parallelism_.moe_expert_parallel_size,
                parallelism_.data_parallel_size, false,
                communication_element_bytes,
                model_.routed_expert_hidden_size);
            value->has_moe_communication = true;
        }
    }

    if (timing_cache_.size() >= kMaxTimingCacheEntries) {
        timing_cache_.clear();
    }
    timing_cache_.emplace(std::move(key), value);
    ++timing_cache_misses_total_;
    ++timing_cache_unique_templates_total_;
    return StageTimingCacheLookup{
        std::move(value), false, timing_cache_hits_total_,
        timing_cache_misses_total_, timing_cache_unique_templates_total_,
        static_cast<std::uint64_t>(timing_cache_.size())};
}

AnalyticalRooflineExecutionTimePredictor::
    AnalyticalRooflineExecutionTimePredictor(
        config::AnalyticalExecutionModelConfig config)
    : AnalyticalRooflineExecutionTimePredictor(
          config,
          [&]() {
              config::ParallelismConfig value{};
              value.tensor_parallel_size = config.tensor_parallel_size;
              return value;
          }(),
          config::ModelConfig{}, config::MoeRoutingConfig{}) {}

AnalyticalRooflineExecutionTimePredictor::
    AnalyticalRooflineExecutionTimePredictor(
        config::AnalyticalExecutionModelConfig config,
        config::ParallelismConfig parallelism)
    : AnalyticalRooflineExecutionTimePredictor(std::move(config), parallelism,
                                               config::ModelConfig{},
                                               config::MoeRoutingConfig{}) {}

AnalyticalRooflineExecutionTimePredictor::
    AnalyticalRooflineExecutionTimePredictor(
        config::AnalyticalExecutionModelConfig config,
        config::ParallelismConfig parallelism, config::ModelConfig model,
        config::MoeRoutingConfig routing)
    : AnalyticalRooflineExecutionTimePredictor(
          config, std::move(parallelism), std::move(model), std::move(routing),
          cc_backend::make_analytical_cc_backend([&]() {
              cc_backend::AnalyticalCommunicationConfig value{};
              value.network_bandwidth_gbps = config.network_bandwidth_gbps;
              value.latency_us = config.network_latency_us;
              value.intra_node_bandwidth_gbps =
                  config.intra_node_bandwidth_gbps;
              return value;
          }())) {}

AnalyticalRooflineExecutionTimePredictor::
    AnalyticalRooflineExecutionTimePredictor(
        config::AnalyticalExecutionModelConfig config,
        config::ParallelismConfig parallelism, config::ModelConfig model,
        config::MoeRoutingConfig routing,
        std::shared_ptr<const cc_backend::BaseCCBackend> communication_backend)
    : config_(std::move(config)),
      device_(detail::DeviceCeilings::from_config(config_)),
      parallelism_(parallelism), model_(std::move(model)), routing_(routing),
      communication_backend_(std::move(communication_backend)) {
    config::apply_model_native_precision_defaults(config_, model_);
    if (parallelism_.tensor_parallel_size == 0) {
        parallelism_.tensor_parallel_size = config_.tensor_parallel_size;
    }
    const auto supported_precision = [](std::string_view value) {
        try {
            static_cast<void>(detail::precision_from_string(value));
            return true;
        } catch (const detail::AnalyticalModelError &) {
            return false;
        }
    };
    if (communication_backend_ == nullptr ||
        !supported_precision(config_.precision) ||
        !supported_precision(config_.attention_precision()) ||
        !supported_precision(config_.dense_precision()) ||
        !supported_precision(config_.moe_expert_precision()) ||
        !supported_precision(config_.moe_router_precision()) ||
        !supported_precision(config_.kv_cache_precision()) ||
        !supported_precision(config_.communication_precision()) ||
        !supported_precision(config_.attention_weight_precision()) ||
        !supported_precision(config_.attention_activation_precision()) ||
        !supported_precision(config_.dense_weight_precision()) ||
        !supported_precision(config_.dense_activation_precision()) ||
        !supported_precision(config_.routed_expert_weight_precision()) ||
        !supported_precision(config_.routed_expert_activation_precision()) ||
        !supported_precision(
            config_.latent_moe_projection_weight_precision()) ||
        !supported_precision(
            config_.latent_moe_projection_activation_precision()) ||
        !supported_precision(config_.shared_expert_weight_precision()) ||
        !supported_precision(config_.shared_expert_activation_precision()) ||
        !supported_precision(config_.dense_mlp_weight_precision()) ||
        !supported_precision(config_.dense_mlp_activation_precision()) ||
        !supported_precision(config_.router_weight_storage_precision()) ||
        !supported_precision(config_.moe_router_activation_precision()) ||
        !supported_precision(config_.router_compute_precision()) ||
        !supported_precision(config_.kda_snapshot_precision()) ||
        !supported_precision(config_.lm_head_weight_precision()) ||
        !supported_precision(config_.lm_head_activation_precision()) ||
        (config_.moe_layer_event_mode != "detailed" &&
         config_.moe_layer_event_mode != "first_layer_scaled" &&
         config_.moe_layer_event_mode != "stage_group_scaled") ||
        model_.attention.memory_layout ==
            attention::AttentionMemoryLayout::kFrozenDsa ||
        parallelism_.tensor_parallel_size == 0 ||
        parallelism_.decode_context_parallel_size == 0 ||
        parallelism_.tensor_parallel_size %
                parallelism_.decode_context_parallel_size !=
            0 ||
        (parallelism_.decode_context_parallel_size > 1 && !model_.use_mla) ||
        parallelism_.pipeline_parallel_size == 0 ||
        parallelism_.pipeline_parallel_size > model_.num_layers) {
        throw ExecutionTimePredictorError(
            "unsupported analytical model configuration");
    }
    build_stage_timing_groups();
}

ExecutionTimePrediction
AnalyticalRooflineExecutionTimePredictor::predict_stage_execution_time(
    const entities::Batch &batch,
    const std::vector<entities::Request> &requests, StageId stage_id) const {
    return predict_execution(batch, requests, stage_id, std::nullopt);
}

MoEGroupLayerPrediction
AnalyticalRooflineExecutionTimePredictor::predict_moe_group_layer(
    const MoEGroupLayerInput &input) const {
    if (!input.layer_id.valid() || input.input_tokens == 0 ||
        input.global_expert_tokens.size() != model_.total_expert_num ||
        input.fallback_lane_times_ms.size() !=
            parallelism_.moe_expert_parallel_size) {
        throw ExecutionTimePredictorError(
            "analytical group MoE input has an invalid domain");
    }
    std::uint64_t routed_tokens = 0;
    for (const std::uint64_t count : input.global_expert_tokens) {
        if (count > std::numeric_limits<std::uint64_t>::max() - routed_tokens) {
            throw ExecutionTimePredictorError(
                "analytical group MoE routed token count overflows");
        }
        routed_tokens += count;
    }
    if (routed_tokens != input.routed_tokens ||
        input.input_tokens >
            std::numeric_limits<std::uint64_t>::max() / model_.router_topk ||
        input.input_tokens * model_.router_topk != input.routed_tokens) {
        throw ExecutionTimePredictorError(
            "analytical group MoE token counts are inconsistent");
    }

    const detail::ExpertParallelDomain domain(
        model_.total_expert_num, parallelism_.moe_expert_parallel_size);
    detail::RoutingAllocation allocation{};
    allocation.input_tokens = input.input_tokens;
    allocation.routed_tokens = input.routed_tokens;
    allocation.global_expert_tokens = input.global_expert_tokens;
    allocation.lane_expert_tokens =
        domain.partition(allocation.global_expert_tokens);
    const detail::MoELanePrediction prediction = detail::predict_moe_lanes(
        device_, detail::AnalyticalConfig{},
        make_moe_model(model_, parallelism_), allocation, model_.router_topk,
        make_moe_operator_precisions(config_));

    MoEGroupLayerPrediction result{};
    result.lane_times_ms = lane_times_ms(prediction);
    result.critical_lane = prediction.critical_lane;
    result.critical_lane_time_ms = prediction.critical_lane_time_ms;
    return result;
}

ExecutionTimePrediction
AnalyticalRooflineExecutionTimePredictor::prepare_moe_stage_execution(
    const entities::Batch &batch,
    const std::vector<entities::Request> &requests, StageId stage_id) const {
    if (!model_.is_moe()) {
        throw ExecutionTimePredictorError(
            "dense model cannot prepare lazy MoE execution");
    }
    if (config_.moe_layer_event_mode == "first_layer_scaled" ||
        config_.moe_layer_event_mode == "stage_group_scaled") {
        return predict_execution(batch, requests, stage_id, std::nullopt);
    }
    return predict_execution(batch, requests, stage_id, std::uint64_t{0});
}

ExecutionTimePrediction
AnalyticalRooflineExecutionTimePredictor::predict_moe_layer_execution(
    const entities::Batch &batch,
    const std::vector<entities::Request> &requests, StageId stage_id,
    std::uint64_t local_moe_layer) const {
    if (!model_.is_moe() ||
        config_.moe_layer_event_mode == "first_layer_scaled" ||
        config_.moe_layer_event_mode == "stage_group_scaled") {
        throw ExecutionTimePredictorError(
            "lazy MoE layer prediction requires detailed analytical mode");
    }
    return predict_execution(batch, requests, stage_id, local_moe_layer);
}

ExecutionTimePrediction
AnalyticalRooflineExecutionTimePredictor::predict_execution(
    const entities::Batch &batch,
    const std::vector<entities::Request> &requests, StageId stage_id,
    std::optional<std::uint64_t> selected_moe_layer) const {
    if (!stage_id.valid() ||
        stage_id.index() >= parallelism_.pipeline_parallel_size) {
        throw ExecutionTimePredictorError(
            "analytical stage ID exceeds pipeline size");
    }

    const StageBatchInfo batch_info = build_stage_batch_info(batch, requests);
    const detail::DenseBatch &dense_batch = batch_info.dense_batch;

    const detail::DenseOperatorPrecisions dense_precisions =
        make_dense_operator_precisions(config_);
    const detail::Precision communication_precision =
        detail::precision_from_string(config_.communication_precision());
    const config::PipelineStageLayerRange stage_layers =
        config::pipeline_stage_layer_range(model_.num_layers,
                                           parallelism_.pipeline_parallel_size,
                                           stage_id.index());
    const bool stage_group_requested =
        config_.moe_layer_event_mode == "stage_group_scaled";
    const bool stage_group_active =
        stage_group_requested && model_.is_moe() &&
        routing_is_layer_invariant(routing_) &&
        has_contiguous_moe_suffix(model_, stage_layers);
    const std::uint32_t timing_group_id =
        timing_catalogue_.stage_to_timing_group.at(stage_id.index());
    const std::uint64_t timing_group_multiplicity =
        timing_catalogue_.timing_group_multiplicity.at(timing_group_id);
    std::vector<detail::DenseLayerTimes> layer_times;
    layer_times.reserve(static_cast<std::size_t>(stage_layers.size()));
    StageTimingCacheLookup timing_cache_lookup{};
    std::shared_ptr<const StageTimingCacheValue> timing_template;
    if (stage_group_active && !selected_moe_layer.has_value()) {
        timing_cache_lookup = lookup_stage_timing_template(
            timing_group_id, dense_batch, batch.cluster_type(),
            selected_moe_layer, stage_layers);
        timing_template = timing_cache_lookup.value;
        layer_times = timing_template->layer_times;
    } else {
        detail::DenseModel dense_model = make_dense_model(model_, parallelism_);
        std::optional<detail::DenseLayerTimes> standard_layer_times;
        std::optional<detail::DenseLayerTimes> mla_layer_times;
        std::optional<detail::DenseLayerTimes> kda_layer_times;
        for (std::uint64_t model_layer = stage_layers.begin;
             model_layer < stage_layers.end; ++model_layer) {
            dense_model.use_kda =
                model_.has_kda() && model_.is_kda_layer(model_layer);
            // KDA and MLA are mutually exclusive attention implementations on
            // a logical layer.  Existing non-hybrid MLA checkpoints retain
            // the old path.
            dense_model.use_mla = !dense_model.use_kda && model_.use_mla;
            std::optional<detail::DenseLayerTimes> *representative =
                dense_model.use_kda
                    ? &kda_layer_times
                    : (dense_model.use_mla ? &mla_layer_times
                                           : &standard_layer_times);
            if (!representative->has_value()) {
                *representative = detail::predict_dense_layer(
                    device_, detail::AnalyticalConfig{}, dense_model,
                    dense_batch, dense_precisions);
            }
            layer_times.push_back(representative->value());
        }
    }
    if (layer_times.empty()) {
        throw ExecutionTimePredictorError(
            "analytical stage has no logical layers");
    }

    const double communication_element_bytes =
        detail::bytes_per_element(communication_precision);
    const std::uint64_t activation_bytes =
        activation_payload_bytes(dense_batch.total_tokens, model_.hidden_size,
                                 communication_element_bytes);
    const double allreduce_ms = timing_template != nullptr
                                    ? timing_template->allreduce_ms
                                    : (parallelism_.tensor_parallel_size > 1
                                           ? communication_backend_->allreduce_ms(
                                                 activation_bytes,
                                                 parallelism_.tensor_parallel_size,
                                                 true)
                                           : 0.0);
    double dcp_attention_communication_ms = 0.0;
    if (timing_template != nullptr) {
        dcp_attention_communication_ms =
            timing_template->dcp_attention_communication_ms;
    } else if (model_.use_mla &&
               parallelism_.decode_context_parallel_size > 1 &&
               !dense_batch.decode_requests.empty() &&
               std::any_of(layer_times.begin(), layer_times.end(),
                           [](const detail::DenseLayerTimes &times) {
                               return times.kda_projection_ms == 0.0 &&
                                      times.kda_recurrent_ms == 0.0;
                           })) {
        const std::uint64_t local_query_heads =
            model_.num_query_heads / parallelism_.tensor_parallel_size;
        std::uint64_t decode_tokens = 0;
        for (const detail::AttentionRequestSlice &request :
             dense_batch.decode_requests) {
            decode_tokens += request.query_tokens;
        }
        const std::uint64_t query_bytes = activation_payload_bytes(
            decode_tokens,
            local_query_heads * (model_.kv_lora_rank + model_.qk_rope_head_dim),
            communication_element_bytes);
        const std::uint64_t gathered_output_bytes = activation_payload_bytes(
            decode_tokens,
            local_query_heads * parallelism_.decode_context_parallel_size *
                model_.v_head_dim,
            communication_element_bytes);
        dcp_attention_communication_ms =
            communication_backend_->allgather_ms(
                query_bytes, parallelism_.decode_context_parallel_size, true) +
            communication_backend_->reduce_scatter_ms(
                gathered_output_bytes,
                parallelism_.decode_context_parallel_size, true);
    }
    const std::uint64_t layers_per_stage = stage_layers.size();
    double dense_compute_ms = 0.0;
    double tp_communication_ms = 0.0;
    for (std::size_t index = 0; index < layer_times.size(); ++index) {
        dense_compute_ms += layer_times[index].total_ms();
        const std::uint64_t model_layer =
            stage_layers.begin + static_cast<std::uint64_t>(index);
        const bool kda_layer =
            model_.has_kda() && model_.is_kda_layer(model_layer);
        tp_communication_ms +=
            2.0 * allreduce_ms +
            (kda_layer ? 0.0 : dcp_attention_communication_ms);
    }
    const double first_layer_compute_ms = layer_times.front().total_ms();
    const double first_tp_layer_ms =
        2.0 * allreduce_ms +
        ((model_.has_kda() && model_.is_kda_layer(stage_layers.begin))
             ? 0.0
             : dcp_attention_communication_ms);
    const double pp_communication_ms =
        stage_id.index() + 1 < parallelism_.pipeline_parallel_size
            ? communication_backend_->point_to_point_ms(activation_bytes, true)
            : 0.0;
    entities::ExecutionTime execution_time{};
    execution_time.dense_compute_ms = dense_compute_ms;
    execution_time.tp_communication_ms = tp_communication_ms;
    execution_time.pp_communication_ms = pp_communication_ms;
    if (stage_id.index() + 1 == parallelism_.pipeline_parallel_size) {
        execution_time.lm_head_ms = detail::predict_output_projection_ms(
            device_, detail::AnalyticalConfig{}, batch_info.lm_head_tokens,
            model_.hidden_size, model_.vocab_size,
            parallelism_.tensor_parallel_size,
            detail::precision_from_string(config_.lm_head_weight_precision()),
            detail::precision_from_string(
                config_.lm_head_activation_precision()));
    }
    if (selected_moe_layer.has_value() && selected_moe_layer.value() > 0) {
        execution_time.lm_head_ms = 0.0;
        execution_time.pp_communication_ms = 0.0;
    }
    std::vector<std::pair<std::string, double>> moe_diagnostics;
    std::vector<MoERoutingDiagnostic> routing_diagnostics;
    std::uint64_t logical_moe_layer_count = 0;
    std::vector<ScaledMoEAttentionGroup> scaled_moe_attention_groups;
    double repeated_moe_layer_pre_compute_ms = 0.0;
    double moe_suffix_compute_ms = 0.0;
    double moe_suffix_tp_communication_ms = 0.0;
    if (model_.is_moe()) {
        const MoEStageContext moe_context{
            batch.cluster_type(),
            batch_info,
            config_,
            device_,
            parallelism_,
            model_,
            routing_,
            *communication_backend_,
            stage_layers,
            layer_times,
            allreduce_ms,
            dcp_attention_communication_ms,
            communication_element_bytes,
            execution_time,
            timing_template != nullptr &&
                    timing_template->has_representative_moe_lane
                ? &timing_template->representative_moe_lane
                : nullptr,
            timing_template != nullptr &&
                    timing_template->has_moe_communication
                ? &timing_template->moe_communication
                : nullptr,
        };
        const MoEStagePrediction moe_prediction =
            selected_moe_layer.has_value()
                ? predict_selected_moe_layer_execution(
                      moe_context, selected_moe_layer.value())
                : predict_moe_stage_execution(moe_context);
        execution_time = moe_prediction.execution_time;
        moe_diagnostics = moe_prediction.diagnostics;
        routing_diagnostics = moe_prediction.routing_diagnostics;
        logical_moe_layer_count = moe_prediction.logical_moe_layer_count;
        scaled_moe_attention_groups =
            std::move(moe_prediction.scaled_attention_groups);
        repeated_moe_layer_pre_compute_ms =
            moe_prediction.repeated_moe_layer_pre_compute_ms;
        moe_suffix_compute_ms = moe_prediction.suffix_compute_ms;
        moe_suffix_tp_communication_ms =
            moe_prediction.suffix_tp_communication_ms;
    }
    // A synchronized lazy-MoE stage is assembled from one initial prediction
    // followed by one prediction per additional MoE layer.  Attribute the
    // stage-level PREFILL attention demand once, on the initial prediction,
    // rather than multiplying it by those layer callbacks.  This field is
    // instrumentation only and does not participate in duration_ms.
    if (!selected_moe_layer.has_value() || selected_moe_layer.value() == 0) {
        execution_time.prefill_attention_token_pairs =
            detail::prefill_attention_token_pairs(dense_batch.prefill_requests);
    }
    dense_compute_ms = execution_time.dense_compute_ms;
    tp_communication_ms = execution_time.tp_communication_ms;
    const double duration_ms = execution_time.total_ms();
    if (!std::isfinite(duration_ms) || duration_ms < 0.0) {
        throw ExecutionTimePredictorError(
            "analytical stage duration is invalid");
    }
    std::uint64_t kda_layer_count = 0;
    for (std::uint64_t model_layer = stage_layers.begin;
         model_layer < stage_layers.end; ++model_layer) {
        kda_layer_count += static_cast<std::uint64_t>(
            model_.has_kda() && model_.is_kda_layer(model_layer));
    }
    std::uint64_t mla_layer_count = 0;
    if (model_.use_mla) {
        mla_layer_count = layers_per_stage - kda_layer_count;
    }
    double kda_projection_ms = 0.0;
    double kda_short_conv_ms = 0.0;
    double kda_recurrent_ms = 0.0;
    double kda_gate_norm_ms = 0.0;
    double attn_res_ms = 0.0;
    for (const detail::DenseLayerTimes &times : layer_times) {
        kda_projection_ms += times.kda_projection_ms;
        kda_short_conv_ms += times.kda_short_conv_ms;
        kda_recurrent_ms += times.kda_recurrent_ms;
        kda_gate_norm_ms += times.kda_gate_norm_ms;
        attn_res_ms += times.attn_res_ms;
    }
    const double kv_cache_bytes_per_token =
        kv_cache_bytes_per_token_per_layer(model_, dense_precisions.kv_cache);
    const double kv_cache_rank_local_bytes_per_token =
        model_.use_mla
            ? kv_cache_bytes_per_token /
                  static_cast<double>(parallelism_.decode_context_parallel_size)
            : kv_cache_bytes_per_token;

    ExecutionTimePrediction result{};
    result.duration_ms = duration_ms;
    result.execution_time = execution_time;
    result.diagnostics = {
        {"total_tokens", static_cast<double>(dense_batch.total_tokens)},
        {"stage_id", static_cast<double>(stage_id.index())},
        {"timing_group_id", static_cast<double>(timing_group_id)},
        {"timing_group_multiplicity",
         static_cast<double>(timing_group_multiplicity)},
        {"timing_cache_enabled", timing_template != nullptr ? 1.0 : 0.0},
        {"timing_cache_hit", timing_cache_lookup.hit ? 1.0 : 0.0},
        {"timing_cache_miss",
         timing_template != nullptr && !timing_cache_lookup.hit ? 1.0 : 0.0},
        {"timing_cache_hits_total",
         static_cast<double>(timing_cache_lookup.hits_total)},
        {"timing_cache_misses_total",
         static_cast<double>(timing_cache_lookup.misses_total)},
        {"timing_cache_unique_templates_total",
         static_cast<double>(timing_cache_lookup.unique_templates_total)},
        {"timing_cache_entries",
         static_cast<double>(timing_cache_lookup.entries)},
        {"timing_group_layer_count",
         static_cast<double>(timing_catalogue_.timing_groups.at(timing_group_id)
                                 .ordered_layers.size())},
        {"timing_owns_input_embedding",
         timing_catalogue_.timing_groups.at(timing_group_id)
                 .owns_input_embedding
             ? 1.0
             : 0.0},
        {"timing_owns_final_norm",
         timing_catalogue_.timing_groups.at(timing_group_id).owns_final_norm
             ? 1.0
             : 0.0},
        {"timing_owns_lm_head",
         timing_catalogue_.timing_groups.at(timing_group_id).owns_lm_head
             ? 1.0
             : 0.0},
        {"timing_emits_pp_send",
         timing_catalogue_.timing_groups.at(timing_group_id).emits_pp_send
             ? 1.0
             : 0.0},
        {"stage_group_requested", stage_group_requested ? 1.0 : 0.0},
        {"stage_group_active", stage_group_active ? 1.0 : 0.0},
        {"stage_group_routing_layer_invariant",
         routing_is_layer_invariant(routing_) ? 1.0 : 0.0},
        {
            "prefill_request_count",
            static_cast<double>(dense_batch.prefill_requests.size()),
        },
        {
            "decode_request_count",
            static_cast<double>(dense_batch.decode_requests.size()),
        },
        // Retain the historical single-layer diagnostic as the first logical
        // layer, and expose stage sums/counts for heterogeneous KDA/MLA
        // schedules.
        {"dense_layer_compute_ms", first_layer_compute_ms},
        {"attention_weight_element_bytes",
         detail::bytes_per_element(*dense_precisions.attention_weight)},
        {"attention_activation_element_bytes",
         detail::bytes_per_element(*dense_precisions.attention_activation)},
        {"dense_weight_element_bytes",
         detail::bytes_per_element(*dense_precisions.dense_weight)},
        {"dense_activation_element_bytes",
         detail::bytes_per_element(*dense_precisions.dense_activation)},
        {"routed_expert_weight_element_bytes",
         detail::bytes_per_element(detail::precision_from_string(
             config_.routed_expert_weight_precision()))},
        {"routed_expert_activation_element_bytes",
         detail::bytes_per_element(detail::precision_from_string(
             config_.routed_expert_activation_precision()))},
        {"latent_moe_projection_weight_element_bytes",
         detail::bytes_per_element(detail::precision_from_string(
             config_.latent_moe_projection_weight_precision()))},
        {"latent_moe_projection_activation_element_bytes",
         detail::bytes_per_element(detail::precision_from_string(
             config_.latent_moe_projection_activation_precision()))},
        {"shared_expert_weight_element_bytes",
         detail::bytes_per_element(detail::precision_from_string(
             config_.shared_expert_weight_precision()))},
        {"shared_expert_activation_element_bytes",
         detail::bytes_per_element(detail::precision_from_string(
             config_.shared_expert_activation_precision()))},
        {"router_weight_storage_element_bytes",
         detail::bytes_per_element(detail::precision_from_string(
             config_.router_weight_storage_precision()))},
        {"router_activation_storage_element_bytes",
         detail::bytes_per_element(detail::precision_from_string(
             config_.moe_router_activation_precision()))},
        {"router_compute_element_bytes",
         detail::bytes_per_element(detail::precision_from_string(
             config_.router_compute_precision()))},
        {"kda_snapshot_element_bytes",
         detail::bytes_per_element(detail::precision_from_string(
             config_.kda_snapshot_precision()))},
        {"kv_cache_element_bytes",
         detail::bytes_per_element(dense_precisions.kv_cache)},
        {"kv_cache_bytes_per_token_per_layer", kv_cache_bytes_per_token},
        {"kv_cache_rank_local_bytes_per_token_per_layer",
         kv_cache_rank_local_bytes_per_token},
        {"communication_element_bytes", communication_element_bytes},
        {"tp_allreduce_ms", allreduce_ms},
        {"dcp_attention_communication_ms", dcp_attention_communication_ms},
        {"decode_context_parallel_size",
         static_cast<double>(parallelism_.decode_context_parallel_size)},
        {
            "dense_layer_total_ms",
            first_layer_compute_ms + first_tp_layer_ms,
        },
        {
            "num_layers",
            static_cast<double>(layers_per_stage),
        },
        {"dense_compute_ms", dense_compute_ms},
        {"tp_communication_ms", tp_communication_ms},
        {"kda_layer_count", static_cast<double>(kda_layer_count)},
        {"mla_layer_count", static_cast<double>(mla_layer_count)},
        {"kda_projection_ms", kda_projection_ms},
        {"kda_short_conv_ms", kda_short_conv_ms},
        {"kda_recurrent_ms", kda_recurrent_ms},
        {"kda_gate_norm_ms", kda_gate_norm_ms},
        {"attn_res_ms", attn_res_ms},
        {"routed_expert_hidden_size",
         static_cast<double>(model_.routed_expert_hidden_size)},
        {"latent_moe_use_norm", model_.latent_moe_use_norm ? 1.0 : 0.0},
        {"attn_res_block_size",
         static_cast<double>(model_.attn_res_block_size)},
        {"pp_communication_ms", pp_communication_ms},
        {"lm_head_ms", execution_time.lm_head_ms},
        {"lm_head_tokens", static_cast<double>(batch_info.lm_head_tokens)},
        {"stage_duration_ms", duration_ms},
        {"batch_duration_ms", duration_ms},
    };
    result.moe_routing = std::move(routing_diagnostics);
    result.logical_moe_layer_count = logical_moe_layer_count;
    result.scaled_moe_attention_groups =
        std::move(scaled_moe_attention_groups);
    result.repeated_moe_layer_pre_compute_ms =
        repeated_moe_layer_pre_compute_ms;
    result.moe_suffix_compute_ms = moe_suffix_compute_ms;
    result.moe_suffix_tp_communication_ms =
        moe_suffix_tp_communication_ms;
    result.lazy_moe_layer_prediction = selected_moe_layer.has_value();
    result.scaled_moe_layer_prediction =
        !selected_moe_layer.has_value() &&
        (config_.moe_layer_event_mode == "first_layer_scaled" ||
         stage_group_active);
    result.diagnostics.insert(result.diagnostics.end(), moe_diagnostics.begin(),
                              moe_diagnostics.end());
    return result;
}

} // namespace frontier::execution_time_predictor

namespace frontier::execution_time_predictor::detail {

DeviceCeilings DeviceCeilings::from_config(
    const config::AnalyticalExecutionModelConfig &config) {
    DeviceCeilings result{};
    if (config.device == "rubin") {
        result = rubin();
    } else if (config.device == "gb300") {
        result = gb300();
    } else if (config.device != "custom") {
        throw AnalyticalModelError("unknown analytical device preset: " +
                                   config.device);
    }

    const config::AnalyticalDeviceOverrides &overrides =
        config.device_overrides;
    const auto apply = [](const std::optional<double> &override_value,
                          double &destination) {
        if (override_value.has_value()) {
            destination = override_value.value();
        }
    };
    apply(overrides.hbm_bandwidth_tbps, result.hbm_bandwidth_tbps);
    apply(overrides.fp32_tflops, result.fp32_tflops);
    apply(overrides.fp16_tflops, result.fp16_tflops);
    apply(overrides.fp8_tflops, result.fp8_tflops);
    apply(overrides.fp4_tflops, result.fp4_tflops);

    for (const auto &[value, name] : {
             std::pair{result.hbm_bandwidth_tbps, "HBM bandwidth"},
             std::pair{result.fp32_tflops, "FP32 ceiling"},
             std::pair{result.fp16_tflops, "FP16 ceiling"},
             std::pair{result.fp8_tflops, "FP8 ceiling"},
             std::pair{result.fp4_tflops, "FP4 ceiling"},
         }) {
        if (!std::isfinite(value) || value <= 0.0) {
            throw AnalyticalModelError(std::string{name} +
                                       " must be finite and positive");
        }
    }
    return result;
}

namespace {

void require_finite_nonnegative(double value, const char *field) {
    if (!std::isfinite(value) || value < 0.0) {
        throw AnalyticalModelError(std::string{field} +
                                   " must be finite and nonnegative");
    }
}

void validate_efficiency(const Efficiency &efficiency) {
    for (const auto &[value, name] : {
             std::pair{efficiency.compute, "compute efficiency"},
             std::pair{efficiency.memory, "memory efficiency"},
         }) {
        if (!std::isfinite(value) || value <= 0.0 || value > 1.0) {
            throw AnalyticalModelError(std::string{name} +
                                       " must satisfy 0 < value <= 1");
        }
    }
    if (!std::isfinite(efficiency.overlap_penalty) ||
        efficiency.overlap_penalty < 0.0 || efficiency.overlap_penalty > 1.0) {
        throw AnalyticalModelError(
            "overlap penalty must satisfy 0 <= value <= 1");
    }
}

std::uint64_t dense_ceil_div(std::uint64_t numerator,
                             std::uint64_t denominator) {
    if (denominator == 0) {
        throw AnalyticalModelError("division denominator must be positive");
    }
    return numerator / denominator +
           static_cast<std::uint64_t>(numerator % denominator != 0);
}

double predict_dense_kernel_ms(const DeviceCeilings &device,
                               Precision precision, const KernelWork &work,
                               const Efficiency &efficiency,
                               double launch_latency_us) {
    return predict_roofline(device, precision, work, efficiency,
                            launch_latency_us)
        .predicted_time_ms;
}

} // namespace

double DenseLayerTimes::total_ms() const noexcept {
    return attention_pre_projection_ms + attention_post_projection_ms +
           rope_ms + kv_cache_save_ms + attention_norm_ms +
           attention_inter_norm_ms + attention_wq_projection_ms +
           prefill_attention_ms + decode_attention_ms + mlp_up_projection_ms +
           mlp_activation_ms + mlp_down_projection_ms + mlp_norm_ms +
           2.0 * residual_add_ms + attn_res_ms;
}

Precision precision_from_string(std::string_view precision) {
    if (precision == "fp32") {
        return Precision::kFp32;
    }
    if (precision == "fp16") {
        return Precision::kFp16;
    }
    if (precision == "bf16") {
        return Precision::kBf16;
    }
    if (precision == "fp8") {
        return Precision::kFp8;
    }
    if (precision == "mxfp8") {
        return Precision::kMxFp8;
    }
    if (precision == "int8") {
        return Precision::kInt8;
    }
    if (precision == "fp4") {
        return Precision::kFp4;
    }
    if (precision == "mxfp4") {
        return Precision::kMxFp4;
    }
    if (precision == "int4") {
        return Precision::kInt4;
    }
    throw AnalyticalModelError("unsupported analytical precision: " +
                               std::string{precision});
}

double bytes_per_element(Precision precision) noexcept {
    switch (precision) {
    case Precision::kFp32:
        return 4.0;
    case Precision::kFp16:
    case Precision::kBf16:
        return 2.0;
    case Precision::kFp8:
    case Precision::kInt8:
        return 1.0;
    case Precision::kMxFp8:
        // One byte per value plus one E8M0 scale byte per 32-value block.
        return 1.0 + 1.0 / 32.0;
    case Precision::kFp4:
    case Precision::kInt4:
        return 0.5;
    case Precision::kMxFp4:
        // Two packed values per byte plus one E8M0 scale byte per 32 values.
        return 0.5 + 1.0 / 32.0;
    }
    return 0.0;
}

double peak_tflops(const DeviceCeilings &device, Precision precision) {
    double peak = 0.0;
    switch (precision) {
    case Precision::kFp32:
        peak = device.fp32_tflops;
        break;
    case Precision::kFp16:
    case Precision::kBf16:
        peak = device.fp16_tflops;
        break;
    case Precision::kFp8:
    case Precision::kMxFp8:
    case Precision::kInt8:
        peak = device.fp8_tflops;
        break;
    case Precision::kFp4:
    case Precision::kMxFp4:
    case Precision::kInt4:
        peak = device.fp4_tflops;
        break;
    }
    if (!std::isfinite(peak) || peak <= 0.0) {
        throw AnalyticalModelError(
            "device compute ceiling must be finite and positive");
    }
    return peak;
}

RooflineResult predict_roofline(const DeviceCeilings &device,
                                Precision precision, const KernelWork &work,
                                const Efficiency &efficiency,
                                double kernel_launch_latency_us) {
    require_finite_nonnegative(work.flops, "FLOPs");
    require_finite_nonnegative(work.hbm_bytes, "HBM bytes");
    validate_efficiency(efficiency);
    require_finite_nonnegative(kernel_launch_latency_us,
                               "kernel launch latency");
    if (!std::isfinite(device.hbm_bandwidth_tbps) ||
        device.hbm_bandwidth_tbps <= 0.0) {
        throw AnalyticalModelError("HBM bandwidth must be finite and positive");
    }

    if (work.flops == 0.0 && work.hbm_bytes == 0.0) {
        return [&]() {
            RooflineResult value{};
            value.compute_time_ms = 0.0;
            value.memory_time_ms = 0.0;
            value.launch_time_ms = 0.0;
            value.predicted_time_ms = 0.0;
            value.bottleneck = Bottleneck::kNone;
            return value;
        }();
    }

    const double compute_time_ms =
        work.flops /
        (peak_tflops(device, precision) * 1e12 * efficiency.compute) * 1e3;
    const double memory_time_ms =
        work.hbm_bytes /
        (device.hbm_bandwidth_tbps * 1e12 * efficiency.memory) * 1e3;
    const double launch_time_ms = kernel_launch_latency_us / 1e3;
    const double maximum = std::max(compute_time_ms, memory_time_ms);
    const double minimum = std::min(compute_time_ms, memory_time_ms);
    const double predicted_time_ms =
        launch_time_ms + maximum + efficiency.overlap_penalty * minimum;

    Bottleneck bottleneck = Bottleneck::kHbm;
    if (launch_time_ms >= maximum) {
        bottleneck = Bottleneck::kLaunch;
    } else if (compute_time_ms >= memory_time_ms) {
        bottleneck = Bottleneck::kCompute;
    }

    return [&]() {
        RooflineResult value{};
        value.compute_time_ms = compute_time_ms;
        value.memory_time_ms = memory_time_ms;
        value.launch_time_ms = launch_time_ms;
        value.predicted_time_ms = predicted_time_ms;
        value.bottleneck = bottleneck;
        return value;
    }();
}

KernelWork gemm_work(std::uint64_t m, std::uint64_t k, std::uint64_t n,
                     double element_bytes, std::uint64_t weight_multiplier) {
    return gemm_work(m, k, n, element_bytes, element_bytes, weight_multiplier);
}

KernelWork gemm_work(std::uint64_t m, std::uint64_t k, std::uint64_t n,
                     double weight_element_bytes,
                     double activation_element_bytes,
                     std::uint64_t weight_multiplier) {
    return gemm_work(m, k, n, weight_element_bytes, activation_element_bytes,
                     activation_element_bytes, weight_multiplier);
}

KernelWork gemm_work(std::uint64_t m, std::uint64_t k, std::uint64_t n,
                     double weight_element_bytes,
                     double activation_element_bytes,
                     double output_element_bytes,
                     std::uint64_t weight_multiplier) {
    require_finite_nonnegative(weight_element_bytes, "weight element bytes");
    require_finite_nonnegative(activation_element_bytes,
                               "activation element bytes");
    require_finite_nonnegative(output_element_bytes, "output element bytes");
    if (weight_element_bytes == 0.0 || activation_element_bytes == 0.0 ||
        output_element_bytes == 0.0) {
        throw AnalyticalModelError("GEMM element bytes must be positive");
    }
    if (weight_multiplier == 0) {
        throw AnalyticalModelError("weight multiplier must be positive");
    }
    if (m == 0 || k == 0 || n == 0) {
        return [&]() {
            KernelWork value{};
            value.flops = 0.0;
            value.hbm_bytes = 0.0;
            return value;
        }();
    }

    const double resolved_m = static_cast<double>(m);
    const double resolved_k = static_cast<double>(k);
    const double resolved_n = static_cast<double>(n);
    const double multiplier = static_cast<double>(weight_multiplier);
    return [&]() {
        KernelWork value{};
        value.flops = 2.0 * resolved_m * resolved_k * resolved_n * multiplier;
        value.hbm_bytes =
            activation_element_bytes * resolved_m * resolved_k +
            weight_element_bytes * multiplier * resolved_k * resolved_n +
            output_element_bytes * multiplier * resolved_m * resolved_n;
        return value;
    }();
}

KernelWork streaming_work(double elements_read, double elements_written,
                          double flops, double element_bytes) {
    require_finite_nonnegative(elements_read, "elements read");
    require_finite_nonnegative(elements_written, "elements written");
    require_finite_nonnegative(flops, "streaming FLOPs");
    require_finite_nonnegative(element_bytes, "element bytes");
    if (element_bytes == 0.0) {
        throw AnalyticalModelError("element bytes must be positive");
    }
    return [&]() {
        KernelWork value{};
        value.flops = flops;
        value.hbm_bytes = (elements_read + elements_written) * element_bytes;
        return value;
    }();
}

std::uint64_t prefill_attention_token_pairs(
    const std::vector<AttentionRequestSlice> &requests) {
    const auto checked_add = [](std::uint64_t lhs, std::uint64_t rhs) {
        if (lhs > std::numeric_limits<std::uint64_t>::max() - rhs) {
            throw AnalyticalModelError(
                "PREFILL attention token-pair count overflows uint64");
        }
        return lhs + rhs;
    };
    const auto checked_mul = [](std::uint64_t lhs, std::uint64_t rhs) {
        if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs) {
            throw AnalyticalModelError(
                "PREFILL attention token-pair count overflows uint64");
        }
        return lhs * rhs;
    };

    std::uint64_t total = 0;
    for (const AttentionRequestSlice &request : requests) {
        // q * (q + 1) / 2 is integral for every integer q.  Divide one
        // factor before multiplying so this remains exact and cannot
        // overflow merely while forming q + 1 for an odd uint64 maximum.
        const std::uint64_t triangular =
            request.query_tokens % 2 == 0
                ? checked_mul(request.query_tokens / 2,
                              request.query_tokens + 1)
                : checked_mul(request.query_tokens,
                              request.query_tokens / 2 + 1);
        const std::uint64_t cached_pairs =
            checked_mul(request.query_tokens, request.past_context);
        total = checked_add(total, checked_add(cached_pairs, triangular));
    }
    return total;
}

KernelWork
attention_context_work(const std::vector<AttentionRequestSlice> &requests,
                       std::uint64_t local_query_heads,
                       std::uint64_t local_kv_heads, std::uint64_t head_dim,
                       double activation_element_bytes,
                       double kv_cache_element_bytes) {
    if (local_query_heads == 0 || local_kv_heads == 0 || head_dim == 0) {
        throw AnalyticalModelError("attention dimensions must be positive");
    }
    require_finite_nonnegative(activation_element_bytes,
                               "activation element bytes");
    require_finite_nonnegative(kv_cache_element_bytes,
                               "KV cache element bytes");
    if (activation_element_bytes == 0.0 || kv_cache_element_bytes == 0.0) {
        throw AnalyticalModelError("attention element bytes must be positive");
    }

    double total_flops = 0.0;
    double activation_elements = 0.0;
    double kv_cache_elements = 0.0;
    for (const AttentionRequestSlice &request : requests) {
        const double query_tokens = static_cast<double>(request.query_tokens);
        const double past_context = static_cast<double>(request.past_context);
        const double average_visible_kv =
            past_context + (query_tokens + 1.0) / 2.0;
        total_flops += 4.0 * static_cast<double>(local_query_heads) *
                       static_cast<double>(head_dim) * query_tokens *
                       average_visible_kv;
        const double q_and_output = 2.0 * query_tokens *
                                    static_cast<double>(local_query_heads) *
                                    static_cast<double>(head_dim);
        const double new_kv_input = 2.0 * query_tokens *
                                    static_cast<double>(local_kv_heads) *
                                    static_cast<double>(head_dim);
        const double cached_kv_reads = 2.0 * past_context *
                                       static_cast<double>(local_kv_heads) *
                                       static_cast<double>(head_dim);
        activation_elements += q_and_output + new_kv_input;
        kv_cache_elements += cached_kv_reads;
    }
    return [&]() {
        KernelWork value{};
        value.flops = total_flops;
        value.hbm_bytes = activation_elements * activation_element_bytes +
                          kv_cache_elements * kv_cache_element_bytes;
        return value;
    }();
}

KernelWork
attention_context_work(const std::vector<AttentionRequestSlice> &requests,
                       std::uint64_t local_query_heads,
                       std::uint64_t local_kv_heads, std::uint64_t head_dim,
                       double element_bytes) {
    return attention_context_work(requests, local_query_heads, local_kv_heads,
                                  head_dim, element_bytes, element_bytes);
}

KernelWork mla_unabsorbed_attention_work(
    const std::vector<AttentionRequestSlice> &requests,
    std::uint64_t local_query_heads, std::uint64_t qk_nope_head_dim,
    std::uint64_t qk_rope_head_dim, std::uint64_t v_head_dim,
    double activation_element_bytes, double rope_cache_element_bytes) {
    if (local_query_heads == 0 || qk_nope_head_dim == 0 ||
        qk_rope_head_dim == 0 || v_head_dim == 0) {
        throw AnalyticalModelError("MLA attention dimensions must be positive");
    }
    require_finite_nonnegative(activation_element_bytes,
                               "activation element bytes");
    require_finite_nonnegative(rope_cache_element_bytes,
                               "RoPE cache element bytes");
    if (activation_element_bytes == 0.0 || rope_cache_element_bytes == 0.0) {
        throw AnalyticalModelError("attention element bytes must be positive");
    }

    const double heads = static_cast<double>(local_query_heads);
    const double qk_dim =
        static_cast<double>(qk_nope_head_dim + qk_rope_head_dim);
    const double expanded_kv_dim =
        static_cast<double>(qk_nope_head_dim + v_head_dim);
    const double value_dim = static_cast<double>(v_head_dim);
    const double rope_dim = static_cast<double>(qk_rope_head_dim);
    double total_flops = 0.0;
    double activation_bytes = 0.0;
    double cached_rope_bytes = 0.0;
    for (const AttentionRequestSlice &request : requests) {
        const double query_tokens = static_cast<double>(request.query_tokens);
        const double past_context = static_cast<double>(request.past_context);
        const double visible_kv = past_context + query_tokens;
        const double average_visible_kv =
            past_context + (query_tokens + 1.0) / 2.0;
        total_flops += 2.0 * heads * query_tokens * average_visible_kv *
                       (qk_dim + value_dim);

        // The compute-friendly path materializes expanded NoPE K and V in an
        // HBM workspace. The projection accounts for the workspace write;
        // attention accounts for the read here. RoPE remains unexpanded in
        // the persistent MLA cache and is broadcast across query heads.
        const double query_and_output =
            query_tokens * heads * (qk_dim + value_dim);
        const double expanded_kv = visible_kv * heads * expanded_kv_dim;
        const double new_rope = query_tokens * rope_dim;
        activation_bytes += (query_and_output + expanded_kv + new_rope) *
                            activation_element_bytes;
        cached_rope_bytes += past_context * rope_dim * rope_cache_element_bytes;
    }
    return KernelWork{total_flops, activation_bytes + cached_rope_bytes};
}

KernelWork mla_absorbed_attention_work(
    const std::vector<AttentionRequestSlice> &requests,
    std::uint64_t local_query_heads, std::uint64_t kv_lora_rank,
    std::uint64_t qk_rope_head_dim, double activation_element_bytes,
    double latent_cache_element_bytes, double rope_cache_element_bytes) {
    if (local_query_heads == 0 || kv_lora_rank == 0 || qk_rope_head_dim == 0) {
        throw AnalyticalModelError("MLA attention dimensions must be positive");
    }
    require_finite_nonnegative(activation_element_bytes,
                               "activation element bytes");
    require_finite_nonnegative(latent_cache_element_bytes,
                               "latent cache element bytes");
    require_finite_nonnegative(rope_cache_element_bytes,
                               "RoPE cache element bytes");
    if (activation_element_bytes == 0.0 || latent_cache_element_bytes == 0.0 ||
        rope_cache_element_bytes == 0.0) {
        throw AnalyticalModelError("attention element bytes must be positive");
    }

    const double heads = static_cast<double>(local_query_heads);
    const double latent_dim = static_cast<double>(kv_lora_rank);
    const double rope_dim = static_cast<double>(qk_rope_head_dim);
    double total_flops = 0.0;
    double activation_bytes = 0.0;
    double cache_bytes = 0.0;
    for (const AttentionRequestSlice &request : requests) {
        const double query_tokens = static_cast<double>(request.query_tokens);
        const double past_context = static_cast<double>(request.past_context);
        const double average_visible_kv =
            past_context + (query_tokens + 1.0) / 2.0;
        total_flops += 2.0 * heads * query_tokens * average_visible_kv *
                       (2.0 * latent_dim + rope_dim);

        const double query_and_output =
            query_tokens * heads * (2.0 * latent_dim + rope_dim);
        const double new_latent_and_rope =
            query_tokens * (latent_dim + rope_dim);
        activation_bytes +=
            (query_and_output + new_latent_and_rope) * activation_element_bytes;
        cache_bytes += past_context * (latent_dim * latent_cache_element_bytes +
                                       rope_dim * rope_cache_element_bytes);
    }
    return KernelWork{total_flops, activation_bytes + cache_bytes};
}

namespace {

struct DenseLayerContext {
    const DeviceCeilings &device;
    const AnalyticalConfig &config;
    const DenseModel &model;
    const DenseBatch &batch;
    Precision attention_weight_precision;
    Precision dense_weight_precision;
    double attention_weight_element_bytes;
    double attention_element_bytes;
    double dense_weight_element_bytes;
    double dense_element_bytes;
    double kv_cache_element_bytes;
    double kda_state_element_bytes;
    std::uint64_t local_query_heads;
    std::uint64_t local_kv_heads;
    std::uint64_t local_intermediate;
    std::uint64_t prefill_tokens;
    std::uint64_t decode_tokens;
    std::uint64_t prefill_past_context;
};

struct AttentionLayerWork {
    double pre_projection_ms = 0.0;
    double post_projection_ms = 0.0;
    double inter_norm_ms = 0.0;
    double wq_projection_ms = 0.0;
    double rope_elements = 0.0;
    KernelWork kv_cache_save{0.0, 0.0};
    KernelWork prefill_attention{0.0, 0.0};
    KernelWork decode_attention{0.0, 0.0};
    KernelWork kda_projection{0.0, 0.0};
    KernelWork kda_short_conv{0.0, 0.0};
    KernelWork kda_recurrent_prefill{0.0, 0.0};
    KernelWork kda_recurrent_decode{0.0, 0.0};
    KernelWork kda_gate_norm{0.0, 0.0};
};

KernelWork add_kernel_work(const KernelWork &lhs, const KernelWork &rhs) {
    return KernelWork{lhs.flops + rhs.flops, lhs.hbm_bytes + rhs.hbm_bytes};
}

std::uint64_t
sum_query_tokens(const std::vector<AttentionRequestSlice> &requests) {
    std::uint64_t total = 0;
    for (const AttentionRequestSlice &request : requests) {
        total += request.query_tokens;
    }
    return total;
}

std::uint64_t
sum_past_context(const std::vector<AttentionRequestSlice> &requests) {
    std::uint64_t total = 0;
    for (const AttentionRequestSlice &request : requests) {
        total += request.past_context;
    }
    return total;
}

void validate_dense_layer_inputs(const AnalyticalConfig &config,
                                 const DenseModel &model,
                                 const DenseBatch &batch) {
    if (model.hidden_size == 0 || model.intermediate_size == 0 ||
        model.num_query_heads == 0 || model.num_kv_heads == 0 ||
        model.head_dim == 0 || model.tensor_parallel_size == 0) {
        throw AnalyticalModelError("dense model dimensions must be positive");
    }
    if (config.small_gemm_token_threshold == 0) {
        throw AnalyticalModelError(
            "small GEMM token threshold must be positive");
    }
    if (sum_query_tokens(batch.prefill_requests) +
            sum_query_tokens(batch.decode_requests) !=
        batch.total_tokens) {
        throw AnalyticalModelError(
            "dense batch total_tokens does not match request slices");
    }
    if (model.use_mla && model.use_mfa) {
        throw AnalyticalModelError("MLA and MFA are mutually exclusive");
    }
    if (model.decode_context_parallel_size == 0 ||
        model.tensor_parallel_size % model.decode_context_parallel_size != 0) {
        throw AnalyticalModelError(
            "tensor parallel size must be divisible by decode context "
            "parallel size");
    }
    // Hybrid K3 reuses the same TP ranks as a DCP group only on MLA layers.
    // A KDA layer still carries the replica-level DCP size in DenseModel, but
    // its recurrent state and projections remain sharded solely by TP and do
    // not issue DCP collectives.
    if (model.decode_context_parallel_size > 1 && !model.use_mla &&
        !model.use_kda) {
        throw AnalyticalModelError(
            "decode context parallelism is currently supported only for "
            "MLA or TP-only KDA layers");
    }
    if (model.use_mla &&
        (model.kv_lora_rank == 0 || model.qk_nope_head_dim == 0 ||
         model.qk_rope_head_dim == 0 || model.qk_head_dim == 0 ||
         model.v_head_dim == 0 ||
         model.qk_head_dim !=
             model.qk_nope_head_dim + model.qk_rope_head_dim)) {
        throw AnalyticalModelError(
            "MLA requires consistent latent attention dimensions");
    }
    if (model.use_mfa && (model.share_q_dim == 0 || model.num_kv_heads != 1)) {
        throw AnalyticalModelError(
            "MFA requires share_q_dim and exactly one KV head");
    }
    if (model.use_kda &&
        (model.kda_num_heads == 0 || model.kda_num_k_heads == 0 ||
         model.kda_num_v_heads == 0 || model.kda_key_head_dim == 0 ||
         model.kda_value_head_dim == 0 || model.kda_head_dim == 0 ||
         model.kda_short_conv_kernel_size == 0 ||
         model.kda_conv_state_dim == 0)) {
        throw AnalyticalModelError(
            "KDA requires positive heads, head dimension, short-conv kernel, "
            "and convolution state dimensions");
    }
    if (model.use_kda &&
        (model.kda_num_heads != model.kda_num_k_heads ||
         model.kda_num_heads != model.kda_num_v_heads ||
         model.kda_head_dim != model.kda_key_head_dim ||
         model.kda_head_dim != model.kda_value_head_dim)) {
        throw AnalyticalModelError(
            "KDA currently supports only symmetric Q/K/V head counts and "
            "dimensions");
    }
    if (model.use_kda &&
        (model.kda_num_heads % model.tensor_parallel_size != 0 ||
         model.kda_num_k_heads % model.tensor_parallel_size != 0 ||
         model.kda_num_v_heads % model.tensor_parallel_size != 0)) {
        throw AnalyticalModelError(
            "KDA Q/K/V head counts must be divisible by tensor parallel "
            "size");
    }
}

DenseLayerContext
make_dense_layer_context(const DeviceCeilings &device,
                         const AnalyticalConfig &config,
                         const DenseModel &model, const DenseBatch &batch,
                         const DenseOperatorPrecisions &precisions) {
    const Precision attention_weight_precision =
        precisions.attention_weight.value_or(precisions.attention);
    const Precision attention_activation_precision =
        precisions.attention_activation.value_or(precisions.attention);
    const Precision dense_weight_precision =
        precisions.dense_weight.value_or(precisions.dense);
    const Precision dense_activation_precision =
        precisions.dense_activation.value_or(precisions.dense);
    return DenseLayerContext{
        device,
        config,
        model,
        batch,
        attention_weight_precision,
        dense_weight_precision,
        bytes_per_element(attention_weight_precision),
        bytes_per_element(attention_activation_precision),
        bytes_per_element(dense_weight_precision),
        bytes_per_element(dense_activation_precision),
        bytes_per_element(precisions.kv_cache),
        bytes_per_element(precisions.kda_state),
        dense_ceil_div(model.num_query_heads, model.tensor_parallel_size),
        dense_ceil_div(model.num_kv_heads, model.tensor_parallel_size),
        dense_ceil_div(model.intermediate_size, model.tensor_parallel_size),
        sum_query_tokens(batch.prefill_requests),
        sum_query_tokens(batch.decode_requests),
        sum_past_context(batch.prefill_requests),
    };
}

const Efficiency &gemm_efficiency_for(const DenseLayerContext &context,
                                      std::uint64_t rows) {
    return rows < context.config.small_gemm_token_threshold
               ? context.config.small_gemm
               : context.config.large_gemm;
}

double predict_attention_work_ms(const DenseLayerContext &context,
                                 const KernelWork &work,
                                 const Efficiency &efficiency) {
    return predict_dense_kernel_ms(
        context.device, context.attention_weight_precision, work, efficiency,
        context.config.kernel_launch_latency_us);
}

double predict_dense_work_ms(const DenseLayerContext &context,
                             const KernelWork &work,
                             const Efficiency &efficiency) {
    return predict_dense_kernel_ms(
        context.device, context.dense_weight_precision, work, efficiency,
        context.config.kernel_launch_latency_us);
}

KernelWork attention_gemm_work(const DenseLayerContext &context,
                               std::uint64_t m, std::uint64_t k,
                               std::uint64_t n, std::uint64_t multiplier = 1) {
    return gemm_work(m, k, n, context.attention_weight_element_bytes,
                     context.attention_element_bytes, multiplier);
}

KernelWork dense_gemm_work(const DenseLayerContext &context, std::uint64_t m,
                           std::uint64_t k, std::uint64_t n,
                           std::uint64_t multiplier = 1) {
    return gemm_work(m, k, n, context.dense_weight_element_bytes,
                     context.dense_element_bytes, multiplier);
}

KernelWork per_head_gemm_work(const DenseLayerContext &context,
                              std::uint64_t rows, std::uint64_t k,
                              std::uint64_t n) {
    if (rows == 0) {
        return KernelWork{0.0, 0.0};
    }
    const double resolved_rows = static_cast<double>(rows);
    const double resolved_heads =
        static_cast<double>(context.local_query_heads);
    const double resolved_k = static_cast<double>(k);
    const double resolved_n = static_cast<double>(n);
    return KernelWork{
        2.0 * resolved_rows * resolved_heads * resolved_k * resolved_n,
        context.attention_element_bytes * resolved_rows * resolved_heads *
                resolved_k +
            context.attention_weight_element_bytes * resolved_heads *
                resolved_k * resolved_n +
            context.attention_element_bytes * resolved_rows * resolved_heads *
                resolved_n,
    };
}

KernelWork mla_unabsorbed_kv_expansion_work(const DenseLayerContext &context,
                                            std::uint64_t expanded_kv_dim) {
    const std::uint64_t visible_tokens =
        context.prefill_tokens + context.prefill_past_context;
    if (visible_tokens == 0) {
        return KernelWork{0.0, 0.0};
    }
    const double visible = static_cast<double>(visible_tokens);
    const double current = static_cast<double>(context.prefill_tokens);
    const double cached = static_cast<double>(context.prefill_past_context);
    const double latent = static_cast<double>(context.model.kv_lora_rank);
    const double output = static_cast<double>(expanded_kv_dim);
    return KernelWork{
        2.0 * visible * latent * output,
        current * latent * context.attention_element_bytes +
            cached * latent * context.kv_cache_element_bytes +
            latent * output * context.attention_weight_element_bytes +
            visible * output * context.attention_element_bytes,
    };
}

std::uint64_t
mla_dcp_busiest_new_token_count(const DenseBatch &batch,
                                std::uint64_t decode_context_parallel_size) {
    std::vector<std::uint64_t> tokens_by_rank(
        static_cast<std::size_t>(decode_context_parallel_size), 0);
    const auto add_requests = [&](const auto &requests) {
        for (const AttentionRequestSlice &request : requests) {
            if (request.query_tokens >
                std::numeric_limits<std::uint64_t>::max() -
                    request.past_context) {
                throw AnalyticalModelError(
                    "MLA DCP token interval overflows uint64");
            }
            const std::uint64_t end =
                request.past_context + request.query_tokens;
            for (std::uint64_t rank = 0; rank < decode_context_parallel_size;
                 ++rank) {
                tokens_by_rank[static_cast<std::size_t>(rank)] +=
                    attention::mla_dcp_local_token_count(
                        end, decode_context_parallel_size, rank) -
                    attention::mla_dcp_local_token_count(
                        request.past_context, decode_context_parallel_size,
                        rank);
            }
        }
    };
    add_requests(batch.prefill_requests);
    add_requests(batch.decode_requests);
    return *std::max_element(tokens_by_rank.begin(), tokens_by_rank.end());
}

AttentionLayerWork
predict_mla_attention_work(const DenseLayerContext &context) {
    constexpr double kMlaRopeCacheElementBytes = 2.0;
    const DenseModel &model = context.model;
    const std::uint64_t latent_and_rope_dim =
        model.kv_lora_rank + model.qk_rope_head_dim;
    const std::uint64_t expanded_kv_dim =
        context.local_query_heads * (model.qk_nope_head_dim + model.v_head_dim);
    const std::uint64_t prefill_visible_tokens =
        context.prefill_tokens + context.prefill_past_context;
    const double tokens = static_cast<double>(context.batch.total_tokens);
    AttentionLayerWork work{};

    if (model.q_lora_rank == 0) {
        work.pre_projection_ms = predict_attention_work_ms(
            context,
            attention_gemm_work(context, context.batch.total_tokens,
                                model.hidden_size,
                                context.local_query_heads * model.qk_head_dim),
            gemm_efficiency_for(context, context.batch.total_tokens));
    } else {
        work.pre_projection_ms =
            predict_attention_work_ms(
                context,
                attention_gemm_work(context, context.batch.total_tokens,
                                    model.hidden_size, model.q_lora_rank),
                gemm_efficiency_for(context, context.batch.total_tokens)) +
            predict_attention_work_ms(
                context,
                attention_gemm_work(
                    context, context.batch.total_tokens, model.q_lora_rank,
                    context.local_query_heads * model.qk_head_dim),
                gemm_efficiency_for(context, context.batch.total_tokens));
        work.inter_norm_ms += predict_attention_work_ms(
            context,
            streaming_work(tokens * static_cast<double>(model.q_lora_rank),
                           tokens * static_cast<double>(model.q_lora_rank),
                           5.0 * tokens *
                               static_cast<double>(model.q_lora_rank),
                           context.attention_element_bytes),
            context.config.streaming);
    }
    work.pre_projection_ms += predict_attention_work_ms(
        context,
        attention_gemm_work(context, context.batch.total_tokens,
                            model.hidden_size, latent_and_rope_dim),
        gemm_efficiency_for(context, context.batch.total_tokens));
    work.inter_norm_ms += predict_attention_work_ms(
        context,
        streaming_work(tokens * static_cast<double>(model.kv_lora_rank),
                       tokens * static_cast<double>(model.kv_lora_rank),
                       5.0 * tokens * static_cast<double>(model.kv_lora_rank),
                       context.attention_element_bytes),
        context.config.streaming);
    work.pre_projection_ms += predict_attention_work_ms(
        context, mla_unabsorbed_kv_expansion_work(context, expanded_kv_dim),
        gemm_efficiency_for(context, prefill_visible_tokens));
    work.pre_projection_ms += predict_attention_work_ms(
        context,
        per_head_gemm_work(context, context.decode_tokens,
                           model.qk_nope_head_dim, model.kv_lora_rank),
        gemm_efficiency_for(context, context.decode_tokens));
    work.post_projection_ms += predict_attention_work_ms(
        context,
        attention_gemm_work(context, context.prefill_tokens,
                            context.local_query_heads * model.v_head_dim,
                            model.hidden_size),
        gemm_efficiency_for(context, context.prefill_tokens));
    work.post_projection_ms += predict_attention_work_ms(
        context,
        per_head_gemm_work(context, context.decode_tokens, model.kv_lora_rank,
                           model.v_head_dim),
        gemm_efficiency_for(context, context.decode_tokens));
    work.post_projection_ms += predict_attention_work_ms(
        context,
        attention_gemm_work(context, context.decode_tokens,
                            context.local_query_heads * model.v_head_dim,
                            model.hidden_size),
        gemm_efficiency_for(context, context.decode_tokens));

    if (model.mla_use_output_gate) {
        // Kimi K3 computes a full-rank sigmoid gate from the layer input,
        // applies it elementwise to the per-head attention output, and only
        // then runs o_proj.  Account for the gate projection with the
        // attention weight/activation dtypes and for the fused sigmoid +
        // multiply as a streaming kernel.  The latter reads both the gate
        // and attention output and writes the gated output once.
        const std::uint64_t gate_dim =
            context.local_query_heads * model.v_head_dim;
        work.post_projection_ms += predict_attention_work_ms(
            context,
            attention_gemm_work(context, context.batch.total_tokens,
                                model.hidden_size, gate_dim),
            gemm_efficiency_for(context, context.batch.total_tokens));
        const double gate_elements = tokens * static_cast<double>(gate_dim);
        work.post_projection_ms += predict_attention_work_ms(
            context,
            streaming_work(2.0 * gate_elements, gate_elements,
                           8.0 * gate_elements,
                           context.attention_element_bytes),
            context.config.streaming);
    }

    work.rope_elements =
        model.mla_use_nope
            ? 0.0
            : tokens * (static_cast<double>(context.local_query_heads) + 1.0) *
                  static_cast<double>(model.qk_rope_head_dim);
    const double mla_cache_bytes_per_token =
        attention::mla_kv_cache_bytes_per_token(attention::MlaKvCacheLayout{
            model.kv_lora_rank,
            model.qk_rope_head_dim,
            context.kv_cache_element_bytes,
            kMlaRopeCacheElementBytes,
        });
    const double rank_local_cache_tokens =
        static_cast<double>(mla_dcp_busiest_new_token_count(
            context.batch, context.model.decode_context_parallel_size));
    work.kv_cache_save = KernelWork{
        0.0,
        tokens * static_cast<double>(latent_and_rope_dim) *
                context.attention_element_bytes +
            rank_local_cache_tokens * mla_cache_bytes_per_token,
    };
    work.prefill_attention = mla_unabsorbed_attention_work(
        context.batch.prefill_requests, context.local_query_heads,
        model.qk_nope_head_dim, model.qk_rope_head_dim, model.v_head_dim,
        context.attention_element_bytes, kMlaRopeCacheElementBytes);
    std::vector<AttentionRequestSlice> dcp_decode_requests =
        context.batch.decode_requests;
    if (context.model.decode_context_parallel_size > 1) {
        for (AttentionRequestSlice &request : dcp_decode_requests) {
            request.past_context = attention::mla_dcp_local_token_count(
                request.past_context,
                context.model.decode_context_parallel_size, 0);
        }
    }
    work.decode_attention = mla_absorbed_attention_work(
        dcp_decode_requests,
        context.local_query_heads * context.model.decode_context_parallel_size,
        model.kv_lora_rank, model.qk_rope_head_dim,
        context.attention_element_bytes, context.kv_cache_element_bytes,
        kMlaRopeCacheElementBytes);
    return work;
}

AttentionLayerWork
predict_mfa_attention_work(const DenseLayerContext &context) {
    const DenseModel &model = context.model;
    const double tokens = static_cast<double>(context.batch.total_tokens);
    const double query_heads = static_cast<double>(context.local_query_heads);
    const double kv_heads = static_cast<double>(context.local_kv_heads);
    const double head_dim = static_cast<double>(model.head_dim);
    const std::uint64_t replicated_qkv_dim =
        model.share_q_dim + 2 * context.local_kv_heads * model.head_dim;
    AttentionLayerWork work{};
    work.pre_projection_ms = predict_attention_work_ms(
        context,
        attention_gemm_work(context, context.batch.total_tokens,
                            model.hidden_size, replicated_qkv_dim),
        gemm_efficiency_for(context, context.batch.total_tokens));
    work.inter_norm_ms = predict_attention_work_ms(
        context,
        streaming_work(tokens * static_cast<double>(model.share_q_dim),
                       tokens * static_cast<double>(model.share_q_dim),
                       5.0 * tokens * static_cast<double>(model.share_q_dim),
                       context.attention_element_bytes),
        context.config.streaming);
    work.wq_projection_ms = predict_attention_work_ms(
        context,
        attention_gemm_work(context, context.batch.total_tokens,
                            model.share_q_dim,
                            context.local_query_heads * model.head_dim),
        gemm_efficiency_for(context, context.batch.total_tokens));
    work.post_projection_ms = predict_attention_work_ms(
        context,
        attention_gemm_work(context, context.batch.total_tokens,
                            context.local_query_heads * model.head_dim,
                            model.hidden_size),
        gemm_efficiency_for(context, context.batch.total_tokens));
    work.rope_elements = tokens * (query_heads + kv_heads) * head_dim;
    const double kv_elements = tokens * 2.0 * kv_heads * head_dim;
    work.kv_cache_save = KernelWork{
        0.0,
        kv_elements *
            (context.attention_element_bytes + context.kv_cache_element_bytes),
    };
    work.prefill_attention = attention_context_work(
        context.batch.prefill_requests, context.local_query_heads,
        context.local_kv_heads, model.head_dim, context.attention_element_bytes,
        context.kv_cache_element_bytes);
    work.decode_attention = attention_context_work(
        context.batch.decode_requests, context.local_query_heads,
        context.local_kv_heads, model.head_dim, context.attention_element_bytes,
        context.kv_cache_element_bytes);
    return work;
}

AttentionLayerWork
predict_mha_attention_work(const DenseLayerContext &context) {
    const DenseModel &model = context.model;
    const double tokens = static_cast<double>(context.batch.total_tokens);
    const double query_heads = static_cast<double>(context.local_query_heads);
    const double kv_heads = static_cast<double>(context.local_kv_heads);
    const double head_dim = static_cast<double>(model.head_dim);
    const std::uint64_t local_qkv_dim =
        (context.local_query_heads + 2 * context.local_kv_heads) *
        model.head_dim;
    AttentionLayerWork work{};
    work.pre_projection_ms = predict_attention_work_ms(
        context,
        attention_gemm_work(context, context.batch.total_tokens,
                            model.hidden_size, local_qkv_dim),
        gemm_efficiency_for(context, context.batch.total_tokens));
    work.post_projection_ms = predict_attention_work_ms(
        context,
        attention_gemm_work(
            context, context.batch.total_tokens,
            std::max<std::uint64_t>(1, model.hidden_size /
                                           model.tensor_parallel_size),
            model.hidden_size),
        gemm_efficiency_for(context, context.batch.total_tokens));
    work.rope_elements = tokens * (query_heads + kv_heads) * head_dim;
    const double kv_elements = tokens * 2.0 * kv_heads * head_dim;
    work.kv_cache_save = KernelWork{
        0.0,
        kv_elements *
            (context.attention_element_bytes + context.kv_cache_element_bytes),
    };
    work.prefill_attention = attention_context_work(
        context.batch.prefill_requests, context.local_query_heads,
        context.local_kv_heads, model.head_dim, context.attention_element_bytes,
        context.kv_cache_element_bytes);
    work.decode_attention = attention_context_work(
        context.batch.decode_requests, context.local_query_heads,
        context.local_kv_heads, model.head_dim, context.attention_element_bytes,
        context.kv_cache_element_bytes);
    return work;
}

// Kimi Delta Attention keeps a fixed recurrent matrix per head instead of
// reading a sequence-length-dependent KV cache.  This helper intentionally
// models the major roofline terms separately: the three Q/K/V projections,
// depthwise short convolutions, the gated delta-rule state update, and the
// output gate/norm.  It is not intended to reproduce a particular CUDA kernel
// schedule; it provides a stable analytical contract that scales with batch
// tokens, KDA dimensions, and the configured short-convolution width.
AttentionLayerWork
predict_kda_attention_work(const DenseLayerContext &context) {
    const DenseModel &model = context.model;
    if (!model.use_kda) {
        throw AnalyticalModelError("KDA work requested for a non-KDA layer");
    }
    const std::uint64_t local_k_heads =
        dense_ceil_div(model.kda_num_k_heads, model.tensor_parallel_size);
    const std::uint64_t local_v_heads =
        dense_ceil_div(model.kda_num_v_heads, model.tensor_parallel_size);
    const std::uint64_t local_key_dim = model.kda_key_head_dim;
    const std::uint64_t local_value_dim = model.kda_value_head_dim;
    const std::uint64_t local_qk_channels = local_k_heads * local_key_dim;
    const std::uint64_t local_v_channels = local_v_heads * local_value_dim;
    if (local_k_heads == 0 || local_v_heads == 0 || local_key_dim == 0 ||
        local_value_dim == 0 || local_qk_channels == 0 ||
        local_v_channels == 0) {
        throw AnalyticalModelError("KDA local dimensions must be positive");
    }
    const std::uint64_t tokens = context.batch.total_tokens;
    const std::uint64_t conv_kernel = model.kda_short_conv_kernel_size;
    const double token_count = static_cast<double>(tokens);
    const double qk_channels = static_cast<double>(local_qk_channels);
    const double v_channels = static_cast<double>(local_v_channels);
    const double conv_channels = static_cast<double>(std::max(
        local_qk_channels * 2 + local_v_channels,
        dense_ceil_div(model.kda_conv_state_dim, model.tensor_parallel_size)));
    const double v_heads = static_cast<double>(local_v_heads);
    const double key_dim = static_cast<double>(local_key_dim);
    const double value_dim = static_cast<double>(local_value_dim);

    AttentionLayerWork work{};

    // q/k use key heads while v and the output gate use value heads.  K3's
    // released dimensions happen to match; keeping them separate also makes
    // the predictor useful for smaller Kimi-Linear checkpoints.
    const KernelWork qk =
        gemm_work(tokens, model.hidden_size, local_qk_channels,
                  context.attention_weight_element_bytes,
                  context.attention_element_bytes, 2);
    const KernelWork v = gemm_work(tokens, model.hidden_size, local_v_channels,
                                   context.attention_weight_element_bytes,
                                   context.attention_element_bytes, 1);
    const KernelWork full_rank_gate =
        gemm_work(tokens, model.hidden_size, local_v_channels,
                  context.attention_weight_element_bytes,
                  context.attention_element_bytes, 1);
    const KernelWork f_a =
        gemm_work(tokens, model.hidden_size, model.kda_head_dim,
                  context.attention_weight_element_bytes,
                  context.attention_element_bytes, 1);
    const KernelWork f_b =
        gemm_work(tokens, model.kda_head_dim, local_v_channels,
                  context.attention_weight_element_bytes,
                  context.attention_element_bytes, 1);
    const KernelWork beta = gemm_work(tokens, model.hidden_size, local_v_heads,
                                      context.attention_weight_element_bytes,
                                      context.attention_element_bytes, 1);
    work.kda_projection = add_kernel_work(
        add_kernel_work(add_kernel_work(qk, v), add_kernel_work(f_a, f_b)),
        add_kernel_work(full_rank_gate, beta));

    // Each of Q/K/V is passed through a depthwise short convolution.  The
    // history window is fixed, so this cost is linear in tokens and does not
    // depend on request past_context.  HBM traffic accounts for a window read
    // and one output write for each channel.
    const double conv_history_elements =
        token_count * conv_channels * static_cast<double>(conv_kernel);
    const double conv_output_elements = token_count * conv_channels;
    const KernelWork short_conv{
        2.0 * conv_history_elements,
        conv_history_elements * context.kda_state_element_bytes +
            conv_output_elements * context.attention_element_bytes,
    };
    work.kda_short_conv = short_conv;

    // The delta-rule update performs a query/state product and a key/value
    // outer-product update for each head.  The fixed recurrent state is
    // touched once per token in this conservative roofline model.  It keeps
    // decode work O(1) in context length while retaining the quadratic head
    // dimension dependence of the state matrix.
    const auto recurrent_work =
        [&](const std::vector<AttentionRequestSlice> &requests) {
            const double request_tokens =
                static_cast<double>(sum_query_tokens(requests));
            const double state_elements = v_heads * key_dim * value_dim;
            // q*S, delta outer-product, decay/gate application, and the output
            // contraction are represented by four matrix-like operations.
            const double flops =
                request_tokens * v_heads * key_dim * value_dim * 8.0;
            const double hbm =
                request_tokens *
                (2.0 * state_elements * context.kda_state_element_bytes +
                 (qk_channels + v_channels) * context.attention_element_bytes);
            return KernelWork{flops, hbm};
        };
    work.kda_recurrent_prefill = recurrent_work(context.batch.prefill_requests);
    work.kda_recurrent_decode = recurrent_work(context.batch.decode_requests);

    // Fused RMSNorm + sigmoid gate: one read/write pass over the projected
    // output with a small constant amount of scalar work per element.
    work.kda_gate_norm = streaming_work(
        token_count * v_channels, token_count * v_channels,
        8.0 * token_count * v_channels, context.attention_element_bytes);

    return work;
}

AttentionLayerWork predict_attention_work(const DenseLayerContext &context) {
    if (context.model.use_kda) {
        return predict_kda_attention_work(context);
    }
    if (context.model.use_mla) {
        return predict_mla_attention_work(context);
    }
    if (context.model.use_mfa) {
        return predict_mfa_attention_work(context);
    }
    return predict_mha_attention_work(context);
}

void populate_attention_times(const DenseLayerContext &context,
                              const AttentionLayerWork &work,
                              DenseLayerTimes &times) {
    if (context.model.use_kda) {
        // Keep the historical attention buckets populated while exposing the
        // KDA sub-components for diagnostics and focused tests.
        times.kda_projection_ms = predict_attention_work_ms(
            context, work.kda_projection,
            gemm_efficiency_for(context, context.batch.total_tokens));
        times.kda_short_conv_ms = predict_attention_work_ms(
            context, work.kda_short_conv, context.config.streaming);
        times.kda_recurrent_ms =
            predict_attention_work_ms(context, work.kda_recurrent_prefill,
                                      context.config.prefill_attention) +
            predict_attention_work_ms(context, work.kda_recurrent_decode,
                                      context.config.decode_attention);
        times.kda_gate_norm_ms = predict_attention_work_ms(
            context, work.kda_gate_norm, context.config.streaming);

        times.attention_pre_projection_ms =
            times.kda_projection_ms + times.kda_short_conv_ms;
        times.attention_post_projection_ms = predict_attention_work_ms(
            context,
            attention_gemm_work(
                context, context.batch.total_tokens,
                dense_ceil_div(context.model.kda_num_v_heads,
                               context.model.tensor_parallel_size) *
                    context.model.kda_value_head_dim,
                context.model.hidden_size),
            gemm_efficiency_for(context, context.batch.total_tokens));
        times.attention_inter_norm_ms = times.kda_gate_norm_ms;
        times.prefill_attention_ms =
            predict_attention_work_ms(context, work.kda_recurrent_prefill,
                                      context.config.prefill_attention);
        times.decode_attention_ms =
            predict_attention_work_ms(context, work.kda_recurrent_decode,
                                      context.config.decode_attention);
        // KDA does not use RoPE or a sequence-growing KV cache.
        times.rope_ms = 0.0;
        times.kv_cache_save_ms = 0.0;
        times.attention_norm_ms = 0.0;
        times.attention_wq_projection_ms = 0.0;
        return;
    }
    const double tokens = static_cast<double>(context.batch.total_tokens);
    const double hidden = static_cast<double>(context.model.hidden_size);
    const double norm_factor = context.model.fused_add_norm ? 3.0 : 2.0;
    times.attention_pre_projection_ms = work.pre_projection_ms;
    times.attention_post_projection_ms = work.post_projection_ms;
    times.rope_ms = predict_attention_work_ms(
        context,
        streaming_work(work.rope_elements, work.rope_elements,
                       6.0 * work.rope_elements,
                       context.attention_element_bytes),
        context.config.streaming);
    times.kv_cache_save_ms = predict_attention_work_ms(
        context, work.kv_cache_save, context.config.streaming);
    times.attention_norm_ms = predict_attention_work_ms(
        context,
        streaming_work(tokens * hidden * (norm_factor - 1.0), tokens * hidden,
                       5.0 * tokens * hidden, context.attention_element_bytes),
        context.config.streaming);
    times.attention_inter_norm_ms = work.inter_norm_ms;
    times.attention_wq_projection_ms = work.wq_projection_ms;
    times.prefill_attention_ms = predict_attention_work_ms(
        context, work.prefill_attention, context.config.prefill_attention);
    times.decode_attention_ms = predict_attention_work_ms(
        context, work.decode_attention, context.config.decode_attention);
}

void populate_dense_mlp_and_norm_times(const DenseLayerContext &context,
                                       DenseLayerTimes &times) {
    const double tokens = static_cast<double>(context.batch.total_tokens);
    const double hidden = static_cast<double>(context.model.hidden_size);
    const double intermediate = static_cast<double>(context.local_intermediate);
    const double activation_elements = tokens * intermediate;
    const std::uint64_t gated_multiplier = context.model.gated_mlp ? 2 : 1;
    const double norm_factor = context.model.fused_add_norm ? 3.0 : 2.0;
    const Efficiency &gemm_efficiency =
        gemm_efficiency_for(context, context.batch.total_tokens);
    times.mlp_up_projection_ms = predict_dense_work_ms(
        context,
        dense_gemm_work(context, context.batch.total_tokens,
                        context.model.hidden_size, context.local_intermediate,
                        gated_multiplier),
        gemm_efficiency);
    times.mlp_activation_ms = predict_dense_work_ms(
        context,
        streaming_work(activation_elements *
                           static_cast<double>(gated_multiplier),
                       activation_elements, 8.0 * activation_elements,
                       context.dense_element_bytes),
        context.config.streaming);
    times.mlp_down_projection_ms = predict_dense_work_ms(
        context,
        dense_gemm_work(context, context.batch.total_tokens,
                        context.local_intermediate, context.model.hidden_size),
        gemm_efficiency);
    times.mlp_norm_ms = predict_dense_work_ms(
        context,
        streaming_work(tokens * hidden * (norm_factor - 1.0), tokens * hidden,
                       5.0 * tokens * hidden, context.dense_element_bytes),
        context.config.streaming);
    if (!context.model.fused_add_norm) {
        const double residual_elements = tokens * hidden;
        times.residual_add_ms = predict_dense_work_ms(
            context,
            streaming_work(2.0 * residual_elements, residual_elements,
                           residual_elements, context.dense_element_bytes),
            context.config.streaming);
    }
    // K3 AttnRes metadata is retained for architecture fidelity, but its
    // operator cost is intentionally omitted.  The operation is expected to
    // be negligible/fused, while the former block-width streaming heuristic
    // had no measured kernel basis and could substantially overcharge HBM.
}

} // namespace

DenseLayerTimes predict_dense_layer(const DeviceCeilings &device,
                                    const AnalyticalConfig &config,
                                    const DenseModel &model,
                                    const DenseBatch &batch,
                                    const DenseOperatorPrecisions &precisions) {
    validate_dense_layer_inputs(config, model, batch);
    const DenseLayerContext context =
        make_dense_layer_context(device, config, model, batch, precisions);
    const AttentionLayerWork attention = predict_attention_work(context);
    DenseLayerTimes times{};
    populate_attention_times(context, attention, times);
    populate_dense_mlp_and_norm_times(context, times);
    return times;
}

DenseLayerTimes predict_dense_layer(const DeviceCeilings &device,
                                    const AnalyticalConfig &config,
                                    const DenseModel &model,
                                    const DenseBatch &batch,
                                    Precision precision) {
    return predict_dense_layer(
        device, config, model, batch,
        DenseOperatorPrecisions{precision, precision, precision});
}

} // namespace frontier::execution_time_predictor::detail

namespace frontier::execution_time_predictor::detail {

ExpertParallelDomain::ExpertParallelDomain(std::uint64_t total_experts,
                                           std::uint64_t expert_parallel_size)
    : total_experts_(total_experts),
      expert_parallel_size_(expert_parallel_size) {
    if (total_experts_ == 0 || expert_parallel_size_ == 0 ||
        total_experts_ % expert_parallel_size_ != 0) {
        throw ParallelDomainError(
            "expert count must be positive and divisible by EP size");
    }
}

ExpertRange ExpertParallelDomain::expert_range(std::uint64_t lane) const {
    if (lane >= expert_parallel_size_) {
        throw ParallelDomainError("EP lane is outside the parallel domain");
    }
    const std::uint64_t width = experts_per_lane();
    return ExpertRange{lane * width, (lane + 1) * width};
}

std::uint64_t ExpertParallelDomain::owner(std::uint64_t expert_id) const {
    if (expert_id >= total_experts_) {
        throw ParallelDomainError("expert ID is outside the parallel domain");
    }
    return expert_id / experts_per_lane();
}

std::vector<std::vector<std::uint64_t>> ExpertParallelDomain::partition(
    const std::vector<std::uint64_t> &global_counts) const {
    if (global_counts.size() != static_cast<std::size_t>(total_experts_)) {
        throw ParallelDomainError(
            "global expert allocation has the wrong size");
    }

    std::vector<std::vector<std::uint64_t>> lanes;
    lanes.reserve(static_cast<std::size_t>(expert_parallel_size_));
    for (std::uint64_t lane = 0; lane < expert_parallel_size_; ++lane) {
        const ExpertRange range = expert_range(lane);
        lanes.emplace_back(
            global_counts.begin() + static_cast<std::ptrdiff_t>(range.begin),
            global_counts.begin() + static_cast<std::ptrdiff_t>(range.end));
    }
    return lanes;
}

} // namespace frontier::execution_time_predictor::detail

namespace frontier::execution_time_predictor::detail {
namespace {

std::uint64_t ceil_div(std::uint64_t numerator, std::uint64_t denominator) {
    if (denominator == 0) {
        throw AnalyticalModelError("MoE TP size must be positive");
    }
    return numerator / denominator +
           static_cast<std::uint64_t>(numerator % denominator != 0);
}

double predict_ms(const DeviceCeilings &device, Precision precision,
                  const KernelWork &work, const Efficiency &efficiency,
                  double launch_latency_us) {
    return predict_roofline(device, precision, work, efficiency,
                            launch_latency_us)
        .predicted_time_ms;
}

std::uint64_t payload_bytes(std::uint64_t tokens, std::uint64_t hidden_size,
                            double element_bytes) {
    const long double bytes = static_cast<long double>(tokens) *
                              static_cast<long double>(hidden_size) *
                              static_cast<long double>(element_bytes);
    if (!std::isfinite(static_cast<double>(bytes)) ||
        bytes > static_cast<long double>(
                    std::numeric_limits<std::uint64_t>::max())) {
        throw AnalyticalModelError(
            "MoE communication payload overflows uint64");
    }
    return static_cast<std::uint64_t>(std::ceil(bytes));
}

struct MoELayerContext {
    const DeviceCeilings &device;
    const AnalyticalConfig &config;
    const MoEModel &model;
    std::uint64_t input_tokens;
    std::uint64_t router_topk;
    Precision expert_weight_precision;
    Precision latent_moe_projection_weight_precision;
    Precision shared_expert_weight_precision;
    Precision router_weight_precision;
    Precision router_compute_precision;
    Precision dense_weight_precision;
    double expert_weight_element_bytes;
    double expert_element_bytes;
    double latent_moe_projection_weight_element_bytes;
    double latent_moe_projection_element_bytes;
    double shared_expert_weight_element_bytes;
    double shared_expert_element_bytes;
    double router_weight_element_bytes;
    double router_element_bytes;
    double dense_weight_element_bytes;
    double dense_element_bytes;
    std::uint64_t local_intermediate;
};

struct ExpertGemmWork {
    KernelWork routed_up;
    KernelWork routed_down;
    KernelWork shared_up;
    KernelWork shared_down;
    std::uint64_t routed_tokens = 0;
};

void validate_moe_layer_inputs(const MoEModel &model,
                               std::uint64_t router_topk) {
    if (model.hidden_size == 0 || model.intermediate_size == 0 ||
        model.model_num_experts == 0 || model.moe_tensor_parallel_size == 0 ||
        router_topk == 0) {
        throw AnalyticalModelError("MoE model dimensions must be positive");
    }
}

MoELayerContext
make_moe_layer_context(const DeviceCeilings &device,
                       const AnalyticalConfig &config, const MoEModel &model,
                       std::uint64_t input_tokens, std::uint64_t router_topk,
                       const MoEOperatorPrecisions &precisions) {
    const Precision expert_weight_precision =
        precisions.expert_weight.value_or(precisions.expert);
    const Precision expert_activation_precision =
        precisions.expert_activation.value_or(precisions.expert);
    const Precision latent_moe_projection_weight_precision =
        precisions.latent_moe_projection_weight.value_or(
            expert_weight_precision);
    const Precision latent_moe_projection_activation_precision =
        precisions.latent_moe_projection_activation.value_or(
            expert_activation_precision);
    const Precision shared_expert_weight_precision =
        precisions.shared_expert_weight.value_or(precisions.shared_expert);
    const Precision shared_expert_activation_precision =
        precisions.shared_expert_activation.value_or(precisions.shared_expert);
    const Precision router_weight_precision =
        precisions.router_weight.value_or(precisions.router);
    const Precision router_activation_precision =
        precisions.router_activation.value_or(precisions.router);
    const Precision router_compute_precision =
        precisions.router_compute.value_or(precisions.router);
    const Precision dense_weight_precision =
        precisions.dense_weight.value_or(precisions.dense);
    const Precision dense_activation_precision =
        precisions.dense_activation.value_or(precisions.dense);
    return MoELayerContext{
        device,
        config,
        model,
        input_tokens,
        router_topk,
        expert_weight_precision,
        latent_moe_projection_weight_precision,
        shared_expert_weight_precision,
        router_weight_precision,
        router_compute_precision,
        dense_weight_precision,
        bytes_per_element(expert_weight_precision),
        bytes_per_element(expert_activation_precision),
        bytes_per_element(latent_moe_projection_weight_precision),
        bytes_per_element(latent_moe_projection_activation_precision),
        bytes_per_element(shared_expert_weight_precision),
        bytes_per_element(shared_expert_activation_precision),
        bytes_per_element(router_weight_precision),
        bytes_per_element(router_activation_precision),
        bytes_per_element(dense_weight_precision),
        bytes_per_element(dense_activation_precision),
        ceil_div(model.intermediate_size, model.moe_tensor_parallel_size),
    };
}

double predict_expert_work_ms(const MoELayerContext &context,
                              const KernelWork &work,
                              const Efficiency &efficiency) {
    return predict_ms(context.device, context.expert_weight_precision, work,
                      efficiency, context.config.kernel_launch_latency_us);
}

double predict_latent_moe_projection_work_ms(
    const MoELayerContext &context, const KernelWork &work,
    const Efficiency &efficiency) {
    return predict_ms(context.device,
                      context.latent_moe_projection_weight_precision, work,
                      efficiency, context.config.kernel_launch_latency_us);
}

double predict_shared_expert_work_ms(const MoELayerContext &context,
                                     const KernelWork &work,
                                     const Efficiency &efficiency) {
    return predict_ms(context.device, context.shared_expert_weight_precision,
                      work, efficiency,
                      context.config.kernel_launch_latency_us);
}

double predict_router_work_ms(const MoELayerContext &context,
                              const KernelWork &work,
                              const Efficiency &efficiency) {
    return predict_ms(context.device, context.router_compute_precision, work,
                      efficiency, context.config.kernel_launch_latency_us);
}

double predict_dense_work_ms(const MoELayerContext &context,
                             const KernelWork &work,
                             const Efficiency &efficiency) {
    return predict_ms(context.device, context.dense_weight_precision, work,
                      efficiency, context.config.kernel_launch_latency_us);
}

void add_kernel_work(KernelWork &target, const KernelWork &source) {
    target.flops += source.flops;
    target.hbm_bytes += source.hbm_bytes;
}

void add_expert_gemm_work(ExpertGemmWork &work, const MoELayerContext &context,
                          std::uint64_t tokens, bool count_as_routed) {
    if (tokens == 0) {
        return;
    }
    if (count_as_routed) {
        work.routed_tokens += tokens;
    }
    const std::uint64_t expert_input_size =
        !count_as_routed || context.model.routed_expert_hidden_size == 0
            ? context.model.hidden_size
            : context.model.routed_expert_hidden_size;
    KernelWork &up = count_as_routed ? work.routed_up : work.shared_up;
    KernelWork &down = count_as_routed ? work.routed_down : work.shared_down;
    const double weight_bytes =
        count_as_routed ? context.expert_weight_element_bytes
                        : context.shared_expert_weight_element_bytes;
    const double activation_bytes =
        count_as_routed ? context.expert_element_bytes
                        : context.shared_expert_element_bytes;
    add_kernel_work(up, gemm_work(tokens, expert_input_size,
                                  context.local_intermediate, weight_bytes,
                                  activation_bytes,
                                  context.model.gated_mlp ? 2 : 1));
    add_kernel_work(down,
                    gemm_work(tokens, context.local_intermediate,
                              expert_input_size, weight_bytes,
                              activation_bytes, 1));
}

ExpertGemmWork
build_expert_gemm_work(const MoELayerContext &context,
                       const std::vector<std::uint64_t> &local_expert_tokens) {
    ExpertGemmWork result{};
    for (const std::uint64_t tokens : local_expert_tokens) {
        add_expert_gemm_work(result, context, tokens, true);
    }
    // Shared experts are replicated on every EP lane. Their weights are only
    // sharded across the MoE TP domain, so each lane processes every input
    // token through every shared expert.
    for (std::uint64_t expert = 0; expert < context.model.num_shared_experts;
         ++expert) {
        add_expert_gemm_work(result, context, context.input_tokens, false);
    }
    return result;
}

const Efficiency &router_gemm_efficiency(const MoELayerContext &context) {
    return context.input_tokens < context.config.small_gemm_token_threshold
               ? context.config.small_gemm
               : context.config.large_gemm;
}

} // namespace

double MoELayerTime::total_ms() const noexcept {
    return gating_linear_ms + gating_routing_topk_ms +
           grouped_up_projection_ms + grouped_down_projection_ms +
           shuffling_ms + post_attention_norm_ms + latent_projection_ms +
           latent_norm_ms + attn_res_ms;
}

double MoECommunicationTime::total_ms() const noexcept {
    return attention_tp_ms + moe_tp_ms + ep_dispatch_ms + ep_combine_ms +
           dp_input_ms + dp_output_ms + pipeline_parallel_ms;
}

MoELayerTime
predict_moe_layer(const DeviceCeilings &device, const AnalyticalConfig &config,
                  const MoEModel &model, std::uint64_t input_tokens,
                  std::uint64_t router_topk,
                  const std::vector<std::uint64_t> &local_expert_tokens,
                  const MoEOperatorPrecisions &precisions) {
    validate_moe_layer_inputs(model, router_topk);
    const MoELayerContext context = make_moe_layer_context(
        device, config, model, input_tokens, router_topk, precisions);
    const ExpertGemmWork expert_work =
        build_expert_gemm_work(context, local_expert_tokens);
    const double tokens = static_cast<double>(context.input_tokens);
    const double hidden = static_cast<double>(context.model.hidden_size);
    const double experts = static_cast<double>(context.model.model_num_experts);
    const double routed = static_cast<double>(expert_work.routed_tokens);
    const double norm_factor = context.model.fused_add_norm ? 3.0 : 2.0;
    const std::uint64_t latent_hidden_size =
        context.model.routed_expert_hidden_size == 0
            ? context.model.hidden_size
            : context.model.routed_expert_hidden_size;

    MoELayerTime result{};
    result.gating_linear_ms = predict_router_work_ms(
        context,
        gemm_work(context.input_tokens, context.model.hidden_size,
                  context.model.model_num_experts,
                  context.router_weight_element_bytes,
                  context.router_element_bytes,
                  bytes_per_element(context.router_compute_precision), 1),
        router_gemm_efficiency(context));
    result.gating_routing_topk_ms = predict_router_work_ms(
        context,
        streaming_work(tokens * experts,
                       tokens * static_cast<double>(context.router_topk),
                       4.0 * tokens * experts,
                       bytes_per_element(context.router_compute_precision)),
        context.config.routing);
    const bool shared_uses_routed_kernel =
        context.shared_expert_weight_precision ==
            context.expert_weight_precision &&
        context.shared_expert_weight_element_bytes ==
            context.expert_weight_element_bytes &&
        context.shared_expert_element_bytes == context.expert_element_bytes;
    if (shared_uses_routed_kernel) {
        KernelWork combined_up = expert_work.routed_up;
        KernelWork combined_down = expert_work.routed_down;
        add_kernel_work(combined_up, expert_work.shared_up);
        add_kernel_work(combined_down, expert_work.shared_down);
        result.grouped_up_projection_ms = predict_expert_work_ms(
            context, combined_up, context.config.moe);
        result.grouped_down_projection_ms = predict_expert_work_ms(
            context, combined_down, context.config.moe);
    } else {
        result.grouped_up_projection_ms =
            predict_expert_work_ms(context, expert_work.routed_up,
                                   context.config.moe) +
            predict_shared_expert_work_ms(context, expert_work.shared_up,
                                          context.config.moe);
        result.grouped_down_projection_ms =
            predict_expert_work_ms(context, expert_work.routed_down,
                                   context.config.moe) +
            predict_shared_expert_work_ms(context, expert_work.shared_down,
                                          context.config.moe);
    }
    result.shuffling_ms = predict_expert_work_ms(
        context,
        streaming_work(routed * static_cast<double>(latent_hidden_size),
                       routed * static_cast<double>(latent_hidden_size), 0.0,
                       context.expert_element_bytes),
        context.config.streaming);
    if (context.model.routed_expert_hidden_size != 0) {
        const KernelWork latent_projection = add_kernel_work(
            gemm_work(context.input_tokens, context.model.hidden_size,
                      latent_hidden_size,
                      context.latent_moe_projection_weight_element_bytes,
                      context.latent_moe_projection_element_bytes, 1),
            gemm_work(context.input_tokens, latent_hidden_size,
                      context.model.hidden_size,
                      context.latent_moe_projection_weight_element_bytes,
                      context.latent_moe_projection_element_bytes, 1));
        result.latent_projection_ms = predict_latent_moe_projection_work_ms(
            context, latent_projection, router_gemm_efficiency(context));
        if (context.model.latent_moe_use_norm) {
            result.latent_norm_ms = predict_dense_work_ms(
                context,
                streaming_work(tokens * static_cast<double>(latent_hidden_size),
                               tokens * static_cast<double>(latent_hidden_size),
                               5.0 * tokens *
                                   static_cast<double>(latent_hidden_size),
                               context.dense_element_bytes),
                context.config.streaming);
        }
    }
    result.post_attention_norm_ms = predict_dense_work_ms(
        context,
        streaming_work(tokens * hidden * (norm_factor - 1.0), tokens * hidden,
                       5.0 * tokens * hidden, context.dense_element_bytes),
        context.config.streaming);
    return result;
}

MoELayerTime
predict_moe_layer(const DeviceCeilings &device, const AnalyticalConfig &config,
                  const MoEModel &model, std::uint64_t input_tokens,
                  std::uint64_t router_topk,
                  const std::vector<std::uint64_t> &local_expert_tokens,
                  Precision precision) {
    return predict_moe_layer(
        device, config, model, input_tokens, router_topk, local_expert_tokens,
        MoEOperatorPrecisions{precision, precision, precision, precision});
}

MoELanePrediction predict_moe_lanes(const DeviceCeilings &device,
                                    const AnalyticalConfig &config,
                                    const MoEModel &model,
                                    const RoutingAllocation &routing,
                                    std::uint64_t router_topk,
                                    const MoEOperatorPrecisions &precisions) {
    if (routing.lane_expert_tokens.empty()) {
        throw AnalyticalModelError(
            "MoE routing must contain at least one lane");
    }
    MoELanePrediction prediction;
    prediction.lane_times.reserve(routing.lane_expert_tokens.size());
    for (const auto &lane : routing.lane_expert_tokens) {
        prediction.lane_times.push_back(
            predict_moe_layer(device, config, model, routing.input_tokens,
                              router_topk, lane, precisions));
    }
    for (std::size_t lane = 0; lane < prediction.lane_times.size(); ++lane) {
        const double time = prediction.lane_times[lane].total_ms();
        if (lane == 0 || time > prediction.critical_lane_time_ms) {
            prediction.critical_lane = static_cast<std::uint64_t>(lane);
            prediction.critical_lane_time_ms = time;
        }
    }
    return prediction;
}

MoELanePrediction
predict_moe_lanes(const DeviceCeilings &device, const AnalyticalConfig &config,
                  const MoEModel &model, const RoutingAllocation &routing,
                  std::uint64_t router_topk, Precision precision) {
    return predict_moe_lanes(
        device, config, model, routing, router_topk,
        MoEOperatorPrecisions{precision, precision, precision, precision});
}

double predict_output_projection_ms(
    const DeviceCeilings &device, const AnalyticalConfig &config,
    std::uint64_t tokens, std::uint64_t hidden_size, std::uint64_t vocab_size,
    std::uint64_t tensor_parallel_size, Precision weight_precision,
    Precision activation_precision) {
    if (hidden_size == 0 || vocab_size == 0 || tensor_parallel_size == 0) {
        throw AnalyticalModelError(
            "output projection dimensions must be positive");
    }
    const std::uint64_t local_vocab =
        ceil_div(vocab_size, tensor_parallel_size);
    const KernelWork work = gemm_work(
        tokens, hidden_size, local_vocab, bytes_per_element(weight_precision),
        bytes_per_element(activation_precision), 1);
    const Efficiency &efficiency = tokens < config.small_gemm_token_threshold
                                       ? config.small_gemm
                                       : config.large_gemm;
    return predict_ms(device, weight_precision, work, efficiency,
                      config.kernel_launch_latency_us);
}

MoECommunicationTime predict_moe_communication(
    const cc_backend::BaseCCBackend &communication, std::uint64_t input_tokens,
    std::uint64_t hidden_size, std::uint64_t routed_tokens,
    std::uint64_t attention_tp_size, std::uint64_t moe_tp_size,
    std::uint64_t expert_parallel_size, std::uint64_t data_parallel_size,
    bool has_pipeline_boundary, double element_bytes,
    std::uint64_t routed_hidden_size) {
    const std::uint64_t activation_bytes =
        payload_bytes(input_tokens, hidden_size, element_bytes);
    const std::uint64_t routed_bytes = payload_bytes(
        routed_tokens,
        routed_hidden_size == 0 ? hidden_size : routed_hidden_size,
        element_bytes);
    return [&]() {
        MoECommunicationTime value{};
        value.attention_tp_ms =
            communication.allreduce_ms(activation_bytes, attention_tp_size);
        value.moe_tp_ms =
            communication.allreduce_ms(activation_bytes, moe_tp_size);
        value.ep_dispatch_ms =
            communication.all_to_all_ms(routed_bytes, expert_parallel_size);
        value.ep_combine_ms =
            communication.all_to_all_ms(routed_bytes, expert_parallel_size);
        value.dp_input_ms =
            communication.allreduce_ms(activation_bytes, data_parallel_size);
        value.dp_output_ms =
            communication.allreduce_ms(activation_bytes, data_parallel_size);
        value.pipeline_parallel_ms =
            has_pipeline_boundary
                ? communication.point_to_point_ms(activation_bytes)
                : 0.0;
        return value;
    }();
}

} // namespace frontier::execution_time_predictor::detail

namespace frontier::execution_time_predictor::detail {
namespace {

// Deterministic routing RNG.  The engine and every conversion below are
// either specified by the C++ standard or implemented here, so one seed
// yields the same routing on every platform and standard library.
// std::uniform_*_distribution is deliberately avoided: its output sequence
// is implementation-defined and would break that guarantee.
class RoutingRng {
  public:
    explicit RoutingRng(std::uint64_t seed) : engine_(seed) {}

    // Uniform in [0, 1) using the top 53 bits, matching double's mantissa.
    double next_double() {
        return static_cast<double>(engine_() >> 11U) *
               (1.0 / 9007199254740992.0);
    }

    // Uniform in [0, exclusive_upper).  Values below the rejection
    // threshold are discarded so ranges that do not divide 2^64 stay
    // unbiased; that threshold is 2^64 mod range in unsigned arithmetic.
    std::uint64_t bounded(std::uint64_t exclusive_upper) {
        if (exclusive_upper == 0) {
            throw RoutingError("bounded routing range must be positive");
        }
        if (exclusive_upper == 1) {
            return 0;
        }
        const std::uint64_t reject_below =
            (0ULL - exclusive_upper) % exclusive_upper;
        for (;;) {
            const std::uint64_t value = engine_();
            if (value >= reject_below) {
                return value % exclusive_upper;
            }
        }
    }

  private:
    std::mt19937_64 engine_;
};

std::vector<double>
distribution_weights(std::uint64_t experts,
                     config::MoeRoutingDistribution distribution,
                     std::uint64_t seed) {
    std::vector<double> weights(static_cast<std::size_t>(experts), 1.0);
    switch (distribution) {
    case config::MoeRoutingDistribution::kBalanced:
        break;
    case config::MoeRoutingDistribution::kRandom: {
        RoutingRng generator(seed);
        for (double &weight : weights) {
            weight = 0.1 + 0.9 * generator.next_double();
        }
        break;
    }
    case config::MoeRoutingDistribution::kSkewed:
        for (std::size_t i = 0; i < weights.size(); ++i) {
            weights[i] = 1.0 / std::pow(static_cast<double>(i + 1), 0.35);
        }
        break;
    case config::MoeRoutingDistribution::kZipf:
        for (std::size_t i = 0; i < weights.size(); ++i) {
            weights[i] = 1.0 / static_cast<double>(i + 1);
        }
        break;
    }
    return weights;
}

} // namespace

std::vector<std::uint64_t>
discretize_expert_weights(std::uint64_t total_tokens,
                          const std::vector<double> &weights) {
    if (weights.empty()) {
        throw RoutingError("expert weights must not be empty");
    }
    double total_weight = 0.0;
    for (const double weight : weights) {
        if (!std::isfinite(weight) || weight < 0.0) {
            throw RoutingError("expert weights must be finite and nonnegative");
        }
        total_weight += weight;
    }
    if (!std::isfinite(total_weight) || total_weight <= 0.0) {
        throw RoutingError("expert weights must have a positive sum");
    }

    std::vector<std::uint64_t> result(weights.size(), 0);
    std::vector<double> normalized(weights.size(), 0.0);
    std::vector<double> fractional(weights.size(), 0.0);
    std::uint64_t allocated = 0;
    for (std::size_t i = 0; i < weights.size(); ++i) {
        normalized[i] = weights[i] / total_weight;
        const double exact = static_cast<double>(total_tokens) * normalized[i];
        result[i] = static_cast<std::uint64_t>(exact);
        fractional[i] = exact - static_cast<double>(result[i]);
        allocated += result[i];
    }

    std::vector<std::size_t> order(weights.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](std::size_t lhs, std::size_t rhs) {
                  if (fractional[lhs] != fractional[rhs]) {
                      return fractional[lhs] > fractional[rhs];
                  }
                  if (normalized[lhs] != normalized[rhs]) {
                      return normalized[lhs] > normalized[rhs];
                  }
                  return lhs < rhs;
              });
    std::uint64_t remainder_index = 0;
    while (allocated < total_tokens) {
        ++result[order[static_cast<std::size_t>(remainder_index %
                                                order.size())]];
        ++allocated;
        ++remainder_index;
    }
    return result;
}

RoutingAllocation
route_tokens(std::uint64_t input_tokens, std::uint64_t router_topk,
             std::uint64_t total_experts, std::uint64_t expert_parallel_size,
             const config::MoeRoutingConfig &config, std::uint64_t layer_id) {
    if (router_topk == 0 || router_topk > total_experts) {
        throw RoutingError("router top-k must be in [1, total experts]");
    }
    if (input_tokens >
        std::numeric_limits<std::uint64_t>::max() / router_topk) {
        throw RoutingError("routed token count overflows uint64");
    }

    ExpertParallelDomain domain(total_experts, expert_parallel_size);
    const std::uint64_t routed_tokens = input_tokens * router_topk;
    std::vector<std::uint64_t> counts(static_cast<std::size_t>(total_experts),
                                      0);

    if (config.mode == config::MoeRoutingMode::kUniformLegacy) {
        const std::uint64_t base = routed_tokens / total_experts;
        const std::uint64_t remainder = routed_tokens % total_experts;
        for (std::uint64_t expert = 0; expert < total_experts; ++expert) {
            counts[static_cast<std::size_t>(expert)] =
                base + static_cast<std::uint64_t>(expert < remainder);
        }
    } else if (config.mode == config::MoeRoutingMode::kUniformRandom) {
        RoutingRng generator(config.seed + layer_id);
        for (std::uint64_t token = 0; token < routed_tokens; ++token) {
            ++counts[static_cast<std::size_t>(
                generator.bounded(total_experts))];
        }
    } else {
        counts = discretize_expert_weights(
            routed_tokens,
            distribution_weights(total_experts, config.distribution,
                                 config.seed + layer_id));
    }

    return [&]() {
        RoutingAllocation value{};
        value.input_tokens = input_tokens;
        value.routed_tokens = routed_tokens;
        value.global_expert_tokens = counts;
        value.lane_expert_tokens = domain.partition(counts);
        return value;
    }();
}

} // namespace frontier::execution_time_predictor::detail
