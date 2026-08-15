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

// A chunk covering [start, start + scheduled_tokens) attends to exactly the
// tokens ahead of its own queries, and that offset is fixed for the chunk's
// entire life: no sibling chunk of the same request can move it.  The
// scheduler advances the optimistic frontier before it builds the snapshot,
// so the offset is recoverable from the snapshot alone -- this is the same
// identity VllmV1Scheduler::apply_batch_completion uses to decide which
// completion belongs to which chunk.
std::uint64_t
snapshot_past_context(const entities::RequestBatchSnapshot &snapshot) {
    if (snapshot.scheduler_frontier < snapshot.scheduled_tokens) {
        throw ExecutionTimePredictorError(
            "batch snapshot scheduler frontier underflows");
    }
    return snapshot.scheduler_frontier - snapshot.scheduled_tokens;
}

detail::AttentionRequestSlice
make_attention_request_slice(const entities::RequestBatchSnapshot &snapshot) {
    return detail::AttentionRequestSlice{
        snapshot.scheduled_tokens,
        snapshot_past_context(snapshot),
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
        // Every attention input comes from the batch's own snapshot rather
        // than from live request state.  A request may hold several prefill
        // chunks at once, and its live frontier then advances while an earlier
        // chunk is still traversing the pipeline, so reading it here would make
        // a stage's predicted work depend on when that stage happened to run.
        // That is what made "collapsed" and "exact" pipeline event modes
        // disagree, and it understated attention under both: the live counter
        // only reaches this chunk's true offset once every earlier chunk has
        // retired, which may be after the chunk has already left the pipeline.
        const detail::AttentionRequestSlice slice =
            make_attention_request_slice(snapshot);
        if (slice.past_context < request.num_prefill_tokens()) {
            result.dense_batch.prefill_requests.push_back(slice);
            // The chunk that reaches the end of the prompt is the one that
            // samples the request's first token.
            if (snapshot.scheduler_frontier >= request.num_prefill_tokens()) {
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
    // Shared layer scope makes one assignment serve every MoE layer of the
    // batch, so it is memoized once per batch and handed to each stage and
    // each lazily predicted layer.  Null means the caller wants this layer's
    // own draw.
    const detail::RoutingAllocation *reusable_routing_allocation = nullptr;
    bool detailed_diagnostics_enabled = true;

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

// Both MoE stage assemblers charge the same per-layer collective set; each
// decides for itself whether to multiply it by the layer count.
detail::MoECommunicationTime
moe_stage_communication(const MoEStageContext &context) {
    if (context.reusable_moe_communication != nullptr) {
        return *context.reusable_moe_communication;
    }
    const std::uint64_t input_tokens =
        context.batch_info.dense_batch.total_tokens;
    return detail::predict_moe_communication(
        context.communication_backend, input_tokens, context.model.hidden_size,
        input_tokens * context.model.router_topk,
        context.parallelism.tensor_parallel_size,
        context.parallelism.moe_tensor_parallel_size,
        context.parallelism.moe_expert_parallel_size,
        context.parallelism.data_parallel_size, false,
        context.communication_element_bytes,
        context.model.routed_expert_hidden_size);
}

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
            std::optional<detail::RoutingAllocation> owned_allocation;
            const detail::RoutingAllocation *allocation =
                context.reusable_routing_allocation;
            if (allocation == nullptr) {
                owned_allocation.emplace(detail::route_tokens(
                    batch_info.dense_batch.total_tokens, model.router_topk,
                    model.total_expert_num,
                    parallelism.moe_expert_parallel_size, context.routing,
                    routing_seed_layer(routing_is_batch_shared(context.config,
                                                               context.routing),
                                       model_layer)));
                allocation = &*owned_allocation;
            }
            std::optional<detail::MoELanePrediction> owned_lane_prediction;
            const detail::MoELanePrediction *lane_prediction =
                context.reusable_moe_lane_prediction;
            if (lane_prediction == nullptr) {
                owned_lane_prediction.emplace(detail::predict_moe_lanes(
                    context.device, detail::AnalyticalConfig{}, moe_model,
                    *allocation, model.router_topk, moe_precisions));
                lane_prediction = &*owned_lane_prediction;
            }
            result.routing_diagnostics.push_back(make_moe_routing_diagnostic(
                local_moe_layer, model_layer, pre_moe_compute_ms,
                pre_moe_tp_communication_ms, *allocation, *lane_prediction));
            const detail::MoELayerTime &critical =
                lane_prediction->lane_times.at(
                    static_cast<std::size_t>(lane_prediction->critical_lane));
            add_critical_moe_layer_time(
                result.execution_time, critical,
                context.layer_time(model_layer).residual_add_ms);
            if (context.detailed_diagnostics_enabled) {
                result.diagnostics.emplace_back(
                    "layer_" + std::to_string(model_layer) + "_critical_lane",
                    static_cast<double>(lane_prediction->critical_lane));
                result.diagnostics.emplace_back(
                    "layer_" + std::to_string(model_layer) +
                        "_critical_lane_ms",
                    lane_prediction->critical_lane_time_ms);
            }
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
        moe_stage_communication(context);
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
    const detail::MoELanePrediction *repeated_lane_prediction = nullptr;
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
            pending_pre_moe_compute_ms = 0.0;
            pending_pre_moe_tp_communication_ms = 0.0;
            ++moe_layer_index;
            continue;
        }
        pending_pre_moe_compute_ms += attention_compute_ms;
        const detail::RoutingAllocation *allocation = shared_allocation;
        if (allocation == nullptr || !layer_shared) {
            owned_allocation.emplace(detail::route_tokens(
                batch_info.dense_batch.total_tokens, model.router_topk,
                model.total_expert_num, parallelism.moe_expert_parallel_size,
                routing, routing_seed_layer(layer_shared, model_layer)));
            allocation = &*owned_allocation;
            if (layer_shared) {
                shared_allocation = allocation;
            }
        }
        const detail::MoELanePrediction *lane_prediction =
            shared_lane_prediction;
        if (lane_prediction == nullptr || !layer_shared) {
            owned_lane_prediction.emplace(detail::predict_moe_lanes(
                context.device, detail::AnalyticalConfig{}, moe_model,
                *allocation, model.router_topk, moe_precisions));
            lane_prediction = &*owned_lane_prediction;
            if (layer_shared) {
                shared_lane_prediction = lane_prediction;
            }
        }
        result.routing_diagnostics.push_back(make_moe_routing_diagnostic(
            moe_layer_index, model_layer, pending_pre_moe_compute_ms,
            pending_pre_moe_tp_communication_ms +
                context.attention_communication_ms(model_layer),
            *allocation, *lane_prediction));
        const detail::MoELayerTime &critical = lane_prediction->lane_times.at(
            static_cast<std::size_t>(lane_prediction->critical_lane));
        add_critical_moe_layer_time(
            result.execution_time, critical,
            context.layer_time(model_layer).residual_add_ms);
        if (context.detailed_diagnostics_enabled) {
            result.diagnostics.emplace_back(
                "layer_" + std::to_string(model_layer) + "_critical_lane",
                static_cast<double>(lane_prediction->critical_lane));
            result.diagnostics.emplace_back(
                "layer_" + std::to_string(model_layer) + "_critical_lane_ms",
                lane_prediction->critical_lane_time_ms);
        }
        pending_pre_moe_compute_ms = 0.0;
        pending_pre_moe_tp_communication_ms = 0.0;
        ++moe_layer_index;
        if (first_layer_scaled || stage_group_scaled) {
            repeated_lane_prediction = lane_prediction;
            result.repeated_moe_layer_pre_compute_ms = attention_compute_ms;
        }
    }
    result.suffix_compute_ms = pending_pre_moe_compute_ms;
    result.suffix_tp_communication_ms = pending_pre_moe_tp_communication_ms;
    const detail::MoECommunicationTime communication_time =
        moe_stage_communication(context);
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

std::uint64_t
model_layer_for_local_moe(const config::ModelConfig &model,
                          const config::PipelineStageLayerRange &stage_layers,
                          std::uint64_t local_moe_layer) {
    std::uint64_t current = 0;
    for (std::uint64_t model_layer = stage_layers.begin;
         model_layer < stage_layers.end; ++model_layer) {
        if (!model.is_moe_layer(model_layer)) {
            continue;
        }
        if (current == local_moe_layer) {
            return model_layer;
        }
        ++current;
    }
    throw ExecutionTimePredictorError(
        "cached MoE layer index is outside the pipeline stage");
}

ExecutionTimePrediction materialize_cached_prediction(
    const ExecutionTimePrediction &cached, const config::ModelConfig &model,
    const config::PipelineStageLayerRange &stage_layers) {
    ExecutionTimePrediction result = cached;
    for (MoERoutingDiagnostic &routing : result.moe_routing) {
        if (!routing.layer_id.valid()) {
            throw ExecutionTimePredictorError(
                "cached MoE routing has an invalid local layer ID");
        }
        routing.model_layer_id = model_layer_for_local_moe(
            model, stage_layers, routing.layer_id.index());
    }
    return result;
}

} // namespace

config::StageTimingSignature
AnalyticalRooflineExecutionTimePredictor::make_stage_timing_signature(
    std::uint64_t stage) const {
    return config::build_pipeline_stage_timing_signature(model_, parallelism_,
                                                         stage);
}

void AnalyticalRooflineExecutionTimePredictor::build_stage_timing_groups() {
    timing_catalogue_.timing_groups.clear();
    timing_catalogue_.stage_to_timing_group.clear();
    timing_catalogue_.timing_group_multiplicity.clear();
    timing_catalogue_.stage_to_timing_group.reserve(
        static_cast<std::size_t>(parallelism_.pipeline_parallel_size));
    for (std::uint64_t stage = 0; stage < parallelism_.pipeline_parallel_size;
         ++stage) {
        config::StageTimingSignature signature =
            make_stage_timing_signature(stage);
        const auto existing =
            std::find(timing_catalogue_.timing_groups.begin(),
                      timing_catalogue_.timing_groups.end(), signature);
        std::uint32_t group_id = 0;
        if (existing == timing_catalogue_.timing_groups.end()) {
            group_id = static_cast<std::uint32_t>(
                timing_catalogue_.timing_groups.size());
            timing_catalogue_.timing_groups.push_back(std::move(signature));
            timing_catalogue_.timing_group_multiplicity.push_back(0);
        } else {
            group_id = static_cast<std::uint32_t>(std::distance(
                timing_catalogue_.timing_groups.begin(), existing));
        }
        timing_catalogue_.stage_to_timing_group.push_back(group_id);
        ++timing_catalogue_.timing_group_multiplicity.at(group_id);
    }
}

std::vector<detail::DenseLayerTimes>
AnalyticalRooflineExecutionTimePredictor::build_stage_layer_times(
    const config::PipelineStageLayerRange &stage_layers,
    const detail::DenseBatch &dense_batch,
    const detail::DenseOperatorPrecisions &precisions) const {
    std::vector<detail::DenseLayerTimes> result;
    result.reserve(static_cast<std::size_t>(stage_layers.size()));
    detail::DenseModel dense_model = make_dense_model(model_, parallelism_);
    // One representative prediction per attention family is enough: layers of
    // the same family in one stage see the same batch and dimensions.
    std::optional<detail::DenseLayerTimes> standard_layer_times;
    std::optional<detail::DenseLayerTimes> mla_layer_times;
    std::optional<detail::DenseLayerTimes> kda_layer_times;
    for (std::uint64_t model_layer = stage_layers.begin;
         model_layer < stage_layers.end; ++model_layer) {
        dense_model.use_kda =
            model_.has_kda() && model_.is_kda_layer(model_layer);
        // KDA and MLA are mutually exclusive on a logical layer.  Non-hybrid
        // MLA checkpoints retain the old path.
        dense_model.use_mla = !dense_model.use_kda && model_.use_mla;
        std::optional<detail::DenseLayerTimes> *representative =
            dense_model.use_kda ? &kda_layer_times
                                : (dense_model.use_mla ? &mla_layer_times
                                                       : &standard_layer_times);
        if (!representative->has_value()) {
            *representative = detail::predict_dense_layer(
                device_, detail::AnalyticalConfig{}, dense_model, dense_batch,
                precisions);
        }
        result.push_back(representative->value());
    }
    return result;
}

double AnalyticalRooflineExecutionTimePredictor::compute_allreduce_ms(
    std::uint64_t activation_bytes) const {
    if (parallelism_.tensor_parallel_size <= 1) {
        return 0.0;
    }
    return communication_backend_->allreduce_ms(
        activation_bytes, parallelism_.tensor_parallel_size, true);
}

double AnalyticalRooflineExecutionTimePredictor::
    compute_dcp_attention_communication_ms(
        const detail::DenseBatch &dense_batch,
        const config::PipelineStageLayerRange &stage_layers,
        double communication_element_bytes) const {
    if (!model_.use_mla || parallelism_.decode_context_parallel_size <= 1 ||
        dense_batch.decode_requests.empty()) {
        return 0.0;
    }
    // DCP collectives are charged per MLA layer, so a stage made entirely of
    // KDA layers pays nothing.  Decide that from model metadata rather than
    // from predicted KDA component times.
    bool has_mla_layer = false;
    for (std::uint64_t model_layer = stage_layers.begin;
         model_layer < stage_layers.end; ++model_layer) {
        if (!(model_.has_kda() && model_.is_kda_layer(model_layer))) {
            has_mla_layer = true;
            break;
        }
    }
    if (!has_mla_layer) {
        return 0.0;
    }

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
    return communication_backend_->allgather_ms(
               query_bytes, parallelism_.decode_context_parallel_size, true) +
           communication_backend_->reduce_scatter_ms(
               gathered_output_bytes, parallelism_.decode_context_parallel_size,
               true);
}

detail::MoECommunicationTime
AnalyticalRooflineExecutionTimePredictor::compute_moe_communication(
    const detail::DenseBatch &dense_batch,
    double communication_element_bytes) const {
    return detail::predict_moe_communication(
        *communication_backend_, dense_batch.total_tokens, model_.hidden_size,
        dense_batch.total_tokens * model_.router_topk,
        parallelism_.tensor_parallel_size,
        parallelism_.moe_tensor_parallel_size,
        parallelism_.moe_expert_parallel_size, parallelism_.data_parallel_size,
        false, communication_element_bytes, model_.routed_expert_hidden_size);
}

AnalyticalRooflineExecutionTimePredictor::StageTimingCacheLookup
AnalyticalRooflineExecutionTimePredictor::lookup_stage_timing_template(
    BatchId batch_id, std::uint32_t timing_group_id,
    const detail::DenseBatch &dense_batch,
    const config::PipelineStageLayerRange &stage_layers) const {
    if (!batch_id.valid()) {
        throw ExecutionTimePredictorError(
            "stage timing cache requires a valid batch ID");
    }

    std::lock_guard<std::mutex> lock(timing_cache_mutex_);
    const auto batch_position = timing_cache_by_batch_.find(batch_id);
    if (batch_position != timing_cache_by_batch_.end()) {
        const auto existing = batch_position->second.find(timing_group_id);
        if (existing != batch_position->second.end()) {
            ++timing_cache_hits_total_;
            return StageTimingCacheLookup{existing->second,
                                          true,
                                          timing_cache_hits_total_,
                                          timing_cache_misses_total_,
                                          timing_cache_unique_templates_total_,
                                          timing_cache_entries_};
        }
    }

    auto value = std::make_shared<StageTimingCacheValue>();
    const detail::DenseOperatorPrecisions dense_precisions =
        make_dense_operator_precisions(config_);
    value->layer_times =
        build_stage_layer_times(stage_layers, dense_batch, dense_precisions);

    const detail::Precision communication_precision =
        detail::precision_from_string(config_.communication_precision());
    const double communication_element_bytes =
        detail::bytes_per_element(communication_precision);
    const std::uint64_t activation_bytes =
        activation_payload_bytes(dense_batch.total_tokens, model_.hidden_size,
                                 communication_element_bytes);
    value->allreduce_ms = compute_allreduce_ms(activation_bytes);
    value->dcp_attention_communication_ms =
        compute_dcp_attention_communication_ms(dense_batch, stage_layers,
                                               communication_element_bytes);

    // The representative expert lane lives in the batch-level shared-routing
    // memo. Keeping it out of each timing group avoids recalculating the same
    // routing allocation and expert roofline on every template miss.
    if (model_.is_moe() &&
        config_.moe_layer_event_mode == "stage_group_scaled" &&
        routing_is_layer_invariant(routing_) &&
        has_contiguous_moe_suffix(model_, stage_layers)) {
        value->moe_communication =
            compute_moe_communication(dense_batch, communication_element_bytes);
        value->has_moe_communication = true;
    }

    timing_cache_by_batch_[batch_id].emplace(timing_group_id, value);
    ++timing_cache_entries_;
    ++timing_cache_misses_total_;
    ++timing_cache_unique_templates_total_;
    return StageTimingCacheLookup{std::move(value),
                                  false,
                                  timing_cache_hits_total_,
                                  timing_cache_misses_total_,
                                  timing_cache_unique_templates_total_,
                                  timing_cache_entries_};
}

std::shared_ptr<const ExecutionTimePrediction>
AnalyticalRooflineExecutionTimePredictor::lookup_complete_prediction(
    BatchId batch_id, std::uint32_t timing_group_id) const {
    std::lock_guard<std::mutex> lock(timing_cache_mutex_);
    const auto batch_position = timing_cache_by_batch_.find(batch_id);
    if (batch_position == timing_cache_by_batch_.end()) {
        return nullptr;
    }
    const auto group_position = batch_position->second.find(timing_group_id);
    if (group_position == batch_position->second.end() ||
        group_position->second->complete_prediction == nullptr) {
        return nullptr;
    }
    ++timing_cache_hits_total_;
    return group_position->second->complete_prediction;
}

void AnalyticalRooflineExecutionTimePredictor::store_complete_prediction(
    BatchId batch_id, std::uint32_t timing_group_id,
    const ExecutionTimePrediction &prediction) const {
    auto stored = std::make_shared<const ExecutionTimePrediction>(prediction);
    std::lock_guard<std::mutex> lock(timing_cache_mutex_);
    const auto batch_position = timing_cache_by_batch_.find(batch_id);
    if (batch_position == timing_cache_by_batch_.end()) {
        throw ExecutionTimePredictorError(
            "complete prediction requires an existing timing template");
    }
    const auto group_position = batch_position->second.find(timing_group_id);
    if (group_position == batch_position->second.end()) {
        throw ExecutionTimePredictorError(
            "complete prediction timing group is not cached");
    }
    group_position->second->complete_prediction = std::move(stored);
}

std::shared_ptr<
    const AnalyticalRooflineExecutionTimePredictor::SharedRoutingValue>
AnalyticalRooflineExecutionTimePredictor::shared_routing_for_batch(
    BatchId batch_id, const detail::DenseBatch &dense_batch) const {
    if (!batch_id.valid()) {
        throw ExecutionTimePredictorError(
            "shared routing memo requires a valid batch ID");
    }
    std::lock_guard<std::mutex> lock(timing_cache_mutex_);
    const auto existing = shared_routing_by_batch_.find(batch_id);
    if (existing != shared_routing_by_batch_.end()) {
        return existing->second;
    }
    auto value = std::make_shared<SharedRoutingValue>();
    // Layer zero is the canonical seed: a shared assignment must not depend on
    // which layer or stage happened to ask for it first.
    value->allocation = detail::route_tokens(
        dense_batch.total_tokens, model_.router_topk, model_.total_expert_num,
        parallelism_.moe_expert_parallel_size, routing_, 0);
    value->lane_prediction = detail::predict_moe_lanes(
        device_, detail::AnalyticalConfig{},
        make_moe_model(model_, parallelism_), value->allocation,
        model_.router_topk, make_moe_operator_precisions(config_));
    shared_routing_by_batch_.emplace(batch_id, value);
    return value;
}

void AnalyticalRooflineExecutionTimePredictor::release_batch_timing_cache(
    BatchId batch_id) const {
    if (!batch_id.valid()) {
        return;
    }
    std::lock_guard<std::mutex> lock(timing_cache_mutex_);
    shared_routing_by_batch_.erase(batch_id);
    const auto position = timing_cache_by_batch_.find(batch_id);
    if (position == timing_cache_by_batch_.end()) {
        return;
    }
    timing_cache_entries_ -=
        static_cast<std::uint64_t>(position->second.size());
    timing_cache_by_batch_.erase(position);
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
    // A singleton timing group has no second stage that can consume a cached
    // value for this batch. Avoid allocating, inserting, copying, and later
    // erasing a template that is structurally guaranteed never to hit.
    const bool reusable_timing_group = stage_group_active &&
                                       timing_group_multiplicity > 1 &&
                                       !selected_moe_layer.has_value();
    if (!detailed_diagnostics_enabled_ && reusable_timing_group) {
        const std::shared_ptr<const ExecutionTimePrediction> cached =
            lookup_complete_prediction(batch.id(), timing_group_id);
        if (cached != nullptr) {
            return materialize_cached_prediction(*cached, model_, stage_layers);
        }
    }

    const StageBatchInfo batch_info = build_stage_batch_info(batch, requests);
    const detail::DenseBatch &dense_batch = batch_info.dense_batch;
    const detail::DenseOperatorPrecisions dense_precisions =
        make_dense_operator_precisions(config_);
    const detail::Precision communication_precision =
        detail::precision_from_string(config_.communication_precision());
    std::vector<detail::DenseLayerTimes> layer_times;
    layer_times.reserve(static_cast<std::size_t>(stage_layers.size()));
    StageTimingCacheLookup timing_cache_lookup{};
    std::shared_ptr<const StageTimingCacheValue> timing_template;
    if (reusable_timing_group) {
        timing_cache_lookup = lookup_stage_timing_template(
            batch.id(), timing_group_id, dense_batch, stage_layers);
        timing_template = timing_cache_lookup.value;
        layer_times = timing_template->layer_times;
    } else {
        {
            std::lock_guard<std::mutex> lock(timing_cache_mutex_);
            timing_cache_lookup.hits_total = timing_cache_hits_total_;
            timing_cache_lookup.misses_total = timing_cache_misses_total_;
            timing_cache_lookup.unique_templates_total =
                timing_cache_unique_templates_total_;
            timing_cache_lookup.entries = timing_cache_entries_;
        }
        layer_times = build_stage_layer_times(stage_layers, dense_batch,
                                              dense_precisions);
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
                                    : compute_allreduce_ms(activation_bytes);
    const double dcp_attention_communication_ms =
        timing_template != nullptr
            ? timing_template->dcp_attention_communication_ms
            : compute_dcp_attention_communication_ms(
                  dense_batch, stage_layers, communication_element_bytes);
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
        // Shared layer scope and first_layer_scaled both reuse one assignment
        // for every MoE layer of the batch. Resolving it here covers whole
        // stages, stage-group templates, and single lazy-layer predictions.
        const std::shared_ptr<const SharedRoutingValue> shared_routing =
            routing_is_batch_shared(config_, routing_)
                ? shared_routing_for_batch(batch.id(), dense_batch)
                : nullptr;
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
            shared_routing != nullptr ? &shared_routing->lane_prediction
                                      : nullptr,
            timing_template != nullptr && timing_template->has_moe_communication
                ? &timing_template->moe_communication
                : nullptr,
            shared_routing != nullptr ? &shared_routing->allocation : nullptr,
            detailed_diagnostics_enabled_,
        };
        MoEStagePrediction moe_prediction =
            selected_moe_layer.has_value()
                ? predict_selected_moe_layer_execution(
                      moe_context, selected_moe_layer.value())
                : predict_moe_stage_execution(moe_context);
        execution_time = moe_prediction.execution_time;
        moe_diagnostics = std::move(moe_prediction.diagnostics);
        routing_diagnostics = std::move(moe_prediction.routing_diagnostics);
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
    ExecutionTimePrediction result{};
    result.duration_ms = duration_ms;
    result.execution_time = execution_time;
    if (detailed_diagnostics_enabled_) {
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
            kv_cache_bytes_per_token_per_layer(model_,
                                               dense_precisions.kv_cache);
        const double kv_cache_rank_local_bytes_per_token =
            model_.use_mla ? kv_cache_bytes_per_token /
                                 static_cast<double>(
                                     parallelism_.decode_context_parallel_size)
                           : kv_cache_bytes_per_token;

        result.diagnostics = {
            {"total_tokens", static_cast<double>(dense_batch.total_tokens)},
            {"stage_id", static_cast<double>(stage_id.index())},
            {"timing_group_id", static_cast<double>(timing_group_id)},
            {"timing_group_multiplicity",
             static_cast<double>(timing_group_multiplicity)},
            {"timing_cache_enabled", timing_template != nullptr ? 1.0 : 0.0},
            {"timing_cache_hit", timing_cache_lookup.hit ? 1.0 : 0.0},
            {"timing_cache_miss",
             timing_template != nullptr && !timing_cache_lookup.hit ? 1.0
                                                                    : 0.0},
            {"timing_cache_hits_total",
             static_cast<double>(timing_cache_lookup.hits_total)},
            {"timing_cache_misses_total",
             static_cast<double>(timing_cache_lookup.misses_total)},
            {"timing_cache_unique_templates_total",
             static_cast<double>(timing_cache_lookup.unique_templates_total)},
            {"timing_cache_entries",
             static_cast<double>(timing_cache_lookup.entries)},
            {"timing_group_layer_count",
             static_cast<double>(
                 timing_catalogue_.timing_groups.at(timing_group_id)
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
            // Retain the historical single-layer diagnostic as the first
            // logical layer, and expose stage sums/counts for heterogeneous
            // KDA/MLA schedules.
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
    }
    result.moe_routing = std::move(routing_diagnostics);
    result.logical_moe_layer_count = logical_moe_layer_count;
    result.scaled_moe_attention_groups = std::move(scaled_moe_attention_groups);
    result.repeated_moe_layer_pre_compute_ms =
        repeated_moe_layer_pre_compute_ms;
    result.moe_suffix_compute_ms = moe_suffix_compute_ms;
    result.moe_suffix_tp_communication_ms = moe_suffix_tp_communication_ms;
    result.lazy_moe_layer_prediction = selected_moe_layer.has_value();
    result.scaled_moe_layer_prediction =
        !selected_moe_layer.has_value() &&
        (config_.moe_layer_event_mode == "first_layer_scaled" ||
         stage_group_active);
    if (detailed_diagnostics_enabled_) {
        result.diagnostics.insert(result.diagnostics.end(),
                                  moe_diagnostics.begin(),
                                  moe_diagnostics.end());
    }
    if (!detailed_diagnostics_enabled_ && reusable_timing_group) {
        store_complete_prediction(batch.id(), timing_group_id, result);
    }
    return result;
}

} // namespace frontier::execution_time_predictor
