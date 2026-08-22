#include "frontier/execution_time_predictor/analytical_roofline_internal.h"

#include <algorithm>
#include <string>
#include <utility>

namespace frontier::execution_time_predictor::roofline_detail {

const entities::Request &
get_request(const entities::RequestCollection &requests, RequestId request_id) {
    if (!requests.contains(request_id)) {
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
                       const entities::RequestCollection &requests) {
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
        // Reading the live counter here understated attention: it reaches this
        // chunk's true offset only after every earlier chunk has retired,
        // which may be after the chunk has already left the pipeline.
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

[[noreturn]] void invalid_predictor_config(std::string_view reason) {
    throw ExecutionTimePredictorError(
        "unsupported analytical model configuration: " + std::string(reason));
}

void validate_predictor_configuration(
    const config::AnalyticalExecutionModelConfig &config,
    const config::ParallelismConfig &parallelism,
    const config::ModelConfig &model,
    const std::shared_ptr<const cc_backend::BaseCCBackend> &backend) {
    if (backend == nullptr) {
        invalid_predictor_config("communication backend is null");
    }

    const std::pair<std::string_view, std::string_view> precisions[] = {
        {"default", config.precision},
        {"attention", config.attention_precision()},
        {"dense", config.dense_precision()},
        {"moe_expert", config.moe_expert_precision()},
        {"moe_router", config.moe_router_precision()},
        {"kv_cache", config.kv_cache_precision()},
        {"communication", config.communication_precision()},
        {"attention_weight", config.attention_weight_precision()},
        {"attention_activation", config.attention_activation_precision()},
        {"dense_weight", config.dense_weight_precision()},
        {"dense_activation", config.dense_activation_precision()},
        {"routed_expert_weight", config.routed_expert_weight_precision()},
        {"routed_expert_activation",
         config.routed_expert_activation_precision()},
        {"latent_moe_projection_weight",
         config.latent_moe_projection_weight_precision()},
        {"latent_moe_projection_activation",
         config.latent_moe_projection_activation_precision()},
        {"shared_expert_weight", config.shared_expert_weight_precision()},
        {"shared_expert_activation",
         config.shared_expert_activation_precision()},
        {"dense_mlp_weight", config.dense_mlp_weight_precision()},
        {"dense_mlp_activation", config.dense_mlp_activation_precision()},
        {"router_weight_storage", config.router_weight_storage_precision()},
        {"moe_router_activation", config.moe_router_activation_precision()},
        {"router_compute", config.router_compute_precision()},
        {"kda_snapshot", config.kda_snapshot_precision()},
        {"lm_head_weight", config.lm_head_weight_precision()},
        {"lm_head_activation", config.lm_head_activation_precision()},
    };
    for (const auto &[name, value] : precisions) {
        try {
            static_cast<void>(detail::precision_from_string(value));
        } catch (const detail::AnalyticalModelError &) {
            invalid_predictor_config(std::string(name) +
                                     " precision is unsupported");
        }
    }

    if (config.moe_layer_event_mode != "detailed" &&
        config.moe_layer_event_mode != "first_layer_scaled" &&
        config.moe_layer_event_mode != "stage_group_scaled") {
        invalid_predictor_config("unknown MoE layer event mode");
    }
    if (model.attention.memory_layout ==
        attention::AttentionMemoryLayout::kFrozenDsa) {
        invalid_predictor_config("frozen DSA attention is unsupported");
    }
    if (parallelism.tensor_parallel_size == 0 ||
        parallelism.decode_context_parallel_size == 0 ||
        parallelism.pipeline_parallel_size == 0) {
        invalid_predictor_config("parallelism sizes must be positive");
    }
    if (parallelism.tensor_parallel_size %
            parallelism.decode_context_parallel_size !=
        0) {
        invalid_predictor_config("DCP size must divide TP size");
    }
    if (parallelism.decode_context_parallel_size > 1 && !model.use_mla) {
        invalid_predictor_config("DCP requires MLA");
    }
    if (parallelism.pipeline_parallel_size > model.num_layers) {
        invalid_predictor_config("PP size exceeds the model layer count");
    }
}

} // namespace frontier::execution_time_predictor::roofline_detail
