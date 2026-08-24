// Dense, MLA, MFA, and KDA per-layer roofline models.
//
// Dense and attention roofline implementation. Public internal contracts live
// in analytical_attention_model.h.

#include "frontier/execution_time_predictor/analytical_attention_model.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "frontier/attention/mla.h"
#include "frontier/core/checked_math.h"

namespace frontier::execution_time_predictor::detail {

namespace {

std::uint64_t dense_ceil_div(std::uint64_t numerator,
                             std::uint64_t denominator) {
    return checked_math::ceil_div<AnalyticalModelError>(
        numerator, denominator, "division denominator must be positive");
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

// MHA/GQA/MQA, MLA, and MFA all execute sequence-attention kernels and share
// this normalized intermediate. KDA has a recurrent-state pipeline instead
// and deliberately uses its own family-local work type below.
struct SequenceAttentionWork {
    double pre_projection_ms = 0.0;
    double post_projection_ms = 0.0;
    double inter_norm_ms = 0.0;
    double wq_projection_ms = 0.0;
    double rope_elements = 0.0;
    KernelWork kv_cache_save{0.0, 0.0};
    KernelWork prefill_attention{0.0, 0.0};
    KernelWork decode_attention{0.0, 0.0};
    // Decode-side GEMM issued on an alternate stream and joined before the
    // post-attention projection (K3 MLA output gate).
    double decode_side_projection_ms = 0.0;
};

AttentionKind attention_kind(const DenseModel &model) {
    const std::uint64_t selected = static_cast<std::uint64_t>(model.use_mla) +
                                   static_cast<std::uint64_t>(model.use_mfa) +
                                   static_cast<std::uint64_t>(model.use_kda);
    if (selected > 1) {
        throw AnalyticalModelError(
            "MLA, MFA, and KDA attention modes are mutually exclusive");
    }
    if (model.use_kda) {
        return AttentionKind::kKda;
    }
    if (model.use_mla) {
        return AttentionKind::kMla;
    }
    if (model.use_mfa) {
        return AttentionKind::kMfa;
    }
    return AttentionKind::kStandard;
}

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

void validate_dense_layer_inputs(const DenseModel &model,
                                 const DenseBatch &batch) {
    if (model.hidden_size == 0 || model.intermediate_size == 0 ||
        model.num_query_heads == 0 || model.num_kv_heads == 0 ||
        model.head_dim == 0 || model.tensor_parallel_size == 0) {
        throw AnalyticalModelError("dense model dimensions must be positive");
    }
    if (sum_query_tokens(batch.prefill_requests) +
            sum_query_tokens(batch.decode_requests) !=
        batch.total_tokens) {
        throw AnalyticalModelError(
            "dense batch total_tokens does not match request slices");
    }
    const AttentionKind kind = attention_kind(model);
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
    if (model.decode_context_parallel_size > 1 && kind != AttentionKind::kMla &&
        kind != AttentionKind::kKda) {
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
    if (model.use_kda && (model.kda_num_heads != model.kda_num_k_heads ||
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

Efficiency gemm_efficiency_for(const DenseLayerContext &context,
                               std::uint64_t m, std::uint64_t k,
                               std::uint64_t n,
                               std::uint64_t independent_matrices = 1) {
    return gemm_efficiency_for_shape(context.config.gemm, m, k, n,
                                     independent_matrices);
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

double predict_attention_gemm_ms(const DenseLayerContext &context,
                                 std::uint64_t m, std::uint64_t k,
                                 std::uint64_t n,
                                 std::uint64_t output_matrices = 1) {
    if (m == 0) {
        return 0.0;
    }
    return predict_attention_work_ms(
        context, attention_gemm_work(context, m, k, n, output_matrices),
        gemm_efficiency_for(context, m, k, n, output_matrices));
}

double predict_dense_gemm_ms(const DenseLayerContext &context,
                             std::uint64_t m, std::uint64_t k,
                             std::uint64_t n,
                             std::uint64_t output_matrices = 1) {
    if (m == 0) {
        return 0.0;
    }
    return predict_dense_work_ms(
        context, dense_gemm_work(context, m, k, n, output_matrices),
        gemm_efficiency_for(context, m, k, n, output_matrices));
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

SequenceAttentionWork
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
    SequenceAttentionWork work{};

    if (model.q_lora_rank == 0) {
        work.pre_projection_ms = predict_attention_gemm_ms(
            context, context.batch.total_tokens, model.hidden_size,
            context.local_query_heads * model.qk_head_dim);
    } else {
        work.pre_projection_ms =
            predict_attention_gemm_ms(context, context.batch.total_tokens,
                                      model.hidden_size, model.q_lora_rank) +
            predict_attention_gemm_ms(
                context, context.batch.total_tokens, model.q_lora_rank,
                context.local_query_heads * model.qk_head_dim);
        work.inter_norm_ms += predict_attention_work_ms(
            context,
            streaming_work(tokens * static_cast<double>(model.q_lora_rank),
                           tokens * static_cast<double>(model.q_lora_rank),
                           5.0 * tokens *
                               static_cast<double>(model.q_lora_rank),
                           context.attention_element_bytes),
            context.config.streaming);
    }
    work.pre_projection_ms += predict_attention_gemm_ms(
        context, context.batch.total_tokens, model.hidden_size,
        latent_and_rope_dim);
    work.inter_norm_ms += predict_attention_work_ms(
        context,
        streaming_work(tokens * static_cast<double>(model.kv_lora_rank),
                       tokens * static_cast<double>(model.kv_lora_rank),
                       5.0 * tokens * static_cast<double>(model.kv_lora_rank),
                       context.attention_element_bytes),
        context.config.streaming);
    work.pre_projection_ms += predict_attention_work_ms(
        context, mla_unabsorbed_kv_expansion_work(context, expanded_kv_dim),
        gemm_efficiency_for(context, prefill_visible_tokens,
                            model.kv_lora_rank, expanded_kv_dim));
    work.pre_projection_ms += predict_attention_work_ms(
        context,
        per_head_gemm_work(context, context.decode_tokens,
                           model.qk_nope_head_dim, model.kv_lora_rank),
        gemm_efficiency_for(context, context.decode_tokens,
                            model.qk_nope_head_dim, model.kv_lora_rank,
                            context.local_query_heads));
    work.post_projection_ms += predict_attention_gemm_ms(
        context, context.prefill_tokens,
        context.local_query_heads * model.v_head_dim, model.hidden_size);
    work.post_projection_ms += predict_attention_work_ms(
        context,
        per_head_gemm_work(context, context.decode_tokens, model.kv_lora_rank,
                           model.v_head_dim),
        gemm_efficiency_for(context, context.decode_tokens,
                            model.kv_lora_rank, model.v_head_dim,
                            context.local_query_heads));
    work.post_projection_ms += predict_attention_gemm_ms(
        context, context.decode_tokens,
        context.local_query_heads * model.v_head_dim, model.hidden_size);

    if (model.mla_use_output_gate) {
        // Kimi K3 computes a full-rank sigmoid gate from the layer input,
        // applies it elementwise to the per-head attention output, and only
        // then runs o_proj.  Account for the gate projection with the
        // attention weight/activation dtypes and for the fused sigmoid +
        // multiply as a streaming kernel.  The latter reads both the gate
        // and attention output and writes the gated output once.
        const std::uint64_t gate_dim =
            context.local_query_heads * model.v_head_dim;
        const bool overlap_decode_gate =
            context.config.overlap_mla_output_gate &&
            context.prefill_tokens == 0 && context.decode_tokens > 0 &&
            context.decode_tokens <= 128;
        const double gate_projection_ms = predict_attention_gemm_ms(
            context, context.batch.total_tokens, model.hidden_size, gate_dim);
        if (overlap_decode_gate) {
            work.decode_side_projection_ms = gate_projection_ms;
        } else {
            work.post_projection_ms += gate_projection_ms;
        }
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
    // DCP deliberately applies to decode only: a context-parallel prefill would
    // pay an all-gather/reduce-scatter pair among the DCP peers that outweighs
    // the sharded read, and sequential PDD runs its prefill cluster at DCP=1.
    // The unmodeled case is co-location with DCP > 1 over a prefix-cache hit,
    // where this term reads latent entries the rank does not own while
    // kv_cache_save below stays sharded.  See kimi-k3-support.md 12.2.
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

SequenceAttentionWork
predict_mfa_attention_work(const DenseLayerContext &context) {
    const DenseModel &model = context.model;
    const double tokens = static_cast<double>(context.batch.total_tokens);
    const double query_heads = static_cast<double>(context.local_query_heads);
    const double kv_heads = static_cast<double>(context.local_kv_heads);
    const double head_dim = static_cast<double>(model.head_dim);
    const std::uint64_t replicated_qkv_dim =
        model.share_q_dim + 2 * context.local_kv_heads * model.head_dim;
    SequenceAttentionWork work{};
    work.pre_projection_ms = predict_attention_gemm_ms(
        context, context.batch.total_tokens, model.hidden_size,
        replicated_qkv_dim);
    work.inter_norm_ms = predict_attention_work_ms(
        context,
        streaming_work(tokens * static_cast<double>(model.share_q_dim),
                       tokens * static_cast<double>(model.share_q_dim),
                       5.0 * tokens * static_cast<double>(model.share_q_dim),
                       context.attention_element_bytes),
        context.config.streaming);
    work.wq_projection_ms = predict_attention_gemm_ms(
        context, context.batch.total_tokens, model.share_q_dim,
        context.local_query_heads * model.head_dim);
    work.post_projection_ms = predict_attention_gemm_ms(
        context, context.batch.total_tokens,
        context.local_query_heads * model.head_dim, model.hidden_size);
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

SequenceAttentionWork
predict_mha_attention_work(const DenseLayerContext &context) {
    const DenseModel &model = context.model;
    const double tokens = static_cast<double>(context.batch.total_tokens);
    const double query_heads = static_cast<double>(context.local_query_heads);
    const double kv_heads = static_cast<double>(context.local_kv_heads);
    const double head_dim = static_cast<double>(model.head_dim);
    const std::uint64_t local_qkv_dim =
        (context.local_query_heads + 2 * context.local_kv_heads) *
        model.head_dim;
    SequenceAttentionWork work{};
    work.pre_projection_ms = predict_attention_gemm_ms(
        context, context.batch.total_tokens, model.hidden_size, local_qkv_dim);
    work.post_projection_ms = predict_attention_gemm_ms(
        context, context.batch.total_tokens,
        std::max<std::uint64_t>(1,
                                model.hidden_size / model.tensor_parallel_size),
        model.hidden_size);
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
DenseLayerTimes predict_kda_attention_times(const DenseLayerContext &context) {
    struct KdaAttentionWork {
        KernelWork projection{0.0, 0.0};
        KernelWork short_conv_prefill{0.0, 0.0};
        KernelWork short_conv_decode{0.0, 0.0};
        KernelWork recurrent_prefill{0.0, 0.0};
        KernelWork recurrent_decode{0.0, 0.0};
        KernelWork gate_norm_prefill{0.0, 0.0};
        KernelWork gate_norm_decode{0.0, 0.0};
    };
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
    const double qk_channels = static_cast<double>(local_qk_channels);
    const double v_channels = static_cast<double>(local_v_channels);
    const double conv_channels = static_cast<double>(std::max(
        local_qk_channels * 2 + local_v_channels,
        dense_ceil_div(model.kda_conv_state_dim, model.tensor_parallel_size)));
    const double v_heads = static_cast<double>(local_v_heads);
    const double key_dim = static_cast<double>(local_key_dim);
    const double value_dim = static_cast<double>(local_value_dim);

    KdaAttentionWork work{};

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
    const bool overlap_aux_projections =
        context.config.overlap_kda_aux_projections &&
        context.prefill_tokens == 0 && context.decode_tokens > 0 &&
        context.decode_tokens <= 128;
    double aux_projection_ms = 0.0;
    if (overlap_aux_projections) {
        // K3's main stream runs one [q|k|v|g] GEMM. A side stream runs the
        // merged [f_a|beta] GEMV followed by f_b; it joins before recurrence.
        // The two branches read hidden_states independently but overlap.
        work.projection = gemm_work(
            tokens, model.hidden_size, local_qk_channels,
            context.attention_weight_element_bytes,
            context.attention_element_bytes, 4);
        const KernelWork fa_beta = gemm_work(
            tokens, model.hidden_size,
            model.kda_head_dim + local_v_heads,
            context.attention_weight_element_bytes,
            context.attention_element_bytes, 1);
        aux_projection_ms =
            predict_attention_work_ms(
                context, fa_beta,
                gemm_efficiency_for(context, tokens, model.hidden_size,
                                    model.kda_head_dim + local_v_heads)) +
            predict_attention_work_ms(
                context, f_b,
                gemm_efficiency_for(context, tokens, model.kda_head_dim,
                                    local_v_channels));
    } else {
        work.projection = add_kernel_work(
            add_kernel_work(add_kernel_work(qk, v), add_kernel_work(f_a, f_b)),
            add_kernel_work(full_rank_gate, beta));
    }

    // Each of Q/K/V is passed through a depthwise short convolution.  The
    // history window is fixed, so this cost is linear in tokens and does not
    // depend on request past_context.  HBM traffic accounts for a window read
    // and one output write for each channel.
    const auto short_conv_work = [&](std::uint64_t count) {
        const double rows = static_cast<double>(count);
        const double history =
            rows * conv_channels * static_cast<double>(conv_kernel);
        const double output = rows * conv_channels;
        return KernelWork{
            2.0 * history,
            history * context.kda_state_element_bytes +
                output * context.attention_element_bytes,
        };
    };
    work.short_conv_prefill = short_conv_work(context.prefill_tokens);
    work.short_conv_decode = short_conv_work(context.decode_tokens);

    // The portable recurrence keeps the historical conservative contract:
    // one full state read/write per token. The released K3 profile replaces
    // prefill below with FlashKDA's two-stage chunk pipeline. Decode still
    // touches one state per request and therefore uses this work record.
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
    work.recurrent_prefill = recurrent_work(context.batch.prefill_requests);
    work.recurrent_decode = recurrent_work(context.batch.decode_requests);

    const auto flashkda_prefill_ms = [&]() {
        if (!context.config.use_flashkda_prefill_two_stage ||
            context.batch.prefill_requests.empty()) {
            return predict_attention_work_ms(
                context, work.recurrent_prefill,
                context.config.prefill_attention);
        }

        const std::uint64_t chunk = context.config.flashkda_chunk_tokens;
        const std::uint64_t sm_count = context.config.flashkda_sm_count;
        if (chunk == 0 || sm_count == 0) {
            throw AnalyticalModelError(
                "FlashKDA chunk size and SM count must be positive");
        }
        std::uint64_t total_chunks = 0;
        std::uint64_t longest_chunks = 0;
        for (const AttentionRequestSlice &request :
             context.batch.prefill_requests) {
            const std::uint64_t chunks =
                dense_ceil_div(request.query_tokens, chunk);
            total_chunks += chunks;
            longest_chunks = std::max(longest_chunks, chunks);
        }

        const double heads = v_heads;
        const double chunks = static_cast<double>(total_chunks);
        const double token_count = static_cast<double>(context.prefill_tokens);
        const double workspace =
            chunks * heads * static_cast<double>(
                                  context.config
                                      .flashkda_workspace_bytes_per_chunk_head);
        const double state_elements_per_sequence =
            heads * key_dim * value_dim;
        const double sequences =
            static_cast<double>(context.batch.prefill_requests.size());

        // K1 grid: (sequence chunks, heads). It normalizes Q/K, constructs
        // the 16x16 decay/Mqk terms and inverse, then writes the exact public
        // FlashKDA workspace: 3x4096 + 3x512 = 13,824 B per chunk/head.
        const KernelWork kernel1{
            4.0 * token_count * heads * key_dim * value_dim,
            token_count * (2.0 * qk_channels + heads) *
                    context.attention_element_bytes +
                workspace,
        };
        // K2 grid: (sequences, heads). It reads the workspace, scans chunks
        // in order, contracts V/output and reads/writes the FP32 boundary
        // state once per sequence rather than once per token.
        const KernelWork kernel2{
            4.0 * token_count * heads * key_dim * value_dim,
            workspace +
                token_count * (2.0 * v_channels + heads) *
                    context.attention_element_bytes +
                2.0 * sequences * state_elements_per_sequence *
                    context.kda_state_element_bytes,
        };
        const double roofline_ms =
            predict_attention_work_ms(context, kernel1,
                                      context.config.prefill_attention) +
            predict_attention_work_ms(context, kernel2,
                                      context.config.prefill_attention);

        // Aggregate roofline misses K2's small N*H grid and sequential chunk
        // tail. Anchor that scheduling envelope to FlashKDA's public GB200
        // 8192-token FP32-state measurements (H64=0.9247 ms,
        // H96=1.0087 ms). GB300 has the same 160-SM topology and HBM class;
        // the roofline above remains the physical lower bound.
        const double single_sequence_8192_ms =
            0.9247 + (heads - 64.0) * ((1.0087 - 0.9247) / 32.0);
        const double sequence_head_ctas = sequences * heads;
        const double effective_chunks =
            sequence_head_ctas <= static_cast<double>(sm_count)
                ? static_cast<double>(longest_chunks)
                : chunks * heads / static_cast<double>(sm_count) +
                      context.config.flashkda_serial_tail_exposure *
                          static_cast<double>(longest_chunks);
        const double scheduling_ms =
            std::max(0.0, single_sequence_8192_ms) * effective_chunks /
            (8192.0 / static_cast<double>(chunk));
        return std::max(roofline_ms, scheduling_ms);
    };

    // Fused RMSNorm + sigmoid gate: one read/write pass over the projected
    // output with a small constant amount of scalar work per element.
    const auto gate_norm_work = [&](std::uint64_t count) {
        const double rows = static_cast<double>(count);
        return streaming_work(rows * v_channels, rows * v_channels,
                              8.0 * rows * v_channels,
                              context.attention_element_bytes);
    };
    work.gate_norm_prefill = gate_norm_work(context.prefill_tokens);
    work.gate_norm_decode = gate_norm_work(context.decode_tokens);

    DenseLayerTimes times{};
    // Keep the historical attention buckets populated while exposing the KDA
    // sub-components for diagnostics and focused tests.
    // The Day-0 path merges skinny Q/K/V/gate projections and overlaps the
    // dependent GEMV chain. A standalone tile-grid prior would double-count
    // the small-M loss already represented by this fused aggregate profile.
    const double main_projection_ms = predict_attention_work_ms(
        context, work.projection,
        gemm_efficiency_for(
            context, tokens, model.hidden_size,
            overlap_aux_projections
                ? 4 * local_qk_channels
                : 2 * local_qk_channels + 2 * local_v_channels +
                      model.kda_head_dim + local_v_heads));
    times.kda_projection_ms =
        overlap_aux_projections
            ? std::max(main_projection_ms, aux_projection_ms)
            : main_projection_ms;
    const double prefill_short_conv_ms = predict_attention_work_ms(
        context, work.short_conv_prefill, context.config.streaming);
    const double decode_short_conv_ms = predict_attention_work_ms(
        context, work.short_conv_decode, context.config.streaming);
    const double prefill_recurrent_ms = flashkda_prefill_ms();
    const double decode_recurrent_ms = predict_attention_work_ms(
        context, work.recurrent_decode, context.config.decode_attention);
    const double prefill_gate_norm_ms = predict_attention_work_ms(
        context, work.gate_norm_prefill, context.config.streaming);
    const double decode_gate_norm_ms = predict_attention_work_ms(
        context, work.gate_norm_decode, context.config.streaming);
    const bool fuse_decode_chain = context.config.fuse_kda_decode_chain &&
                                   context.prefill_tokens == 0 &&
                                   context.decode_tokens > 0;
    double fused_decode_ms = 0.0;
    if (fuse_decode_chain) {
        const KernelWork fused_decode_work = add_kernel_work(
            add_kernel_work(work.short_conv_decode, work.recurrent_decode),
            work.gate_norm_decode);
        fused_decode_ms = predict_attention_work_ms(
            context, fused_decode_work, context.config.decode_attention);
    }
    times.kda_short_conv_ms =
        prefill_short_conv_ms +
        (fuse_decode_chain ? 0.0 : decode_short_conv_ms);
    times.kda_recurrent_ms =
        prefill_recurrent_ms +
        (fuse_decode_chain ? fused_decode_ms : decode_recurrent_ms);
    times.kda_gate_norm_ms =
        prefill_gate_norm_ms +
        (fuse_decode_chain ? 0.0 : decode_gate_norm_ms);
    times.attention_pre_projection_ms =
        times.kda_projection_ms + times.kda_short_conv_ms;
    times.attention_post_projection_ms = predict_attention_gemm_ms(
        context, context.batch.total_tokens,
        dense_ceil_div(context.model.kda_num_v_heads,
                       context.model.tensor_parallel_size) *
            context.model.kda_value_head_dim,
        context.model.hidden_size);
    times.attention_inter_norm_ms = times.kda_gate_norm_ms;
    times.prefill_attention_ms = prefill_recurrent_ms;
    times.decode_attention_ms =
        fuse_decode_chain ? fused_decode_ms : decode_recurrent_ms;
    // KDA does not use RoPE or a sequence-growing KV cache.
    times.rope_ms = 0.0;
    times.kv_cache_save_ms = 0.0;
    times.attention_norm_ms = 0.0;
    times.attention_wq_projection_ms = 0.0;
    return times;
}

DenseLayerTimes
predict_sequence_attention_times(const DenseLayerContext &context,
                                 const SequenceAttentionWork &work) {
    DenseLayerTimes times{};
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
    if (work.decode_side_projection_ms > 0.0) {
        times.decode_attention_ms =
            std::max(times.decode_attention_ms,
                     work.decode_side_projection_ms);
    }
    return times;
}

DenseLayerTimes predict_attention_times(const DenseLayerContext &context) {
    switch (attention_kind(context.model)) {
    case AttentionKind::kStandard:
        return predict_sequence_attention_times(
            context, predict_mha_attention_work(context));
    case AttentionKind::kMla:
        return predict_sequence_attention_times(
            context, predict_mla_attention_work(context));
    case AttentionKind::kMfa:
        return predict_sequence_attention_times(
            context, predict_mfa_attention_work(context));
    case AttentionKind::kKda:
        return predict_kda_attention_times(context);
    }
    throw AnalyticalModelError("unknown analytical attention kind");
}

void populate_dense_mlp_and_norm_times(const DenseLayerContext &context,
                                       DenseLayerTimes &times) {
    const double tokens = static_cast<double>(context.batch.total_tokens);
    const double hidden = static_cast<double>(context.model.hidden_size);
    const double intermediate = static_cast<double>(context.local_intermediate);
    const double activation_elements = tokens * intermediate;
    const std::uint64_t gated_multiplier = context.model.gated_mlp ? 2 : 1;
    const double norm_factor = context.model.fused_add_norm ? 3.0 : 2.0;
    times.mlp_up_projection_ms = predict_dense_gemm_ms(
        context, context.batch.total_tokens, context.model.hidden_size,
        context.local_intermediate, gated_multiplier);
    times.mlp_activation_ms = predict_dense_work_ms(
        context,
        streaming_work(activation_elements *
                           static_cast<double>(gated_multiplier),
                       activation_elements, 8.0 * activation_elements,
                       context.dense_element_bytes),
        context.config.streaming);
    times.mlp_down_projection_ms = predict_dense_gemm_ms(
        context, context.batch.total_tokens, context.local_intermediate,
        context.model.hidden_size);
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
    validate_dense_layer_inputs(model, batch);
    const DenseLayerContext context =
        make_dense_layer_context(device, config, model, batch, precisions);
    DenseLayerTimes times = predict_attention_times(context);
    populate_dense_mlp_and_norm_times(context, times);
    return times;
}

DenseLayerTimes predict_dense_layer(const DeviceCeilings &device,
                                    const AnalyticalConfig &config,
                                    const DenseModel &model,
                                    const DenseBatch &batch,
                                    Precision precision) {
    DenseOperatorPrecisions precisions{};
    precisions.attention = precision;
    precisions.dense = precision;
    precisions.kv_cache = precision;
    return predict_dense_layer(device, config, model, batch, precisions);
}

} // namespace frontier::execution_time_predictor::detail
