#include "frontier/execution_time_predictor/analytical_attention_model.h"

#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "frontier/core/checked_math.h"

namespace frontier::execution_time_predictor::detail {
namespace {

void require_finite_nonnegative(double value, const char *field) {
    if (!std::isfinite(value) || value < 0.0) {
        throw AnalyticalModelError(std::string{field} +
                                   " must be finite and nonnegative");
    }
}

} // namespace

std::uint64_t prefill_attention_token_pairs(
    const std::vector<AttentionRequestSlice> &requests) {
    std::uint64_t total = 0;
    for (const AttentionRequestSlice &request : requests) {
        const std::string_view message =
            "PREFILL attention token-pair count overflows uint64";
        total = checked_math::add<AnalyticalModelError>(
            total,
            checked_math::prefill_attention_token_pairs<AnalyticalModelError>(
                request.query_tokens, request.past_context, message),
            message);
    }
    return total;
}

KernelWork
attention_context_work(const std::vector<AttentionRequestSlice> &requests,
                       std::uint64_t local_query_heads,
                       std::uint64_t local_kv_heads, std::uint64_t head_dim,
                       double core_element_bytes,
                       double activation_element_bytes,
                       double kv_cache_element_bytes) {
    if (local_query_heads == 0 || local_kv_heads == 0 || head_dim == 0) {
        throw AnalyticalModelError("attention dimensions must be positive");
    }
    require_finite_nonnegative(core_element_bytes, "core element bytes");
    require_finite_nonnegative(activation_element_bytes,
                               "activation element bytes");
    require_finite_nonnegative(kv_cache_element_bytes,
                               "KV cache element bytes");
    if (core_element_bytes == 0.0 || activation_element_bytes == 0.0 ||
        kv_cache_element_bytes == 0.0) {
        throw AnalyticalModelError("attention element bytes must be positive");
    }

    double total_flops = 0.0;
    double core_input_elements = 0.0;
    double output_elements = 0.0;
    double kv_cache_elements = 0.0;
    for (const AttentionRequestSlice &request : requests) {
        const double query_tokens = static_cast<double>(request.query_tokens);
        const double past_context = static_cast<double>(request.past_context);
        const double average_visible_kv =
            past_context + (query_tokens + 1.0) / 2.0;
        total_flops += 4.0 * static_cast<double>(local_query_heads) *
                       static_cast<double>(head_dim) * query_tokens *
                       average_visible_kv;
        const double query = query_tokens *
                             static_cast<double>(local_query_heads) *
                             static_cast<double>(head_dim);
        const double output = query;
        const double new_kv_input = 2.0 * query_tokens *
                                    static_cast<double>(local_kv_heads) *
                                    static_cast<double>(head_dim);
        const double cached_kv_reads = 2.0 * past_context *
                                       static_cast<double>(local_kv_heads) *
                                       static_cast<double>(head_dim);
        core_input_elements += query + new_kv_input;
        output_elements += output;
        kv_cache_elements += cached_kv_reads;
    }
    return [&]() {
        KernelWork value{};
        value.flops = total_flops;
        value.hbm_bytes = core_input_elements * core_element_bytes +
                          output_elements * activation_element_bytes +
                          kv_cache_elements * kv_cache_element_bytes;
        if (core_element_bytes != activation_element_bytes) {
            // Projection outputs are materialized in the attention activation
            // dtype. TRT-LLM's FP8 FMHA path converts Q/K/V before QK/PV;
            // conservatively charge the conversion read/write traffic.
            value.flops += core_input_elements;
            value.hbm_bytes += core_input_elements *
                               (activation_element_bytes + core_element_bytes);
        }
        return value;
    }();
}

KernelWork
attention_context_work(const std::vector<AttentionRequestSlice> &requests,
                       std::uint64_t local_query_heads,
                       std::uint64_t local_kv_heads, std::uint64_t head_dim,
                       double activation_element_bytes,
                       double kv_cache_element_bytes) {
    return attention_context_work(
        requests, local_query_heads, local_kv_heads, head_dim,
        activation_element_bytes, activation_element_bytes,
        kv_cache_element_bytes);
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
    double core_element_bytes, double activation_element_bytes,
    double rope_cache_element_bytes) {
    if (local_query_heads == 0 || qk_nope_head_dim == 0 ||
        qk_rope_head_dim == 0 || v_head_dim == 0) {
        throw AnalyticalModelError("MLA attention dimensions must be positive");
    }
    require_finite_nonnegative(core_element_bytes, "core element bytes");
    require_finite_nonnegative(activation_element_bytes,
                               "activation element bytes");
    require_finite_nonnegative(rope_cache_element_bytes,
                               "RoPE cache element bytes");
    if (core_element_bytes == 0.0 || activation_element_bytes == 0.0 ||
        rope_cache_element_bytes == 0.0) {
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
    double core_input_elements = 0.0;
    double output_elements = 0.0;
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
        const double query = query_tokens * heads * qk_dim;
        const double output = query_tokens * heads * value_dim;
        const double expanded_kv = visible_kv * heads * expanded_kv_dim;
        const double new_rope = query_tokens * rope_dim;
        core_input_elements += query + expanded_kv + new_rope;
        output_elements += output;
        cached_rope_bytes += past_context * rope_dim * rope_cache_element_bytes;
    }
    double hbm_bytes = core_input_elements * core_element_bytes +
                       output_elements * activation_element_bytes +
                       cached_rope_bytes;
    if (core_element_bytes != activation_element_bytes) {
        total_flops += core_input_elements;
        hbm_bytes += core_input_elements *
                     (activation_element_bytes + core_element_bytes);
    }
    return KernelWork{total_flops, hbm_bytes};
}

KernelWork mla_unabsorbed_attention_work(
    const std::vector<AttentionRequestSlice> &requests,
    std::uint64_t local_query_heads, std::uint64_t qk_nope_head_dim,
    std::uint64_t qk_rope_head_dim, std::uint64_t v_head_dim,
    double activation_element_bytes, double rope_cache_element_bytes) {
    return mla_unabsorbed_attention_work(
        requests, local_query_heads, qk_nope_head_dim, qk_rope_head_dim,
        v_head_dim, activation_element_bytes, activation_element_bytes,
        rope_cache_element_bytes);
}

KernelWork mla_absorbed_attention_work(
    const std::vector<AttentionRequestSlice> &requests,
    std::uint64_t local_query_heads, std::uint64_t kv_lora_rank,
    std::uint64_t qk_rope_head_dim, double core_element_bytes,
    double activation_element_bytes,
    double latent_cache_element_bytes, double rope_cache_element_bytes) {
    if (local_query_heads == 0 || kv_lora_rank == 0 || qk_rope_head_dim == 0) {
        throw AnalyticalModelError("MLA attention dimensions must be positive");
    }
    require_finite_nonnegative(core_element_bytes, "core element bytes");
    require_finite_nonnegative(activation_element_bytes,
                               "activation element bytes");
    require_finite_nonnegative(latent_cache_element_bytes,
                               "latent cache element bytes");
    require_finite_nonnegative(rope_cache_element_bytes,
                               "RoPE cache element bytes");
    if (core_element_bytes == 0.0 || activation_element_bytes == 0.0 ||
        latent_cache_element_bytes == 0.0 ||
        rope_cache_element_bytes == 0.0) {
        throw AnalyticalModelError("attention element bytes must be positive");
    }

    const double heads = static_cast<double>(local_query_heads);
    const double latent_dim = static_cast<double>(kv_lora_rank);
    const double rope_dim = static_cast<double>(qk_rope_head_dim);
    double total_flops = 0.0;
    double core_input_elements = 0.0;
    double output_elements = 0.0;
    double cache_bytes = 0.0;
    for (const AttentionRequestSlice &request : requests) {
        const double query_tokens = static_cast<double>(request.query_tokens);
        const double past_context = static_cast<double>(request.past_context);
        const double average_visible_kv =
            past_context + (query_tokens + 1.0) / 2.0;
        total_flops += 2.0 * heads * query_tokens * average_visible_kv *
                       (2.0 * latent_dim + rope_dim);

        const double query =
            query_tokens * heads * (latent_dim + rope_dim);
        const double output = query_tokens * heads * latent_dim;
        const double new_latent_and_rope =
            query_tokens * (latent_dim + rope_dim);
        core_input_elements += query + new_latent_and_rope;
        output_elements += output;
        cache_bytes += past_context * (latent_dim * latent_cache_element_bytes +
                                       rope_dim * rope_cache_element_bytes);
    }
    double hbm_bytes = core_input_elements * core_element_bytes +
                       output_elements * activation_element_bytes + cache_bytes;
    if (core_element_bytes != activation_element_bytes) {
        total_flops += core_input_elements;
        hbm_bytes += core_input_elements *
                     (activation_element_bytes + core_element_bytes);
    }
    return KernelWork{total_flops, hbm_bytes};
}

KernelWork mla_absorbed_attention_work(
    const std::vector<AttentionRequestSlice> &requests,
    std::uint64_t local_query_heads, std::uint64_t kv_lora_rank,
    std::uint64_t qk_rope_head_dim, double activation_element_bytes,
    double latent_cache_element_bytes, double rope_cache_element_bytes) {
    return mla_absorbed_attention_work(
        requests, local_query_heads, kv_lora_rank, qk_rope_head_dim,
        activation_element_bytes, activation_element_bytes,
        latent_cache_element_bytes, rope_cache_element_bytes);
}

} // namespace frontier::execution_time_predictor::detail
