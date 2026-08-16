// Dense, MLA, MFA, and KDA per-layer roofline models.
//
// Dense and attention roofline implementation. Public internal contracts live
// in analytical_attention_model.h.

#include "frontier/execution_time_predictor/analytical_attention_model.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "frontier/attention/mla.h"
#include "frontier/core/checked_math.h"

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

Precision precision_from_string(std::string_view precision) {
    const std::optional<Precision> parsed = parse_precision(precision);
    if (parsed.has_value()) {
        return *parsed;
    }
    throw AnalyticalModelError("unsupported analytical precision: " +
                               std::string{precision});
}

double bytes_per_element(Precision precision) noexcept {
    return storage_bytes_per_element(precision);
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
DenseLayerTimes predict_kda_attention_times(const DenseLayerContext &context) {
    struct KdaAttentionWork {
        KernelWork projection{0.0, 0.0};
        KernelWork short_conv{0.0, 0.0};
        KernelWork recurrent_prefill{0.0, 0.0};
        KernelWork recurrent_decode{0.0, 0.0};
        KernelWork gate_norm{0.0, 0.0};
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
    const double token_count = static_cast<double>(tokens);
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
    work.projection = add_kernel_work(
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
    work.short_conv = short_conv;

    // The delta-rule update performs a query/state product and a key/value
    // outer-product update for each head.  The fixed recurrent state is
    // touched once per token in this conservative roofline model.  It keeps
    // decode work O(1) in context length while retaining the quadratic head
    // dimension dependence of the state matrix.
    //
    // FUTURE WORK (docs/design/kimi-k3-support.md 12.1): this models a strictly
    // sequential scan.  FlashKDA-style kernels process a chunk of tokens
    // against a state held in registers and only combine chunk results
    // sequentially, so the state reaches HBM once per chunk.  Per-token traffic
    // makes prefill memory bound by roughly 675:1 and overstates a KDA layer by
    // about an order of magnitude; decode, which touches the state once per
    // request either way, is unaffected.
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

    // Fused RMSNorm + sigmoid gate: one read/write pass over the projected
    // output with a small constant amount of scalar work per element.
    work.gate_norm = streaming_work(
        token_count * v_channels, token_count * v_channels,
        8.0 * token_count * v_channels, context.attention_element_bytes);

    DenseLayerTimes times{};
    // Keep the historical attention buckets populated while exposing the KDA
    // sub-components for diagnostics and focused tests.
    times.kda_projection_ms = predict_attention_work_ms(
        context, work.projection,
        gemm_efficiency_for(context, context.batch.total_tokens));
    times.kda_short_conv_ms = predict_attention_work_ms(
        context, work.short_conv, context.config.streaming);
    times.kda_recurrent_ms =
        predict_attention_work_ms(context, work.recurrent_prefill,
                                  context.config.prefill_attention) +
        predict_attention_work_ms(context, work.recurrent_decode,
                                  context.config.decode_attention);
    times.kda_gate_norm_ms = predict_attention_work_ms(
        context, work.gate_norm, context.config.streaming);
    times.attention_pre_projection_ms =
        times.kda_projection_ms + times.kda_short_conv_ms;
    times.attention_post_projection_ms = predict_attention_work_ms(
        context,
        attention_gemm_work(context, context.batch.total_tokens,
                            dense_ceil_div(context.model.kda_num_v_heads,
                                           context.model.tensor_parallel_size) *
                                context.model.kda_value_head_dim,
                            context.model.hidden_size),
        gemm_efficiency_for(context, context.batch.total_tokens));
    times.attention_inter_norm_ms = times.kda_gate_norm_ms;
    times.prefill_attention_ms = predict_attention_work_ms(
        context, work.recurrent_prefill, context.config.prefill_attention);
    times.decode_attention_ms = predict_attention_work_ms(
        context, work.recurrent_decode, context.config.decode_attention);
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
