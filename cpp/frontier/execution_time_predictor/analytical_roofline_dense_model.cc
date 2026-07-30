#include "frontier/execution_time_predictor/analytical_roofline_execution_time_predictor.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace frontier::execution_time_predictor::detail {
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

std::uint64_t ceil_div(std::uint64_t numerator, std::uint64_t denominator) {
    if (denominator == 0) {
        throw AnalyticalModelError("division denominator must be positive");
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

} // namespace

double DenseLayerTimes::total_ms() const noexcept {
    return attention_pre_projection_ms + attention_post_projection_ms +
           rope_ms + kv_cache_save_ms + attention_norm_ms +
           prefill_attention_ms + decode_attention_ms + mlp_up_projection_ms +
           mlp_activation_ms + mlp_down_projection_ms + mlp_norm_ms +
           2.0 * residual_add_ms;
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
    case Precision::kFp4:
    case Precision::kInt4:
        return 0.5;
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
    case Precision::kInt8:
        peak = device.fp8_tflops;
        break;
    case Precision::kFp4:
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
    require_finite_nonnegative(element_bytes, "element bytes");
    if (element_bytes == 0.0) {
        throw AnalyticalModelError("element bytes must be positive");
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
            element_bytes *
            (resolved_m * resolved_k + multiplier * resolved_k * resolved_n +
             multiplier * resolved_m * resolved_n);
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

KernelWork
attention_context_work(const std::vector<AttentionRequestSlice> &requests,
                       std::uint64_t local_query_heads,
                       std::uint64_t local_kv_heads, std::uint64_t head_dim,
                       double element_bytes) {
    if (local_query_heads == 0 || local_kv_heads == 0 || head_dim == 0) {
        throw AnalyticalModelError("attention dimensions must be positive");
    }
    require_finite_nonnegative(element_bytes, "element bytes");
    if (element_bytes == 0.0) {
        throw AnalyticalModelError("element bytes must be positive");
    }

    double total_flops = 0.0;
    double total_elements = 0.0;
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
        total_elements += q_and_output + new_kv_input + cached_kv_reads;
    }
    return [&]() {
        KernelWork value{};
        value.flops = total_flops;
        value.hbm_bytes = total_elements * element_bytes;
        return value;
    }();
}

DenseLayerTimes predict_dense_layer(const DeviceCeilings &device,
                                    const AnalyticalConfig &config,
                                    const DenseModel &model,
                                    const DenseBatch &batch,
                                    Precision precision) {
    if (model.hidden_size == 0 || model.intermediate_size == 0 ||
        model.num_query_heads == 0 || model.num_kv_heads == 0 ||
        model.head_dim == 0 || model.tensor_parallel_size == 0) {
        throw AnalyticalModelError("dense model dimensions must be positive");
    }
    if (config.small_gemm_token_threshold == 0) {
        throw AnalyticalModelError(
            "small GEMM token threshold must be positive");
    }

    const std::uint64_t scheduled_tokens = [&batch] {
        std::uint64_t total = 0;
        for (const AttentionRequestSlice &request : batch.prefill_requests) {
            total += request.query_tokens;
        }
        for (const AttentionRequestSlice &request : batch.decode_requests) {
            total += request.query_tokens;
        }
        return total;
    }();
    if (scheduled_tokens != batch.total_tokens) {
        throw AnalyticalModelError(
            "dense batch total_tokens does not match request slices");
    }

    const double element_bytes = bytes_per_element(precision);
    const std::uint64_t local_query_heads =
        ceil_div(model.num_query_heads, model.tensor_parallel_size);
    const std::uint64_t local_kv_heads =
        ceil_div(model.num_kv_heads, model.tensor_parallel_size);
    const std::uint64_t local_qkv_dim =
        (local_query_heads + 2 * local_kv_heads) * model.head_dim;
    const std::uint64_t local_intermediate =
        ceil_div(model.intermediate_size, model.tensor_parallel_size);
    const std::uint64_t gated_multiplier = model.gated_mlp ? 2 : 1;
    const Efficiency &gemm_efficiency =
        batch.total_tokens < config.small_gemm_token_threshold
            ? config.small_gemm
            : config.large_gemm;

    const auto kernel_ms = [&](const KernelWork &work,
                               const Efficiency &efficiency) {
        return predict_ms(device, precision, work, efficiency,
                          config.kernel_launch_latency_us);
    };

    const double query_heads = static_cast<double>(local_query_heads);
    const double kv_heads = static_cast<double>(local_kv_heads);
    const double head_dim = static_cast<double>(model.head_dim);
    const double tokens = static_cast<double>(batch.total_tokens);
    const double hidden = static_cast<double>(model.hidden_size);
    const double intermediate = static_cast<double>(local_intermediate);

    const KernelWork prefill_attention =
        attention_context_work(batch.prefill_requests, local_query_heads,
                               local_kv_heads, model.head_dim, element_bytes);
    const KernelWork decode_attention =
        attention_context_work(batch.decode_requests, local_query_heads,
                               local_kv_heads, model.head_dim, element_bytes);

    const double rope_elements = tokens * (query_heads + kv_heads) * head_dim;
    const double kv_elements = tokens * 2.0 * kv_heads * head_dim;
    const double attention_norm_factor = model.fused_add_norm ? 3.0 : 2.0;
    const double mlp_norm_factor = model.fused_add_norm ? 3.0 : 2.0;
    const double activation_elements = tokens * intermediate;

    double residual_add_ms = 0.0;
    if (!model.fused_add_norm) {
        const double residual_elements = tokens * hidden;
        residual_add_ms =
            kernel_ms(streaming_work(2.0 * residual_elements, residual_elements,
                                     residual_elements, element_bytes),
                      config.streaming);
    }

    return [&]() {
        DenseLayerTimes value{};
        value.attention_pre_projection_ms =
            kernel_ms(gemm_work(batch.total_tokens, model.hidden_size,
                                local_qkv_dim, element_bytes),
                      gemm_efficiency);
        value.attention_post_projection_ms = kernel_ms(
            gemm_work(batch.total_tokens,
                      std::max<std::uint64_t>(
                          1, model.hidden_size / model.tensor_parallel_size),
                      model.hidden_size, element_bytes),
            gemm_efficiency);
        value.rope_ms =
            kernel_ms(streaming_work(rope_elements, rope_elements,
                                     6.0 * rope_elements, element_bytes),
                      config.streaming);
        value.kv_cache_save_ms = kernel_ms(
            streaming_work(kv_elements, kv_elements, 0.0, element_bytes),
            config.streaming);
        value.attention_norm_ms = kernel_ms(
            streaming_work(tokens * hidden * (attention_norm_factor - 1.0),
                           tokens * hidden, 5.0 * tokens * hidden,
                           element_bytes),
            config.streaming);
        value.prefill_attention_ms =
            kernel_ms(prefill_attention, config.prefill_attention);
        value.decode_attention_ms =
            kernel_ms(decode_attention, config.decode_attention);
        value.mlp_up_projection_ms = kernel_ms(
            gemm_work(batch.total_tokens, model.hidden_size, local_intermediate,
                      element_bytes, gated_multiplier),
            gemm_efficiency);
        value.mlp_activation_ms = kernel_ms(
            streaming_work(
                activation_elements * static_cast<double>(gated_multiplier),
                activation_elements, 8.0 * activation_elements, element_bytes),
            config.streaming);
        value.mlp_down_projection_ms =
            kernel_ms(gemm_work(batch.total_tokens, local_intermediate,
                                model.hidden_size, element_bytes),
                      gemm_efficiency);
        value.mlp_norm_ms =
            kernel_ms(streaming_work(tokens * hidden * (mlp_norm_factor - 1.0),
                                     tokens * hidden, 5.0 * tokens * hidden,
                                     element_bytes),
                      config.streaming);
        value.residual_add_ms = residual_add_ms;
        return value;
    }();
}

} // namespace frontier::execution_time_predictor::detail
