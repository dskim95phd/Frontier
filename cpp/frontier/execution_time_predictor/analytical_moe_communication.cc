#include "frontier/execution_time_predictor/analytical_moe_model.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>

#include "frontier/core/checked_math.h"

namespace frontier::execution_time_predictor::detail {
namespace {

std::uint64_t ceil_div(std::uint64_t numerator, std::uint64_t denominator) {
    return checked_math::ceil_div<AnalyticalModelError>(
        numerator, denominator, "MoE TP size must be positive");
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

double interpolate_public_prior(
    std::uint64_t expert_parallel_size,
    const std::array<double, 4> &ep8_to_ep64_values) {
    const double log_ep =
        std::clamp(std::log2(static_cast<double>(expert_parallel_size)), 3.0,
                   6.0);
    const std::size_t lower = static_cast<std::size_t>(std::floor(log_ep)) - 3;
    if (lower >= ep8_to_ep64_values.size() - 1) {
        return ep8_to_ep64_values.back();
    }
    const double fraction = log_ep - std::floor(log_ep);
    return ep8_to_ep64_values[lower] * (1.0 - fraction) +
           ep8_to_ep64_values[lower + 1] * fraction;
}

} // namespace

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
    std::uint64_t routed_hidden_size,
    std::string_view moe_communication_backend,
    const RoutingAllocation *routing, double fused_expert_compute_ms,
    double overlap_residual, double a2a_bandwidth_scale,
    double a2a_startup_scale, double dispatch_element_bytes,
    std::uint64_t logical_unique_token_copies_override) {
    if (!std::isfinite(overlap_residual) || overlap_residual < 0.0 ||
        overlap_residual > 1.0) {
        throw AnalyticalModelError(
            "MoE communication overlap residual must be in [0, 1]");
    }
    if (!std::isfinite(a2a_bandwidth_scale) ||
        !std::isfinite(a2a_startup_scale) || a2a_bandwidth_scale <= 0.0 ||
        a2a_startup_scale <= 0.0) {
        throw AnalyticalModelError(
            "MoE A2A bandwidth and startup scales must be positive and "
            "finite");
    }
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
        if (moe_communication_backend == "sm100_megamoe_public" &&
            expert_parallel_size > 1) {
            // Linear fits of NVIDIA's public GB200 NVL72 measurements. Each
            // EP knot is fit across batch sizes 1--2048; intermediate EP sizes
            // interpolate in log2(EP). The fitted slope is deliberately
            // separate from the fixed barrier/startup intercept.
            const double dispatch_startup_us =
                interpolate_public_prior(expert_parallel_size,
                                         {15.763, 16.260, 17.550, 20.910}) *
                a2a_startup_scale;
            const double combine_startup_us =
                interpolate_public_prior(expert_parallel_size,
                                         {26.537, 26.892, 26.892, 28.726}) *
                a2a_startup_scale;
            const double dispatch_gbps =
                interpolate_public_prior(expert_parallel_size,
                                         {792.1, 718.2, 657.6, 622.2}) *
                a2a_bandwidth_scale;
            const double combine_gbps =
                interpolate_public_prior(expert_parallel_size,
                                         {800.3, 736.8, 692.4, 690.1}) *
                a2a_bandwidth_scale;

            // The rank-major one-sided buffer stores one copy per
            // (source token, destination EP lane), even when several selected
            // experts for that token live on the same lane.  Expert-route
            // counts still drive lane-local GEMM work, but communication must
            // use the deduplicated token counts.
            std::uint64_t logical_unique_token_copies =
                expert_parallel_size == 0
                    ? 0
                    : std::min(
                          routed_tokens,
                          checked_math::multiply<AnalyticalModelError>(
                              input_tokens, expert_parallel_size,
                              "MoE logical communication count overflows"));
            if (logical_unique_token_copies_override > 0) {
                logical_unique_token_copies =
                    logical_unique_token_copies_override;
            } else if (routing != nullptr &&
                       !routing->lane_unique_tokens.empty()) {
                logical_unique_token_copies = std::accumulate(
                    routing->lane_unique_tokens.begin(),
                    routing->lane_unique_tokens.end(), std::uint64_t{0});
            }
            const std::uint64_t routed_width =
                routed_hidden_size == 0 ? hidden_size : routed_hidden_size;
            const double dispatch_bytes_per_element =
                dispatch_element_bytes > 0.0 ? dispatch_element_bytes
                                             : element_bytes;
            const std::uint64_t maximum_dispatch_bytes = payload_bytes(
                logical_unique_token_copies, routed_width,
                dispatch_bytes_per_element);
            // Dispatch transports the post-quant routed activation. MegaMoE
            // combines BF16 expert outputs by default, independently of the
            // dispatch dtype.
            const std::uint64_t maximum_combine_bytes =
                payload_bytes(logical_unique_token_copies, routed_width,
                              std::max(2.0, element_bytes));
            // NVIDIA reports logical bandwidth, including the local-rank
            // fraction. Keep the same byte convention as the fitted prior.
            value.raw_ep_dispatch_ms =
                dispatch_startup_us * 1e-3 +
                static_cast<double>(maximum_dispatch_bytes) /
                    (dispatch_gbps * 1e6);
            value.raw_ep_combine_ms =
                combine_startup_us * 1e-3 +
                static_cast<double>(maximum_combine_bytes) /
                    (combine_gbps * 1e6);

            const double raw_total =
                value.raw_ep_dispatch_ms + value.raw_ep_combine_ms;
            // Fused dispatch/GEMM/combine may overlap the shorter path.  The
            // portable public prior exposes 35%; synchronized profiles can
            // explicitly request the full producer/consumer critical path.
            const double fused_total =
                std::max(fused_expert_compute_ms, raw_total) +
                overlap_residual * std::min(fused_expert_compute_ms, raw_total);
            const double exposed =
                std::max(0.0, fused_total - fused_expert_compute_ms);
            value.ep_dispatch_ms =
                raw_total > 0.0 ? exposed * value.raw_ep_dispatch_ms / raw_total
                                : 0.0;
            value.ep_combine_ms = exposed - value.ep_dispatch_ms;
            // With DP attention + EP experts, dispatch/combine are the layout
            // transition. Charging another DP all-reduce would double-count
            // it. TP collectives remain separate above.
            value.dp_input_ms = 0.0;
            value.dp_output_ms = 0.0;
        } else {
            value.ep_dispatch_ms =
                communication.all_to_all_ms(routed_bytes, expert_parallel_size);
            value.ep_combine_ms =
                communication.all_to_all_ms(routed_bytes, expert_parallel_size);
            value.raw_ep_dispatch_ms = value.ep_dispatch_ms;
            value.raw_ep_combine_ms = value.ep_combine_ms;
            value.dp_input_ms = communication.allreduce_ms(activation_bytes,
                                                           data_parallel_size);
            value.dp_output_ms = communication.allreduce_ms(activation_bytes,
                                                            data_parallel_size);
        }
        value.pipeline_parallel_ms =
            has_pipeline_boundary
                ? communication.point_to_point_ms(activation_bytes)
                : 0.0;
        return value;
    }();
}

} // namespace frontier::execution_time_predictor::detail
