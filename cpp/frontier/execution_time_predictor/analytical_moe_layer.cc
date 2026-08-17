// MoE expert-lane roofline model and collective communication costs.
//
// MoE operator roofline implementation. Public internal contracts live in
// analytical_moe_model.h.

#include "frontier/execution_time_predictor/analytical_moe_model.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "frontier/cc_backend/analytical_model.h"
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

MoEOperatorPrecisions uniform_operator_precisions(Precision precision) {
    MoEOperatorPrecisions result{};
    result.expert = precision;
    result.router = precision;
    result.dense = precision;
    result.shared_expert = precision;
    return result;
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

double predict_latent_moe_projection_work_ms(const MoELayerContext &context,
                                             const KernelWork &work,
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

// Value-returning companion to add_kernel_work, for the places that combine
// two temporaries rather than accumulating into an existing total.
KernelWork combined_kernel_work(const KernelWork &lhs, const KernelWork &rhs) {
    return KernelWork{lhs.flops + rhs.flops, lhs.hbm_bytes + rhs.hbm_bytes};
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
    const double activation_bytes = count_as_routed
                                        ? context.expert_element_bytes
                                        : context.shared_expert_element_bytes;
    add_kernel_work(up, gemm_work(tokens, expert_input_size,
                                  context.local_intermediate, weight_bytes,
                                  activation_bytes,
                                  context.model.gated_mlp ? 2 : 1));
    add_kernel_work(down, gemm_work(tokens, context.local_intermediate,
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
        result.grouped_up_projection_ms =
            predict_expert_work_ms(context, combined_up, context.config.moe);
        result.grouped_down_projection_ms =
            predict_expert_work_ms(context, combined_down, context.config.moe);
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
        const KernelWork latent_projection = combined_kernel_work(
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
    return predict_moe_layer(device, config, model, input_tokens, router_topk,
                             local_expert_tokens,
                             uniform_operator_precisions(precision));
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
    return predict_moe_lanes(device, config, model, routing, router_topk,
                             uniform_operator_precisions(precision));
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
    std::uint64_t routed_hidden_size,
    std::string_view moe_communication_backend,
    const RoutingAllocation *routing, double fused_expert_compute_ms) {
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
            // Public-prior GB200/GB300 NVL72 one-sided A2A envelope.  The
            // startup terms (roughly 18--22 us dispatch and 31--33 us combine)
            // and effective bandwidth range (about 0.59--0.75 TB/s) follow
            // published DeepEP/MegaMoE-family measurements rather than peak
            // NVLink bandwidth. Interpolate conservatively in log2(EP).
            const double log_ep = std::log2(
                static_cast<double>(expert_parallel_size));
            const double ep_position = std::clamp((log_ep - 3.0) / 3.0,
                                                  0.0, 1.0);
            const double dispatch_startup_us = 18.0 + 4.0 * ep_position;
            const double combine_startup_us = 31.0 + 2.0 * ep_position;
            const double dispatch_gbps = 753.0 - 165.0 * ep_position;
            const double combine_gbps = 728.0 - 97.0 * ep_position;

            // The rank-major one-sided buffer stores one copy per
            // (source token, destination EP lane), even when several selected
            // experts for that token live on the same lane.  Expert-route
            // counts still drive lane-local GEMM work, but communication must
            // use the deduplicated token counts.
            std::uint64_t maximum_lane_unique_tokens =
                expert_parallel_size == 0
                    ? 0
                    : std::min(
                          input_tokens,
                          routed_tokens / expert_parallel_size +
                              static_cast<std::uint64_t>(
                                  routed_tokens % expert_parallel_size != 0));
            if (routing != nullptr && !routing->lane_unique_tokens.empty()) {
                maximum_lane_unique_tokens = *std::max_element(
                    routing->lane_unique_tokens.begin(),
                    routing->lane_unique_tokens.end());
            }
            const std::uint64_t routed_width =
                routed_hidden_size == 0 ? hidden_size : routed_hidden_size;
            const std::uint64_t maximum_dispatch_bytes = payload_bytes(
                maximum_lane_unique_tokens, routed_width, element_bytes);
            // MegaMoE combines BF16 expert outputs even when the dispatch
            // activation is FP8.
            const std::uint64_t maximum_combine_bytes = payload_bytes(
                maximum_lane_unique_tokens, routed_width,
                std::max(2.0, element_bytes));
            const double remote_fraction =
                static_cast<double>(expert_parallel_size - 1) /
                static_cast<double>(expert_parallel_size);
            value.raw_ep_dispatch_ms = dispatch_startup_us * 1e-3 +
                remote_fraction * static_cast<double>(maximum_dispatch_bytes) /
                    (dispatch_gbps * 1e6);
            value.raw_ep_combine_ms = combine_startup_us * 1e-3 +
                remote_fraction * static_cast<double>(maximum_combine_bytes) /
                    (combine_gbps * 1e6);

            const double raw_total =
                value.raw_ep_dispatch_ms + value.raw_ep_combine_ms;
            // Fused dispatch/GEMM/combine overlaps most, but not all, of the
            // shorter path. A 35% residual is a named public prior, not a
            // fitted result for the simulator workload.
            const double fused_total =
                std::max(fused_expert_compute_ms, raw_total) +
                0.35 * std::min(fused_expert_compute_ms, raw_total);
            const double exposed =
                std::max(0.0, fused_total - fused_expert_compute_ms);
            value.ep_dispatch_ms =
                raw_total > 0.0
                    ? exposed * value.raw_ep_dispatch_ms / raw_total
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
            value.dp_input_ms = communication.allreduce_ms(
                activation_bytes, data_parallel_size);
            value.dp_output_ms = communication.allreduce_ms(
                activation_bytes, data_parallel_size);
        }
        value.pipeline_parallel_ms =
            has_pipeline_boundary
                ? communication.point_to_point_ms(activation_bytes)
                : 0.0;
        return value;
    }();
}

} // namespace frontier::execution_time_predictor::detail
