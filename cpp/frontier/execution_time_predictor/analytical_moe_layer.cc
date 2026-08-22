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

struct MegaMoeGeometryWork {
    ExpertGemmWork padded_work;
    MoEGroupedGemmGeometry geometry;
    std::uint64_t routed_up_cluster_tasks = 0;
    std::uint64_t routed_down_cluster_tasks = 0;
    std::uint64_t shared_up_cluster_tasks = 0;
    std::uint64_t shared_down_cluster_tasks = 0;
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

double predict_router_gemm_ms(const MoELayerContext &context,
                              const KernelWork &work,
                              const Efficiency &efficiency) {
    // FP32 router logits describe accumulation/output width, not an FP32
    // CUDA-core GEMM. BF16 inputs and weights execute on the BF16 tensor-core
    // roofline while still writing FP32 logits.
    return predict_ms(context.device, context.router_weight_precision, work,
                      efficiency, context.config.kernel_launch_latency_us);
}

double predict_router_topk_ms(const MoELayerContext &context,
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

std::uint64_t mega_moe_block_m(const MoELayerContext &context) {
    const long double expected_tokens_per_expert =
        static_cast<long double>(context.input_tokens) *
        static_cast<long double>(context.router_topk) /
        static_cast<long double>(context.model.model_num_experts);
    if (expected_tokens_per_expert <= 8.5L) {
        return 16;
    }
    if (expected_tokens_per_expert <= 16.5L) {
        return 32;
    }
    if (expected_tokens_per_expert <= 32.5L) {
        return 64;
    }
    if (expected_tokens_per_expert <= 64.5L) {
        return 96;
    }
    if (expected_tokens_per_expert <= 96.5L) {
        return 128;
    }
    return 192;
}

std::uint64_t padded_m_tokens(std::uint64_t tokens, std::uint64_t block_m) {
    return checked_math::multiply<AnalyticalModelError>(
        ceil_div(tokens, block_m), block_m,
        "MegaMoE padded token count overflows");
}

std::uint64_t add_geometry_count(std::uint64_t lhs, std::uint64_t rhs) {
    return checked_math::add<AnalyticalModelError>(
        lhs, rhs, "MegaMoE geometry count overflows");
}

std::uint64_t multiply_geometry_count(std::uint64_t lhs, std::uint64_t rhs) {
    return checked_math::multiply<AnalyticalModelError>(
        lhs, rhs, "MegaMoE geometry count overflows");
}

void validate_mega_moe_geometry_config(const AnalyticalConfig &config) {
    if (config.mega_moe_sm_count == 0 || config.mega_moe_block_n == 0 ||
        config.mega_moe_cluster_size == 0 ||
        config.mega_moe_sm_count < config.mega_moe_cluster_size) {
        throw AnalyticalModelError(
            "MegaMoE SM, block-N, and cluster sizes must be positive");
    }
    if (!std::isfinite(config.mega_moe_tail_io_fraction) ||
        config.mega_moe_tail_io_fraction < 0.0 ||
        config.mega_moe_tail_io_fraction > 1.0 ||
        !std::isfinite(config.mega_moe_wave_exposure) ||
        config.mega_moe_wave_exposure < 0.0 ||
        config.mega_moe_wave_exposure > 1.0 ||
        !std::isfinite(config.mega_moe_cluster_task_latency_us) ||
        config.mega_moe_cluster_task_latency_us < 0.0) {
        throw AnalyticalModelError(
            "MegaMoE tail IO/wave exposure must be in [0, 1] and cluster "
            "task latency must be finite and non-negative");
    }
}

double mega_moe_wave_utilization(const AnalyticalConfig &config,
                                 std::uint64_t cluster_tasks) {
    if (cluster_tasks == 0) {
        return 1.0;
    }
    const std::uint64_t clusters_per_wave =
        config.mega_moe_sm_count / config.mega_moe_cluster_size;
    const std::uint64_t waves = ceil_div(cluster_tasks, clusters_per_wave);
    const std::uint64_t wave_capacity =
        multiply_geometry_count(waves, clusters_per_wave);
    return static_cast<double>(cluster_tasks) /
           static_cast<double>(wave_capacity);
}

MegaMoeGeometryWork build_mega_moe_geometry_work(
    const MoELayerContext &context,
    const std::vector<std::uint64_t> &local_expert_tokens) {
    validate_mega_moe_geometry_config(context.config);
    MegaMoeGeometryWork result{};
    result.geometry.enabled = true;
    result.geometry.block_m = mega_moe_block_m(context);

    for (const std::uint64_t tokens : local_expert_tokens) {
        if (tokens == 0) {
            continue;
        }
        const std::uint64_t padded =
            padded_m_tokens(tokens, result.geometry.block_m);
        add_expert_gemm_work(result.padded_work, context, padded, true);
        result.geometry.routed_m_blocks =
            add_geometry_count(result.geometry.routed_m_blocks,
                               ceil_div(tokens, result.geometry.block_m));
        result.geometry.routed_padded_tokens =
            add_geometry_count(result.geometry.routed_padded_tokens, padded);
    }
    for (std::uint64_t expert = 0; expert < context.model.num_shared_experts;
         ++expert) {
        if (context.input_tokens == 0) {
            continue;
        }
        const std::uint64_t padded =
            padded_m_tokens(context.input_tokens, result.geometry.block_m);
        add_expert_gemm_work(result.padded_work, context, padded, false);
        result.geometry.shared_m_blocks = add_geometry_count(
            result.geometry.shared_m_blocks,
            ceil_div(context.input_tokens, result.geometry.block_m));
        result.geometry.shared_padded_tokens =
            add_geometry_count(result.geometry.shared_padded_tokens, padded);
    }

    const std::uint64_t cluster_n = multiply_geometry_count(
        context.config.mega_moe_cluster_size, context.config.mega_moe_block_n);
    const std::uint64_t up_width =
        context.model.gated_mlp
            ? multiply_geometry_count(context.local_intermediate, 2)
            : context.local_intermediate;
    const std::uint64_t up_n_clusters = ceil_div(up_width, cluster_n);
    const std::uint64_t routed_down_width =
        context.model.routed_expert_hidden_size == 0
            ? context.model.hidden_size
            : context.model.routed_expert_hidden_size;
    const std::uint64_t routed_down_n_clusters =
        ceil_div(routed_down_width, cluster_n);
    const std::uint64_t shared_down_n_clusters =
        ceil_div(context.model.hidden_size, cluster_n);

    result.routed_up_cluster_tasks =
        multiply_geometry_count(result.geometry.routed_m_blocks, up_n_clusters);
    result.shared_up_cluster_tasks =
        multiply_geometry_count(result.geometry.shared_m_blocks, up_n_clusters);
    result.routed_down_cluster_tasks = multiply_geometry_count(
        result.geometry.routed_m_blocks, routed_down_n_clusters);
    result.shared_down_cluster_tasks = multiply_geometry_count(
        result.geometry.shared_m_blocks, shared_down_n_clusters);
    result.geometry.up_cluster_tasks = add_geometry_count(
        result.routed_up_cluster_tasks, result.shared_up_cluster_tasks);
    result.geometry.down_cluster_tasks = add_geometry_count(
        result.routed_down_cluster_tasks, result.shared_down_cluster_tasks);
    result.geometry.up_wave_utilization = mega_moe_wave_utilization(
        context.config, result.geometry.up_cluster_tasks);
    result.geometry.down_wave_utilization = mega_moe_wave_utilization(
        context.config, result.geometry.down_cluster_tasks);
    result.geometry.up_cluster_task_overhead_ms =
        static_cast<double>(result.geometry.up_cluster_tasks) *
        context.config.mega_moe_cluster_task_latency_us / 1000.0;
    result.geometry.down_cluster_task_overhead_ms =
        static_cast<double>(result.geometry.down_cluster_tasks) *
        context.config.mega_moe_cluster_task_latency_us / 1000.0;
    return result;
}

KernelWork mega_moe_tail_work(const KernelWork &actual,
                              const KernelWork &padded,
                              double tail_io_fraction) {
    if (padded.flops < actual.flops || padded.hbm_bytes < actual.hbm_bytes) {
        throw AnalyticalModelError(
            "MegaMoE padded work must not be smaller than actual work");
    }
    return KernelWork{
        padded.flops,
        actual.hbm_bytes +
            tail_io_fraction * (padded.hbm_bytes - actual.hbm_bytes),
    };
}

double predict_mega_moe_work_ms(const MoELayerContext &context,
                                Precision precision, const KernelWork &actual,
                                const KernelWork &padded,
                                std::uint64_t cluster_tasks) {
    const KernelWork work = mega_moe_tail_work(
        actual, padded, context.config.mega_moe_tail_io_fraction);
    const RooflineResult roofline =
        predict_roofline(context.device, precision, work, context.config.moe,
                         context.config.kernel_launch_latency_us);
    if (roofline.predicted_time_ms == 0.0) {
        return 0.0;
    }
    const double utilization =
        mega_moe_wave_utilization(context.config, cluster_tasks);
    const double wave_factor =
        1.0 + context.config.mega_moe_wave_exposure * (1.0 / utilization - 1.0);
    return roofline.launch_time_ms +
           (roofline.predicted_time_ms - roofline.launch_time_ms) *
               wave_factor +
           static_cast<double>(cluster_tasks) *
               context.config.mega_moe_cluster_task_latency_us / 1000.0;
}

const Efficiency &router_gemm_efficiency(const MoELayerContext &context) {
    return context.input_tokens < context.config.small_gemm_token_threshold
               ? context.config.small_gemm
               : context.config.large_gemm;
}

const Efficiency &
latent_moe_front_efficiency(const MoELayerContext &context) {
    return context.input_tokens < context.config.small_gemm_token_threshold
               ? context.config.latent_moe_front
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
    const MegaMoeGeometryWork mega_moe =
        context.config.mega_moe_geometry_enabled
            ? build_mega_moe_geometry_work(context, local_expert_tokens)
            : MegaMoeGeometryWork{};
    const double tokens = static_cast<double>(context.input_tokens);
    const double hidden = static_cast<double>(context.model.hidden_size);
    const double experts = static_cast<double>(context.model.model_num_experts);
    const double routed = static_cast<double>(expert_work.routed_tokens);
    const double norm_factor = context.model.fused_add_norm ? 3.0 : 2.0;
    const std::uint64_t latent_hidden_size =
        context.model.routed_expert_hidden_size == 0
            ? context.model.hidden_size
            : context.model.routed_expert_hidden_size;
    const bool fuse_latent_moe_front =
        context.config.fuse_latent_moe_front &&
        context.model.routed_expert_hidden_size != 0 &&
        context.router_weight_precision ==
            context.latent_moe_projection_weight_precision &&
        context.router_element_bytes ==
            context.latent_moe_projection_element_bytes;
    const bool plain_tp_decode = context.model.decode_only &&
                                 context.model.expert_parallel_size == 1 &&
                                 context.model.moe_tensor_parallel_size > 1;
    const bool fuse_shared_expert_front =
        fuse_latent_moe_front && plain_tp_decode &&
        context.config.fuse_shared_expert_moe_front &&
        context.model.num_shared_experts > 0 &&
        context.shared_expert_weight_precision ==
            context.router_weight_precision &&
        context.shared_expert_element_bytes == context.router_element_bytes;

    MoELayerTime result{};
    const KernelWork router_work = gemm_work(
        context.input_tokens, context.model.hidden_size,
        context.model.model_num_experts, context.router_weight_element_bytes,
        context.router_element_bytes,
        bytes_per_element(context.router_compute_precision), 1);
    if (fuse_latent_moe_front) {
        // SGLang concatenates the 896-column gate and 3584-column LatentMoE
        // down-projection weights. Both slices emit FP32 from one BF16 GEMM;
        // the routed slice is rounded to BF16 by its consumer. Combining the
        // two logical work records must remove the duplicate hidden-state
        // read that a pair of standalone GEMMs would incur.
        const KernelWork latent_down_work = gemm_work(
            context.input_tokens, context.model.hidden_size,
            latent_hidden_size,
            context.latent_moe_projection_weight_element_bytes,
            context.latent_moe_projection_element_bytes,
            bytes_per_element(context.router_compute_precision), 1);
        KernelWork fused_front_work =
            combined_kernel_work(router_work, latent_down_work);
        fused_front_work.hbm_bytes -=
            tokens * hidden * context.router_element_bytes;
        if (fuse_shared_expert_front) {
            // Plain TP K3 merges the TP-local shared gate/up slice as the
            // third output of [shared gate_up | router | latent down].  The
            // multiplier represents the two shared experts and SiTU's gated
            // pair while retaining a single hidden-state read.
            const KernelWork shared_front_work = gemm_work(
                context.input_tokens, context.model.hidden_size,
                context.local_intermediate,
                context.shared_expert_weight_element_bytes,
                context.shared_expert_element_bytes,
                bytes_per_element(context.router_compute_precision),
                context.model.num_shared_experts *
                    (context.model.gated_mlp ? 2 : 1));
            fused_front_work =
                combined_kernel_work(fused_front_work, shared_front_work);
            fused_front_work.hbm_bytes -=
                tokens * hidden * context.router_element_bytes;
        }
        result.gating_linear_ms = predict_router_gemm_ms(
            context, fused_front_work,
            latent_moe_front_efficiency(context));
    } else {
        result.gating_linear_ms = predict_router_gemm_ms(
            context, router_work, router_gemm_efficiency(context));
    }
    KernelWork routing_work = streaming_work(
        tokens * experts,
        tokens * static_cast<double>(context.router_topk),
        4.0 * tokens * experts,
        bytes_per_element(context.router_compute_precision));
    const bool fuse_route_quant =
        context.model.decode_only && context.config.fuse_route_quant &&
        context.model.routed_expert_hidden_size != 0;
    if (fuse_route_quant) {
        // SGLang's K3 route_quant_fused launch runs the radix-routing CTAs and
        // one MXFP8 quant CTA per source token together. The packed expert ids
        // are tiny; include the activation and scale traffic without creating
        // the old second launch or top-k-expanded activation copies.
        const double latent = static_cast<double>(latent_hidden_size);
        routing_work.hbm_bytes +=
            tokens * latent *
                (context.latent_moe_projection_element_bytes +
                 context.expert_element_bytes) +
            tokens * std::ceil(latent / 32.0) * 4.0;
        routing_work.flops += 4.0 * tokens * latent;
    }
    result.gating_routing_topk_ms = predict_router_topk_ms(
        context, routing_work, context.config.routing);
    const bool shared_uses_routed_kernel =
        context.shared_expert_weight_precision ==
            context.expert_weight_precision &&
        context.shared_expert_weight_element_bytes ==
            context.expert_weight_element_bytes &&
        context.shared_expert_element_bytes == context.expert_element_bytes;
    const double routed_up_projection_ms =
        context.config.mega_moe_geometry_enabled
            ? predict_mega_moe_work_ms(context, context.expert_weight_precision,
                                       expert_work.routed_up,
                                       mega_moe.padded_work.routed_up,
                                       mega_moe.routed_up_cluster_tasks)
            : predict_expert_work_ms(context, expert_work.routed_up,
                                     context.config.moe);
    const double routed_down_projection_ms =
        context.config.mega_moe_geometry_enabled
            ? predict_mega_moe_work_ms(context, context.expert_weight_precision,
                                       expert_work.routed_down,
                                       mega_moe.padded_work.routed_down,
                                       mega_moe.routed_down_cluster_tasks)
            : predict_expert_work_ms(context, expert_work.routed_down,
                                     context.config.moe);
    const double shared_up_projection_ms =
        context.config.mega_moe_geometry_enabled
            ? predict_mega_moe_work_ms(
                  context, context.shared_expert_weight_precision,
                  expert_work.shared_up, mega_moe.padded_work.shared_up,
                  mega_moe.shared_up_cluster_tasks)
            : predict_shared_expert_work_ms(context, expert_work.shared_up,
                                            context.config.moe);
    const double shared_down_projection_ms =
        context.config.mega_moe_geometry_enabled
            ? predict_mega_moe_work_ms(
                  context, context.shared_expert_weight_precision,
                  expert_work.shared_down, mega_moe.padded_work.shared_down,
                  mega_moe.shared_down_cluster_tasks)
            : predict_shared_expert_work_ms(context, expert_work.shared_down,
                                            context.config.moe);
    if (shared_uses_routed_kernel && !fuse_shared_expert_front) {
        KernelWork combined_up = expert_work.routed_up;
        KernelWork combined_down = expert_work.routed_down;
        add_kernel_work(combined_up, expert_work.shared_up);
        add_kernel_work(combined_down, expert_work.shared_down);
        if (context.config.mega_moe_geometry_enabled) {
            const KernelWork padded_up = combined_kernel_work(
                mega_moe.padded_work.routed_up, mega_moe.padded_work.shared_up);
            const KernelWork padded_down =
                combined_kernel_work(mega_moe.padded_work.routed_down,
                                     mega_moe.padded_work.shared_down);
            result.grouped_up_projection_ms = predict_mega_moe_work_ms(
                context, context.expert_weight_precision, combined_up,
                padded_up, mega_moe.geometry.up_cluster_tasks);
            result.grouped_down_projection_ms = predict_mega_moe_work_ms(
                context, context.expert_weight_precision, combined_down,
                padded_down, mega_moe.geometry.down_cluster_tasks);
        } else {
            result.grouped_up_projection_ms = predict_expert_work_ms(
                context, combined_up, context.config.moe);
            result.grouped_down_projection_ms = predict_expert_work_ms(
                context, combined_down, context.config.moe);
        }
    } else {
        result.grouped_up_projection_ms =
            routed_up_projection_ms +
            (fuse_shared_expert_front ? 0.0 : shared_up_projection_ms);
        result.grouped_down_projection_ms =
            routed_down_projection_ms + shared_down_projection_ms;
    }
    result.grouped_gemm_geometry = mega_moe.geometry;
    result.shuffling_ms =
        fuse_route_quant
            ? 0.0
            : predict_expert_work_ms(
                  context,
                  streaming_work(
                      routed * static_cast<double>(latent_hidden_size),
                      routed * static_cast<double>(latent_hidden_size), 0.0,
                      context.expert_element_bytes),
                  context.config.streaming);
    if (fuse_shared_expert_front &&
        context.config.overlap_shared_routed_moe) {
        // After the merged front, the BF16 shared down GEMM runs on the side
        // stream while routed MXFP4 experts run on the main stream. Charge the
        // producer/consumer critical path rather than adding both branches.
        const double routed_path = routed_up_projection_ms +
                                   routed_down_projection_ms +
                                   result.shuffling_ms;
        const double critical_path =
            std::max(routed_path, shared_down_projection_ms);
        result.grouped_up_projection_ms = routed_up_projection_ms;
        result.grouped_down_projection_ms = std::max(
            0.0, critical_path - routed_up_projection_ms -
                     result.shuffling_ms);
    }
    if (context.model.routed_expert_hidden_size != 0) {
        KernelWork latent_projection = gemm_work(
            context.input_tokens, latent_hidden_size, context.model.hidden_size,
            context.latent_moe_projection_weight_element_bytes,
            context.latent_moe_projection_element_bytes, 1);
        if (!fuse_latent_moe_front) {
            latent_projection = combined_kernel_work(
                gemm_work(
                    context.input_tokens, context.model.hidden_size,
                    latent_hidden_size,
                    context.latent_moe_projection_weight_element_bytes,
                    context.latent_moe_projection_element_bytes, 1),
                latent_projection);
        }
        const bool use_gemm_allgather =
            plain_tp_decode &&
            context.config.use_tp8_latent_up_gemm_allgather &&
            context.model.moe_tensor_parallel_size == 8 &&
            context.input_tokens <=
                context.config.tp8_latent_up_gemm_allgather_max_tokens;
        if (use_gemm_allgather) {
            // TP8 K3 reads one eighth of the replicated up-projection weight
            // per rank, multicasts the local columns, and folds add3 into the
            // consumer. The producer and consumer are separate PDL launches.
            const std::uint64_t local_hidden =
                ceil_div(context.model.hidden_size,
                         context.model.moe_tensor_parallel_size);
            const KernelWork local_projection = gemm_work(
                context.input_tokens, latent_hidden_size, local_hidden,
                context.latent_moe_projection_weight_element_bytes,
                context.latent_moe_projection_element_bytes, 1);
            const double local_ms = predict_latent_moe_projection_work_ms(
                context, local_projection, router_gemm_efficiency(context));
            const double gathered_bytes =
                tokens * hidden *
                context.latent_moe_projection_element_bytes;
            const double gather_ms =
                context.config.kernel_launch_latency_us / 1000.0 +
                gathered_bytes /
                    (context.config
                         .tp8_latent_up_gemm_allgather_bandwidth_tbps *
                     1e12) *
                    1e3;
            result.latent_projection_ms = local_ms + gather_ms;
        } else {
            result.latent_projection_ms =
                predict_latent_moe_projection_work_ms(
                    context, latent_projection,
                    router_gemm_efficiency(context));
        }
        if (context.model.latent_moe_use_norm && !use_gemm_allgather) {
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
    result.routed_path_ms = routed_up_projection_ms +
                            routed_down_projection_ms + result.shuffling_ms;
    result.shared_expert_path_ms =
        (fuse_shared_expert_front ? 0.0 : shared_up_projection_ms) +
        shared_down_projection_ms;
    result.source_local_ms =
        result.gating_linear_ms + result.gating_routing_topk_ms +
        result.shared_expert_path_ms + result.post_attention_norm_ms +
        result.latent_projection_ms + result.latent_norm_ms +
        result.attn_res_ms;
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
                                    const MoEOperatorPrecisions &precisions,
                                    bool enable_group_mega_moe_geometry) {
    if (routing.lane_expert_tokens.empty()) {
        throw AnalyticalModelError(
            "MoE routing must contain at least one lane");
    }
    AnalyticalConfig effective_config = config;
    effective_config.mega_moe_geometry_enabled =
        config.mega_moe_geometry_enabled && enable_group_mega_moe_geometry;
    MoELanePrediction prediction;
    prediction.lane_times.reserve(routing.lane_expert_tokens.size());
    prediction.routed_lane_times_ms.reserve(routing.lane_expert_tokens.size());
    prediction.source_local_lane_times_ms.reserve(
        routing.lane_expert_tokens.size());
    for (const auto &lane : routing.lane_expert_tokens) {
        MoELayerTime layer = predict_moe_layer(device, effective_config, model,
                                               routing.input_tokens,
                                               router_topk, lane, precisions);
        if (prediction.lane_times.empty()) {
            prediction.shared_expert_path_ms = layer.shared_expert_path_ms;
        } else if (layer.shared_expert_path_ms !=
                   prediction.shared_expert_path_ms) {
            throw AnalyticalModelError(
                "source-local shared-expert time must be lane invariant");
        }
        prediction.routed_lane_times_ms.push_back(layer.routed_path_ms);
        prediction.source_local_lane_times_ms.push_back(layer.source_local_ms);
        prediction.lane_times.push_back(std::move(layer));
    }
    for (std::size_t lane = 0; lane < prediction.lane_times.size(); ++lane) {
        const double time = prediction.lane_times[lane].total_ms();
        if (lane == 0 || time > prediction.critical_lane_time_ms) {
            prediction.critical_lane = static_cast<std::uint64_t>(lane);
            prediction.critical_lane_time_ms = time;
        }
        const double routed_time = prediction.routed_lane_times_ms.at(lane);
        if (lane == 0 ||
            routed_time > prediction.routed_critical_lane_time_ms) {
            prediction.routed_critical_lane = static_cast<std::uint64_t>(lane);
            prediction.routed_critical_lane_time_ms = routed_time;
        }
    }
    return prediction;
}

MoELanePrediction
predict_moe_lanes(const DeviceCeilings &device, const AnalyticalConfig &config,
                  const MoEModel &model, const RoutingAllocation &routing,
                  std::uint64_t router_topk, Precision precision,
                  bool enable_group_mega_moe_geometry) {
    return predict_moe_lanes(device, config, model, routing, router_topk,
                             uniform_operator_precisions(precision),
                             enable_group_mega_moe_geometry);
}

MoERoutedLanePrediction predict_routed_moe_lanes(
    const DeviceCeilings &device, const AnalyticalConfig &config,
    const MoEModel &model, const RoutingAllocation &routing,
    std::uint64_t router_topk, const MoEOperatorPrecisions &precisions,
    bool enable_group_mega_moe_geometry) {
    if (routing.lane_expert_tokens.empty()) {
        throw AnalyticalModelError(
            "MoE routing must contain at least one lane");
    }
    AnalyticalConfig effective_config = config;
    effective_config.mega_moe_geometry_enabled =
        config.mega_moe_geometry_enabled && enable_group_mega_moe_geometry;
    MoEModel routed_only_model = model;
    routed_only_model.num_shared_experts = 0;

    MoERoutedLanePrediction result{};
    result.lane_times_ms.reserve(routing.lane_expert_tokens.size());
    for (const auto &lane : routing.lane_expert_tokens) {
        const MoELayerTime time = predict_moe_layer(
            device, effective_config, routed_only_model, routing.input_tokens,
            router_topk, lane, precisions);
        result.lane_times_ms.push_back(time.routed_path_ms);
        const std::size_t lane_index = result.lane_times_ms.size() - 1;
        if (lane_index == 0 ||
            time.routed_path_ms > result.critical_lane_time_ms) {
            result.critical_lane = static_cast<std::uint64_t>(lane_index);
            result.critical_lane_time_ms = time.routed_path_ms;
            result.critical_lane_geometry = time.grouped_gemm_geometry;
        }
    }
    return result;
}

} // namespace frontier::execution_time_predictor::detail
