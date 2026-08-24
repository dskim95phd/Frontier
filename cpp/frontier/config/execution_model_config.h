#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <variant>
#include <vector>

#include "frontier/attention/ops.h"
#include "frontier/core/cluster_type.h"
#include "frontier/core/ids.h"

#include "frontier/config/config_types.h"

namespace frontier::config {

enum class ExecutionModelType {
    kFixed,
    kAnalytical,
};

struct FixedExecutionModelConfig {
    double batch_latency_ms = 1.0;
    std::vector<double> stage_latencies_ms;

    friend bool operator==(const FixedExecutionModelConfig &lhs,
                           const FixedExecutionModelConfig &rhs) {
        return std::tie(lhs.batch_latency_ms, lhs.stage_latencies_ms) ==
               std::tie(rhs.batch_latency_ms, rhs.stage_latencies_ms);
    }
};

struct OperatorPrecisionConfig {
    // Empty fields inherit AnalyticalExecutionModelConfig::precision.
    std::string attention;
    std::string dense;
    std::string moe_expert;
    std::string moe_router;
    std::string kv_cache;
    std::string communication;
    std::string attention_weight;
    std::string attention_activation;
    std::string dense_weight;
    std::string dense_activation;
    std::string moe_expert_weight;
    std::string moe_expert_activation;
    std::string moe_router_weight;
    std::string moe_router_activation;
    // Canonical router weight storage precision.  `moe_router_weight` is
    // retained as a legacy fallback for configs written before router
    // compute/storage precisions were split.
    std::string router_weight_storage;
    std::string lm_head;
    std::string lm_head_weight;
    std::string lm_head_activation;
    // K3-native mixed-precision overrides.  Empty values are optional and
    // inherit the legacy family/suffix fields through the accessors below.
    std::string routed_expert_weight;
    std::string routed_expert_activation;
    // Stable LatentMoE's dense down/up projections sit outside the routed
    // expert bank and may use a different dtype.  Empty values retain the
    // historical routed-expert precision for backward compatibility.
    std::string latent_moe_projection_weight;
    std::string latent_moe_projection_activation;
    std::string shared_expert_weight;
    std::string shared_expert_activation;
    std::string dense_mlp_weight;
    std::string dense_mlp_activation;
    std::string router_compute;
    std::string kda_snapshot;

    [[nodiscard]] bool empty() const noexcept {
        return attention.empty() && dense.empty() && moe_expert.empty() &&
               moe_router.empty() && kv_cache.empty() &&
               communication.empty() && attention_weight.empty() &&
               attention_activation.empty() && dense_weight.empty() &&
               dense_activation.empty() && moe_expert_weight.empty() &&
               moe_expert_activation.empty() && moe_router_weight.empty() &&
               moe_router_activation.empty() && lm_head.empty() &&
               lm_head_weight.empty() && lm_head_activation.empty() &&
               router_weight_storage.empty() && routed_expert_weight.empty() &&
               routed_expert_activation.empty() &&
               latent_moe_projection_weight.empty() &&
               latent_moe_projection_activation.empty() &&
               shared_expert_weight.empty() &&
               shared_expert_activation.empty() && dense_mlp_weight.empty() &&
               dense_mlp_activation.empty() && router_compute.empty() &&
               kda_snapshot.empty();
    }

    friend bool operator==(const OperatorPrecisionConfig &lhs,
                           const OperatorPrecisionConfig &rhs) {
        return std::tie(lhs.attention, lhs.dense, lhs.moe_expert,
                        lhs.moe_router, lhs.kv_cache, lhs.communication,
                        lhs.attention_weight, lhs.attention_activation,
                        lhs.dense_weight, lhs.dense_activation,
                        lhs.moe_expert_weight, lhs.moe_expert_activation,
                        lhs.moe_router_weight, lhs.moe_router_activation,
                        lhs.router_weight_storage, lhs.lm_head,
                        lhs.lm_head_weight, lhs.lm_head_activation,
                        lhs.routed_expert_weight, lhs.routed_expert_activation,
                        lhs.latent_moe_projection_weight,
                        lhs.latent_moe_projection_activation,
                        lhs.shared_expert_weight, lhs.shared_expert_activation,
                        lhs.dense_mlp_weight, lhs.dense_mlp_activation,
                        lhs.router_compute, lhs.kda_snapshot) ==
               std::tie(rhs.attention, rhs.dense, rhs.moe_expert,
                        rhs.moe_router, rhs.kv_cache, rhs.communication,
                        rhs.attention_weight, rhs.attention_activation,
                        rhs.dense_weight, rhs.dense_activation,
                        rhs.moe_expert_weight, rhs.moe_expert_activation,
                        rhs.moe_router_weight, rhs.moe_router_activation,
                        rhs.router_weight_storage, rhs.lm_head,
                        rhs.lm_head_weight, rhs.lm_head_activation,
                        rhs.routed_expert_weight, rhs.routed_expert_activation,
                        rhs.latent_moe_projection_weight,
                        rhs.latent_moe_projection_activation,
                        rhs.shared_expert_weight, rhs.shared_expert_activation,
                        rhs.dense_mlp_weight, rhs.dense_mlp_activation,
                        rhs.router_compute, rhs.kda_snapshot);
    }
};

struct AnalyticalDeviceOverrides {
    std::optional<double> hbm_bandwidth_tbps;
    std::optional<double> fp32_tflops;
    std::optional<double> fp16_tflops;
    std::optional<double> fp8_tflops;
    std::optional<double> fp4_tflops;

    [[nodiscard]] bool empty() const noexcept {
        return !hbm_bandwidth_tbps.has_value() && !fp32_tflops.has_value() &&
               !fp16_tflops.has_value() && !fp8_tflops.has_value() &&
               !fp4_tflops.has_value();
    }

    [[nodiscard]] bool complete() const noexcept {
        return hbm_bandwidth_tbps.has_value() && fp32_tflops.has_value() &&
               fp16_tflops.has_value() && fp8_tflops.has_value() &&
               fp4_tflops.has_value();
    }

    friend bool operator==(const AnalyticalDeviceOverrides &lhs,
                           const AnalyticalDeviceOverrides &rhs) {
        return std::tie(lhs.hbm_bandwidth_tbps, lhs.fp32_tflops,
                        lhs.fp16_tflops, lhs.fp8_tflops, lhs.fp4_tflops) ==
               std::tie(rhs.hbm_bandwidth_tbps, rhs.fp32_tflops,
                        rhs.fp16_tflops, rhs.fp8_tflops, rhs.fp4_tflops);
    }
};

struct AnalyticalExecutionModelConfig {
    std::string device = "rubin";
    AnalyticalDeviceOverrides device_overrides;
    std::string precision = "fp16";
    OperatorPrecisionConfig operator_precisions;
    // Selects an explicit operator-efficiency/fusion profile.  "generic"
    // preserves the historical roofline constants.  K3 profiles are opt-in
    // because they describe backend-specific Blackwell kernel stacks. Rubin
    // use is an explicit forward projection of those public priors.
    std::string kernel_profile = "generic";
    // "detailed" predicts and emits synchronization events one MoE layer at a
    // time.
    // "first_layer_scaled" emits the first MoE layer normally, uses one
    // batch-shared routing draw regardless of moe_routing.layer_scope, reuses
    // its expert path, and accumulates attention delays by implementation
    // family.
    // "stage_group_scaled" additionally exposes canonical PP stage groups,
    // uses the same family-aware compression only for layer-invariant routing,
    // and otherwise falls back to exact per-layer prediction.
    std::string moe_layer_event_mode = "detailed";
    // "generic" uses the configured collective backend. The SM100 profile is
    // an opt-in public-prior model for fused MegaMoE dispatch/expert/combine;
    // on Rubin it is a forward projection rather than a calibrated profile.
    std::string moe_communication_backend = "generic";
    // Optional absolute MegaMoE sensitivity/calibration overrides. They are
    // valid only with k3_deepgemm_megamoe; absent values use profile priors.
    std::optional<double> mega_moe_tail_io_fraction;
    std::optional<double> mega_moe_wave_exposure;
    std::uint64_t tensor_parallel_size = 8;
    double network_bandwidth_gbps = 400.0;
    double network_latency_us = 1.0;
    double intra_node_bandwidth_gbps = 14'400.0;

    friend bool operator==(const AnalyticalExecutionModelConfig &lhs,
                           const AnalyticalExecutionModelConfig &rhs) {
        return std::tie(
                   lhs.device, lhs.device_overrides, lhs.precision,
                   lhs.operator_precisions, lhs.kernel_profile,
                   lhs.moe_layer_event_mode, lhs.moe_communication_backend,
                   lhs.mega_moe_tail_io_fraction, lhs.mega_moe_wave_exposure,
                   lhs.tensor_parallel_size, lhs.network_bandwidth_gbps,
                   lhs.network_latency_us, lhs.intra_node_bandwidth_gbps) ==
               std::tie(rhs.device, rhs.device_overrides, rhs.precision,
                        rhs.operator_precisions, rhs.kernel_profile,
                        rhs.moe_layer_event_mode, rhs.moe_communication_backend,
                        rhs.mega_moe_tail_io_fraction,
                        rhs.mega_moe_wave_exposure,
                        rhs.tensor_parallel_size, rhs.network_bandwidth_gbps,
                        rhs.network_latency_us, rhs.intra_node_bandwidth_gbps);
    }

    [[nodiscard]] const std::string &attention_precision() const noexcept {
        return operator_precisions.attention.empty()
                   ? precision
                   : operator_precisions.attention;
    }
    [[nodiscard]] const std::string &dense_precision() const noexcept {
        return operator_precisions.dense.empty() ? precision
                                                 : operator_precisions.dense;
    }
    [[nodiscard]] const std::string &moe_expert_precision() const noexcept {
        return operator_precisions.moe_expert.empty()
                   ? precision
                   : operator_precisions.moe_expert;
    }
    [[nodiscard]] const std::string &moe_router_precision() const noexcept {
        return operator_precisions.moe_router.empty()
                   ? precision
                   : operator_precisions.moe_router;
    }
    [[nodiscard]] const std::string &kv_cache_precision() const noexcept {
        return operator_precisions.kv_cache.empty()
                   ? precision
                   : operator_precisions.kv_cache;
    }
    [[nodiscard]] const std::string &communication_precision() const noexcept {
        return operator_precisions.communication.empty()
                   ? precision
                   : operator_precisions.communication;
    }
    [[nodiscard]] const std::string &
    attention_weight_precision() const noexcept {
        return operator_precisions.attention_weight.empty()
                   ? attention_precision()
                   : operator_precisions.attention_weight;
    }
    [[nodiscard]] const std::string &
    attention_activation_precision() const noexcept {
        return operator_precisions.attention_activation.empty()
                   ? attention_precision()
                   : operator_precisions.attention_activation;
    }
    [[nodiscard]] const std::string &dense_weight_precision() const noexcept {
        return operator_precisions.dense_weight.empty()
                   ? dense_precision()
                   : operator_precisions.dense_weight;
    }
    [[nodiscard]] const std::string &
    dense_activation_precision() const noexcept {
        return operator_precisions.dense_activation.empty()
                   ? dense_precision()
                   : operator_precisions.dense_activation;
    }
    [[nodiscard]] const std::string &
    moe_expert_weight_precision() const noexcept {
        return operator_precisions.moe_expert_weight.empty()
                   ? moe_expert_precision()
                   : operator_precisions.moe_expert_weight;
    }
    [[nodiscard]] const std::string &
    moe_expert_activation_precision() const noexcept {
        return operator_precisions.moe_expert_activation.empty()
                   ? moe_expert_precision()
                   : operator_precisions.moe_expert_activation;
    }
    [[nodiscard]] const std::string &
    moe_router_weight_precision() const noexcept {
        // Keep the historical getter as an alias.  New configs should use
        // router_weight_storage_precision() so the storage dtype cannot be
        // confused with the FP32 router compute dtype.
        return router_weight_storage_precision();
    }
    [[nodiscard]] const std::string &
    router_weight_storage_precision() const noexcept {
        if (!operator_precisions.router_weight_storage.empty()) {
            return operator_precisions.router_weight_storage;
        }
        // Legacy configs used moe_router_weight for the router's resident
        // weight dtype.  Preserve that fallback when no canonical field is
        // present.
        return operator_precisions.moe_router_weight.empty()
                   ? moe_router_precision()
                   : operator_precisions.moe_router_weight;
    }
    [[nodiscard]] const std::string &
    moe_router_activation_precision() const noexcept {
        return operator_precisions.moe_router_activation.empty()
                   ? moe_router_precision()
                   : operator_precisions.moe_router_activation;
    }
    [[nodiscard]] const std::string &lm_head_precision() const noexcept {
        return operator_precisions.lm_head.empty()
                   ? dense_precision()
                   : operator_precisions.lm_head;
    }
    [[nodiscard]] const std::string &lm_head_weight_precision() const noexcept {
        return operator_precisions.lm_head_weight.empty()
                   ? lm_head_precision()
                   : operator_precisions.lm_head_weight;
    }
    [[nodiscard]] const std::string &
    lm_head_activation_precision() const noexcept {
        return operator_precisions.lm_head_activation.empty()
                   ? lm_head_precision()
                   : operator_precisions.lm_head_activation;
    }

    // K3-native precision families retain the legacy operator-specific
    // fields as fallbacks.  The older *_weight/*_activation overrides are
    // checked before their unsuffixed family so existing configs preserve
    // their most specific setting.
    [[nodiscard]] const std::string &
    routed_expert_weight_precision() const noexcept {
        return operator_precisions.routed_expert_weight.empty()
                   ? moe_expert_weight_precision()
                   : operator_precisions.routed_expert_weight;
    }
    [[nodiscard]] const std::string &
    routed_expert_activation_precision() const noexcept {
        return operator_precisions.routed_expert_activation.empty()
                   ? moe_expert_activation_precision()
                   : operator_precisions.routed_expert_activation;
    }
    [[nodiscard]] const std::string &
    latent_moe_projection_weight_precision() const noexcept {
        return operator_precisions.latent_moe_projection_weight.empty()
                   ? routed_expert_weight_precision()
                   : operator_precisions.latent_moe_projection_weight;
    }
    [[nodiscard]] const std::string &
    latent_moe_projection_activation_precision() const noexcept {
        return operator_precisions.latent_moe_projection_activation.empty()
                   ? routed_expert_activation_precision()
                   : operator_precisions.latent_moe_projection_activation;
    }
    [[nodiscard]] const std::string &
    shared_expert_weight_precision() const noexcept {
        return operator_precisions.shared_expert_weight.empty()
                   ? moe_expert_weight_precision()
                   : operator_precisions.shared_expert_weight;
    }
    [[nodiscard]] const std::string &
    shared_expert_activation_precision() const noexcept {
        return operator_precisions.shared_expert_activation.empty()
                   ? moe_expert_activation_precision()
                   : operator_precisions.shared_expert_activation;
    }
    [[nodiscard]] const std::string &
    dense_mlp_weight_precision() const noexcept {
        return operator_precisions.dense_mlp_weight.empty()
                   ? dense_weight_precision()
                   : operator_precisions.dense_mlp_weight;
    }
    [[nodiscard]] const std::string &
    dense_mlp_activation_precision() const noexcept {
        return operator_precisions.dense_mlp_activation.empty()
                   ? dense_activation_precision()
                   : operator_precisions.dense_mlp_activation;
    }
    [[nodiscard]] const std::string &router_compute_precision() const noexcept {
        if (!operator_precisions.router_compute.empty()) {
            return operator_precisions.router_compute;
        }
        // A legacy router weight override was historically also used as the
        // router compute dtype.  Keep that fallback for old configs while
        // allowing the canonical router_weight_storage field to be
        // independent.  K3 native defaults install an explicit FP32 value.
        return operator_precisions.moe_router_weight.empty()
                   ? moe_router_precision()
                   : operator_precisions.moe_router_weight;
    }
    [[nodiscard]] const std::string &kda_snapshot_precision() const noexcept {
        // This is an execution-config override only. A model-aware resolver
        // installs K3's native BF16 snapshot default before memory sizing.
        return operator_precisions.kda_snapshot.empty()
                   ? precision
                   : operator_precisions.kda_snapshot;
    }
};

// Fill model-native operator defaults without overwriting explicit modern or
// legacy precision overrides. Kimi K3 uses its published mixed-precision
// execution policy; other models are unchanged.
void apply_model_native_precision_defaults(
    AnalyticalExecutionModelConfig &execution, const ModelConfig &model);

struct ExecutionModelConfig {
    ExecutionModelType type = ExecutionModelType::kFixed;
    FixedExecutionModelConfig fixed;
    AnalyticalExecutionModelConfig analytical;

    friend bool operator==(const ExecutionModelConfig &lhs,
                           const ExecutionModelConfig &rhs) {
        return std::tie(lhs.type, lhs.fixed, lhs.analytical) ==
               std::tie(rhs.type, rhs.fixed, rhs.analytical);
    }
};

} // namespace frontier::config
