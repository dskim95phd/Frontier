#include <cmath>
#include <initializer_list>
#include <limits>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "frontier/config/config_parse_internal.h"
#include "frontier/core/precision.h"

namespace frontier::config::parse_detail {

std::vector<double> require_finite_number_array(const Json &object,
                                                std::string_view field,
                                                std::string_view context) {
    const Json &value = object.at(field);
    if (!value.is_array()) {
        throw ConfigError(std::string{context} + "." + std::string{field} +
                          " must be an array");
    }
    std::vector<double> result;
    result.reserve(value.size());
    for (std::size_t index = 0; index < value.size(); ++index) {
        const Json &element = value[index];
        if (!element.is_number()) {
            throw ConfigError(std::string{context} + "." + std::string{field} +
                              "[" + std::to_string(index) +
                              "] must be numeric");
        }
        double parsed = 0.0;
        try {
            parsed = element.get<double>();
        } catch (const Json::exception &) {
            throw ConfigError(std::string{context} + "." + std::string{field} +
                              "[" + std::to_string(index) +
                              "] is outside the supported numeric range");
        }
        if (!std::isfinite(parsed) || parsed < 0.0) {
            throw ConfigError(std::string{context} + "." + std::string{field} +
                              "[" + std::to_string(index) +
                              "] must be finite and nonnegative");
        }
        result.push_back(parsed);
    }
    return result;
}

bool is_supported_analytical_precision(std::string_view precision) noexcept {
    return parse_precision(precision).has_value();
}

double analytical_precision_size_bytes(std::string_view precision) {
    const std::optional<Precision> parsed = parse_precision(precision);
    if (parsed.has_value()) {
        return storage_bytes_per_element(*parsed);
    }
    throw ConfigError("unsupported analytical precision: " +
                      std::string{precision});
}

AnalyticalDeviceOverrides
parse_analytical_device_overrides(const Json &execution) {
    AnalyticalDeviceOverrides result{};
    if (!execution.contains("device_overrides")) {
        return result;
    }
    const Json &overrides = execution.at("device_overrides");
    constexpr std::string_view context =
        "config.execution_model.device_overrides";
    require_keys(overrides, {},
                 {"hbm_bandwidth_tbps", "fp32_tflops", "fp16_tflops",
                  "fp8_tflops", "fp4_tflops"},
                 context);
    const auto parse_optional = [&](std::string_view field,
                                    std::optional<double> &destination) {
        if (!overrides.contains(field)) {
            return;
        }
        const double value = require_finite_number(overrides, field, context);
        if (value <= 0.0) {
            throw ConfigError(std::string{context} + "." + std::string{field} +
                              " must be positive");
        }
        destination = value;
    };
    parse_optional("hbm_bandwidth_tbps", result.hbm_bandwidth_tbps);
    parse_optional("fp32_tflops", result.fp32_tflops);
    parse_optional("fp16_tflops", result.fp16_tflops);
    parse_optional("fp8_tflops", result.fp8_tflops);
    parse_optional("fp4_tflops", result.fp4_tflops);
    return result;
}

OperatorPrecisionConfig parse_operator_precisions(const Json &execution) {
    OperatorPrecisionConfig result{};
    if (!execution.contains("operator_precisions")) {
        return result;
    }
    const Json &operators = execution.at("operator_precisions");
    constexpr std::string_view context =
        "config.execution_model.operator_precisions";
    require_keys(operators, {},
                 {"attention",
                  "attention_core",
                  "dense",
                  "moe_expert",
                  "moe_router",
                  "kv_cache",
                  "communication",
                  "attention_weight",
                  "attention_activation",
                  "dense_weight",
                  "dense_activation",
                  "moe_expert_weight",
                  "moe_expert_activation",
                  "moe_router_weight",
                  "moe_router_activation",
                  "router_weight_storage",
                  "lm_head",
                  "lm_head_weight",
                  "lm_head_activation",
                  "routed_expert_weight",
                  "routed_expert_activation",
                  "latent_moe_projection_weight",
                  "latent_moe_projection_activation",
                  "shared_expert_weight",
                  "shared_expert_activation",
                  "dense_mlp_weight",
                  "dense_mlp_activation",
                  "router_compute",
                  "kda_snapshot"},
                 context);
    const auto parse_optional = [&](std::string_view field,
                                    std::string &destination) {
        if (!operators.contains(field)) {
            return;
        }
        destination = require_string(operators, field, context);
        if (!is_supported_analytical_precision(destination)) {
            throw ConfigError(std::string{context} + "." + std::string{field} +
                              " must be one of fp32, fp16, bf16, fp8, int8, "
                              "fp4, int4, mxfp8, or mxfp4");
        }
    };
    parse_optional("attention", result.attention);
    parse_optional("attention_core", result.attention_core);
    parse_optional("dense", result.dense);
    parse_optional("moe_expert", result.moe_expert);
    parse_optional("moe_router", result.moe_router);
    parse_optional("kv_cache", result.kv_cache);
    parse_optional("communication", result.communication);
    parse_optional("attention_weight", result.attention_weight);
    parse_optional("attention_activation", result.attention_activation);
    parse_optional("dense_weight", result.dense_weight);
    parse_optional("dense_activation", result.dense_activation);
    parse_optional("moe_expert_weight", result.moe_expert_weight);
    parse_optional("moe_expert_activation", result.moe_expert_activation);
    parse_optional("moe_router_weight", result.moe_router_weight);
    parse_optional("moe_router_activation", result.moe_router_activation);
    parse_optional("router_weight_storage", result.router_weight_storage);
    parse_optional("lm_head", result.lm_head);
    parse_optional("lm_head_weight", result.lm_head_weight);
    parse_optional("lm_head_activation", result.lm_head_activation);
    parse_optional("routed_expert_weight", result.routed_expert_weight);
    parse_optional("routed_expert_activation", result.routed_expert_activation);
    parse_optional("latent_moe_projection_weight",
                   result.latent_moe_projection_weight);
    parse_optional("latent_moe_projection_activation",
                   result.latent_moe_projection_activation);
    parse_optional("shared_expert_weight", result.shared_expert_weight);
    parse_optional("shared_expert_activation", result.shared_expert_activation);
    parse_optional("dense_mlp_weight", result.dense_mlp_weight);
    parse_optional("dense_mlp_activation", result.dense_mlp_activation);
    parse_optional("router_compute", result.router_compute);
    parse_optional("kda_snapshot", result.kda_snapshot);
    return result;
}

ExecutionModelConfig parse_execution_model(const Json &root,
                                           const ParallelismConfig &parallelism,
                                           const ModelConfig &model) {
    const Json &execution = root.at("execution_model");
    require_object(execution, "config.execution_model");
    if (!execution.contains("type")) {
        throw ConfigError(
            "config.execution_model is missing required field 'type'");
    }
    const std::string type =
        require_string(execution, "type", "config.execution_model");

    if (type == "fixed") {
        require_exact_keys(execution, {"type", "stage_latencies_ms"},
                           "config.execution_model");
        std::vector<double> stage_latencies = require_finite_number_array(
            execution, "stage_latencies_ms", "config.execution_model");
        if (stage_latencies.size() != parallelism.pipeline_parallel_size) {
            throw ConfigError(
                "config.execution_model.stage_latencies_ms length must equal "
                "config.parallelism.pipeline_parallel_size");
        }
        return [&]() {
            ExecutionModelConfig value{};
            value.type = ExecutionModelType::kFixed;
            value.fixed = [&]() {
                FixedExecutionModelConfig value{};
                value.batch_latency_ms = stage_latencies.front();
                value.stage_latencies_ms = std::move(stage_latencies);
                return value;
            }();
            value.analytical = AnalyticalExecutionModelConfig{};
            return value;
        }();
    }

    if (type == "analytical") {
        require_keys(execution,
                     {
                         "type",
                         "device",
                         "precision",
                         "network_bandwidth_gbps",
                         "network_latency_us",
                         "intra_node_bandwidth_gbps",
                     },
                     {"operator_precisions", "device_overrides",
                      "kernel_profile", "moe_layer_event_mode",
                      "moe_communication_backend", "mega_moe_tail_io_fraction",
                      "mega_moe_wave_exposure"},
                     "config.execution_model");
        AnalyticalExecutionModelConfig analytical = [&]() {
            AnalyticalExecutionModelConfig value{};
            value.device =
                require_string(execution, "device", "config.execution_model");
            value.device_overrides =
                parse_analytical_device_overrides(execution);
            value.precision = require_string(execution, "precision",
                                             "config.execution_model");
            value.operator_precisions = parse_operator_precisions(execution);
            if (execution.contains("kernel_profile")) {
                value.kernel_profile = require_string(
                    execution, "kernel_profile", "config.execution_model");
            }
            if (execution.contains("moe_layer_event_mode")) {
                value.moe_layer_event_mode =
                    require_string(execution, "moe_layer_event_mode",
                                   "config.execution_model");
            }
            if (execution.contains("moe_communication_backend")) {
                value.moe_communication_backend =
                    require_string(execution, "moe_communication_backend",
                                   "config.execution_model");
            }
            if (execution.contains("mega_moe_tail_io_fraction")) {
                value.mega_moe_tail_io_fraction = require_finite_number(
                    execution, "mega_moe_tail_io_fraction",
                    "config.execution_model");
            }
            if (execution.contains("mega_moe_wave_exposure")) {
                value.mega_moe_wave_exposure =
                    require_finite_number(execution, "mega_moe_wave_exposure",
                                          "config.execution_model");
            }
            value.tensor_parallel_size = parallelism.tensor_parallel_size;
            value.network_bandwidth_gbps = require_finite_number(
                execution, "network_bandwidth_gbps", "config.execution_model");
            value.network_latency_us = require_finite_number(
                execution, "network_latency_us", "config.execution_model");
            value.intra_node_bandwidth_gbps =
                require_finite_number(execution, "intra_node_bandwidth_gbps",
                                      "config.execution_model");
            return value;
        }();
        if (analytical.device != "rubin" && analytical.device != "gb300" &&
            analytical.device != "custom") {
            throw ConfigError(
                "config.execution_model.device must be 'rubin', 'gb300', or "
                "'custom'");
        }
        if (analytical.device == "custom" &&
            !analytical.device_overrides.complete()) {
            throw ConfigError(
                "custom analytical device requires all device_overrides "
                "fields");
        }
        if (!is_supported_analytical_precision(analytical.precision)) {
            throw ConfigError(
                "analytical execution precision must be fp32, fp16, bf16, "
                "fp8, int8, fp4, int4, mxfp8, or mxfp4");
        }
        if (analytical.moe_layer_event_mode != "detailed" &&
            analytical.moe_layer_event_mode != "first_layer_scaled" &&
            analytical.moe_layer_event_mode != "stage_group_scaled") {
            throw ConfigError(
                "config.execution_model.moe_layer_event_mode must be "
                "'detailed', 'first_layer_scaled', or 'stage_group_scaled'");
        }
        if (analytical.kernel_profile != "generic" &&
            analytical.kernel_profile != "k3_flashkda_prefill" &&
            analytical.kernel_profile != "k3_sglang_mxfp4" &&
            analytical.kernel_profile != "k3_deepgemm_megamoe") {
            throw ConfigError(
                "config.execution_model.kernel_profile must be 'generic', "
                "'k3_flashkda_prefill', 'k3_sglang_mxfp4', or "
                "'k3_deepgemm_megamoe'");
        }
        if (analytical.kernel_profile != "generic" &&
            ((analytical.device != "gb300" && analytical.device != "rubin") ||
             model.model_type != "kimi_k3")) {
            throw ConfigError(
                "K3 analytical kernel profiles require model Kimi-K3 and "
                "execution device='gb300' or 'rubin'");
        }
        const bool has_mega_moe_override =
            analytical.mega_moe_tail_io_fraction.has_value() ||
            analytical.mega_moe_wave_exposure.has_value();
        if (has_mega_moe_override &&
            analytical.kernel_profile != "k3_deepgemm_megamoe") {
            throw ConfigError("MegaMoE timing overrides require "
                              "kernel_profile='k3_deepgemm_megamoe'");
        }
        const auto valid_fraction = [](const std::optional<double> &value) {
            return !value.has_value() || (*value >= 0.0 && *value <= 1.0);
        };
        if (!valid_fraction(analytical.mega_moe_tail_io_fraction) ||
            !valid_fraction(analytical.mega_moe_wave_exposure)) {
            throw ConfigError(
                "MegaMoE tail IO/wave overrides must be in [0, 1]");
        }
        if (analytical.moe_communication_backend != "generic" &&
            analytical.moe_communication_backend != "sm100_megamoe_public") {
            throw ConfigError(
                "config.execution_model.moe_communication_backend must be "
                "'generic' or 'sm100_megamoe_public'");
        }
        if (analytical.moe_communication_backend == "sm100_megamoe_public" &&
            analytical.device != "gb300" && analytical.device != "rubin") {
            throw ConfigError(
                "sm100_megamoe_public requires execution device='gb300' or "
                "'rubin'");
        }
        if (!model.attention.execution_enabled ||
            model.attention.memory_layout ==
                attention::AttentionMemoryLayout::kFrozenDsa) {
            throw ConfigError(
                "analytical execution does not support frozen DSA attention");
        }
        if (analytical.network_bandwidth_gbps <= 0.0 ||
            analytical.intra_node_bandwidth_gbps <= 0.0 ||
            analytical.network_latency_us < 0.0) {
            throw ConfigError(
                "analytical bandwidths must be positive and latency must be "
                "nonnegative");
        }
        return [&]() {
            ExecutionModelConfig value{};
            value.type = ExecutionModelType::kAnalytical;
            value.fixed = FixedExecutionModelConfig{};
            value.analytical = std::move(analytical);
            return value;
        }();
    }

    throw ConfigError(
        "config.execution_model.type must be 'fixed' or 'analytical', got '" +
        type + "'");
}

} // namespace frontier::config::parse_detail
