#include "frontier/config/config.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "frontier/attention/model_binding.h"
#include "frontier/core/runtime_paths.h"

namespace frontier::config {
namespace {

using Json = nlohmann::json;

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](char ch) {
        return static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    });
    return value;
}

std::uint64_t require_positive_u64(const Json &root, std::string_view key,
                                   std::string_view source) {
    if (!root.contains(key) || !root.at(key).is_number_integer() ||
        root.at(key).is_number_float()) {
        throw ConfigError(std::string{source} + " requires positive integer '" +
                          std::string{key} + "'");
    }
    try {
        if (root.at(key).is_number_unsigned()) {
            const std::uint64_t value = root.at(key).get<std::uint64_t>();
            if (value == 0) {
                throw ConfigError(std::string{source} + "." + std::string{key} +
                                  " must be positive");
            }
            return value;
        }
        const std::int64_t value = root.at(key).get<std::int64_t>();
        if (value <= 0) {
            throw ConfigError(std::string{source} + "." + std::string{key} +
                              " must be positive");
        }
        return static_cast<std::uint64_t>(value);
    } catch (const ConfigError &) {
        throw;
    } catch (const Json::exception &) {
        throw ConfigError(std::string{source} + "." + std::string{key} +
                          " is outside the supported integer range");
    }
}

std::uint64_t optional_positive_u64(const Json &root, std::string_view key,
                                    std::uint64_t fallback,
                                    std::string_view source) {
    return root.contains(key) && !root.at(key).is_null()
               ? require_positive_u64(root, key, source)
               : fallback;
}

std::uint64_t optional_nonnegative_u64(const Json &root, std::string_view key,
                                       std::uint64_t fallback,
                                       std::string_view source) {
    if (!root.contains(key) || root.at(key).is_null()) {
        return fallback;
    }
    if (!root.at(key).is_number_integer() || root.at(key).is_number_float()) {
        throw ConfigError(std::string{source} + "." + std::string{key} +
                          " must be a nonnegative integer");
    }
    try {
        if (root.at(key).is_number_unsigned()) {
            return root.at(key).get<std::uint64_t>();
        }
        const std::int64_t value = root.at(key).get<std::int64_t>();
        if (value < 0) {
            throw ConfigError(std::string{source} + "." + std::string{key} +
                              " must be nonnegative");
        }
        return static_cast<std::uint64_t>(value);
    } catch (const ConfigError &) {
        throw;
    } catch (const Json::exception &) {
        throw ConfigError(std::string{source} + "." + std::string{key} +
                          " is outside the supported integer range");
    }
}

bool optional_bool(const Json &root, std::string_view key, bool fallback,
                   std::string_view source) {
    if (!root.contains(key)) {
        return fallback;
    }
    if (!root.at(key).is_boolean()) {
        throw ConfigError(std::string{source} + "." + std::string{key} +
                          " must be boolean");
    }
    return root.at(key).get<bool>();
}

std::uint64_t first_positive_u64(const Json &root,
                                 const std::vector<std::string_view> &keys,
                                 std::uint64_t fallback,
                                 std::string_view source) {
    for (const std::string_view key : keys) {
        if (root.contains(key)) {
            return require_positive_u64(root, key, source);
        }
    }
    return fallback;
}

const Json *find_json_value(const Json &primary, const Json &fallback,
                            const std::vector<std::string_view> &keys) {
    for (const std::string_view key : keys) {
        if (primary.contains(key)) {
            return &primary.at(key);
        }
    }
    for (const std::string_view key : keys) {
        if (fallback.contains(key)) {
            return &fallback.at(key);
        }
    }
    return nullptr;
}

std::uint64_t first_positive_u64_from(const Json &primary, const Json &fallback,
                                      const std::vector<std::string_view> &keys,
                                      std::uint64_t fallback_value,
                                      std::string_view source) {
    const Json *value = find_json_value(primary, fallback, keys);
    if (value == nullptr) {
        return fallback_value;
    }
    // Keep the existing diagnostics (including integer range checks) by
    // parsing the selected value through a one-field object.
    Json selected = Json::object();
    selected[std::string{keys.front()}] = *value;
    return require_positive_u64(selected, keys.front(), source);
}

std::uint64_t
optional_nonnegative_u64_from(const Json &primary, const Json &fallback,
                              const std::vector<std::string_view> &keys,
                              std::uint64_t fallback_value,
                              std::string_view source) {
    const Json *value = find_json_value(primary, fallback, keys);
    if (value == nullptr || value->is_null()) {
        return fallback_value;
    }
    Json selected = Json::object();
    selected[std::string{keys.front()}] = *value;
    return optional_nonnegative_u64(selected, keys.front(), fallback_value,
                                    source);
}

std::vector<std::uint64_t> parse_layer_ids(const Json &value,
                                           std::string_view key,
                                           std::string_view source,
                                           std::uint64_t num_layers,
                                           bool one_based) {
    std::vector<std::uint64_t> raw;
    if (value.is_array()) {
        raw.reserve(value.size());
        for (const Json &entry : value) {
            if (!entry.is_number_integer() || entry.is_number_float()) {
                throw ConfigError(std::string{source} + "." + std::string{key} +
                                  " entries must be integers");
            }
            try {
                if (entry.is_number_unsigned()) {
                    raw.push_back(entry.get<std::uint64_t>());
                } else {
                    const std::int64_t parsed = entry.get<std::int64_t>();
                    if (parsed < 0) {
                        throw ConfigError(std::string{source} + "." +
                                          std::string{key} +
                                          " entries must be nonnegative");
                    }
                    raw.push_back(static_cast<std::uint64_t>(parsed));
                }
            } catch (const ConfigError &) {
                throw;
            } catch (const Json::exception &) {
                throw ConfigError(std::string{source} + "." + std::string{key} +
                                  " entry is outside the supported integer "
                                  "range");
            }
        }
    } else if (value.is_string()) {
        std::string text = value.get<std::string>();
        for (char &ch : text) {
            if (ch == ',' || ch == ';' || ch == '[' || ch == ']') {
                ch = ' ';
            }
        }
        std::istringstream stream{text};
        std::string token;
        while (stream >> token) {
            try {
                std::size_t consumed = 0;
                const long long parsed = std::stoll(token, &consumed);
                if (consumed != token.size()) {
                    throw ConfigError(std::string{source} + "." +
                                      std::string{key} +
                                      " entries must be integers");
                }
                if (parsed < 0) {
                    throw ConfigError(std::string{source} + "." +
                                      std::string{key} +
                                      " entries must be nonnegative");
                }
                raw.push_back(static_cast<std::uint64_t>(parsed));
            } catch (const ConfigError &) {
                throw;
            } catch (const std::exception &) {
                throw ConfigError(std::string{source} + "." + std::string{key} +
                                  " entries must be integers");
            }
        }
    } else {
        throw ConfigError(std::string{source} + "." + std::string{key} +
                          " must be an integer array or comma-separated "
                          "string");
    }

    std::vector<std::uint64_t> parsed;
    parsed.reserve(raw.size());
    std::set<std::uint64_t> seen;
    for (const std::uint64_t value_id : raw) {
        if (one_based) {
            if (value_id == 0 || value_id > num_layers) {
                throw ConfigError(std::string{source} + "." + std::string{key} +
                                  " contains a one-based layer outside "
                                  "[1, num_hidden_layers]");
            }
            const std::uint64_t zero_based = value_id - 1;
            if (!seen.insert(zero_based).second) {
                throw ConfigError(std::string{source} + "." + std::string{key} +
                                  " contains duplicate layer ids");
            }
            parsed.push_back(zero_based);
        } else {
            if (value_id >= num_layers) {
                throw ConfigError(std::string{source} + "." + std::string{key} +
                                  " contains a zero-based layer outside "
                                  "[0, num_hidden_layers)");
            }
            if (!seen.insert(value_id).second) {
                throw ConfigError(std::string{source} + "." + std::string{key} +
                                  " contains duplicate layer ids");
            }
            parsed.push_back(value_id);
        }
    }
    std::sort(parsed.begin(), parsed.end());
    return parsed;
}

std::vector<std::uint64_t>
complement_layer_ids(std::uint64_t num_layers,
                     const std::vector<std::uint64_t> &selected) {
    std::vector<std::uint64_t> complement;
    complement.reserve(num_layers - selected.size());
    std::size_t selected_index = 0;
    for (std::uint64_t layer = 0; layer < num_layers; ++layer) {
        if (selected_index < selected.size() &&
            selected[selected_index] == layer) {
            ++selected_index;
        } else {
            complement.push_back(layer);
        }
    }
    return complement;
}

void validate_layer_partition(const std::vector<std::uint64_t> &kda,
                              const std::vector<std::uint64_t> &mla,
                              std::uint64_t num_layers,
                              std::string_view source) {
    if (kda.size() + mla.size() != num_layers) {
        throw ConfigError(std::string{source} +
                          " hybrid attention layer metadata must partition "
                          "all decoder layers");
    }
    std::vector<std::uint64_t> combined;
    combined.reserve(num_layers);
    combined.insert(combined.end(), kda.begin(), kda.end());
    combined.insert(combined.end(), mla.begin(), mla.end());
    std::sort(combined.begin(), combined.end());
    for (std::uint64_t layer = 0; layer < num_layers; ++layer) {
        if (combined[layer] != layer) {
            throw ConfigError(std::string{source} +
                              " hybrid attention layer metadata contains "
                              "overlap or a missing layer");
        }
    }
}

std::string canonical_model_alias(std::string_view model_name) {
    std::string lower = lowercase(std::string{model_name});
    if (lower == "kimi-k3" || lower == "kimi_k3" ||
        lower == "kimi-k3-instruct" || lower == "kimi_k3_instruct" ||
        lower == "moonshotai/kimi-k3" ||
        lower == "moonshotai/kimi-k3-instruct") {
        return "moonshotai/Kimi-K3";
    }
    return std::string{model_name};
}

ModelConfig
registered_dense_model(std::string name, std::string model_type,
                       std::uint64_t num_layers, std::uint64_t num_query_heads,
                       std::uint64_t num_kv_heads, std::uint64_t hidden_size,
                       std::uint64_t intermediate_size, bool gated_mlp = true,
                       bool fused_add_norm = true) {
    ModelConfig model{};
    model.name = std::move(name);
    model.model_type = std::move(model_type);
    model.kind = ModelKind::kDense;
    model.num_layers = num_layers;
    model.hidden_size = hidden_size;
    model.intermediate_size = intermediate_size;
    model.dense_intermediate_size = intermediate_size;
    model.moe_intermediate_size = intermediate_size;
    model.num_query_heads = num_query_heads;
    model.num_kv_heads = num_kv_heads;
    if (hidden_size % num_query_heads != 0) {
        throw ConfigError("registered model hidden size must divide evenly "
                          "over query heads");
    }
    model.head_dim = hidden_size / num_query_heads;
    model.gated_mlp = gated_mlp;
    model.fused_add_norm = fused_add_norm;
    model.num_experts = 1;
    model.num_experts_per_token = 1;
    model.total_expert_num = 1;
    model.router_topk = 1;
    model.attention = attention::bind_attention_family(model);
    return model;
}

std::optional<ModelConfig> registered_model(std::string_view name) {
    if (name == "codellama/CodeLlama-34b-Instruct-hf") {
        return registered_dense_model(std::string{name}, "llama", 48, 64, 8,
                                      8'192, 22'016);
    }
    if (name == "meta-llama/Llama-2-7b-hf") {
        return registered_dense_model(std::string{name}, "llama", 32, 32, 32,
                                      4'096, 11'008);
    }
    if (name == "meta-llama/Llama-2-tiny") {
        return registered_dense_model(std::string{name}, "llama", 2, 2, 2, 2,
                                      2);
    }
    if (name == "meta-llama/Llama-2-70b-hf") {
        return registered_dense_model(std::string{name}, "llama", 80, 64, 8,
                                      8'192, 28'672);
    }
    if (name == "meta-llama/Meta-Llama-3-8B") {
        return registered_dense_model(std::string{name}, "llama", 32, 32, 8,
                                      4'096, 14'336);
    }
    if (name == "meta-llama/Meta-Llama-3-70B") {
        return registered_dense_model(std::string{name}, "llama", 80, 64, 8,
                                      8'192, 28'672);
    }
    if (name == "internlm/internlm-20b") {
        return registered_dense_model(std::string{name}, "internlm", 60, 40, 40,
                                      5'120, 13'824);
    }
    if (name == "internlm/internlm2-20b") {
        return registered_dense_model(std::string{name}, "internlm2", 48, 48, 8,
                                      6'144, 16'384);
    }
    if (name == "microsoft/phi-2") {
        return registered_dense_model(std::string{name}, "phi", 32, 32, 32,
                                      2'560, 10'240, false, false);
    }
    if (name == "Qwen/Qwen-72B") {
        return registered_dense_model(std::string{name}, "qwen", 80, 64, 64,
                                      8'192, 24'576);
    }
    if (name == "Qwen/Qwen3-4B") {
        return registered_dense_model(std::string{name}, "qwen", 36, 32, 8,
                                      2'560, 9'728);
    }
    if (name == "Qwen/Qwen3-32B") {
        return registered_dense_model(std::string{name}, "qwen", 64, 64, 8,
                                      5'120, 25'600);
    }
    return std::nullopt;
}

void parse_hybrid_attention_metadata(ModelConfig &model, const Json &config,
                                     const Json &fallback,
                                     std::string_view source) {
    const Json empty_object = Json::object();
    const Json *linear_config_value = find_json_value(
        config, fallback, {"linear_attn_config", "linear_attention_config"});
    const Json &linear_config =
        linear_config_value == nullptr ? empty_object : *linear_config_value;
    if (!linear_config.is_object()) {
        throw ConfigError(std::string{source} +
                          ".linear_attn_config must be an object");
    }

    model.kda_num_heads = first_positive_u64_from(
        linear_config, config,
        {"num_heads", "kda_num_heads", "linear_num_heads"}, 0, source);
    model.kda_num_k_heads = first_positive_u64_from(
        linear_config, config,
        {"num_k_heads", "kda_num_k_heads", "linear_num_k_heads"},
        model.kda_num_heads, source);
    model.kda_num_v_heads = first_positive_u64_from(
        linear_config, config,
        {"num_v_heads", "kda_num_v_heads", "linear_num_v_heads"},
        model.kda_num_heads, source);
    model.kda_head_dim = first_positive_u64_from(
        linear_config, config, {"head_dim", "kda_head_dim", "linear_head_dim"},
        0, source);
    model.kda_key_head_dim = first_positive_u64_from(
        linear_config, config,
        {"key_head_dim", "kda_key_head_dim", "linear_key_head_dim"},
        model.kda_head_dim, source);
    model.kda_value_head_dim = first_positive_u64_from(
        linear_config, config,
        {"value_head_dim", "kda_value_head_dim", "linear_value_head_dim"},
        model.kda_head_dim, source);
    model.kda_short_conv_kernel_size = first_positive_u64_from(
        linear_config, config,
        {"short_conv_kernel_size", "kda_short_conv_kernel_size",
         "linear_conv_kernel_size", "linear_conv_kernel_dim"},
        0, source);
    model.kda_conv_state_dim = first_positive_u64_from(
        linear_config, config,
        {"conv_state_dim", "kda_conv_state_dim", "linear_conv_dim"}, 0, source);
    if (model.kda_conv_state_dim == 0 && model.kda_num_k_heads != 0 &&
        model.kda_num_v_heads != 0 && model.kda_key_head_dim != 0 &&
        model.kda_value_head_dim != 0) {
        model.kda_conv_state_dim =
            2 * model.kda_num_k_heads * model.kda_key_head_dim +
            model.kda_num_v_heads * model.kda_value_head_dim;
    }

    const auto find_layer_metadata =
        [](const Json &primary, const Json &secondary,
           const std::vector<std::string_view> &keys)
        -> std::pair<const Json *, std::string_view> {
        for (const std::string_view key : keys) {
            if (primary.contains(key)) {
                return {&primary.at(key), key};
            }
        }
        for (const std::string_view key : keys) {
            if (secondary.contains(key)) {
                return {&secondary.at(key), key};
            }
        }
        return {nullptr, {}};
    };

    auto [kda_layers_value, kda_layers_key] =
        find_layer_metadata(linear_config, config,
                            {"kda_layers", "linear_layers", "delta_layers",
                             "kda_layer_ids", "kda_layer_indices"});
    auto [mla_layers_value, mla_layers_key] = find_layer_metadata(
        linear_config, config,
        {"full_attn_layers", "full_attention_layers", "mla_layers",
         "mla_layer_ids", "mla_layer_indices"});
    const bool kda_layers_present = kda_layers_value != nullptr;
    const bool mla_layers_present = mla_layers_value != nullptr;

    std::vector<std::uint64_t> kda_layers;
    std::vector<std::uint64_t> mla_layers;
    const auto is_zero_based_key = [](std::string_view key) {
        return key.find("indices") != std::string_view::npos ||
               key.find("_ids") != std::string_view::npos;
    };
    if (kda_layers_value != nullptr) {
        kda_layers = parse_layer_ids(*kda_layers_value, kda_layers_key, source,
                                     model.num_layers,
                                     !is_zero_based_key(kda_layers_key));
    }
    if (mla_layers_value != nullptr) {
        mla_layers = parse_layer_ids(*mla_layers_value, mla_layers_key, source,
                                     model.num_layers,
                                     !is_zero_based_key(mla_layers_key));
    }

    // A few Kimi/Kimi-Linear exports use an ordered layer_types array instead
    // of the two explicit one-based lists.  Accept the common aliases while
    // retaining a strict exact partition contract.
    const auto [layer_types_value, layer_types_key] =
        find_layer_metadata(config, fallback,
                            {"layer_types", "attention_layer_types",
                             "layer_attention_types", "attention_types"});
    const bool layer_types_present = layer_types_value != nullptr;
    if (!kda_layers_present && !mla_layers_present &&
        layer_types_value != nullptr) {
        if (!layer_types_value->is_array() ||
            layer_types_value->size() != model.num_layers) {
            throw ConfigError(std::string{source} + "." +
                              std::string{layer_types_key} +
                              " must contain exactly num_hidden_layers "
                              "entries");
        }
        for (std::uint64_t layer = 0; layer < model.num_layers; ++layer) {
            const Json &entry = layer_types_value->at(layer);
            if (!entry.is_string()) {
                throw ConfigError(std::string{source} + "." +
                                  std::string{layer_types_key} +
                                  " entries must be strings");
            }
            std::string type = lowercase(entry.get<std::string>());
            std::replace(type.begin(), type.end(), '-', '_');
            if (type == "kda" || type == "kda_attention" || type == "linear" ||
                type == "linear_attention" || type == "delta_attention" ||
                type == "kimi_delta_attention") {
                kda_layers.push_back(layer);
            } else if (type == "mla" || type == "gated_mla" || type == "full" ||
                       type == "full_attention" || type == "self_attention") {
                mla_layers.push_back(layer);
            } else {
                throw ConfigError(
                    std::string{source} + "." + std::string{layer_types_key} +
                    " contains unsupported layer type '" + type + "'");
            }
        }
    }

    // Infer only a genuinely absent counterpart. An explicitly empty list is
    // authoritative and must participate in exact partition validation.
    if (kda_layers_present && !mla_layers_present) {
        mla_layers = complement_layer_ids(model.num_layers, kda_layers);
    } else if (!kda_layers_present && mla_layers_present) {
        kda_layers = complement_layer_ids(model.num_layers, mla_layers);
    }

    const Json *declared_kda_count_value = find_json_value(
        config, fallback, {"num_kda_layers", "kda_layer_count"});
    const Json *declared_mla_count_value = find_json_value(
        config, fallback, {"num_mla_layers", "mla_layer_count"});
    const bool declared_kda_count_present =
        declared_kda_count_value != nullptr &&
        !declared_kda_count_value->is_null();
    const bool declared_mla_count_present =
        declared_mla_count_value != nullptr &&
        !declared_mla_count_value->is_null();
    const std::uint64_t declared_kda_count = optional_nonnegative_u64_from(
        config, fallback, {"num_kda_layers", "kda_layer_count"}, 0, source);
    const std::uint64_t declared_mla_count = optional_nonnegative_u64_from(
        config, fallback, {"num_mla_layers", "mla_layer_count"}, 0, source);

    const bool per_layer_metadata_present =
        kda_layers_present || mla_layers_present || layer_types_present;
    if (per_layer_metadata_present) {
        validate_layer_partition(kda_layers, mla_layers, model.num_layers,
                                 source);
        if (declared_kda_count_present &&
            declared_kda_count != kda_layers.size()) {
            throw ConfigError(std::string{source} +
                              ".num_kda_layers disagrees with layer metadata");
        }
        if (declared_mla_count_present &&
            declared_mla_count != mla_layers.size()) {
            throw ConfigError(std::string{source} +
                              ".num_mla_layers disagrees with layer metadata");
        }
        model.kda_layer_indices = std::move(kda_layers);
        model.mla_layer_indices = std::move(mla_layers);
        model.num_kda_layers = model.kda_layer_indices.size();
        model.num_mla_layers = model.mla_layer_indices.size();
    } else if (declared_kda_count_present || declared_mla_count_present) {
        throw ConfigError(std::string{source} +
                          " declares hybrid layer counts without per-layer "
                          "metadata");
    } else if (model.use_mla) {
        // Legacy flat MLA assets (DeepSeek/Kimi K2) have no hybrid list and
        // historically apply latent MLA semantics to every decoder layer.
        model.num_mla_layers = model.num_layers;
    }

    if (model.num_kda_layers != 0) {
        if (model.kda_num_heads == 0) {
            model.kda_num_heads = model.num_query_heads;
        }
        if (model.kda_num_k_heads == 0) {
            model.kda_num_k_heads = model.kda_num_heads;
        }
        if (model.kda_num_v_heads == 0) {
            model.kda_num_v_heads = model.kda_num_heads;
        }
        if (model.kda_head_dim == 0) {
            model.kda_head_dim = model.head_dim;
        }
        if (model.kda_key_head_dim == 0) {
            model.kda_key_head_dim = model.kda_head_dim;
        }
        if (model.kda_value_head_dim == 0) {
            model.kda_value_head_dim = model.kda_head_dim;
        }
        if (model.kda_conv_state_dim == 0) {
            model.kda_conv_state_dim =
                2 * model.kda_num_k_heads * model.kda_key_head_dim +
                model.kda_num_v_heads * model.kda_value_head_dim;
        }
        if (!model.kda_topology_is_symmetric()) {
            throw ConfigError(
                std::string{source} +
                " currently supports only symmetric KDA Q/K/V topology: "
                "num_heads, num_k_heads, and num_v_heads must match, and "
                "head_dim, key_head_dim, and value_head_dim must match");
        }
    }
}

bool json_truthy(const Json &value) {
    if (value.is_null()) {
        return false;
    }
    if (value.is_boolean()) {
        return value.get<bool>();
    }
    if (value.is_number()) {
        return value.get<double>() != 0.0;
    }
    if (value.is_string()) {
        return !value.get_ref<const std::string &>().empty();
    }
    return !value.empty();
}

std::string sanitize_model_name(std::string_view model_name) {
    std::string sanitized;
    sanitized.reserve(model_name.size() * 2);
    for (const char ch : model_name) {
        if (ch == '/') {
            sanitized += "__";
        } else {
            sanitized.push_back(ch);
        }
    }
    return sanitized;
}

std::vector<std::filesystem::path> model_directories() {
    std::vector<std::filesystem::path> directories;
#ifdef _WIN32
    char *configured = nullptr;
    std::size_t configured_size = 0;
    if (_dupenv_s(&configured, &configured_size, "FRONTIER_MODEL_CONFIG_DIR") ==
            0 &&
        configured != nullptr && *configured != '\0') {
        directories.emplace_back(configured);
    }
    std::free(configured);
#else
    if (const char *configured = std::getenv("FRONTIER_MODEL_CONFIG_DIR");
        configured != nullptr && *configured != '\0') {
        directories.emplace_back(configured);
    }
#endif
    if (const auto executable_dir = core::executable_directory();
        executable_dir.has_value()) {
        // Installed layouts place the binary in <prefix>/bin and model assets
        // in <prefix>/share/frontier/models.  The data/config candidate also
        // supports relocatable source-tree bundles without relying on cwd.
        directories.push_back(*executable_dir / ".." / "share" / "frontier" /
                              "models");
        directories.push_back(*executable_dir / "share" / "frontier" /
                              "models");
        directories.push_back(*executable_dir / ".." / "data" / "config" /
                              "models");
    }
    directories.emplace_back("data/config/models");
    directories.emplace_back("../data/config/models");
    directories.emplace_back("../../data/config/models");
    return directories;
}

std::filesystem::path find_model_file(std::string_view model_name) {
    const std::string filename = sanitize_model_name(model_name) + ".json";
    std::vector<std::filesystem::path> searched;
    for (const std::filesystem::path &directory : model_directories()) {
        const std::filesystem::path candidate = directory / filename;
        searched.push_back(candidate);
        if (std::filesystem::is_regular_file(candidate)) {
            return candidate;
        }
    }
    std::ostringstream message;
    message << "unknown model '" << model_name
            << "'; no registered model or config asset found (searched";
    for (const auto &path : searched) {
        message << " " << path.string();
    }
    message << ")";
    throw ConfigError(message.str());
}

Json read_model_json(const std::filesystem::path &path) {
    std::ifstream input{path, std::ios::binary};
    if (!input) {
        throw ConfigError("failed to open model config: " + path.string());
    }
    try {
        return Json::parse(input);
    } catch (const Json::exception &error) {
        throw ConfigError("failed to parse model config " + path.string() +
                          ": " + error.what());
    }
}

bool inferred_fused_add_norm(const Json &root, const std::string &model_type) {
    if (root.contains("fused_add_norm_capability")) {
        return optional_bool(root, "fused_add_norm_capability", false,
                             "model config");
    }
    if (model_type == "phimoe") {
        return false;
    }
    if (root.contains("rms_norm_eps")) {
        return true;
    }
    return model_type == "llama" || model_type == "mixtral" ||
           model_type == "qwen2" || model_type == "qwen2_moe" ||
           model_type == "qwen3_moe" || model_type == "deepseek_v3" ||
           model_type == "kimi_k3" || model_type == "kimi_linear";
}

ModelConfig parse_model_asset(std::string_view model_name, const Json &root,
                              const std::filesystem::path &path) {
    if (!root.is_object()) {
        throw ConfigError("model config must be a JSON object: " +
                          path.string());
    }
    const std::string source = "model config " + path.string();
    // Hugging Face's multimodal Kimi K3 config keeps decoder settings in a
    // nested text_config object. Merge that object over top-level values so
    // nested and legacy flat assets share one parser. Keep the outer
    // model_type authoritative when both levels provide it (K3's nested
    // text model reports kimi_linear).
    Json config = root;
    if (root.contains("text_config") && !root.at("text_config").is_null()) {
        if (!root.at("text_config").is_object()) {
            throw ConfigError(source + ".text_config must be an object");
        }
        for (const auto &[key, value] : root.at("text_config").items()) {
            if (key == "model_type" && root.contains("model_type") &&
                root.at("model_type").is_string()) {
                continue;
            }
            config[key] = value;
        }
    }
    ModelConfig model{};
    model.name = std::string{model_name};
    model.model_type = lowercase(
        config.contains("model_type") && config.at("model_type").is_string()
            ? config.at("model_type").get<std::string>()
            : std::string{});
    model.num_layers =
        require_positive_u64(config, "num_hidden_layers", source);
    model.hidden_size = require_positive_u64(config, "hidden_size", source);
    model.num_query_heads = first_positive_u64(
        config, {"num_attention_heads", "num_query_heads"}, 0, source);
    if (model.num_query_heads == 0) {
        throw ConfigError(source +
                          " requires positive integer 'num_attention_heads'");
    }
    model.num_kv_heads = first_positive_u64(
        config, {"num_key_value_heads", "num_kv_heads"}, 0, source);
    if (model.num_kv_heads == 0) {
        // Kimi Linear's Python config defaults num_key_value_heads to the
        // query-head count when a flat export omits the field.
        model.num_kv_heads = model.num_query_heads;
    }
    model.num_experts =
        first_positive_u64(config,
                           {"num_experts", "num_local_experts",
                            "n_routed_experts", "num_routed_experts"},
                           1, source);
    model.kind = model.num_experts > 1 ? ModelKind::kMoe : ModelKind::kDense;
    model.intermediate_size =
        model.is_moe()
            ? first_positive_u64(config,
                                 {"moe_intermediate_size", "intermediate_size"},
                                 0, source)
            : require_positive_u64(config, "intermediate_size", source);
    if (model.intermediate_size == 0) {
        throw ConfigError(source +
                          " requires moe_intermediate_size for an MoE model");
    }
    model.dense_intermediate_size = first_positive_u64(
        config, {"intermediate_size"}, model.intermediate_size, source);
    model.moe_intermediate_size = model.is_moe()
                                      ? model.intermediate_size
                                      : model.dense_intermediate_size;
    model.routed_expert_hidden_size =
        optional_positive_u64(config, "routed_expert_hidden_size", 0, source);
    model.latent_moe_use_norm =
        optional_bool(config, "latent_moe_use_norm", false, source);
    model.attn_res_block_size =
        optional_positive_u64(config, "attn_res_block_size", 0, source);
    if (config.contains("head_dim")) {
        model.head_dim = require_positive_u64(config, "head_dim", source);
    } else {
        // Kimi K3's nested linear_attn_config carries the 128-wide KDA head
        // dimension even though hidden_size/num_attention_heads is not an
        // integer.  Prefer that explicit hybrid dimension before applying
        // the legacy dense-head divisibility check.
        const Json *linear_config = find_json_value(
            config, root, {"linear_attn_config", "linear_attention_config"});
        if (linear_config != nullptr && linear_config->is_object() &&
            linear_config->contains("head_dim")) {
            model.head_dim =
                require_positive_u64(*linear_config, "head_dim", source);
        } else {
            if (model.hidden_size % model.num_query_heads != 0) {
                throw ConfigError(source +
                                  " hidden_size must divide evenly over query "
                                  "heads when head_dim is omitted");
            }
            model.head_dim = model.hidden_size / model.num_query_heads;
        }
    }
    const std::string hidden_act = lowercase(
        config.contains("hidden_act") && config.at("hidden_act").is_string()
            ? config.at("hidden_act").get<std::string>()
            : std::string{});
    model.gated_mlp =
        hidden_act == "silu" || hidden_act == "swish" || hidden_act == "situ";
    model.fused_add_norm = inferred_fused_add_norm(config, model.model_type);
    model.num_experts_per_token =
        model.is_moe()
            ? first_positive_u64(
                  config, {"num_experts_per_tok", "num_experts_per_token"}, 0,
                  source)
            : 1;
    if (model.is_moe() && model.num_experts_per_token == 0) {
        throw ConfigError(source +
                          " requires positive integer 'num_experts_per_tok'");
    }
    model.total_expert_num = model.num_experts;
    model.router_topk = model.num_experts_per_token;
    model.num_shared_experts =
        model.is_moe()
            ? optional_nonnegative_u64_from(
                  config, root, {"n_shared_experts", "num_shared_experts"}, 0,
                  source)
            : 0;
    model.first_k_dense_replace =
        model.is_moe() ? optional_nonnegative_u64(
                             config, "first_k_dense_replace", 0, source)
                       : model.num_layers;
    model.moe_layer_freq =
        model.is_moe()
            ? optional_positive_u64(config, "moe_layer_freq", 1, source)
            : 1;
    model.vocab_size =
        optional_positive_u64(config, "vocab_size", model.vocab_size, source);
    if (model.first_k_dense_replace > model.num_layers) {
        throw ConfigError(source +
                          ".first_k_dense_replace must not exceed num_layers");
    }
    model.use_mla =
        optional_bool(config, "use_mla", false, source) ||
        ((model.model_type == "deepseek_v2" ||
          model.model_type == "deepseek_v3" ||
          model.model_type == "deepseek_mtp" || model.model_type == "kimi_k2" ||
          model.model_type == "kimi_k3" || model.model_type == "kimi_linear") &&
         config.contains("kv_lora_rank"));
    // Kimi K3 publishes these MLA behavior switches alongside the nested
    // decoder fields. Keep them explicit so analytical execution can model
    // gated MLA and the no-PE path without inferring from model type. Older
    // MLA assets do not carry the fields and retain the false defaults.
    model.mla_use_output_gate =
        optional_bool(config, "mla_use_output_gate", false, source);
    model.mla_use_nope = optional_bool(config, "mla_use_nope", false, source);
    model.use_mfa = optional_bool(config, "use_mfa", false, source);
    model.q_lora_rank = optional_positive_u64(config, "q_lora_rank", 0, source);
    model.kv_lora_rank =
        optional_positive_u64(config, "kv_lora_rank", 0, source);
    model.qk_nope_head_dim =
        optional_positive_u64(config, "qk_nope_head_dim", 0, source);
    model.qk_rope_head_dim =
        optional_positive_u64(config, "qk_rope_head_dim", 0, source);
    model.qk_head_dim = optional_positive_u64(config, "qk_head_dim", 0, source);
    if (model.qk_head_dim == 0 && model.qk_nope_head_dim > 0 &&
        model.qk_rope_head_dim > 0) {
        model.qk_head_dim = model.qk_nope_head_dim + model.qk_rope_head_dim;
    }
    if (model.use_mla &&
        model.qk_head_dim != model.qk_nope_head_dim + model.qk_rope_head_dim) {
        throw ConfigError(source + " qk_head_dim must equal qk_nope_head_dim + "
                                   "qk_rope_head_dim for MLA");
    }
    model.v_head_dim = optional_positive_u64(config, "v_head_dim", 0, source);
    model.share_q_dim = optional_positive_u64(config, "share_q_dim", 0, source);
    parse_hybrid_attention_metadata(model, config, root, source);
    for (const std::string_view marker :
         {"dsa_topk", "dsa_top_k", "dsa_index_topk", "dsa_indexer"}) {
        if (config.contains(marker) && json_truthy(config.at(marker))) {
            model.has_dsa_marker = true;
        }
    }
    for (const std::string_view marker :
         {"sliding_window_pattern", "dual_chunk_attention",
          "attention_chunk_size"}) {
        if (config.contains(marker) && json_truthy(config.at(marker))) {
            model.exotic_attention_fields.emplace_back(marker);
        }
    }
    try {
        model.attention = attention::bind_attention_family(model);
    } catch (const std::invalid_argument &error) {
        throw ConfigError(source + ": " + error.what());
    }
    return model;
}

} // namespace

ModelConfig load_model_config(std::string_view model_name) {
    if (model_name.empty()) {
        throw ConfigError("model_name must not be empty");
    }
    if (auto registered = registered_model(model_name);
        registered.has_value()) {
        return std::move(*registered);
    }
    const std::string canonical_name = canonical_model_alias(model_name);
    const std::filesystem::path path = find_model_file(canonical_name);
    return parse_model_asset(model_name, read_model_json(path), path);
}

} // namespace frontier::config
