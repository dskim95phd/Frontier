#include "frontier/config/config.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "frontier/attention/model_binding.h"

namespace frontier::config {
namespace {

using Json = nlohmann::json;

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](char ch) {
        return static_cast<char>(
            std::tolower(static_cast<unsigned char>(ch)));
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
                throw ConfigError(std::string{source} + "." +
                                  std::string{key} + " must be positive");
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

ModelConfig registered_dense_model(std::string name, std::string model_type,
                                   std::uint64_t num_layers,
                                   std::uint64_t num_query_heads,
                                   std::uint64_t num_kv_heads,
                                   std::uint64_t hidden_size,
                                   std::uint64_t intermediate_size,
                                   bool gated_mlp = true,
                                   bool fused_add_norm = true) {
    ModelConfig model{};
    model.name = std::move(name);
    model.model_type = std::move(model_type);
    model.kind = ModelKind::kDense;
    model.num_layers = num_layers;
    model.hidden_size = hidden_size;
    model.intermediate_size = intermediate_size;
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
        return registered_dense_model(std::string{name}, "internlm", 60, 40,
                                      40, 5'120, 13'824);
    }
    if (name == "internlm/internlm2-20b") {
        return registered_dense_model(std::string{name}, "internlm2", 48, 48,
                                      8, 6'144, 16'384);
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
    if (_dupenv_s(&configured, &configured_size,
                  "FRONTIER_MODEL_CONFIG_DIR") == 0 &&
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
#ifdef FRONTIER_MODEL_CONFIG_DIR
    directories.emplace_back(FRONTIER_MODEL_CONFIG_DIR);
#endif
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
           model_type == "qwen3_moe" || model_type == "deepseek_v3";
}

ModelConfig parse_model_asset(std::string_view model_name, const Json &root,
                              const std::filesystem::path &path) {
    if (!root.is_object()) {
        throw ConfigError("model config must be a JSON object: " +
                          path.string());
    }
    const std::string source = "model config " + path.string();
    ModelConfig model{};
    model.name = std::string{model_name};
    model.model_type = lowercase(
        root.contains("model_type") && root.at("model_type").is_string()
            ? root.at("model_type").get<std::string>()
            : std::string{});
    model.num_layers =
        require_positive_u64(root, "num_hidden_layers", source);
    model.hidden_size = require_positive_u64(root, "hidden_size", source);
    model.num_query_heads =
        require_positive_u64(root, "num_attention_heads", source);
    model.num_kv_heads =
        require_positive_u64(root, "num_key_value_heads", source);
    model.num_experts = first_positive_u64(
        root, {"num_experts", "num_local_experts", "n_routed_experts"}, 1,
        source);
    model.kind = model.num_experts > 1 ? ModelKind::kMoe : ModelKind::kDense;
    model.intermediate_size = model.is_moe()
                                  ? first_positive_u64(
                                        root,
                                        {"moe_intermediate_size",
                                         "intermediate_size"},
                                        0, source)
                                  : require_positive_u64(
                                        root, "intermediate_size", source);
    if (model.intermediate_size == 0) {
        throw ConfigError(source +
                          " requires moe_intermediate_size for an MoE model");
    }
    if (root.contains("head_dim")) {
        model.head_dim = require_positive_u64(root, "head_dim", source);
    } else {
        if (model.hidden_size % model.num_query_heads != 0) {
            throw ConfigError(source +
                              " hidden_size must divide evenly over query "
                              "heads when head_dim is omitted");
        }
        model.head_dim = model.hidden_size / model.num_query_heads;
    }
    const std::string hidden_act = lowercase(
        root.contains("hidden_act") && root.at("hidden_act").is_string()
            ? root.at("hidden_act").get<std::string>()
            : std::string{});
    model.gated_mlp = hidden_act == "silu" || hidden_act == "swish";
    model.fused_add_norm = inferred_fused_add_norm(root, model.model_type);
    model.num_experts_per_token = model.is_moe()
                                      ? require_positive_u64(
                                            root, "num_experts_per_tok", source)
                                      : 1;
    model.total_expert_num = model.num_experts;
    model.router_topk = model.num_experts_per_token;
    model.use_mla = optional_bool(root, "use_mla", false, source) ||
                    ((model.model_type == "deepseek_v2" ||
                      model.model_type == "deepseek_v3" ||
                      model.model_type == "deepseek_mtp" ||
                      model.model_type == "kimi_k2") &&
                     root.contains("kv_lora_rank"));
    model.use_mfa = optional_bool(root, "use_mfa", false, source);
    model.q_lora_rank = optional_positive_u64(root, "q_lora_rank", 0, source);
    model.kv_lora_rank =
        optional_positive_u64(root, "kv_lora_rank", 0, source);
    model.qk_nope_head_dim =
        optional_positive_u64(root, "qk_nope_head_dim", 0, source);
    model.qk_rope_head_dim =
        optional_positive_u64(root, "qk_rope_head_dim", 0, source);
    model.qk_head_dim =
        optional_positive_u64(root, "qk_head_dim", 0, source);
    if (model.qk_head_dim == 0 && model.qk_nope_head_dim > 0 &&
        model.qk_rope_head_dim > 0) {
        model.qk_head_dim = model.qk_nope_head_dim + model.qk_rope_head_dim;
    }
    if (model.use_mla &&
        model.qk_head_dim !=
            model.qk_nope_head_dim + model.qk_rope_head_dim) {
        throw ConfigError(source +
                          " qk_head_dim must equal qk_nope_head_dim + "
                          "qk_rope_head_dim for MLA");
    }
    model.v_head_dim =
        optional_positive_u64(root, "v_head_dim", 0, source);
    model.share_q_dim =
        optional_positive_u64(root, "share_q_dim", 0, source);
    for (const std::string_view marker : {"dsa_topk", "dsa_top_k",
                                          "dsa_index_topk", "dsa_indexer"}) {
        if (root.contains(marker) && json_truthy(root.at(marker))) {
            model.has_dsa_marker = true;
        }
    }
    for (const std::string_view marker : {"sliding_window_pattern",
                                          "dual_chunk_attention",
                                          "attention_chunk_size"}) {
        if (root.contains(marker) && json_truthy(root.at(marker))) {
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
    const std::filesystem::path path = find_model_file(model_name);
    return parse_model_asset(model_name, read_model_json(path), path);
}

} // namespace frontier::config
