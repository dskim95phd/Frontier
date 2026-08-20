#include "frontier/config/model_registry_internal.h"

#include <algorithm>
#include <cctype>
#include <string>
#include <utility>

#include "frontier/attention/model_binding.h"

namespace frontier::config::model_detail {
namespace {

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](char ch) {
        return static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    });
    return value;
}

} // namespace

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

} // namespace frontier::config::model_detail
