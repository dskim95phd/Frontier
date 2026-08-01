#include "frontier/attention/model_binding.h"

#include <stdexcept>
#include <string>

#include "frontier/config/config.h"

namespace frontier::attention {
namespace {

bool is_dsa_model_type(const std::string &model_type) {
    return model_type.find("deepseek_v32") != std::string::npos ||
           model_type.find("deepseek_v3_2") != std::string::npos ||
           model_type.find("deepseek_v3.2") != std::string::npos;
}

} // namespace

AttentionFamilyBinding bind_attention_family(const config::ModelConfig &model) {
    if (model.has_dsa_marker || is_dsa_model_type(model.model_type)) {
        return AttentionFamilyBinding{AttentionMemoryLayout::kFrozenDsa,
                                      AttentionVariant::kDsa, false};
    }
    if (model.use_mla && model.use_mfa) {
        throw std::invalid_argument(
            "model attention use_mla and use_mfa are mutually exclusive");
    }
    if (model.use_mla) {
        if (model.kv_lora_rank == 0 || model.qk_nope_head_dim == 0 ||
            model.qk_rope_head_dim == 0 || model.v_head_dim == 0) {
            throw std::invalid_argument(
                "MLA attention requires kv_lora_rank, qk_nope_head_dim, "
                "qk_rope_head_dim, and v_head_dim");
        }
        return AttentionFamilyBinding{AttentionMemoryLayout::kLatentMla,
                                      AttentionVariant::kMla, true};
    }
    if (!model.exotic_attention_fields.empty()) {
        std::string fields;
        for (const auto &field : model.exotic_attention_fields) {
            if (!fields.empty()) {
                fields += ", ";
            }
            fields += field;
        }
        throw std::invalid_argument(
            "unrecognized exotic attention fields require an explicit "
            "family binding: " +
            fields);
    }
    if (model.num_query_heads == 0 || model.num_kv_heads == 0) {
        throw std::invalid_argument(
            "attention query and KV head counts must be positive");
    }
    if (model.use_mfa &&
        (model.share_q_dim == 0 || model.head_dim == 0 ||
         model.num_kv_heads != 1)) {
        throw std::invalid_argument(
            "MFA attention requires share_q_dim, head_dim, and exactly one "
            "dense KV head");
    }
    if (model.num_kv_heads == model.num_query_heads) {
        return AttentionFamilyBinding{AttentionMemoryLayout::kDenseKv,
                                      AttentionVariant::kMha, true};
    }
    if (model.num_kv_heads == 1) {
        return AttentionFamilyBinding{AttentionMemoryLayout::kDenseKv,
                                      AttentionVariant::kMqa, true};
    }
    if (model.num_kv_heads < model.num_query_heads) {
        return AttentionFamilyBinding{AttentionMemoryLayout::kDenseKv,
                                      AttentionVariant::kGqa, true};
    }
    throw std::invalid_argument(
        "unsupported attention topology: num_kv_heads exceeds "
        "num_query_heads");
}

std::string_view to_string(AttentionMemoryLayout layout) noexcept {
    switch (layout) {
    case AttentionMemoryLayout::kDenseKv:
        return "dense_kv";
    case AttentionMemoryLayout::kLatentMla:
        return "latent_mla";
    case AttentionMemoryLayout::kFrozenDsa:
        return "frozen_dsa";
    }
    return "unknown";
}

std::string_view to_string(AttentionVariant variant) noexcept {
    switch (variant) {
    case AttentionVariant::kMha:
        return "mha";
    case AttentionVariant::kGqa:
        return "gqa";
    case AttentionVariant::kMqa:
        return "mqa";
    case AttentionVariant::kMla:
        return "mla";
    case AttentionVariant::kDsa:
        return "dsa";
    }
    return "unknown";
}

} // namespace frontier::attention
