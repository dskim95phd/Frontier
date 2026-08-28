#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "frontier/execution_time_predictor/analytical_roofline_primitives.h"

namespace frontier::execution_time_predictor::detail {

enum class AttentionKind {
    kStandard,
    kMla,
    kMfa,
    kKda,
};

struct AttentionRequestSlice {
    std::uint64_t query_tokens;
    std::uint64_t past_context;
};

[[nodiscard]] std::uint64_t prefill_attention_token_pairs(
    const std::vector<AttentionRequestSlice> &requests);

struct DenseBatch {
    std::uint64_t total_tokens;
    std::vector<AttentionRequestSlice> prefill_requests;
    std::vector<AttentionRequestSlice> decode_requests;
};

struct DenseModel {
    std::uint64_t hidden_size;
    std::uint64_t intermediate_size;
    std::uint64_t num_query_heads;
    std::uint64_t num_kv_heads;
    std::uint64_t head_dim;
    std::uint64_t tensor_parallel_size;
    bool gated_mlp;
    bool fused_add_norm;
    bool use_mla = false;
    bool mla_use_output_gate = false;
    bool mla_use_nope = false;
    bool use_mfa = false;
    std::uint64_t q_lora_rank = 0;
    std::uint64_t kv_lora_rank = 0;
    std::uint64_t qk_nope_head_dim = 0;
    std::uint64_t qk_rope_head_dim = 0;
    std::uint64_t qk_head_dim = 0;
    std::uint64_t v_head_dim = 0;
    std::uint64_t share_q_dim = 0;
    std::uint64_t decode_context_parallel_size = 1;
    bool use_kda = false;
    std::uint64_t kda_num_heads = 0;
    std::uint64_t kda_num_k_heads = 0;
    std::uint64_t kda_num_v_heads = 0;
    std::uint64_t kda_key_head_dim = 0;
    std::uint64_t kda_value_head_dim = 0;
    std::uint64_t kda_head_dim = 0;
    std::uint64_t kda_short_conv_kernel_size = 0;
    std::uint64_t kda_conv_state_dim = 0;
    std::uint64_t attn_res_block_size = 0;

    [[nodiscard]] static constexpr DenseModel llama2_7b_tp8() noexcept {
        return llama2_7b(8);
    }

    [[nodiscard]] static constexpr DenseModel
    llama2_7b(std::uint64_t tensor_parallel_size) noexcept {
        return DenseModel{
            4'096, 11'008, 32, 32, 128, tensor_parallel_size, true, true,
        };
    }
};

struct DenseLayerTimes {
    double attention_pre_projection_ms;
    double attention_post_projection_ms;
    double rope_ms;
    double kv_cache_save_ms;
    double attention_norm_ms;
    double attention_inter_norm_ms;
    double attention_wq_projection_ms;
    double prefill_attention_ms;
    double decode_attention_ms;
    double mlp_up_projection_ms;
    double mlp_activation_ms;
    double mlp_down_projection_ms;
    double mlp_norm_ms;
    double residual_add_ms;
    double kda_projection_ms = 0.0;
    double kda_short_conv_ms = 0.0;
    double kda_recurrent_ms = 0.0;
    double kda_gate_norm_ms = 0.0;
    double attn_res_ms = 0.0;

    [[nodiscard]] double total_ms() const noexcept;
};

struct DenseOperatorPrecisions {
    Precision attention = Precision::kFp16;
    Precision dense = Precision::kFp16;
    Precision kv_cache = Precision::kFp16;
    std::optional<Precision> attention_weight;
    std::optional<Precision> attention_activation;
    std::optional<Precision> dense_weight;
    std::optional<Precision> dense_activation;
    Precision kda_state = Precision::kFp32;
    // QK/PV sequence-attention compute. When omitted, inherit `attention`.
    // KDA is deliberately excluded because it is a recurrent BF16/FP32 path.
    std::optional<Precision> attention_core;
};

[[nodiscard]] KernelWork
attention_context_work(const std::vector<AttentionRequestSlice> &requests,
                       std::uint64_t local_query_heads,
                       std::uint64_t local_kv_heads, std::uint64_t head_dim,
                       double core_element_bytes,
                       double activation_element_bytes,
                       double kv_cache_element_bytes);
[[nodiscard]] KernelWork
attention_context_work(const std::vector<AttentionRequestSlice> &requests,
                       std::uint64_t local_query_heads,
                       std::uint64_t local_kv_heads, std::uint64_t head_dim,
                       double activation_element_bytes,
                       double kv_cache_element_bytes);
[[nodiscard]] KernelWork
attention_context_work(const std::vector<AttentionRequestSlice> &requests,
                       std::uint64_t local_query_heads,
                       std::uint64_t local_kv_heads, std::uint64_t head_dim,
                       double element_bytes);
[[nodiscard]] KernelWork mla_unabsorbed_attention_work(
    const std::vector<AttentionRequestSlice> &requests,
    std::uint64_t local_query_heads, std::uint64_t qk_nope_head_dim,
    std::uint64_t qk_rope_head_dim, std::uint64_t v_head_dim,
    double core_element_bytes, double activation_element_bytes,
    double rope_cache_element_bytes);
[[nodiscard]] KernelWork mla_unabsorbed_attention_work(
    const std::vector<AttentionRequestSlice> &requests,
    std::uint64_t local_query_heads, std::uint64_t qk_nope_head_dim,
    std::uint64_t qk_rope_head_dim, std::uint64_t v_head_dim,
    double activation_element_bytes, double rope_cache_element_bytes);
[[nodiscard]] KernelWork mla_absorbed_attention_work(
    const std::vector<AttentionRequestSlice> &requests,
    std::uint64_t local_query_heads, std::uint64_t kv_lora_rank,
    std::uint64_t qk_rope_head_dim, double core_element_bytes,
    double activation_element_bytes, double latent_cache_element_bytes,
    double rope_cache_element_bytes);
[[nodiscard]] KernelWork mla_absorbed_attention_work(
    const std::vector<AttentionRequestSlice> &requests,
    std::uint64_t local_query_heads, std::uint64_t kv_lora_rank,
    std::uint64_t qk_rope_head_dim, double activation_element_bytes,
    double latent_cache_element_bytes, double rope_cache_element_bytes);
[[nodiscard]] DenseLayerTimes
predict_dense_layer(const DeviceCeilings &device,
                    const AnalyticalConfig &config, const DenseModel &model,
                    const DenseBatch &batch,
                    const DenseOperatorPrecisions &precisions);
[[nodiscard]] DenseLayerTimes
predict_dense_layer(const DeviceCeilings &device,
                    const AnalyticalConfig &config, const DenseModel &model,
                    const DenseBatch &batch, Precision precision);

} // namespace frontier::execution_time_predictor::detail
