#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace frontier::execution_time_predictor {

enum class Precision {
  kFp32,
  kFp16,
  kBf16,
  kFp8,
  kInt8,
  kFp4,
  kInt4,
};

enum class Bottleneck {
  kNone,
  kLaunch,
  kCompute,
  kHbm,
};

struct DeviceCeilings {
  double hbm_bandwidth_tbps;
  double fp32_tflops;
  double fp16_tflops;
  double fp8_tflops;
  double fp4_tflops;

  [[nodiscard]] static constexpr DeviceCeilings rubin() noexcept {
    return DeviceCeilings{
        .hbm_bandwidth_tbps = 22.0,
        .fp32_tflops = 400.0,
        .fp16_tflops = 4'000.0,
        .fp8_tflops = 17'500.0,
        .fp4_tflops = 35'000.0,
    };
  }
};

struct KernelWork {
  double flops;
  double hbm_bytes;
};

struct Efficiency {
  double compute;
  double memory;
  double overlap_penalty;
};

struct RooflineResult {
  double compute_time_ms;
  double memory_time_ms;
  double launch_time_ms;
  double predicted_time_ms;
  Bottleneck bottleneck;
};

struct AnalyticalConfig {
  Efficiency large_gemm{0.65, 0.75, 0.10};
  Efficiency small_gemm{0.25, 0.60, 0.50};
  Efficiency prefill_attention{0.55, 0.65, 0.20};
  Efficiency decode_attention{0.20, 0.60, 0.50};
  Efficiency streaming{0.20, 0.75, 0.0};
  double kernel_launch_latency_us = 5.0;
  std::uint64_t small_gemm_token_threshold = 128;
};

struct AttentionRequestSlice {
  std::uint64_t query_tokens;
  std::uint64_t past_context;
};

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

  [[nodiscard]] static constexpr DenseModel llama2_7b_tp8() noexcept {
    return DenseModel{
        .hidden_size = 4'096,
        .intermediate_size = 11'008,
        .num_query_heads = 32,
        .num_kv_heads = 32,
        .head_dim = 128,
        .tensor_parallel_size = 8,
        .gated_mlp = true,
        .fused_add_norm = true,
    };
  }
};

struct DenseLayerTimes {
  double attention_pre_projection_ms;
  double attention_post_projection_ms;
  double rope_ms;
  double kv_cache_save_ms;
  double attention_norm_ms;
  double prefill_attention_ms;
  double decode_attention_ms;
  double mlp_up_projection_ms;
  double mlp_activation_ms;
  double mlp_down_projection_ms;
  double mlp_norm_ms;
  double residual_add_ms;

  [[nodiscard]] double total_ms() const noexcept;
};

class AnalyticalModelError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

[[nodiscard]] double bytes_per_element(Precision precision) noexcept;
[[nodiscard]] double peak_tflops(
    const DeviceCeilings& device,
    Precision precision);
[[nodiscard]] RooflineResult predict_roofline(
    const DeviceCeilings& device,
    Precision precision,
    const KernelWork& work,
    const Efficiency& efficiency,
    double kernel_launch_latency_us);
[[nodiscard]] KernelWork gemm_work(
    std::uint64_t m,
    std::uint64_t k,
    std::uint64_t n,
    double element_bytes,
    std::uint64_t weight_multiplier = 1);
[[nodiscard]] KernelWork streaming_work(
    double elements_read,
    double elements_written,
    double flops,
    double element_bytes);
[[nodiscard]] KernelWork attention_context_work(
    const std::vector<AttentionRequestSlice>& requests,
    std::uint64_t local_query_heads,
    std::uint64_t local_kv_heads,
    std::uint64_t head_dim,
    double element_bytes);
[[nodiscard]] DenseLayerTimes predict_dense_layer(
    const DeviceCeilings& device,
    const AnalyticalConfig& config,
    const DenseModel& model,
    const DenseBatch& batch,
    Precision precision);

}  // namespace frontier::execution_time_predictor
