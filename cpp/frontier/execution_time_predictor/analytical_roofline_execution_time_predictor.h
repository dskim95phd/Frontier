#pragma once

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "frontier/cc_backend/base_cc_backend.h"
#include "frontier/config/config.h"
#include "frontier/entities/batch.h"
#include "frontier/entities/execution_time.h"
#include "frontier/entities/request.h"
#include "frontier/execution_time_predictor/base_execution_time_predictor.h"

namespace frontier::execution_time_predictor {

// Internal analytical primitives are declared next to the predictor so the
// Python AnalyticalRooflineExecutionTimePredictor remains the only public
// runtime abstraction.
namespace detail {

enum class Precision {
    kFp32,
    kFp16,
    kBf16,
    kFp8,
    kInt8,
    kFp4,
    kInt4,
};

[[nodiscard]] Precision precision_from_string(std::string_view precision);

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
            22.0, 400.0, 4'000.0, 17'500.0, 35'000.0,
        };
    }

    // Dense, non-sparse per-GPU ceilings derived from the public GB300 NVL72
    // rack totals. Overrides in AnalyticalExecutionModelConfig are applied
    // after selecting this preset.
    [[nodiscard]] static constexpr DeviceCeilings gb300() noexcept {
        return DeviceCeilings{
            8.0, 83.33333333333333, 2'500.0, 5'000.0, 15'000.0,
        };
    }

    [[nodiscard]] static DeviceCeilings
    from_config(const config::AnalyticalExecutionModelConfig &config);
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
    Efficiency moe{0.45, 0.65, 0.30};
    Efficiency routing{0.15, 0.55, 0.75};
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
    bool use_mla = false;
    bool use_mfa = false;
    std::uint64_t q_lora_rank = 0;
    std::uint64_t kv_lora_rank = 0;
    std::uint64_t qk_nope_head_dim = 0;
    std::uint64_t qk_rope_head_dim = 0;
    std::uint64_t qk_head_dim = 0;
    std::uint64_t v_head_dim = 0;
    std::uint64_t share_q_dim = 0;
    std::uint64_t decode_context_parallel_size = 1;

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
};

class AnalyticalModelError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

[[nodiscard]] double bytes_per_element(Precision precision) noexcept;
[[nodiscard]] double peak_tflops(const DeviceCeilings &device,
                                 Precision precision);
[[nodiscard]] RooflineResult predict_roofline(const DeviceCeilings &device,
                                              Precision precision,
                                              const KernelWork &work,
                                              const Efficiency &efficiency,
                                              double kernel_launch_latency_us);
[[nodiscard]] KernelWork gemm_work(std::uint64_t m, std::uint64_t k,
                                   std::uint64_t n, double element_bytes,
                                   std::uint64_t weight_multiplier = 1);
[[nodiscard]] KernelWork gemm_work(std::uint64_t m, std::uint64_t k,
                                   std::uint64_t n, double weight_element_bytes,
                                   double activation_element_bytes,
                                   std::uint64_t weight_multiplier);
[[nodiscard]] KernelWork streaming_work(double elements_read,
                                        double elements_written, double flops,
                                        double element_bytes);
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
    double activation_element_bytes, double rope_cache_element_bytes);
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

class ParallelDomainError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

struct ExpertRange {
    std::uint64_t begin = 0;
    std::uint64_t end = 0;

    [[nodiscard]] std::uint64_t size() const noexcept { return end - begin; }

    friend bool operator==(const ExpertRange &lhs, const ExpertRange &rhs) {
        return lhs.begin == rhs.begin && lhs.end == rhs.end;
    }
};

class ExpertParallelDomain {
  public:
    ExpertParallelDomain(std::uint64_t total_experts,
                         std::uint64_t expert_parallel_size);

    [[nodiscard]] std::uint64_t total_experts() const noexcept {
        return total_experts_;
    }
    [[nodiscard]] std::uint64_t size() const noexcept {
        return expert_parallel_size_;
    }
    [[nodiscard]] std::uint64_t experts_per_lane() const noexcept {
        return total_experts_ / expert_parallel_size_;
    }
    [[nodiscard]] ExpertRange expert_range(std::uint64_t lane) const;
    [[nodiscard]] std::uint64_t owner(std::uint64_t expert_id) const;
    [[nodiscard]] std::vector<std::vector<std::uint64_t>>
    partition(const std::vector<std::uint64_t> &global_counts) const;

  private:
    std::uint64_t total_experts_;
    std::uint64_t expert_parallel_size_;
};

class RoutingError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

struct RoutingAllocation {
    std::uint64_t input_tokens = 0;
    std::uint64_t routed_tokens = 0;
    std::vector<std::uint64_t> global_expert_tokens;
    std::vector<std::vector<std::uint64_t>> lane_expert_tokens;

    friend bool operator==(const RoutingAllocation &lhs,
                           const RoutingAllocation &rhs) {
        return std::tie(lhs.input_tokens, lhs.routed_tokens,
                        lhs.global_expert_tokens, lhs.lane_expert_tokens) ==
               std::tie(rhs.input_tokens, rhs.routed_tokens,
                        rhs.global_expert_tokens, rhs.lane_expert_tokens);
    }
};

[[nodiscard]] std::vector<std::uint64_t>
discretize_expert_weights(std::uint64_t total_tokens,
                          const std::vector<double> &weights);
[[nodiscard]] RoutingAllocation
route_tokens(std::uint64_t input_tokens, std::uint64_t router_topk,
             std::uint64_t total_experts, std::uint64_t expert_parallel_size,
             const config::MoeRoutingConfig &config, std::uint64_t layer_id);

struct MoEModel {
    std::uint64_t hidden_size = 0;
    std::uint64_t intermediate_size = 0;
    std::uint64_t model_num_experts = 0;
    std::uint64_t num_shared_experts = 0;
    std::uint64_t moe_tensor_parallel_size = 1;
    bool gated_mlp = true;
    bool fused_add_norm = false;
};

struct MoELayerTime {
    double gating_linear_ms = 0.0;
    double gating_routing_topk_ms = 0.0;
    double grouped_up_projection_ms = 0.0;
    double grouped_down_projection_ms = 0.0;
    double shuffling_ms = 0.0;
    double post_attention_norm_ms = 0.0;

    [[nodiscard]] double total_ms() const noexcept;
};

struct MoECommunicationTime {
    double attention_tp_ms = 0.0;
    double moe_tp_ms = 0.0;
    double ep_dispatch_ms = 0.0;
    double ep_combine_ms = 0.0;
    double dp_input_ms = 0.0;
    double dp_output_ms = 0.0;
    double pipeline_parallel_ms = 0.0;

    [[nodiscard]] double total_ms() const noexcept;
};

struct MoELanePrediction {
    std::vector<MoELayerTime> lane_times;
    std::uint64_t critical_lane = 0;
    double critical_lane_time_ms = 0.0;
};

struct MoEOperatorPrecisions {
    Precision expert = Precision::kFp16;
    Precision router = Precision::kFp16;
    Precision dense = Precision::kFp16;
    std::optional<Precision> expert_weight;
    std::optional<Precision> expert_activation;
    std::optional<Precision> router_weight;
    std::optional<Precision> router_activation;
    std::optional<Precision> dense_weight;
    std::optional<Precision> dense_activation;
};

[[nodiscard]] double predict_output_projection_ms(
    const DeviceCeilings &device, const AnalyticalConfig &config,
    std::uint64_t tokens, std::uint64_t hidden_size, std::uint64_t vocab_size,
    std::uint64_t tensor_parallel_size, Precision weight_precision,
    Precision activation_precision);

[[nodiscard]] MoELayerTime
predict_moe_layer(const DeviceCeilings &device, const AnalyticalConfig &config,
                  const MoEModel &model, std::uint64_t input_tokens,
                  std::uint64_t router_topk,
                  const std::vector<std::uint64_t> &local_expert_tokens,
                  const MoEOperatorPrecisions &precisions);
[[nodiscard]] MoELayerTime
predict_moe_layer(const DeviceCeilings &device, const AnalyticalConfig &config,
                  const MoEModel &model, std::uint64_t input_tokens,
                  std::uint64_t router_topk,
                  const std::vector<std::uint64_t> &local_expert_tokens,
                  Precision precision);
[[nodiscard]] MoELanePrediction
predict_moe_lanes(const DeviceCeilings &device, const AnalyticalConfig &config,
                  const MoEModel &model, const RoutingAllocation &routing,
                  std::uint64_t router_topk,
                  const MoEOperatorPrecisions &precisions);
[[nodiscard]] MoELanePrediction
predict_moe_lanes(const DeviceCeilings &device, const AnalyticalConfig &config,
                  const MoEModel &model, const RoutingAllocation &routing,
                  std::uint64_t router_topk, Precision precision);
[[nodiscard]] MoECommunicationTime predict_moe_communication(
    const cc_backend::BaseCCBackend &communication, std::uint64_t input_tokens,
    std::uint64_t hidden_size, std::uint64_t routed_tokens,
    std::uint64_t attention_tp_size, std::uint64_t moe_tp_size,
    std::uint64_t expert_parallel_size, std::uint64_t data_parallel_size,
    bool has_pipeline_boundary, double element_bytes);

} // namespace detail

class AnalyticalRooflineExecutionTimePredictor final
    : public BaseExecutionTimePredictor {
  public:
    explicit AnalyticalRooflineExecutionTimePredictor(
        config::AnalyticalExecutionModelConfig config);
    AnalyticalRooflineExecutionTimePredictor(
        config::AnalyticalExecutionModelConfig config,
        config::ParallelismConfig parallelism);
    AnalyticalRooflineExecutionTimePredictor(
        config::AnalyticalExecutionModelConfig config,
        config::ParallelismConfig parallelism, config::ModelConfig model,
        config::MoeRoutingConfig routing);
    AnalyticalRooflineExecutionTimePredictor(
        config::AnalyticalExecutionModelConfig config,
        config::ParallelismConfig parallelism, config::ModelConfig model,
        config::MoeRoutingConfig routing,
        std::shared_ptr<const cc_backend::BaseCCBackend> communication_backend);

    [[nodiscard]] ExecutionTimePrediction
    predict_stage_execution_time(const entities::Batch &batch,
                                 const std::vector<entities::Request> &requests,
                                 StageId stage_id) const override;
    [[nodiscard]] bool supports_lazy_moe_prediction() const noexcept override {
        return true;
    }
    [[nodiscard]] ExecutionTimePrediction prepare_moe_stage_execution(
        const entities::Batch &batch,
        const std::vector<entities::Request> &requests,
        StageId stage_id) const override;
    [[nodiscard]] ExecutionTimePrediction predict_moe_layer_execution(
        const entities::Batch &batch,
        const std::vector<entities::Request> &requests, StageId stage_id,
        std::uint64_t local_moe_layer) const override;

  private:
    [[nodiscard]] ExecutionTimePrediction predict_execution(
        const entities::Batch &batch,
        const std::vector<entities::Request> &requests, StageId stage_id,
        std::optional<std::uint64_t> selected_moe_layer) const;

    config::AnalyticalExecutionModelConfig config_;
    detail::DeviceCeilings device_;
    config::ParallelismConfig parallelism_;
    config::ModelConfig model_;
    config::MoeRoutingConfig routing_;
    std::shared_ptr<const cc_backend::BaseCCBackend> communication_backend_;
};

} // namespace frontier::execution_time_predictor
