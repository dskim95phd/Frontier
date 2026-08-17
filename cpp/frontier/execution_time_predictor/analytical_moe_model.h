#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <vector>

#include "frontier/cc_backend/base_cc_backend.h"
#include "frontier/config/config.h"
#include "frontier/execution_time_predictor/analytical_roofline_primitives.h"

namespace frontier::execution_time_predictor::detail {

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
    // Receiver-side load and source-to-destination traffic summaries.  The
    // latter counts each input token once per destination EP lane, even when
    // several of its top-k experts live on that same lane.
    std::vector<std::uint64_t> lane_routed_tokens;
    std::vector<std::uint64_t> lane_active_experts;
    std::vector<std::uint64_t> lane_unique_tokens;

    friend bool operator==(const RoutingAllocation &lhs,
                           const RoutingAllocation &rhs) {
        return std::tie(lhs.input_tokens, lhs.routed_tokens,
                        lhs.global_expert_tokens, lhs.lane_expert_tokens,
                        lhs.lane_routed_tokens, lhs.lane_active_experts,
                        lhs.lane_unique_tokens) ==
               std::tie(rhs.input_tokens, rhs.routed_tokens,
                        rhs.global_expert_tokens, rhs.lane_expert_tokens,
                        rhs.lane_routed_tokens, rhs.lane_active_experts,
                        rhs.lane_unique_tokens);
    }
};

[[nodiscard]] std::vector<std::uint64_t>
discretize_expert_weights(std::uint64_t total_tokens,
                          const std::vector<double> &weights);
[[nodiscard]] RoutingAllocation
route_tokens(std::uint64_t input_tokens, std::uint64_t router_topk,
             std::uint64_t total_experts, std::uint64_t expert_parallel_size,
             const config::MoeRoutingConfig &config, std::uint64_t layer_id,
             std::optional<std::uint64_t> sample_id = std::nullopt);

struct MoEModel {
    std::uint64_t hidden_size = 0;
    std::uint64_t intermediate_size = 0;
    std::uint64_t model_num_experts = 0;
    std::uint64_t num_shared_experts = 0;
    std::uint64_t moe_tensor_parallel_size = 1;
    std::uint64_t routed_expert_hidden_size = 0;
    std::uint64_t attn_res_block_size = 0;
    bool latent_moe_use_norm = false;
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
    double latent_projection_ms = 0.0;
    double latent_norm_ms = 0.0;
    double attn_res_ms = 0.0;

    [[nodiscard]] double total_ms() const noexcept;
};

struct MoECommunicationTime {
    double attention_tp_ms = 0.0;
    double moe_tp_ms = 0.0;
    double ep_dispatch_ms = 0.0;
    double ep_combine_ms = 0.0;
    // Unhidden transport time before MegaMoE overlap is applied. Equal to the
    // exposed values for the generic backend.
    double raw_ep_dispatch_ms = 0.0;
    double raw_ep_combine_ms = 0.0;
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
    Precision shared_expert = Precision::kFp16;
    std::optional<Precision> expert_weight;
    std::optional<Precision> expert_activation;
    std::optional<Precision> latent_moe_projection_weight;
    std::optional<Precision> latent_moe_projection_activation;
    std::optional<Precision> shared_expert_weight;
    std::optional<Precision> shared_expert_activation;
    std::optional<Precision> router_weight;
    std::optional<Precision> router_activation;
    std::optional<Precision> dense_weight;
    std::optional<Precision> dense_activation;
    std::optional<Precision> router_compute;
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
    bool has_pipeline_boundary, double element_bytes,
    std::uint64_t routed_hidden_size = 0,
    std::string_view moe_communication_backend = "generic",
    const RoutingAllocation *routing = nullptr,
    double fused_expert_compute_ms = 0.0,
    double overlap_residual = 0.35);

} // namespace frontier::execution_time_predictor::detail
