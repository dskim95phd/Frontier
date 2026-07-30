#pragma once

#include <cstdint>
#include <vector>

#include "frontier/cc_backend/base_cc_backend.h"
#include "frontier/execution_time_predictor/analytical_model.h"
#include "frontier/moe/routing.h"

namespace frontier::moe {

struct MoEModel {
  std::uint64_t hidden_size = 0;
  std::uint64_t intermediate_size = 0;
  std::uint64_t model_num_experts = 0;
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

[[nodiscard]] MoELayerTime predict_moe_layer(
    const execution_time_predictor::DeviceCeilings& device,
    const execution_time_predictor::AnalyticalConfig& config,
    const MoEModel& model,
    std::uint64_t input_tokens,
    std::uint64_t router_topk,
    const std::vector<std::uint64_t>& local_expert_tokens,
    execution_time_predictor::Precision precision);

[[nodiscard]] MoELanePrediction predict_moe_lanes(
    const execution_time_predictor::DeviceCeilings& device,
    const execution_time_predictor::AnalyticalConfig& config,
    const MoEModel& model,
    const RoutingAllocation& routing,
    std::uint64_t router_topk,
    execution_time_predictor::Precision precision);

[[nodiscard]] MoECommunicationTime predict_moe_communication(
    const cc_backend::BaseCCBackend& communication,
    std::uint64_t input_tokens,
    std::uint64_t hidden_size,
    std::uint64_t routed_tokens,
    std::uint64_t attention_tp_size,
    std::uint64_t moe_tp_size,
    std::uint64_t expert_parallel_size,
    std::uint64_t data_parallel_size,
    bool has_pipeline_boundary,
    double element_bytes);

}  // namespace frontier::moe
