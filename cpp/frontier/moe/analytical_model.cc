#include "frontier/moe/analytical_model.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>

namespace frontier::moe {
namespace {

using execution_time_predictor::AnalyticalModelError;
using execution_time_predictor::Efficiency;
using execution_time_predictor::KernelWork;

std::uint64_t ceil_div(
    std::uint64_t numerator,
    std::uint64_t denominator) {
  if (denominator == 0) {
    throw AnalyticalModelError("MoE TP size must be positive");
  }
  return numerator / denominator +
      static_cast<std::uint64_t>(numerator % denominator != 0);
}

double predict_ms(
    const execution_time_predictor::DeviceCeilings& device,
    execution_time_predictor::Precision precision,
    const KernelWork& work,
    const Efficiency& efficiency,
    double launch_latency_us) {
  return execution_time_predictor::predict_roofline(
             device,
             precision,
             work,
             efficiency,
             launch_latency_us)
      .predicted_time_ms;
}

std::uint64_t payload_bytes(
    std::uint64_t tokens,
    std::uint64_t hidden_size,
    double element_bytes) {
  const long double bytes =
      static_cast<long double>(tokens) *
      static_cast<long double>(hidden_size) *
      static_cast<long double>(element_bytes);
  if (!std::isfinite(static_cast<double>(bytes)) ||
      bytes > static_cast<long double>(
                  std::numeric_limits<std::uint64_t>::max())) {
    throw AnalyticalModelError("MoE communication payload overflows uint64");
  }
  return static_cast<std::uint64_t>(std::ceil(bytes));
}

}  // namespace

double MoELayerTime::total_ms() const noexcept {
  return gating_linear_ms + gating_routing_topk_ms +
      grouped_up_projection_ms + grouped_down_projection_ms +
      shuffling_ms + post_attention_norm_ms;
}

double MoECommunicationTime::total_ms() const noexcept {
  return attention_tp_ms + moe_tp_ms + ep_dispatch_ms +
      ep_combine_ms + dp_input_ms + dp_output_ms +
      pipeline_parallel_ms;
}

MoELayerTime predict_moe_layer(
    const execution_time_predictor::DeviceCeilings& device,
    const execution_time_predictor::AnalyticalConfig& config,
    const MoEModel& model,
    std::uint64_t input_tokens,
    std::uint64_t router_topk,
    const std::vector<std::uint64_t>& local_expert_tokens,
    execution_time_predictor::Precision precision) {
  if (model.hidden_size == 0 || model.intermediate_size == 0 ||
      model.model_num_experts == 0 ||
      model.moe_tensor_parallel_size == 0 ||
      router_topk == 0) {
    throw AnalyticalModelError("MoE model dimensions must be positive");
  }

  const double element_bytes =
      execution_time_predictor::bytes_per_element(precision);
  const std::uint64_t local_intermediate =
      ceil_div(
          model.intermediate_size,
          model.moe_tensor_parallel_size);
  const std::uint64_t gated_multiplier = model.gated_mlp ? 2 : 1;
  const auto roofline = [&](const KernelWork& work,
                            const Efficiency& efficiency) {
    return predict_ms(
        device,
        precision,
        work,
        efficiency,
        config.kernel_launch_latency_us);
  };

  KernelWork grouped_up{};
  KernelWork grouped_down{};
  std::uint64_t local_routed_tokens = 0;
  for (const std::uint64_t tokens : local_expert_tokens) {
    if (tokens == 0) {
      continue;
    }
    local_routed_tokens += tokens;
    const KernelWork up = execution_time_predictor::gemm_work(
        tokens,
        model.hidden_size,
        local_intermediate,
        element_bytes,
        gated_multiplier);
    grouped_up.flops += up.flops;
    grouped_up.hbm_bytes += up.hbm_bytes;
    const KernelWork down = execution_time_predictor::gemm_work(
        tokens,
        local_intermediate,
        model.hidden_size,
        element_bytes);
    grouped_down.flops += down.flops;
    grouped_down.hbm_bytes += down.hbm_bytes;
  }

  const double tokens = static_cast<double>(input_tokens);
  const double hidden = static_cast<double>(model.hidden_size);
  const double experts = static_cast<double>(model.model_num_experts);
  const double routed = static_cast<double>(local_routed_tokens);
  const double norm_factor = model.fused_add_norm ? 3.0 : 2.0;

  return MoELayerTime{
      .gating_linear_ms = roofline(
          execution_time_predictor::gemm_work(
              input_tokens,
              model.hidden_size,
              model.model_num_experts,
              element_bytes),
          input_tokens < config.small_gemm_token_threshold
              ? config.small_gemm
              : config.large_gemm),
      .gating_routing_topk_ms = roofline(
          execution_time_predictor::streaming_work(
              tokens * experts,
              tokens * static_cast<double>(router_topk),
              4.0 * tokens * experts,
              element_bytes),
          config.routing),
      .grouped_up_projection_ms = roofline(grouped_up, config.moe),
      .grouped_down_projection_ms =
          roofline(grouped_down, config.moe),
      .shuffling_ms = roofline(
          execution_time_predictor::streaming_work(
              routed * hidden,
              routed * hidden,
              0.0,
              element_bytes),
          config.streaming),
      .post_attention_norm_ms = roofline(
          execution_time_predictor::streaming_work(
              tokens * hidden * (norm_factor - 1.0),
              tokens * hidden,
              5.0 * tokens * hidden,
              element_bytes),
          config.streaming),
  };
}

MoELanePrediction predict_moe_lanes(
    const execution_time_predictor::DeviceCeilings& device,
    const execution_time_predictor::AnalyticalConfig& config,
    const MoEModel& model,
    const RoutingAllocation& routing,
    std::uint64_t router_topk,
    execution_time_predictor::Precision precision) {
  if (routing.lane_expert_tokens.empty()) {
    throw AnalyticalModelError("MoE routing must contain at least one lane");
  }
  MoELanePrediction prediction;
  prediction.lane_times.reserve(routing.lane_expert_tokens.size());
  for (const auto& lane : routing.lane_expert_tokens) {
    prediction.lane_times.push_back(predict_moe_layer(
        device,
        config,
        model,
        routing.input_tokens,
        router_topk,
        lane,
        precision));
  }
  for (std::size_t lane = 0; lane < prediction.lane_times.size(); ++lane) {
    const double time = prediction.lane_times[lane].total_ms();
    if (lane == 0 || time > prediction.critical_lane_time_ms) {
      prediction.critical_lane = static_cast<std::uint64_t>(lane);
      prediction.critical_lane_time_ms = time;
    }
  }
  return prediction;
}

MoECommunicationTime predict_moe_communication(
    const cc_backend::BaseCCBackend& communication,
    std::uint64_t input_tokens,
    std::uint64_t hidden_size,
    std::uint64_t routed_tokens,
    std::uint64_t attention_tp_size,
    std::uint64_t moe_tp_size,
    std::uint64_t expert_parallel_size,
    std::uint64_t data_parallel_size,
    bool has_pipeline_boundary,
    double element_bytes) {
  const std::uint64_t activation_bytes =
      payload_bytes(input_tokens, hidden_size, element_bytes);
  const std::uint64_t routed_bytes =
      payload_bytes(routed_tokens, hidden_size, element_bytes);
  return MoECommunicationTime{
      .attention_tp_ms = communication.allreduce_ms(
          activation_bytes, attention_tp_size),
      .moe_tp_ms =
          communication.allreduce_ms(activation_bytes, moe_tp_size),
      .ep_dispatch_ms =
          communication.all_to_all_ms(routed_bytes, expert_parallel_size),
      .ep_combine_ms =
          communication.all_to_all_ms(routed_bytes, expert_parallel_size),
      .dp_input_ms =
          communication.allreduce_ms(activation_bytes, data_parallel_size),
      .dp_output_ms =
          communication.allreduce_ms(activation_bytes, data_parallel_size),
      .pipeline_parallel_ms = has_pipeline_boundary
          ? communication.point_to_point_ms(activation_bytes)
          : 0.0,
  };
}

}  // namespace frontier::moe
