#pragma once

namespace frontier::entities {

struct ExecutionTime {
  double dense_compute_ms = 0.0;
  double tp_communication_ms = 0.0;
  double pp_communication_ms = 0.0;
  double moe_gating_linear_ms = 0.0;
  double moe_gating_routing_topk_ms = 0.0;
  double moe_grouped_gemm_ms = 0.0;
  double moe_shuffling_ms = 0.0;
  double moe_post_attention_norm_ms = 0.0;
  double moe_tp_communication_ms = 0.0;
  double ep_dispatch_ms = 0.0;
  double ep_combine_ms = 0.0;
  double dp_input_communication_ms = 0.0;
  double dp_output_communication_ms = 0.0;
  double synchronization_wait_ms = 0.0;

  [[nodiscard]] double total_ms() const noexcept {
    return dense_compute_ms + tp_communication_ms +
        pp_communication_ms + moe_gating_linear_ms +
        moe_gating_routing_topk_ms + moe_grouped_gemm_ms +
        moe_shuffling_ms + moe_post_attention_norm_ms +
        moe_tp_communication_ms + ep_dispatch_ms +
        ep_combine_ms + dp_input_communication_ms +
        dp_output_communication_ms + synchronization_wait_ms;
  }

  friend bool operator==(
      const ExecutionTime&,
      const ExecutionTime&) = default;
};

}  // namespace frontier::entities
