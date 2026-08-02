#pragma once

#include <tuple>

namespace frontier::entities {

struct ExecutionTime {
    double dense_compute_ms = 0.0;
    double lm_head_ms = 0.0;
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
        return dense_compute_ms + lm_head_ms + tp_communication_ms +
               pp_communication_ms + moe_gating_linear_ms +
               moe_gating_routing_topk_ms + moe_grouped_gemm_ms +
               moe_shuffling_ms + moe_post_attention_norm_ms +
               moe_tp_communication_ms + ep_dispatch_ms + ep_combine_ms +
               dp_input_communication_ms + dp_output_communication_ms +
               synchronization_wait_ms;
    }

    friend bool operator==(const ExecutionTime &lhs, const ExecutionTime &rhs) {
        return std::tie(lhs.dense_compute_ms, lhs.lm_head_ms,
                        lhs.tp_communication_ms, lhs.pp_communication_ms,
                        lhs.moe_gating_linear_ms,
                        lhs.moe_gating_routing_topk_ms, lhs.moe_grouped_gemm_ms,
                        lhs.moe_shuffling_ms, lhs.moe_post_attention_norm_ms,
                        lhs.moe_tp_communication_ms, lhs.ep_dispatch_ms,
                        lhs.ep_combine_ms, lhs.dp_input_communication_ms,
                        lhs.dp_output_communication_ms,
                        lhs.synchronization_wait_ms) ==
               std::tie(rhs.dense_compute_ms, rhs.lm_head_ms,
                        rhs.tp_communication_ms, rhs.pp_communication_ms,
                        rhs.moe_gating_linear_ms,
                        rhs.moe_gating_routing_topk_ms, rhs.moe_grouped_gemm_ms,
                        rhs.moe_shuffling_ms, rhs.moe_post_attention_norm_ms,
                        rhs.moe_tp_communication_ms, rhs.ep_dispatch_ms,
                        rhs.ep_combine_ms, rhs.dp_input_communication_ms,
                        rhs.dp_output_communication_ms,
                        rhs.synchronization_wait_ms);
    }
};

} // namespace frontier::entities
