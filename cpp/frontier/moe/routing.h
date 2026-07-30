#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "frontier/config/config.h"

namespace frontier::moe {

class RoutingError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

struct RoutingAllocation {
  std::uint64_t input_tokens = 0;
  std::uint64_t routed_tokens = 0;
  std::vector<std::uint64_t> global_expert_tokens;
  std::vector<std::vector<std::uint64_t>> lane_expert_tokens;

  friend bool operator==(
      const RoutingAllocation&,
      const RoutingAllocation&) = default;
};

[[nodiscard]] std::vector<std::uint64_t>
discretize_expert_weights(
    std::uint64_t total_tokens,
    const std::vector<double>& weights);

[[nodiscard]] RoutingAllocation route_tokens(
    std::uint64_t input_tokens,
    std::uint64_t router_topk,
    std::uint64_t total_experts,
    std::uint64_t expert_parallel_size,
    const config::MoeRoutingConfig& config,
    std::uint64_t layer_id);

}  // namespace frontier::moe
