// Deterministic MoE token routing and its reproducible RNG.
//
// MoE routing implementation. Public internal contracts live in
// analytical_moe_model.h.

#include "frontier/execution_time_predictor/analytical_moe_model.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace frontier::execution_time_predictor::detail {
namespace {

// Deterministic routing RNG.  The engine and every conversion below are
// either specified by the C++ standard or implemented here, so one seed
// yields the same routing on every platform and standard library.
// std::uniform_*_distribution is deliberately avoided: its output sequence
// is implementation-defined and would break that guarantee.
class RoutingRng {
  public:
    explicit RoutingRng(std::uint64_t seed) : engine_(seed) {}

    // Uniform in [0, 1) using the top 53 bits, matching double's mantissa.
    double next_double() {
        return static_cast<double>(engine_() >> 11U) *
               (1.0 / 9007199254740992.0);
    }

    // Uniform in [0, exclusive_upper).  Values below the rejection
    // threshold are discarded so ranges that do not divide 2^64 stay
    // unbiased; that threshold is 2^64 mod range in unsigned arithmetic.
    std::uint64_t bounded(std::uint64_t exclusive_upper) {
        if (exclusive_upper == 0) {
            throw RoutingError("bounded routing range must be positive");
        }
        if (exclusive_upper == 1) {
            return 0;
        }
        const std::uint64_t reject_below =
            (0ULL - exclusive_upper) % exclusive_upper;
        for (;;) {
            const std::uint64_t value = engine_();
            if (value >= reject_below) {
                return value % exclusive_upper;
            }
        }
    }

  private:
    std::mt19937_64 engine_;
};

std::vector<double>
distribution_weights(std::uint64_t experts,
                     config::MoeRoutingDistribution distribution,
                     std::uint64_t seed) {
    std::vector<double> weights(static_cast<std::size_t>(experts), 1.0);
    switch (distribution) {
    case config::MoeRoutingDistribution::kBalanced:
        break;
    case config::MoeRoutingDistribution::kRandom: {
        RoutingRng generator(seed);
        for (double &weight : weights) {
            weight = 0.1 + 0.9 * generator.next_double();
        }
        break;
    }
    case config::MoeRoutingDistribution::kSkewed:
        for (std::size_t i = 0; i < weights.size(); ++i) {
            weights[i] = 1.0 / std::pow(static_cast<double>(i + 1), 0.35);
        }
        break;
    case config::MoeRoutingDistribution::kZipf:
        for (std::size_t i = 0; i < weights.size(); ++i) {
            weights[i] = 1.0 / static_cast<double>(i + 1);
        }
        break;
    }
    return weights;
}

void accumulate_uniform_topk_counts(std::uint64_t input_tokens,
                                    std::uint64_t router_topk,
                                    std::uint64_t total_experts,
                                    RoutingRng &generator,
                                    std::vector<std::uint64_t> &counts) {
    // A router selects a set of k distinct experts for each input token. Keep
    // only the aggregate expert loads needed by the timing model, but obtain
    // them from a real without-replacement top-k draw. The partial
    // Fisher-Yates shuffle costs O(input_tokens * topk), and undoing its swaps
    // leaves the expert pool canonical for the next token without an
    // O(total_experts) reset.
    std::vector<std::uint64_t> expert_pool(
        static_cast<std::size_t>(total_experts));
    std::iota(expert_pool.begin(), expert_pool.end(), 0);
    std::vector<std::uint64_t> swap_positions(
        static_cast<std::size_t>(router_topk));

    for (std::uint64_t token = 0; token < input_tokens; ++token) {
        for (std::uint64_t pick = 0; pick < router_topk; ++pick) {
            const std::uint64_t swap_position =
                pick + generator.bounded(total_experts - pick);
            swap_positions[static_cast<std::size_t>(pick)] = swap_position;
            std::swap(expert_pool[static_cast<std::size_t>(pick)],
                      expert_pool[static_cast<std::size_t>(swap_position)]);
            ++counts[static_cast<std::size_t>(
                expert_pool[static_cast<std::size_t>(pick)])];
        }
        for (std::uint64_t pick = router_topk; pick > 0; --pick) {
            const std::uint64_t index = pick - 1;
            const std::uint64_t swap_position =
                swap_positions[static_cast<std::size_t>(index)];
            std::swap(expert_pool[static_cast<std::size_t>(index)],
                      expert_pool[static_cast<std::size_t>(swap_position)]);
        }
    }
}

} // namespace

std::vector<std::uint64_t>
discretize_expert_weights(std::uint64_t total_tokens,
                          const std::vector<double> &weights) {
    if (weights.empty()) {
        throw RoutingError("expert weights must not be empty");
    }
    double total_weight = 0.0;
    for (const double weight : weights) {
        if (!std::isfinite(weight) || weight < 0.0) {
            throw RoutingError("expert weights must be finite and nonnegative");
        }
        total_weight += weight;
    }
    if (!std::isfinite(total_weight) || total_weight <= 0.0) {
        throw RoutingError("expert weights must have a positive sum");
    }

    std::vector<std::uint64_t> result(weights.size(), 0);
    std::vector<double> normalized(weights.size(), 0.0);
    std::vector<double> fractional(weights.size(), 0.0);
    std::uint64_t allocated = 0;
    for (std::size_t i = 0; i < weights.size(); ++i) {
        normalized[i] = weights[i] / total_weight;
        const double exact = static_cast<double>(total_tokens) * normalized[i];
        result[i] = static_cast<std::uint64_t>(exact);
        fractional[i] = exact - static_cast<double>(result[i]);
        allocated += result[i];
    }

    std::vector<std::size_t> order(weights.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](std::size_t lhs, std::size_t rhs) {
                  if (fractional[lhs] != fractional[rhs]) {
                      return fractional[lhs] > fractional[rhs];
                  }
                  if (normalized[lhs] != normalized[rhs]) {
                      return normalized[lhs] > normalized[rhs];
                  }
                  return lhs < rhs;
              });
    std::uint64_t remainder_index = 0;
    while (allocated < total_tokens) {
        ++result[order[static_cast<std::size_t>(remainder_index %
                                                order.size())]];
        ++allocated;
        ++remainder_index;
    }
    return result;
}

RoutingAllocation
route_tokens(std::uint64_t input_tokens, std::uint64_t router_topk,
             std::uint64_t total_experts, std::uint64_t expert_parallel_size,
             const config::MoeRoutingConfig &config, std::uint64_t layer_id) {
    if (router_topk == 0 || router_topk > total_experts) {
        throw RoutingError("router top-k must be in [1, total experts]");
    }
    if (input_tokens >
        std::numeric_limits<std::uint64_t>::max() / router_topk) {
        throw RoutingError("routed token count overflows uint64");
    }

    ExpertParallelDomain domain(total_experts, expert_parallel_size);
    const std::uint64_t routed_tokens = input_tokens * router_topk;
    std::vector<std::uint64_t> counts(static_cast<std::size_t>(total_experts),
                                      0);

    if (config.mode == config::MoeRoutingMode::kUniformLegacy) {
        const std::uint64_t base = routed_tokens / total_experts;
        const std::uint64_t remainder = routed_tokens % total_experts;
        for (std::uint64_t expert = 0; expert < total_experts; ++expert) {
            counts[static_cast<std::size_t>(expert)] =
                base + static_cast<std::uint64_t>(expert < remainder);
        }
    } else if (config.mode == config::MoeRoutingMode::kUniformRandom) {
        RoutingRng generator(config.seed + layer_id);
        accumulate_uniform_topk_counts(input_tokens, router_topk, total_experts,
                                       generator, counts);
    } else {
        counts = discretize_expert_weights(
            routed_tokens,
            distribution_weights(total_experts, config.distribution,
                                 config.seed + layer_id));
    }

    return [&]() {
        RoutingAllocation value{};
        value.input_tokens = input_tokens;
        value.routed_tokens = routed_tokens;
        value.global_expert_tokens = counts;
        value.lane_expert_tokens = domain.partition(counts);
        return value;
    }();
}

} // namespace frontier::execution_time_predictor::detail
