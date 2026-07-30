#include "frontier/execution_time_predictor/analytical_roofline_execution_time_predictor.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <utility>

namespace frontier::execution_time_predictor::detail {
namespace {

constexpr std::uint32_t kInitA = 0x43b0d7e5U;
constexpr std::uint32_t kMultA = 0x931e8875U;
constexpr std::uint32_t kInitB = 0x8b51f9ddU;
constexpr std::uint32_t kMultB = 0x58f38dedU;
constexpr std::uint32_t kMixMultL = 0xca01f9ddU;
constexpr std::uint32_t kMixMultR = 0x4973f715U;

constexpr std::uint64_t rotate_right(std::uint64_t value,
                                     unsigned int shift) noexcept {
    shift &= 63U;
    return shift == 0U ? value : (value >> shift) | (value << (64U - shift));
}

std::uint32_t hashmix(std::uint32_t value, std::uint32_t &hash_constant) {
    value ^= hash_constant;
    hash_constant *= kMultA;
    value *= hash_constant;
    value ^= value >> 16U;
    return value;
}

std::uint32_t mix(std::uint32_t x, std::uint32_t y) {
    std::uint32_t result = kMixMultL * x - kMixMultR * y;
    result ^= result >> 16U;
    return result;
}

std::array<std::uint64_t, 4> seed_sequence_state(std::uint64_t seed) {
    std::vector<std::uint32_t> entropy;
    entropy.push_back(static_cast<std::uint32_t>(seed));
    if (seed > std::numeric_limits<std::uint32_t>::max()) {
        entropy.push_back(static_cast<std::uint32_t>(seed >> 32U));
    }

    std::array<std::uint32_t, 4> pool{};
    std::uint32_t hash_constant = kInitA;
    for (std::size_t i = 0; i < pool.size(); ++i) {
        pool[i] = hashmix(i < entropy.size() ? entropy[i] : 0U, hash_constant);
    }
    for (std::size_t source = 0; source < pool.size(); ++source) {
        for (std::size_t destination = 0; destination < pool.size();
             ++destination) {
            if (source != destination) {
                pool[destination] = mix(pool[destination],
                                        hashmix(pool[source], hash_constant));
            }
        }
    }
    for (std::size_t source = pool.size(); source < entropy.size(); ++source) {
        for (std::size_t destination = 0; destination < pool.size();
             ++destination) {
            pool[destination] =
                mix(pool[destination], hashmix(entropy[source], hash_constant));
        }
    }

    std::array<std::uint32_t, 8> words{};
    hash_constant = kInitB;
    for (std::size_t i = 0; i < words.size(); ++i) {
        std::uint32_t value = pool[i % pool.size()];
        value ^= hash_constant;
        hash_constant *= kMultB;
        value *= hash_constant;
        value ^= value >> 16U;
        words[i] = value;
    }

    std::array<std::uint64_t, 4> result{};
    for (std::size_t i = 0; i < result.size(); ++i) {
        result[i] = static_cast<std::uint64_t>(words[i * 2]) |
                    (static_cast<std::uint64_t>(words[i * 2 + 1]) << 32U);
    }
    return result;
}

struct Uint128 {
    std::uint64_t high = 0;
    std::uint64_t low = 0;
};

Uint128 add128(Uint128 lhs, Uint128 rhs) {
    const std::uint64_t low = lhs.low + rhs.low;
    return Uint128{
        lhs.high + rhs.high + static_cast<std::uint64_t>(low < rhs.low), low};
}

std::pair<std::uint64_t, std::uint64_t> multiply64(std::uint64_t lhs,
                                                   std::uint64_t rhs) {
    const std::uint64_t lhs_low = lhs & 0xffffffffULL;
    const std::uint64_t lhs_high = lhs >> 32U;
    const std::uint64_t rhs_low = rhs & 0xffffffffULL;
    const std::uint64_t rhs_high = rhs >> 32U;
    const std::uint64_t word0 = lhs_low * rhs_low;
    const std::uint64_t temp = lhs_high * rhs_low + (word0 >> 32U);
    std::uint64_t word1 = temp & 0xffffffffULL;
    const std::uint64_t word2 = temp >> 32U;
    word1 += lhs_low * rhs_high;
    const std::uint64_t high = lhs_high * rhs_high + word2 + (word1 >> 32U);
    return {high, lhs * rhs};
}

Uint128 multiply128(Uint128 lhs, Uint128 rhs) {
    auto [high, low] = multiply64(lhs.low, rhs.low);
    high += lhs.high * rhs.low + lhs.low * rhs.high;
    return Uint128{high, low};
}

class NumpyPcg64 {
  public:
    explicit NumpyPcg64(std::uint64_t seed) {
        const auto generated = seed_sequence_state(seed);
        seed_state(Uint128{generated[0], generated[1]},
                   Uint128{generated[2], generated[3]});
    }

    std::uint64_t next64() {
        step();
        return rotate_right(state_.high ^ state_.low,
                            static_cast<unsigned int>(state_.high >> 58U));
    }

    std::uint32_t next32() {
        if (has_uint32_) {
            has_uint32_ = false;
            return buffered_uint32_;
        }
        const std::uint64_t value = next64();
        has_uint32_ = true;
        buffered_uint32_ = static_cast<std::uint32_t>(value >> 32U);
        return static_cast<std::uint32_t>(value);
    }

    std::uint32_t bounded32(std::uint32_t exclusive_upper) {
        if (exclusive_upper == 0) {
            throw RoutingError("bounded PCG64 range must be positive");
        }
        const std::uint32_t inclusive_range = exclusive_upper - 1U;
        if (inclusive_range == 0) {
            return 0;
        }
        const std::uint64_t range = exclusive_upper;
        for (;;) {
            const std::uint64_t product =
                static_cast<std::uint64_t>(next32()) * range;
            const std::uint32_t leftover = static_cast<std::uint32_t>(product);
            if (leftover < range) {
                const std::uint32_t threshold =
                    (std::numeric_limits<std::uint32_t>::max() -
                     inclusive_range) %
                    exclusive_upper;
                if (leftover < threshold) {
                    continue;
                }
            }
            return static_cast<std::uint32_t>(product >> 32U);
        }
    }

    double next_double() {
        return static_cast<double>(next64() >> 11U) *
               (1.0 / 9007199254740992.0);
    }

  private:
    void step() {
        constexpr Uint128 multiplier{2549297995355413924ULL,
                                     4865540595714422341ULL};
        state_ = add128(multiply128(state_, multiplier), increment_);
    }

    void seed_state(Uint128 initial_state, Uint128 initial_sequence) {
        state_ = Uint128{};
        increment_.high =
            (initial_sequence.high << 1U) | (initial_sequence.low >> 63U);
        increment_.low = (initial_sequence.low << 1U) | 1U;
        step();
        state_ = add128(state_, initial_state);
        step();
    }

    Uint128 state_;
    Uint128 increment_;
    bool has_uint32_ = false;
    std::uint32_t buffered_uint32_ = 0;
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
        NumpyPcg64 generator(seed);
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
        NumpyPcg64 generator(config.seed + layer_id);
        for (std::uint64_t token = 0; token < routed_tokens; ++token) {
            const std::uint32_t expert =
                generator.bounded32(static_cast<std::uint32_t>(total_experts));
            ++counts[expert];
        }
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
