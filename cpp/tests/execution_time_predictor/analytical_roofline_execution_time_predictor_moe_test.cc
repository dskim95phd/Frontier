// Internal analytical MoE behavior is tested through the predictor module.
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "frontier/cc_backend/analytical_model.h"
#include "frontier/config/config.h"
#include "frontier/execution_time_predictor/analytical_roofline_execution_time_predictor.h"

namespace {

void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

void test_parallel_domain() {
    const frontier::execution_time_predictor::detail::ExpertParallelDomain
        domain(16, 4);
    require(domain.expert_range(2) ==
                frontier::execution_time_predictor::detail::ExpertRange{8, 12},
            "EP expert range must be contiguous");
    require(domain.owner(15) == 3, "expert owner must match range");

    std::vector<std::uint64_t> counts(16);
    std::iota(counts.begin(), counts.end(), 0);
    const auto lanes = domain.partition(counts);
    require(lanes.size() == 4, "EP lane count mismatch");
    require(lanes[2] == std::vector<std::uint64_t>({8, 9, 10, 11}),
            "EP lane-local allocation mismatch");
}

void test_deterministic_distributions() {
    using frontier::config::MoeRoutingConfig;
    using frontier::config::MoeRoutingDistribution;
    using frontier::config::MoeRoutingMode;

    const auto balanced =
        frontier::execution_time_predictor::detail::route_tokens(
            19, 2, 8, 4,
            [&]() {
                MoeRoutingConfig value{};
                value.mode = MoeRoutingMode::kSimulation;
                value.distribution = MoeRoutingDistribution::kBalanced;
                value.seed = 42;
                return value;
            }(),
            0);
    require(balanced.global_expert_tokens ==
                std::vector<std::uint64_t>({5, 5, 5, 5, 5, 5, 4, 4}),
            "balanced largest-remainder routing mismatch");
    require(balanced.routed_tokens == 38, "routed tokens must include top-k");

    const auto legacy =
        frontier::execution_time_predictor::detail::route_tokens(
            19, 2, 8, 2,
            [&]() {
                MoeRoutingConfig value{};
                value.mode = MoeRoutingMode::kUniformLegacy;
                value.distribution = MoeRoutingDistribution::kZipf;
                value.seed = 999;
                return value;
            }(),
            31);
    require(legacy.global_expert_tokens == balanced.global_expert_tokens,
            "uniform legacy routing must use exact quotient/remainder");

    const auto skewed =
        frontier::execution_time_predictor::detail::route_tokens(
            37, 1, 16, 4,
            [&]() {
                MoeRoutingConfig value{};
                value.mode = MoeRoutingMode::kSimulation;
                value.distribution = MoeRoutingDistribution::kSkewed;
                value.seed = 42;
                return value;
            }(),
            0);
    require(std::accumulate(skewed.global_expert_tokens.begin(),
                            skewed.global_expert_tokens.end(),
                            std::uint64_t{0}) == 37,
            "skewed routing must conserve tokens");
    require(skewed.global_expert_tokens.front() >
                skewed.global_expert_tokens.back(),
            "skewed routing must favor low-rank experts");
}

void test_numpy_random_golden_vectors() {
    using frontier::config::MoeRoutingConfig;
    using frontier::config::MoeRoutingDistribution;
    using frontier::config::MoeRoutingMode;

    const auto simulation_random =
        frontier::execution_time_predictor::detail::route_tokens(
            37, 1, 16, 4,
            [&]() {
                MoeRoutingConfig value{};
                value.mode = MoeRoutingMode::kSimulation;
                value.distribution = MoeRoutingDistribution::kRandom;
                value.seed = 42;
                return value;
            }(),
            0);
    require(simulation_random.global_expert_tokens ==
                std::vector<std::uint64_t>(
                    {3, 2, 3, 3, 1, 4, 3, 3, 1, 2, 1, 3, 2, 3, 2, 1}),
            "NumPy default_rng PCG64 random-routing vector mismatch");

    const auto uniform_random =
        frontier::execution_time_predictor::detail::route_tokens(
            37, 1, 16, 4,
            [&]() {
                MoeRoutingConfig value{};
                value.mode = MoeRoutingMode::kUniformRandom;
                value.distribution = MoeRoutingDistribution::kBalanced;
                value.seed = 42;
                return value;
            }(),
            0);
    require(uniform_random.global_expert_tokens ==
                std::vector<std::uint64_t>(
                    {0, 5, 2, 2, 0, 1, 2, 4, 5, 0, 2, 3, 4, 4, 2, 1}),
            "NumPy default_rng PCG64 uniform-random vector mismatch");

    const auto second_layer =
        frontier::execution_time_predictor::detail::route_tokens(
            37, 1, 16, 4,
            [&]() {
                MoeRoutingConfig value{};
                value.mode = MoeRoutingMode::kUniformRandom;
                value.distribution = MoeRoutingDistribution::kBalanced;
                value.seed = 123456789;
                return value;
            }(),
            31);
    require(second_layer.global_expert_tokens ==
                std::vector<std::uint64_t>(
                    {1, 3, 4, 0, 2, 2, 1, 0, 3, 4, 3, 3, 4, 1, 1, 5}),
            "PCG64 multi-seed/multi-layer golden vector mismatch");
}

void test_moe_lane_analytical_model() {
    using frontier::config::MoeRoutingConfig;
    using frontier::config::MoeRoutingDistribution;
    using frontier::config::MoeRoutingMode;
    using frontier::execution_time_predictor::detail::AnalyticalConfig;
    using frontier::execution_time_predictor::detail::DeviceCeilings;
    using frontier::execution_time_predictor::detail::Precision;

    const auto routing =
        frontier::execution_time_predictor::detail::route_tokens(
            128, 2, 16, 4,
            [&]() {
                MoeRoutingConfig value{};
                value.mode = MoeRoutingMode::kSimulation;
                value.distribution = MoeRoutingDistribution::kZipf;
                value.seed = 42;
                return value;
            }(),
            7);
    const auto prediction =
        frontier::execution_time_predictor::detail::predict_moe_lanes(
            DeviceCeilings::rubin(), AnalyticalConfig{},
            [&]() {
                frontier::execution_time_predictor::detail::MoEModel value{};
                value.hidden_size = 4096;
                value.intermediate_size = 448;
                value.model_num_experts = 16;
                value.moe_tensor_parallel_size = 2;
                value.gated_mlp = true;
                value.fused_add_norm = false;
                return value;
            }(),
            routing, 2, Precision::kBf16);
    require(prediction.lane_times.size() == 4,
            "analytical MoE must predict every EP lane");
    require(prediction.critical_lane == 0,
            "Zipf allocation must select the hot first EP lane");
    require(prediction.critical_lane_time_ms ==
                prediction.lane_times.front().total_ms(),
            "critical-lane time must be the lane maximum");
    require(prediction.lane_times.front().grouped_up_projection_ms >
                prediction.lane_times.back().grouped_up_projection_ms,
            "lane-local grouped GEMM must reflect routing imbalance");

    const frontier::cc_backend::AnalyticalCommunicationModel communication(
        [&]() {
            frontier::cc_backend::AnalyticalCommunicationConfig value{};
            value.network_bandwidth_gbps = 400.0;
            value.latency_us = 1.0;
            value.intra_node_bandwidth_gbps = 14'400.0;
            return value;
        }());
    const auto local =
        frontier::execution_time_predictor::detail::predict_moe_communication(
            communication, 128, 4096, 256, 1, 1, 1, 1, false, 2.0);
    require(local.total_ms() == 0.0,
            "single-device MoE communication must be skipped");
    const auto distributed =
        frontier::execution_time_predictor::detail::predict_moe_communication(
            communication, 128, 4096, 256, 2, 2, 4, 2, true, 2.0);
    require(distributed.attention_tp_ms > 0.0 && distributed.moe_tp_ms > 0.0 &&
                distributed.ep_dispatch_ms > 0.0 &&
                distributed.ep_combine_ms > 0.0 &&
                distributed.dp_input_ms > 0.0 &&
                distributed.dp_output_ms > 0.0 &&
                distributed.pipeline_parallel_ms > 0.0,
            "distributed MoE must retain distinct communication terms");
}

void test_moe_overflow_and_nonfinite_inputs_fail_fast() {
    using frontier::config::MoeRoutingConfig;
    bool routing_overflow_rejected = false;
    try {
        static_cast<void>(
            frontier::execution_time_predictor::detail::route_tokens(
                std::numeric_limits<std::uint64_t>::max(), 2, 16, 4,
                MoeRoutingConfig{}, 0));
    } catch (const frontier::execution_time_predictor::detail::RoutingError &) {
        routing_overflow_rejected = true;
    }
    require(routing_overflow_rejected, "routed-token overflow must fail fast");

    const frontier::cc_backend::AnalyticalCommunicationModel communication(
        [&]() {
            frontier::cc_backend::AnalyticalCommunicationConfig value{};
            value.network_bandwidth_gbps = 400.0;
            value.latency_us = 1.0;
            value.intra_node_bandwidth_gbps = 14'400.0;
            return value;
        }());
    bool nonfinite_payload_rejected = false;
    try {
        static_cast<void>(frontier::execution_time_predictor::detail::
                              predict_moe_communication(
                                  communication, 1, 4096, 2, 1, 1, 1, 1, false,
                                  std::numeric_limits<double>::quiet_NaN()));
    } catch (
        const frontier::execution_time_predictor::detail::AnalyticalModelError
            &) {
        nonfinite_payload_rejected = true;
    }
    require(nonfinite_payload_rejected,
            "nonfinite MoE communication payload must fail fast");

    bool payload_overflow_rejected = false;
    try {
        static_cast<void>(
            frontier::execution_time_predictor::detail::
                predict_moe_communication(
                    communication, std::numeric_limits<std::uint64_t>::max(),
                    std::numeric_limits<std::uint64_t>::max(), 1, 1, 1, 1, 1,
                    false, 2.0));
    } catch (
        const frontier::execution_time_predictor::detail::AnalyticalModelError
            &) {
        payload_overflow_rejected = true;
    }
    require(payload_overflow_rejected,
            "MoE communication payload overflow must fail fast");
}

} // namespace

int main() {
    try {
        test_parallel_domain();
        test_deterministic_distributions();
        test_numpy_random_golden_vectors();
        test_moe_lane_analytical_model();
        test_moe_overflow_and_nonfinite_inputs_fail_fast();
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
