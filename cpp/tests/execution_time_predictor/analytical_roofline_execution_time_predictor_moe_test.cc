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

// Routing reproducibility regression.  These vectors are not tied to any
// external library; they pin the simulator's own routing engine so a given
// seed keeps producing the same expert allocation across releases and
// across platforms.  Regenerate them only for a deliberate, documented
// routing change.
void test_routing_reproducibility_golden_vectors() {
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
                    {3, 3, 3, 1, 4, 1, 2, 2, 1, 2, 0, 2, 3, 3, 3, 4}),
            "random-distribution routing vector is not reproducible");

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
                    {3, 2, 2, 0, 2, 2, 5, 4, 3, 3, 2, 2, 4, 1, 2, 0}),
            "uniform-random routing vector is not reproducible");

    // A different seed and a different layer id must both perturb routing.
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
                    {2, 1, 2, 1, 1, 1, 3, 1, 8, 4, 2, 2, 3, 2, 2, 2}),
            "multi-seed/multi-layer routing vector is not reproducible");

    // Every routed token must land on exactly one expert.
    for (const auto *allocation :
         {&simulation_random, &uniform_random, &second_layer}) {
        std::uint64_t total = 0;
        for (const std::uint64_t count : allocation->global_expert_tokens) {
            total += count;
        }
        require(total == allocation->routed_tokens,
                "routing must conserve routed tokens");
    }
}

void test_uniform_random_topk_is_distinct_per_input_token() {
    frontier::config::MoeRoutingConfig routing{};
    routing.mode = frontier::config::MoeRoutingMode::kUniformRandom;
    routing.seed = 42;

    constexpr std::uint64_t kInputTokens = 31;
    constexpr std::uint64_t kTopK = 8;
    const auto allocation =
        frontier::execution_time_predictor::detail::route_tokens(
            kInputTokens, kTopK, 16, 4, routing, 0);

    require(std::accumulate(allocation.global_expert_tokens.begin(),
                            allocation.global_expert_tokens.end(),
                            std::uint64_t{0}) == kInputTokens * kTopK,
            "uniform top-k routing must conserve all selections");
    require(std::all_of(allocation.global_expert_tokens.begin(),
                        allocation.global_expert_tokens.end(),
                        [](std::uint64_t count) {
                            return count <= kInputTokens;
                        }),
            "one input token must not select the same expert twice");

    // A one-token prefill-sized draw makes the distinctness contract directly
    // observable from the aggregate histogram: exactly k experts have count 1.
    const auto one_token =
        frontier::execution_time_predictor::detail::route_tokens(
            1, kTopK, 16, 4, routing, 0);
    require(std::count(one_token.global_expert_tokens.begin(),
                       one_token.global_expert_tokens.end(),
                       std::uint64_t{1}) == kTopK &&
                std::all_of(one_token.global_expert_tokens.begin(),
                            one_token.global_expert_tokens.end(),
                            [](std::uint64_t count) { return count <= 1; }),
            "one input token must route to k distinct experts");
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

void test_shared_expert_is_replicated_across_ep_and_sharded_by_tp() {
    using frontier::execution_time_predictor::detail::AnalyticalConfig;
    using frontier::execution_time_predictor::detail::DeviceCeilings;
    using frontier::execution_time_predictor::detail::MoEModel;
    using frontier::execution_time_predictor::detail::Precision;

    const std::vector<std::uint64_t> cold_lane = {0, 0};
    MoEModel without_shared{};
    without_shared.hidden_size = 4'096;
    without_shared.intermediate_size = 2'048;
    without_shared.model_num_experts = 4;
    without_shared.moe_tensor_parallel_size = 1;
    without_shared.gated_mlp = true;

    MoEModel tp1 = without_shared;
    tp1.num_shared_experts = 1;
    const auto no_shared =
        frontier::execution_time_predictor::detail::predict_moe_layer(
            DeviceCeilings::rubin(), AnalyticalConfig{}, without_shared, 128, 1,
            cold_lane, Precision::kFp8);
    const auto replicated_tp1 =
        frontier::execution_time_predictor::detail::predict_moe_layer(
            DeviceCeilings::rubin(), AnalyticalConfig{}, tp1, 128, 1, cold_lane,
            Precision::kFp8);
    require(replicated_tp1.grouped_up_projection_ms >
                    no_shared.grouped_up_projection_ms &&
                replicated_tp1.grouped_down_projection_ms >
                    no_shared.grouped_down_projection_ms,
            "a shared expert must execute even on an EP lane with no routed "
            "tokens");
    require(replicated_tp1.shuffling_ms == no_shared.shuffling_ms,
            "a replicated shared expert must not add routed-token shuffle "
            "work");

    MoEModel tp2 = tp1;
    tp2.moe_tensor_parallel_size = 2;
    const auto replicated_tp2 =
        frontier::execution_time_predictor::detail::predict_moe_layer(
            DeviceCeilings::rubin(), AnalyticalConfig{}, tp2, 128, 1, cold_lane,
            Precision::kFp8);
    require(replicated_tp2.grouped_up_projection_ms <
                    replicated_tp1.grouped_up_projection_ms &&
                replicated_tp2.grouped_down_projection_ms <
                    replicated_tp1.grouped_down_projection_ms,
            "MoE TP must shard the replicated shared expert");
}

void test_latent_moe_projection_precision_is_independent() {
    using frontier::execution_time_predictor::detail::AnalyticalConfig;
    using frontier::execution_time_predictor::detail::DeviceCeilings;
    using frontier::execution_time_predictor::detail::MoEModel;
    using frontier::execution_time_predictor::detail::MoEOperatorPrecisions;
    using frontier::execution_time_predictor::detail::Precision;

    MoEModel latent{};
    latent.hidden_size = 7'168;
    latent.intermediate_size = 2'048;
    latent.model_num_experts = 4;
    latent.moe_tensor_parallel_size = 1;
    latent.routed_expert_hidden_size = 3'584;
    latent.gated_mlp = true;
    const std::vector<std::uint64_t> local_expert_tokens = {64, 64, 64, 64};

    MoEOperatorPrecisions bf16_projection{};
    bf16_projection.expert = Precision::kFp16;
    bf16_projection.router = Precision::kFp16;
    bf16_projection.dense = Precision::kFp16;
    bf16_projection.shared_expert = Precision::kFp16;
    bf16_projection.expert_weight = Precision::kFp4;
    bf16_projection.expert_activation = Precision::kFp8;
    bf16_projection.latent_moe_projection_weight = Precision::kBf16;
    bf16_projection.latent_moe_projection_activation = Precision::kBf16;

    auto quantized_projection = bf16_projection;
    quantized_projection.latent_moe_projection_weight = Precision::kFp4;
    quantized_projection.latent_moe_projection_activation = Precision::kFp8;
    const auto bf16 =
        frontier::execution_time_predictor::detail::predict_moe_layer(
            DeviceCeilings::rubin(), AnalyticalConfig{}, latent, 128, 2,
            local_expert_tokens, bf16_projection);
    const auto quantized =
        frontier::execution_time_predictor::detail::predict_moe_layer(
            DeviceCeilings::rubin(), AnalyticalConfig{}, latent, 128, 2,
            local_expert_tokens, quantized_projection);
    require(bf16.latent_projection_ms > quantized.latent_projection_ms,
            "BF16 Stable LatentMoE projections must use their own HBM and "
            "compute roofline");
    require(bf16.grouped_up_projection_ms ==
                    quantized.grouped_up_projection_ms &&
                bf16.grouped_down_projection_ms ==
                    quantized.grouped_down_projection_ms,
            "Stable LatentMoE projection precision must not change routed "
            "expert kernels");

    MoEOperatorPrecisions legacy = bf16_projection;
    legacy.latent_moe_projection_weight.reset();
    legacy.latent_moe_projection_activation.reset();
    const auto legacy_prediction =
        frontier::execution_time_predictor::detail::predict_moe_layer(
            DeviceCeilings::rubin(), AnalyticalConfig{}, latent, 128, 2,
            local_expert_tokens, legacy);
    require(legacy_prediction.latent_projection_ms ==
                quantized.latent_projection_ms,
            "legacy latent-MoE callers must inherit routed-expert precision "
            "when projection fields are absent");

    auto non_latent = latent;
    non_latent.routed_expert_hidden_size = 0;
    const auto non_latent_bf16 =
        frontier::execution_time_predictor::detail::predict_moe_layer(
            DeviceCeilings::rubin(), AnalyticalConfig{}, non_latent, 128, 2,
            local_expert_tokens, bf16_projection);
    const auto non_latent_quantized =
        frontier::execution_time_predictor::detail::predict_moe_layer(
            DeviceCeilings::rubin(), AnalyticalConfig{}, non_latent, 128, 2,
            local_expert_tokens, quantized_projection);
    require(non_latent_bf16.total_ms() == non_latent_quantized.total_ms() &&
                non_latent_bf16.latent_projection_ms == 0.0,
            "non-latent MoE execution must ignore latent projection "
            "precision overrides");
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
        test_routing_reproducibility_golden_vectors();
        test_uniform_random_topk_is_distinct_per_input_token();
        test_moe_lane_analytical_model();
        test_shared_expert_is_replicated_across_ep_and_sharded_by_tp();
        test_latent_moe_projection_precision_is_independent();
        test_moe_overflow_and_nonfinite_inputs_fail_fast();
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
