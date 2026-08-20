// Internal analytical MoE behavior is tested through the predictor module.
#include <cmath>
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
    require(
        std::all_of(allocation.global_expert_tokens.begin(),
                    allocation.global_expert_tokens.end(),
                    [](std::uint64_t count) { return count <= kInputTokens; }),
        "one input token must not select the same expert twice");

    // A one-token prefill-sized draw makes the distinctness contract directly
    // observable from the aggregate histogram: exactly k experts have count 1.
    const auto one_token =
        frontier::execution_time_predictor::detail::route_tokens(1, kTopK, 16,
                                                                 4, routing, 0);
    require(std::count(one_token.global_expert_tokens.begin(),
                       one_token.global_expert_tokens.end(),
                       std::uint64_t{1}) == kTopK &&
                std::all_of(one_token.global_expert_tokens.begin(),
                            one_token.global_expert_tokens.end(),
                            [](std::uint64_t count) { return count <= 1; }),
            "one input token must route to k distinct experts");
}

void test_uniform_routing_varies_by_batch_and_reports_lane_traffic() {
    frontier::config::MoeRoutingConfig routing{};
    routing.mode = frontier::config::MoeRoutingMode::kUniformRandom;
    routing.layer_scope = frontier::config::MoeRoutingLayerScope::kShared;
    routing.seed = 42;

    const auto first = frontier::execution_time_predictor::detail::route_tokens(
        64, 8, 256, 32, routing, 7, 1001);
    const auto repeated =
        frontier::execution_time_predictor::detail::route_tokens(
            64, 8, 256, 32, routing, 7, 1001);
    const auto next_batch =
        frontier::execution_time_predictor::detail::route_tokens(
            64, 8, 256, 32, routing, 7, 1002);

    require(first == repeated,
            "uniform routing must be deterministic for one batch/layer");
    require(first.global_expert_tokens != next_batch.global_expert_tokens,
            "different batches must not repeat one routing draw");
    require(first.lane_routed_tokens.size() == 32 &&
                first.lane_active_experts.size() == 32 &&
                first.lane_unique_tokens.size() == 32,
            "routing must expose one traffic summary per EP lane");
    require(std::accumulate(first.lane_routed_tokens.begin(),
                            first.lane_routed_tokens.end(),
                            std::uint64_t{0}) == first.routed_tokens,
            "lane routed-token summaries must conserve assignments");
    require(std::all_of(first.lane_unique_tokens.begin(),
                        first.lane_unique_tokens.end(),
                        [](std::uint64_t value) { return value <= 64; }),
            "one source token must count at most once per destination lane");
}

void test_sm100_megamoe_public_profile_models_overlap_and_layout_transition() {
    frontier::cc_backend::AnalyticalCommunicationConfig communication_config{};
    communication_config.network_bandwidth_gbps = 400.0;
    communication_config.latency_us = 1.0;
    communication_config.intra_node_bandwidth_gbps = 14'400.0;
    const frontier::cc_backend::AnalyticalCommunicationModel communication(
        communication_config);

    frontier::execution_time_predictor::detail::RoutingAllocation balanced{};
    balanced.input_tokens = 64;
    balanced.routed_tokens = 512;
    balanced.lane_routed_tokens = std::vector<std::uint64_t>(32, 16);
    balanced.lane_unique_tokens = std::vector<std::uint64_t>(32, 12);
    auto imbalanced = balanced;
    imbalanced.lane_unique_tokens.front() = 48;
    auto duplicate_heavy = balanced;
    duplicate_heavy.lane_routed_tokens.front() = 64;

    const auto public_balanced =
        frontier::execution_time_predictor::detail::predict_moe_communication(
            communication, 64, 7168, 512, 4, 1, 32, 4, false, 1.0, 2048,
            "sm100_megamoe_public", &balanced, 0.04);
    const auto public_imbalanced =
        frontier::execution_time_predictor::detail::predict_moe_communication(
            communication, 64, 7168, 512, 4, 1, 32, 4, false, 1.0, 2048,
            "sm100_megamoe_public", &imbalanced, 0.04);
    const auto public_duplicate_heavy =
        frontier::execution_time_predictor::detail::predict_moe_communication(
            communication, 64, 7168, 512, 4, 1, 32, 4, false, 1.0, 2048,
            "sm100_megamoe_public", &duplicate_heavy, 0.04);
    const auto synchronized =
        frontier::execution_time_predictor::detail::predict_moe_communication(
            communication, 64, 7168, 512, 4, 1, 32, 4, false, 1.0, 2048,
            "sm100_megamoe_public", &balanced, 0.04, 1.0);
    const auto rubin_bandwidth =
        frontier::execution_time_predictor::detail::predict_moe_communication(
            communication, 64, 7168, 512, 4, 1, 32, 4, false, 1.0, 2048,
            "sm100_megamoe_public", &balanced, 0.04, 0.35, 2.0, 1.0);

    require(public_balanced.raw_ep_dispatch_ms > 0.018 &&
                public_balanced.raw_ep_combine_ms > 0.031,
            "public profile must include measured A2A startup floors");
    require(public_balanced.ep_dispatch_ms + public_balanced.ep_combine_ms <
                public_balanced.raw_ep_dispatch_ms +
                    public_balanced.raw_ep_combine_ms,
            "MegaMoE profile must expose only the non-overlapped tail");
    require(public_balanced.dp_input_ms == 0.0 &&
                public_balanced.dp_output_ms == 0.0,
            "DP-attention to EP-expert transition must not be double charged");
    require(public_imbalanced.raw_ep_dispatch_ms >
                public_balanced.raw_ep_dispatch_ms,
            "receiver-side unique-token imbalance must lengthen dispatch");
    require(public_duplicate_heavy.raw_ep_dispatch_ms ==
                    public_balanced.raw_ep_dispatch_ms &&
                public_duplicate_heavy.raw_ep_combine_ms ==
                    public_balanced.raw_ep_combine_ms,
            "multiple expert routes to one destination lane must not duplicate "
            "the one-sided communication payload");
    require(std::abs(synchronized.ep_dispatch_ms + synchronized.ep_combine_ms -
                     synchronized.raw_ep_dispatch_ms -
                     synchronized.raw_ep_combine_ms) < 1e-12,
            "synchronized MegaMoE must expose the full A2A path");
    const double ep32_position = 2.0 / 3.0;
    const double dispatch_startup_ms = (18.0 + 4.0 * ep32_position) / 1000.0;
    const double combine_startup_ms = (31.0 + 2.0 * ep32_position) / 1000.0;
    require(std::abs(rubin_bandwidth.raw_ep_dispatch_ms - dispatch_startup_ms -
                     0.5 * (public_balanced.raw_ep_dispatch_ms -
                            dispatch_startup_ms)) < 1e-12 &&
                std::abs(rubin_bandwidth.raw_ep_combine_ms -
                         combine_startup_ms -
                         0.5 * (public_balanced.raw_ep_combine_ms -
                                combine_startup_ms)) < 1e-12,
            "Rubin NVLink scaling must halve payload time without scaling "
            "the fixed A2A startup");
}

void test_group_moe_communication_aggregates_dp_source_rows() {
    frontier::config::AnalyticalExecutionModelConfig execution{};
    execution.device = "gb300";
    execution.precision = "fp8";
    execution.moe_communication_backend = "sm100_megamoe_public";
    execution.network_bandwidth_gbps = 28'800.0;
    execution.intra_node_bandwidth_gbps = 28'800.0;

    frontier::config::ParallelismConfig parallelism{};
    parallelism.tensor_parallel_size = 8;
    parallelism.decode_context_parallel_size = 8;
    parallelism.data_parallel_size = 4;
    parallelism.moe_tensor_parallel_size = 1;
    parallelism.moe_expert_parallel_size = 32;

    const auto model =
        frontier::config::load_model_config("moonshotai/Kimi-K3");
    frontier::config::MoeRoutingConfig routing{};
    routing.mode = frontier::config::MoeRoutingMode::kUniformRandom;
    routing.distribution = frontier::config::MoeRoutingDistribution::kBalanced;
    routing.layer_scope = frontier::config::MoeRoutingLayerScope::kShared;

    const auto source0 =
        frontier::execution_time_predictor::detail::route_tokens(
            64, model.router_topk, model.total_expert_num,
            parallelism.moe_expert_parallel_size, routing, 0, 1001);
    const auto source1 =
        frontier::execution_time_predictor::detail::route_tokens(
            64, model.router_topk, model.total_expert_num,
            parallelism.moe_expert_parallel_size, routing, 0, 1002);

    const auto make_input = [&](bool include_second_source) {
        frontier::execution_time_predictor::MoEGroupLayerInput input{};
        input.layer_id = frontier::LayerId{0};
        input.model_layer_id = 0;
        input.input_tokens = include_second_source ? 128 : 64;
        input.routed_tokens = input.input_tokens * model.router_topk;
        input.global_expert_tokens = source0.global_expert_tokens;
        input.source_input_tokens = {64};
        input.source_shared_expert_path_ms = {0.0};
        input.source_lane_routed_tokens = {source0.lane_routed_tokens};
        input.source_lane_unique_tokens = {source0.lane_unique_tokens};
        if (include_second_source) {
            for (std::size_t expert = 0;
                 expert < input.global_expert_tokens.size(); ++expert) {
                input.global_expert_tokens.at(expert) +=
                    source1.global_expert_tokens.at(expert);
            }
            input.source_lane_routed_tokens.push_back(
                source1.lane_routed_tokens);
            input.source_lane_unique_tokens.push_back(
                source1.lane_unique_tokens);
            input.source_input_tokens.push_back(64);
            input.source_shared_expert_path_ms.push_back(0.0);
        }
        input.fallback_lane_times_ms.assign(
            static_cast<std::size_t>(parallelism.moe_expert_parallel_size),
            0.0);
        return input;
    };

    const frontier::execution_time_predictor::
        AnalyticalRooflineExecutionTimePredictor predictor(
            execution, parallelism, model, routing);
    const auto single = predictor.predict_moe_group_layer(make_input(false));
    const auto group = predictor.predict_moe_group_layer(make_input(true));
    execution.kernel_profile = "k3_deepgemm_megamoe";
    const frontier::execution_time_predictor::
        AnalyticalRooflineExecutionTimePredictor synchronized_predictor(
            execution, parallelism, model, routing);
    const auto synchronized_group =
        synchronized_predictor.predict_moe_group_layer(make_input(true));
    auto sensitivity_execution = execution;
    sensitivity_execution.mega_moe_cluster_task_latency_us = 0.0;
    sensitivity_execution.mega_moe_wave_exposure = 0.0;
    sensitivity_execution.mega_moe_tail_io_fraction = 1.0;
    const frontier::execution_time_predictor::
        AnalyticalRooflineExecutionTimePredictor sensitivity_predictor(
            sensitivity_execution, parallelism, model, routing);
    const auto sensitivity_group =
        sensitivity_predictor.predict_moe_group_layer(make_input(true));
    require(single.has_source_aware_ep_communication &&
                group.has_source_aware_ep_communication,
            "GB300 group predictor must expose source-aware EP communication");
    require(single.lane_times_are_routed_only &&
                group.lane_times_are_routed_only,
            "analytical DP-group prediction must expose only destination "
            "routed work");
    for (std::size_t lane = 0;
         lane < group.destination_lane_routed_tokens.size(); ++lane) {
        require(group.destination_lane_routed_tokens.at(lane) ==
                    source0.lane_routed_tokens.at(lane) +
                        source1.lane_routed_tokens.at(lane),
                "destination load must equal the DP-source column sum");
        require(group.destination_lane_unique_tokens.at(lane) ==
                    source0.lane_unique_tokens.at(lane) +
                        source1.lane_unique_tokens.at(lane),
                "destination unique-token load must equal the DP-source "
                "column sum");
    }
    require(group.raw_ep_dispatch_ms >= single.raw_ep_dispatch_ms &&
                group.raw_ep_combine_ms >= single.raw_ep_combine_ms,
            "adding an active DP source must not reduce receiver-side A2A "
            "time");
    require(synchronized_group.critical_lane_time_ms >
                    group.critical_lane_time_ms &&
                synchronized_group.ep_dispatch_ms +
                        synchronized_group.ep_combine_ms >
                    group.ep_dispatch_ms + group.ep_combine_ms,
            "wide-EP K3 profile must expose receiver-side kernel and barrier "
            "costs");
    require(sensitivity_group.critical_lane_time_ms !=
                synchronized_group.critical_lane_time_ms,
            "MegaMoE rho_tail/lambda_wave/c_grid config overrides must "
            "change the live routed group prediction");
    require(synchronized_group.grouped_gemm_geometry.enabled &&
                synchronized_group.grouped_gemm_geometry.block_m > 0 &&
                synchronized_group.grouped_gemm_geometry.up_cluster_tasks > 0 &&
                synchronized_group.grouped_gemm_geometry.down_cluster_tasks > 0,
            "production group prediction must retain critical-lane MegaMoE "
            "geometry");

    auto no_shared_model = model;
    no_shared_model.num_shared_experts = 0;
    const frontier::execution_time_predictor::
        AnalyticalRooflineExecutionTimePredictor no_shared_predictor(
            execution, parallelism, no_shared_model, routing);
    const auto no_shared_group =
        no_shared_predictor.predict_moe_group_layer(make_input(true));
    require(no_shared_group.lane_times_ms == synchronized_group.lane_times_ms,
            "DP-group destination work must not include source-local shared "
            "experts");
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

void test_split_expert_decomposition_is_additive() {
    using frontier::execution_time_predictor::detail::AnalyticalConfig;
    using frontier::execution_time_predictor::detail::DeviceCeilings;
    using frontier::execution_time_predictor::detail::MoEModel;
    using frontier::execution_time_predictor::detail::MoEOperatorPrecisions;
    using frontier::execution_time_predictor::detail::Precision;

    MoEModel model{};
    model.hidden_size = 4096;
    model.intermediate_size = 2048;
    model.model_num_experts = 8;
    model.num_shared_experts = 1;
    model.moe_tensor_parallel_size = 1;
    model.gated_mlp = true;

    MoEOperatorPrecisions precisions{};
    precisions.expert_weight = Precision::kFp4;
    precisions.expert_activation = Precision::kFp8;
    precisions.shared_expert_weight = Precision::kFp8;
    precisions.shared_expert_activation = Precision::kFp8;
    const auto layer =
        frontier::execution_time_predictor::detail::predict_moe_layer(
            DeviceCeilings::gb300(), AnalyticalConfig{}, model, 32, 2,
            std::vector<std::uint64_t>(8, 8), precisions);
    require(std::abs(layer.routed_path_ms + layer.source_local_ms -
                     layer.total_ms()) < 1e-12,
            "split routed/shared MoE decomposition must add to total_ms");
}

void test_mega_moe_geometry_uses_exact_expert_histogram() {
    using frontier::execution_time_predictor::detail::AnalyticalConfig;
    using frontier::execution_time_predictor::detail::DeviceCeilings;
    using frontier::execution_time_predictor::detail::MoEModel;
    using frontier::execution_time_predictor::detail::Precision;

    MoEModel model{};
    model.hidden_size = 4'096;
    model.intermediate_size = 2'048;
    model.model_num_experts = 16;
    model.moe_tensor_parallel_size = 1;
    model.gated_mlp = true;

    const std::vector<std::uint64_t> aligned = {16, 16, 16, 16, 16, 0, 0, 0,
                                                0,  0,  0,  0,  0,  0, 0, 0};
    const std::vector<std::uint64_t> fragmented = {31, 16, 16, 16, 1, 0, 0, 0,
                                                   0,  0,  0,  0,  0, 0, 0, 0};
    const AnalyticalConfig generic{};
    const AnalyticalConfig mega = frontier::execution_time_predictor::detail::
        analytical_config_from_profile("k3_deepgemm_megamoe");
    const AnalyticalConfig rubin_mega = frontier::execution_time_predictor::
        detail::analytical_config_from_profile("k3_deepgemm_megamoe", "rubin");
    const AnalyticalConfig rubin_generic = frontier::execution_time_predictor::
        detail::analytical_config_from_profile("generic", "rubin");
    require(rubin_mega.mega_moe_sm_count == 224 &&
                rubin_mega.mega_moe_a2a_bandwidth_scale == 2.0 &&
                rubin_mega.mega_moe_a2a_startup_scale == 1.0 &&
                rubin_mega.mega_moe_tail_io_fraction == 1.0 &&
                rubin_mega.mega_moe_wave_exposure == 0.0 &&
                std::abs(rubin_mega.mega_moe_cluster_task_latency_us -
                         0.06325) < 1e-12 &&
                rubin_mega.moe_a2a_overlap_residual == 1.0,
            "Rubin MegaMoE projection must scale SM waves and NVLink payload "
            "without assuming a lower unmeasured cluster-task latency");
    require(!rubin_generic.mega_moe_geometry_enabled &&
                rubin_generic.mega_moe_sm_count == 224 &&
                rubin_generic.mega_moe_a2a_bandwidth_scale == 2.0,
            "Rubin device resources must remain independent of the optional "
            "K3 kernel profile");

    const auto generic_aligned =
        frontier::execution_time_predictor::detail::predict_moe_layer(
            DeviceCeilings::gb300(), generic, model, 80, 1, aligned,
            Precision::kFp8);
    const auto generic_fragmented =
        frontier::execution_time_predictor::detail::predict_moe_layer(
            DeviceCeilings::gb300(), generic, model, 80, 1, fragmented,
            Precision::kFp8);
    const auto mega_aligned =
        frontier::execution_time_predictor::detail::predict_moe_layer(
            DeviceCeilings::gb300(), mega, model, 80, 1, aligned,
            Precision::kFp8);
    const auto mega_fragmented =
        frontier::execution_time_predictor::detail::predict_moe_layer(
            DeviceCeilings::gb300(), mega, model, 80, 1, fragmented,
            Precision::kFp8);

    require(generic_aligned.grouped_up_projection_ms ==
                    generic_fragmented.grouped_up_projection_ms &&
                generic_aligned.grouped_down_projection_ms ==
                    generic_fragmented.grouped_down_projection_ms,
            "portable roofline must retain aggregate-work behavior");
    require(mega_fragmented.grouped_up_projection_ms >
                    mega_aligned.grouped_up_projection_ms &&
                mega_fragmented.grouped_down_projection_ms >
                    mega_aligned.grouped_down_projection_ms,
            "MegaMoE geometry must charge the extra expert M block");
    require(
        mega_aligned.grouped_gemm_geometry.enabled &&
            mega_aligned.grouped_gemm_geometry.block_m == 16 &&
            mega_aligned.grouped_gemm_geometry.routed_m_blocks == 5 &&
            mega_fragmented.grouped_gemm_geometry.routed_m_blocks == 6 &&
            mega_aligned.grouped_gemm_geometry.routed_padded_tokens == 80 &&
            mega_fragmented.grouped_gemm_geometry.routed_padded_tokens == 96 &&
            std::abs(
                mega_aligned.grouped_gemm_geometry.up_cluster_task_overhead_ms -
                static_cast<double>(
                    mega_aligned.grouped_gemm_geometry.up_cluster_tasks) *
                    0.06325 / 1000.0) < 1e-12,
        "MegaMoE diagnostics must expose block-M padding geometry");
    require(mega_aligned.shuffling_ms == generic_aligned.shuffling_ms,
            "expert GEMM geometry must not scale shuffling work");

    frontier::execution_time_predictor::detail::RoutingAllocation allocation{};
    allocation.input_tokens = 80;
    allocation.routed_tokens = 80;
    allocation.lane_expert_tokens = {aligned};
    const auto source_local =
        frontier::execution_time_predictor::detail::predict_moe_lanes(
            DeviceCeilings::gb300(), mega, model, allocation, 1,
            Precision::kFp8);
    const auto group_composed =
        frontier::execution_time_predictor::detail::predict_routed_moe_lanes(
            DeviceCeilings::gb300(), mega, model, allocation, 1,
            frontier::execution_time_predictor::detail::MoEOperatorPrecisions{},
            true);
    require(!source_local.lane_times.front().grouped_gemm_geometry.enabled &&
                group_composed.critical_lane_geometry.enabled &&
                group_composed.critical_lane_geometry.block_m == 16 &&
                group_composed.critical_lane_geometry.up_cluster_tasks > 0,
            "live routed-only group prediction must retain its critical "
            "MegaMoE geometry");

    auto threshold_model = model;
    threshold_model.model_num_experts = 8;
    const std::vector<std::uint64_t> threshold_low = {68, 0, 0, 0, 0, 0, 0, 0};
    const std::vector<std::uint64_t> threshold_high = {69, 0, 0, 0, 0, 0, 0, 0};
    const auto low =
        frontier::execution_time_predictor::detail::predict_moe_layer(
            DeviceCeilings::gb300(), mega, threshold_model, 68, 1,
            threshold_low, Precision::kFp8);
    const auto high =
        frontier::execution_time_predictor::detail::predict_moe_layer(
            DeviceCeilings::gb300(), mega, threshold_model, 69, 1,
            threshold_high, Precision::kFp8);
    require(low.grouped_gemm_geometry.block_m == 16 &&
                high.grouped_gemm_geometry.block_m == 32,
            "MegaMoE block-M policy must switch above 8.5 tokens/expert");
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
        test_uniform_routing_varies_by_batch_and_reports_lane_traffic();
        test_sm100_megamoe_public_profile_models_overlap_and_layout_transition();
        test_group_moe_communication_aggregates_dp_source_rows();
        test_moe_lane_analytical_model();
        test_split_expert_decomposition_is_additive();
        test_shared_expert_is_replicated_across_ep_and_sharded_by_tp();
        test_latent_moe_projection_precision_is_independent();
        test_mega_moe_geometry_uses_exact_expert_histogram();
        test_moe_overflow_and_nonfinite_inputs_fail_fast();
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
