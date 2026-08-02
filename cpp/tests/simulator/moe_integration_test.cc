#include "frontier/config/config.h"
#include "frontier/core/event.h"
#include "frontier/request_generator/workload.h"
#include "frontier/simulator/simulator.h"
#include "tests/test_support.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <string>

#ifndef FRONTIER_TEST_FIXTURE_DIR
#error "FRONTIER_TEST_FIXTURE_DIR must be defined for simulator tests"
#endif

namespace {

using frontier::EventType;
using frontier::config::parse_simulation_config_json;
using frontier::request_generator::parse_workload_csv;
using frontier::simulator::run_simulation;
using frontier::test::expect;
using frontier::test::read_text_file;

const std::filesystem::path kFixtureRoot{FRONTIER_TEST_FIXTURE_DIR};

frontier::config::SimulationConfig load_config(const char *name) {
    return parse_simulation_config_json(
        read_text_file(kFixtureRoot / "config" / name));
}

auto load_small_workload() {
    return parse_workload_csv(
        read_text_file(kFixtureRoot / "workloads/step3_pdd_small.csv"));
}

std::size_t count_events(const frontier::metrics::SimulationOutput &output,
                         EventType type) {
    return static_cast<std::size_t>(std::count_if(
        output.event_trace.begin(), output.event_trace.end(),
        [type](const frontier::Event &event) { return event.type() == type; }));
}

void test_local_moe_uses_fast_path() {
    const auto output = run_simulation(
        load_config("fixed_moe_local_colocation.json"), load_small_workload());

    expect(output.requests.size() == 2 &&
               count_events(output, EventType::kPrefillSync) == 0 &&
               count_events(output, EventType::kPrefillSyncCollective) == 0 &&
               count_events(output, EventType::kDecodeSync) == 0 &&
               count_events(output, EventType::kDecodeSyncCollective) == 0,
           "EP1/DP1 MoE must retain the direct stage path");
    expect(std::all_of(output.batch_stages.begin(), output.batch_stages.end(),
                       [](const auto &stage) {
                           return stage.execution_time.moe_grouped_gemm_ms >
                                      0.0 &&
                                  stage.execution_time.ep_dispatch_ms == 0.0 &&
                                  stage.execution_time.ep_combine_ms == 0.0;
                       }),
           "local MoE stages must expose compute without EP collectives");
}

void test_colocation_runs_all_sync_event_families() {
    const auto output = run_simulation(
        load_config("analytical_moe_ep4_colocation.json"),
        parse_workload_csv("arrived_at,num_prefill_tokens,num_decode_tokens\n"
                           "0,4,3\n"));
    const std::array types{
        EventType::kPrefillSync,
        EventType::kPrefillSyncCollective,
        EventType::kDecodeSync,
        EventType::kDecodeSyncCollective,
    };
    expect(
        std::all_of(types.begin(), types.end(),
                    [&output](EventType type) {
                        return count_events(output, type) > 0;
                    }),
        "synchronized co-location MoE must exercise all four event families");

    const bool has_prefill_idle = std::any_of(
        output.event_trace.begin(), output.event_trace.end(),
        [](const frontier::Event &event) {
            if (event.type() == EventType::kPrefillSync) {
                return event.as<frontier::PrefillSyncPayload>().is_idle;
            }
            return false;
        });
    const bool has_decode_pre_idle_event = std::any_of(
        output.event_trace.begin(), output.event_trace.end(),
        [](const frontier::Event &event) {
            if (event.type() != EventType::kDecodeSync) {
                return false;
            }
            const auto &payload = event.as<frontier::DecodeSyncPayload>();
            return payload.is_idle &&
                   payload.sync_phase == frontier::MoESyncPhase::kPreMoe;
        });
    const bool has_decode_post_idle_event = std::any_of(
        output.event_trace.begin(), output.event_trace.end(),
        [](const frontier::Event &event) {
            if (event.type() != EventType::kDecodeSync) {
                return false;
            }
            const auto &payload = event.as<frontier::DecodeSyncPayload>();
            return payload.is_idle &&
                   payload.sync_phase == frontier::MoESyncPhase::kPostMoe;
        });
    expect(has_prefill_idle && !has_decode_pre_idle_event &&
               has_decode_post_idle_event,
           "prefill emits idle events while monolithic decode compacts pre-MoE "
           "idles and emits lane-timed post-MoE arrivals");
    expect(output.requests.size() == 1 &&
               std::all_of(output.batch_stages.begin(),
                           output.batch_stages.end(),
                           [](const auto &stage) {
                               return stage.execution_time.total_ms() > 0.0;
                           }),
           "synchronized MoE must finish each real request and stage");
}

void test_pdd_moe_preserves_phi_kv_contract() {
    const auto workload = load_small_workload();
    const auto output =
        run_simulation(load_config("fixed_moe_sequential_pdd.json"), workload);
    expect(output.requests.size() == workload.size() &&
               output.kv_cache_transfers.size() == workload.size() &&
               count_events(output, EventType::kPrefillSync) > 0 &&
               count_events(output, EventType::kDecodeSync) > 0,
           "sequential PDD MoE must synchronize both clusters and complete "
           "transfers");

    constexpr std::uint64_t kBytesPerPrefillToken =
        32ULL * 4ULL * 128ULL * 2ULL * 2ULL;
    for (std::size_t index = 0; index < workload.size(); ++index) {
        expect(output.kv_cache_transfers.at(index).size_bytes ==
                   workload.at(index).num_prefill_tokens *
                       kBytesPerPrefillToken,
               "Phi KV transfer size must use model layers, KV heads, and head "
               "dim");
    }
}

void test_kimi_k2_dense_prefix_runs_with_uneven_pp() {
    auto config = load_config("analytical_moe_ep4_colocation.json");
    auto &runtime = config.cluster();
    runtime.model =
        frontier::config::load_model_config("moonshotai/Kimi-K2-Instruct");
    runtime.parallelism.tensor_parallel_size = 4;
    runtime.parallelism.pipeline_parallel_size = 4;
    runtime.parallelism.data_parallel_size = 2;
    runtime.parallelism.moe_tensor_parallel_size = 1;
    runtime.parallelism.moe_expert_parallel_size = 8;
    runtime.execution_model.analytical.tensor_parallel_size = 4;
    runtime.execution_model.analytical.precision = "fp8";
    runtime.execution_model.analytical.operator_precisions.moe_expert_weight =
        "fp4";
    runtime.execution_model.analytical.operator_precisions
        .moe_expert_activation = "fp8";

    const auto output = run_simulation(
        config,
        parse_workload_csv("arrived_at,num_prefill_tokens,num_decode_tokens\n"
                           "0,1,1\n"));
    expect(output.requests.size() == 1 && output.batch_stages.size() >= 4 &&
               output.batch_stages.size() % 4 == 0,
           "Kimi K2 PP4 must complete every four-stage pipeline pass");
    expect(std::any_of(output.batch_stages.begin(), output.batch_stages.end(),
                       [](const auto &stage) {
                           return stage.stage_id.value() == 3 &&
                                  stage.execution_time.lm_head_ms > 0.0;
                       }),
           "Kimi K2 final stage must report LM-head projection time");
}

void test_kimi_k2_first_layer_scaled_preserves_completion_time() {
    auto detailed_config = load_config("analytical_moe_ep4_colocation.json");
    auto &runtime = detailed_config.cluster();
    runtime.model =
        frontier::config::load_model_config("moonshotai/Kimi-K2-Instruct");
    runtime.parallelism.tensor_parallel_size = 4;
    runtime.parallelism.pipeline_parallel_size = 4;
    runtime.parallelism.data_parallel_size = 2;
    runtime.parallelism.moe_tensor_parallel_size = 1;
    runtime.parallelism.moe_expert_parallel_size = 8;
    runtime.execution_model.analytical.tensor_parallel_size = 4;
    runtime.execution_model.analytical.precision = "fp8";
    runtime.execution_model.analytical.operator_precisions.moe_expert_weight =
        "fp4";
    runtime.execution_model.analytical.operator_precisions
        .moe_expert_activation = "fp8";
    auto scaled_config = detailed_config;
    scaled_config.cluster()
        .execution_model.analytical.moe_layer_event_mode =
        "first_layer_scaled";
    const auto workload = parse_workload_csv(
        "arrived_at,num_prefill_tokens,num_decode_tokens\n0,8,2\n");

    const auto detailed = run_simulation(detailed_config, workload);
    const auto scaled = run_simulation(scaled_config, workload);
    expect(detailed.requests.size() == 1 && scaled.requests.size() == 1 &&
               std::abs(detailed.requests.front().completed_at.seconds() -
                        scaled.requests.front().completed_at.seconds()) <
                   1e-12,
           "scaled MoE layers must preserve request completion time: " +
               std::to_string(detailed.requests.front().completed_at.seconds()) +
               " vs " +
               std::to_string(scaled.requests.front().completed_at.seconds()));
    expect(scaled.aggregate.event_count < detailed.aggregate.event_count &&
               scaled.aggregate.moe_routing_count * 10 <
                   detailed.aggregate.moe_routing_count,
           "scaled MoE layers must materially reduce events and routing "
           "records");
}

void test_kimi_k2_first_layer_scaled_preserves_pdd_completion_time() {
    auto detailed_config = load_config("fixed_moe_sequential_pdd.json");
    const auto model =
        frontier::config::load_model_config("moonshotai/Kimi-K2-Instruct");
    const auto configure_cluster = [&model](
                                       frontier::config::ClusterRuntimeConfig
                                           &cluster,
                                       std::uint64_t data_parallel,
                                       std::uint64_t expert_parallel) {
        cluster.model = model;
        cluster.parallelism.tensor_parallel_size = 4;
        cluster.parallelism.pipeline_parallel_size = 4;
        cluster.parallelism.data_parallel_size = data_parallel;
        cluster.parallelism.moe_tensor_parallel_size = 1;
        cluster.parallelism.moe_expert_parallel_size = expert_parallel;
        cluster.execution_model.type =
            frontier::config::ExecutionModelType::kAnalytical;
        cluster.execution_model.analytical.tensor_parallel_size = 4;
        cluster.execution_model.analytical.precision = "fp8";
        cluster.execution_model.analytical.operator_precisions
            .moe_expert_weight = "fp4";
        cluster.execution_model.analytical.operator_precisions
            .moe_expert_activation = "fp8";
    };
    configure_cluster(detailed_config.pdd().clusters.prefill, 1, 4);
    configure_cluster(detailed_config.pdd().clusters.decode, 2, 8);
    detailed_config.pdd().clusters.prefill.moe_routing.distribution =
        frontier::config::MoeRoutingDistribution::kBalanced;
    detailed_config.pdd().clusters.decode.moe_routing.distribution =
        frontier::config::MoeRoutingDistribution::kBalanced;
    detailed_config.pdd().clusters.prefill.moe_routing.mode =
        frontier::config::MoeRoutingMode::kUniformLegacy;
    detailed_config.pdd().clusters.decode.moe_routing.mode =
        frontier::config::MoeRoutingMode::kUniformLegacy;
    detailed_config.pdd().kv_cache_transfer.kv_cache_dtype_size_bytes = 1.0;
    auto scaled_config = detailed_config;
    scaled_config.pdd()
        .clusters.prefill.execution_model.analytical.moe_layer_event_mode =
        "first_layer_scaled";
    scaled_config.pdd()
        .clusters.decode.execution_model.analytical.moe_layer_event_mode =
        "first_layer_scaled";
    const auto workload = parse_workload_csv(
        "arrived_at,num_prefill_tokens,num_decode_tokens\n0,8,2\n");

    const auto detailed = run_simulation(detailed_config, workload);
    const auto scaled = run_simulation(scaled_config, workload);
    expect(detailed.batch_stages.size() == scaled.batch_stages.size(),
           "scaled PDD run must preserve the batch-stage count");
    for (std::size_t index = 0; index < detailed.batch_stages.size(); ++index) {
        const double detailed_ms =
            (detailed.batch_stages.at(index).completed_at.seconds() -
             detailed.batch_stages.at(index).started_at.seconds()) *
            1e3;
        const double scaled_ms =
            (scaled.batch_stages.at(index).completed_at.seconds() -
             scaled.batch_stages.at(index).started_at.seconds()) *
            1e3;
        expect(std::abs(detailed_ms - scaled_ms) < 1e-9,
               "scaled PDD stage duration mismatch at index " +
                   std::to_string(index) + ": " +
                   std::to_string(detailed_ms) + " vs " +
                   std::to_string(scaled_ms));
    }
    expect(std::abs(detailed.requests.front().completed_at.seconds() -
                    scaled.requests.front().completed_at.seconds()) < 1e-12,
           "scaled PDD MoE layers must preserve request completion time: " +
               std::to_string(detailed.requests.front().completed_at.seconds()) +
               " vs " +
               std::to_string(scaled.requests.front().completed_at.seconds()));
    expect(scaled.aggregate.event_count < detailed.aggregate.event_count &&
               scaled.aggregate.moe_routing_count * 10 <
                   detailed.aggregate.moe_routing_count,
           "scaled PDD MoE layers must materially reduce detailed records");
}

} // namespace

int main() {
    int failures = 0;
    failures += frontier::test::run("local MoE fast path",
                                    test_local_moe_uses_fast_path);
    failures +=
        frontier::test::run("co-location MoE synchronization",
                            test_colocation_runs_all_sync_event_families);
    failures += frontier::test::run("PDD MoE Phi KV contract",
                                    test_pdd_moe_preserves_phi_kv_contract);
    failures +=
        frontier::test::run("Kimi K2 dense prefix and uneven PP",
                            test_kimi_k2_dense_prefix_runs_with_uneven_pp);
    failures += frontier::test::run(
        "Kimi K2 first-layer-scaled completion",
        test_kimi_k2_first_layer_scaled_preserves_completion_time);
    failures += frontier::test::run(
        "Kimi K2 first-layer-scaled PDD completion",
        test_kimi_k2_first_layer_scaled_preserves_pdd_completion_time);
    return failures == 0 ? 0 : 1;
}
