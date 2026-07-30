#include "frontier/config/config.h"
#include "frontier/core/event.h"
#include "frontier/request_generator/workload.h"
#include "frontier/simulator/simulator.h"
#include "tests/test_support.h"

#include <algorithm>
#include <array>
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
    return failures == 0 ? 0 : 1;
}
