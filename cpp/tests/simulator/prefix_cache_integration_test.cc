#include "frontier/config/config.h"
#include "frontier/request_generator/workload.h"
#include "frontier/simulator/simulator.h"
#include "tests/test_support.h"

#include <algorithm>
#include <filesystem>
#include <vector>

#ifndef FRONTIER_TEST_FIXTURE_DIR
#error "FRONTIER_TEST_FIXTURE_DIR must be defined for integration tests"
#endif

namespace {

using frontier::RequestId;
using frontier::SessionId;
using frontier::config::ClusterSchedulerType;
using frontier::config::parse_simulation_config_json;
using frontier::config::SimulationConfig;
using frontier::metrics::RequestMetricsRecord;
using frontier::metrics::SimulationOutput;
using frontier::request_generator::parse_workload_csv;
using frontier::request_generator::WorkloadRequest;
using frontier::simulator::run_simulation;
using frontier::simulator::SimulationError;
using frontier::simulator::Simulator;
using frontier::test::expect;
using frontier::test::expect_throws;
using frontier::test::read_text_file;

std::filesystem::path fixture(std::string_view relative) {
    return std::filesystem::path{FRONTIER_TEST_FIXTURE_DIR} / relative;
}

SimulationConfig load_config(std::string_view name) {
    SimulationConfig config = parse_simulation_config_json(
        read_text_file(fixture(std::string{"config/"} + std::string{name})));
    config.prefix_cache.enabled = true;
    config.cluster_scheduler.type = ClusterSchedulerType::kStickyRoundRobin;
    return config;
}

const RequestMetricsRecord &request(const SimulationOutput &output,
                                    RequestId request_id) {
    const auto position =
        std::find_if(output.requests.begin(), output.requests.end(),
                     [request_id](const RequestMetricsRecord &value) {
                         return value.request_id == request_id;
                     });
    if (position == output.requests.end()) {
        throw std::runtime_error("integration output is missing a request");
    }
    return *position;
}

void test_colocation_and_pdd_session_prefix_cache() {
    const auto workload = parse_workload_csv(
        read_text_file(fixture("workloads/session_prefix.csv")));
    const SimulationOutput colocation =
        run_simulation(load_config("fixed_parallel_colocation.json"), workload);
    const SimulationOutput pdd =
        run_simulation(load_config("fixed_sequential_pdd.json"), workload);

    const RequestMetricsRecord &colocation_followup =
        request(colocation, RequestId{2});
    const RequestMetricsRecord &pdd_followup = request(pdd, RequestId{2});
    expect(colocation_followup.num_prefill_tokens == 48 &&
               pdd_followup.num_prefill_tokens == 48,
           "both architectures must consume the same materialized context");
    expect(colocation_followup.cached_prefill_tokens == 40,
           "co-location cache may publish decode-produced full blocks");
    expect(pdd_followup.cached_prefill_tokens == 32,
           "PDD cache must publish PREFILL blocks only");
    expect(request(colocation, RequestId{0}).cached_prefill_tokens == 0 &&
               request(colocation, RequestId{1}).cached_prefill_tokens == 0,
           "first requests of independent sessions must miss");
    expect(colocation_followup.arrived_at.seconds() ==
                   request(colocation, RequestId{0}).completed_at.seconds() +
                       2.0 &&
               pdd_followup.arrived_at.seconds() ==
                   request(pdd, RequestId{0}).completed_at.seconds() + 2.0,
           "both architectures must inject a successor only after terminal "
           "completion plus think time");
    expect(colocation_followup.replica_id ==
                   request(colocation, RequestId{0}).replica_id &&
               colocation_followup.dp_id ==
                   request(colocation, RequestId{0}).dp_id,
           "sticky affinity must retain the session's replica/DP target");
    expect(request(colocation, RequestId{0}).replica_id !=
                   request(colocation, RequestId{1}).replica_id ||
               request(colocation, RequestId{0}).dp_id !=
                   request(colocation, RequestId{1}).dp_id,
           "new sessions must consume distinct sticky targets");
    expect(colocation.aggregate.prefix_cache.hit_blocks == 10 &&
               pdd.aggregate.prefix_cache.hit_blocks == 8,
           "aggregate cache hits must reflect architecture ownership");
    expect(pdd.kv_cache_transfers.size() == workload.size(),
           "prefix caching must preserve one PDD transfer per request");
    expect(!colocation.prefix_cache_targets.empty() &&
               !pdd.prefix_cache_targets.empty(),
           "quiescent cache diagnostics must retain keyed free blocks");
    expect(colocation.prefix_cache_targets.front().active_blocks == 0 &&
               colocation.prefix_cache_targets.front().resident_blocks > 0 &&
               colocation.aggregate.prefix_cache.storage_model ==
                   "analytical_session",
           "free cached blocks must be idle while still GPU-resident");
}

void test_multi_target_cache_requires_sticky_affinity() {
    SimulationConfig config = load_config("fixed_parallel_colocation.json");
    config.cluster_scheduler.type = ClusterSchedulerType::kRoundRobin;
    const auto workload = parse_workload_csv(
        read_text_file(fixture("workloads/session_prefix.csv")));
    expect_throws<SimulationError>(
        [&] { static_cast<void>(run_simulation(config, workload)); },
        "multi-target prefix cache must reject non-sticky routing");
    expect_throws<SimulationError>(
        [&] { static_cast<void>(Simulator{config, workload}.run()); },
        "direct Simulator entry points must enforce sticky routing too");
}

void test_direct_simulator_entry_materializes_sessions() {
    const auto workload = parse_workload_csv(
        read_text_file(fixture("workloads/session_prefix.csv")));
    const SimulationOutput output =
        Simulator{load_config("fixed_parallel_colocation.json"), workload}
            .run();
    expect(request(output, RequestId{2}).num_prefill_tokens == 48 &&
               request(output, RequestId{2}).cached_prefill_tokens == 40,
           "direct Simulator users must receive the same one-time workload "
           "materialization as run_simulation");
}

void test_moe_and_eviction_pressure_configs() {
    const auto workload = parse_workload_csv(
        read_text_file(fixture("workloads/session_prefix.csv")));
    const SimulationOutput moe_colocation = run_simulation(
        load_config("fixed_moe_local_colocation.json"), workload);
    const SimulationOutput moe_pdd =
        run_simulation(load_config("fixed_moe_sequential_pdd.json"), workload);
    expect(moe_colocation.requests.size() == workload.size() &&
               moe_pdd.requests.size() == workload.size() &&
               !moe_colocation.moe_routing.empty() &&
               !moe_pdd.moe_routing.empty(),
           "MoE co-location and PDD prefix-cache configs must drain");
    expect(request(moe_colocation, RequestId{2}).cached_prefill_tokens > 0 &&
               request(moe_pdd, RequestId{2}).cached_prefill_tokens > 0 &&
               moe_pdd.kv_cache_transfers.size() == workload.size(),
           "MoE execution must preserve cache hits and PDD transfers");

    SimulationConfig pressure = load_config("fixed_parallel_colocation.json");
    pressure.cluster().parallelism.num_replicas = 1;
    pressure.cluster().parallelism.data_parallel_size = 1;
    pressure.cluster().scheduler.num_blocks = 13;
    const SimulationOutput evicting = run_simulation(pressure, workload);
    expect(evicting.aggregate.prefix_cache.evicted_blocks > 0 &&
               evicting.requests.size() == workload.size(),
           "tight analytical capacity must exercise session-range eviction and "
           "still quiesce");
}

std::vector<WorkloadRequest> make_stress_workload(std::size_t request_count,
                                                  std::uint64_t sessions) {
    std::vector<WorkloadRequest> workload;
    workload.reserve(request_count);
    for (std::size_t index = 0; index < request_count; ++index) {
        WorkloadRequest value{};
        value.request_id = RequestId{index};
        if (index < sessions) {
            value.session_start_at = frontier::SimTime::from_seconds(
                static_cast<double>(index) * 0.01);
        } else {
            value.think_time = frontier::SimTime::from_seconds(0.01);
        }
        value.num_prefill_tokens = 1 + index % 4;
        value.num_decode_tokens = 1 + index % 3;
        value.session_id = frontier::SessionId{index % sessions};
        value.session_turn_index = index / sessions;
        workload.push_back(value);
    }
    return workload;
}

void test_dense_online_offline_stress_matrix() {
    const auto workload = make_stress_workload(120, 20);
    SimulationConfig online = load_config("fixed_parallel_colocation.json");
    online.cluster().scheduler.num_blocks = 64;
    const SimulationOutput online_output = run_simulation(online, workload);
    expect(online_output.requests.size() == workload.size() &&
               online_output.aggregate.prefix_cache.hit_blocks > 0,
           "multi-target online stress workload must hit and drain");

    SimulationConfig offline = online;
    offline.simulation_mode = frontier::config::SimulationMode::kOffline;
    const SimulationOutput offline_output = run_simulation(offline, workload);
    expect(offline_output.requests.size() == workload.size(),
           "offline simultaneous-arrival stress workload must drain");

    SimulationConfig pdd = load_config("fixed_sequential_pdd.json");
    const SimulationOutput pdd_output = run_simulation(pdd, workload);
    expect(pdd_output.requests.size() == workload.size() &&
               pdd_output.kv_cache_transfers.size() == workload.size() &&
               pdd_output.aggregate.prefix_cache.hit_blocks > 0,
           "sequential PDD stress workload must hit, transfer, and drain");
}

void test_pdd_same_session_waits_for_terminal_decode() {
    SimulationConfig config = load_config("fixed_sequential_pdd.json");
    std::vector<WorkloadRequest> workload;
    for (std::uint64_t turn = 0; turn < 2; ++turn) {
        WorkloadRequest value{};
        value.request_id = RequestId{turn};
        if (turn == 0) {
            value.session_start_at = frontier::SimTime::from_seconds(0.0);
        } else {
            value.think_time = frontier::SimTime::from_seconds(0.0);
        }
        value.num_prefill_tokens = turn == 0 ? 8 : 4;
        value.num_decode_tokens = 3;
        value.session_id = SessionId{77};
        value.session_turn_index = turn;
        workload.push_back(value);
    }
    const SimulationOutput output = run_simulation(config, workload);
    const auto successor_prefill = std::find_if(
        output.batches.begin(), output.batches.end(), [](const auto &batch) {
            return batch.cluster_type == frontier::ClusterType::kPrefill &&
                   std::find(batch.request_ids.begin(), batch.request_ids.end(),
                             RequestId{1}) != batch.request_ids.end();
        });
    expect(successor_prefill != output.batches.end() &&
               successor_prefill->scheduled_at.seconds() >=
                   request(output, RequestId{0}).completed_at.seconds(),
           "PDD successor PREFILL must wait for predecessor terminal DECODE");
}

} // namespace

int main() {
    int failures = 0;
    failures +=
        frontier::test::run("co-location and PDD session prefix cache",
                            test_colocation_and_pdd_session_prefix_cache);
    failures +=
        frontier::test::run("multi-target cache requires sticky affinity",
                            test_multi_target_cache_requires_sticky_affinity);
    failures +=
        frontier::test::run("direct simulator entry materializes sessions",
                            test_direct_simulator_entry_materializes_sessions);
    failures += frontier::test::run("MoE and eviction pressure configs",
                                    test_moe_and_eviction_pressure_configs);
    failures += frontier::test::run("dense online/offline stress matrix",
                                    test_dense_online_offline_stress_matrix);
    failures +=
        frontier::test::run("PDD same-session terminal serialization",
                            test_pdd_same_session_waits_for_terminal_decode);
    return failures == 0 ? 0 : 1;
}
