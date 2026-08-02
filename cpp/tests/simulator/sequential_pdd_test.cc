#include "frontier/config/config.h"
#include "frontier/metrics/output_contract.h"
#include "frontier/request_generator/workload.h"
#include "frontier/simulator/simulator.h"
#include "tests/test_support.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <numeric>
#include <set>
#include <string>
#include <vector>

#ifndef FRONTIER_TEST_FIXTURE_DIR
#error "FRONTIER_TEST_FIXTURE_DIR must be defined for simulator tests"
#endif

namespace {

using frontier::ClusterType;
using frontier::EventType;
using frontier::config::ParallelismConfig;
using frontier::config::parse_simulation_config_json;
using frontier::config::SimulationConfig;
using frontier::config::SimulationMode;
using frontier::metrics::serialize_simulation_output_json;
using frontier::request_generator::parse_workload_csv;
using frontier::simulator::run_simulation;
using frontier::test::expect;
using frontier::test::read_text_file;

SimulationConfig load_config() {
    const std::filesystem::path fixture_root{FRONTIER_TEST_FIXTURE_DIR};
    return parse_simulation_config_json(
        read_text_file(fixture_root / "config/fixed_sequential_pdd.json"));
}

std::string load_small_workload() {
    const std::filesystem::path fixture_root{FRONTIER_TEST_FIXTURE_DIR};
    return read_text_file(fixture_root / "workloads/step3_pdd_small.csv");
}

void test_online_prefill_transfer_decode_lifecycle() {
    const SimulationConfig config = load_config();
    const auto output =
        run_simulation(config, parse_workload_csv(load_small_workload()));

    expect(output.schema_version == frontier::config::kSchemaVersion &&
               output.requests.size() == 2 &&
               output.kv_cache_transfers.size() == 2,
           "PDD must complete one transfer per request");
    for (const auto &request : output.requests) {
        expect(
            request.prefill_replica_id.valid() &&
                request.prefill_dp_id.valid() &&
                request.decode_replica_id.valid() &&
                request.decode_dp_id.valid() && request.transfer_id.valid() &&
                request.kv_cache_transfer_start_time.valid() &&
                request.kv_cache_transfer_end_time.valid() &&
                request.decode_arrived_at.valid(),
            "PDD request metrics must expose both owners and transfer times");
        expect(request.prefill_completed_at <=
                       request.kv_cache_transfer_start_time &&
                   request.kv_cache_transfer_start_time <
                       request.kv_cache_transfer_end_time &&
                   request.kv_cache_transfer_end_time ==
                       request.decode_arrived_at &&
                   request.decode_arrived_at < request.completed_at,
               "request lifecycle must be PREFILL -> transfer -> DECODE");
    }
    expect(std::all_of(output.kv_cache_transfers.begin(),
                       output.kv_cache_transfers.end(),
                       [](const auto &transfer) {
                           return transfer.source_cluster_type ==
                                      ClusterType::kPrefill &&
                                  transfer.target_cluster_type ==
                                      ClusterType::kDecode &&
                                  transfer.started_at < transfer.completed_at;
                       }),
           "transfer records must retain their PDD direction and interval");
}

void test_offline_barrier_and_deterministic_drain() {
    SimulationConfig config = load_config();
    config.simulation_mode = SimulationMode::kOffline;
    const auto workload = parse_workload_csv(load_small_workload());
    const auto output = run_simulation(config, workload);

    const auto final_transfer = std::max_element(
        output.kv_cache_transfers.begin(), output.kv_cache_transfers.end(),
        [](const auto &lhs, const auto &rhs) {
            return lhs.completed_at < rhs.completed_at;
        });
    expect(final_transfer != output.kv_cache_transfers.end(),
           "offline workload must contain completed transfers");
    const auto first_decode_schedule = std::find_if(
        output.event_trace.begin(), output.event_trace.end(),
        [](const auto &event) {
            return event.type() == EventType::kClusterSchedule &&
                   event.template as<frontier::ClusterSchedulePayload>()
                           .cluster_type == ClusterType::kDecode;
        });
    expect(first_decode_schedule != output.event_trace.end() &&
               first_decode_schedule->time >= final_transfer->completed_at,
           "offline decode scheduling must wait until all transfers drain");
    expect(std::none_of(output.event_trace.begin(), output.event_trace.end(),
                        [](const auto &event) {
                            return event.type() == EventType::kRequestArrival;
                        }),
           "offline mode must preload requests instead of replaying arrivals");

    const std::string first = serialize_simulation_output_json(output);
    const std::string second =
        serialize_simulation_output_json(run_simulation(config, workload));
    expect(first == second, "identical offline PDD runs must be byte stable");
}

void test_prefill_and_decode_use_independent_topologies() {
    SimulationConfig config = load_config();
    auto &prefill = config.pdd().clusters.prefill;
    auto &decode = config.pdd().clusters.decode;
    prefill.parallelism = [&]() {
        ParallelismConfig value{};
        value.num_replicas = 2;
        value.tensor_parallel_size = 2;
        value.pipeline_parallel_size = 2;
        value.data_parallel_size = 2;
        return value;
    }();
    prefill.execution_model.fixed.stage_latencies_ms = {0.2, 0.2};
    decode.parallelism = [&]() {
        ParallelismConfig value{};
        value.num_replicas = 1;
        value.tensor_parallel_size = 4;
        value.pipeline_parallel_size = 2;
        value.data_parallel_size = 2;
        return value;
    }();
    decode.execution_model.fixed.stage_latencies_ms = {0.3, 0.3};

    const auto output = run_simulation(
        config,
        parse_workload_csv("arrived_at,num_prefill_tokens,num_decode_tokens\n"
                           "0,1,2\n"
                           "0,2,2\n"
                           "0,3,2\n"
                           "0,4,2\n"
                           "0,1,2\n"
                           "0,2,2\n"
                           "0,3,2\n"
                           "0,4,2\n"));

    std::set<std::uint64_t> prefill_replicas;
    std::set<std::uint64_t> prefill_dps;
    std::set<std::uint64_t> decode_replicas;
    std::set<std::uint64_t> decode_dps;
    for (const auto &request : output.requests) {
        prefill_replicas.insert(request.prefill_replica_id.value());
        prefill_dps.insert(request.prefill_dp_id.value());
        decode_replicas.insert(request.decode_replica_id.value());
        decode_dps.insert(request.decode_dp_id.value());
    }
    expect(prefill_replicas == std::set<std::uint64_t>{0, 1} &&
               prefill_dps == std::set<std::uint64_t>{0, 1},
           "prefill routing must exercise its 2x2 replica/DP topology");
    expect(decode_replicas == std::set<std::uint64_t>{0} &&
               decode_dps == std::set<std::uint64_t>{0, 1},
           "decode routing must use its independent 1x2 topology");

    for (const ClusterType cluster_type :
         {ClusterType::kPrefill, ClusterType::kDecode}) {
        const bool saw_terminal_stage =
            std::any_of(output.batch_stages.begin(), output.batch_stages.end(),
                        [cluster_type](const auto &stage) {
                            return stage.cluster_type == cluster_type &&
                                   stage.stage_id.value() == 1;
                        });
        expect(saw_terminal_stage,
               "each cluster must execute all configured PP stages");
    }
}

void test_decode_preemption_recovers_to_quiescence() {
    SimulationConfig config = load_config();
    config.pdd().clusters.decode.scheduler.num_blocks = 3;
    const auto output = run_simulation(
        config,
        parse_workload_csv("arrived_at,num_prefill_tokens,num_decode_tokens\n"
                           "0,3,3\n"
                           "0,4,2\n"
                           "0,3,3\n"
                           "0,4,2\n"));

    const std::uint64_t preemptions = std::accumulate(
        output.requests.begin(), output.requests.end(), std::uint64_t{0},
        [](std::uint64_t total, const auto &request) {
            return total + request.preemption_count;
        });
    expect(output.requests.size() == 4 &&
               output.kv_cache_transfers.size() == 4 && preemptions > 0,
           "decode memory pressure must preempt and still drain every request");
}

void test_closed_loop_releases_one_replacement_per_completion() {
    SimulationConfig config = load_config();
    config.closed_loop_max_concurrency = 2;
    const auto output = run_simulation(
        config,
        parse_workload_csv("arrived_at,num_prefill_tokens,num_decode_tokens\n"
                           "0,3,3\n"
                           "100,4,2\n"
                           "200,3,3\n"
                           "300,4,2\n"));

    std::vector<const frontier::metrics::RequestMetricsRecord *> requests;
    requests.reserve(output.requests.size());
    for (const auto &request : output.requests) {
        requests.push_back(&request);
    }
    std::sort(requests.begin(), requests.end(), [](const auto *left,
                                                   const auto *right) {
        return left->request_id.value() < right->request_id.value();
    });
    expect(requests.size() == 4 &&
               requests.at(0)->arrived_at ==
                   frontier::SimTime::from_seconds(0.0) &&
               requests.at(1)->arrived_at ==
                   frontier::SimTime::from_seconds(0.0) &&
               requests.at(2)->arrived_at <
                   frontier::SimTime::from_seconds(100.0) &&
               requests.at(3)->arrived_at <
                   frontier::SimTime::from_seconds(100.0),
           "closed loop must ignore later trace times and release replacements");
    for (const auto *request : requests) {
        const auto active = std::count_if(
            requests.begin(), requests.end(), [request](const auto *candidate) {
                return candidate->arrived_at <= request->arrived_at &&
                       candidate->completed_at > request->arrived_at;
            });
        expect(active == 2,
               "closed loop must replenish exactly two in-flight requests");
    }
}

} // namespace

int main() {
    int failures = 0;
    failures += frontier::test::run(
        "online PDD lifecycle", test_online_prefill_transfer_decode_lifecycle);
    failures +=
        frontier::test::run("offline PDD barrier and deterministic drain",
                            test_offline_barrier_and_deterministic_drain);
    failures +=
        frontier::test::run("independent PDD topologies",
                            test_prefill_and_decode_use_independent_topologies);
    failures +=
        frontier::test::run("decode preemption reaches quiescence",
                            test_decode_preemption_recovers_to_quiescence);
    failures += frontier::test::run(
        "closed-loop request replenishment",
        test_closed_loop_releases_one_replacement_per_completion);
    return failures == 0 ? 0 : 1;
}
