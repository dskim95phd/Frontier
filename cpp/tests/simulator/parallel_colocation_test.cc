#include "frontier/config/config.h"
#include "frontier/core/event.h"
#include "frontier/metrics/output_contract.h"
#include "frontier/request_generator/workload.h"
#include "frontier/simulator/simulator.h"
#include "tests/test_support.h"

#include <algorithm>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include <nlohmann/json.hpp>

#ifndef FRONTIER_TEST_FIXTURE_DIR
#error "FRONTIER_TEST_FIXTURE_DIR must be defined"
#endif

namespace {

using frontier::EventType;
using frontier::config::ConfigError;
using frontier::config::parse_simulation_config_json;
using frontier::config::serialize_simulation_config_json;
using frontier::config::SimulationConfig;
using frontier::metrics::BatchStageMetricsRecord;
using frontier::metrics::serialize_simulation_output_json;
using frontier::metrics::SimulationOutput;
using frontier::request_generator::parse_workload_csv;
using frontier::simulator::run_simulation;
using frontier::test::expect;
using frontier::test::expect_throws;
using frontier::test::read_text_file;
using Json = nlohmann::json;

const std::filesystem::path fixture_root{FRONTIER_TEST_FIXTURE_DIR};

SimulationConfig load_config(const char *name) {
    return parse_simulation_config_json(
        read_text_file(fixture_root / "config" / name));
}

auto load_workload() {
    return parse_workload_csv(
        read_text_file(fixture_root / "workloads/step25_parallel.csv"));
}

auto load_workload(const char *name) {
    return parse_workload_csv(
        read_text_file(fixture_root / "workloads" / name));
}

void test_config_round_trip_and_validation() {
    const SimulationConfig config =
        load_config("fixed_parallel_colocation.json");
    expect(config.schema_version == frontier::config::kSchemaVersion &&
               config.cluster().parallelism.num_replicas == 2 &&
               config.cluster().parallelism.data_parallel_size == 2 &&
               config.cluster().parallelism.pipeline_parallel_size == 2 &&
               config.cluster().parallelism.tensor_parallel_size == 2,
           "cluster topology must parse");
    expect(parse_simulation_config_json(
               serialize_simulation_config_json(config)) == config,
           "config must round-trip");

    std::string invalid =
        read_text_file(fixture_root / "config/fixed_parallel_colocation.json");
    const std::string from = "\"pipeline_parallel_size\": 2";
    invalid.replace(invalid.find(from), from.size(),
                    "\"pipeline_parallel_size\": 3");
    expect_throws<ConfigError>(
        [&invalid] {
            static_cast<void>(parse_simulation_config_json(invalid));
        },
        "PP that does not divide model layers must fail");
}

void test_round_robin_dp_replica_and_event_pipeline() {
    const SimulationOutput output = run_simulation(
        load_config("fixed_parallel_colocation.json"), load_workload());
    expect(output.requests.size() == 8 &&
               output.batches.size() * 2 == output.batch_stages.size(),
           "all requests and both PP stages must complete");

    std::map<std::uint64_t, std::pair<std::uint64_t, std::uint64_t>> owners;
    for (const auto &request : output.requests) {
        owners.emplace(
            request.request_id.value(),
            std::pair{request.replica_id.value(), request.dp_id.value()});
    }
    const std::map<std::uint64_t, std::pair<std::uint64_t, std::uint64_t>>
        expected_owners{
            {0, {0, 0}}, {1, {1, 0}}, {2, {0, 1}}, {3, {1, 1}},
            {4, {0, 0}}, {5, {1, 0}}, {6, {0, 0}}, {7, {1, 0}},
        };
    expect(owners == expected_owners,
           "target assignment must match Python replica-first batch routing");

    const std::set<EventType> required{
        EventType::kRequestArrival,    EventType::kGlobalSchedule,
        EventType::kClusterSchedule,   EventType::kReplicaSchedule,
        EventType::kBatchStageArrival, EventType::kReplicaStageSchedule,
        EventType::kBatchStageEnd,     EventType::kClusterBatchEnd,
        EventType::kGlobalBatchEnd,
    };
    std::set<EventType> seen;
    for (const auto &event : output.event_trace) {
        seen.insert(event.type());
    }
    for (const EventType type : required) {
        expect(seen.find(type) != seen.end(), "required event type is missing");
    }
}

void test_pp4_fill_drain_and_terminal_release() {
    const SimulationOutput output = run_simulation(
        load_config("fixed_pp4_colocation.json"), load_workload());
    expect(output.requests.size() == 8 &&
               output.batch_stages.size() == output.batches.size() * 4,
           "PP4 must execute every batch through all four stages");

    std::set<std::uint64_t> stages;
    std::size_t empty_schedule_iterations = 0;
    for (const auto &stage : output.batch_stages) {
        stages.insert(stage.stage_id.value());
    }
    for (const auto &trace : output.scheduler_trace) {
        if (trace.batch_request_ids.empty()) {
            ++empty_schedule_iterations;
        }
    }
    expect(stages == std::set<std::uint64_t>{0, 1, 2, 3},
           "PP4 output must contain every stage");
    expect(empty_schedule_iterations > 0,
           "PP4 terminal drain must emit an observable empty scheduler poll");
}

void test_dp_target_local_pressure_and_preemption() {
    const SimulationOutput output =
        run_simulation(load_config("fixed_dp2_pressure_colocation.json"),
                       load_workload("step25_parallel_pressure.csv"));
    std::map<std::uint64_t, std::uint64_t> preemptions_by_dp;
    for (const auto &request : output.requests) {
        preemptions_by_dp[request.dp_id.value()] += request.preemption_count;
    }
    expect(output.requests.size() == 4 && preemptions_by_dp.size() == 2 &&
               preemptions_by_dp.at(0) > 0 && preemptions_by_dp.at(1) > 0,
           "each DP target must resolve its own KV pressure by preemption");
}

void test_analytical_primary_tp_pp_matrix() {
    const SimulationConfig base =
        load_config("analytical_parallel_colocation.json");
    for (const std::uint64_t tp : {1U, 2U, 4U, 8U}) {
        for (const std::uint64_t pp : {1U, 2U, 4U}) {
            SimulationConfig config = base;
            config.cluster().parallelism.tensor_parallel_size = tp;
            config.cluster().parallelism.pipeline_parallel_size = pp;
            config.cluster().parallelism.data_parallel_size = 1;
            const SimulationOutput output =
                run_simulation(config, load_workload());
            expect(output.batch_stages.size() == output.batches.size() * pp,
                   "analytical TP/PP matrix must execute every PP stage");
            for (const auto &stage : output.batch_stages) {
                expect((tp == 1
                            ? stage.execution_time.tp_communication_ms == 0.0
                            : stage.execution_time.tp_communication_ms > 0.0),
                       "TP communication must be zero only for TP1");
                expect((stage.stage_id.index() + 1 == pp
                            ? stage.execution_time.pp_communication_ms == 0.0
                            : stage.execution_time.pp_communication_ms > 0.0),
                       "PP communication must appear only on non-final stages");
            }
        }
    }
}

void test_pipeline_serialization_overlap_and_fixed_timing() {
    const SimulationOutput output = run_simulation(
        load_config("fixed_parallel_colocation.json"), load_workload());
    using Key = std::tuple<std::uint64_t, std::uint64_t, std::uint64_t>;
    std::map<Key, std::vector<const BatchStageMetricsRecord *>> by_stage;
    for (const auto &stage : output.batch_stages) {
        const double expected = stage.stage_id.value() == 0 ? 1.0 : 3.0;
        expect(stage.execution_time.total_ms() == expected,
               "fixed stage latency must select by PP stage");
        by_stage[Key{stage.replica_id.value(), stage.dp_id.value(),
                     stage.stage_id.value()}]
            .push_back(&stage);
    }
    for (auto &[key, records] : by_stage) {
        static_cast<void>(key);
        std::sort(records.begin(), records.end(),
                  [](const auto *left, const auto *right) {
                      return left->started_at.seconds() <
                             right->started_at.seconds();
                  });
        for (std::size_t index = 1; index < records.size(); ++index) {
            expect(records[index - 1]->completed_at.seconds() <=
                       records[index]->started_at.seconds(),
                   "one stage must serialize its batches");
        }
    }

    bool found_cross_stage_overlap = false;
    for (const auto &left : output.batch_stages) {
        for (const auto &right : output.batch_stages) {
            if (left.replica_id != right.replica_id ||
                left.dp_id != right.dp_id || left.stage_id == right.stage_id) {
                continue;
            }
            if (left.started_at.seconds() < right.completed_at.seconds() &&
                right.started_at.seconds() < left.completed_at.seconds()) {
                found_cross_stage_overlap = true;
            }
        }
    }
    expect(found_cross_stage_overlap,
           "different PP stages must overlap on a loaded target");
}

void test_analytical_tp_pp_components_and_determinism() {
    const SimulationConfig config =
        load_config("analytical_parallel_colocation.json");
    const auto workload = load_workload();
    const SimulationOutput first = run_simulation(config, workload);
    const SimulationOutput second = run_simulation(config, workload);
    expect(serialize_simulation_output_json(first) ==
               serialize_simulation_output_json(second),
           "parallel run output must be byte stable");
    for (const auto &stage : first.batch_stages) {
        expect(stage.execution_time.dense_compute_ms > 0.0 &&
                   stage.execution_time.tp_communication_ms > 0.0,
               "TP4 analytical stages require compute and TP communication");
        if (stage.stage_id.value() == 0) {
            expect(stage.execution_time.pp_communication_ms > 0.0,
                   "non-final PP stage requires transfer time");
        } else {
            expect(stage.execution_time.pp_communication_ms == 0.0,
                   "final PP stage must not include transfer time");
        }
    }
    const Json json = Json::parse(serialize_simulation_output_json(first));
    expect(json.at("schema_version") == 1 &&
               json.at("batch_stages").size() == first.batch_stages.size() &&
               json.at("requests").at(0).contains("replica_id") &&
               json.at("scheduler_trace").at(0).contains("dp_id"),
           "output must expose target and stage contracts");
}

} // namespace

int main() {
    int failures = 0;
    failures += frontier::test::run("topology contract",
                                    test_config_round_trip_and_validation);
    failures +=
        frontier::test::run("round-robin DP/replica event pipeline",
                            test_round_robin_dp_replica_and_event_pipeline);
    failures += frontier::test::run(
        "PP serialization and overlap",
        test_pipeline_serialization_overlap_and_fixed_timing);
    failures += frontier::test::run("PP4 fill, drain, and terminal release",
                                    test_pp4_fill_drain_and_terminal_release);
    failures +=
        frontier::test::run("DP target-local pressure and preemption",
                            test_dp_target_local_pressure_and_preemption);
    failures +=
        frontier::test::run("analytical TP/PP components and determinism",
                            test_analytical_tp_pp_components_and_determinism);
    failures += frontier::test::run("analytical primary TP/PP matrix",
                                    test_analytical_primary_tp_pp_matrix);
    return failures == 0 ? 0 : 1;
}
