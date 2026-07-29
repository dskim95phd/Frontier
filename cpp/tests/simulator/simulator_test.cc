#include "frontier/config/config.h"
#include "frontier/metrics/output_contract.h"
#include "frontier/request_generator/workload.h"
#include "frontier/simulator/simulator.h"
#include "tests/test_support.h"

#include <filesystem>
#include <limits>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#ifndef FRONTIER_TEST_FIXTURE_DIR
#error "FRONTIER_TEST_FIXTURE_DIR must be defined for simulator tests"
#endif

namespace {

using frontier::EventType;
using frontier::RequestId;
using frontier::SimTime;
using frontier::config::SimulationConfig;
using frontier::config::parse_simulation_config_json;
using frontier::metrics::serialize_simulation_output_json;
using frontier::request_generator::WorkloadRequest;
using frontier::request_generator::parse_workload_csv;
using frontier::simulator::FoundationLifecycleOptions;
using frontier::simulator::FoundationSimulationError;
using frontier::simulator::run_foundation_lifecycle;
using frontier::test::expect;
using frontier::test::expect_throws;
using frontier::test::read_text_file;
using Json = nlohmann::json;

SimulationConfig load_config() {
  const std::filesystem::path fixture_root{
      FRONTIER_TEST_FIXTURE_DIR};
  return parse_simulation_config_json(
      read_text_file(
          fixture_root / "config/minimal_foundation_colocation.json"));
}

std::vector<WorkloadRequest> load_workload() {
  const std::filesystem::path fixture_root{
      FRONTIER_TEST_FIXTURE_DIR};
  return parse_workload_csv(
      read_text_file(
          fixture_root / "workloads/session_prefix.csv"));
}

void test_one_request_completes_exactly_once() {
  const SimulationConfig config = load_config();
  const auto workload = parse_workload_csv(
      "arrived_at,num_prefill_tokens,num_decode_tokens,session_id\n"
      "2,32,8,7\n");

  const auto output = run_foundation_lifecycle(
      config,
      workload,
      FoundationLifecycleOptions{.service_time_ms = 250.0});

  expect(output.requests.size() == 1, "one request must complete once");
  expect(output.event_trace.size() == 2, "one request needs two events");
  expect(
      output.event_trace[0].type == EventType::kRequestArrival,
      "arrival must be the first lifecycle event");
  expect(
      output.event_trace[1].type ==
          EventType::kFoundationCompletion,
      "foundation completion must be the second lifecycle event");

  const auto& record = output.requests.front();
  expect(record.request_id == RequestId{0}, "request ID must be retained");
  expect(
      record.arrived_at == SimTime::from_seconds(2.0),
      "arrival timestamp must be retained");
  expect(
      record.prefill_completed_at ==
          SimTime::from_seconds(2.25),
      "placeholder prefill completion must use fixed service time");
  expect(
      record.completed_at == record.prefill_completed_at,
      "foundation TTFT and E2E timestamps must be explicitly equal");
}

void test_checked_in_fixture_has_deterministic_trace() {
  const SimulationConfig config = load_config();
  const auto workload = load_workload();
  const FoundationLifecycleOptions options{
      .service_time_ms = 250.0,
  };

  const auto first =
      run_foundation_lifecycle(config, workload, options);
  const auto second =
      run_foundation_lifecycle(config, workload, options);
  const std::string first_json =
      serialize_simulation_output_json(first);
  const std::string second_json =
      serialize_simulation_output_json(second);

  expect(first_json == second_json, "repeated output must be byte-stable");
  expect(first.requests.size() == 3, "every fixture request must complete");
  expect(first.event_trace.size() == 6, "each request needs two events");

  const std::vector<std::uint64_t> expected_sequences{1, 4, 2, 5, 3, 6};
  for (std::size_t index = 0; index < expected_sequences.size(); ++index) {
    expect(
        first.event_trace[index].sequence.value() ==
            expected_sequences[index],
        "event trace must preserve deterministic queue ordering");
  }

  const Json json = Json::parse(first_json);
  expect(
      json.at("completed_request_ids") ==
          Json::array({0, 1, 2}),
      "completed request IDs must follow completion order");
  expect(
      json.at("requests").at(0).at("ttft_ms").get<double>() ==
          250.0,
      "foundation TTFT placeholder must be serialized in milliseconds");
  expect(
      json.at("requests").at(0).at("e2e_ms").get<double>() ==
          250.0,
      "foundation E2E placeholder must be serialized in milliseconds");
  expect(
      json.at("analytical_diagnostics")
              .empty(),
      "foundation lifecycle must not invent analytical diagnostics");
  expect(
      json.at("run")
              .at("metrics_semantics")
              .get<std::string>() == "foundation-placeholder",
      "output must identify placeholder lifecycle metrics");
}

void test_equal_time_events_use_creation_sequence() {
  const SimulationConfig config = load_config();
  const auto workload = parse_workload_csv(
      "arrived_at,num_prefill_tokens,num_decode_tokens,session_id\n"
      "0,32,8,7\n"
      "0,16,4,9\n");

  const auto output = run_foundation_lifecycle(
      config,
      workload,
      FoundationLifecycleOptions{.service_time_ms = 0.0});
  const std::vector<std::uint64_t> expected_sequences{1, 2, 3, 4};
  for (std::size_t index = 0; index < expected_sequences.size(); ++index) {
    expect(
        output.event_trace[index].sequence.value() ==
            expected_sequences[index],
        "equal-time events must remain in creation-sequence order");
  }
  expect(
      output.requests[0].request_id == RequestId{0} &&
          output.requests[1].request_id == RequestId{1},
      "equal-time completions must be deterministic");
}

void test_invalid_foundation_inputs_are_rejected() {
  const SimulationConfig config = load_config();
  auto workload = load_workload();
  workload[1].request_id = RequestId{0};
  expect_throws<FoundationSimulationError>(
      [&config, &workload] {
        static_cast<void>(
            run_foundation_lifecycle(config, workload));
      },
      "noncontiguous request IDs must be rejected");

  expect_throws<FoundationSimulationError>(
      [&config] {
        static_cast<void>(run_foundation_lifecycle(
            config,
            {},
            FoundationLifecycleOptions{
                .service_time_ms =
                    std::numeric_limits<double>::infinity(),
            }));
      },
      "nonfinite service time must be rejected");

  expect_throws<FoundationSimulationError>(
      [&config] {
        static_cast<void>(run_foundation_lifecycle(
            config,
            {},
            FoundationLifecycleOptions{.service_time_ms = -1.0}));
      },
      "negative service time must be rejected");

  auto prefix_config = config;
  prefix_config.prefix_cache.enabled = true;
  expect_throws<FoundationSimulationError>(
      [&prefix_config] {
        static_cast<void>(
            run_foundation_lifecycle(prefix_config, {}));
      },
      "unimplemented prefix caching must be rejected");

  auto pdd_config = config;
  pdd_config.system_architecture =
      frontier::config::SystemArchitecture::kPdDisaggregation;
  expect_throws<FoundationSimulationError>(
      [&pdd_config] {
        static_cast<void>(
            run_foundation_lifecycle(pdd_config, {}));
      },
      "unimplemented PDD lifecycle must be rejected");
}

}  // namespace

int main() {
  int failures = 0;
  failures += frontier::test::run(
      "one request completes exactly once",
      test_one_request_completes_exactly_once);
  failures += frontier::test::run(
      "checked-in fixture has deterministic trace",
      test_checked_in_fixture_has_deterministic_trace);
  failures += frontier::test::run(
      "equal-time events use creation sequence",
      test_equal_time_events_use_creation_sequence);
  failures += frontier::test::run(
      "invalid foundation inputs are rejected",
      test_invalid_foundation_inputs_are_rejected);
  return failures == 0 ? 0 : 1;
}
