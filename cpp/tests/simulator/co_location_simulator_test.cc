#include "frontier/config/config.h"
#include "frontier/metrics/output_contract.h"
#include "frontier/request_generator/workload.h"
#include "frontier/simulator/co_location_simulator.h"
#include "tests/test_support.h"

#include <algorithm>
#include <filesystem>
#include <string>

#include <nlohmann/json.hpp>

#ifndef FRONTIER_TEST_FIXTURE_DIR
#error "FRONTIER_TEST_FIXTURE_DIR must be defined for simulator tests"
#endif

namespace {

using frontier::config::ExecutionModelType;
using frontier::config::SimulationConfig;
using frontier::config::parse_simulation_config_json;
using frontier::metrics::serialize_simulation_output_json;
using frontier::request_generator::parse_workload_csv;
using frontier::simulator::CoLocationSimulationError;
using frontier::simulator::run_co_location_simulation;
using frontier::test::expect;
using frontier::test::expect_throws;
using frontier::test::read_text_file;
using Json = nlohmann::json;

SimulationConfig load_config() {
  const std::filesystem::path fixture_root{
      FRONTIER_TEST_FIXTURE_DIR};
  return parse_simulation_config_json(read_text_file(
      fixture_root / "config/step2_fixed_colocation.json"));
}

void test_canonical_single_request_lifecycle() {
  const SimulationConfig config = load_config();
  const auto workload = parse_workload_csv(
      "arrived_at,num_prefill_tokens,num_decode_tokens\n"
      "0,32,1\n");
  const auto output = run_co_location_simulation(config, workload);
  expect(
      output.requests.size() == 1 && output.batches.size() == 1,
      "single decode-length-one request needs one batch");
  expect(
      output.requests[0].prefill_completed_at.seconds() == 0.001 &&
          output.requests[0].completed_at.seconds() == 0.001,
      "fixed latency must define canonical TTFT and E2E");
  expect(
      output.requests[0].num_processed_tokens == 33,
      "request output must expose final token progress");
  const Json json =
      Json::parse(serialize_simulation_output_json(output));
  expect(
      json.at("schema_version") == 2 &&
          json.at("run").at("metrics_semantics") == "canonical",
      "scheduler output must use canonical schema v2");
}

void test_arrival_during_batch_joins_next_continuous_batch() {
  const SimulationConfig config = load_config();
  const auto workload = parse_workload_csv(
      "arrived_at,num_prefill_tokens,num_decode_tokens\n"
      "0,4,2\n"
      "0.0005,2,1\n");
  const auto output = run_co_location_simulation(config, workload);
  expect(
      output.batches.size() == 2,
      "mid-flight arrival must wait for next scheduler iteration");
  expect(
      output.batches[1].request_ids.size() == 2 &&
          output.batches[1].request_ids[0].value() == 1 &&
          output.batches[1].request_ids[1].value() == 0,
      "new admission must precede running decode in batch output order");
  expect(
      output.requests.size() == 2,
      "both requests must complete exactly once");
}

void test_decode_iterations_and_chunked_prefill() {
  SimulationConfig decode_config = load_config();
  const auto decode_output = run_co_location_simulation(
      decode_config,
      parse_workload_csv(
          "arrived_at,num_prefill_tokens,num_decode_tokens\n"
          "0,4,3\n"));
  expect(
      decode_output.batches.size() == 3,
      "decode length three needs prefill plus two decode batches");
  expect(
      decode_output.requests[0].completed_at.seconds() == 0.003,
      "fixed latency must accumulate over decode iterations");

  SimulationConfig chunked_config = load_config();
  chunked_config.scheduler->max_tokens_in_batch = 4;
  chunked_config.scheduler->enable_chunked_prefill = true;
  const auto chunked_output = run_co_location_simulation(
      chunked_config,
      parse_workload_csv(
          "arrived_at,num_prefill_tokens,num_decode_tokens\n"
          "0,10,1\n"));
  expect(
      chunked_output.batches.size() == 3 &&
          chunked_output.batches[0].scheduled_tokens[0] == 4 &&
          chunked_output.batches[1].scheduled_tokens[0] == 4 &&
          chunked_output.batches[2].scheduled_tokens[0] == 2,
      "chunked prefill must preserve exact prompt token sum");
}

void test_preemption_recovers_and_conserves_blocks() {
  SimulationConfig config = load_config();
  config.scheduler->block_size = 4;
  config.scheduler->num_blocks = 2;
  config.scheduler->max_tokens_in_batch = 8;
  config.scheduler->enable_preemption = true;
  const auto output = run_co_location_simulation(
      config,
      parse_workload_csv(
          "arrived_at,num_prefill_tokens,num_decode_tokens\n"
          "0,3,3\n"
          "0,4,2\n"));
  expect(
      output.requests.size() == 2,
      "preemption workload must complete both requests");
  const auto request0 = std::find_if(
      output.requests.begin(),
      output.requests.end(),
      [](const auto& record) {
        return record.request_id.value() == 0;
      });
  expect(
      request0 != output.requests.end() &&
          request0->preemption_count == 1,
      "FCFS victim must report one preemption");
}

void test_analytical_execution_produces_batch_diagnostics() {
  SimulationConfig config = load_config();
  config.execution_model->type = ExecutionModelType::kAnalytical;
  const auto output = run_co_location_simulation(
      config,
      parse_workload_csv(
          "arrived_at,num_prefill_tokens,num_decode_tokens\n"
          "0,128,1\n"));
  expect(
      output.batches.size() == 1 &&
          output.batches[0].predicted_execution_ms > 0.0,
      "analytical model must determine batch completion time");
  expect(
      output.analytical_diagnostics.size() == 1,
      "analytical batch must emit component diagnostics");
}

void test_unschedulable_workload_fails_without_hanging() {
  SimulationConfig config = load_config();
  config.scheduler->max_tokens_in_batch = 4;
  config.scheduler->enable_chunked_prefill = false;
  expect_throws<CoLocationSimulationError>(
      [&config] {
        static_cast<void>(run_co_location_simulation(
            config,
            parse_workload_csv(
                "arrived_at,num_prefill_tokens,num_decode_tokens\n"
                "0,8,1\n")));
      },
      "quiescent unschedulable request must fail deterministically");
}

void test_repeated_runs_are_byte_stable() {
  const SimulationConfig config = load_config();
  const auto workload = parse_workload_csv(
      "arrived_at,num_prefill_tokens,num_decode_tokens\n"
      "0,4,3\n"
      "0,2,1\n"
      "0.0005,3,2\n");
  const std::string first = serialize_simulation_output_json(
      run_co_location_simulation(config, workload));
  const std::string second = serialize_simulation_output_json(
      run_co_location_simulation(config, workload));
  expect(
      first == second,
      "identical Step 2 runs must produce byte-stable JSON");
}

void test_online_mode_uses_same_trace_semantics() {
  SimulationConfig config = load_config();
  config.simulation_mode =
      frontier::config::SimulationMode::kOnline;
  const auto output = run_co_location_simulation(
      config,
      parse_workload_csv(
          "arrived_at,num_prefill_tokens,num_decode_tokens\n"
          "0.25,4,2\n"));
  expect(
      output.run.simulation_mode ==
          frontier::config::SimulationMode::kOnline,
      "online metadata must be preserved");
  expect(
      output.requests[0].prefill_completed_at.seconds() == 0.251 &&
          output.requests[0].completed_at.seconds() == 0.252,
      "online mode must use the same trace-driven lifecycle");
}

}  // namespace

int main() {
  int failures = 0;
  failures += frontier::test::run(
      "canonical single request lifecycle",
      test_canonical_single_request_lifecycle);
  failures += frontier::test::run(
      "arrival during batch joins next continuous batch",
      test_arrival_during_batch_joins_next_continuous_batch);
  failures += frontier::test::run(
      "decode iterations and chunked prefill",
      test_decode_iterations_and_chunked_prefill);
  failures += frontier::test::run(
      "preemption recovers and conserves blocks",
      test_preemption_recovers_and_conserves_blocks);
  failures += frontier::test::run(
      "analytical execution produces diagnostics",
      test_analytical_execution_produces_batch_diagnostics);
  failures += frontier::test::run(
      "unschedulable workload fails without hanging",
      test_unschedulable_workload_fails_without_hanging);
  failures += frontier::test::run(
      "repeated runs are byte stable",
      test_repeated_runs_are_byte_stable);
  failures += frontier::test::run(
      "online mode uses trace semantics",
      test_online_mode_uses_same_trace_semantics);
  return failures == 0 ? 0 : 1;
}
