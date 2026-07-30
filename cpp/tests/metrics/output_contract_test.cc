#include "frontier/metrics/output_contract.h"
#include "tests/test_support.h"

#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

namespace {

using frontier::Event;
using frontier::EventPayload;
using frontier::EventSequence;
using frontier::EventType;
using frontier::DataParallelId;
using frontier::RequestId;
using frontier::ReplicaId;
using frontier::SimTime;
using frontier::config::SimulationMode;
using frontier::config::SystemArchitecture;
using frontier::metrics::AnalyticalDiagnostic;
using frontier::metrics::RequestMetricsRecord;
using frontier::metrics::RunMetadata;
using frontier::metrics::SimulationOutput;
using frontier::metrics::serialize_request_metrics_csv;
using frontier::metrics::serialize_simulation_output_json;
using frontier::test::expect;
using frontier::test::expect_throws;

SimulationOutput make_output() {
  EventPayload payload;
  payload.request_id = RequestId{0};

  return SimulationOutput{
      .schema_version = 1,
      .run = RunMetadata{
          .run_id = "output-contract",
          .simulation_mode = SimulationMode::kOffline,
          .system_architecture = SystemArchitecture::kCoLocation,
      },
      .requests = {
          RequestMetricsRecord{
              .request_id = RequestId{0},
              .arrived_at = SimTime::from_seconds(0.0),
              .prefill_completed_at = SimTime::from_seconds(0.25),
              .completed_at = SimTime::from_seconds(1.0),
              .first_scheduled_at = std::nullopt,
              .first_token_completed_at = std::nullopt,
              .num_processed_tokens = 0,
              .preemption_count = 0,
              .tokens_at_preemption = {},
              .replica_id = ReplicaId{0},
              .dp_id = DataParallelId{0},
              .prefill_replica_id = std::nullopt,
              .prefill_dp_id = std::nullopt,
              .decode_replica_id = std::nullopt,
              .decode_dp_id = std::nullopt,
              .transfer_id = std::nullopt,
              .kv_cache_transfer_start_time = std::nullopt,
              .kv_cache_transfer_end_time = std::nullopt,
              .decode_arrived_at = std::nullopt,
              .kv_cache_transfer_size_bytes = 0,
          },
      },
      .batches = {},
      .batch_stages = {},
      .scheduler_trace = {},
      .event_trace = {
          Event{
              .time = SimTime::from_seconds(0.0),
              .sequence = EventSequence{1},
              .type = EventType::kRequestArrival,
              .payload = payload,
          },
      },
      .analytical_diagnostics = {
          AnalyticalDiagnostic{
              .name = "fixture",
              .values = {
                  {"latency_ms", 0.25},
                  {"tokens", 32.0},
              },
          },
      },
      .kv_cache_transfers = {},
  };
}

void test_json_contract_is_versioned_and_ordered() {
  const std::string json = serialize_simulation_output_json(make_output());
  expect(
      json.find("\"schema_version\": 1") != std::string::npos,
      "output schema version must be present");
  expect(
      json.find("\"timestamp_unit\": \"seconds\"") !=
          std::string::npos,
      "timestamp unit must be explicit");
  expect(
      json.find("\"latency_unit\": \"milliseconds\"") !=
          std::string::npos,
      "latency unit must be explicit");
  expect(
      json.find("\"metrics_semantics\": \"canonical\"") !=
          std::string::npos,
      "canonical metrics semantics must be explicit");
  expect(
      json.find("\"ttft_ms\": 250.0") != std::string::npos,
      "TTFT must use milliseconds");
  expect(
      json.find("\"e2e_ms\": 1000.0") != std::string::npos,
      "E2E must use milliseconds");
  expect(
      json.find("\"completed_request_ids\"") <
          json.find("\"requests\""),
      "top-level output field order must be stable");
  expect(
      json.find("\"time_s\"") < json.find("\"sequence\""),
      "event fields must have stable order");
}

void test_request_metrics_csv_contract() {
  const auto output = make_output();
  const std::string csv = serialize_request_metrics_csv(output.requests);
  expect(
      csv ==
          "request_id,arrived_at_s,prefill_completed_at_s,completed_at_s,"
          "ttft_ms,e2e_ms\n"
          "0,0,0.25,1,250,1000\n",
      "request metrics CSV must be byte-stable");
}

void test_invalid_output_is_rejected() {
  auto duplicate = make_output();
  duplicate.requests.push_back(duplicate.requests.front());
  expect_throws<std::invalid_argument>(
      [&duplicate] {
        static_cast<void>(serialize_simulation_output_json(duplicate));
      },
      "duplicate request IDs must be rejected");

  auto reversed = make_output();
  reversed.requests.front().completed_at =
      SimTime::from_seconds(0.1);
  expect_throws<std::invalid_argument>(
      [&reversed] {
        static_cast<void>(serialize_request_metrics_csv(
            reversed.requests));
      },
      "completion before prefill must be rejected");

  auto nonfinite = make_output();
  nonfinite.analytical_diagnostics.front().values.front().second =
      std::numeric_limits<double>::infinity();
  expect_throws<std::invalid_argument>(
      [&nonfinite] {
        static_cast<void>(serialize_simulation_output_json(nonfinite));
      },
      "nonfinite diagnostics must be rejected");
}

}  // namespace

int main() {
  int failures = 0;
  failures += frontier::test::run(
      "JSON contract is versioned and ordered",
      test_json_contract_is_versioned_and_ordered);
  failures += frontier::test::run(
      "request metrics CSV contract",
      test_request_metrics_csv_contract);
  failures += frontier::test::run(
      "invalid output is rejected",
      test_invalid_output_is_rejected);
  return failures == 0 ? 0 : 1;
}
