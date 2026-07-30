#include "frontier/metrics/output_contract.h"
#include "tests/test_support.h"

#include <string>

#include <nlohmann/json.hpp>

namespace {

using frontier::BatchId;
using frontier::DataParallelId;
using frontier::Event;
using frontier::EventPayload;
using frontier::EventSequence;
using frontier::EventType;
using frontier::IterationId;
using frontier::RequestId;
using frontier::ReplicaId;
using frontier::SimTime;
using frontier::config::SimulationMode;
using frontier::config::SystemArchitecture;
using frontier::metrics::BatchMetricsRecord;
using frontier::metrics::MetricsSemantics;
using frontier::metrics::RequestMetricsRecord;
using frontier::metrics::RunMetadata;
using frontier::metrics::SchedulerDecisionRecord;
using frontier::metrics::SchedulerTraceRecord;
using frontier::metrics::SimulationOutput;
using frontier::metrics::serialize_request_metrics_csv;
using frontier::metrics::serialize_simulation_output_json;
using frontier::test::expect;
using Json = nlohmann::json;

SimulationOutput make_output() {
  EventPayload arrival;
  arrival.request_id = RequestId{0};
  return SimulationOutput{
      .schema_version = 2,
      .run = RunMetadata{
          .run_id = "output-v2",
          .simulation_mode = SimulationMode::kOnline,
          .system_architecture = SystemArchitecture::kCoLocation,
          .metrics_semantics = MetricsSemantics::kCanonical,
      },
      .requests = {
          RequestMetricsRecord{
              .request_id = RequestId{0},
              .arrived_at = SimTime::from_seconds(0.0),
              .prefill_completed_at = SimTime::from_seconds(0.001),
              .completed_at = SimTime::from_seconds(0.002),
              .first_scheduled_at = SimTime::from_seconds(0.0),
              .first_token_completed_at =
                  SimTime::from_seconds(0.001),
              .num_processed_tokens = 6,
              .preemption_count = 1,
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
      .batches = {
          BatchMetricsRecord{
              .batch_id = BatchId{0},
              .iteration_id = IterationId{0},
              .scheduled_at = SimTime::from_seconds(0.0),
              .completed_at = SimTime::from_seconds(0.001),
              .request_ids = {RequestId{0}},
              .scheduled_tokens = {4},
              .total_scheduled_tokens = 4,
              .num_prefill_tokens = 4,
              .num_decode_tokens = 0,
              .predicted_execution_ms = 1.0,
          },
      },
      .batch_stages = {},
      .scheduler_trace = {
          SchedulerTraceRecord{
              .iteration_id = IterationId{0},
              .simulation_time = SimTime::from_seconds(0.0),
              .token_budget_before = 8,
              .token_budget_after = 4,
              .available_blocks_before = 8,
              .available_blocks_after = 7,
              .waiting_count_before = 1,
              .waiting_count_after = 0,
              .running_count_before = 0,
              .running_count_after = 1,
              .preempted_count = 0,
              .decisions = {
                  SchedulerDecisionRecord{
                      .decision_result = "ADMISSION",
                      .request_id = RequestId{0},
                      .num_tokens = 4,
                      .token_budget_after = 4,
                      .available_blocks_after = 7,
                  },
              },
              .batch_request_ids = {RequestId{0}},
              .request_num_tokens = {4},
          },
      },
      .event_trace = {
          Event{
              .time = SimTime::from_seconds(0.0),
              .sequence = EventSequence{1},
              .type = EventType::kRequestArrival,
              .payload = arrival,
          },
      },
      .analytical_diagnostics = {},
      .kv_cache_transfers = {},
  };
}

void test_schema_v2_json_contract() {
  const Json json =
      Json::parse(serialize_simulation_output_json(make_output()));
  expect(json.at("schema_version") == 2, "v2 version must serialize");
  expect(
      json.at("requests").at(0).at("first_scheduled_at_s") == 0.0,
      "canonical scheduling timestamp must serialize");
  expect(
      json.at("batches").at(0).at("request_ids") ==
          Json::array({0}),
      "batch request order must be stable");
  expect(
      json.at("scheduler_trace")
              .at(0)
              .at("decisions")
              .at(0)
              .at("decision_result") == "ADMISSION",
      "scheduler decision sequence must serialize");
}

void test_schema_v2_csv_contract() {
  const std::string csv =
      serialize_request_metrics_csv(make_output().requests, 2);
  expect(
      csv.starts_with(
          "request_id,arrived_at_s,first_scheduled_at_s,"),
      "v2 CSV must expose canonical scheduling fields");
  expect(
      csv.find(",6,1\n") != std::string::npos,
      "v2 CSV must include final progress and preemption count");
}

}  // namespace

int main() {
  int failures = 0;
  failures += frontier::test::run(
      "schema v2 JSON contract",
      test_schema_v2_json_contract);
  failures += frontier::test::run(
      "schema v2 CSV contract",
      test_schema_v2_csv_contract);
  return failures == 0 ? 0 : 1;
}
