#include "frontier/metrics/output_contract.h"
#include "tests/test_support.h"

#include <string>

#include <nlohmann/json.hpp>

namespace {

using frontier::BatchId;
using frontier::DataParallelId;
using frontier::Event;
using frontier::EventSequence;
using frontier::IterationId;
using frontier::RequestArrivalPayload;
using frontier::RequestId;
using frontier::ReplicaId;
using frontier::SimTime;
using frontier::config::SimulationMode;
using frontier::config::SystemArchitecture;
using frontier::metrics::BatchMetricsRecord;
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
  return SimulationOutput{
      .schema_version = 1,
      .run = RunMetadata{
          .run_id = "output-contract",
          .simulation_mode = SimulationMode::kOnline,
          .system_architecture = SystemArchitecture::kCoLocation,
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
              .prefill_replica_id = ReplicaId{},
              .prefill_dp_id = DataParallelId{},
              .decode_replica_id = ReplicaId{},
              .decode_dp_id = DataParallelId{},
              .transfer_id = frontier::TransferId{},
              .kv_cache_transfer_start_time = SimTime{},
              .kv_cache_transfer_end_time = SimTime{},
              .decode_arrived_at = SimTime{},
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
              SimTime::from_seconds(0.0),
              EventSequence{1},
              RequestArrivalPayload{
                  .request_id = RequestId{0},
                  .cluster_type =
                      frontier::ClusterType::kMonolithic,
              },
          },
      },
      .analytical_diagnostics = {},
      .kv_cache_transfers = {},
  };
}

void test_json_contract() {
  const Json json =
      Json::parse(serialize_simulation_output_json(make_output()));
  expect(json.at("schema_version") == 1, "version must serialize");
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

void test_csv_contract() {
  const std::string csv =
      serialize_request_metrics_csv(make_output().requests);
  expect(
      csv.starts_with(
          "request_id,arrived_at_s,first_scheduled_at_s,"),
      "CSV must expose canonical scheduling fields");
  expect(
      csv.find(",6,1,0,0\n") != std::string::npos,
      "CSV must include progress, preemption, and target fields");
}

}  // namespace

int main() {
  int failures = 0;
  failures += frontier::test::run(
      "JSON contract",
      test_json_contract);
  failures += frontier::test::run(
      "CSV contract",
      test_csv_contract);
  return failures == 0 ? 0 : 1;
}
