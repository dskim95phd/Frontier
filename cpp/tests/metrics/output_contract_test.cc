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
using frontier::ReplicaId;
using frontier::RequestArrivalPayload;
using frontier::RequestId;
using frontier::SimTime;
using frontier::config::SimulationMode;
using frontier::config::SystemArchitecture;
using frontier::metrics::BatchMetricsRecord;
using frontier::metrics::RequestMetricsRecord;
using frontier::metrics::RunMetadata;
using frontier::metrics::SchedulerDecisionRecord;
using frontier::metrics::SchedulerTraceRecord;
using frontier::metrics::serialize_request_metrics_csv;
using frontier::metrics::serialize_simulation_output_json;
using frontier::metrics::SimulationOutput;
using frontier::test::expect;
using Json = nlohmann::json;

SimulationOutput make_output() {
    return [&]() {
        SimulationOutput value{};
        value.schema_version = 1;
        value.run = [&]() {
            RunMetadata value{};
            value.run_id = "output-contract";
            value.simulation_mode = SimulationMode::kOnline;
            value.system_architecture = SystemArchitecture::kCoLocation;
            return value;
        }();
        value.requests = {
            [&]() {
                RequestMetricsRecord value{};
                value.request_id = RequestId{0};
                value.arrived_at = SimTime::from_seconds(0.0);
                value.prefill_completed_at = SimTime::from_seconds(0.001);
                value.completed_at = SimTime::from_seconds(0.002);
                value.first_scheduled_at = SimTime::from_seconds(0.0);
                value.first_token_completed_at = SimTime::from_seconds(0.001);
                value.num_processed_tokens = 6;
                value.preemption_count = 1;
                value.tokens_at_preemption = {};
                value.replica_id = ReplicaId{0};
                value.dp_id = DataParallelId{0};
                value.prefill_replica_id = ReplicaId{};
                value.prefill_dp_id = DataParallelId{};
                value.decode_replica_id = ReplicaId{};
                value.decode_dp_id = DataParallelId{};
                value.transfer_id = frontier::TransferId{};
                value.kv_cache_transfer_start_time = SimTime{};
                value.kv_cache_transfer_end_time = SimTime{};
                value.decode_arrived_at = SimTime{};
                value.kv_cache_transfer_size_bytes = 0;
                return value;
            }(),
        };
        value.batches = {
            [&]() {
                BatchMetricsRecord value{};
                value.batch_id = BatchId{0};
                value.iteration_id = IterationId{0};
                value.scheduled_at = SimTime::from_seconds(0.0);
                value.completed_at = SimTime::from_seconds(0.001);
                value.request_ids = {RequestId{0}};
                value.scheduled_tokens = {4};
                value.total_scheduled_tokens = 4;
                value.num_prefill_tokens = 4;
                value.num_decode_tokens = 0;
                value.predicted_execution_ms = 1.0;
                return value;
            }(),
        };
        value.batch_stages = {};
        value.scheduler_trace = {
            [&]() {
                SchedulerTraceRecord value{};
                value.iteration_id = IterationId{0};
                value.simulation_time = SimTime::from_seconds(0.0);
                value.token_budget_before = 8;
                value.token_budget_after = 4;
                value.available_blocks_before = 8;
                value.available_blocks_after = 7;
                value.waiting_count_before = 1;
                value.waiting_count_after = 0;
                value.running_count_before = 0;
                value.running_count_after = 1;
                value.preempted_count = 0;
                value.decisions = {
                    [&]() {
                        SchedulerDecisionRecord value{};
                        value.decision_result = "ADMISSION";
                        value.request_id = RequestId{0};
                        value.num_tokens = 4;
                        value.token_budget_after = 4;
                        value.available_blocks_after = 7;
                        return value;
                    }(),
                };
                value.batch_request_ids = {RequestId{0}};
                value.request_num_tokens = {4};
                return value;
            }(),
        };
        value.event_trace = {
            Event{
                SimTime::from_seconds(0.0),
                EventSequence{1},
                [&]() {
                    RequestArrivalPayload value{};
                    value.request_id = RequestId{0};
                    value.cluster_type = frontier::ClusterType::kMonolithic;
                    return value;
                }(),
            },
        };
        value.analytical_diagnostics = {};
        value.kv_cache_transfers = {};
        return value;
    }();
}

void test_json_contract() {
    const Json json =
        Json::parse(serialize_simulation_output_json(make_output()));
    expect(json.at("schema_version") == 1, "version must serialize");
    expect(json.at("requests").at(0).at("first_scheduled_at_s") == 0.0,
           "canonical scheduling timestamp must serialize");
    expect(json.at("batches").at(0).at("request_ids") == Json::array({0}),
           "batch request order must be stable");
    expect(json.at("scheduler_trace")
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
        csv.compare(
            0,
            std::string{"request_id,arrived_at_s,first_scheduled_at_s,"}.size(),
            "request_id,arrived_at_s,first_scheduled_at_s,") == 0,
        "CSV must expose canonical scheduling fields");
    expect(csv.find(",6,1,0,0\n") != std::string::npos,
           "CSV must include progress, preemption, and target fields");
}

} // namespace

int main() {
    int failures = 0;
    failures += frontier::test::run("JSON contract", test_json_contract);
    failures += frontier::test::run("CSV contract", test_csv_contract);
    return failures == 0 ? 0 : 1;
}
