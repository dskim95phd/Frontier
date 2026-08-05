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
using frontier::metrics::GpuKVCacheOccupancyRecord;
using frontier::metrics::RequestMetricsRecord;
using frontier::metrics::RunMetadata;
using frontier::metrics::SchedulerDecisionRecord;
using frontier::metrics::SchedulerTraceRecord;
using frontier::metrics::serialize_request_metrics_csv;
using frontier::metrics::serialize_gpu_kv_occupancy_csv;
using frontier::metrics::serialize_simulation_output_json;
using frontier::metrics::serialize_simulation_summary_json;
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
                value.num_prefill_tokens = 4;
                value.num_decode_tokens = 2;
                value.arrived_at = SimTime::from_seconds(0.0);
                value.prefill_completed_at = SimTime::from_seconds(0.001);
                value.completed_at = SimTime::from_seconds(0.002);
                value.first_scheduled_at = SimTime::from_seconds(0.0);
                value.first_token_completed_at = SimTime::from_seconds(0.0015);
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
            Event{
                SimTime::from_seconds(0.5),
                EventSequence{2},
                [&]() {
                    frontier::CpuKVCacheOffloadStartPayload value{};
                    value.transfer_id = frontier::CpuKvTransferId{3};
                    value.request_id = RequestId{0};
                    value.replica_id = ReplicaId{0};
                    value.dp_id = DataParallelId{0};
                    value.cpu_generation = frontier::CpuOffloadGeneration{7};
                    value.cluster_type = frontier::ClusterType::kPrefill;
                    return value;
                }(),
            },
        };
        value.analytical_diagnostics = {};
        value.kv_cache_transfers = {};
        value.gpu_kv_occupancy = {
            [&]() {
                GpuKVCacheOccupancyRecord value{};
                value.time = SimTime::from_seconds(0.0);
                value.cluster_type = frontier::ClusterType::kPrefill;
                value.replica_id = ReplicaId{0};
                value.dp_id = DataParallelId{0};
                value.active_blocks = 2;
                value.capacity_blocks = 8;
                value.active_bytes_per_gpu = 1'249'280;
                value.active_fraction_of_kv_budget = 0.25;
                const double total_hbm_fraction =
                    static_cast<double>(value.active_bytes_per_gpu) /
                    288'000'000'000.0;
                value.hbm_fraction = total_hbm_fraction;
                value.active_fraction_of_total_hbm = total_hbm_fraction;
                return value;
            }(),
            [&]() {
                GpuKVCacheOccupancyRecord value{};
                value.time = SimTime::from_seconds(1.0);
                value.cluster_type = frontier::ClusterType::kPrefill;
                value.replica_id = ReplicaId{0};
                value.dp_id = DataParallelId{0};
                value.capacity_blocks = 8;
                return value;
            }(),
        };
        value.aggregate.prefix_cache.block_size = 4;
        value.aggregate.prefix_cache.successful_admissions = 3;
        value.aggregate.prefix_cache.query_blocks = 5;
        value.aggregate.prefix_cache.hit_blocks = 3;
        value.aggregate.prefix_cache.evicted_blocks = 7;
        value.aggregate.prefix_cache.evicted_sessions = 2;
        auto &batch_aggregate =
            value.aggregate
                .batches_by_cluster[frontier::ClusterType::kMonolithic];
        batch_aggregate.batch_count = 1;
        batch_aggregate.request_slots = 1;
        batch_aggregate.prefill_scheduled_tokens = 4;
        batch_aggregate.predicted_execution_ms = 1.0;
        batch_aggregate.execution_time.dense_compute_ms = 0.25;
        batch_aggregate.execution_time.tp_communication_ms = 0.125;
        batch_aggregate.execution_time.moe_grouped_gemm_ms = 0.5;
        batch_aggregate.execution_time.synchronization_wait_ms = 0.125;
        batch_aggregate.batch_size_histogram[1] = 1;
        auto &time_bucket =
            value.aggregate.batch_time_buckets_by_cluster
                [frontier::ClusterType::kMonolithic][0];
        time_bucket.batch_count = 1;
        time_bucket.request_slots = 1;
        time_bucket.predicted_execution_ms = 1.0;
        time_bucket.execution_time = batch_aggregate.execution_time;
        value.prefix_cache_targets = {
            [&]() {
                frontier::metrics::PrefixCacheTargetMetricsRecord value{};
                value.capacity_blocks = 16;
                value.available_blocks = 12;
                value.active_blocks = 2;
                value.resident_blocks = 6;
                value.evictable_blocks = 4;
                value.evictable_sessions = 2;
                value.sessions_with_nonzero_frontier = 3;
                return value;
            }(),
        };
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
        expect(json.at("requests").at(0).at("num_prefill_tokens") == 4,
           "prefix-cache request shape must serialize");
    expect(json.at("requests").at(0).at("scheduled_prefill_tokens") == 0 &&
               json.at("requests").at(0)
                       .at("preemption_recomputed_prefill_tokens") == 0,
           "compact scheduled-PREFILL request fields must serialize");
    expect(json.at("requests").at(0).at("prefill_latency_ms") == 1.0 &&
               json.at("requests").at(0).at("ttft_ms") == 1.5,
           "TTFT must end at first-token completion while prefill latency is "
           "reported separately");
    expect(json.contains("prefix_cache"),
           "aggregate prefix-cache metrics must serialize");
    expect(json.at("prefix_cache").at("storage_model") ==
                   "analytical_session" &&
               json.at("prefix_cache").at("evicted_blocks") == 7 &&
               json.at("prefix_cache").at("evicted_sessions") == 2,
           "analytical cache identity and eviction metrics must serialize");
    expect(json.at("prefix_cache_targets").at(0).at("resident_blocks") == 6 &&
               json.at("prefix_cache_targets").at(0).at("evictable_sessions") ==
                   2,
           "analytical target counters must serialize");
    expect(!json.at("prefix_cache").contains("physical_evictions"),
           "removed physical allocator metrics must not leak into output");
    expect(json.at("event_trace").at(1).at("type") ==
                   "cpu_kv_cache_offload_start" &&
               json.at("event_trace").at(1).at("cpu_generation") == 7,
           "CPU offload trace must serialize its CPU generation");
    expect(json.at("gpu_kv_occupancy").size() == 2 &&
               json.at("gpu_kv_occupancy").at(0).at("active_blocks") == 2 &&
               json.at("gpu_kv_occupancy").at(0).at("hbm_fraction") ==
                   static_cast<double>(1'249'280) / 288'000'000'000.0 &&
               json.at("gpu_kv_occupancy").at(1)
                       .at("active_fraction_of_total_hbm")
                       .is_null(),
           "GPU KV occupancy event samples must serialize with optional HBM basis");
}

void test_csv_contract() {
    const std::string csv =
        serialize_request_metrics_csv(make_output().requests);
    expect(csv.compare(
               0,
               std::string{"request_id,session_id,num_prefill_tokens,"}.size(),
               "request_id,session_id,num_prefill_tokens,") == 0,
           "CSV must expose canonical scheduling fields");
    expect(csv.find("prefill_latency_ms,ttft_ms") != std::string::npos,
           "CSV must distinguish prefill latency from TTFT");
    expect(csv.find(",6,1,0,0,0,0\n") != std::string::npos,
           "CSV must include progress, preemption, target, and PREFILL work fields");
    const std::string occupancy_csv =
        serialize_gpu_kv_occupancy_csv(make_output().gpu_kv_occupancy);
    expect(occupancy_csv.rfind(
               "time_s,cluster_type,replica_id,dp_id,active_blocks,"
               "capacity_blocks,active_bytes_per_gpu,hbm_fraction,"
               "active_fraction_of_kv_budget,active_fraction_of_total_hbm\n",
               0) == 0,
           "GPU KV occupancy CSV must expose per-GPU and fraction fields");
    expect(occupancy_csv.find("0,PREFILL,0,0,2,8,1249280,") !=
               std::string::npos &&
               occupancy_csv.find("1,PREFILL,0,0,0,8,0,,0,\n") !=
                   std::string::npos,
           "GPU KV occupancy CSV must retain terminal zero snapshots");
}

void test_summary_contract() {
    const Json json =
        Json::parse(serialize_simulation_summary_json(make_output(), 0.25));
    expect(json.at("latency_ms").at("prefill").at("mean") == 1.0 &&
               json.at("latency_ms").at("ttft").at("mean") == 1.5,
           "summary must use first-token TTFT semantics");
    expect(json.at("latency_ms").at("tpot").at("mean") == 0.5,
           "summary must derive TPOT from the post-first-token decode tail");
    expect(json.at("prefill_work").at("scheduled_prefill_tokens") == 0 &&
               json.at("counts").at("prefill_scheduled_tokens") == 4,
           "summary must expose compact scheduled-PREFILL totals");
    expect(json.at("batch_time_bucket_seconds") == 60 &&
               json.at("batch_summary_by_cluster_time_bucket").is_object(),
           "summary must expose compact batch time buckets");
    const Json &cluster =
        json.at("batch_summary_by_cluster").at("MONOLITHIC");
    const Json &components = cluster.at("execution_time_components_ms");
    expect(components.at("dense_compute_ms") == 0.25 &&
               components.at("tp_communication_ms") == 0.125 &&
               components.at("moe_grouped_gemm_ms") == 0.5 &&
               components.at("synchronization_wait_ms") == 0.125 &&
               components.at("total_ms") == 1.0,
           "summary must expose compact execution-time component totals");
    const Json &bucket_components =
        json.at("batch_summary_by_cluster_time_bucket")
            .at("MONOLITHIC")
            .at(0)
            .at("execution_time_components_ms");
    expect(bucket_components == components,
           "time buckets must preserve execution-time component totals");
}

} // namespace

int main() {
    int failures = 0;
    failures += frontier::test::run("JSON contract", test_json_contract);
    failures += frontier::test::run("CSV contract", test_csv_contract);
    failures += frontier::test::run("summary contract", test_summary_contract);
    return failures == 0 ? 0 : 1;
}
