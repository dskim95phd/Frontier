#include "frontier/config/config.h"
#include "frontier/core/event.h"
#include "frontier/metrics/output_contract.h"
#include "frontier/request_generator/workload.h"
#include "frontier/simulator/simulator.h"
#include "tests/test_support.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#ifndef FRONTIER_EXAMPLE_DIR
#error "FRONTIER_EXAMPLE_DIR must be defined for K3 PP offload tests"
#endif

namespace {

using frontier::BatchId;
using frontier::ClusterType;
using frontier::EventType;
using frontier::RequestId;
using frontier::SimTime;
using frontier::config::ClusterRuntimeConfig;
using frontier::config::PrefillOnlyConfig;
using frontier::config::parse_simulation_config_json;
using frontier::config::SimulationConfig;
using frontier::metrics::CpuKVCacheTransferKind;
using frontier::metrics::CpuKVCacheTransferMetricsRecord;
using frontier::metrics::RequestMetricsRecord;
using frontier::metrics::SimulationOutput;
using frontier::request_generator::WorkloadRequest;
using frontier::simulator::run_simulation;
using frontier::test::expect;
using frontier::test::read_text_file;

const std::filesystem::path kExamples{FRONTIER_EXAMPLE_DIR};
constexpr std::uint64_t kK3SnapshotBytes = 232'316'928ULL;

void replace_all(std::string &text, const std::string &from,
                 const std::string &to) {
    std::size_t position = 0;
    std::size_t replacements = 0;
    while ((position = text.find(from, position)) != std::string::npos) {
        text.replace(position, from.size(), to);
        position += to.size();
        ++replacements;
    }
    expect(replacements > 0, "K3 PP fixture replacement is missing");
}

std::vector<WorkloadRequest> pdd_workload() {
    // Two close initial sessions exercise concurrent/staggered PP calendars;
    // their later turns require CPU-tier prefix lookup and restore.
    return frontier::request_generator::parse_workload_csv(
        "session_start_at,think_time,num_prefill_tokens,num_decode_tokens,"
        "session_id,session_turn_index\n"
        "0,0,8,2,700,0\n"
        "0.0001,0,12,2,701,0\n"
        "0.2,0,12,2,702,0\n"
        ",0.5,4,2,700,1\n"
        ",0.05,4,2,701,1\n");
}

SimulationConfig make_config(std::uint64_t pp, const char *event_mode) {
    std::string config_text = read_text_file(kExamples / "configs" /
                                             "06_cpu_kv_cache_pdd_online.json");
    replace_all(config_text, "meta-llama/Llama-2-7b-hf", "moonshotai/Kimi-K3");
    replace_all(config_text, "\"capacity_bytes\": 67108864",
                "\"capacity_bytes\": 2147483648");
    // Keep a generous logical KV frontier while making a second resident K3
    // snapshot impossible.  The physical capacity is set below from the
    // stage-local snapshot profile rather than from a full-model estimate.
    replace_all(config_text, "\"num_blocks\": 4", "\"num_blocks\": 16");
    replace_all(config_text, "\"num_blocks\": 32", "\"num_blocks\": 16");

    auto config = parse_simulation_config_json(config_text);
    config.run_id = std::string{"k3-pp-exclusive-"} + event_mode + "-pp" +
                    std::to_string(pp);

    auto configure_cluster = [pp, event_mode](ClusterRuntimeConfig &cluster) {
        cluster.parallelism.pipeline_parallel_size = pp;
        cluster.parallelism.pipeline_exclusive = pp > 1;
        cluster.scheduler.pipeline_event_mode = event_mode;
        const double stage_latency =
            cluster.execution_model.fixed.stage_latencies_ms.front();
        cluster.execution_model.fixed.stage_latencies_ms.assign(
            static_cast<std::size_t>(pp), stage_latency);

        // Resolve the exact PP-local profiles first, then use twice the
        // largest snapshot shard as per-GPU HBM.  This leaves one snapshot's
        // normalized charge available but forces eviction when a second
        // session arrives.
        const auto provisional =
            frontier::config::build_pipeline_stage_memory_profiles(cluster);
        std::uint64_t max_snapshot = 0;
        for (const auto &profile : provisional) {
            for (const std::uint64_t bytes :
                 profile.kda_snapshot_bytes_by_rank) {
                max_snapshot = std::max(max_snapshot, bytes);
            }
        }
        expect(max_snapshot > 0,
               "K3 PP profile must expose a local KDA snapshot shard");
        cluster.gpu_memory.capacity_bytes_per_gpu = 2 * max_snapshot;
        cluster.gpu_memory.auto_calculate_num_blocks = false;
        frontier::config::resolve_gpu_memory_config(
            cluster, cluster.scheduler.num_blocks);
    };
    configure_cluster(config.pdd().clusters.prefill);
    configure_cluster(config.pdd().clusters.decode);
    return config;
}

const RequestMetricsRecord &request(const SimulationOutput &output,
                                    RequestId id) {
    const auto position =
        std::find_if(output.requests.begin(), output.requests.end(),
                     [id](const RequestMetricsRecord &value) {
                         return value.request_id == id;
                     });
    if (position == output.requests.end()) {
        throw std::runtime_error("missing request metrics");
    }
    return *position;
}

bool batch_contains_request(const SimulationOutput &output, BatchId batch_id,
                            RequestId request_id) {
    const auto batch = std::find_if(
        output.batches.begin(), output.batches.end(),
        [batch_id](const auto &record) { return record.batch_id == batch_id; });
    return batch != output.batches.end() &&
           std::find(batch->request_ids.begin(), batch->request_ids.end(),
                     request_id) != batch->request_ids.end();
}

SimTime stage_time(const SimulationOutput &output, RequestId request_id,
                   std::uint64_t stage, bool completion) {
    SimTime selected{};
    for (const auto &record : output.batch_stages) {
        if (record.cluster_type != ClusterType::kPrefill ||
            record.stage_id != frontier::StageId{stage} ||
            !batch_contains_request(output, record.batch_id, request_id)) {
            continue;
        }
        const SimTime candidate =
            completion ? record.completed_at : record.arrived_at;
        if (!selected.valid() ||
            (completion ? selected < candidate : candidate < selected)) {
            selected = candidate;
        }
    }
    return selected;
}

void check_stage_local_profiles(const SimulationOutput &output,
                                std::uint64_t pp) {
    expect(output.pipeline_memory_diagnostics.size() == 2,
           "PDD output must expose PREFILL and DECODE memory diagnostics");
    for (const auto &diagnostic : output.pipeline_memory_diagnostics) {
        expect(diagnostic.stages.size() == pp,
               "diagnostics must contain every physical PP stage");
        std::uint64_t kda_layers = 0;
        std::uint64_t mla_layers = 0;
        bool local_profile_varies = false;
        for (std::size_t index = 0; index < diagnostic.stages.size(); ++index) {
            const auto &stage = diagnostic.stages[index];
            expect(stage.layer_count > 0 &&
                       stage.kv_bytes_per_block_by_rank.size() == 1 &&
                       stage.kda_snapshot_bytes_by_rank.size() == 1 &&
                       stage.free_bytes > 0,
                   "every PP stage must carry rank-local memory vectors");
            kda_layers += stage.kda_layer_count;
            mla_layers += stage.mla_layer_count;
            if (index != 0 &&
                stage.kda_snapshot_bytes_by_rank !=
                    diagnostic.stages.front().kda_snapshot_bytes_by_rank) {
                local_profile_varies = true;
            }
        }
        expect(kda_layers == 69 && mla_layers == 24,
               "K3 PP profiles must partition all KDA and MLA layers");
        expect(local_profile_varies,
               "K3 PP profiles must remain stage-local rather than PP-wide");
    }
}

void check_offload_boundaries(const SimulationOutput &output,
                              std::uint64_t pp) {
    const auto &cpu = output.aggregate.cpu_kv_cache;
    expect(cpu.kda_snapshot_bytes_per_session == kK3SnapshotBytes,
           "CPU K3 snapshot must be one full-model payload");
    expect(cpu.offload_operations > 0 && cpu.restore_operations > 0,
           "workload must exercise both CPU offload and restore");
    expect(cpu.kda_snapshot_offload_bytes % kK3SnapshotBytes == 0 &&
               cpu.kda_snapshot_restore_bytes % kK3SnapshotBytes == 0,
           "CPU snapshot traffic must use whole-model payload units");
    for (const CpuKVCacheTransferMetricsRecord &transfer :
         output.cpu_kv_cache_transfers) {
        expect(transfer.completed_at >= transfer.started_at &&
                   transfer.started_at >= transfer.submitted_at,
               "CPU transfer timing must be monotonic");
        if (transfer.kda_snapshot_bytes != 0) {
            expect(
                transfer.kda_snapshot_bytes == kK3SnapshotBytes,
                "CPU transfer must carry one atomic snapshot, not PP copies");
        }
        if (transfer.kind == CpuKVCacheTransferKind::kOffload) {
            const SimTime final_stage = stage_time(output, transfer.request_id,
                                                   pp - 1, /*completion=*/true);
            expect(final_stage.valid() && final_stage <= transfer.submitted_at,
                   "offload must submit after final PP stage completion");
        } else if (transfer.request_id == RequestId{3} ||
                   transfer.request_id == RequestId{4}) {
            const SimTime pp0_arrival = stage_time(output, transfer.request_id,
                                                   0, /*completion=*/false);
            expect(pp0_arrival.valid() && transfer.completed_at <= pp0_arrival,
                   "PP0 admission must wait for the complete CPU restore");
        }
    }
}

void check_collapsed_calendar(const SimulationOutput &output, bool collapsed) {
    const bool has_pipeline_end = std::any_of(
        output.event_trace.begin(), output.event_trace.end(),
        [](const auto &e) { return e.type() == EventType::kBatchPipelineEnd; });
    const bool has_sync =
        std::any_of(output.event_trace.begin(), output.event_trace.end(),
                    [](const auto &e) {
                        return e.type() == EventType::kPrefillSync ||
                               e.type() == EventType::kDecodeSync;
                    });
    expect(has_pipeline_end == collapsed && !has_sync,
           "pipeline-exclusive PDD must use one collapsed calendar without "
           "DP/EP synchronization events");
}

void compare_request_and_offload_contract(const SimulationOutput &exact,
                                          const SimulationOutput &collapsed) {
    expect(exact.requests.size() == collapsed.requests.size() &&
               exact.cpu_kv_cache_transfers.size() ==
                   collapsed.cpu_kv_cache_transfers.size(),
           "exact and collapsed runs must complete the same request/transfer "
           "count");
    for (const auto &record : exact.requests) {
        const auto &other = request(collapsed, record.request_id);
        expect(
            record.cpu_prefix_query_blocks == other.cpu_prefix_query_blocks &&
                record.cpu_prefix_hit_blocks == other.cpu_prefix_hit_blocks &&
                record.cpu_restore_transferred_blocks ==
                    other.cpu_restore_transferred_blocks &&
                record.cpu_restore_consumed_blocks ==
                    other.cpu_restore_consumed_blocks &&
                record.cpu_restore_bytes == other.cpu_restore_bytes &&
                record.cpu_offload_bytes == other.cpu_offload_bytes &&
                record.cached_prefill_tokens == other.cached_prefill_tokens,
            "exact and collapsed CPU/prefix request metrics must match");
    }
    expect(exact.aggregate.cpu_kv_cache.offload_operations ==
                   collapsed.aggregate.cpu_kv_cache.offload_operations &&
               exact.aggregate.cpu_kv_cache.restore_operations ==
                   collapsed.aggregate.cpu_kv_cache.restore_operations &&
               exact.aggregate.cpu_kv_cache.offload_bytes ==
                   collapsed.aggregate.cpu_kv_cache.offload_bytes &&
               exact.aggregate.cpu_kv_cache.restore_bytes ==
                   collapsed.aggregate.cpu_kv_cache.restore_bytes,
           "exact and collapsed CPU transfer aggregates must match");
}

void check_prefill_only_contract(const SimulationOutput &output,
                                 std::size_t request_count,
                                 double decode_tokens_per_second) {
    expect(output.run.prefill_only &&
               output.run.synthetic_decode_tokens_per_second ==
                   decode_tokens_per_second &&
               output.requests.size() == request_count &&
               output.prefill_completions.size() == request_count,
           "K3 PREFILL-only run must complete every PREFILL and synthetic "
           "output interval");
    const bool has_decode_batch = std::any_of(
        output.batches.begin(), output.batches.end(), [](const auto &batch) {
            return batch.cluster_type == ClusterType::kDecode;
        });
    const std::size_t synthetic_events = static_cast<std::size_t>(std::count_if(
        output.event_trace.begin(), output.event_trace.end(),
        [](const auto &event) {
            return event.type() == EventType::kSyntheticDecodeEnd;
        }));
    expect(!has_decode_batch && synthetic_events == request_count,
           "K3 PREFILL-only run must bypass DECODE scheduling exactly once "
           "per request");
    for (const auto &record : output.requests) {
        expect(record.prefill_only && record.prefill_replica_id.valid() &&
                   !record.decode_replica_id.valid() &&
                   record.first_token_completed_at.seconds() ==
                       record.decode_arrived_at.seconds() +
                           1.0 / decode_tokens_per_second &&
                   record.completed_at.seconds() ==
                       record.decode_arrived_at.seconds() +
                           static_cast<double>(record.num_decode_tokens) /
                               decode_tokens_per_second,
               "K3 PREFILL-only request must retain PREFILL ownership and "
               "use the configured synthetic token rate");
    }
}

void test_k3_pipeline_exclusive_offload_matrix() {
    for (const std::uint64_t pp : {2ULL, 4ULL, 8ULL}) {
        const auto workload = pdd_workload();
        auto exact_config = make_config(pp, "exact");
        auto collapsed_config = make_config(pp, "collapsed");
        const auto exact = run_simulation(exact_config, workload);
        const auto collapsed = run_simulation(collapsed_config, workload);

        expect(exact.requests.size() == workload.size() &&
                   collapsed.requests.size() == workload.size(),
               "all K3 PDD multiturn requests must complete");
        check_stage_local_profiles(exact, pp);
        check_stage_local_profiles(collapsed, pp);
        check_offload_boundaries(exact, pp);
        check_offload_boundaries(collapsed, pp);
        check_collapsed_calendar(exact, false);
        check_collapsed_calendar(collapsed, true);
        compare_request_and_offload_contract(exact, collapsed);
    }
}

void test_k3_prefill_only_pipeline_exclusive_offload_matrix() {
    for (const std::uint64_t pp : {2ULL, 4ULL, 8ULL}) {
        for (const double decode_tokens_per_second : {12.5, 50.0, 200.0}) {
            const auto workload = pdd_workload();
            auto exact_config = make_config(pp, "exact");
            auto collapsed_config = make_config(pp, "collapsed");
            exact_config.prefill_only =
                PrefillOnlyConfig{decode_tokens_per_second};
            collapsed_config.prefill_only =
                PrefillOnlyConfig{decode_tokens_per_second};

            const auto exact = run_simulation(exact_config, workload);
            const auto collapsed = run_simulation(collapsed_config, workload);

            check_prefill_only_contract(exact, workload.size(),
                                        decode_tokens_per_second);
            check_prefill_only_contract(collapsed, workload.size(),
                                        decode_tokens_per_second);
            check_stage_local_profiles(exact, pp);
            check_stage_local_profiles(collapsed, pp);
            check_offload_boundaries(exact, pp);
            check_offload_boundaries(collapsed, pp);
            check_collapsed_calendar(exact, false);
            check_collapsed_calendar(collapsed, true);
            compare_request_and_offload_contract(exact, collapsed);
        }
    }
}

} // namespace

int main() {
    int failures = 0;
    failures += frontier::test::run(
        "K3 pipeline-exclusive PDD CPU offload matrix",
        test_k3_pipeline_exclusive_offload_matrix);
    failures += frontier::test::run(
        "K3 PREFILL-only pipeline-exclusive PDD CPU offload matrix",
        test_k3_prefill_only_pipeline_exclusive_offload_matrix);
    return failures == 0 ? 0 : 1;
}
