#include "frontier/config/config.h"
#include "frontier/request_generator/workload.h"
#include "frontier/simulator/simulator.h"
#include "tests/test_support.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <utility>
#include <vector>

#ifndef FRONTIER_EXAMPLE_DIR
#error "FRONTIER_EXAMPLE_DIR must be defined"
#endif
#ifndef FRONTIER_TEST_FIXTURE_DIR
#error "FRONTIER_TEST_FIXTURE_DIR must be defined"
#endif

namespace {

using frontier::RequestId;
using frontier::SessionId;
using frontier::SimTime;
using frontier::config::ClusterSchedulerType;
using frontier::config::ExecutionModelType;
using frontier::config::SimulationConfig;
using frontier::config::parse_simulation_config_json;
using frontier::request_generator::WorkloadRequest;
using frontier::simulator::run_simulation;
using frontier::test::expect;
using frontier::test::read_text_file;

const std::filesystem::path kExamples{FRONTIER_EXAMPLE_DIR};
const std::filesystem::path kFixtures{FRONTIER_TEST_FIXTURE_DIR};

SimulationConfig load(const std::filesystem::path &path) {
    return parse_simulation_config_json(read_text_file(path));
}

std::vector<WorkloadRequest> workload(std::uint64_t sessions) {
    std::vector<WorkloadRequest> result;
    for (std::uint64_t turn = 0; turn < 2; ++turn) {
        for (std::uint64_t session = 0; session < sessions; ++session) {
            WorkloadRequest value{};
            value.request_id = RequestId{result.size()};
            if (turn == 0) {
                value.session_start_at =
                    SimTime::from_seconds(session * 0.0005);
                value.num_prefill_tokens = 8;
            } else {
                value.think_time = SimTime::from_seconds(0.01);
                value.num_prefill_tokens = 4;
            }
            value.num_decode_tokens = 2;
            value.session_id = SessionId{2'000 + session};
            value.session_turn_index = turn;
            result.push_back(value);
        }
    }
    return result;
}

void prepare(SimulationConfig &config,
             const frontier::config::CpuKVCacheConfig &cpu) {
    config.prefix_cache.enabled = true;
    config.cpu_kv_cache = cpu;
    config.cluster_scheduler.type = ClusterSchedulerType::kStickyRoundRobin;
    config.pdd().clusters.prefill.scheduler.num_blocks = 4;
    config.pdd().clusters.prefill.scheduler.max_tokens_in_batch = 8;
    config.pdd().clusters.decode.scheduler.num_blocks = 64;
}

void require_matrix_result(const SimulationConfig &config,
                           std::uint64_t sessions,
                           std::uint64_t expected_targets,
                           const std::string &context) {
    const auto output = run_simulation(config, workload(sessions));
    expect(output.requests.size() == sessions * 2,
           context + ": requests did not drain");
    expect(output.aggregate.cpu_kv_cache.target_count == expected_targets &&
               output.cpu_kv_cache_targets.size() == expected_targets,
           context + ": CPU target count differs from PREFILL replica*DP");
    expect(output.aggregate.cpu_kv_cache.offload_operations > 0 &&
               output.aggregate.cpu_kv_cache.restore_operations > 0,
           context + ": topology did not exercise both transfer directions");
    std::map<std::int64_t, std::pair<std::int64_t, std::int64_t>> affinity;
    for (const auto &record : output.requests) {
        const auto target = std::make_pair(record.prefill_replica_id.value(),
                                           record.prefill_dp_id.value());
        const auto [position, inserted] =
            affinity.emplace(record.session_id.value(), target);
        expect(inserted || position->second == target,
               context + ": session affinity changed CPU cache target");
    }
    std::uint64_t target_queries = 0;
    std::uint64_t target_hits = 0;
    std::uint64_t capacity_blocks = 0;
    std::uint64_t resident_blocks = 0;
    std::uint64_t reserved_blocks = 0;
    std::uint64_t free_blocks = 0;
    std::uint64_t evicted_blocks = 0;
    std::uint64_t evicted_sessions = 0;
    for (const auto &target : output.cpu_kv_cache_targets) {
        target_queries += target.cpu_query_blocks;
        target_hits += target.cpu_hit_blocks;
        capacity_blocks += target.capacity_blocks;
        resident_blocks += target.resident_blocks;
        reserved_blocks += target.reserved_blocks;
        free_blocks += target.free_blocks;
        evicted_blocks += target.evicted_blocks;
        evicted_sessions += target.evicted_sessions;
        expect(target.resident_blocks + target.reserved_blocks <=
                       target.capacity_blocks &&
                   target.active_offload_reservations == 0 &&
                   target.active_restore_leases == 0,
               context + ": target did not quiesce independently");
    }
    const auto &aggregate = output.aggregate.cpu_kv_cache;
    expect(target_queries == aggregate.query_blocks &&
               target_hits == aggregate.hit_blocks &&
               capacity_blocks == aggregate.capacity_blocks &&
               resident_blocks == aggregate.resident_blocks &&
               reserved_blocks == aggregate.reserved_blocks &&
               free_blocks == aggregate.free_blocks &&
               evicted_blocks == aggregate.evicted_blocks &&
               evicted_sessions == aggregate.evicted_sessions,
           context + ": aggregate metrics must equal target sums");

    std::map<std::pair<std::int64_t, std::int64_t>, std::uint64_t>
        operations_by_target;
    for (const auto &transfer : output.cpu_kv_cache_transfers) {
        ++operations_by_target[{transfer.replica_id.value(),
                                transfer.dp_id.value()}];
    }
    expect(operations_by_target.size() == expected_targets,
           context + ": every CPU target must exercise its own transfer engine");
    if (expected_targets > 1) {
        bool cross_target_overlap = false;
        for (const auto &left : output.cpu_kv_cache_transfers) {
            for (const auto &right : output.cpu_kv_cache_transfers) {
                if (left.kind == right.kind &&
                    std::make_pair(left.replica_id.value(),
                                   left.dp_id.value()) !=
                        std::make_pair(right.replica_id.value(),
                                       right.dp_id.value()) &&
                    left.started_at < right.completed_at &&
                    right.started_at < left.completed_at) {
                    cross_target_overlap = true;
                }
            }
        }
        expect(cross_target_overlap,
               context + ": different targets must have independent overlapping queues");
    }

    std::uint64_t offload_operations = 0;
    std::uint64_t offload_blocks = 0;
    std::uint64_t offload_bytes = 0;
    std::uint64_t restore_operations = 0;
    std::uint64_t restore_blocks = 0;
    std::uint64_t restore_bytes = 0;
    double d2h_queue_ms = 0.0;
    double d2h_service_ms = 0.0;
    double h2d_queue_ms = 0.0;
    double h2d_service_ms = 0.0;
    double source_hold_ms = 0.0;
    for (const auto &transfer : output.cpu_kv_cache_transfers) {
        if (transfer.kind ==
            frontier::metrics::CpuKVCacheTransferKind::kOffload) {
            ++offload_operations;
            offload_blocks += transfer.blocks;
            offload_bytes += transfer.size_bytes;
            d2h_queue_ms += transfer.queue_time_ms;
            d2h_service_ms += transfer.service_time_ms;
            source_hold_ms += transfer.source_gpu_hold_ms;
        } else {
            ++restore_operations;
            restore_blocks += transfer.blocks;
            restore_bytes += transfer.size_bytes;
            h2d_queue_ms += transfer.queue_time_ms;
            h2d_service_ms += transfer.service_time_ms;
        }
    }
    expect(offload_operations == aggregate.offload_operations &&
               offload_blocks == aggregate.offload_blocks &&
               offload_bytes == aggregate.offload_bytes &&
               restore_operations == aggregate.restore_operations &&
               restore_blocks == aggregate.restore_blocks &&
               restore_bytes == aggregate.restore_bytes &&
               std::abs(d2h_queue_ms - aggregate.d2h_queue_time_ms) < 1e-9 &&
               std::abs(d2h_service_ms - aggregate.d2h_service_time_ms) <
                   1e-9 &&
               std::abs(h2d_queue_ms - aggregate.h2d_queue_time_ms) < 1e-9 &&
               std::abs(h2d_service_ms - aggregate.h2d_service_time_ms) <
                   1e-9 &&
               std::abs(source_hold_ms -
                        aggregate.source_gpu_hold_time_ms) < 1e-9,
           context + ": aggregate transfer totals must equal detailed records");
}

void test_dense_multi_target_tp_pp_dp() {
    auto config = load(kExamples / "configs" /
                       "06_cpu_kv_cache_pdd_online.json");
    const auto cpu = config.cpu_kv_cache;
    prepare(config, cpu);
    auto &prefill = config.pdd().clusters.prefill;
    auto &decode = config.pdd().clusters.decode;
    prefill.parallelism.num_replicas = 2;
    prefill.parallelism.tensor_parallel_size = 2;
    prefill.parallelism.pipeline_parallel_size = 2;
    prefill.parallelism.data_parallel_size = 2;
    prefill.execution_model.fixed.stage_latencies_ms = {0.01, 0.01};
    decode.parallelism.tensor_parallel_size = 2;
    decode.parallelism.pipeline_parallel_size = 2;
    decode.parallelism.data_parallel_size = 2;
    decode.execution_model.fixed.stage_latencies_ms = {0.01, 0.01};
    require_matrix_result(config, 12, 4, "dense TP2/PP2/DP2");
}

void test_dense_analytical_runtime() {
    auto config = load(kExamples / "configs" /
                       "06_cpu_kv_cache_pdd_online.json");
    const auto cpu = config.cpu_kv_cache;
    prepare(config, cpu);
    for (auto *runtime : {&config.pdd().clusters.prefill,
                          &config.pdd().clusters.decode}) {
        runtime->execution_model.type = ExecutionModelType::kAnalytical;
        runtime->execution_model.analytical.tensor_parallel_size = 1;
    }
    require_matrix_result(config, 3, 1, "dense analytical");
}

void test_moe_tp_pp_runtime() {
    auto cpu_source = load(kExamples / "configs" /
                           "06_cpu_kv_cache_pdd_online.json");
    auto config = load(kFixtures / "config" /
                       "fixed_moe_sequential_pdd.json");
    prepare(config, cpu_source.cpu_kv_cache);
    require_matrix_result(config, 3, 1, "MoE TP/PP");
}

} // namespace

int main() {
    int failures = 0;
    failures += frontier::test::run("CPU KV dense multi-target topology",
                                    test_dense_multi_target_tp_pp_dp);
    failures += frontier::test::run("CPU KV analytical execution topology",
                                    test_dense_analytical_runtime);
    failures += frontier::test::run("CPU KV MoE topology",
                                    test_moe_tp_pp_runtime);
    return failures == 0 ? 0 : 1;
}
