#include "frontier/config/config.h"
#include "frontier/metrics/output_contract.h"
#include "frontier/request_generator/workload.h"
#include "frontier/simulator/simulator.h"
#include "tests/test_support.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#ifndef FRONTIER_EXAMPLE_DIR
#error "FRONTIER_EXAMPLE_DIR must be defined for CPU KV-cache matrix tests"
#endif

namespace {

using frontier::ClusterType;
using frontier::RequestId;
using frontier::SessionId;
using frontier::SimTime;
using frontier::config::CpuKVCacheCapacityPressurePolicy;
using frontier::config::SimulationConfig;
using frontier::config::SimulationMode;
using frontier::config::parse_simulation_config_json;
using frontier::metrics::SimulationOutput;
using frontier::request_generator::WorkloadRequest;
using frontier::simulator::run_simulation;
using frontier::test::expect;
using frontier::test::read_text_file;

const std::filesystem::path kExamples{FRONTIER_EXAMPLE_DIR};

struct MatrixScenario {
    const char *name;
    SimulationMode mode;
    std::uint64_t capacity_blocks;
    CpuKVCacheCapacityPressurePolicy pressure_policy;
    bool static_slice;
    bool slow_transfer;
    bool multi_target;
};

struct TierSummary {
    std::uint64_t prefill_scheduled_tokens = 0;
    std::uint64_t cached_tokens = 0;
    double successor_ttft_sum_ms = 0.0;
    std::uint64_t successor_count = 0;
    double makespan_ms = 0.0;
    std::uint64_t offloads = 0;
    std::uint64_t restores = 0;
    std::uint64_t evicted_blocks = 0;
    std::uint64_t evicted_sessions = 0;
    std::uint64_t skipped_offloads = 0;
    std::uint64_t truncated_offloads = 0;
};

SimulationConfig load_config(SimulationMode mode) {
    const char *filename = mode == SimulationMode::kOnline
                               ? "06_cpu_kv_cache_pdd_online.json"
                               : "07_cpu_kv_cache_pdd_offline.json";
    SimulationConfig config = parse_simulation_config_json(
        read_text_file(kExamples / "configs" / filename));
    config.run_id = std::string{"cpu-kv-cache-stress-matrix-"} +
                    (mode == SimulationMode::kOnline ? "online" : "offline");
    // Keep the GPU tier finite enough to force CPU snapshots, while retaining
    // enough pages for every request to make progress in two-turn workloads.
    config.pdd().clusters.prefill.scheduler.num_blocks = 5;
    config.pdd().clusters.prefill.scheduler.max_tokens_in_batch = 8;
    config.pdd().clusters.prefill.scheduler.batch_size_cap = 4;
    config.pdd().clusters.decode.scheduler.num_blocks = 64;
    config.cluster_scheduler.type =
        frontier::config::ClusterSchedulerType::kStickyRoundRobin;
    return config;
}

std::vector<WorkloadRequest> make_workload(std::uint64_t sessions,
                                           std::uint64_t rounds = 2,
                                           bool burst = false,
                                           bool oversized_first = false) {
    std::vector<WorkloadRequest> workload;
    workload.reserve(static_cast<std::size_t>(sessions * rounds));
    for (std::uint64_t round = 0; round < rounds; ++round) {
        for (std::uint64_t session = 0; session < sessions; ++session) {
            WorkloadRequest value{};
            value.request_id = RequestId{workload.size()};
            value.session_id = SessionId{10'000 + session};
            value.session_turn_index = round;
            value.num_decode_tokens = 2;
            if (round == 0) {
                value.session_start_at =
                    burst ? SimTime::from_seconds(static_cast<double>(session) *
                                                  0.0001)
                          : SimTime::from_seconds((session / 2) * 0.2 +
                                                  (session % 2) * 0.01);
                value.num_prefill_tokens =
                    oversized_first && session == 0
                        ? 40
                        : 8 + (session % 2) * 4;
            } else {
                // Session-relative arrivals exercise restore after other
                // sessions have evicted the initial prefix.
                value.think_time =
                    SimTime::from_seconds(burst ? 0.001 : 0.05);
                // Keep the accumulated per-session GPU frontier within the
                // five-block stress tier (the decode tokens consume pages as
                // well as the prefill prompt).
                value.num_prefill_tokens = 4;
            }
            workload.push_back(value);
        }
    }
    return workload;
}

std::uint64_t bytes_per_block(const SimulationConfig &config) {
    return frontier::config::resolve_cpu_kv_cache_target(config).bytes_per_block;
}

SimulationConfig configure(const MatrixScenario &scenario) {
    SimulationConfig config = load_config(scenario.mode);
    auto &prefill = config.pdd().clusters.prefill;
    auto &decode = config.pdd().clusters.decode;
    const std::uint64_t block_bytes = bytes_per_block(config);

    if (scenario.multi_target) {
        // Four independent (replica, DP) CPU targets.  Keep TP/PP at one so
        // direct and static-slice capacities have the same logical block unit.
        prefill.parallelism.num_replicas = 2;
        prefill.parallelism.data_parallel_size = 2;
        decode.parallelism.num_replicas = 2;
        decode.parallelism.data_parallel_size = 2;
        prefill.execution_model.fixed.stage_latencies_ms = {0.05};
        decode.execution_model.fixed.stage_latencies_ms = {0.05};
    }
    if (scenario.static_slice) {
        // Exercise the physical-slice multiplication path.  TP2/PP2 gives
        // four physical slices, so capacity_bytes_per_gpu is multiplied by 4.
        prefill.parallelism.tensor_parallel_size = 2;
        prefill.parallelism.pipeline_parallel_size = 2;
        decode.parallelism.tensor_parallel_size = 2;
        decode.parallelism.pipeline_parallel_size = 2;
        prefill.execution_model.fixed.stage_latencies_ms = {0.05, 0.05};
        decode.execution_model.fixed.stage_latencies_ms = {0.05, 0.05};
    }

    config.cpu_kv_cache.enabled = true;
    config.cpu_kv_cache.capacity_pressure_policy = scenario.pressure_policy;
    config.cpu_kv_cache.static_slice_per_gpu = scenario.static_slice;
    if (scenario.pressure_policy ==
            CpuKVCacheCapacityPressurePolicy::kSkipOffload &&
        scenario.capacity_blocks == 8) {
        // The oversized pressure request below needs a GPU working set large
        // enough for chunked prefill; its CPU snapshot still exceeds the
        // eight-block tier and is therefore skipped.
        config.pdd().clusters.prefill.scheduler.num_blocks = 16;
    }
    if (scenario.static_slice) {
        config.cpu_kv_cache.capacity_bytes_per_gpu =
            block_bytes * scenario.capacity_blocks;
    } else {
        config.cpu_kv_cache.capacity_bytes =
            block_bytes * scenario.capacity_blocks;
    }
    if (scenario.slow_transfer) {
        // A deliberately slow, finite transfer tier.  Assertions below use
        // service/queue metrics as the causal observation; no latency
        // improvement is assumed.
        config.cpu_kv_cache.write_bandwidth_gbps = 8.0;
        config.cpu_kv_cache.read_bandwidth_gbps = 8.0;
        config.cpu_kv_cache.write_latency_ms = 0.25;
        config.cpu_kv_cache.read_latency_ms = 0.25;
    } else {
        config.cpu_kv_cache.write_bandwidth_gbps = 64.0;
        config.cpu_kv_cache.read_bandwidth_gbps = 64.0;
        config.cpu_kv_cache.write_latency_ms = 0.01;
        config.cpu_kv_cache.read_latency_ms = 0.01;
    }
    config.run_id = std::string{"cpu-kv-cache-matrix-"} + scenario.name;
    return config;
}

void require_quiescent(const SimulationOutput &output,
                       const SimulationConfig &config,
                       std::uint64_t expected_requests,
                       std::uint64_t expected_targets,
                       const std::string &context) {
    expect(output.requests.size() == expected_requests,
           context + ": every request must complete");
    expect(output.aggregate.cpu_kv_cache.target_count == expected_targets &&
               output.cpu_kv_cache_targets.size() == expected_targets,
           context + ": CPU target count mismatch");

    std::uint64_t capacity_blocks = 0;
    std::uint64_t resident_blocks = 0;
    std::uint64_t reserved_blocks = 0;
    std::uint64_t free_blocks = 0;
    std::uint64_t evicted_blocks = 0;
    std::uint64_t evicted_sessions = 0;
    std::uint64_t skipped_offloads = 0;
    std::uint64_t truncated_offloads = 0;
    for (const auto &target : output.cpu_kv_cache_targets) {
        expect(target.resident_blocks + target.reserved_blocks <=
                   target.capacity_blocks,
               context + ": CPU target exceeds capacity");
        expect(target.free_blocks + target.resident_blocks +
                       target.reserved_blocks ==
                   target.capacity_blocks,
               context + ": CPU free/resident/reserved accounting diverged");
        expect(target.pending_restore_operations == 0 &&
                   target.staged_restore_payloads == 0 &&
                   target.active_restore_leases == 0 &&
                   target.active_offload_reservations == 0,
               context + ": CPU target retained transient state");
        capacity_blocks += target.capacity_blocks;
        resident_blocks += target.resident_blocks;
        reserved_blocks += target.reserved_blocks;
        free_blocks += target.free_blocks;
        evicted_blocks += target.evicted_blocks;
        evicted_sessions += target.evicted_sessions;
        skipped_offloads += target.skipped_offloads;
        truncated_offloads += target.truncated_offloads;
    }

    const auto &cpu = output.aggregate.cpu_kv_cache;
    expect(cpu.capacity_blocks == capacity_blocks &&
               cpu.resident_blocks == resident_blocks &&
               cpu.reserved_blocks == reserved_blocks &&
               cpu.free_blocks == free_blocks &&
               cpu.evicted_blocks == evicted_blocks &&
               cpu.evicted_sessions == evicted_sessions &&
               cpu.skipped_offloads == skipped_offloads &&
               cpu.truncated_offloads == truncated_offloads,
           context + ": aggregate CPU diagnostics differ from target sums");
    const auto resolved = frontier::config::resolve_cpu_kv_cache_target(config);
    expect(cpu.bytes_per_block == resolved.bytes_per_block,
           context + ": CPU block size differs from resolved configuration");

    for (const auto &request : output.requests) {
        expect(request.completed_at.valid() &&
                   request.first_token_completed_at.valid() &&
                   request.prefill_completed_at.valid() &&
                   request.completed_at >= request.first_token_completed_at,
               context + ": request timing is incomplete");
    }
    for (const auto &transfer : output.cpu_kv_cache_transfers) {
        expect(transfer.blocks > 0 && transfer.size_bytes > 0 &&
                   transfer.submitted_at.valid() && transfer.started_at.valid() &&
                   transfer.completed_at.valid() &&
                   transfer.submitted_at <= transfer.started_at &&
                   transfer.started_at <= transfer.completed_at,
               context + ": CPU transfer timing is invalid");
    }
}

void require_affinity(const SimulationOutput &output,
                      const std::string &context) {
    std::map<std::int64_t, std::pair<std::int64_t, std::int64_t>> affinity;
    for (const auto &record : output.requests) {
        const auto target = std::make_pair(record.prefill_replica_id.value(),
                                            record.prefill_dp_id.value());
        const auto [position, inserted] =
            affinity.emplace(record.session_id.value(), target);
        expect(inserted || position->second == target,
               context + ": session affinity changed across turns");
    }
}

TierSummary summarize(const SimulationOutput &output,
                      std::uint64_t sessions) {
    TierSummary summary{};
    for (const auto &batch : output.batches) {
        if (batch.cluster_type != ClusterType::kPrefill) {
            continue;
        }
        for (const std::uint64_t tokens : batch.scheduled_tokens) {
            summary.prefill_scheduled_tokens += tokens;
        }
    }
    double first_arrival = 0.0;
    double last_completion = 0.0;
    bool have_request = false;
    for (const auto &record : output.requests) {
        summary.cached_tokens += record.cached_prefill_tokens;
        if (!have_request) {
            first_arrival = record.arrived_at.seconds();
            have_request = true;
        } else {
            first_arrival = std::min(first_arrival, record.arrived_at.seconds());
        }
        last_completion = std::max(last_completion,
                                   record.completed_at.seconds());
        if (record.request_id.value() >=
            static_cast<std::int64_t>(sessions)) {
            const double ttft_ms =
                (record.first_token_completed_at.seconds() -
                 record.arrived_at.seconds()) *
                1'000.0;
            summary.successor_ttft_sum_ms += ttft_ms;
            ++summary.successor_count;
        }
    }
    summary.makespan_ms = (last_completion - first_arrival) * 1'000.0;
    summary.offloads = output.aggregate.cpu_kv_cache.offload_operations;
    summary.restores = output.aggregate.cpu_kv_cache.restore_operations;
    summary.evicted_blocks = output.aggregate.cpu_kv_cache.evicted_blocks;
    summary.evicted_sessions = output.aggregate.cpu_kv_cache.evicted_sessions;
    summary.skipped_offloads =
        output.aggregate.cpu_kv_cache.skipped_offloads;
    summary.truncated_offloads =
        output.aggregate.cpu_kv_cache.truncated_offloads;
    return summary;
}

void print_summary(const std::string &label, const TierSummary &summary) {
    const double average = summary.successor_count == 0
                               ? 0.0
                               : summary.successor_ttft_sum_ms /
                                     static_cast<double>(summary.successor_count);
    std::cout << "[CPU-MATRIX] " << label
              << " prefill_scheduled_tokens="
              << summary.prefill_scheduled_tokens
              << " cached_prefill_tokens=" << summary.cached_tokens
              << " successor_ttft_sum_ms=" << summary.successor_ttft_sum_ms
              << " successor_ttft_avg_ms=" << average
              << " makespan_ms=" << summary.makespan_ms
              << " offloads=" << summary.offloads
              << " restores=" << summary.restores
              << " evicted_blocks=" << summary.evicted_blocks
              << " evicted_sessions=" << summary.evicted_sessions
              << " skipped_offloads=" << summary.skipped_offloads
              << " truncated_offloads=" << summary.truncated_offloads
              << '\n';
}

void require_pdd_functional_equivalence(const SimulationOutput &enabled,
                                        const SimulationOutput &disabled,
                                        const std::string &context) {
    expect(enabled.requests.size() == disabled.requests.size(),
           context + ": CPU toggle changed completion count");
    expect(enabled.aggregate.kv_cache_transfer_count ==
               disabled.aggregate.kv_cache_transfer_count &&
               enabled.aggregate.kv_cache_transfer_count > 0,
           context + ": CPU toggle changed PDD transfer count");
    std::map<std::int64_t, std::tuple<std::int64_t, std::int64_t,
                                     std::int64_t, std::int64_t, std::uint64_t>>
        enabled_requests;
    for (const auto &record : enabled.requests) {
        enabled_requests.emplace(
            record.request_id.value(),
            std::make_tuple(record.prefill_replica_id.value(),
                            record.prefill_dp_id.value(),
                            record.decode_replica_id.value(),
                            record.decode_dp_id.value(),
                            record.kv_cache_transfer_size_bytes));
    }
    for (const auto &record : disabled.requests) {
        const auto position = enabled_requests.find(record.request_id.value());
        expect(position != enabled_requests.end(),
               context + ": CPU toggle dropped a request identity");
        expect(position->second ==
                   std::make_tuple(record.prefill_replica_id.value(),
                                   record.prefill_dp_id.value(),
                                   record.decode_replica_id.value(),
                                   record.decode_dp_id.value(),
                                   record.kv_cache_transfer_size_bytes),
               context + ": CPU toggle changed PDD affinity/transfer result");
    }
}

void require_cpu_disabled_zero_metrics(const SimulationOutput &output,
                                       const std::string &context) {
    const auto &cpu = output.aggregate.cpu_kv_cache;
    expect(cpu.target_count == 0 && cpu.offload_operations == 0 &&
               cpu.restore_operations == 0 && cpu.query_blocks == 0 &&
               cpu.hit_blocks == 0 && cpu.offload_bytes == 0 &&
               cpu.restore_bytes == 0 && output.cpu_kv_cache_targets.empty() &&
               output.cpu_kv_cache_transfers.empty(),
           context + ": CPU-disabled aggregate metrics are nonzero");
    for (const auto &record : output.requests) {
        expect(record.cpu_prefix_query_blocks == 0 &&
                   record.cpu_prefix_hit_blocks == 0 &&
                   record.cpu_restore_transferred_blocks == 0 &&
                   record.cpu_restore_bytes == 0 && record.cpu_offload_bytes == 0,
               context + ": CPU-disabled request metrics are nonzero");
    }
}

void test_capacity_policy_mode_matrix() {
    const std::vector<MatrixScenario> scenarios = {
        {"online_direct_prefix_capacity_1", SimulationMode::kOnline, 1,
         CpuKVCacheCapacityPressurePolicy::kPrefixFit, false, false, false},
        {"online_direct_prefix_capacity_2", SimulationMode::kOnline, 2,
         CpuKVCacheCapacityPressurePolicy::kPrefixFit, false, false, false},
        {"online_direct_skip_capacity_1", SimulationMode::kOnline, 1,
         CpuKVCacheCapacityPressurePolicy::kSkipOffload, false, false, false},
        {"online_direct_skip_capacity_8", SimulationMode::kOnline, 8,
         CpuKVCacheCapacityPressurePolicy::kSkipOffload, false, false, false},
        {"online_direct_prefix_capacity_8", SimulationMode::kOnline, 8,
         CpuKVCacheCapacityPressurePolicy::kPrefixFit, false, false, false},
        {"offline_direct_prefix_capacity_8", SimulationMode::kOffline, 8,
         CpuKVCacheCapacityPressurePolicy::kPrefixFit, false, false, false},
        {"online_static_prefix_capacity_1", SimulationMode::kOnline, 1,
         CpuKVCacheCapacityPressurePolicy::kPrefixFit, true, false, false},
        {"online_multi_target_prefix_capacity_4", SimulationMode::kOnline, 4,
         CpuKVCacheCapacityPressurePolicy::kPrefixFit, false, false, true},
        {"online_direct_prefix_capacity_8_slow", SimulationMode::kOnline, 8,
         CpuKVCacheCapacityPressurePolicy::kPrefixFit, false, true, false},
    };

    for (const auto &scenario : scenarios) {
        const SimulationConfig config = configure(scenario);
        const std::uint64_t sessions = scenario.multi_target ? 12 : 8;
        const std::uint64_t expected_targets =
            config.pdd().clusters.prefill.parallelism.num_replicas *
            config.pdd().clusters.prefill.parallelism.data_parallel_size;
        const bool burst_pressure =
            scenario.pressure_policy ==
                CpuKVCacheCapacityPressurePolicy::kSkipOffload &&
            scenario.capacity_blocks == 8;
        const bool oversized_pressure = burst_pressure;
        const auto output =
            run_simulation(config, make_workload(sessions, 2, burst_pressure,
                                                 oversized_pressure));
        require_quiescent(output, config, sessions * 2, expected_targets,
                          scenario.name);
        require_affinity(output, scenario.name);
        print_summary(
            std::string{"matrix "} + scenario.name,
            summarize(output, sessions));

        const auto &cpu = output.aggregate.cpu_kv_cache;
        const auto resolved =
            frontier::config::resolve_cpu_kv_cache_target(config);
        expect(cpu.capacity_blocks == expected_targets *
                                      resolved.capacity_blocks,
               std::string{scenario.name} + ": capacity resolution mismatch");
        if (scenario.pressure_policy ==
            CpuKVCacheCapacityPressurePolicy::kPrefixFit &&
            scenario.capacity_blocks == 1 && !scenario.static_slice) {
            expect(cpu.truncated_offloads > 0,
                   std::string{scenario.name} +
                       ": prefix_fit must report truncation");
        }
        if (scenario.pressure_policy ==
                CpuKVCacheCapacityPressurePolicy::kSkipOffload &&
            scenario.capacity_blocks == 1 && !scenario.static_slice) {
            expect(cpu.skipped_offloads > 0,
                   std::string{scenario.name} +
                       ": skip_offload must report skipped snapshots");
        }
        if (scenario.pressure_policy ==
                CpuKVCacheCapacityPressurePolicy::kPrefixFit) {
            expect(cpu.offload_operations > 0 && cpu.evicted_blocks > 0,
                   std::string{scenario.name} +
                       ": prefix_fit must materialize and evict snapshots");
        }
        if (scenario.pressure_policy ==
                CpuKVCacheCapacityPressurePolicy::kPrefixFit &&
            scenario.capacity_blocks == 8 && !scenario.multi_target &&
            !scenario.slow_transfer && scenario.mode == SimulationMode::kOnline) {
            expect(cpu.offload_operations > sessions &&
                       cpu.offload_operations >= 16 &&
                       cpu.restore_operations >= 4 && cpu.hit_blocks > 0 &&
                       cpu.evicted_blocks >= 16 && cpu.evicted_sessions >= 4,
                   std::string{scenario.name} +
                       ": pressure workload must repeatedly evict/offload/restore");
        }
        if (scenario.pressure_policy ==
                CpuKVCacheCapacityPressurePolicy::kSkipOffload &&
            scenario.capacity_blocks == 8) {
            expect(cpu.offload_operations > 0 && cpu.skipped_offloads > 0,
                   std::string{scenario.name} +
                       ": skip_offload must commit fits and skip pressure");
        }
        if (scenario.multi_target) {
            expect(output.cpu_kv_cache_targets.size() == 4,
                   "multi-target scenario must expose four CPU targets");
            std::map<std::pair<std::int64_t, std::int64_t>, std::uint64_t>
                operations_by_target;
            for (const auto &transfer : output.cpu_kv_cache_transfers) {
                ++operations_by_target[{transfer.replica_id.value(),
                                         transfer.dp_id.value()}];
            }
            expect(operations_by_target.size() == 4,
                   "multi-target scenario must exercise every CPU target");
        }
    }
}

void test_cpu_on_off_and_fast_slow_observability() {
    MatrixScenario fast_scenario{
        "online_direct_prefix_capacity_8_fast_compare", SimulationMode::kOnline,
        8, CpuKVCacheCapacityPressurePolicy::kPrefixFit, false, false, false};
    MatrixScenario slow_scenario = fast_scenario;
    slow_scenario.name = "online_direct_prefix_capacity_8_slow_compare";
    slow_scenario.slow_transfer = true;

    const auto workload = make_workload(8);
    const SimulationConfig fast_config = configure(fast_scenario);
    const auto fast_enabled = run_simulation(fast_config, workload);
    auto fast_disabled_config = fast_config;
    fast_disabled_config.run_id = "cpu-kv-cache-matrix-fast-off";
    fast_disabled_config.cpu_kv_cache = frontier::config::CpuKVCacheConfig{};
    const auto fast_disabled = run_simulation(fast_disabled_config, workload);

    const SimulationConfig slow_config = configure(slow_scenario);
    const auto slow_enabled = run_simulation(slow_config, workload);
    auto slow_disabled_config = slow_config;
    slow_disabled_config.run_id = "cpu-kv-cache-matrix-slow-off";
    slow_disabled_config.cpu_kv_cache = frontier::config::CpuKVCacheConfig{};
    const auto slow_disabled = run_simulation(slow_disabled_config, workload);

    require_quiescent(fast_enabled, fast_config, workload.size(), 1,
                      "fast CPU-on");
    require_quiescent(slow_enabled, slow_config, workload.size(), 1,
                      "slow CPU-on");
    require_quiescent(fast_disabled, fast_disabled_config, workload.size(), 0,
                      "fast CPU-off");
    require_quiescent(slow_disabled, slow_disabled_config, workload.size(), 0,
                      "slow CPU-off");
    require_cpu_disabled_zero_metrics(fast_disabled, "fast CPU-off");
    require_cpu_disabled_zero_metrics(slow_disabled, "slow CPU-off");
    require_affinity(fast_enabled, "fast CPU-on");
    require_affinity(fast_disabled, "fast CPU-off");
    require_affinity(slow_enabled, "slow CPU-on");
    require_affinity(slow_disabled, "slow CPU-off");

    require_pdd_functional_equivalence(fast_enabled, fast_disabled,
                                       "fast CPU toggle");
    require_pdd_functional_equivalence(slow_enabled, slow_disabled,
                                       "slow CPU toggle");
    require_pdd_functional_equivalence(fast_enabled, slow_enabled,
                                       "fast/slow CPU-on transfer");

    expect(fast_enabled.aggregate.cpu_kv_cache.offload_operations > 0 &&
               fast_enabled.aggregate.cpu_kv_cache.restore_operations > 0 &&
               fast_enabled.aggregate.cpu_kv_cache.hit_blocks > 0,
           "CPU-on tier must observe offload, restore, and CPU hits");
    const TierSummary fast_on_summary = summarize(fast_enabled, 8);
    const TierSummary fast_off_summary = summarize(fast_disabled, 8);
    const TierSummary slow_on_summary = summarize(slow_enabled, 8);
    const TierSummary slow_off_summary = summarize(slow_disabled, 8);
    print_summary("fast-tier ON", fast_on_summary);
    print_summary("fast-tier OFF", fast_off_summary);
    print_summary("slow-tier ON", slow_on_summary);
    print_summary("slow-tier OFF", slow_off_summary);

    // Repeat the same ON/OFF contract under offline PDD.  Offline execution
    // may legitimately consume no restore (the prefill/decode barrier is
    // different), but completion, affinity, transfer identity, and zeroed
    // disabled-tier metrics remain observable contracts.
    MatrixScenario offline_scenario = fast_scenario;
    offline_scenario.name = "offline_direct_prefix_capacity_8_compare";
    offline_scenario.mode = SimulationMode::kOffline;
    const SimulationConfig offline_config = configure(offline_scenario);
    const auto offline_enabled = run_simulation(offline_config, workload);
    auto offline_disabled_config = offline_config;
    offline_disabled_config.run_id = "cpu-kv-cache-matrix-offline-off";
    offline_disabled_config.cpu_kv_cache = frontier::config::CpuKVCacheConfig{};
    const auto offline_disabled =
        run_simulation(offline_disabled_config, workload);
    require_quiescent(offline_enabled, offline_config, workload.size(), 1,
                      "offline CPU-on");
    require_quiescent(offline_disabled, offline_disabled_config, workload.size(),
                      0, "offline CPU-off");
    require_cpu_disabled_zero_metrics(offline_disabled, "offline CPU-off");
    require_affinity(offline_enabled, "offline CPU-on");
    require_affinity(offline_disabled, "offline CPU-off");
    require_pdd_functional_equivalence(offline_enabled, offline_disabled,
                                       "offline CPU toggle");
    expect(offline_enabled.aggregate.cpu_kv_cache.offload_operations > 0,
           "offline CPU-on tier must still expose D2H snapshots");
    print_summary("offline-tier ON", summarize(offline_enabled, 8));
    print_summary("offline-tier OFF", summarize(offline_disabled, 8));

    expect(fast_on_summary.cached_tokens > fast_off_summary.cached_tokens ||
               fast_on_summary.prefill_scheduled_tokens <
                   fast_off_summary.prefill_scheduled_tokens,
           "CPU-on tier must add cached prefix reuse or reduce PREFILL work");
    expect(slow_on_summary.cached_tokens > slow_off_summary.cached_tokens ||
               slow_on_summary.prefill_scheduled_tokens <
                   slow_off_summary.prefill_scheduled_tokens,
           "slow CPU-on tier must retain a measurable reuse/work effect");
    // The causal latency observation is transfer service, not an assumed
    // end-to-end speedup: slower bandwidth must increase H2D service time when
    // a restore is actually consumed.
    expect(slow_enabled.aggregate.cpu_kv_cache.h2d_service_time_ms >
               fast_enabled.aggregate.cpu_kv_cache.h2d_service_time_ms &&
               slow_enabled.aggregate.cpu_kv_cache.d2h_service_time_ms >
                   fast_enabled.aggregate.cpu_kv_cache.d2h_service_time_ms,
           "slow tier must expose greater causal transfer service time");
}

} // namespace

int main() {
    int failures = 0;
    failures += frontier::test::run("CPU KV-cache stress capacity/policy/mode matrix",
                                    test_capacity_policy_mode_matrix);
    failures += frontier::test::run(
        "CPU KV-cache stress CPU-on/off and fast/slow observability",
        test_cpu_on_off_and_fast_slow_observability);
    return failures == 0 ? 0 : 1;
}
