#include "frontier/config/config.h"
#include "frontier/metrics/output_contract.h"
#include "frontier/request_generator/workload.h"
#include "frontier/simulator/simulator.h"
#include "tests/test_support.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#ifndef FRONTIER_EXAMPLE_DIR
#error "FRONTIER_EXAMPLE_DIR must be defined for CPU KV-cache integration tests"
#endif

namespace {

using frontier::EventType;
using frontier::RequestId;
using frontier::SessionId;
using frontier::SimTime;
using frontier::config::parse_simulation_config_json;
using frontier::metrics::RequestMetricsRecord;
using frontier::metrics::SimulationOutput;
using frontier::request_generator::parse_workload_csv;
using frontier::simulator::run_simulation;
using frontier::test::expect;
using frontier::test::read_text_file;

const std::filesystem::path kExampleRoot{FRONTIER_EXAMPLE_DIR};

SimulationOutput run_example(const char *config_name) {
    const auto config = parse_simulation_config_json(
        read_text_file(kExampleRoot / "configs" / config_name));
    const auto workload = parse_workload_csv(read_text_file(
        kExampleRoot / "workloads" / "06_cpu_kv_cache_sessions.csv"));
    return run_simulation(config, workload);
}

SimulationOutput run_config(frontier::config::SimulationConfig config) {
    const auto workload = parse_workload_csv(read_text_file(
        kExampleRoot / "workloads" / "06_cpu_kv_cache_sessions.csv"));
    return run_simulation(config, workload);
}

SimulationOutput run_config_without_details(
    const frontier::config::SimulationConfig &config) {
    const auto workload = parse_workload_csv(read_text_file(
        kExampleRoot / "workloads" / "06_cpu_kv_cache_sessions.csv"));
    frontier::simulator::Simulator simulator{config, workload};
    simulator.metrics().set_detailed_traces_enabled(false);
    return simulator.run();
}

SimulationOutput run_workload(
    const frontier::config::SimulationConfig &config,
    const std::vector<frontier::request_generator::WorkloadRequest> &workload) {
    return run_simulation(config, workload);
}

const RequestMetricsRecord &request(const SimulationOutput &output,
                                    RequestId id) {
    const auto position = std::find_if(
        output.requests.begin(), output.requests.end(),
        [id](const RequestMetricsRecord &value) {
            return value.request_id == id;
        });
    if (position == output.requests.end()) {
        throw std::runtime_error("missing request metrics");
    }
    return *position;
}

void validate_cpu_tier(const SimulationOutput &output,
                       const std::string &context) {
    expect(output.requests.size() == 3,
           context + ": all session turns must complete");
    const RequestMetricsRecord &successor = request(output, RequestId{2});
    expect(successor.gpu_prefix_hit_blocks == 1 &&
               successor.cpu_prefix_query_blocks == 2 &&
               successor.cpu_prefix_hit_blocks == 1 &&
               successor.prefix_cache_hit_blocks == 2 &&
               successor.cached_prefill_tokens == 8,
           context + ": GPU and CPU ranges must form the expected frontier");
    expect(successor.cpu_restore_transferred_blocks == 1 &&
               successor.cpu_restore_consumed_blocks == 1 &&
               successor.cpu_restore_discarded_blocks == 0 &&
               successor.cpu_restored_tokens == 4 &&
               successor.cpu_restore_bytes > 0,
           context + ": restored traffic and consumed reuse must be distinct");

    const auto &cpu = output.aggregate.cpu_kv_cache;
    expect(cpu.target_count == 1 && cpu.offload_operations == 3 &&
               cpu.offload_blocks == 6 && cpu.restore_operations == 1 &&
               cpu.restore_blocks == 1 && cpu.query_blocks == 7 &&
               cpu.hit_blocks == 1,
           context + ": CPU target aggregate differs from the logical oracle");
    expect(output.cpu_kv_cache_targets.size() == 1 &&
               output.cpu_kv_cache_targets.front().resident_blocks == 6 &&
               output.cpu_kv_cache_targets.front().reserved_blocks == 0 &&
               output.cpu_kv_cache_targets.front()
                       .active_offload_reservations == 0 &&
               output.cpu_kv_cache_targets.front().active_restore_leases == 0,
           context + ": final target diagnostics must be quiescent");
    const auto &target = output.cpu_kv_cache_targets.front();
    expect(cpu.capacity_bytes == target.capacity_bytes &&
               cpu.capacity_blocks == target.capacity_blocks &&
               cpu.bytes_per_block == target.bytes_per_block &&
               cpu.resident_bytes == target.resident_bytes &&
               cpu.resident_blocks == target.resident_blocks &&
               cpu.reserved_bytes == target.reserved_bytes &&
               cpu.reserved_blocks == target.reserved_blocks &&
               cpu.free_blocks == target.free_blocks &&
               cpu.peak_resident_bytes == target.peak_resident_bytes &&
               cpu.peak_reserved_bytes == target.peak_reserved_bytes &&
               cpu.resident_sessions == target.resident_sessions &&
               cpu.evicted_bytes == target.evicted_bytes &&
               cpu.skipped_offloads == target.skipped_offloads &&
               cpu.truncated_offloads == target.truncated_offloads &&
               cpu.sessions_with_cpu_hits == target.sessions_with_cpu_hits,
           context + ": single-target system statistics must equal target statistics");
    expect(output.cpu_kv_cache_transfers.size() == 4,
           context + ": full output must retain detailed CPU transfers");
    std::uint64_t request_offload_bytes = 0;
    std::uint64_t request_restore_bytes = 0;
    for (const auto &value : output.requests) {
        request_offload_bytes += value.cpu_offload_bytes;
        request_restore_bytes += value.cpu_restore_bytes;
    }
    expect(request_offload_bytes == cpu.offload_bytes &&
               request_restore_bytes == cpu.restore_bytes,
           context + ": request and system CPU byte totals must agree");
    for (const auto &transfer : output.cpu_kv_cache_transfers) {
        expect(transfer.size_bytes > 0 && transfer.blocks > 0 &&
                   transfer.completed_at >= transfer.started_at &&
                   transfer.started_at >= transfer.submitted_at,
               context + ": CPU transfer timing is invalid");
    }
    const auto has_event = [&](EventType type) {
        return std::any_of(output.event_trace.begin(), output.event_trace.end(),
                           [type](const auto &event) {
                               return event.type() == type;
                           });
    };
    expect(has_event(EventType::kCpuKvCacheOffloadStart) &&
               has_event(EventType::kCpuKvCacheOffloadEnd) &&
               has_event(EventType::kCpuKvCacheRestoreStart) &&
               has_event(EventType::kCpuKvCacheRestoreEnd),
           context + ": typed CPU events must remain in the full trace");
    const std::string json =
        frontier::metrics::serialize_simulation_output_json(output);
    expect(json.find("\"cpu_restore_transferred_blocks\"") !=
                   std::string::npos &&
               json.find("\"cpu_kv_cache_targets\"") != std::string::npos &&
               json.find("\"cpu_kv_cache_transfers\"") !=
                   std::string::npos,
           context + ": serialized output is missing CPU KV-cache contracts");
}

void test_capacity_policies_and_slow_restore_latency() {
    auto base = parse_simulation_config_json(read_text_file(
        kExampleRoot / "configs" / "06_cpu_kv_cache_pdd_online.json"));
    const auto fast = run_config(base);
    auto slow = base;
    slow.run_id = "cpu-kv-cache-slow-h2d";
    slow.cpu_kv_cache.read_bandwidth_gbps = 0.01;
    const auto slow_output = run_config(slow);
    const auto ttft = [](const RequestMetricsRecord &record) {
        return record.first_token_completed_at.seconds() -
               record.arrived_at.seconds();
    };
    expect(ttft(request(slow_output, RequestId{2})) >
                   ttft(request(fast, RequestId{2})) + 1.0 &&
               slow_output.aggregate.cpu_kv_cache.h2d_service_time_ms >
                   fast.aggregate.cpu_kv_cache.h2d_service_time_ms,
           "slow H2D service must contribute to follow-up TTFT");

    auto prefix_fit = base;
    prefix_fit.run_id = "cpu-kv-cache-one-block-prefix-fit";
    prefix_fit.cpu_kv_cache.capacity_bytes = 2'097'152;
    const auto fit_output = run_config(prefix_fit);
    expect(fit_output.cpu_kv_cache_targets.front().capacity_blocks == 1 &&
               fit_output.cpu_kv_cache_targets.front().truncated_offloads > 0,
           "one-block prefix_fit runtime must report truncation");

    auto skip = prefix_fit;
    skip.run_id = "cpu-kv-cache-one-block-skip";
    skip.cpu_kv_cache.capacity_pressure_policy =
        frontier::config::CpuKVCacheCapacityPressurePolicy::kSkipOffload;
    const auto skip_output = run_config(skip);
    expect(skip_output.cpu_kv_cache_targets.front().skipped_offloads > 0 &&
               skip_output.cpu_kv_cache_targets.front().resident_blocks <= 1,
           "one-block skip_offload runtime must report skipped snapshots");
}

void test_online_and_offline_cpu_kv_cache_tiering() {
    validate_cpu_tier(run_example("06_cpu_kv_cache_pdd_online.json"),
                      "online");
    validate_cpu_tier(run_example("07_cpu_kv_cache_pdd_offline.json"),
                      "offline");
}

void test_cpu_disabled_and_detail_suppression_controls() {
    auto enabled = parse_simulation_config_json(read_text_file(
        kExampleRoot / "configs" / "06_cpu_kv_cache_pdd_online.json"));
    const auto compact = run_config_without_details(enabled);
    expect(compact.requests.size() == 3 &&
               compact.aggregate.cpu_kv_cache.offload_operations == 3 &&
               compact.aggregate.cpu_kv_cache.restore_operations == 1 &&
               compact.cpu_kv_cache_targets.size() == 1 &&
               compact.cpu_kv_cache_transfers.empty() &&
               compact.event_trace.empty(),
           "summary/request collection must retain CPU aggregates without detailed traces");

    enabled.run_id = "cpu-kv-cache-disabled-control";
    enabled.cpu_kv_cache = frontier::config::CpuKVCacheConfig{};
    const auto disabled = run_config(enabled);
    expect(disabled.requests.size() == 3 &&
               disabled.aggregate.kv_cache_transfer_count == 3 &&
               disabled.aggregate.cpu_kv_cache.target_count == 0 &&
               disabled.aggregate.cpu_kv_cache.offload_operations == 0 &&
               disabled.aggregate.cpu_kv_cache.restore_operations == 0 &&
               disabled.cpu_kv_cache_targets.empty() &&
               disabled.cpu_kv_cache_transfers.empty(),
           "CPU-disabled control must preserve Step 4 PDD completion with additive zero metrics");
    for (const auto &record : disabled.requests) {
        expect(record.cpu_prefix_query_blocks == 0 &&
                   record.cpu_prefix_hit_blocks == 0 &&
                   record.cpu_restore_transferred_blocks == 0 &&
                   record.cpu_restore_bytes == 0 &&
                   record.cpu_offload_bytes == 0,
               "CPU-disabled requests must retain zero CPU-tier fields");
    }
}

void test_slow_queues_are_full_duplex_and_source_hold_is_attributable() {
    auto config = parse_simulation_config_json(read_text_file(
        kExampleRoot / "configs" / "06_cpu_kv_cache_pdd_online.json"));
    config.run_id = "cpu-kv-cache-slow-full-duplex";
    config.pdd().clusters.prefill.scheduler.batch_size_cap = 1;
    config.pdd().clusters.prefill.scheduler.num_blocks = 5;
    config.cpu_kv_cache.write_latency_ms = 20.0;
    config.cpu_kv_cache.read_latency_ms = 20.0;

    std::vector<frontier::request_generator::WorkloadRequest> workload;
    const auto add_initial = [&](std::uint64_t id, std::int64_t session,
                                 double arrival, std::uint64_t prompt) {
        frontier::request_generator::WorkloadRequest value{};
        value.request_id = RequestId{id};
        value.session_start_at = SimTime::from_seconds(arrival);
        value.num_prefill_tokens = prompt;
        value.num_decode_tokens = 2;
        value.session_id = SessionId{session};
        value.session_turn_index = 0;
        workload.push_back(value);
    };
    add_initial(0, 70, 0.0, 8);
    add_initial(1, 71, 0.0001, 12);
    add_initial(2, 72, 0.0002, 8);
    frontier::request_generator::WorkloadRequest successor{};
    successor.request_id = RequestId{3};
    successor.think_time = SimTime::from_seconds(0.055);
    successor.num_prefill_tokens = 4;
    successor.num_decode_tokens = 2;
    successor.session_id = SessionId{70};
    successor.session_turn_index = 1;
    workload.push_back(successor);

    const auto output = run_workload(config, workload);
    expect(output.requests.size() == workload.size() &&
               output.aggregate.cpu_kv_cache.d2h_queue_time_ms > 0.0 &&
               output.aggregate.cpu_kv_cache.restore_operations > 0 &&
               output.aggregate.cpu_kv_cache.source_gpu_hold_time_ms > 0.0,
           "slow system transfers must expose queueing, restore, and attributable source hold");
    bool duplex_overlap = false;
    for (const auto &left : output.cpu_kv_cache_transfers) {
        for (const auto &right : output.cpu_kv_cache_transfers) {
            if (left.kind == frontier::metrics::CpuKVCacheTransferKind::kOffload &&
                right.kind == frontier::metrics::CpuKVCacheTransferKind::kRestore &&
                left.started_at < right.completed_at &&
                right.started_at < left.completed_at) {
                duplex_overlap = true;
            }
        }
    }
    expect(duplex_overlap,
           "an H2D restore must overlap an independently queued D2H offload");
    const auto &target = output.cpu_kv_cache_targets.front();
    expect(target.active_offload_reservations == 0 &&
               target.active_restore_leases == 0 &&
               target.pending_restore_operations == 0 &&
               target.staged_restore_payloads == 0,
           "slow full-duplex workload must release all transient ownership");
}

} // namespace

int main() {
    int failures = 0;
    failures += frontier::test::run(
        "CPU KV-cache PDD online/offline E2E",
        test_online_and_offline_cpu_kv_cache_tiering);
    failures += frontier::test::run(
        "CPU KV-cache policies and slow restore",
        test_capacity_policies_and_slow_restore_latency);
    failures += frontier::test::run(
        "CPU KV-cache disabled and compact-output controls",
        test_cpu_disabled_and_detail_suppression_controls);
    failures += frontier::test::run(
        "CPU KV-cache slow full-duplex queues",
        test_slow_queues_are_full_duplex_and_source_hold_is_attributable);
    return failures == 0 ? 0 : 1;
}
