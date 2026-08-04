#include "frontier/config/config.h"
#include "frontier/request_generator/workload.h"
#include "frontier/simulator/simulator.h"
#include "tests/test_support.h"

#include <cstdint>
#include <filesystem>
#include <vector>

#ifndef FRONTIER_EXAMPLE_DIR
#error "FRONTIER_EXAMPLE_DIR must be defined for CPU KV-cache stress tests"
#endif

namespace {

using frontier::RequestId;
using frontier::SessionId;
using frontier::SimTime;
using frontier::config::parse_simulation_config_json;
using frontier::request_generator::WorkloadRequest;
using frontier::simulator::run_simulation;
using frontier::test::expect;
using frontier::test::read_text_file;

std::vector<WorkloadRequest> make_workload(std::uint64_t sessions,
                                           std::uint64_t rounds) {
    std::vector<WorkloadRequest> result;
    result.reserve(static_cast<std::size_t>(sessions * rounds));
    for (std::uint64_t round = 0; round < rounds; ++round) {
        for (std::uint64_t session = 0; session < sessions; ++session) {
            WorkloadRequest value{};
            value.request_id = RequestId{result.size()};
            if (round == 0) {
                value.session_start_at =
                    SimTime::from_seconds((session / 2) * 0.2 +
                                          (session % 2) * 0.01);
                value.num_prefill_tokens = 8 + (session % 2) * 4;
            } else {
                value.think_time = SimTime::from_seconds(0.05);
                value.num_prefill_tokens = 4;
            }
            value.num_decode_tokens = 2;
            value.session_id = SessionId{1'000 + session};
            value.session_turn_index = round;
            result.push_back(value);
        }
    }
    return result;
}

void test_cpu_kv_cache_pressure_reaches_quiescence() {
    const std::filesystem::path root{FRONTIER_EXAMPLE_DIR};
    auto config = parse_simulation_config_json(read_text_file(
        root / "configs" / "06_cpu_kv_cache_pdd_online.json"));
    config.run_id = "cpu-kv-cache-stress";
    config.pdd().clusters.prefill.scheduler.num_blocks = 5;
    const std::uint64_t bytes_per_block =
        frontier::config::resolve_cpu_kv_cache_target(config).bytes_per_block;
    config.cpu_kv_cache.capacity_bytes = bytes_per_block * 8;
    const std::uint64_t sessions = 8;
    // Two rounds keep each accumulated session within the five-block GPU
    // target while eight competing sessions still force CPU-tier eviction.
    const std::uint64_t rounds = 2;
    const auto output = run_simulation(config, make_workload(sessions, rounds));
    expect(output.requests.size() == sessions * rounds,
           "CPU tier pressure workload did not drain");
    expect(output.aggregate.cpu_kv_cache.offload_operations > sessions &&
               output.aggregate.cpu_kv_cache.restore_operations > 0 &&
               output.aggregate.cpu_kv_cache.hit_blocks > 0,
           "CPU tier pressure workload did not exercise offload and restore: "
           "offloads=" +
               std::to_string(
                   output.aggregate.cpu_kv_cache.offload_operations) +
               ", restores=" +
               std::to_string(
                   output.aggregate.cpu_kv_cache.restore_operations) +
               ", hit_blocks=" +
               std::to_string(output.aggregate.cpu_kv_cache.hit_blocks));
    expect(output.cpu_kv_cache_targets.size() == 1,
           "CPU tier stress must expose one PREFILL target");
    const auto &target = output.cpu_kv_cache_targets.front();
    expect(target.capacity_blocks == 8 && target.evicted_blocks > 0 &&
               target.evicted_sessions > 0 &&
               target.resident_blocks + target.reserved_blocks <=
                   target.capacity_blocks &&
               target.active_offload_reservations == 0 &&
               target.active_restore_leases == 0 &&
               target.pending_restore_operations == 0 &&
               target.staged_restore_payloads == 0,
           "five-GPU/eight-CPU-block stress must evict and then quiesce");
}

} // namespace

int main() {
    return frontier::test::run("CPU KV-cache pressure quiescence",
                               test_cpu_kv_cache_pressure_reaches_quiescence);
}
