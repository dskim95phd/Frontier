#include "frontier/config/config.h"
#include "frontier/metrics/output_contract.h"
#include "frontier/request_generator/workload.h"
#include "frontier/simulator/simulator.h"
#include "tests/test_support.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#ifndef FRONTIER_TEST_FIXTURE_DIR
#error "FRONTIER_TEST_FIXTURE_DIR must be defined for integration tests"
#endif

namespace {

using Target = std::pair<std::uint64_t, std::uint64_t>;
using frontier::ClusterType;
using frontier::RequestId;
using frontier::SessionId;
using frontier::SimTime;
using frontier::config::ClusterRuntimeConfig;
using frontier::config::ClusterSchedulerType;
using frontier::config::ExecutionModelType;
using frontier::config::MoeRoutingDistribution;
using frontier::config::MoeRoutingMode;
using frontier::config::ParallelismConfig;
using frontier::config::parse_simulation_config_json;
using frontier::config::PrefixCachingKeyMode;
using frontier::config::SimulationConfig;
using frontier::config::SimulationMode;
using frontier::metrics::SimulationOutput;
using frontier::request_generator::materialize_workload_for_config;
using frontier::request_generator::WorkloadRequest;
using frontier::simulator::run_simulation;
using frontier::test::expect;
using frontier::test::read_text_file;

const std::filesystem::path kFixtureRoot{FRONTIER_TEST_FIXTURE_DIR};

struct Topology {
    std::uint64_t replicas;
    std::uint64_t tp;
    std::uint64_t pp;
    std::uint64_t dp;
    std::uint64_t moe_tp;
    std::uint64_t ep;
};

struct StressCase {
    const char *name;
    bool moe;
    bool pdd;
    Topology prefill_or_monolithic;
    Topology decode;
    std::uint64_t cache_blocks;
    std::uint64_t decode_blocks;
    std::uint64_t sessions;
    std::uint64_t rounds;
    bool pressure_prefill;
    std::uint64_t seed;
};

ParallelismConfig parallelism(const Topology &topology) {
    ParallelismConfig result{};
    result.num_replicas = topology.replicas;
    result.tensor_parallel_size = topology.tp;
    result.pipeline_parallel_size = topology.pp;
    result.data_parallel_size = topology.dp;
    result.moe_tensor_parallel_size = topology.moe_tp;
    result.moe_expert_parallel_size = topology.ep;
    return result;
}

std::uint64_t target_count(const Topology &topology) {
    return topology.replicas * topology.dp;
}

std::set<Target> expected_targets(const Topology &topology) {
    std::set<Target> result;
    for (std::uint64_t replica = 0; replica < topology.replicas; ++replica) {
        for (std::uint64_t dp = 0; dp < topology.dp; ++dp) {
            result.emplace(replica, dp);
        }
    }
    return result;
}

SimulationConfig load_base(bool moe, bool pdd) {
    const char *name = nullptr;
    if (moe) {
        name = pdd ? "fixed_moe_sequential_pdd.json"
                   : "fixed_moe_local_colocation.json";
    } else {
        name = pdd ? "fixed_sequential_pdd.json"
                   : "fixed_parallel_colocation.json";
    }
    return parse_simulation_config_json(
        read_text_file(kFixtureRoot / "config" / name));
}

void configure_runtime(ClusterRuntimeConfig &runtime, const Topology &topology,
                       std::uint64_t blocks, bool chunked_prefill) {
    runtime.parallelism = parallelism(topology);
    runtime.scheduler.block_size = 4;
    runtime.scheduler.num_blocks = blocks;
    runtime.scheduler.batch_size_cap = 8;
    runtime.scheduler.max_tokens_in_batch = 16;
    runtime.scheduler.enable_preemption = true;
    runtime.scheduler.enable_chunked_prefill = chunked_prefill;
    runtime.scheduler.long_prefill_token_threshold = 0;
    runtime.scheduler.watermark_blocks_fraction = 0.0;
    runtime.scheduler.num_preallocate_tokens = 0;
    runtime.execution_model.type = ExecutionModelType::kFixed;
    runtime.execution_model.fixed.batch_latency_ms = 0.001;
    runtime.execution_model.fixed.stage_latencies_ms.assign(
        static_cast<std::size_t>(topology.pp), 0.001);
    runtime.moe_routing.mode = MoeRoutingMode::kSimulation;
    runtime.moe_routing.distribution = MoeRoutingDistribution::kBalanced;
    runtime.moe_routing.seed = 42;
}

std::vector<WorkloadRequest> make_seeded_workload(std::uint64_t sessions,
                                                  std::uint64_t rounds,
                                                  std::uint64_t seed) {
    std::mt19937_64 generator{seed};
    std::uniform_int_distribution<std::uint64_t> prefill_distribution{1, 8};
    std::uniform_int_distribution<std::uint64_t> decode_distribution{1, 4};
    std::vector<WorkloadRequest> workload;
    workload.reserve(static_cast<std::size_t>(sessions * rounds));
    for (std::uint64_t round = 0; round < rounds; ++round) {
        const double think_time = static_cast<double>(round / 2) * 0.25;
        for (std::uint64_t session = 0; session < sessions; ++session) {
            WorkloadRequest request{};
            request.request_id = RequestId{workload.size()};
            if (round == 0) {
                request.session_start_at = SimTime::from_seconds(0.0);
            } else {
                request.think_time = SimTime::from_seconds(think_time);
            }
            request.num_prefill_tokens = prefill_distribution(generator);
            request.num_decode_tokens = decode_distribution(generator);
            request.session_id = SessionId{10'000 + session};
            request.session_turn_index = round;
            workload.push_back(request);
        }
    }
    return workload;
}

std::uint64_t ceil_div(std::uint64_t numerator, std::uint64_t denominator) {
    return numerator / denominator +
           static_cast<std::uint64_t>(numerator % denominator != 0);
}

void expect_individually_admissible(
    const std::vector<WorkloadRequest> &materialized,
    const StressCase &test_case) {
    for (const WorkloadRequest &request : materialized) {
        const std::uint64_t prefill_blocks =
            ceil_div(request.num_prefill_tokens, 4);
        const std::uint64_t total_blocks =
            ceil_div(request.num_prefill_tokens + request.num_decode_tokens, 4);
        expect(prefill_blocks <= test_case.cache_blocks,
               std::string{test_case.name} +
                   ": one PREFILL request exceeds cache-owner capacity");
        const std::uint64_t execution_capacity =
            test_case.pdd ? test_case.decode_blocks : test_case.cache_blocks;
        expect(total_blocks <= execution_capacity,
               std::string{test_case.name} +
                   ": one request exceeds execution-target capacity");
    }
}

void expect_preemption_on_every_target(const SimulationOutput &output,
                                       ClusterType cluster_type,
                                       const Topology &topology,
                                       const std::string &context) {
    std::set<Target> observed;
    for (const auto &trace : output.scheduler_trace) {
        if (trace.cluster_type != cluster_type) {
            continue;
        }
        if (std::any_of(trace.decisions.begin(), trace.decisions.end(),
                        [](const auto &decision) {
                            return decision.decision_result == "PREEMPTED";
                        })) {
            observed.emplace(trace.replica_id.value(), trace.dp_id.value());
        }
    }
    const std::set<Target> expected = expected_targets(topology);
    std::ostringstream detail;
    detail << context << ": preemption targets observed=";
    for (const auto &[replica, dp] : observed) {
        detail << replica << ':' << dp << ',';
    }
    detail << " expected=";
    for (const auto &[replica, dp] : expected) {
        detail << replica << ':' << dp << ',';
    }
    expect(observed == expected, detail.str());
}

void validate_output(const SimulationOutput &output,
                     const std::vector<WorkloadRequest> &materialized,
                     const StressCase &test_case) {
    const std::string context{test_case.name};
    expect(output.requests.size() == materialized.size(),
           context + ": stress workload did not drain");
    std::uint64_t request_query_blocks = 0;
    std::uint64_t request_hit_blocks = 0;
    std::uint64_t total_preemptions = 0;
    std::uint64_t maximum_preemptions = 0;
    for (const auto &record : output.requests) {
        expect(record.request_id.valid() &&
                   record.request_id.index() < materialized.size(),
               context + ": output contains an unknown request");
        const WorkloadRequest &expected_request =
            materialized.at(record.request_id.index());
        expect(record.session_id == expected_request.session_id &&
                   record.num_prefill_tokens ==
                       expected_request.num_prefill_tokens &&
                   record.num_decode_tokens ==
                       expected_request.num_decode_tokens &&
                   record.num_processed_tokens ==
                       record.num_prefill_tokens + record.num_decode_tokens,
               context + ": completed request metadata/progress mismatch");
        expect(record.prefix_cache_query_blocks ==
                       record.num_prefill_tokens / 4 &&
                   record.prefix_cache_hit_blocks <=
                       record.prefix_cache_query_blocks &&
                   record.cached_prefill_tokens ==
                       record.prefix_cache_hit_blocks * 4 &&
                   record.cached_prefill_tokens < record.num_prefill_tokens,
               context + ": request prefix lookup metrics are inconsistent");
        request_query_blocks += record.prefix_cache_query_blocks;
        request_hit_blocks += record.prefix_cache_hit_blocks;
        total_preemptions += record.preemption_count;
        maximum_preemptions =
            std::max(maximum_preemptions, record.preemption_count);
    }

    const auto &cache = output.aggregate.prefix_cache;
    expect(cache.successful_admissions >= output.requests.size() &&
               cache.query_blocks >= request_query_blocks &&
               cache.hit_blocks >= request_hit_blocks && cache.hit_blocks > 0 &&
               cache.evicted_blocks > 0,
           context + ": stress run missed reuse or eviction (admissions=" +
               std::to_string(cache.successful_admissions) +
               ", hits=" + std::to_string(cache.hit_blocks) +
               ", evictions=" + std::to_string(cache.evicted_blocks) +
               ", sessions=" + std::to_string(cache.evicted_sessions) + ")");
    expect(total_preemptions > 0 && maximum_preemptions >= 2,
           context + ": no request experienced repeated preemption");

    const ClusterType cache_owner =
        test_case.pdd ? ClusterType::kPrefill : ClusterType::kMonolithic;
    const std::uint64_t owner_targets =
        target_count(test_case.prefill_or_monolithic);
    expect(output.prefix_cache_targets.size() == owner_targets,
           context + ": cache-owner diagnostics are incomplete");
    std::uint64_t resident_blocks = 0;
    std::uint64_t resident_sessions = 0;
    for (const auto &target : output.prefix_cache_targets) {
        expect(target.cluster_type == cache_owner &&
                   target.capacity_blocks == test_case.cache_blocks &&
                   target.available_blocks == test_case.cache_blocks &&
                   target.active_blocks == 0 &&
                   target.resident_blocks <= target.capacity_blocks &&
                   target.evictable_blocks == target.resident_blocks,
               context + ": cache target did not quiesce consistently");
        resident_blocks += target.resident_blocks;
        resident_sessions += target.sessions_with_nonzero_frontier;
    }
    expect(resident_blocks > 0 && resident_sessions > 0 &&
               resident_sessions <= test_case.sessions,
           context + ": final resident cache diagnostics are invalid");

    const ClusterType pressure_cluster =
        !test_case.pdd ? ClusterType::kMonolithic
                       : (test_case.pressure_prefill ? ClusterType::kPrefill
                                                     : ClusterType::kDecode);
    const Topology &pressure_topology =
        test_case.pdd && !test_case.pressure_prefill
            ? test_case.decode
            : test_case.prefill_or_monolithic;
    expect_preemption_on_every_target(output, pressure_cluster,
                                      pressure_topology, context);
    const std::uint64_t pressure_capacity =
        test_case.pdd && !test_case.pressure_prefill ? test_case.decode_blocks
                                                     : test_case.cache_blocks;
    for (const auto &trace : output.scheduler_trace) {
        if (trace.cluster_type == pressure_cluster) {
            expect(trace.available_blocks_before <= pressure_capacity &&
                       trace.available_blocks_after <= pressure_capacity,
                   context + ": scheduler trace overcommitted KV capacity");
        }
    }

    if (test_case.pdd) {
        expect(output.kv_cache_transfers.size() == output.requests.size() &&
                   output.aggregate.kv_cache_transfer_count ==
                       output.requests.size(),
               context + ": PDD stress run lost a KV transfer");
    }
    if (test_case.moe) {
        const std::uint64_t expected_ep =
            std::max(test_case.prefill_or_monolithic.ep,
                     test_case.pdd ? test_case.decode.ep : 0);
        const bool saw_expected_ep =
            std::any_of(output.moe_routing.begin(), output.moe_routing.end(),
                        [expected_ep](const auto &routing) {
                            return routing.input_tokens > 0 &&
                                   routing.lane_times_ms.size() == expected_ep;
                        });
        expect(saw_expected_ep,
               context + ": MoE stress run did not exercise EP routing");
    }
}

void run_stress_case(const StressCase &test_case) {
    SimulationConfig config = load_base(test_case.moe, test_case.pdd);
    config.simulation_mode = SimulationMode::kOnline;
    config.prefix_cache.enabled = true;
    config.prefix_cache.key_mode = PrefixCachingKeyMode::kSession;
    config.cluster_scheduler.type = ClusterSchedulerType::kStickyRoundRobin;
    if (test_case.pdd) {
        configure_runtime(config.pdd().clusters.prefill,
                          test_case.prefill_or_monolithic,
                          test_case.cache_blocks, true);
        configure_runtime(config.pdd().clusters.decode, test_case.decode,
                          test_case.decode_blocks, false);
    } else {
        configure_runtime(config.cluster(), test_case.prefill_or_monolithic,
                          test_case.cache_blocks, true);
    }

    const std::vector<WorkloadRequest> workload = make_seeded_workload(
        test_case.sessions, test_case.rounds, test_case.seed);
    const std::vector<WorkloadRequest> materialized =
        materialize_workload_for_config(workload, config);
    expect_individually_admissible(materialized, test_case);
    const SimulationOutput output = run_simulation(config, workload);
    validate_output(output, materialized, test_case);
}

const std::vector<StressCase> kStressCases{
    {"dense-colocation-randomized",
     false,
     false,
     {2, 2, 4, 2, 1, 1},
     {1, 1, 1, 1, 1, 1},
     32,
     32,
     32,
     10,
     false,
     0xD301ULL},
    {"moe-colocation-randomized",
     true,
     false,
     {1, 4, 2, 2, 2, 4},
     {1, 1, 1, 1, 1, 1},
     32,
     32,
     24,
     10,
     false,
     0xA401ULL},
    {"dense-pdd-prefill1-decode8",
     false,
     true,
     {1, 2, 2, 1, 1, 1},
     {2, 2, 4, 4, 1, 1},
     48,
     20,
     64,
     8,
     true,
     0xD801ULL},
    {"moe-pdd-prefill8-decode1",
     true,
     true,
     {2, 2, 2, 4, 2, 4},
     {1, 4, 4, 1, 1, 4},
     32,
     32,
     32,
     8,
     false,
     0xA801ULL},
    {"dense-colocation-1000-request-soak",
     false,
     false,
     {1, 4, 4, 1, 1, 1},
     {1, 1, 1, 1, 1, 1},
     64,
     64,
     50,
     20,
     false,
     0x5010ULL},
};

} // namespace

int main() {
    int failures = 0;
    for (const StressCase &test_case : kStressCases) {
        failures += frontier::test::run(test_case.name, [&] {
            try {
                run_stress_case(test_case);
            } catch (const std::exception &error) {
                throw std::runtime_error(std::string{test_case.name} + ": " +
                                         error.what());
            }
        });
    }
    return failures == 0 ? 0 : 1;
}
