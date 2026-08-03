#include "frontier/config/config.h"
#include "frontier/metrics/output_contract.h"
#include "frontier/request_generator/workload.h"
#include "frontier/simulator/simulator.h"
#include "tests/test_support.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <map>
#include <set>
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
using frontier::config::ModelKind;
using frontier::config::MoeRoutingDistribution;
using frontier::config::MoeRoutingMode;
using frontier::config::ParallelismConfig;
using frontier::config::parse_simulation_config_json;
using frontier::config::PrefixCachingKeyMode;
using frontier::config::SimulationConfig;
using frontier::config::SimulationMode;
using frontier::metrics::RequestMetricsRecord;
using frontier::metrics::SimulationOutput;
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

struct ColocationCase {
    const char *name;
    bool moe;
    Topology topology;
};

struct PddCase {
    const char *name;
    bool moe;
    Topology prefill;
    Topology decode;
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
                       std::uint64_t blocks, std::uint64_t batch_size,
                       std::uint64_t token_budget) {
    runtime.parallelism = parallelism(topology);
    runtime.scheduler.block_size = 4;
    runtime.scheduler.num_blocks = blocks;
    runtime.scheduler.batch_size_cap = batch_size;
    runtime.scheduler.max_tokens_in_batch = token_budget;
    runtime.scheduler.enable_preemption = true;
    runtime.scheduler.enable_chunked_prefill = true;
    runtime.scheduler.long_prefill_token_threshold = 0;
    runtime.scheduler.watermark_blocks_fraction = 0.0;
    runtime.scheduler.num_preallocate_tokens = 0;
    runtime.execution_model.type = ExecutionModelType::kFixed;
    runtime.execution_model.fixed.batch_latency_ms = 0.01;
    runtime.execution_model.fixed.stage_latencies_ms.assign(
        static_cast<std::size_t>(topology.pp), 0.01);
    runtime.moe_routing.mode = MoeRoutingMode::kSimulation;
    runtime.moe_routing.distribution = MoeRoutingDistribution::kBalanced;
    runtime.moe_routing.seed = 42;
}

void enable_prefix_cache(SimulationConfig &config, SimulationMode mode) {
    config.simulation_mode = mode;
    config.prefix_cache.enabled = true;
    config.prefix_cache.key_mode = PrefixCachingKeyMode::kSession;
    config.cluster_scheduler.type = ClusterSchedulerType::kStickyRoundRobin;
}

WorkloadRequest make_request(std::size_t id, double delay,
                             std::uint64_t prefill, std::uint64_t decode,
                             std::uint64_t session, std::uint64_t turn) {
    WorkloadRequest result{};
    result.request_id = RequestId{id};
    if (turn == 0) {
        result.session_start_at = SimTime::from_seconds(delay);
    } else {
        result.think_time = SimTime::from_seconds(delay);
    }
    result.num_prefill_tokens = prefill;
    result.num_decode_tokens = decode;
    result.session_id = SessionId{session};
    result.session_turn_index = turn;
    return result;
}

std::vector<WorkloadRequest> make_reuse_workload(std::uint64_t sessions) {
    std::vector<WorkloadRequest> workload;
    workload.reserve(static_cast<std::size_t>(sessions * 2));
    for (std::uint64_t session = 0; session < sessions; ++session) {
        workload.push_back(
            make_request(workload.size(), 0.0, 8, 4, 1'000 + session, 0));
    }
    for (std::uint64_t session = 0; session < sessions; ++session) {
        workload.push_back(
            make_request(workload.size(), 1.0, 4, 2, 1'000 + session, 1));
    }
    return workload;
}

std::vector<WorkloadRequest>
make_colocation_pressure_workload(std::uint64_t targets) {
    std::vector<WorkloadRequest> workload;
    workload.reserve(static_cast<std::size_t>(targets * 3));
    for (std::uint64_t target = 0; target < targets; ++target) {
        workload.push_back(
            make_request(workload.size(), 0.0, 5, 3, 10'000 + target, 0));
    }
    for (std::uint64_t target = 0; target < targets; ++target) {
        workload.push_back(
            make_request(workload.size(), 0.0, 5, 1, 20'000 + target, 0));
    }
    for (std::uint64_t target = 0; target < targets; ++target) {
        workload.push_back(
            make_request(workload.size(), 1.0, 1, 1, 10'000 + target, 1));
    }
    return workload;
}

std::vector<WorkloadRequest>
make_pdd_pressure_workload(std::uint64_t sessions) {
    std::vector<WorkloadRequest> workload;
    workload.reserve(static_cast<std::size_t>(sessions * 2));
    for (std::uint64_t session = 0; session < sessions; ++session) {
        workload.push_back(
            make_request(workload.size(), 0.0, 5, 5, 30'000 + session, 0));
    }
    for (std::uint64_t session = 0; session < sessions; ++session) {
        workload.push_back(
            make_request(workload.size(), 1.0, 1, 5, 30'000 + session, 1));
    }
    return workload;
}

const RequestMetricsRecord &request(const SimulationOutput &output,
                                    std::uint64_t id) {
    const auto position =
        std::find_if(output.requests.begin(), output.requests.end(),
                     [id](const RequestMetricsRecord &record) {
                         return record.request_id == RequestId{id};
                     });
    if (position == output.requests.end()) {
        throw std::runtime_error("matrix output is missing request " +
                                 std::to_string(id));
    }
    return *position;
}

SimulationOutput run_named(const SimulationConfig &config,
                           const std::vector<WorkloadRequest> &workload,
                           const std::string &context) {
    try {
        return run_simulation(config, workload);
    } catch (const std::exception &error) {
        throw std::runtime_error(context + ": " + error.what());
    }
}

std::set<Target> request_targets(const SimulationOutput &output,
                                 ClusterType cluster_type) {
    std::set<Target> result;
    for (const auto &record : output.requests) {
        if (cluster_type == ClusterType::kMonolithic) {
            result.emplace(record.replica_id.value(), record.dp_id.value());
        } else if (cluster_type == ClusterType::kPrefill) {
            result.emplace(record.prefill_replica_id.value(),
                           record.prefill_dp_id.value());
        } else {
            result.emplace(record.decode_replica_id.value(),
                           record.decode_dp_id.value());
        }
    }
    return result;
}

void expect_topology_exercised(const SimulationOutput &output,
                               ClusterType cluster_type,
                               const ClusterRuntimeConfig &runtime,
                               const Topology &topology,
                               const std::string &context) {
    const ParallelismConfig expected = parallelism(topology);
    bool saw_batch = false;
    std::set<std::uint64_t> stages;
    for (const auto &batch : output.batches) {
        if (batch.cluster_type != cluster_type) {
            continue;
        }
        saw_batch = true;
        expect(batch.parallelism == expected &&
                   batch.num_pipeline_stages == topology.pp,
               context + ": batch did not retain the configured topology");
    }
    for (const auto &stage : output.batch_stages) {
        if (stage.cluster_type != cluster_type) {
            continue;
        }
        stages.insert(stage.stage_id.value());
        expect(stage.parallelism == expected,
               context + ": stage did not retain the configured topology");
    }
    expect(saw_batch && stages.size() == topology.pp,
           context + ": not every configured PP stage executed");
    for (std::uint64_t stage = 0; stage < topology.pp; ++stage) {
        expect(stages.find(stage) != stages.end(),
               context + ": missing PP stage " + std::to_string(stage));
    }

    if (runtime.model.kind == ModelKind::kMoe) {
        const bool saw_real_routing = std::any_of(
            output.moe_routing.begin(), output.moe_routing.end(),
            [&](const auto &routing) {
                return routing.cluster_type == cluster_type &&
                       routing.input_tokens > 0 &&
                       routing.global_expert_tokens.size() ==
                           runtime.model.total_expert_num &&
                       routing.lane_expert_tokens.size() == topology.ep &&
                       routing.lane_times_ms.size() == topology.ep &&
                       routing.routed_tokens ==
                           routing.input_tokens * runtime.model.router_topk;
            });
        expect(saw_real_routing,
               context + ": MoE EP routing was not exercised");
    }
}

void expect_cache_targets(const SimulationOutput &output,
                          ClusterType owner_cluster, std::uint64_t targets,
                          std::uint64_t capacity, std::uint64_t sessions,
                          std::uint64_t resident_blocks,
                          const std::string &context) {
    expect(output.prefix_cache_targets.size() == targets,
           context + ": wrong number of cache-owner targets");
    std::uint64_t observed_sessions = 0;
    std::uint64_t observed_resident = 0;
    for (const auto &target : output.prefix_cache_targets) {
        expect(
            target.cluster_type == owner_cluster &&
                target.capacity_blocks == capacity &&
                target.available_blocks == capacity &&
                target.active_blocks == 0 &&
                target.evictable_blocks == target.resident_blocks,
            context +
                ": cache target did not quiesce with all blocks allocatable");
        observed_sessions += target.sessions_with_nonzero_frontier;
        observed_resident += target.resident_blocks;
    }
    expect(observed_sessions == sessions &&
               observed_resident == resident_blocks,
           context + ": final session frontier/resident-block totals mismatch");
}

void expect_preemption_trace_coverage(const SimulationOutput &output,
                                      ClusterType cluster_type,
                                      const std::set<Target> &targets,
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
    expect(observed == targets,
           context + ": preemption did not occur on every target");
}

void validate_reuse_output(const SimulationOutput &output,
                           std::uint64_t sessions, bool pdd,
                           std::uint64_t owner_targets,
                           const std::string &context) {
    const std::uint64_t expected_cached = pdd ? 8 : 12;
    const std::uint64_t expected_hits_per_followup = pdd ? 2 : 3;
    expect(output.requests.size() == sessions * 2,
           context + ": reuse workload did not drain");
    for (std::uint64_t session = 0; session < sessions; ++session) {
        const auto &first = request(output, session);
        const auto &followup = request(output, sessions + session);
        expect(first.cached_prefill_tokens == 0 &&
                   first.prefix_cache_query_blocks == 2 &&
                   first.prefix_cache_hit_blocks == 0,
               context + ": cold session turn unexpectedly hit");
        expect(followup.num_prefill_tokens == 16 &&
                   followup.cached_prefill_tokens == expected_cached &&
                   followup.prefix_cache_query_blocks == 4 &&
                   followup.prefix_cache_hit_blocks ==
                       expected_hits_per_followup,
               context + ": warm session turn did not reuse the exact prefix");
        if (pdd) {
            expect(first.prefill_replica_id == followup.prefill_replica_id &&
                       first.prefill_dp_id == followup.prefill_dp_id,
                   context + ": PDD session changed PREFILL cache owner");
        } else {
            expect(first.replica_id == followup.replica_id &&
                       first.dp_id == followup.dp_id,
                   context + ": co-location session changed cache owner");
        }
    }
    expect(output.aggregate.prefix_cache.successful_admissions ==
                   sessions * 2 &&
               output.aggregate.prefix_cache.query_blocks == sessions * 6 &&
               output.aggregate.prefix_cache.hit_blocks ==
                   sessions * expected_hits_per_followup &&
               output.aggregate.prefix_cache.evicted_blocks == 0 &&
               output.aggregate.prefix_cache.evicted_sessions == 0,
           context + ": aggregate reuse metrics mismatch");
    expect_cache_targets(output,
                         pdd ? ClusterType::kPrefill : ClusterType::kMonolithic,
                         owner_targets, 16, sessions, sessions * 4, context);
    if (pdd) {
        expect(output.kv_cache_transfers.size() == sessions * 2,
               context + ": PDD did not transfer every request");
    }
}

void validate_colocation_pressure(const SimulationOutput &output,
                                  const Topology &topology,
                                  const std::string &context) {
    const std::uint64_t targets = target_count(topology);
    expect(output.requests.size() == targets * 3,
           context + ": pressure workload did not drain");
    std::uint64_t recorded_request_hits = 0;
    for (const auto &record : output.requests) {
        recorded_request_hits += record.prefix_cache_hit_blocks;
    }
    for (std::uint64_t target = 0; target < targets; ++target) {
        const auto &producer = request(output, target);
        const auto &followup = request(output, targets * 2 + target);
        expect(
            producer.preemption_count > 0 &&
                producer.cached_prefill_tokens == 0,
            context +
                ": the two-block producer was not preempted after publishing");
        expect(
            followup.num_prefill_tokens == 9 &&
                followup.cached_prefill_tokens == 8 &&
                followup.prefix_cache_query_blocks == 2 &&
                followup.prefix_cache_hit_blocks == 2 &&
                followup.replica_id == producer.replica_id &&
                followup.dp_id == producer.dp_id,
            context +
                ": post-preemption follow-up did not recover resident cache");
    }
    expect(output.aggregate.prefix_cache.successful_admissions >
                   output.requests.size() &&
               output.aggregate.prefix_cache.hit_blocks >
                   recorded_request_hits &&
               output.aggregate.prefix_cache.evicted_blocks >= targets,
           context +
               ": system metrics did not expose cached re-entry and eviction");
    expect_preemption_trace_coverage(output, ClusterType::kMonolithic,
                                     expected_targets(topology), context);
    expect_cache_targets(output, ClusterType::kMonolithic, targets, 3, targets,
                         targets * 2, context);
}

void validate_pdd_pressure(const SimulationOutput &output,
                           std::uint64_t sessions,
                           const Topology &prefill_topology,
                           const Topology &decode_topology,
                           const std::string &context) {
    expect(output.requests.size() == sessions * 2 &&
               output.kv_cache_transfers.size() == sessions * 2,
           context + ": PDD pressure workload did not transfer and drain");
    std::map<Target, std::uint64_t> preemptions_by_decode_target;
    for (const auto &record : output.requests) {
        preemptions_by_decode_target[Target{record.decode_replica_id.value(),
                                            record.decode_dp_id.value()}] +=
            record.preemption_count;
    }
    for (const Target target : expected_targets(decode_topology)) {
        expect(preemptions_by_decode_target[target] > 0,
               context + ": a DECODE target did not preempt under pressure");
    }
    for (std::uint64_t session = 0; session < sessions; ++session) {
        const auto &first = request(output, session);
        const auto &followup = request(output, sessions + session);
        expect(first.cached_prefill_tokens == 0 &&
                   followup.num_prefill_tokens == 11 &&
                   followup.cached_prefill_tokens == 4 &&
                   followup.prefix_cache_query_blocks == 2 &&
                   followup.prefix_cache_hit_blocks == 1 &&
                   first.prefill_replica_id == followup.prefill_replica_id &&
                   first.prefill_dp_id == followup.prefill_dp_id,
               context + ": PDD follow-up lost its PREFILL cache across DECODE "
                         "preemption");
    }
    expect(output.aggregate.prefix_cache.successful_admissions ==
                   sessions * 2 &&
               output.aggregate.prefix_cache.query_blocks == sessions * 3 &&
               output.aggregate.prefix_cache.hit_blocks == sessions &&
               output.aggregate.prefix_cache.evicted_blocks == 0,
           context + ": PDD prefix metrics changed under DECODE pressure");
    expect_preemption_trace_coverage(output, ClusterType::kDecode,
                                     expected_targets(decode_topology),
                                     context);
    expect_cache_targets(output, ClusterType::kPrefill,
                         target_count(prefill_topology), 64, sessions,
                         sessions * 2, context);
}

void run_colocation_case(const ColocationCase &test_case) {
    const std::uint64_t targets = target_count(test_case.topology);
    const std::uint64_t sessions = std::max<std::uint64_t>(2, targets);

    for (const SimulationMode mode :
         {SimulationMode::kOnline, SimulationMode::kOffline}) {
        SimulationConfig config = load_base(test_case.moe, false);
        enable_prefix_cache(config, mode);
        configure_runtime(config.cluster(), test_case.topology, 16,
                          mode == SimulationMode::kOffline ? 1 : 8, 32);
        const std::string context =
            std::string{test_case.name} + (mode == SimulationMode::kOnline
                                               ? "/online-reuse"
                                               : "/offline-reuse");
        const SimulationOutput output =
            run_named(config, make_reuse_workload(sessions), context);
        validate_reuse_output(output, sessions, false, targets, context);
        expect(request_targets(output, ClusterType::kMonolithic) ==
                   expected_targets(test_case.topology),
               context + ": sticky routing did not cover every target");
        expect_topology_exercised(output, ClusterType::kMonolithic,
                                  config.cluster(), test_case.topology,
                                  context);
    }

    SimulationConfig pressure = load_base(test_case.moe, false);
    enable_prefix_cache(pressure, SimulationMode::kOnline);
    configure_runtime(pressure.cluster(), test_case.topology, 3, 2, 8);
    const std::string context = std::string{test_case.name} + "/preemption";
    const SimulationOutput output = run_named(
        pressure, make_colocation_pressure_workload(targets), context);
    validate_colocation_pressure(output, test_case.topology, context);
    expect_topology_exercised(output, ClusterType::kMonolithic,
                              pressure.cluster(), test_case.topology, context);
}

void run_pdd_case(const PddCase &test_case) {
    const std::uint64_t prefill_targets = target_count(test_case.prefill);
    const std::uint64_t decode_targets = target_count(test_case.decode);
    const std::uint64_t reuse_sessions =
        std::max<std::uint64_t>(2, std::max(prefill_targets, decode_targets));

    for (const SimulationMode mode :
         {SimulationMode::kOnline, SimulationMode::kOffline}) {
        SimulationConfig config = load_base(test_case.moe, true);
        enable_prefix_cache(config, mode);
        configure_runtime(config.pdd().clusters.prefill, test_case.prefill, 16,
                          mode == SimulationMode::kOffline ? 1 : 8, 32);
        configure_runtime(config.pdd().clusters.decode, test_case.decode, 16,
                          mode == SimulationMode::kOffline ? 1 : 8, 32);
        config.pdd().clusters.decode.scheduler.enable_chunked_prefill = false;
        const std::string context =
            std::string{test_case.name} + (mode == SimulationMode::kOnline
                                               ? "/online-reuse"
                                               : "/offline-reuse");
        const SimulationOutput output =
            run_named(config, make_reuse_workload(reuse_sessions), context);
        validate_reuse_output(output, reuse_sessions, true, prefill_targets,
                              context);
        expect(request_targets(output, ClusterType::kPrefill) ==
                       expected_targets(test_case.prefill) &&
                   request_targets(output, ClusterType::kDecode) ==
                       expected_targets(test_case.decode),
               context + ": PDD sticky routing did not cover both topologies");
        expect_topology_exercised(output, ClusterType::kPrefill,
                                  config.pdd().clusters.prefill,
                                  test_case.prefill, context + "/prefill");
        expect_topology_exercised(output, ClusterType::kDecode,
                                  config.pdd().clusters.decode,
                                  test_case.decode, context + "/decode");
    }

    const std::uint64_t pressure_sessions =
        2 * std::max(prefill_targets, decode_targets);
    SimulationConfig pressure = load_base(test_case.moe, true);
    enable_prefix_cache(pressure, SimulationMode::kOnline);
    configure_runtime(pressure.pdd().clusters.prefill, test_case.prefill, 64, 8,
                      32);
    configure_runtime(pressure.pdd().clusters.decode, test_case.decode, 4, 2,
                      8);
    pressure.pdd().clusters.decode.scheduler.enable_chunked_prefill = false;
    const std::string context = std::string{test_case.name} + "/preemption";
    const SimulationOutput output = run_named(
        pressure, make_pdd_pressure_workload(pressure_sessions), context);
    validate_pdd_pressure(output, pressure_sessions, test_case.prefill,
                          test_case.decode, context);
    expect_topology_exercised(output, ClusterType::kPrefill,
                              pressure.pdd().clusters.prefill,
                              test_case.prefill, context + "/prefill");
    expect_topology_exercised(output, ClusterType::kDecode,
                              pressure.pdd().clusters.decode, test_case.decode,
                              context + "/decode");
}

const std::vector<ColocationCase> kColocationCases{
    {"dense-colo-r1-tp1-dp1-pp1", false, {1, 1, 1, 1, 1, 1}},
    {"dense-colo-r1-tp2-dp2-pp2", false, {1, 2, 2, 2, 1, 1}},
    {"dense-colo-r2-tp4-dp1-pp4", false, {2, 4, 4, 1, 1, 1}},
    {"dense-colo-r1-tp1-dp4-pp2", false, {1, 1, 2, 4, 1, 1}},
    {"moe-colo-r1-tp2-dp1-pp2-mtp2-ep1", true, {1, 2, 2, 1, 2, 1}},
    {"moe-colo-r1-tp1-dp4-pp2-mtp1-ep4", true, {1, 1, 2, 4, 1, 4}},
    {"moe-colo-r2-tp2-dp2-pp4-mtp1-ep4", true, {2, 2, 4, 2, 1, 4}},
    {"moe-colo-r1-tp4-dp2-pp1-mtp2-ep4", true, {1, 4, 1, 2, 2, 4}},
};

const std::vector<PddCase> kPddCases{
    {"dense-pdd-p-tp1-dp1-pp1-d-tp2-dp2-pp2",
     false,
     {1, 1, 1, 1, 1, 1},
     {1, 2, 2, 2, 1, 1}},
    {"dense-pdd-p-tp2-dp2-pp2-d-r2-tp4-dp1-pp4",
     false,
     {1, 2, 2, 2, 1, 1},
     {2, 4, 4, 1, 1, 1}},
    {"dense-pdd-p-r2-tp4-dp1-pp4-d-tp1-dp4-pp2",
     false,
     {2, 4, 4, 1, 1, 1},
     {1, 1, 2, 4, 1, 1}},
    {"moe-pdd-p-tp2-dp2-pp2-ep4-d-tp4-dp1-pp1-ep4",
     true,
     {1, 2, 2, 2, 1, 4},
     {1, 4, 1, 1, 1, 4}},
    {"moe-pdd-p-r2-tp1-dp4-pp4-ep4-d-tp2-dp2-pp2-mtp2-ep2",
     true,
     {2, 1, 4, 4, 1, 4},
     {1, 2, 2, 2, 2, 2}},
    {"moe-pdd-p-tp4-dp2-pp1-mtp2-ep4-d-r2-tp1-dp4-pp4-ep4",
     true,
     {1, 4, 1, 2, 2, 4},
     {2, 1, 4, 4, 1, 4}},
};

} // namespace

int main() {
    int failures = 0;
    for (const ColocationCase &test_case : kColocationCases) {
        failures += frontier::test::run(
            test_case.name, [&] { run_colocation_case(test_case); });
    }
    for (const PddCase &test_case : kPddCases) {
        failures += frontier::test::run(test_case.name,
                                        [&] { run_pdd_case(test_case); });
    }
    return failures == 0 ? 0 : 1;
}
