#include "frontier/config/config.h"
#include "frontier/core/event.h"
#include "frontier/metrics/output_contract.h"
#include "frontier/request_generator/workload.h"
#include "frontier/simulator/simulator.h"
#include "tests/test_support.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <map>
#include <numeric>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#ifndef FRONTIER_EXAMPLE_DIR
#error "FRONTIER_EXAMPLE_DIR must be defined for pipeline-exclusive tests"
#endif

namespace {

using frontier::EventType;
using frontier::config::ConfigError;
using frontier::config::ExecutionModelType;
using frontier::config::parse_simulation_config_json;
using frontier::config::serialize_simulation_config_json;
using frontier::config::SimulationConfig;
using frontier::config::SimulationMode;
using frontier::metrics::SimulationOutput;
using frontier::request_generator::parse_workload_csv;
using frontier::simulator::run_simulation;
using frontier::test::expect;
using frontier::test::expect_throws;
using frontier::test::read_text_file;
using Json = nlohmann::json;

const std::filesystem::path kExampleRoot{FRONTIER_EXAMPLE_DIR};

enum class ExecutionFlavor { kFixed, kAnalytical };

struct Topology {
    std::uint64_t tensor_parallel_size;
    std::uint64_t pipeline_parallel_size;
    std::uint64_t decode_context_parallel_size;
};

void replace_all(std::string &text, std::string_view from,
                 std::string_view to) {
    std::size_t position = 0;
    bool replaced = false;
    while ((position = text.find(from, position)) != std::string::npos) {
        text.replace(position, from.size(), to);
        position += to.size();
        replaced = true;
    }
    expect(replaced, "K3 example must contain the base model name");
}

SimulationConfig make_k3_config(ExecutionFlavor flavor, Topology topology,
                                std::string_view pipeline_event_mode,
                                std::uint64_t batch_size_cap = 2) {
    // Parse the checked-in example first so model loading and all ordinary
    // scheduler defaults remain covered by the public configuration path.
    std::string config_text = read_text_file(kExampleRoot / "configs" /
                                             "00_hello_colocation_fixed.json");
    replace_all(config_text, "meta-llama/Llama-2-7b-hf", "moonshotai/Kimi-K3");
    SimulationConfig config = parse_simulation_config_json(config_text);

    auto &cluster = config.cluster();
    auto &parallelism = cluster.parallelism;
    parallelism.num_replicas = 1;
    parallelism.tensor_parallel_size = topology.tensor_parallel_size;
    parallelism.pipeline_parallel_size = topology.pipeline_parallel_size;
    parallelism.data_parallel_size = 1;
    parallelism.moe_tensor_parallel_size = topology.tensor_parallel_size;
    parallelism.moe_expert_parallel_size = 1;
    parallelism.decode_context_parallel_size =
        topology.decode_context_parallel_size;
    parallelism.pipeline_exclusive = topology.pipeline_parallel_size > 1;

    cluster.scheduler.batch_size_cap = batch_size_cap;
    cluster.scheduler.max_tokens_in_batch = 256;
    cluster.scheduler.enable_chunked_prefill = true;
    cluster.scheduler.pipeline_event_mode = std::string{pipeline_event_mode};
    cluster.scheduler.num_blocks = 50'000;
    cluster.scheduler.block_size = 16;

    // K3's model footprint is large, and PP24 intentionally creates tiny
    // uneven stages.  A positive, explicit HBM contract keeps every matrix
    // point in the same safe admission regime without relying on a host GPU.
    cluster.gpu_memory.capacity_bytes_per_gpu = 1'000'000'000'000'000ULL;
    cluster.gpu_memory.auto_calculate_num_blocks = false;
    cluster.gpu_memory.runtime_reserve_fraction = 0.1;

    if (flavor == ExecutionFlavor::kFixed) {
        cluster.execution_model.type = ExecutionModelType::kFixed;
        cluster.execution_model.fixed.stage_latencies_ms.resize(
            static_cast<std::size_t>(topology.pipeline_parallel_size));
        for (std::uint64_t stage = 0; stage < topology.pipeline_parallel_size;
             ++stage) {
            // Deliberately vary stage costs so serialization checks cannot
            // pass by observing one constant interval accidentally.
            cluster.execution_model.fixed.stage_latencies_ms.at(stage) =
                0.20 + 0.01 * static_cast<double>(stage % 5);
        }
    } else {
        cluster.execution_model.type = ExecutionModelType::kAnalytical;
        cluster.execution_model.analytical = {};
        cluster.execution_model.analytical.tensor_parallel_size =
            topology.tensor_parallel_size;
        cluster.execution_model.analytical.moe_layer_event_mode =
            "stage_group_scaled";
        cluster.execution_model.analytical.precision = "bf16";
    }

    // Mutation happens after parsing, so rematerialize the PP stage/rank
    // profiles and validate the positive HBM/explicit block contract.
    frontier::config::resolve_gpu_memory_config(cluster,
                                                cluster.scheduler.num_blocks);
    config.simulation_mode = SimulationMode::kOnline;
    config.run_id = "pipeline-exclusive-k3-matrix";
    return config;
}

std::vector<frontier::request_generator::WorkloadRequest>
workload(std::string_view csv) {
    return parse_workload_csv(csv);
}

std::size_t count_events(const SimulationOutput &output, EventType type) {
    return static_cast<std::size_t>(std::count_if(
        output.event_trace.begin(), output.event_trace.end(),
        [type](const auto &event) { return event.type() == type; }));
}

void expect_no_moe_sync_events(const SimulationOutput &output) {
    for (const EventType type :
         {EventType::kPrefillSync, EventType::kPrefillSyncCollective,
          EventType::kDecodeSync, EventType::kDecodeSyncCollective}) {
        expect(
            count_events(output, type) == 0,
            "pipeline-exclusive K3 must not emit MoE synchronization events");
    }
}

void expect_positive_hbm_contract(const SimulationOutput &output,
                                  const Topology &topology) {
    expect(output.pipeline_memory_diagnostics.size() == 1,
           "co-location run must expose one pipeline memory diagnostic");
    const auto &diagnostic = output.pipeline_memory_diagnostics.front();
    expect(diagnostic.capacity_bytes_per_gpu > 0 &&
               diagnostic.ordinary_kv_capacity_blocks > 0 &&
               diagnostic.configured_num_blocks <=
                   diagnostic.ordinary_kv_capacity_blocks,
           "pipeline-exclusive K3 requires a positive HBM/KV contract");
    expect(diagnostic.stages.size() ==
                   static_cast<std::size_t>(topology.pipeline_parallel_size) &&
               diagnostic.timing_group_count > 0 &&
               diagnostic.memory_group_count > 0,
           "K3 memory diagnostics must cover every PP stage and groups");
    for (std::size_t index = 0; index < diagnostic.stages.size(); ++index) {
        const auto &stage = diagnostic.stages.at(index);
        expect(stage.stage_id.value() ==
                       static_cast<decltype(stage.stage_id.value())>(index) &&
                   stage.layer_count > 0 &&
                   stage.layer_end > stage.layer_begin && stage.free_bytes > 0,
               "every uneven K3 PP stage must retain positive free HBM");
        expect(stage.layer_count == stage.layer_end - stage.layer_begin &&
                   stage.timing_group_id.has_value() &&
                   stage.memory_group_id.has_value(),
               "K3 stage diagnostics must expose complete signatures");
        // PP24 can contain a KDA-only stage.  Such a stage legitimately has
        // zero ordinary KV bytes while still carrying a positive HBM budget
        // and KDA snapshot footprint.
        for (const std::uint64_t bytes : stage.kda_snapshot_bytes_by_rank) {
            expect(bytes <= stage.free_bytes,
                   "one K3 KDA snapshot must fit each stage/rank HBM budget");
        }
    }
}

void expect_all_stages_visited(const SimulationOutput &output,
                               const Topology &topology) {
    expect(!output.requests.empty() && !output.batches.empty(),
           "matrix workload must complete at least one request and batch");
    expect(output.batch_stages.size() ==
               output.batches.size() * topology.pipeline_parallel_size,
           "each K3 batch must visit every physical PP stage");
    std::set<std::uint64_t> stages;
    for (const auto &record : output.batch_stages) {
        stages.insert(record.stage_id.value());
        expect(record.started_at.valid() && record.completed_at.valid() &&
                   record.arrived_at <= record.started_at &&
                   record.started_at <= record.completed_at,
               "K3 stage timestamps must be finite and causal");
    }
    std::set<std::uint64_t> expected;
    for (std::uint64_t stage = 0; stage < topology.pipeline_parallel_size;
         ++stage) {
        expected.insert(stage);
    }
    expect(stages == expected, "all K3 pipeline stages must be visited");
}

void expect_stage_serialization(const SimulationOutput &output) {
    using Key = std::tuple<std::uint64_t, std::uint64_t, std::uint64_t>;
    std::map<Key,
             std::vector<const frontier::metrics::BatchStageMetricsRecord *>>
        by_stage;
    for (const auto &record : output.batch_stages) {
        by_stage[Key{record.replica_id.value(), record.dp_id.value(),
                     record.stage_id.value()}]
            .push_back(&record);
    }
    for (auto &[key, records] : by_stage) {
        static_cast<void>(key);
        std::sort(records.begin(), records.end(),
                  [](const auto *left, const auto *right) {
                      return left->started_at < right->started_at;
                  });
        for (std::size_t index = 1; index < records.size(); ++index) {
            expect(records.at(index - 1)->completed_at <=
                       records.at(index)->started_at,
                   "pipeline-exclusive stages must serialize batches locally");
        }
    }
}

Json sorted_stage_json(const SimulationOutput &output) {
    const Json root = Json::parse(
        frontier::metrics::serialize_simulation_output_json(output));
    Json stages = root.at("batch_stages");
    std::sort(stages.begin(), stages.end(),
              [](const Json &left, const Json &right) {
                  if (left.at("batch_id") != right.at("batch_id")) {
                      return left.at("batch_id") < right.at("batch_id");
                  }
                  return left.at("stage_id") < right.at("stage_id");
              });
    return stages;
}

void expect_exact_collapsed_parity(const SimulationOutput &exact,
                                   const SimulationOutput &collapsed) {
    const Json exact_json =
        Json::parse(frontier::metrics::serialize_simulation_output_json(exact));
    const Json collapsed_json = Json::parse(
        frontier::metrics::serialize_simulation_output_json(collapsed));
    expect(exact_json.at("requests") == collapsed_json.at("requests") &&
               exact_json.at("batches") == collapsed_json.at("batches"),
           "collapsed K3 PP must preserve requests and batches exactly");
    expect(sorted_stage_json(exact) == sorted_stage_json(collapsed),
           "collapsed K3 PP must preserve every stage timestamp and duration");
    expect(count_events(collapsed, EventType::kBatchPipelineEnd) ==
                   collapsed.batches.size() &&
               count_events(collapsed, EventType::kBatchStageEnd) == 0 &&
               collapsed.aggregate.event_count < exact.aggregate.event_count &&
               count_events(exact, EventType::kBatchStageEnd) > 0,
           "collapsed K3 PP must reduce intermediate stage events");
}

const std::string kPrefillHeavy =
    "session_start_at,think_time,num_prefill_tokens,num_decode_tokens\n"
    "0,0,96,1\n";
const std::string kDecodeHeavy =
    "session_start_at,think_time,num_prefill_tokens,num_decode_tokens\n"
    "0,0,2,16\n";
const std::string kConcurrent =
    "session_start_at,think_time,num_prefill_tokens,num_decode_tokens\n"
    "0,0,24,4\n"
    "0,0,16,6\n"
    "0,0,8,8\n";
const std::string kStaggered =
    "session_start_at,think_time,num_prefill_tokens,num_decode_tokens\n"
    "0,0,32,3\n"
    "0.0001,0,12,5\n"
    "0.0002,0,8,4\n";
// Enough staggered arrivals, and enough spread in prefill length, that the
// pipeline holds three or more batches with visibly different stage durations
// at the same time.  That combination is what lets a stage's reservation list
// develop an interior hole, so it is the workload the collapsed calendar has
// to be checked against; the smaller fixtures above never produce one.
const std::string kPipelined =
    "session_start_at,think_time,num_prefill_tokens,num_decode_tokens\n"
    "0,0,32,8\n"
    "0.002,0,64,16\n"
    "0.003,0,48,12\n"
    "0.006,0,96,8\n"
    "0.008,0,40,20\n"
    "0.011,0,80,12\n"
    "0.014,0,24,8\n"
    "0.018,0,128,16\n";

void test_k3_pipeline_exclusive_topology_matrix() {
    // Keep this a release smoke matrix rather than a Cartesian explosion.
    // These five pairwise points cover TP1/2/4/8, PP2/3/4/8/24, an uneven
    // PP3/PP24 partition, and DCP1 versus DCP==TP.
    const std::vector<Topology> topologies{
        {1, 2, 1}, {2, 3, 1}, {4, 4, 1}, {8, 8, 1}, {8, 24, 8}};
    for (const Topology topology : topologies) {
        const auto config =
            make_k3_config(ExecutionFlavor::kAnalytical, topology, "collapsed");
        const SimulationOutput output =
            run_simulation(config, workload(kPrefillHeavy));
        expect_all_stages_visited(output, topology);
        expect_no_moe_sync_events(output);
        expect_positive_hbm_contract(output, topology);
        expect(std::all_of(output.batch_stages.begin(),
                           output.batch_stages.end(),
                           [](const auto &stage) {
                               return stage.model_kind ==
                                          frontier::config::ModelKind::kMoe &&
                                      stage.execution_time.total_ms() > 0.0;
                           }),
               "analytical K3 stage_group_scaled timings must be positive");
    }
}

void test_k3_fixed_and_analytical_workload_matrix() {
    const std::vector<std::tuple<std::string_view, std::string_view,
                                 ExecutionFlavor, Topology>>
        cases{
            {"prefill-heavy",
             kPrefillHeavy,
             ExecutionFlavor::kFixed,
             {2, 3, 1}},
            {"decode-heavy",
             kDecodeHeavy,
             ExecutionFlavor::kAnalytical,
             {4, 4, 1}},
            {"concurrent-batches",
             kConcurrent,
             ExecutionFlavor::kFixed,
             {8, 8, 1}},
            {"staggered-arrivals",
             kStaggered,
             ExecutionFlavor::kAnalytical,
             {8, 24, 8}},
        };
    for (const auto &[name, csv, flavor, topology] : cases) {
        static_cast<void>(name);
        const auto config = make_k3_config(flavor, topology, "exact",
                                           /*batch_size_cap=*/1);
        const SimulationOutput output = run_simulation(config, workload(csv));
        expect_all_stages_visited(output, topology);
        expect_stage_serialization(output);
        expect_no_moe_sync_events(output);
        expect_positive_hbm_contract(output, topology);
    }
}

void test_k3_exact_collapsed_parity_and_event_reduction() {
    // PP3/PP8/PP24 cover exact uneven partitions and the DCP-sharded endpoint.
    // Analytical stage_group_scaled is the path that benefits from collapsed
    // event reduction; fixed timing is covered by the exact workload matrix.
    //
    // The batch size cap is varied deliberately.  A cap of one keeps a single
    // batch in the pipeline, where the collapsed calendar holds at most one
    // reservation per stage and can never disagree with the exact chain.  The
    // larger caps put three or more batches in flight simultaneously, which is
    // the only regime that exercises how the calendar orders competing batches
    // on one stage; parity there is the property that actually needs guarding.
    struct Case {
        Topology topology;
        std::uint64_t batch_size_cap;
        const std::string &csv;
    };
    const std::vector<Case> cases{
        {{2, 3, 1}, 1, kConcurrent},  {{4, 8, 4}, 1, kConcurrent},
        {{8, 24, 8}, 1, kConcurrent}, {{2, 3, 1}, 4, kPipelined},
        {{4, 8, 4}, 4, kPipelined},   {{4, 8, 4}, 16, kPipelined},
        {{8, 24, 8}, 8, kPipelined},
    };
    for (const auto &[topology, batch_size_cap, csv] : cases) {
        const auto exact_config = make_k3_config(
            ExecutionFlavor::kAnalytical, topology, "exact", batch_size_cap);
        auto collapsed_config = exact_config;
        collapsed_config.cluster().scheduler.pipeline_event_mode = "collapsed";
        const auto requests = workload(csv);
        const SimulationOutput exact = run_simulation(exact_config, requests);
        const SimulationOutput collapsed =
            run_simulation(collapsed_config, requests);
        expect_all_stages_visited(exact, topology);
        expect_all_stages_visited(collapsed, topology);
        expect_stage_serialization(exact);
        expect_stage_serialization(collapsed);
        expect_no_moe_sync_events(exact);
        expect_no_moe_sync_events(collapsed);
        expect_exact_collapsed_parity(exact, collapsed);
    }
}

void test_k3_pipeline_exclusive_rejects_invalid_matrix_points() {
    const Topology topology{4, 4, 1};
    const auto valid =
        make_k3_config(ExecutionFlavor::kFixed, topology, "exact");
    {
        auto invalid = valid;
        invalid.cluster().parallelism.data_parallel_size = 2;
        expect(invalid.cluster().parallelism.pipeline_exclusive,
               "matrix baseline must be pipeline-exclusive");
        expect_throws<ConfigError>(
            [&invalid] {
                static_cast<void>(parse_simulation_config_json(
                    serialize_simulation_config_json(invalid)));
            },
            "pipeline-exclusive matrix must reject DP greater than one");
    }
    {
        auto invalid = valid;
        invalid.cluster().parallelism.moe_expert_parallel_size = 2;
        expect_throws<ConfigError>(
            [&invalid] {
                static_cast<void>(parse_simulation_config_json(
                    serialize_simulation_config_json(invalid)));
            },
            "pipeline-exclusive matrix must reject MoE EP greater than one");
    }
}

} // namespace

int main() {
    int failures = 0;
    failures +=
        frontier::test::run("K3 pipeline-exclusive TP/PP/DCP analytical matrix",
                            test_k3_pipeline_exclusive_topology_matrix);
    failures += frontier::test::run(
        "K3 pipeline-exclusive fixed/analytical workload matrix",
        test_k3_fixed_and_analytical_workload_matrix);
    failures +=
        frontier::test::run("K3 exact/collapsed parity and event reduction",
                            test_k3_exact_collapsed_parity_and_event_reduction);
    failures += frontier::test::run(
        "K3 pipeline-exclusive invalid matrix points",
        test_k3_pipeline_exclusive_rejects_invalid_matrix_points);
    return failures == 0 ? 0 : 1;
}
