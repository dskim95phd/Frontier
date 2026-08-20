#include "frontier/config/config.h"
#include "frontier/metrics/output_contract.h"
#include "frontier/request_generator/workload.h"
#include "frontier/simulator/simulator.h"
#include "tests/test_support.h"

#include <algorithm>
#include <filesystem>
#include <string>

#ifndef FRONTIER_EXAMPLE_DIR
#error "FRONTIER_EXAMPLE_DIR must be defined for K3 CPU KV-cache tests"
#endif

namespace {

using frontier::EventType;
using frontier::RequestId;
using frontier::config::parse_simulation_config_json;
using frontier::metrics::CpuKVCacheTransferKind;
using frontier::request_generator::parse_workload_csv;
using frontier::simulator::run_simulation;
using frontier::test::expect;
using frontier::test::read_text_file;

const std::filesystem::path kExampleRoot{FRONTIER_EXAMPLE_DIR};

void replace_all(std::string &text, const std::string &from,
                 const std::string &to) {
    std::size_t position = 0;
    std::size_t replacements = 0;
    while ((position = text.find(from, position)) != std::string::npos) {
        text.replace(position, from.size(), to);
        position += to.size();
        ++replacements;
    }
    expect(replacements > 0, "K3 integration fixture replacement is missing");
}

void run_k3_cpu_snapshot_offload_and_restore(std::uint64_t pp_size) {
    std::string config_text = read_text_file(kExampleRoot / "configs" /
                                             "06_cpu_kv_cache_pdd_online.json");
    replace_all(config_text, "meta-llama/Llama-2-7b-hf", "moonshotai/Kimi-K3");
    replace_all(config_text, "\"capacity_bytes\": 67108864",
                "\"capacity_bytes\": 2147483648");
    replace_all(config_text, "\"num_blocks\": 4", "\"num_blocks\": 16");
    replace_all(config_text, "\"num_blocks\": 32", "\"num_blocks\": 16");

    auto config = parse_simulation_config_json(config_text);
    auto &clusters = config.pdd().clusters;
    for (auto *cluster : {&clusters.prefill, &clusters.decode}) {
        cluster->parallelism.pipeline_parallel_size = pp_size;
        if (pp_size > 1) {
            cluster->parallelism.pipeline_exclusive = true;
            cluster->scheduler.pipeline_event_mode = "collapsed";
        }
        const double stage_latency =
            cluster->execution_model.fixed.stage_latencies_ms.front();
        cluster->execution_model.fixed.stage_latencies_ms.assign(
            static_cast<std::size_t>(pp_size), stage_latency);
        // Keep enough physical HBM for one KDA snapshot plus ordinary KV, but
        // not for two concurrent snapshots. With the fixture's default 10%
        // reserve, capacity=2*S_max yields F=1.8*S_max and the normalized
        // logical charge is ceil(16/1.8)=9 blocks. Two sessions therefore
        // exceed the sixteen-block manager and force the CPU restore path,
        // while one prompt still has room for its ordinary KV blocks.
        const auto provisional_profiles =
            frontier::config::build_pipeline_stage_memory_profiles(*cluster);
        std::uint64_t maximum_snapshot_bytes = 0;
        for (const auto &profile : provisional_profiles) {
            for (const auto snapshot_bytes :
                 profile.kda_snapshot_bytes_by_rank) {
                maximum_snapshot_bytes =
                    std::max(maximum_snapshot_bytes, snapshot_bytes);
            }
        }
        expect(maximum_snapshot_bytes > 0,
               "K3 PP fixture must expose a stage-local snapshot shard");
        cluster->gpu_memory.capacity_bytes_per_gpu = 2 * maximum_snapshot_bytes;
        frontier::config::resolve_gpu_memory_config(
            *cluster, cluster->scheduler.num_blocks);
    }
    // Serialize the two initial turns so the second atomic GPU snapshot
    // deterministically reclaims the first session.  The later successor can
    // then reuse that session only through CPU MLA KV + KDA restoration.
    const auto workload = parse_workload_csv(
        "session_start_at,think_time,num_prefill_tokens,num_decode_tokens,"
        "session_id,session_turn_index\n"
        "0,0,8,2,7,0\n"
        "0.2,0,12,2,9,0\n"
        ",0.5,4,2,7,1\n");
    const auto output = run_simulation(config, workload);

    if (pp_size > 1) {
        expect(std::any_of(output.event_trace.begin(), output.event_trace.end(),
                           [](const auto &event) {
                               return event.type() ==
                                      EventType::kBatchPipelineEnd;
                           }) &&
                   std::none_of(
                       output.event_trace.begin(), output.event_trace.end(),
                       [](const auto &event) {
                           return event.type() == EventType::kPrefillSync ||
                                  event.type() == EventType::kDecodeSync;
                       }),
               "K3 pipeline-exclusive PP must use the collapsed calendar "
               "without DP/EP synchronization events");
    }

    expect(output.requests.size() == 3,
           "K3 CPU tier must complete both initial turns and the successor");
    const auto successor = std::find_if(
        output.requests.begin(), output.requests.end(),
        [](const auto &record) { return record.request_id == RequestId{2}; });
    expect(successor != output.requests.end() &&
               successor->cpu_prefix_hit_blocks > 0 &&
               successor->cpu_restore_bytes > 232'316'928ULL,
           "K3 successor must restore MLA KV together with one KDA snapshot");

    const auto &cpu = output.aggregate.cpu_kv_cache;
    expect(cpu.kda_snapshot_bytes_per_session == 232'316'928ULL &&
               cpu.kda_snapshot_offload_operations >= 2 &&
               cpu.kda_snapshot_restore_operations >= 1 &&
               cpu.kda_snapshot_restore_bytes >= 232'316'928ULL &&
               cpu.kda_snapshot_reserved_blocks == 0,
           "K3 CPU aggregate must expose quiescent atomic snapshot traffic");
    expect(
        output.cpu_kv_cache_targets.size() == 1 &&
            output.cpu_kv_cache_targets.front().kda_snapshot_occupied_blocks >
                0 &&
            output.cpu_kv_cache_targets.front().kda_snapshot_reserved_blocks ==
                0,
        "K3 CPU target must retain whole snapshots without partial state");
    const bool restored_snapshot = std::any_of(
        output.cpu_kv_cache_transfers.begin(),
        output.cpu_kv_cache_transfers.end(), [](const auto &transfer) {
            return transfer.kind == CpuKVCacheTransferKind::kRestore &&
                   transfer.kda_snapshot_bytes == 232'316'928ULL;
        });
    expect(restored_snapshot,
           "detailed CPU transfer trace must identify the atomic KDA payload");

    for (std::uint64_t stage = 0; stage < pp_size; ++stage) {
        expect(
            std::any_of(output.batch_stages.begin(), output.batch_stages.end(),
                        [stage](const auto &record) {
                            return record.cluster_type ==
                                       frontier::ClusterType::kPrefill &&
                                   record.stage_id == frontier::StageId{stage};
                        }),
            "K3 CPU offload run must execute every PREFILL PP stage");
    }

    const auto batch_contains_request = [&](frontier::BatchId batch_id,
                                            RequestId request_id) {
        const auto batch =
            std::find_if(output.batches.begin(), output.batches.end(),
                         [batch_id](const auto &record) {
                             return record.batch_id == batch_id;
                         });
        return batch != output.batches.end() &&
               std::find(batch->request_ids.begin(), batch->request_ids.end(),
                         request_id) != batch->request_ids.end();
    };
    for (const auto &transfer : output.cpu_kv_cache_transfers) {
        if (transfer.kind != CpuKVCacheTransferKind::kOffload) {
            continue;
        }
        frontier::SimTime final_stage_completion{};
        for (const auto &stage : output.batch_stages) {
            if (stage.cluster_type == frontier::ClusterType::kPrefill &&
                stage.stage_id == frontier::StageId{pp_size - 1} &&
                batch_contains_request(stage.batch_id, transfer.request_id) &&
                (!final_stage_completion.valid() ||
                 final_stage_completion < stage.completed_at)) {
                final_stage_completion = stage.completed_at;
            }
        }
        expect(final_stage_completion.valid() &&
                   final_stage_completion <= transfer.submitted_at,
               "CPU offload must be submitted after the request's final PP "
               "stage completes");
    }

    const auto restore = std::find_if(
        output.cpu_kv_cache_transfers.begin(),
        output.cpu_kv_cache_transfers.end(), [](const auto &transfer) {
            return transfer.kind == CpuKVCacheTransferKind::kRestore &&
                   transfer.request_id == RequestId{2};
        });
    frontier::SimTime successor_stage0_arrival{};
    for (const auto &stage : output.batch_stages) {
        if (stage.cluster_type == frontier::ClusterType::kPrefill &&
            stage.stage_id == frontier::StageId{0} &&
            batch_contains_request(stage.batch_id, RequestId{2}) &&
            (!successor_stage0_arrival.valid() ||
             stage.arrived_at < successor_stage0_arrival)) {
            successor_stage0_arrival = stage.arrived_at;
        }
    }
    expect(restore != output.cpu_kv_cache_transfers.end() &&
               successor_stage0_arrival.valid() &&
               restore->completed_at <= successor_stage0_arrival,
           "PP0 admission must wait for the full atomic CPU restore");
}

void test_k3_cpu_snapshot_offload_and_restore() {
    run_k3_cpu_snapshot_offload_and_restore(1);
    run_k3_cpu_snapshot_offload_and_restore(2);
    run_k3_cpu_snapshot_offload_and_restore(4);
}

void test_k3_pdd_tp8_dcp8_analytical_decode() {
    std::string config_text =
        read_text_file(kExampleRoot / "configs" / "03_sequential_pdd.json");
    replace_all(config_text, "meta-llama/Llama-2-7b-hf", "moonshotai/Kimi-K3");
    auto config = parse_simulation_config_json(config_text);
    auto &clusters = config.pdd().clusters;
    for (auto *cluster : {&clusters.prefill, &clusters.decode}) {
        cluster->parallelism.tensor_parallel_size = 8;
        cluster->parallelism.pipeline_parallel_size = 2;
        cluster->parallelism.data_parallel_size = 1;
        cluster->parallelism.moe_tensor_parallel_size = 8;
        cluster->parallelism.moe_expert_parallel_size = 1;
        cluster->parallelism.pipeline_exclusive = true;
        cluster->scheduler.pipeline_event_mode = "collapsed";
        cluster->scheduler.num_blocks = 5'000;
        cluster->execution_model.type =
            frontier::config::ExecutionModelType::kAnalytical;
        cluster->execution_model.analytical = {};
        cluster->execution_model.analytical.tensor_parallel_size = 8;
        cluster->execution_model.analytical.moe_layer_event_mode =
            "stage_group_scaled";
    }
    clusters.prefill.parallelism.decode_context_parallel_size = 1;
    clusters.decode.parallelism.decode_context_parallel_size = 8;
    config.pdd().kv_cache_transfer.kv_cache_dtype_size_bytes = 1.0;
    frontier::config::resolve_gpu_memory_config(
        clusters.prefill, clusters.prefill.scheduler.num_blocks);
    frontier::config::resolve_gpu_memory_config(
        clusters.decode, clusters.decode.scheduler.num_blocks);

    const auto workload = parse_workload_csv(
        "session_start_at,think_time,num_prefill_tokens,num_decode_tokens\n"
        "0,0,32,2\n");
    const auto output = run_simulation(config, workload);
    const auto decode_stage = std::find_if(
        output.batch_stages.begin(), output.batch_stages.end(),
        [](const auto &stage) {
            return stage.cluster_type == frontier::ClusterType::kDecode;
        });
    expect(output.requests.size() == 1 &&
               output.kv_cache_transfers.size() == 1 &&
               decode_stage != output.batch_stages.end(),
           "K3 TP8/DCP8 PDD run must complete prefill, transfer, and decode");
    expect(decode_stage->parallelism.tensor_parallel_size == 8 &&
               decode_stage->parallelism.decode_context_parallel_size == 8 &&
               decode_stage->execution_time.dense_compute_ms > 0.0 &&
               decode_stage->execution_time.tp_communication_ms > 0.0,
           "K3 decode stage must execute analytical TP8/DCP8 compute and "
           "collectives");
    expect(
        std::any_of(output.event_trace.begin(), output.event_trace.end(),
                    [](const auto &event) {
                        return event.type() == EventType::kBatchPipelineEnd;
                    }) &&
            std::none_of(output.event_trace.begin(), output.event_trace.end(),
                         [](const auto &event) {
                             return event.type() == EventType::kPrefillSync ||
                                    event.type() == EventType::kDecodeSync;
                         }),
        "K3 TP8 pipeline-exclusive analytical run must collapse PP "
        "without DP/EP synchronization");
    expect(output.kv_cache_transfers.front().size_bytes > 232'316'928ULL,
           "K3 PDD transfer must include MLA KV and the full KDA snapshot");
}

} // namespace

int main() {
    int failures = 0;
    failures += frontier::test::run("K3 CPU KV snapshot offload and restore",
                                    test_k3_cpu_snapshot_offload_and_restore);
    failures += frontier::test::run("K3 PDD TP8 DCP8 analytical decode",
                                    test_k3_pdd_tp8_dcp8_analytical_decode);
    return failures == 0 ? 0 : 1;
}
