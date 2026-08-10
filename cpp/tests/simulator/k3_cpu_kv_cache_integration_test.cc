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

void test_k3_cpu_snapshot_offload_and_restore() {
    std::string config_text = read_text_file(kExampleRoot / "configs" /
                                             "06_cpu_kv_cache_pdd_online.json");
    replace_all(config_text, "meta-llama/Llama-2-7b-hf", "moonshotai/Kimi-K3");
    replace_all(config_text, "\"capacity_bytes\": 67108864",
                "\"capacity_bytes\": 2147483648");
    replace_all(config_text, "\"num_blocks\": 4", "\"num_blocks\": 5000");
    replace_all(config_text, "\"num_blocks\": 32", "\"num_blocks\": 5000");

    const auto config = parse_simulation_config_json(config_text);
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
}

void test_k3_pdd_tp8_dcp8_analytical_decode() {
    std::string config_text = read_text_file(kExampleRoot / "configs" /
                                             "03_sequential_pdd.json");
    replace_all(config_text, "meta-llama/Llama-2-7b-hf", "moonshotai/Kimi-K3");
    auto config = parse_simulation_config_json(config_text);
    auto &clusters = config.pdd().clusters;
    for (auto *cluster : {&clusters.prefill, &clusters.decode}) {
        cluster->parallelism.tensor_parallel_size = 8;
        cluster->parallelism.pipeline_parallel_size = 1;
        cluster->parallelism.data_parallel_size = 1;
        cluster->parallelism.moe_tensor_parallel_size = 1;
        cluster->parallelism.moe_expert_parallel_size = 8;
        cluster->scheduler.num_blocks = 5'000;
        cluster->execution_model.type =
            frontier::config::ExecutionModelType::kAnalytical;
        cluster->execution_model.analytical = {};
        cluster->execution_model.analytical.tensor_parallel_size = 8;
        cluster->execution_model.analytical.moe_layer_event_mode =
            "first_layer_scaled";
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
