#include "frontier/config/config.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

#include "frontier/core/checked_math.h"
#include "frontier/core/precision.h"
#include "frontier/kv_cache_transfer/analytical_transfer.h"

namespace frontier::config {
namespace {

double storage_bytes_per_element(std::string_view precision) {
    const std::optional<Precision> parsed = parse_precision(precision);
    if (parsed.has_value()) {
        return frontier::storage_bytes_per_element(*parsed);
    }
    throw ConfigError("unsupported storage precision: " +
                      std::string{precision});
}

std::uint64_t ceil_div(std::uint64_t numerator, std::uint64_t denominator) {
    return checked_math::ceil_div<ConfigError>(
        numerator, denominator,
        "CPU KV cache block size divisor must be positive");
}

} // namespace

void resolve_gpu_memory_config(
    ClusterRuntimeConfig &cluster,
    std::optional<std::uint64_t> explicit_num_blocks) {
    if (cluster.execution_model.type == ExecutionModelType::kAnalytical) {
        apply_model_native_precision_defaults(
            cluster.execution_model.analytical, cluster.model);
    }
    GpuMemoryConfig &memory = cluster.gpu_memory;
    memory.pipeline_stage_memory_profiles.clear();
    memory.pipeline_stage_group_catalogue = {};
    memory.ordinary_kv_capacity_blocks = 0;
    memory.ordinary_kv_limiting_stage = 0;
    memory.ordinary_kv_limiting_rank = 0;
    memory.kda_snapshot_limiting_stage = 0;
    memory.kda_snapshot_limiting_rank = 0;
    if (memory.capacity_bytes_per_gpu == 0) {
        throw ConfigError("gpu_memory.capacity_bytes_per_gpu must be positive");
    }
    if (!memory.auto_calculate_num_blocks) {
        if (!explicit_num_blocks.has_value() ||
            explicit_num_blocks.value() == 0) {
            throw ConfigError(
                "manual GPU memory config requires scheduler.num_blocks");
        }
        cluster.scheduler.num_blocks = explicit_num_blocks.value();
        const auto profiles = build_pipeline_stage_memory_profiles(cluster);
        const std::uint64_t physical_capacity =
            resolve_pipeline_logical_kv_capacity(
                profiles, &memory.ordinary_kv_limiting_stage,
                &memory.ordinary_kv_limiting_rank);
        if (cluster.scheduler.num_blocks > physical_capacity) {
            throw ConfigError(
                "manual scheduler.num_blocks exceeds stage-local GPU KV "
                "capacity");
        }
        memory.pipeline_stage_memory_profiles = profiles;
        memory.pipeline_stage_group_catalogue =
            build_pipeline_stage_group_catalogue(cluster, profiles);
        memory.ordinary_kv_capacity_blocks = physical_capacity;
        memory.model_weight_bytes_per_gpu = 0;
        memory.kv_cache_budget_bytes_per_gpu =
            std::numeric_limits<std::uint64_t>::max();
        memory.kv_cache_bytes_per_block = 0;
        for (const auto &profile : profiles) {
            memory.model_weight_bytes_per_gpu =
                std::max(memory.model_weight_bytes_per_gpu,
                         profile.resident_weight_bytes);
            memory.kv_cache_budget_bytes_per_gpu = std::min(
                memory.kv_cache_budget_bytes_per_gpu, profile.free_bytes);
            for (const auto bytes : profile.kv_bytes_per_block_by_rank) {
                memory.kv_cache_bytes_per_block =
                    std::max(memory.kv_cache_bytes_per_block, bytes);
            }
        }
        memory.kda_snapshot_limiting_stage = 0;
        memory.kda_snapshot_limiting_rank = 0;
        cluster.scheduler.kda_snapshot_blocks_per_session =
            resolve_pipeline_kda_snapshot_charge(
                profiles, cluster.scheduler.num_blocks,
                &memory.kda_snapshot_limiting_stage,
                &memory.kda_snapshot_limiting_rank);
        return;
    }
    if (cluster.execution_model.type != ExecutionModelType::kAnalytical) {
        throw ConfigError(
            "automatic GPU block calculation requires analytical execution");
    }
    if (!std::isfinite(memory.weight_overhead_fraction) ||
        memory.weight_overhead_fraction < 0.0) {
        throw ConfigError(
            "gpu_memory.weight_overhead_fraction must be nonnegative");
    }
    const auto profiles = build_pipeline_stage_memory_profiles(cluster);
    memory.pipeline_stage_memory_profiles = profiles;
    memory.pipeline_stage_group_catalogue =
        build_pipeline_stage_group_catalogue(cluster, profiles);
    memory.model_weight_bytes_per_gpu = 0;
    memory.kv_cache_budget_bytes_per_gpu =
        std::numeric_limits<std::uint64_t>::max();
    memory.kv_cache_bytes_per_block = 0;
    for (const auto &profile : profiles) {
        memory.model_weight_bytes_per_gpu = std::max(
            memory.model_weight_bytes_per_gpu, profile.resident_weight_bytes);
        // Compatibility scalar: expose the least-free physical stage while
        // retaining exact F/B vectors in pipeline_stage_memory_profiles.
        memory.kv_cache_budget_bytes_per_gpu =
            std::min(memory.kv_cache_budget_bytes_per_gpu, profile.free_bytes);
        for (const auto bytes : profile.kv_bytes_per_block_by_rank) {
            memory.kv_cache_bytes_per_block =
                std::max(memory.kv_cache_bytes_per_block, bytes);
        }
    }
    const std::uint64_t calculated_blocks =
        resolve_pipeline_logical_kv_capacity(profiles,
                                             &memory.ordinary_kv_limiting_stage,
                                             &memory.ordinary_kv_limiting_rank);
    if (calculated_blocks == 0) {
        throw ConfigError(
            "remaining stage-local GPU KV budget cannot hold one cache block");
    }
    if (explicit_num_blocks.has_value() &&
        explicit_num_blocks.value() != calculated_blocks) {
        throw ConfigError(
            "scheduler.num_blocks does not match automatic GPU memory "
            "calculation");
    }
    cluster.scheduler.num_blocks = calculated_blocks;
    memory.ordinary_kv_capacity_blocks = calculated_blocks;
    cluster.scheduler.kda_snapshot_blocks_per_session =
        resolve_pipeline_kda_snapshot_charge(
            profiles, calculated_blocks, &memory.kda_snapshot_limiting_stage,
            &memory.kda_snapshot_limiting_rank);
}

ResolvedCpuKVCacheTargetConfig
resolve_cpu_kv_cache_target(const SimulationConfig &config) {
    ResolvedCpuKVCacheTargetConfig result{};
    result.enabled = config.cpu_kv_cache.enabled;
    if (!result.enabled) {
        return result;
    }
    if (config.system_architecture != SystemArchitecture::kPdDisaggregation) {
        throw ConfigError("CPU KV cache requires sequential PDD");
    }
    const auto &prefill = config.pdd().clusters.prefill;
    if (prefill.parallelism.decode_context_parallel_size != 1) {
        throw ConfigError("PDD PREFILL decode_context_parallel_size must be 1");
    }
    const auto &cpu = config.cpu_kv_cache;
    result.capacity_pressure_policy = cpu.capacity_pressure_policy;
    if (prefill.parallelism.tensor_parallel_size >
        std::numeric_limits<std::uint64_t>::max() /
            prefill.parallelism.pipeline_parallel_size) {
        throw ConfigError("CPU KV cache physical slice count overflows uint64");
    }
    const std::uint64_t slices = prefill.parallelism.tensor_parallel_size *
                                 prefill.parallelism.pipeline_parallel_size;
    if (slices == 0) {
        throw ConfigError("CPU KV cache target has no physical GPU slices");
    }
    if (cpu.static_slice_per_gpu) {
        if (cpu.capacity_bytes_per_gpu >
            std::numeric_limits<std::uint64_t>::max() / slices) {
            throw ConfigError(
                "CPU KV cache resolved capacity overflows uint64");
        }
        result.capacity_bytes = cpu.capacity_bytes_per_gpu * slices;
        const double bandwidth = std::min(cpu.dram_bandwidth_gbps_per_gpu,
                                          cpu.c2c_bandwidth_gbps_per_gpu) *
                                 static_cast<double>(slices);
        if (!std::isfinite(bandwidth)) {
            throw ConfigError("CPU KV cache resolved bandwidth overflows");
        }
        result.d2h_bandwidth_gbps = bandwidth;
        result.h2d_bandwidth_gbps = bandwidth;
    } else {
        result.capacity_bytes = cpu.capacity_bytes;
        result.d2h_bandwidth_gbps = cpu.write_bandwidth_gbps;
        result.h2d_bandwidth_gbps = cpu.read_bandwidth_gbps;
    }
    result.d2h_latency_ms = cpu.write_latency_ms;
    result.h2d_latency_ms = cpu.read_latency_ms;
    try {
        result.bytes_per_block =
            kv_cache_transfer::model_kv_cache_size_bytes_target_physical(
                prefill.scheduler.block_size, prefill.model,
                config.pdd().kv_cache_transfer.kv_cache_dtype_size_bytes,
                prefill.parallelism.tensor_parallel_size,
                prefill.parallelism.decode_context_parallel_size);
    } catch (const kv_cache_transfer::TransferModelError &error) {
        throw ConfigError(std::string{"invalid CPU KV cache block layout: "} +
                          error.what());
    }
    if (result.bytes_per_block == 0) {
        throw ConfigError("CPU KV cache bytes per block must be positive");
    }
    result.capacity_blocks = result.capacity_bytes / result.bytes_per_block;
    if (result.capacity_blocks == 0) {
        throw ConfigError(
            "CPU KV cache capacity must hold at least one logical KV block");
    }
    if (prefill.model.has_kda()) {
        try {
            result.kda_snapshot_bytes =
                kv_cache_transfer::model_kda_state_snapshot_size_bytes(
                    prefill.model, prefill.execution_model.type ==
                                           ExecutionModelType::kAnalytical
                                       ? storage_bytes_per_element(
                                             prefill.execution_model.analytical
                                                 .kda_snapshot_precision())
                                       : 2.0);
        } catch (const kv_cache_transfer::TransferModelError &error) {
            throw ConfigError(std::string{"invalid CPU KDA snapshot layout: "} +
                              error.what());
        }
        result.kda_snapshot_blocks =
            ceil_div(result.kda_snapshot_bytes, result.bytes_per_block);
        if (result.kda_snapshot_blocks >= result.capacity_blocks) {
            throw ConfigError(
                "CPU KV cache must hold one atomic KDA snapshot plus at "
                "least one logical KV block");
        }
    }
    return result;
}

} // namespace frontier::config
