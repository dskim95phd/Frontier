#include "frontier/config/config.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "frontier/kv_cache_transfer/analytical_transfer.h"

namespace frontier::config {

std::string_view to_string(SimulationMode mode) noexcept {
    switch (mode) {
    case SimulationMode::kOffline:
        return "offline";
    case SimulationMode::kOnline:
        return "online";
    }
    return "unknown";
}

std::string_view to_string(SystemArchitecture architecture) noexcept {
    switch (architecture) {
    case SystemArchitecture::kCoLocation:
        return "co-location";
    case SystemArchitecture::kPdDisaggregation:
        return "pd-disaggregation";
    }
    return "unknown";
}

std::string_view to_string(PrefixCachingKeyMode key_mode) noexcept {
    switch (key_mode) {
    case PrefixCachingKeyMode::kSession:
        return "session";
    }
    return "unknown";
}

std::string_view to_string(SchedulerType type) noexcept {
    switch (type) {
    case SchedulerType::kVllmV1:
        return "vllm_v1";
    }
    return "unknown";
}

std::string_view to_string(SchedulingPolicy policy) noexcept {
    switch (policy) {
    case SchedulingPolicy::kFcfs:
        return "fcfs";
    }
    return "unknown";
}

std::string_view to_string(ClusterSchedulerType type) noexcept {
    switch (type) {
    case ClusterSchedulerType::kRoundRobin:
        return "round_robin";
    case ClusterSchedulerType::kStickyRoundRobin:
        return "sticky_round_robin";
    }
    return "unknown";
}

std::string_view to_string(ModelKind kind) noexcept {
    switch (kind) {
    case ModelKind::kDense:
        return "dense";
    case ModelKind::kMoe:
        return "moe";
    }
    return "unknown";
}

std::string_view to_string(MoeRoutingMode mode) noexcept {
    switch (mode) {
    case MoeRoutingMode::kSimulation:
        return "simulation";
    case MoeRoutingMode::kUniformLegacy:
        return "uniform_legacy";
    case MoeRoutingMode::kUniformRandom:
        return "uniform_random";
    }
    return "unknown";
}

std::string_view to_string(MoeRoutingDistribution distribution) noexcept {
    switch (distribution) {
    case MoeRoutingDistribution::kBalanced:
        return "balanced";
    case MoeRoutingDistribution::kRandom:
        return "random";
    case MoeRoutingDistribution::kSkewed:
        return "skewed";
    case MoeRoutingDistribution::kZipf:
        return "zipf";
    }
    return "unknown";
}

std::string_view to_string(ExecutionModelType type) noexcept {
    switch (type) {
    case ExecutionModelType::kFixed:
        return "fixed";
    case ExecutionModelType::kAnalytical:
        return "analytical";
    }
    return "unknown";
}

std::string_view to_string(CpuKVCacheEvictionPolicy policy) noexcept {
    switch (policy) {
    case CpuKVCacheEvictionPolicy::kSessionLruSuffix:
        return "session_lru_suffix";
    }
    return "unknown";
}

std::string_view
to_string(CpuKVCacheCapacityPressurePolicy policy) noexcept {
    switch (policy) {
    case CpuKVCacheCapacityPressurePolicy::kPrefixFit:
        return "prefix_fit";
    case CpuKVCacheCapacityPressurePolicy::kSkipOffload:
        return "skip_offload";
    }
    return "unknown";
}

std::string_view
to_string(CpuKVCacheTransferConcurrency concurrency) noexcept {
    switch (concurrency) {
    case CpuKVCacheTransferConcurrency::kFullDuplexSerialized:
        return "full_duplex_serialized";
    }
    return "unknown";
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
            throw ConfigError("CPU KV cache resolved capacity overflows uint64");
        }
        result.capacity_bytes = cpu.capacity_bytes_per_gpu * slices;
        const double bandwidth =
            std::min(cpu.dram_bandwidth_gbps_per_gpu,
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
                prefill.parallelism.tensor_parallel_size);
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
    return result;
}

ClusterRuntimeConfig &SimulationConfig::cluster() {
    auto *value = std::get_if<ClusterRuntimeConfig>(&runtime);
    if (value == nullptr) {
        throw ConfigError("simulation config has no single cluster runtime");
    }
    return *value;
}

const ClusterRuntimeConfig &SimulationConfig::cluster() const {
    const auto *value = std::get_if<ClusterRuntimeConfig>(&runtime);
    if (value == nullptr) {
        throw ConfigError("simulation config has no single cluster runtime");
    }
    return *value;
}

PddRuntimeConfig &SimulationConfig::pdd() {
    auto *value = std::get_if<PddRuntimeConfig>(&runtime);
    if (value == nullptr) {
        throw ConfigError("simulation config is not a PDD config");
    }
    return *value;
}

const PddRuntimeConfig &SimulationConfig::pdd() const {
    const auto *value = std::get_if<PddRuntimeConfig>(&runtime);
    if (value == nullptr) {
        throw ConfigError("simulation config is not a PDD config");
    }
    return *value;
}

} // namespace frontier::config
