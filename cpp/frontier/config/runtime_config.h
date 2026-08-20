#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <variant>
#include <vector>

#include "frontier/attention/ops.h"
#include "frontier/core/cluster_type.h"
#include "frontier/core/ids.h"

#include "frontier/config/execution_model_config.h"

namespace frontier::config {

struct GpuMemoryConfig {
    bool auto_calculate_num_blocks = false;
    // Required for every runtime cluster.  Zero denotes an unbound in-memory
    // object and is rejected by parsing and resolve_gpu_memory_config().
    std::uint64_t capacity_bytes_per_gpu = 0;
    double runtime_reserve_fraction = 0.0;
    std::uint64_t runtime_reserve_bytes = 0;
    double weight_overhead_fraction = 0.0;

    // Materialized diagnostics. These are emitted in normalized configs and
    // make the automatic block calculation auditable and reproducible.
    std::uint64_t model_weight_bytes_per_gpu = 0;
    std::uint64_t kv_cache_budget_bytes_per_gpu = 0;
    std::uint64_t kv_cache_bytes_per_block = 0;

    // Exact PP stage/rank-local memory diagnostics.  These are derived fields
    // populated by resolve_gpu_memory_config; they are deliberately omitted
    // from the value-equality contract and legacy JSON surface so older
    // normalized configs continue to round-trip.  Consumers that need physical
    // occupancy should use these profiles instead of reconstructing a full
    // model footprint from kv_cache_bytes_per_block.
    std::vector<PipelineStageMemoryProfile> pipeline_stage_memory_profiles;
    PipelineStageGroupCatalogue pipeline_stage_group_catalogue;
    std::uint64_t ordinary_kv_capacity_blocks = 0;
    std::uint64_t ordinary_kv_limiting_stage = 0;
    std::uint64_t ordinary_kv_limiting_rank = 0;
    std::uint64_t kda_snapshot_limiting_stage = 0;
    std::uint64_t kda_snapshot_limiting_rank = 0;

    friend bool operator==(const GpuMemoryConfig &lhs,
                           const GpuMemoryConfig &rhs) {
        return std::tie(lhs.auto_calculate_num_blocks,
                        lhs.capacity_bytes_per_gpu,
                        lhs.runtime_reserve_fraction, lhs.runtime_reserve_bytes,
                        lhs.weight_overhead_fraction) ==
               std::tie(rhs.auto_calculate_num_blocks,
                        rhs.capacity_bytes_per_gpu,
                        rhs.runtime_reserve_fraction, rhs.runtime_reserve_bytes,
                        rhs.weight_overhead_fraction);
    }
};

struct ClusterRuntimeConfig {
    ParallelismConfig parallelism;
    SchedulerConfig scheduler;
    ExecutionModelConfig execution_model;
    GpuMemoryConfig gpu_memory;
    ModelConfig model;
    MoeRoutingConfig moe_routing;

    friend bool operator==(const ClusterRuntimeConfig &lhs,
                           const ClusterRuntimeConfig &rhs) {
        return std::tie(lhs.parallelism, lhs.scheduler, lhs.execution_model,
                        lhs.gpu_memory, lhs.model, lhs.moe_routing) ==
               std::tie(rhs.parallelism, rhs.scheduler, rhs.execution_model,
                        rhs.gpu_memory, rhs.model, rhs.moe_routing);
    }
};

struct PddClustersConfig {
    ClusterRuntimeConfig prefill;
    ClusterRuntimeConfig decode;

    friend bool operator==(const PddClustersConfig &lhs,
                           const PddClustersConfig &rhs) {
        return std::tie(lhs.prefill, lhs.decode) ==
               std::tie(rhs.prefill, rhs.decode);
    }
};

struct KvCacheTransferConfig {
    double network_bandwidth_gbps = 100.0;
    double network_latency_ms = 0.1;
    double kv_cache_dtype_size_bytes = 2.0;
    bool enable_compression = false;

    friend bool operator==(const KvCacheTransferConfig &lhs,
                           const KvCacheTransferConfig &rhs) {
        return std::tie(lhs.network_bandwidth_gbps, lhs.network_latency_ms,
                        lhs.kv_cache_dtype_size_bytes,
                        lhs.enable_compression) ==
               std::tie(rhs.network_bandwidth_gbps, rhs.network_latency_ms,
                        rhs.kv_cache_dtype_size_bytes, rhs.enable_compression);
    }
};

enum class CpuKVCacheEvictionPolicy { kSessionLruSuffix };

enum class CpuKVCacheCapacityPressurePolicy { kPrefixFit, kSkipOffload };

enum class CpuKVCacheTransferConcurrency { kFullDuplexSerialized };

struct CpuKVCacheConfig {
    bool enabled = false;
    std::uint64_t capacity_bytes = 0;
    bool static_slice_per_gpu = false;
    std::uint64_t capacity_bytes_per_gpu = 750'000'000'000ULL;
    double dram_bandwidth_gbps_per_gpu = 4'800.0;
    double c2c_bandwidth_gbps_per_gpu = 3'600.0;
    double write_bandwidth_gbps = 64.0;
    double write_latency_ms = 0.01;
    double read_bandwidth_gbps = 64.0;
    double read_latency_ms = 0.01;
    CpuKVCacheEvictionPolicy eviction_policy =
        CpuKVCacheEvictionPolicy::kSessionLruSuffix;
    CpuKVCacheCapacityPressurePolicy capacity_pressure_policy =
        CpuKVCacheCapacityPressurePolicy::kPrefixFit;
    CpuKVCacheTransferConcurrency transfer_concurrency =
        CpuKVCacheTransferConcurrency::kFullDuplexSerialized;

    friend bool operator==(const CpuKVCacheConfig &lhs,
                           const CpuKVCacheConfig &rhs) {
        return std::tie(
                   lhs.enabled, lhs.capacity_bytes, lhs.static_slice_per_gpu,
                   lhs.capacity_bytes_per_gpu, lhs.dram_bandwidth_gbps_per_gpu,
                   lhs.c2c_bandwidth_gbps_per_gpu, lhs.write_bandwidth_gbps,
                   lhs.write_latency_ms, lhs.read_bandwidth_gbps,
                   lhs.read_latency_ms, lhs.eviction_policy,
                   lhs.capacity_pressure_policy, lhs.transfer_concurrency) ==
               std::tie(
                   rhs.enabled, rhs.capacity_bytes, rhs.static_slice_per_gpu,
                   rhs.capacity_bytes_per_gpu, rhs.dram_bandwidth_gbps_per_gpu,
                   rhs.c2c_bandwidth_gbps_per_gpu, rhs.write_bandwidth_gbps,
                   rhs.write_latency_ms, rhs.read_bandwidth_gbps,
                   rhs.read_latency_ms, rhs.eviction_policy,
                   rhs.capacity_pressure_policy, rhs.transfer_concurrency);
    }
};

struct ResolvedCpuKVCacheTargetConfig {
    bool enabled = false;
    std::uint64_t capacity_bytes = 0;
    std::uint64_t capacity_blocks = 0;
    std::uint64_t bytes_per_block = 0;
    // Hybrid KDA models keep one latest recurrent-state snapshot per cached
    // session. The CPU manager charges it as one indivisible group while
    // transfer accounting uses the exact byte size.
    std::uint64_t kda_snapshot_bytes = 0;
    std::uint64_t kda_snapshot_blocks = 0;
    double d2h_bandwidth_gbps = 0.0;
    double d2h_latency_ms = 0.0;
    double h2d_bandwidth_gbps = 0.0;
    double h2d_latency_ms = 0.0;
    CpuKVCacheCapacityPressurePolicy capacity_pressure_policy =
        CpuKVCacheCapacityPressurePolicy::kPrefixFit;
};

struct PddRuntimeConfig {
    PddClustersConfig clusters;
    KvCacheTransferConfig kv_cache_transfer;

    friend bool operator==(const PddRuntimeConfig &lhs,
                           const PddRuntimeConfig &rhs) {
        return std::tie(lhs.clusters, lhs.kv_cache_transfer) ==
               std::tie(rhs.clusters, rhs.kv_cache_transfer);
    }
};

using RuntimeConfig = std::variant<ClusterRuntimeConfig, PddRuntimeConfig>;

class ConfigError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

struct SimulationConfig {
    int schema_version;
    std::string run_id;
    SimulationMode simulation_mode;
    SystemArchitecture system_architecture;
    bool enable_parallel_clusters;
    PrefixCacheConfig prefix_cache;
    CpuKVCacheConfig cpu_kv_cache;
    ClusterSchedulerConfig cluster_scheduler;
    RuntimeConfig runtime;

    [[nodiscard]] ClusterRuntimeConfig &cluster();
    [[nodiscard]] const ClusterRuntimeConfig &cluster() const;
    [[nodiscard]] PddRuntimeConfig &pdd();
    [[nodiscard]] const PddRuntimeConfig &pdd() const;

    friend bool operator==(const SimulationConfig &lhs,
                           const SimulationConfig &rhs) {
        return std::tie(lhs.schema_version, lhs.run_id, lhs.simulation_mode,
                        lhs.system_architecture, lhs.enable_parallel_clusters,
                        lhs.prefix_cache, lhs.cpu_kv_cache,
                        lhs.cluster_scheduler, lhs.runtime) ==
               std::tie(rhs.schema_version, rhs.run_id, rhs.simulation_mode,
                        rhs.system_architecture, rhs.enable_parallel_clusters,
                        rhs.prefix_cache, rhs.cpu_kv_cache,
                        rhs.cluster_scheduler, rhs.runtime);
    }
};

} // namespace frontier::config
