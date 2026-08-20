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
#include "frontier/config/runtime_config.h"
#include "frontier/core/cluster_type.h"
#include "frontier/core/ids.h"

namespace frontier::config {

[[nodiscard]] std::string_view to_string(SimulationMode mode) noexcept;
[[nodiscard]] std::string_view
to_string(SystemArchitecture architecture) noexcept;
[[nodiscard]] std::string_view
to_string(PrefixCachingKeyMode key_mode) noexcept;
[[nodiscard]] std::string_view to_string(SchedulerType type) noexcept;
[[nodiscard]] std::string_view to_string(SchedulingPolicy policy) noexcept;
[[nodiscard]] std::string_view to_string(ClusterSchedulerType type) noexcept;
[[nodiscard]] std::string_view to_string(ModelKind kind) noexcept;
[[nodiscard]] std::string_view to_string(MoeRoutingMode mode) noexcept;
[[nodiscard]] std::string_view
to_string(MoeRoutingLayerScope layer_scope) noexcept;
[[nodiscard]] std::string_view
to_string(MoeRoutingDistribution distribution) noexcept;
[[nodiscard]] std::string_view to_string(ExecutionModelType type) noexcept;
[[nodiscard]] std::string_view
to_string(CpuKVCacheEvictionPolicy policy) noexcept;
[[nodiscard]] std::string_view
to_string(CpuKVCacheCapacityPressurePolicy policy) noexcept;
[[nodiscard]] std::string_view
to_string(CpuKVCacheTransferConcurrency concurrency) noexcept;

[[nodiscard]] ResolvedCpuKVCacheTargetConfig
resolve_cpu_kv_cache_target(const SimulationConfig &config);

// Resolve the common KDA recurrent-state snapshot element size used by a
// sequential PDD transfer. Fixed execution follows the native BF16 snapshot;
// analytical PREFILL/DECODE clusters must agree on the exact precision when
// the model exposes KDA layers. A mixed analytical/fixed pair is therefore
// valid only when the analytical side also uses BF16.
[[nodiscard]] double
resolve_pdd_kda_snapshot_dtype_size_bytes(const PddClustersConfig &clusters);

// Build the exact static PP memory view for one logical (replica, DP) target.
// The returned profiles contain one entry per physical pipeline stage and one
// rank-local KV/KDA byte value per decode-context-parallel shard.  CPU/PDD
// aggregate transfer helpers intentionally remain separate from this view.
[[nodiscard]] std::vector<PipelineStageMemoryProfile>
build_pipeline_stage_memory_profiles(const ClusterRuntimeConfig &cluster);

// Build the canonical execution signature for one physical PP stage. Both
// configuration/metrics grouping and predictor caching must use this helper.
[[nodiscard]] StageTimingSignature
build_pipeline_stage_timing_signature(const ModelConfig &model,
                                      const ParallelismConfig &parallelism,
                                      std::uint64_t stage);

// Build deterministic first-occurrence-order timing and memory groups for a
// profile set.  `profiles` must be ordered by stage id and cover every PP
// stage in `cluster.parallelism.pipeline_parallel_size`.
[[nodiscard]] PipelineStageGroupCatalogue build_pipeline_stage_group_catalogue(
    const ClusterRuntimeConfig &cluster,
    const std::vector<PipelineStageMemoryProfile> &profiles);

// Resolve the shared logical KV-block capacity from real stage/rank physical
// profiles.  Ranks with zero KV bytes do not constrain ordinary KV capacity,
// but remain part of the profile set for KDA snapshot admission.
[[nodiscard]] std::uint64_t resolve_pipeline_logical_kv_capacity(
    const std::vector<PipelineStageMemoryProfile> &profiles,
    std::uint64_t *limiting_stage = nullptr,
    std::uint64_t *limiting_rank = nullptr);

// Derive the conservative scalar logical charge for one resident KDA
// snapshot.  This includes KDA-only stages with B == 0 and checks that every
// nonzero snapshot shard fits in its stage/rank's free HBM budget.
[[nodiscard]] std::uint64_t resolve_pipeline_kda_snapshot_charge(
    const std::vector<PipelineStageMemoryProfile> &profiles,
    std::uint64_t logical_kv_capacity, std::uint64_t *limiting_stage = nullptr,
    std::uint64_t *limiting_rank = nullptr);

// Resolve per-rank model weight storage and the remaining rank-local KV block
// capacity. Every caller must bind a positive gpu_memory capacity. Manual
// configs retain their explicit scheduler.num_blocks but are checked against
// exact PP stage/rank profiles; an explicit value on an automatic config is
// treated as a normalized-config consistency check.
void resolve_gpu_memory_config(
    ClusterRuntimeConfig &cluster,
    std::optional<std::uint64_t> explicit_num_blocks = std::nullopt);

[[nodiscard]] SimulationConfig
parse_simulation_config_json(std::string_view json_text);

[[nodiscard]] ModelConfig load_model_config(std::string_view model_name);
[[nodiscard]] std::string
serialize_simulation_config_json(const SimulationConfig &config);

} // namespace frontier::config
