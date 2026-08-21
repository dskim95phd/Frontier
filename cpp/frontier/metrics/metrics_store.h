#pragma once

#include <cstddef>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "frontier/config/config.h"
#include "frontier/metrics/output_contract.h"

namespace frontier::simulator {
class EntityArena;
}

namespace frontier::entities {
class Batch;
class BatchStage;
class KVCacheTransferInfo;
class CpuKVCacheOffloadInfo;
class CpuKVCacheRestoreInfo;
class Request;
} // namespace frontier::entities

namespace frontier::execution_time_predictor {
struct MoERoutingDiagnostic;
}

namespace frontier::scheduler {
class BaseReplicaScheduler;
struct ReplicaTarget;
struct ScheduleResult;
} // namespace frontier::scheduler

namespace frontier::kv_cache {
struct PrefixCacheStats;
struct PrefixCacheDiagnostics;
struct CpuKVCacheStats;
struct CpuKVCacheDiagnostics;
} // namespace frontier::kv_cache

namespace frontier::metrics {

// Runtime metrics ownership corresponding to Python's MetricsStore.
// Event handlers report observations here; output serialization remains in
// output_contract.
class MetricsStore {
  public:
    explicit MetricsStore(const config::SimulationConfig &config,
                          bool detailed_traces_enabled = true);

    void set_detailed_traces_enabled(bool enabled) noexcept {
        detailed_traces_enabled_ = enabled;
    }
    void set_gpu_kv_occupancy_enabled(bool enabled) noexcept {
        gpu_kv_occupancy_enabled_ = enabled;
    }

    void record_event(Event event);
    void record_batch(const entities::Batch &batch,
                      const entities::RequestCollection &requests,
                      double predicted_execution_ms,
                      const config::ClusterRuntimeConfig &runtime);
    void record_batch_stage(const entities::BatchStage &batch_stage,
                            const entities::Batch &batch,
                            const config::ClusterRuntimeConfig &runtime);
    void record_scheduler_trace(const scheduler::ScheduleResult &schedule,
                                scheduler::ReplicaTarget target,
                                ClusterType cluster_type);
    void
    record_kv_cache_transfer(const entities::KVCacheTransferInfo &transfer);
    void record_prefill_completion(const entities::Request &request,
                                   scheduler::ReplicaTarget target);
    void record_analytical_diagnostic(
        std::string name, std::vector<std::pair<std::string, double>> values);
    void record_moe_routing(
        const entities::Batch &batch, StageId stage_id,
        const execution_time_predictor::MoERoutingDiagnostic &diagnostic,
        const config::ClusterRuntimeConfig &runtime);
    void update_moe_grouped_gemm_geometry(
        BatchId batch_id, StageId stage_id, LayerId layer_id,
        const execution_time_predictor::MoEGroupedGemmGeometryDiagnostic
            &geometry);
    void release_batch_diagnostics(BatchId batch_id);

    void collect_completed_requests(const config::SimulationConfig &config,
                                    const simulator::EntityArena &entities);
    void set_observation_window_seconds(double seconds);
    void record_prefix_cache_target(
        const kv_cache::PrefixCacheStats &stats,
        const kv_cache::PrefixCacheDiagnostics &diagnostics,
        scheduler::ReplicaTarget target, ClusterType cluster_type,
        std::uint64_t block_size, config::PrefixCachingKeyMode key_mode);
    void record_cpu_kv_cache_target(
        const config::ResolvedCpuKVCacheTargetConfig &config,
        const kv_cache::CpuKVCacheStats &stats,
        const kv_cache::CpuKVCacheDiagnostics &diagnostics,
        scheduler::ReplicaTarget target, ClusterType cluster_type,
        std::size_t pending_restores, std::size_t staged_restores);
    void record_cpu_kv_cache_offload(
        const entities::CpuKVCacheOffloadInfo &operation,
        ClusterType cluster_type);
    void record_cpu_kv_cache_restore(
        const entities::CpuKVCacheRestoreInfo &operation,
        ClusterType cluster_type);
    void record_gpu_kv_cache_occupancy(
        SimTime time, const scheduler::BaseReplicaScheduler &scheduler,
        std::uint64_t max_rank_bytes_per_block,
        std::uint64_t pipeline_bytes_per_block,
        std::optional<std::uint64_t> total_hbm_bytes = std::nullopt,
        bool force = false);

    [[nodiscard]] SimulationOutput take_output() noexcept;

  private:
    using OccupancyTarget = std::tuple<ClusterType, ReplicaId, DataParallelId>;

    struct MoERoutingKey {
        BatchId batch_id;
        StageId stage_id;
        LayerId layer_id;

        friend bool operator==(const MoERoutingKey &lhs,
                               const MoERoutingKey &rhs) noexcept {
            return lhs.batch_id == rhs.batch_id &&
                   lhs.stage_id == rhs.stage_id && lhs.layer_id == rhs.layer_id;
        }
    };

    struct MoERoutingKeyHash {
        [[nodiscard]] std::size_t
        operator()(const MoERoutingKey &key) const noexcept {
            std::size_t seed = StrongIdHash<BatchId>{}(key.batch_id);
            const auto combine = [&seed](std::size_t value) {
                seed ^= value + 0x9e3779b9U + (seed << 6U) + (seed >> 2U);
            };
            combine(StrongIdHash<StageId>{}(key.stage_id));
            combine(StrongIdHash<LayerId>{}(key.layer_id));
            return seed;
        }
    };

    void record_request(RequestMetricsRecord record);
    SimulationOutput output_;
    std::unordered_map<MoERoutingKey, std::size_t, MoERoutingKeyHash>
        moe_routing_positions_;
    bool detailed_traces_enabled_ = true;
    bool gpu_kv_occupancy_enabled_ = true;
    bool arrival_demand_collected_ = false;
    std::map<OccupancyTarget, std::size_t> occupancy_positions_;
};

} // namespace frontier::metrics
