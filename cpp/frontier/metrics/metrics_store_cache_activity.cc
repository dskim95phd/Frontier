#include "frontier/metrics/metrics_store.h"

#include <limits>
#include <stdexcept>
#include <utility>

#include "frontier/entities/cpu_kv_cache_transfer_info.h"
#include "frontier/kv_cache/replica_kv_cache_manager.h"
#include "frontier/scheduler/replica_scheduler/base_replica_scheduler.h"

namespace frontier::metrics {

void MetricsStore::record_cpu_kv_cache_offload(
    const entities::CpuKVCacheOffloadInfo &operation,
    ClusterType cluster_type) {
    if (operation.state() != entities::CpuKVCacheTransferState::kCompleted) {
        return;
    }
    const auto &timing = operation.timing();
    CpuKVCacheTransferMetricsRecord record{};
    record.transfer_id = operation.transfer_id();
    record.kind = CpuKVCacheTransferKind::kOffload;
    record.request_id = operation.request_id();
    record.cluster_type = cluster_type;
    record.replica_id = operation.replica_id();
    record.dp_id = operation.dp_id();
    record.blocks = operation.transferred_kv_blocks();
    record.size_bytes = timing.size_bytes;
    record.kda_snapshot_bytes = operation.kda_snapshot_bytes();
    record.submitted_at = timing.submitted_at;
    record.started_at = timing.started_at;
    record.completed_at = timing.completed_at;
    record.queue_time_ms = timing.queue_time_ms;
    record.service_time_ms = timing.service_time_ms;
    record.source_gpu_hold_ms = operation.attributable_source_hold_ms();
    auto &aggregate = output_.aggregate.cpu_kv_cache;
    ++aggregate.offload_operations;
    aggregate.offload_blocks += record.blocks;
    aggregate.offload_bytes += record.size_bytes;
    if (record.kda_snapshot_bytes != 0) {
        ++aggregate.kda_snapshot_offload_operations;
        aggregate.kda_snapshot_offload_bytes += record.kda_snapshot_bytes;
    }
    aggregate.d2h_queue_time_ms += record.queue_time_ms;
    aggregate.d2h_service_time_ms += record.service_time_ms;
    aggregate.source_gpu_hold_time_ms += record.source_gpu_hold_ms;
    if (detailed_traces_enabled_) {
        output_.cpu_kv_cache_transfers.push_back(record);
    }
}

void MetricsStore::record_cpu_kv_cache_restore(
    const entities::CpuKVCacheRestoreInfo &operation,
    ClusterType cluster_type) {
    if (operation.state() != entities::CpuKVCacheTransferState::kCompleted) {
        return;
    }
    const auto &timing = operation.timing();
    CpuKVCacheTransferMetricsRecord record{};
    record.transfer_id = operation.transfer_id();
    record.kind = CpuKVCacheTransferKind::kRestore;
    record.request_id = operation.request_id();
    record.cluster_type = cluster_type;
    record.replica_id = operation.replica_id();
    record.dp_id = operation.dp_id();
    record.blocks =
        operation.plan().cpu_end_block - operation.plan().cpu_begin_block;
    record.size_bytes = timing.size_bytes;
    record.kda_snapshot_bytes = operation.kda_snapshot_bytes();
    record.submitted_at = timing.submitted_at;
    record.started_at = timing.started_at;
    record.completed_at = timing.completed_at;
    record.queue_time_ms = timing.queue_time_ms;
    record.service_time_ms = timing.service_time_ms;
    auto &aggregate = output_.aggregate.cpu_kv_cache;
    ++aggregate.restore_operations;
    aggregate.restore_blocks += record.blocks;
    aggregate.restore_bytes += record.size_bytes;
    if (record.kda_snapshot_bytes != 0) {
        ++aggregate.kda_snapshot_restore_operations;
        aggregate.kda_snapshot_restore_bytes += record.kda_snapshot_bytes;
    }
    aggregate.h2d_queue_time_ms += record.queue_time_ms;
    aggregate.h2d_service_time_ms += record.service_time_ms;
    if (detailed_traces_enabled_) {
        output_.cpu_kv_cache_transfers.push_back(record);
    }
}

void MetricsStore::record_gpu_kv_cache_occupancy(
    SimTime time, const scheduler::BaseReplicaScheduler &scheduler,
    std::uint64_t max_rank_bytes_per_block,
    std::uint64_t pipeline_bytes_per_block,
    std::optional<std::uint64_t> total_hbm_bytes, bool force) {
    if (!gpu_kv_occupancy_enabled_) {
        return;
    }
    if (!time.valid()) {
        throw std::invalid_argument(
            "GPU KV occupancy time must be finite and nonnegative");
    }
    if (max_rank_bytes_per_block == 0 || pipeline_bytes_per_block == 0 ||
        pipeline_bytes_per_block < max_rank_bytes_per_block) {
        throw std::invalid_argument(
            "GPU KV occupancy block byte layout is invalid");
    }
    const kv_cache::PrefixCacheDiagnostics diagnostics =
        scheduler.prefix_cache_diagnostics();
    if (diagnostics.active_blocks > diagnostics.capacity_blocks) {
        throw std::logic_error("GPU KV occupancy exceeds target capacity");
    }
    if (diagnostics.active_blocks > std::numeric_limits<std::uint64_t>::max() /
                                        max_rank_bytes_per_block ||
        diagnostics.active_blocks > std::numeric_limits<std::uint64_t>::max() /
                                        pipeline_bytes_per_block) {
        throw std::overflow_error(
            "GPU KV occupancy byte count overflows uint64");
    }
    if (total_hbm_bytes.has_value() && total_hbm_bytes.value() == 0) {
        throw std::invalid_argument(
            "GPU KV occupancy total HBM bytes must be positive");
    }

    GpuKVCacheOccupancyRecord record{};
    record.time = time;
    record.cluster_type = scheduler.cluster_type();
    record.replica_id = scheduler.replica_id();
    record.dp_id = scheduler.dp_id();
    record.active_blocks = diagnostics.active_blocks;
    record.capacity_blocks = diagnostics.capacity_blocks;
    record.active_bytes_per_gpu =
        diagnostics.active_blocks * max_rank_bytes_per_block;
    record.active_bytes_across_pipeline =
        diagnostics.active_blocks * pipeline_bytes_per_block;
    record.active_fraction_of_kv_budget =
        diagnostics.capacity_blocks == 0
            ? 0.0
            : static_cast<double>(diagnostics.active_blocks) /
                  static_cast<double>(diagnostics.capacity_blocks);
    if (total_hbm_bytes.has_value()) {
        const double total_hbm_fraction =
            static_cast<double>(record.active_bytes_per_gpu) /
            static_cast<double>(total_hbm_bytes.value());
        record.hbm_fraction = total_hbm_fraction;
        record.active_fraction_of_total_hbm = total_hbm_fraction;
    }

    const OccupancyTarget key{record.cluster_type, record.replica_id,
                              record.dp_id};
    const auto previous = occupancy_positions_.find(key);
    if (previous != occupancy_positions_.end()) {
        GpuKVCacheOccupancyRecord &last =
            output_.gpu_kv_occupancy.at(previous->second);
        const bool same_state =
            last.active_blocks == record.active_blocks &&
            last.capacity_blocks == record.capacity_blocks &&
            last.active_bytes_per_gpu == record.active_bytes_per_gpu &&
            last.active_bytes_across_pipeline ==
                record.active_bytes_across_pipeline &&
            last.active_fraction_of_kv_budget ==
                record.active_fraction_of_kv_budget &&
            last.hbm_fraction == record.hbm_fraction &&
            last.active_fraction_of_total_hbm ==
                record.active_fraction_of_total_hbm;
        if (last.time == time) {
            // Multiple scheduler mutations can occur at one simulation time;
            // retain only the final target state for that timestamp.
            last = record;
            return;
        }
        if (!force && same_state) {
            return;
        }
    }
    occupancy_positions_[key] = output_.gpu_kv_occupancy.size();
    output_.gpu_kv_occupancy.push_back(std::move(record));
}

} // namespace frontier::metrics
