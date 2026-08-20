#include "frontier/metrics/metrics_store.h"

#include <cmath>
#include <limits>
#include <stdexcept>

#include "frontier/kv_cache/cpu_kv_cache_manager.h"
#include "frontier/kv_cache/replica_kv_cache_manager.h"
#include "frontier/scheduler/scheduler_types.h"

namespace frontier::metrics {

void MetricsStore::set_observation_window_seconds(double seconds) {
    if (!std::isfinite(seconds) || seconds <= 0.0) {
        throw std::invalid_argument(
            "observation window must be finite and positive");
    }
    output_.observation_window_seconds = seconds;
}

void MetricsStore::record_prefix_cache_target(
    const kv_cache::PrefixCacheStats &stats,
    const kv_cache::PrefixCacheDiagnostics &diagnostics,
    scheduler::ReplicaTarget target, ClusterType cluster_type,
    std::uint64_t block_size, config::PrefixCachingKeyMode key_mode) {
    PrefixCacheMetricsAggregate &aggregate = output_.aggregate.prefix_cache;
    if (aggregate.block_size != 0 && aggregate.block_size != block_size) {
        throw std::logic_error("prefix-cache targets disagree on block size");
    }
    aggregate.key_mode = key_mode;
    aggregate.block_size = block_size;
    aggregate.successful_admissions += stats.successful_admissions;
    aggregate.query_blocks += stats.query_blocks;
    aggregate.hit_blocks += stats.hit_blocks;
    aggregate.evicted_blocks += stats.evicted_blocks;
    aggregate.evicted_sessions += stats.evicted_sessions;
    aggregate.evicted_kda_snapshots += stats.evicted_kda_snapshots;
    aggregate.evicted_kda_snapshot_blocks += stats.evicted_kda_snapshot_blocks;
    aggregate.kda_snapshot_occupied_blocks +=
        diagnostics.kda_snapshot_occupied_blocks;
    aggregate.kda_snapshot_sessions += diagnostics.kda_snapshot_sessions;
    output_.prefix_cache_targets.push_back(PrefixCacheTargetMetricsRecord{
        cluster_type, target.replica_id, target.dp_id,
        diagnostics.capacity_blocks, diagnostics.available_blocks,
        diagnostics.active_blocks, diagnostics.resident_blocks,
        diagnostics.evictable_blocks, diagnostics.evictable_sessions,
        diagnostics.sessions_with_nonzero_frontier,
        diagnostics.kda_snapshot_occupied_blocks,
        diagnostics.kda_snapshot_sessions,
        diagnostics.kda_snapshot_evictable_sessions});
}

void MetricsStore::record_cpu_kv_cache_target(
    const config::ResolvedCpuKVCacheTargetConfig &config,
    const kv_cache::CpuKVCacheStats &stats,
    const kv_cache::CpuKVCacheDiagnostics &diagnostics,
    scheduler::ReplicaTarget target, ClusterType cluster_type,
    std::size_t pending_restores, std::size_t staged_restores) {
    const auto bytes = [&](std::uint64_t blocks) {
        if (blocks > std::numeric_limits<std::uint64_t>::max() /
                         config.bytes_per_block) {
            throw std::overflow_error("CPU KV-cache metric bytes overflow");
        }
        return blocks * config.bytes_per_block;
    };
    const std::uint64_t used = diagnostics.resident_blocks +
                               diagnostics.reserved_blocks +
                               diagnostics.kda_snapshot_occupied_blocks +
                               diagnostics.kda_snapshot_reserved_blocks;
    if (used > diagnostics.capacity_blocks) {
        throw std::logic_error("CPU KV-cache target exceeds capacity");
    }
    CpuKVCacheTargetMetricsRecord record{};
    record.cluster_type = cluster_type;
    record.replica_id = target.replica_id;
    record.dp_id = target.dp_id;
    record.capacity_bytes = config.capacity_bytes;
    record.capacity_blocks = diagnostics.capacity_blocks;
    record.bytes_per_block = config.bytes_per_block;
    record.kda_snapshot_bytes_per_session = config.kda_snapshot_bytes;
    record.kda_snapshot_blocks_per_session = config.kda_snapshot_blocks;
    record.kda_snapshot_occupied_blocks =
        diagnostics.kda_snapshot_occupied_blocks;
    record.kda_snapshot_reserved_blocks =
        diagnostics.kda_snapshot_reserved_blocks;
    record.kda_snapshot_sessions = diagnostics.kda_snapshot_sessions;
    record.kda_snapshot_evictable_sessions =
        diagnostics.kda_snapshot_evictable_sessions;
    const auto snapshot_bytes = [&](std::uint64_t charged_blocks) {
        if (charged_blocks == 0) {
            return std::uint64_t{0};
        }
        if (config.kda_snapshot_blocks == 0 ||
            charged_blocks % config.kda_snapshot_blocks != 0) {
            throw std::logic_error(
                "CPU KDA snapshot metric charge is not atomic");
        }
        const std::uint64_t groups =
            charged_blocks / config.kda_snapshot_blocks;
        if (groups > std::numeric_limits<std::uint64_t>::max() /
                         config.kda_snapshot_bytes) {
            throw std::overflow_error("CPU KDA snapshot metric bytes overflow");
        }
        return groups * config.kda_snapshot_bytes;
    };
    record.kda_snapshot_occupied_bytes =
        snapshot_bytes(record.kda_snapshot_occupied_blocks);
    record.kda_snapshot_reserved_bytes =
        snapshot_bytes(record.kda_snapshot_reserved_blocks);
    record.resident_blocks = diagnostics.resident_blocks;
    record.resident_bytes = bytes(record.resident_blocks);
    record.reserved_blocks = diagnostics.reserved_blocks;
    record.reserved_bytes = bytes(record.reserved_blocks);
    record.free_blocks = diagnostics.capacity_blocks - used;
    record.free_bytes = bytes(record.free_blocks);
    record.peak_resident_blocks = stats.peak_resident_blocks;
    record.peak_resident_bytes = bytes(record.peak_resident_blocks);
    record.peak_reserved_blocks = stats.peak_reserved_blocks;
    record.peak_reserved_bytes = bytes(record.peak_reserved_blocks);
    record.resident_sessions = diagnostics.sessions;
    record.evicted_sessions = stats.evicted_sessions;
    record.evicted_blocks = stats.evicted_blocks;
    record.evicted_bytes = bytes(stats.evicted_blocks);
    record.evicted_kda_snapshots = stats.evicted_kda_snapshots;
    record.evicted_kda_snapshot_blocks = stats.evicted_kda_snapshot_blocks;
    record.evicted_kda_snapshot_bytes =
        snapshot_bytes(record.evicted_kda_snapshot_blocks);
    record.skipped_offloads = stats.skipped_offloads;
    record.truncated_offloads = stats.truncated_offloads;
    record.stale_generation_completions = stats.stale_generation_completions;
    record.cpu_query_blocks = stats.query_blocks;
    record.cpu_hit_blocks = stats.hit_blocks;
    record.sessions_with_cpu_hits = stats.sessions_with_hits;
    record.pending_restore_operations = pending_restores;
    record.staged_restore_payloads = staged_restores;
    record.active_restore_leases = diagnostics.active_restore_leases;
    record.active_offload_reservations = diagnostics.active_reservations;
    output_.cpu_kv_cache_targets.push_back(record);

    CpuKVCacheMetricsAggregate &aggregate = output_.aggregate.cpu_kv_cache;
    ++aggregate.target_count;
    aggregate.capacity_bytes += record.capacity_bytes;
    aggregate.capacity_blocks += record.capacity_blocks;
    if (aggregate.bytes_per_block == 0) {
        aggregate.bytes_per_block = record.bytes_per_block;
    } else if (aggregate.bytes_per_block != record.bytes_per_block) {
        throw std::logic_error(
            "CPU KV-cache targets reported inconsistent block sizes");
    }
    if (aggregate.kda_snapshot_bytes_per_session == 0) {
        aggregate.kda_snapshot_bytes_per_session =
            record.kda_snapshot_bytes_per_session;
        aggregate.kda_snapshot_blocks_per_session =
            record.kda_snapshot_blocks_per_session;
    } else if (record.kda_snapshot_bytes_per_session != 0 &&
               (aggregate.kda_snapshot_bytes_per_session !=
                    record.kda_snapshot_bytes_per_session ||
                aggregate.kda_snapshot_blocks_per_session !=
                    record.kda_snapshot_blocks_per_session)) {
        throw std::logic_error(
            "CPU KV-cache targets reported inconsistent KDA snapshot sizes");
    }
    aggregate.kda_snapshot_occupied_bytes += record.kda_snapshot_occupied_bytes;
    aggregate.kda_snapshot_occupied_blocks +=
        record.kda_snapshot_occupied_blocks;
    aggregate.kda_snapshot_reserved_bytes += record.kda_snapshot_reserved_bytes;
    aggregate.kda_snapshot_reserved_blocks +=
        record.kda_snapshot_reserved_blocks;
    aggregate.kda_snapshot_sessions += record.kda_snapshot_sessions;
    aggregate.kda_snapshot_evictable_sessions +=
        record.kda_snapshot_evictable_sessions;
    aggregate.evicted_kda_snapshots += record.evicted_kda_snapshots;
    aggregate.evicted_kda_snapshot_blocks += record.evicted_kda_snapshot_blocks;
    aggregate.evicted_kda_snapshot_bytes += record.evicted_kda_snapshot_bytes;
    aggregate.query_blocks += record.cpu_query_blocks;
    aggregate.hit_blocks += record.cpu_hit_blocks;
    aggregate.resident_bytes += record.resident_bytes;
    aggregate.resident_blocks += record.resident_blocks;
    aggregate.reserved_bytes += record.reserved_bytes;
    aggregate.reserved_blocks += record.reserved_blocks;
    aggregate.free_bytes += record.free_bytes;
    aggregate.free_blocks += record.free_blocks;
    aggregate.peak_resident_bytes += record.peak_resident_bytes;
    aggregate.peak_resident_blocks += record.peak_resident_blocks;
    aggregate.peak_reserved_bytes += record.peak_reserved_bytes;
    aggregate.peak_reserved_blocks += record.peak_reserved_blocks;
    aggregate.resident_sessions += record.resident_sessions;
    aggregate.evicted_blocks += record.evicted_blocks;
    aggregate.evicted_sessions += record.evicted_sessions;
    aggregate.evicted_bytes += record.evicted_bytes;
    aggregate.skipped_offloads += record.skipped_offloads;
    aggregate.truncated_offloads += record.truncated_offloads;
    aggregate.stale_generation_completions +=
        record.stale_generation_completions;
    aggregate.sessions_with_cpu_hits += record.sessions_with_cpu_hits;
    aggregate.pending_restore_operations += record.pending_restore_operations;
    aggregate.staged_restore_payloads += record.staged_restore_payloads;
    aggregate.active_restore_leases += record.active_restore_leases;
    aggregate.active_offload_reservations += record.active_offload_reservations;
}

} // namespace frontier::metrics
