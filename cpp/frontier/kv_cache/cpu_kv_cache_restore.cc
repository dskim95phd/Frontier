#include "frontier/kv_cache/cpu_kv_cache_manager.h"

#include <algorithm>
#include <limits>
#include <tuple>
#include <vector>

#include "frontier/core/checked_math.h"

namespace frontier::kv_cache {

CpuRestoreLeaseId CpuKVCacheManager::pin_restore(SessionId session_id,
                                                 std::uint64_t begin_block,
                                                 std::uint64_t end_block,
                                                 SimTime started_at) {
    if (!session_id.valid() || !started_at.valid() || begin_block > end_block) {
        throw CpuKVCacheError("invalid CPU restore lease range/time");
    }
    auto position = sessions_.find(session_id);
    if (position == sessions_.end() || position->second.discard_pending ||
        end_block > position->second.committed_frontier_blocks) {
        throw CpuKVCacheError("CPU restore range is not committed");
    }
    SessionState &session = position->second;
    const auto snapshot = kda_snapshots_.find(session_id);
    const bool include_snapshot = kda_snapshot_charge_blocks_ != 0;
    if (begin_block == end_block &&
        (!include_snapshot || snapshot == kda_snapshots_.end())) {
        throw CpuKVCacheError(
            "empty CPU restore range requires a KDA snapshot");
    }
    if (include_snapshot && (snapshot == kda_snapshots_.end() ||
                             end_block > snapshot->second.frontier_blocks)) {
        throw CpuKVCacheError(
            "CPU restore range has no matching KDA snapshot frontier");
    }
    RestoreLease lease{};
    lease.id = lease_ids_.next("CPU restore lease ID space exhausted");
    lease.session_id = session_id;
    lease.begin_block = begin_block;
    lease.end_block = end_block;
    lease.started_at = started_at;
    lease.includes_kda_snapshot = include_snapshot;
    if (include_snapshot) {
        if (snapshot->second.in_lru) {
            remove_kda_snapshot_from_lru(snapshot->second);
        }
        lease.kda_snapshot_blocks = kda_snapshot_charge_blocks_;
        lease.kda_snapshot_bytes = kda_snapshot_charge_bytes_;
        ++session.aggregate_snapshot_pins;
        ++pinned_kda_snapshots_;
    }
    const std::uint64_t length = end_block - begin_block;
    if (length > std::numeric_limits<std::uint64_t>::max() -
                     session.aggregate_restore_pins) {
        throw CpuKVCacheError("CPU restore pin accounting overflows");
    }
    const CpuRestoreLeaseId lease_id = lease.id;
    if (!leases_.emplace(lease_id, std::move(lease)).second ||
        !session.active_restore_leases.insert(lease_id).second) {
        throw CpuKVCacheError("duplicate CPU restore lease");
    }
    session.aggregate_restore_pins += length;
    recompute_distinct_pinned_blocks(session);
    validate_local_invariants();
    return lease_id;
}

bool CpuKVCacheManager::release_restore(CpuRestoreLeaseId lease_id, bool used,
                                        SimTime released_at) {
    auto position = leases_.find(lease_id);
    if (position == leases_.end()) {
        if (lease_ids_.was_issued(lease_id)) {
            return false;
        }
        throw CpuKVCacheError("unknown CPU restore lease");
    }
    RestoreLease &lease = position->second;
    if (lease.released) {
        return false;
    }
    if (!released_at.valid() || released_at < lease.started_at) {
        throw CpuKVCacheError("invalid CPU restore release time");
    }
    auto session_position = sessions_.find(lease.session_id);
    if (session_position == sessions_.end()) {
        throw CpuKVCacheError("pinned CPU restore session disappeared");
    }
    SessionState &session = session_position->second;
    const std::uint64_t length = lease.end_block - lease.begin_block;
    if (length > session.aggregate_restore_pins ||
        session.active_restore_leases.erase(lease_id) != 1) {
        throw CpuKVCacheError("CPU restore pin accounting underflow");
    }
    session.aggregate_restore_pins -= length;
    recompute_distinct_pinned_blocks(session);
    if (lease.includes_kda_snapshot) {
        if (session.aggregate_snapshot_pins == 0 ||
            pinned_kda_snapshots_ == 0) {
            throw CpuKVCacheError("CPU KDA snapshot pin accounting underflow");
        }
        --session.aggregate_snapshot_pins;
        --pinned_kda_snapshots_;
        const auto snapshot = kda_snapshots_.find(lease.session_id);
        if (!session.discard_pending && snapshot != kda_snapshots_.end() &&
            session.committed_blocks == 0 && session.reserved_blocks == 0 &&
            session.active_reservations.empty() &&
            session.aggregate_restore_pins == 0 &&
            session.aggregate_snapshot_pins == 0 && !snapshot->second.in_lru) {
            append_kda_snapshot_to_lru(lease.session_id, snapshot->second);
        }
    }
    if (used && !session.discard_pending) {
        session.last_access_time = released_at;
    }
    lease.released = true;
    const SessionId session_id = lease.session_id;
    leases_.erase(position);
    maybe_reap_discarded_session(session_id);
    validate_local_invariants();
    return true;
}

bool CpuKVCacheManager::restore_includes_kda_snapshot(
    CpuRestoreLeaseId lease_id) const noexcept {
    const auto position = leases_.find(lease_id);
    return position != leases_.end() && position->second.includes_kda_snapshot;
}

std::uint64_t CpuKVCacheManager::restore_kda_snapshot_blocks(
    CpuRestoreLeaseId lease_id) const noexcept {
    const auto position = leases_.find(lease_id);
    return position == leases_.end() ? 0 : position->second.kda_snapshot_blocks;
}

std::uint64_t CpuKVCacheManager::restore_kda_snapshot_bytes(
    CpuRestoreLeaseId lease_id) const noexcept {
    const auto position = leases_.find(lease_id);
    return position == leases_.end() ? 0 : position->second.kda_snapshot_bytes;
}

std::uint64_t CpuKVCacheManager::committed_frontier_blocks(
    SessionId session_id) const noexcept {
    const auto position = sessions_.find(session_id);
    return position == sessions_.end() || position->second.discard_pending
               ? 0
               : position->second.committed_frontier_blocks;
}

bool CpuKVCacheManager::reservation_pending(
    CpuOffloadReservationId reservation_id) const noexcept {
    const auto position = reservations_.find(reservation_id);
    return position != reservations_.end() &&
           position->second.state == CpuOffloadReservationState::kPending;
}

bool CpuKVCacheManager::lease_active(
    CpuRestoreLeaseId lease_id) const noexcept {
    const auto position = leases_.find(lease_id);
    return position != leases_.end() && !position->second.released;
}

} // namespace frontier::kv_cache
