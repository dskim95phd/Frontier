#include "frontier/kv_cache/replica_kv_cache_manager.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "frontier/core/checked_math.h"
#include "frontier/entities/request.h"

namespace frontier::kv_cache {

void ReplicaKVCacheManager::configure_kda_snapshot(
    std::uint64_t charged_blocks) {
    if (charged_blocks > capacity_blocks_) {
        throw ReplicaKVCacheError(
            "KDA snapshot charge exceeds GPU KV cache capacity");
    }
    if (charged_blocks == kda_snapshot_charge_blocks_) {
        return;
    }
    if (!kda_snapshots_.empty()) {
        throw ReplicaKVCacheError(
            "cannot reconfigure KDA snapshot charge while snapshots exist");
    }
    kda_snapshot_charge_blocks_ = charged_blocks;
    validate_accounting();
}

void ReplicaKVCacheManager::remove_kda_snapshot_from_lru(
    KdaSnapshotEntry &entry) {
    if (!entry.in_lru) {
        throw ReplicaKVCacheError("KDA snapshot is not in its LRU");
    }
    kda_snapshot_lru_.erase(entry.lru_position);
    entry.in_lru = false;
}

void ReplicaKVCacheManager::append_kda_snapshot_to_lru(
    SessionId session_id, KdaSnapshotEntry &entry) {
    if (entry.in_lru) {
        throw ReplicaKVCacheError("KDA snapshot already appears in its LRU");
    }
    kda_snapshot_lru_.push_back(session_id);
    entry.lru_position = std::prev(kda_snapshot_lru_.end());
    entry.in_lru = true;
}

void ReplicaKVCacheManager::remove_from_reclaim_lru(SessionCacheEntry &entry) {
    if (!entry.in_reclaim_lru) {
        return;
    }
    reclaim_lru_.erase(entry.reclaim_lru_position);
    entry.in_reclaim_lru = false;
}

void ReplicaKVCacheManager::append_to_reclaim_lru(SessionId session_id,
                                                  SessionCacheEntry &entry) {
    if (entry.in_reclaim_lru) {
        throw ReplicaKVCacheError("session already appears in reclaim LRU");
    }
    reclaim_lru_.push_back(session_id);
    entry.reclaim_lru_position = std::prev(reclaim_lru_.end());
    entry.in_reclaim_lru = true;
}

void ReplicaKVCacheManager::sync_reclaim_lru(SessionId session_id) {
    const auto position = sessions_.find(session_id);
    if (position == sessions_.end()) {
        return;
    }
    SessionCacheEntry &entry = position->second;
    const auto snapshot = kda_snapshots_.find(session_id);
    const bool inactive = !entry.active_request.valid();
    const bool has_reclaimable_state =
        inactive &&
        (entry.resident_prefix_blocks > 0 ||
         (snapshot != kda_snapshots_.end() && snapshot->second.in_lru));
    if (has_reclaimable_state) {
        if (!entry.in_reclaim_lru) {
            append_to_reclaim_lru(session_id, entry);
        }
    } else {
        remove_from_reclaim_lru(entry);
    }
}

void ReplicaKVCacheManager::touch_reclaim_lru(SessionId session_id) {
    const auto position = sessions_.find(session_id);
    if (position == sessions_.end()) {
        return;
    }
    SessionCacheEntry &entry = position->second;
    remove_from_reclaim_lru(entry);
    sync_reclaim_lru(session_id);
}

std::uint64_t ReplicaKVCacheManager::reclaimable_snapshot_blocks(
    std::optional<SessionId> excluded_session) const noexcept {
    if (kda_snapshot_charge_blocks_ == 0) {
        return 0;
    }
    std::uint64_t result = 0;
    for (const SessionId session_id : kda_snapshot_lru_) {
        if (excluded_session.has_value() && session_id == *excluded_session) {
            continue;
        }
        if (result > std::numeric_limits<std::uint64_t>::max() -
                         kda_snapshot_charge_blocks_) {
            return std::numeric_limits<std::uint64_t>::max();
        }
        result += kda_snapshot_charge_blocks_;
    }
    return result;
}

std::uint64_t ReplicaKVCacheManager::reclaimable_commitment_blocks(
    std::optional<SessionId> excluded_session,
    std::uint64_t virtual_credit) const noexcept {
    // available_commitment_blocks() excludes all future-only reservations.
    // A continuation may spend only its own virtual suffix; add that suffix
    // back, then include inactive KDA groups that the unified reclaim LRU can
    // release.  Saturate on overflow so a precheck cannot wrap around.
    std::uint64_t result = available_commitment_blocks();
    if (virtual_credit > std::numeric_limits<std::uint64_t>::max() - result) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    result += virtual_credit;
    const std::uint64_t snapshots =
        reclaimable_snapshot_blocks(excluded_session);
    if (snapshots > std::numeric_limits<std::uint64_t>::max() - result) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return result + snapshots;
}

bool ReplicaKVCacheManager::can_allocate_kda_snapshot() const {
    if (kda_snapshot_charge_blocks_ == 0) {
        return false;
    }
    return reclaimable_commitment_blocks() >= kda_snapshot_charge_blocks_;
}

void ReplicaKVCacheManager::reserve_kda_snapshot_capacity(
    SessionId session_id) {
    if (kda_snapshot_charge_blocks_ == 0 ||
        kda_snapshots_.find(session_id) != kda_snapshots_.end()) {
        return;
    }
    if (!session_id.valid()) {
        throw ReplicaKVCacheError(
            "KDA snapshot reservation requires a valid session");
    }
    // Frontier zero is a capacity reservation for the initial recurrent
    // state.  It cannot produce a prefix hit, but it guarantees that the first
    // real snapshot publication only updates an already charged atomic group.
    if (!publish_kda_snapshot(session_id, 0)) {
        throw ReplicaKVCacheError(
            "KDA snapshot capacity was not reserved during admission");
    }
    // publish_kda_snapshot() is also the public materialization API and marks
    // even frontier zero as valid. Undo only that semantic bit for this
    // internal cold-capacity reservation.
    kda_snapshots_.at(session_id).published = false;
}

std::uint64_t ReplicaKVCacheManager::effective_resident_frontier_blocks(
    SessionId session_id) const noexcept {
    const auto session = sessions_.find(session_id);
    if (session == sessions_.end()) {
        return 0;
    }
    const std::uint64_t resident = session->second.resident_prefix_blocks;
    if (kda_snapshot_charge_blocks_ == 0) {
        return resident;
    }
    const auto snapshot = kda_snapshots_.find(session_id);
    return snapshot == kda_snapshots_.end()
               ? 0
               : std::min(resident, snapshot->second.frontier_blocks);
}

void ReplicaKVCacheManager::trim_resident_frontier(SessionId session_id,
                                                   std::uint64_t keep_blocks) {
    auto position = sessions_.find(session_id);
    if (position == sessions_.end()) {
        return;
    }
    SessionCacheEntry &entry = position->second;
    if (keep_blocks > entry.resident_prefix_blocks) {
        throw ReplicaKVCacheError("KDA frontier trim exceeds resident KV");
    }
    const std::uint64_t excess = entry.resident_prefix_blocks - keep_blocks;
    if (excess == 0) {
        return;
    }
    if (entry.active_request.valid()) {
        throw ReplicaKVCacheError(
            "cannot trim a resident KV frontier owned by an active request");
    }
    if (!entry.in_evictable_lru) {
        throw ReplicaKVCacheError("resident KV frontier is not evictable");
    }
    remove_from_evictable_lru(entry);
    if (excess > resident_blocks_ ||
        blank_blocks_ > std::numeric_limits<std::uint64_t>::max() - excess) {
        throw ReplicaKVCacheError("resident KV frontier accounting underflow");
    }
    entry.resident_prefix_blocks = keep_blocks;
    resident_blocks_ -= excess;
    blank_blocks_ += excess;
    stats_.evicted_blocks += excess;
    if (keep_blocks > 0) {
        append_to_evictable_lru(session_id, entry);
    }
}

void ReplicaKVCacheManager::consume_kda_snapshot_capacity() {
    const std::uint64_t charge = kda_snapshot_charge_blocks_;
    if (charge == 0 || !can_allocate_kda_snapshot()) {
        throw ReplicaKVCacheError(
            "KDA snapshot occupancy exceeds reclaimable GPU capacity");
    }
    // Snapshot publication uses the same unified session reclaim chain as an
    // ordinary allocation.  A victim's resident suffix is trimmed first;
    // when that suffix reaches zero, its inactive snapshot is removed as one
    // atomic group before the next session can be considered.  Any group
    // surplus remains blank capacity for the newly published snapshot.
    consume_available_blocks(charge);
    if (kda_snapshot_occupied_blocks_ >
        std::numeric_limits<std::uint64_t>::max() - charge) {
        throw ReplicaKVCacheError("KDA snapshot occupancy overflows uint64");
    }
    kda_snapshot_occupied_blocks_ += charge;
}

void ReplicaKVCacheManager::erase_session_if_unowned(SessionId session_id) {
    const auto position = sessions_.find(session_id);
    if (position == sessions_.end() ||
        position->second.active_request.valid() ||
        position->second.resident_prefix_blocks != 0 ||
        kda_snapshots_.find(session_id) != kda_snapshots_.end()) {
        return;
    }
    remove_from_reclaim_lru(position->second);
    sessions_.erase(position);
    discard_on_release_.erase(session_id);
}

void ReplicaKVCacheManager::remove_kda_snapshot(SessionId session_id,
                                                bool count_eviction) {
    auto position = kda_snapshots_.find(session_id);
    if (position == kda_snapshots_.end()) {
        return;
    }
    KdaSnapshotEntry &snapshot = position->second;
    // Active snapshots are pinned and therefore absent from the LRU.  The
    // explicit bit is authoritative for inactive snapshots.
    if (snapshot.in_lru) {
        remove_kda_snapshot_from_lru(snapshot);
    }
    if (kda_snapshot_occupied_blocks_ < kda_snapshot_charge_blocks_ ||
        blank_blocks_ > std::numeric_limits<std::uint64_t>::max() -
                            kda_snapshot_charge_blocks_) {
        throw ReplicaKVCacheError(
            "KDA snapshot occupancy accounting underflow");
    }
    kda_snapshot_occupied_blocks_ -= kda_snapshot_charge_blocks_;
    blank_blocks_ += kda_snapshot_charge_blocks_;
    if (count_eviction) {
        ++stats_.evicted_kda_snapshots;
        stats_.evicted_kda_snapshot_blocks += kda_snapshot_charge_blocks_;
    }
    kda_snapshots_.erase(position);
    sync_reclaim_lru(session_id);
    erase_session_if_unowned(session_id);
}

bool ReplicaKVCacheManager::publish_kda_snapshot(
    SessionId session_id, std::uint64_t frontier_blocks) {
    if (kda_snapshot_charge_blocks_ == 0 || !session_id.valid()) {
        throw ReplicaKVCacheError("KDA snapshots are not configured");
    }
    auto existing = kda_snapshots_.find(session_id);
    if (existing != kda_snapshots_.end()) {
        existing->second.frontier_blocks = frontier_blocks;
        existing->second.published = true;
        const auto session = sessions_.find(session_id);
        if (session != sessions_.end() &&
            session->second.active_request.valid()) {
            if (existing->second.in_lru) {
                remove_kda_snapshot_from_lru(existing->second);
            }
        } else {
            if (existing->second.in_lru) {
                remove_kda_snapshot_from_lru(existing->second);
            }
            append_kda_snapshot_to_lru(session_id, existing->second);
            touch_reclaim_lru(session_id);
        }
        validate_accounting();
        return true;
    }
    if (!can_allocate_kda_snapshot()) {
        return false;
    }
    consume_kda_snapshot_capacity();
    SessionCacheEntry &session = sessions_[session_id];
    KdaSnapshotEntry snapshot{};
    snapshot.frontier_blocks = frontier_blocks;
    snapshot.published = true;
    const bool active = session.active_request.valid();
    auto inserted = kda_snapshots_.emplace(session_id, std::move(snapshot));
    if (!inserted.second) {
        throw ReplicaKVCacheError("KDA snapshot replacement raced allocation");
    }
    if (!active) {
        append_kda_snapshot_to_lru(session_id, inserted.first->second);
        touch_reclaim_lru(session_id);
    }
    validate_accounting();
    return true;
}

bool ReplicaKVCacheManager::has_kda_snapshot(
    SessionId session_id) const noexcept {
    return kda_snapshot_charge_blocks_ != 0 &&
           kda_snapshots_.find(session_id) != kda_snapshots_.end();
}

std::uint64_t ReplicaKVCacheManager::kda_snapshot_frontier_blocks(
    SessionId session_id) const noexcept {
    const auto position = kda_snapshots_.find(session_id);
    return position == kda_snapshots_.end() ? 0
                                            : position->second.frontier_blocks;
}

KdaSnapshotCheckpoint ReplicaKVCacheManager::checkpoint_kda_snapshot(
    SessionId session_id) const noexcept {
    const auto position = kda_snapshots_.find(session_id);
    if (position == kda_snapshots_.end()) {
        return {};
    }
    return KdaSnapshotCheckpoint{true, position->second.frontier_blocks,
                                 position->second.published};
}

void ReplicaKVCacheManager::restore_kda_snapshot(
    SessionId session_id, KdaSnapshotCheckpoint checkpoint) {
    auto current = kda_snapshots_.find(session_id);
    if (!checkpoint.present) {
        if (current != kda_snapshots_.end()) {
            remove_kda_snapshot(session_id, false);
        }
        validate_accounting();
        return;
    }
    if (current == kda_snapshots_.end()) {
        throw ReplicaKVCacheError(
            "cannot restore a missing GPU KDA snapshot checkpoint");
    }
    current->second.frontier_blocks = checkpoint.frontier_blocks;
    current->second.published = checkpoint.published;
    validate_accounting();
}

std::uint64_t
ReplicaKVCacheManager::discard_kda_snapshot(SessionId session_id) {
    if (!has_kda_snapshot(session_id)) {
        return 0;
    }
    remove_kda_snapshot(session_id, false);
    validate_accounting();
    return kda_snapshot_charge_blocks_;
}

} // namespace frontier::kv_cache
