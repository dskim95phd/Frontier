#include "frontier/kv_cache/replica_kv_cache_manager.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "frontier/core/checked_math.h"
#include "frontier/entities/request.h"

namespace frontier::kv_cache {

void ReplicaKVCacheManager::enable_full_sequence_commitments(bool enabled) {
    if (!enabled && virtual_committed_blocks_ != 0) {
        throw ReplicaKVCacheError("cannot disable full-sequence commitments "
                                  "while reservations exist");
    }
    full_sequence_commitments_enabled_ = enabled;
    validate_accounting();
}

bool ReplicaKVCacheManager::can_commit(
    RequestId request_id, std::uint64_t full_sequence_tokens) const {
    if (!full_sequence_commitments_enabled_ || !request_id.valid() ||
        allocations_.find(request_id) != allocations_.end()) {
        return false;
    }
    const std::uint64_t blocks = full_sequence_blocks(full_sequence_tokens);
    return can_commit_blocks(blocks);
}

bool ReplicaKVCacheManager::can_commit(
    RequestId request_id, SessionId session_id,
    std::uint64_t full_sequence_tokens) const {
    if (prefix_cache_enabled_ && session_id.valid()) {
        const auto session = sessions_.find(session_id);
        if (session != sessions_.end() &&
            session->second.active_request.valid() &&
            session->second.active_request != request_id) {
            return false;
        }
    }
    if (!full_sequence_commitments_enabled_ || !request_id.valid() ||
        allocations_.find(request_id) != allocations_.end()) {
        return false;
    }
    const std::uint64_t blocks = full_sequence_blocks(full_sequence_tokens);
    return can_commit_blocks(blocks, session_id.valid()
                                         ? std::optional<SessionId>{session_id}
                                         : std::nullopt);
}

void ReplicaKVCacheManager::commit_virtual(RequestId request_id,
                                           SessionId session_id,
                                           std::uint64_t full_sequence_tokens) {
    if (!full_sequence_commitments_enabled_ || !request_id.valid() ||
        allocations_.find(request_id) != allocations_.end()) {
        throw ReplicaKVCacheError("invalid full-sequence commitment request");
    }
    const std::uint64_t committed = full_sequence_blocks(full_sequence_tokens);

    // Complete every request/session validation before reserving a cold KDA
    // snapshot. Snapshot reservation is a mutating operation and may reclaim
    // inactive sessions; a later validation failure cannot be repaired via
    // release_commitment() because no request allocation exists yet.
    if (kda_snapshot_charge_blocks_ != 0 && !session_id.valid()) {
        throw ReplicaKVCacheError(
            "KDA snapshot reservation requires a valid session");
    }
    std::uint64_t resident = 0;
    if (prefix_cache_enabled_ && session_id.valid()) {
        const auto position = sessions_.find(session_id);
        if (position != sessions_.end()) {
            if (position->second.active_request.valid() &&
                position->second.active_request != request_id) {
                throw ReplicaKVCacheError(
                    "session already has an active KV commitment");
            }
            resident = position->second.resident_prefix_blocks;
        }
    }
    if (resident > committed) {
        throw ReplicaKVCacheError(
            "resident prefix exceeds full-sequence commitment");
    }
    const std::uint64_t future = committed - resident;
    if (future >
        std::numeric_limits<std::uint64_t>::max() - virtual_committed_blocks_) {
        throw ReplicaKVCacheError("virtual commitment count overflows uint64");
    }
    if (!can_commit_blocks(committed, session_id.valid()
                                          ? std::optional<SessionId>{session_id}
                                          : std::nullopt)) {
        throw ReplicaKVCacheError(
            "full-sequence commitment exceeds capacity or watermark");
    }
    reserve_kda_snapshot_capacity(session_id);

    // Pin any GPU-resident prefix before the asynchronous restore starts.  A
    // resident cache suffix is physical capacity already in the cache, so
    // converting it to an active allocation preserves the partition while
    // preventing another request from evicting it during the restore.
    if (prefix_cache_enabled_ && session_id.valid()) {
        SessionCacheEntry &entry = sessions_[session_id];
        resident = entry.resident_prefix_blocks;
        if (resident > 0 && entry.in_evictable_lru) {
            remove_from_evictable_lru(entry);
            active_blocks_ += resident;
        }
        entry.active_request = request_id;
        const auto snapshot = kda_snapshots_.find(session_id);
        if (snapshot != kda_snapshots_.end() && snapshot->second.in_lru) {
            remove_kda_snapshot_from_lru(snapshot->second);
        }
        remove_from_reclaim_lru(entry);
    }

    // A virtual commitment consumes logical capacity without materializing a
    // physical block.  If reclaimable inactive state was needed to satisfy
    // the precheck, evict it now.  consume_available_blocks may consume a
    // blank slot as part of that operation; restore that consumption because
    // the slot remains reserved by the new virtual suffix.
    const std::uint64_t before_blank = blank_blocks_;
    const std::uint64_t strict_available = available_commitment_blocks();
    const std::uint64_t watermark_required =
        future > std::numeric_limits<std::uint64_t>::max() - watermark_blocks_
            ? std::numeric_limits<std::uint64_t>::max()
            : future + watermark_blocks_;
    if (watermark_required > strict_available) {
        const std::uint64_t deficit = watermark_required - strict_available;
        consume_available_blocks(deficit);
        if (blank_blocks_ < before_blank) {
            blank_blocks_ += before_blank - blank_blocks_;
        }
    }

    RequestKVAllocation allocation{};
    allocation.session_id = session_id;
    allocation.allocated_blocks = resident;
    allocation.published_blocks = resident;
    allocation.committed_blocks = committed;
    if (!allocations_.emplace(request_id, allocation).second) {
        throw ReplicaKVCacheError("request already owns KV commitment");
    }
    virtual_committed_blocks_ += future;
    validate_accounting();
}

void ReplicaKVCacheManager::release_commitment(RequestId request_id) {
    const auto allocation = allocations_.find(request_id);
    if (allocation == allocations_.end()) {
        return;
    }
    // If a GPU prefix was pinned while a restore was pending, free() both
    // unpins those physical blocks and releases the remaining future suffix.
    if (allocation->second.allocated_blocks > 0) {
        static_cast<void>(free(request_id));
        return;
    }
    const RequestKVAllocation value = allocation->second;
    if (value.committed_blocks < value.allocated_blocks ||
        value.committed_blocks - value.allocated_blocks >
            virtual_committed_blocks_) {
        throw ReplicaKVCacheError("invalid virtual commitment accounting");
    }
    if (prefix_cache_enabled_ && value.session_id.valid()) {
        auto session = sessions_.find(value.session_id);
        if (session != sessions_.end()) {
            if (session->second.active_request != request_id) {
                throw ReplicaKVCacheError(
                    "virtual commitment does not own its session");
            }
            const auto reserved_snapshot =
                kda_snapshots_.find(value.session_id);
            if (reserved_snapshot != kda_snapshots_.end() &&
                !reserved_snapshot->second.published) {
                // A restore/admission rollback occurred before any reusable
                // KDA frontier was published.  Release the cold reservation
                // instead of retaining a useless snapshot-only session.
                remove_kda_snapshot(value.session_id, false);
            }
            session->second.active_request = RequestId{};
            const auto snapshot = kda_snapshots_.find(value.session_id);
            if (snapshot != kda_snapshots_.end()) {
                if (snapshot->second.in_lru) {
                    remove_kda_snapshot_from_lru(snapshot->second);
                }
                append_kda_snapshot_to_lru(value.session_id, snapshot->second);
            }
            if (session->second.resident_prefix_blocks > 0) {
                append_to_evictable_lru(value.session_id, session->second);
            }
            sync_reclaim_lru(value.session_id);
            erase_session_if_unowned(value.session_id);
        }
        discard_on_release_.erase(value.session_id);
    }
    virtual_committed_blocks_ -= value.committed_blocks;
    allocations_.erase(allocation);
    validate_accounting();
}

} // namespace frontier::kv_cache
