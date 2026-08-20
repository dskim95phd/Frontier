#include "frontier/kv_cache/replica_kv_cache_manager.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "frontier/core/checked_math.h"
#include "frontier/entities/request.h"

namespace frontier::kv_cache {

bool ReplicaKVCacheManager::can_admit(
    RequestId request_id, SessionId session_id, std::uint64_t cached_tokens,
    std::uint64_t scheduled_tokens, std::uint64_t full_sequence_tokens) const {
    if (!prefix_cache_enabled_ || !session_id.valid() ||
        allocations_.find(request_id) != allocations_.end() ||
        cached_tokens % block_size_ != 0 || scheduled_tokens == 0 ||
        cached_tokens >
            std::numeric_limits<std::uint64_t>::max() - scheduled_tokens) {
        return false;
    }
    const auto position = sessions_.find(session_id);
    const std::uint64_t resident =
        position == sessions_.end() ? 0
                                    : position->second.resident_prefix_blocks;
    const std::uint64_t effective_resident =
        kda_snapshot_charge_blocks_ == 0
            ? resident
            : effective_resident_frontier_blocks(session_id);
    if (position != sessions_.end() &&
        position->second.active_request.valid() &&
        position->second.active_request != request_id) {
        return false;
    }
    const std::uint64_t cached_blocks = cached_tokens / block_size_;
    const std::uint64_t required =
        ceil_div(cached_tokens + scheduled_tokens, block_size_);
    const std::uint64_t snapshot_charge =
        additional_kda_snapshot_charge(std::optional<SessionId>{session_id});
    if (cached_blocks > effective_resident || required < effective_resident ||
        !required_capacity_fits(required, snapshot_charge,
                                reclaimable_commitment_blocks(session_id))) {
        return false;
    }
    if (full_sequence_commitments_enabled_ && full_sequence_tokens > 0) {
        const std::uint64_t committed =
            full_sequence_blocks(full_sequence_tokens);
        if (committed < required || !can_commit_blocks(committed, session_id)) {
            return false;
        }
        // The full-sequence commitment already accounts for watermark
        // headroom.  Do not apply it a second time to this first chunk.
        return true;
    }
    return true;
}

void ReplicaKVCacheManager::remove_from_evictable_lru(
    SessionCacheEntry &entry) {
    if (!entry.in_evictable_lru || entry.active_request.valid() ||
        entry.resident_prefix_blocks == 0 ||
        entry.resident_prefix_blocks > evictable_blocks_) {
        throw ReplicaKVCacheError("invalid evictable-session removal");
    }
    evictable_blocks_ -= entry.resident_prefix_blocks;
    evictable_lru_.erase(entry.lru_position);
    entry.in_evictable_lru = false;
}

void ReplicaKVCacheManager::append_to_evictable_lru(SessionId session_id,
                                                    SessionCacheEntry &entry) {
    if (entry.in_evictable_lru || entry.active_request.valid() ||
        entry.resident_prefix_blocks == 0 ||
        entry.resident_prefix_blocks >
            std::numeric_limits<std::uint64_t>::max() - evictable_blocks_) {
        throw ReplicaKVCacheError("invalid evictable-session insertion");
    }
    evictable_lru_.push_back(session_id);
    entry.lru_position = std::prev(evictable_lru_.end());
    entry.in_evictable_lru = true;
    evictable_blocks_ += entry.resident_prefix_blocks;
}

void ReplicaKVCacheManager::consume_available_blocks(
    std::uint64_t blocks, std::uint64_t virtual_credit) {
    const std::uint64_t reclaimable =
        reclaimable_commitment_blocks(std::nullopt, virtual_credit);
    if (blocks > reclaimable) {
        throw ReplicaKVCacheError("analytical KV cache is exhausted");
    }

    // Future-only commitments are overlaid on blank/ordinary capacity.  A
    // continuation may spend its own virtual suffix (virtual_credit), while
    // all other virtual reservations remain protected.  Consume the usable
    // blank portion first; the unified LRU below then reclaims session state.
    const std::uint64_t ordinary_budget =
        available_commitment_blocks() >
                std::numeric_limits<std::uint64_t>::max() - virtual_credit
            ? std::numeric_limits<std::uint64_t>::max()
            : available_commitment_blocks() + virtual_credit;
    const std::uint64_t blank =
        std::min(blocks, std::min(blank_blocks_, ordinary_budget));
    blank_blocks_ -= blank;
    blocks -= blank;

    while (blocks > 0) {
        if (reclaim_lru_.empty()) {
            throw ReplicaKVCacheError(
                "reclaimable accounting has no evictable session");
        }
        const SessionId victim_id = reclaim_lru_.front();
        auto victim = sessions_.find(victim_id);
        if (victim == sessions_.end() || !victim->second.in_reclaim_lru ||
            victim->second.active_request.valid() ||
            (victim->second.resident_prefix_blocks == 0 &&
             kda_snapshots_.find(victim_id) == kda_snapshots_.end())) {
            throw ReplicaKVCacheError("unified reclaim LRU is corrupt");
        }

        SessionCacheEntry &entry = victim->second;
        if (entry.resident_prefix_blocks > 0) {
            // Ordinary KV is always the first leg of a session's reclaim
            // chain.  Partial suffix eviction leaves this session at the
            // unified LRU front so subsequent pressure continues the same
            // victim before considering a newer session.
            const std::uint64_t reclaimed =
                std::min(blocks, entry.resident_prefix_blocks);
            entry.resident_prefix_blocks -= reclaimed;
            evictable_blocks_ -= reclaimed;
            resident_blocks_ -= reclaimed;
            stats_.evicted_blocks += reclaimed;
            blocks -= reclaimed;
            if (entry.resident_prefix_blocks > 0) {
                continue;
            }

            // The ordinary LRU node is now empty.  Remove it before touching
            // the snapshot so all diagnostic counters remain synchronized.
            evictable_lru_.erase(entry.lru_position);
            entry.in_evictable_lru = false;
            --sessions_with_nonzero_frontier_;
            ++stats_.evicted_sessions;

            // Once the resident suffix reaches zero, evict this same
            // session's inactive snapshot atomically.  A larger snapshot
            // charge may leave surplus blank capacity after satisfying the
            // current request; never advance to the next session for that
            // surplus.
            const bool has_snapshot =
                kda_snapshots_.find(victim_id) != kda_snapshots_.end();
            if (has_snapshot && blocks > 0) {
                const std::uint64_t charge = kda_snapshot_charge_blocks_;
                remove_kda_snapshot(victim_id, true);
                const std::uint64_t used = std::min(blocks, charge);
                if (used > blank_blocks_) {
                    throw ReplicaKVCacheError(
                        "KDA snapshot reclaim accounting underflow");
                }
                blank_blocks_ -= used;
                blocks -= used;
            } else {
                // Exact-fit ordinary reclaim leaves an inactive snapshot
                // snapshot-only victim at the unified LRU front.  It is
                // retired on the next pressure event, before any newer
                // session's ordinary KV is considered.
                sync_reclaim_lru(victim_id);
                if (!has_snapshot) {
                    erase_session_if_unowned(victim_id);
                }
            }
            continue;
        }

        // Snapshot-only sessions are still full reclaim victims and must be
        // selected by the same unified recency order as ordinary sessions.
        const std::uint64_t charge = kda_snapshot_charge_blocks_;
        if (charge == 0 || !has_kda_snapshot(victim_id)) {
            throw ReplicaKVCacheError("snapshot-only reclaim LRU is corrupt");
        }
        remove_kda_snapshot(victim_id, true);
        const std::uint64_t used = std::min(blocks, charge);
        if (used > blank_blocks_) {
            throw ReplicaKVCacheError(
                "KDA snapshot reclaim accounting underflow");
        }
        blank_blocks_ -= used;
        blocks -= used;
    }
}

void ReplicaKVCacheManager::admit(RequestId request_id, SessionId session_id,
                                  std::uint64_t cached_tokens,
                                  std::uint64_t scheduled_tokens,
                                  std::uint64_t full_sequence_tokens) {
    if (!can_admit(request_id, session_id, cached_tokens, scheduled_tokens,
                   full_sequence_tokens)) {
        throw ReplicaKVCacheError(
            "KV prefix admission exceeds capacity or watermark");
    }
    reserve_kda_snapshot_capacity(session_id);
    SessionCacheEntry &session = sessions_[session_id];
    const std::uint64_t resident = session.resident_prefix_blocks;
    const std::uint64_t effective_resident =
        kda_snapshot_charge_blocks_ == 0
            ? resident
            : effective_resident_frontier_blocks(session_id);
    if (resident > effective_resident) {
        trim_resident_frontier(session_id, effective_resident);
    }
    const std::uint64_t reusable_resident = effective_resident;
    if (reusable_resident > 0) {
        remove_from_evictable_lru(session);
    }
    const auto snapshot = kda_snapshots_.find(session_id);
    if (snapshot != kda_snapshots_.end() && snapshot->second.in_lru) {
        remove_kda_snapshot_from_lru(snapshot->second);
    }
    // Pin the target session before reclaiming capacity so the unified LRU
    // cannot evict its own resident range or snapshot while materializing.
    session.active_request = request_id;
    remove_from_reclaim_lru(session);
    const std::uint64_t required =
        ceil_div(cached_tokens + scheduled_tokens, block_size_);
    const std::uint64_t committed =
        full_sequence_commitments_enabled_ && full_sequence_tokens > 0
            ? full_sequence_blocks(full_sequence_tokens)
            : required;
    if (committed < required) {
        throw ReplicaKVCacheError(
            "full-sequence commitment is smaller than materialization");
    }
    const std::uint64_t future = committed - required;
    if (future >
        std::numeric_limits<std::uint64_t>::max() - virtual_committed_blocks_) {
        throw ReplicaKVCacheError("invalid prefix commitment accounting");
    }
    consume_available_blocks(required - reusable_resident);
    active_blocks_ += required;
    if (future > 0) {
        virtual_committed_blocks_ += future;
    }
    if (!allocations_
             .emplace(request_id,
                      RequestKVAllocation{session_id, required,
                                          reusable_resident, committed})
             .second) {
        throw ReplicaKVCacheError("request already owns KV blocks");
    }
    validate_accounting();
}

bool ReplicaKVCacheManager::can_admit_tiered(
    RequestId request_id, SessionId session_id,
    std::uint64_t reusable_frontier_blocks, std::uint64_t scheduled_tokens,
    std::uint64_t full_sequence_tokens) const {
    if (!prefix_cache_enabled_ || !session_id.valid() ||
        scheduled_tokens == 0 ||
        reusable_frontier_blocks >
            std::numeric_limits<std::uint64_t>::max() / block_size_) {
        return false;
    }
    const auto allocation = allocations_.find(request_id);
    const bool has_virtual_commitment = allocation != allocations_.end() &&
                                        full_sequence_commitments_enabled_ &&
                                        allocation->second.committed_blocks >
                                            allocation->second.allocated_blocks;
    if (allocation != allocations_.end() && !has_virtual_commitment) {
        return false;
    }
    const auto position = sessions_.find(session_id);
    const std::uint64_t resident =
        position == sessions_.end() ? 0
                                    : position->second.resident_prefix_blocks;
    const std::uint64_t effective_resident =
        kda_snapshot_charge_blocks_ == 0
            ? resident
            : effective_resident_frontier_blocks(session_id);
    if (position != sessions_.end() &&
        position->second.active_request.valid() &&
        position->second.active_request != request_id) {
        return false;
    }
    if (reusable_frontier_blocks < effective_resident) {
        return false;
    }
    const std::uint64_t reusable_tokens =
        reusable_frontier_blocks * block_size_;
    if (reusable_tokens >
        std::numeric_limits<std::uint64_t>::max() - scheduled_tokens) {
        return false;
    }
    const std::uint64_t required =
        ceil_div(reusable_tokens + scheduled_tokens, block_size_);
    const std::uint64_t allocated =
        has_virtual_commitment ? allocation->second.allocated_blocks : 0;
    const std::uint64_t virtual_credit =
        has_virtual_commitment ? allocation->second.committed_blocks - allocated
                               : 0;
    const std::uint64_t reclaimable =
        reclaimable_commitment_blocks(session_id, virtual_credit);
    const std::uint64_t snapshot_charge =
        additional_kda_snapshot_charge(std::optional<SessionId>{session_id});
    if (required < reusable_frontier_blocks || required < effective_resident ||
        required < allocated) {
        return false;
    }
    if (has_virtual_commitment) {
        if (snapshot_charge != 0 || required - allocated > reclaimable ||
            allocation->second.committed_blocks < required) {
            return false;
        }
        return true;
    }
    if (!required_capacity_fits(required - allocated, snapshot_charge,
                                reclaimable)) {
        return false;
    }
    if (full_sequence_commitments_enabled_ && full_sequence_tokens > 0) {
        const std::uint64_t committed =
            full_sequence_blocks(full_sequence_tokens);
        if (committed < required || !can_commit_blocks(committed, session_id)) {
            return false;
        }
        return true;
    }
    return true;
}

void ReplicaKVCacheManager::admit_tiered(RequestId request_id,
                                         SessionId session_id,
                                         std::uint64_t reusable_frontier_blocks,
                                         std::uint64_t scheduled_tokens,
                                         std::uint64_t full_sequence_tokens) {
    if (!can_admit_tiered(request_id, session_id, reusable_frontier_blocks,
                          scheduled_tokens, full_sequence_tokens)) {
        throw ReplicaKVCacheError(
            "tiered KV admission exceeds capacity or watermark");
    }
    reserve_kda_snapshot_capacity(session_id);
    SessionCacheEntry &session = sessions_[session_id];
    const std::uint64_t resident = session.resident_prefix_blocks;
    const auto allocation = allocations_.find(request_id);
    const bool has_virtual_commitment = allocation != allocations_.end() &&
                                        full_sequence_commitments_enabled_ &&
                                        allocation->second.committed_blocks >
                                            allocation->second.allocated_blocks;
    const std::uint64_t effective_resident =
        kda_snapshot_charge_blocks_ == 0
            ? resident
            : effective_resident_frontier_blocks(session_id);
    if (resident > effective_resident) {
        if (!has_virtual_commitment) {
            trim_resident_frontier(session_id, effective_resident);
        } else {
            const std::uint64_t excess = resident - effective_resident;
            RequestKVAllocation &owned = allocation->second;
            if (owned.allocated_blocks < excess || active_blocks_ < excess ||
                resident_blocks_ < excess ||
                blank_blocks_ >
                    std::numeric_limits<std::uint64_t>::max() - excess ||
                virtual_committed_blocks_ >
                    std::numeric_limits<std::uint64_t>::max() - excess) {
                throw ReplicaKVCacheError(
                    "active KDA frontier trim accounting underflow");
            }
            owned.allocated_blocks -= excess;
            owned.published_blocks =
                std::min(owned.published_blocks, effective_resident);
            active_blocks_ -= excess;
            resident_blocks_ -= excess;
            blank_blocks_ += excess;
            virtual_committed_blocks_ += excess;
            session.resident_prefix_blocks = effective_resident;
        }
    }
    const std::uint64_t reusable_resident = effective_resident;
    const std::uint64_t already_allocated =
        has_virtual_commitment ? allocation->second.allocated_blocks : 0;
    const std::uint64_t virtual_credit =
        has_virtual_commitment
            ? allocation->second.committed_blocks - already_allocated
            : 0;
    if (reusable_resident > 0 && !has_virtual_commitment) {
        remove_from_evictable_lru(session);
    }
    const auto snapshot = kda_snapshots_.find(session_id);
    if (snapshot != kda_snapshots_.end() && snapshot->second.in_lru) {
        remove_kda_snapshot_from_lru(snapshot->second);
    }
    // Pin the target before reclaiming any physical continuation blocks.
    session.active_request = request_id;
    remove_from_reclaim_lru(session);
    const std::uint64_t reusable_tokens =
        reusable_frontier_blocks * block_size_;
    const std::uint64_t required =
        ceil_div(reusable_tokens + scheduled_tokens, block_size_);
    const std::uint64_t physical_additional = required - already_allocated;
    const std::uint64_t committed =
        full_sequence_commitments_enabled_ && full_sequence_tokens > 0
            ? full_sequence_blocks(full_sequence_tokens)
            : (has_virtual_commitment ? allocation->second.committed_blocks
                                      : required);
    if (committed < required) {
        throw ReplicaKVCacheError(
            "full-sequence commitment is smaller than materialization");
    }
    const std::uint64_t future = committed - required;
    if (has_virtual_commitment) {
        if (allocation->second.committed_blocks < already_allocated ||
            allocation->second.committed_blocks - already_allocated <
                physical_additional ||
            physical_additional > virtual_committed_blocks_) {
            throw ReplicaKVCacheError("invalid staged commitment accounting");
        }
    } else if (future > std::numeric_limits<std::uint64_t>::max() -
                            virtual_committed_blocks_) {
        throw ReplicaKVCacheError("invalid staged commitment accounting");
    }
    consume_available_blocks(physical_additional, virtual_credit);
    active_blocks_ += physical_additional;
    resident_blocks_ += reusable_frontier_blocks - reusable_resident;
    if (reusable_resident == 0 && reusable_frontier_blocks > 0) {
        ++sessions_with_nonzero_frontier_;
    }
    session.resident_prefix_blocks = reusable_frontier_blocks;
    if (has_virtual_commitment) {
        RequestKVAllocation &owned = allocation->second;
        owned.allocated_blocks = required;
        owned.published_blocks = reusable_frontier_blocks;
        if (physical_additional > virtual_committed_blocks_) {
            throw ReplicaKVCacheError("staged commitment underflow");
        }
        virtual_committed_blocks_ -= physical_additional;
    } else {
        if (!allocations_
                 .emplace(request_id,
                          RequestKVAllocation{session_id, required,
                                              reusable_frontier_blocks,
                                              committed})
                 .second) {
            throw ReplicaKVCacheError("tiered request already owns KV blocks");
        }
        virtual_committed_blocks_ += future;
    }
    validate_accounting();
}

void ReplicaKVCacheManager::record_successful_admission(
    std::uint64_t query_blocks, std::uint64_t hit_blocks) {
    if (!prefix_cache_enabled_ || hit_blocks > query_blocks) {
        throw ReplicaKVCacheError("invalid prefix-cache admission metrics");
    }
    ++stats_.successful_admissions;
    stats_.query_blocks += query_blocks;
    stats_.hit_blocks += hit_blocks;
}

} // namespace frontier::kv_cache
