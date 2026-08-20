#include "frontier/kv_cache/replica_kv_cache_manager.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "frontier/core/checked_math.h"
#include "frontier/entities/request.h"

namespace frontier::kv_cache {

ReplicaKVCacheManager::ReplicaKVCacheManager(
    const config::SchedulerConfig &scheduler_config,
    config::PrefixCacheConfig prefix_config, bool cache_owner)
    : block_size_(scheduler_config.block_size),
      watermark_blocks_(static_cast<std::uint64_t>(
          scheduler_config.watermark_blocks_fraction *
          static_cast<double>(scheduler_config.num_blocks))),
      prefix_cache_enabled_(prefix_config.enabled && cache_owner),
      capacity_blocks_(scheduler_config.num_blocks),
      blank_blocks_(scheduler_config.num_blocks) {
    if (block_size_ == 0 || capacity_blocks_ == 0) {
        throw ReplicaKVCacheError(
            "KV block size and capacity must be positive");
    }
    if (!std::isfinite(scheduler_config.watermark_blocks_fraction) ||
        scheduler_config.watermark_blocks_fraction < 0.0 ||
        scheduler_config.watermark_blocks_fraction >= 1.0) {
        throw ReplicaKVCacheError(
            "KV watermark fraction must be finite and in [0, 1)");
    }
    if (prefix_config.key_mode != config::PrefixCachingKeyMode::kSession) {
        throw ReplicaKVCacheError("unsupported prefix-cache key mode");
    }
}

std::uint64_t ReplicaKVCacheManager::ceil_div(std::uint64_t numerator,
                                              std::uint64_t denominator) {
    return checked_math::ceil_div<ReplicaKVCacheError>(
        numerator, denominator, "KV division denominator is zero");
}

std::uint64_t ReplicaKVCacheManager::full_sequence_blocks(
    std::uint64_t full_sequence_tokens) const {
    if (full_sequence_tokens == 0) {
        throw ReplicaKVCacheError(
            "full-sequence commitment requires positive token count");
    }
    return ceil_div(full_sequence_tokens, block_size_);
}

bool ReplicaKVCacheManager::can_commit_blocks(
    std::uint64_t blocks, std::optional<SessionId> excluded_session) const {
    if (!full_sequence_commitments_enabled_ || blocks == 0) {
        return false;
    }
    return required_capacity_fits(
        blocks, additional_kda_snapshot_charge(excluded_session),
        reclaimable_commitment_blocks(excluded_session));
}

std::uint64_t ReplicaKVCacheManager::additional_kda_snapshot_charge(
    std::optional<SessionId> session_id) const noexcept {
    if (kda_snapshot_charge_blocks_ == 0 || !session_id.has_value() ||
        !session_id->valid() ||
        kda_snapshots_.find(*session_id) != kda_snapshots_.end()) {
        return 0;
    }
    return kda_snapshot_charge_blocks_;
}

bool ReplicaKVCacheManager::required_capacity_fits(
    std::uint64_t blocks, std::uint64_t extra_blocks,
    std::uint64_t available_blocks) const noexcept {
    if (blocks == 0 || blocks > std::numeric_limits<std::uint64_t>::max() -
                                    watermark_blocks_) {
        return false;
    }
    const std::uint64_t with_watermark = blocks + watermark_blocks_;
    return extra_blocks <=
               std::numeric_limits<std::uint64_t>::max() - with_watermark &&
           with_watermark + extra_blocks <= available_blocks;
}

std::uint64_t ReplicaKVCacheManager::additional_blocks_required(
    RequestId request_id, std::uint64_t kv_accounted_tokens,
    std::uint64_t scheduled_tokens, bool for_materialization) const {
    if (scheduled_tokens == 0) {
        throw ReplicaKVCacheError(
            "KV reservation requires positive scheduled tokens");
    }
    if (kv_accounted_tokens >
        std::numeric_limits<std::uint64_t>::max() - scheduled_tokens) {
        throw ReplicaKVCacheError("KV token reservation overflows uint64");
    }
    const auto allocation = allocations_.find(request_id);
    if (allocation == allocations_.end()) {
        const std::uint64_t tokens =
            for_materialization ? scheduled_tokens
                                : kv_accounted_tokens + scheduled_tokens;
        return ceil_div(tokens, block_size_);
    }
    const std::uint64_t owned = allocation->second.allocated_blocks;
    if (owned > std::numeric_limits<std::uint64_t>::max() / block_size_) {
        throw ReplicaKVCacheError("KV reserved token count overflows uint64");
    }
    const std::uint64_t reserved_tokens = owned * block_size_;
    const std::uint64_t required_tokens =
        kv_accounted_tokens + scheduled_tokens;
    return required_tokens <= reserved_tokens
               ? 0
               : ceil_div(required_tokens - reserved_tokens, block_size_);
}

PrefixLookupResult
ReplicaKVCacheManager::lookup(const entities::Request &request) const {
    PrefixLookupResult result{};
    if (!prefix_cache_enabled_ || !request.session_id().valid() ||
        request.is_prefill_complete()) {
        return result;
    }
    result.query_blocks = request.num_prefill_tokens() / block_size_;
    const auto session = sessions_.find(request.session_id());
    if (session == sessions_.end() || session->second.active_request.valid()) {
        return result;
    }
    result.hit_blocks =
        std::min(result.query_blocks, session->second.resident_prefix_blocks);
    if (kda_snapshot_charge_blocks_ != 0) {
        const auto snapshot = kda_snapshots_.find(request.session_id());
        if (snapshot == kda_snapshots_.end()) {
            result.hit_blocks = 0;
        } else {
            // A newer immutable snapshot may be valid for a shorter GPU KV
            // frontier; never expose blocks beyond that KDA frontier.
            result.hit_blocks =
                std::min(result.hit_blocks, snapshot->second.frontier_blocks);
        }
    }
    result.cached_tokens = result.hit_blocks * block_size_;
    return result;
}

bool ReplicaKVCacheManager::can_reserve(
    RequestId request_id, std::uint64_t kv_accounted_tokens,
    std::uint64_t scheduled_tokens, std::uint64_t full_sequence_tokens) const {
    const std::uint64_t required = additional_blocks_required(
        request_id, kv_accounted_tokens, scheduled_tokens, false);
    const auto allocation = allocations_.find(request_id);
    const std::uint64_t virtual_credit =
        allocation != allocations_.end() &&
                allocation->second.committed_blocks >
                    allocation->second.allocated_blocks
            ? allocation->second.committed_blocks -
                  allocation->second.allocated_blocks
            : 0;
    const std::uint64_t reclaimable =
        reclaimable_commitment_blocks(std::nullopt, virtual_credit);
    if (required > reclaimable) {
        return false;
    }
    if (allocation != allocations_.end()) {
        if (full_sequence_commitments_enabled_ && full_sequence_tokens > 0 &&
            allocation->second.committed_blocks <
                full_sequence_blocks(full_sequence_tokens)) {
            return false;
        }
        return true;
    }
    if (full_sequence_commitments_enabled_ && full_sequence_tokens > 0) {
        if (!can_commit(request_id, full_sequence_tokens)) {
            return false;
        }
        // Full-sequence admission already reserves watermark headroom.  The
        // first physical chunk must consume only its own reservation.
        return true;
    }
    return reclaimable - required >= watermark_blocks_;
}

void ReplicaKVCacheManager::reserve(RequestId request_id,
                                    std::uint64_t kv_accounted_tokens,
                                    std::uint64_t scheduled_tokens,
                                    std::uint64_t full_sequence_tokens) {
    if (!can_reserve(request_id, kv_accounted_tokens, scheduled_tokens,
                     full_sequence_tokens)) {
        throw ReplicaKVCacheError(
            "KV reservation exceeds capacity or watermark");
    }
    const std::uint64_t additional = additional_blocks_required(
        request_id, kv_accounted_tokens, scheduled_tokens, true);
    auto position = allocations_.find(request_id);
    const bool inserted = position == allocations_.end();
    const std::uint64_t old_allocated =
        inserted ? 0 : position->second.allocated_blocks;
    const std::uint64_t old_committed =
        inserted ? 0 : position->second.committed_blocks;
    const bool had_virtual_suffix = !inserted && old_committed > old_allocated;
    const std::uint64_t committed = [&]() {
        if (inserted) {
            return full_sequence_commitments_enabled_ &&
                           full_sequence_tokens > 0
                       ? full_sequence_blocks(full_sequence_tokens)
                       : additional;
        }
        const std::uint64_t old = old_committed;
        if (had_virtual_suffix) {
            // A previously committed request is materializing its own
            // reserved suffix.  Do not grow the target on continuation calls.
            if (full_sequence_commitments_enabled_ &&
                full_sequence_tokens > 0) {
                return std::max(old,
                                full_sequence_blocks(full_sequence_tokens));
            }
            return old;
        }
        if (full_sequence_commitments_enabled_ && full_sequence_tokens > 0) {
            return std::max(old, full_sequence_blocks(full_sequence_tokens));
        }
        // Legacy/materialized allocations have no future suffix to preserve;
        // continuation materialization extends their logical target by the
        // newly allocated physical blocks.
        if (old > std::numeric_limits<std::uint64_t>::max() - additional) {
            throw ReplicaKVCacheError("KV commitment count overflows uint64");
        }
        return old + additional;
    }();
    if (committed < old_allocated || committed - old_allocated < additional) {
        throw ReplicaKVCacheError(
            "KV materialization exceeds full-sequence commitment");
    }
    const std::uint64_t future_before = committed - old_allocated;
    const std::uint64_t future_after = future_before - additional;
    if (inserted) {
        if (future_after > std::numeric_limits<std::uint64_t>::max() -
                               virtual_committed_blocks_) {
            throw ReplicaKVCacheError(
                "virtual commitment count overflows uint64");
        }
    } else if (had_virtual_suffix && additional > virtual_committed_blocks_) {
        throw ReplicaKVCacheError(
            "virtual commitment underflows on materialization");
    }
    consume_available_blocks(
        additional, had_virtual_suffix ? old_committed - old_allocated : 0);
    active_blocks_ += additional;
    if (inserted) {
        const RequestKVAllocation allocation{SessionId{}, additional, 0,
                                             committed};
        position = allocations_.emplace(request_id, allocation).first;
        virtual_committed_blocks_ += future_after;
    } else {
        position->second.allocated_blocks += additional;
        position->second.committed_blocks = committed;
        if (had_virtual_suffix) {
            virtual_committed_blocks_ -= additional;
        }
    }
    validate_accounting();
}

std::uint64_t ReplicaKVCacheManager::free(RequestId request_id) {
    const auto allocation = allocations_.find(request_id);
    if (allocation == allocations_.end()) {
        return 0;
    }
    const RequestKVAllocation value = allocation->second;
    if (value.allocated_blocks > active_blocks_ ||
        value.published_blocks > value.allocated_blocks ||
        value.committed_blocks < value.allocated_blocks ||
        value.committed_blocks - value.allocated_blocks >
            virtual_committed_blocks_) {
        throw ReplicaKVCacheError("request allocation accounting is corrupt");
    }
    active_blocks_ -= value.allocated_blocks;
    virtual_committed_blocks_ -=
        value.committed_blocks - value.allocated_blocks;
    if (prefix_cache_enabled_ && value.session_id.valid()) {
        auto session = sessions_.find(value.session_id);
        if (session == sessions_.end() ||
            session->second.active_request != request_id ||
            session->second.in_evictable_lru) {
            throw ReplicaKVCacheError("active session allocation is corrupt");
        }
        SessionCacheEntry &entry = session->second;
        const bool discard = discard_on_release_.erase(value.session_id) != 0;
        if (discard) {
            // Explicit migration discard retires both ordinary GPU KV and the
            // pinned KDA snapshot before dropping session metadata.
            remove_kda_snapshot(value.session_id, false);
            if (entry.resident_prefix_blocks > resident_blocks_) {
                throw ReplicaKVCacheError(
                    "discarded session resident accounting is corrupt");
            }
            resident_blocks_ -= entry.resident_prefix_blocks;
            if (entry.resident_prefix_blocks > 0) {
                --sessions_with_nonzero_frontier_;
                stats_.evicted_blocks += entry.resident_prefix_blocks;
                ++stats_.evicted_sessions;
            }
            blank_blocks_ += value.allocated_blocks;
            remove_from_reclaim_lru(entry);
            sessions_.erase(session);
            allocations_.erase(allocation);
            validate_accounting();
            return value.allocated_blocks;
        }
        entry.resident_prefix_blocks = value.published_blocks;
        const auto reserved_snapshot = kda_snapshots_.find(value.session_id);
        if (reserved_snapshot != kda_snapshots_.end() &&
            !reserved_snapshot->second.published) {
            remove_kda_snapshot(value.session_id, false);
        }
        entry.active_request = RequestId{};
        blank_blocks_ += value.allocated_blocks - value.published_blocks;
        const auto snapshot = kda_snapshots_.find(value.session_id);
        if (snapshot != kda_snapshots_.end()) {
            if (snapshot->second.in_lru) {
                remove_kda_snapshot_from_lru(snapshot->second);
            }
            append_kda_snapshot_to_lru(value.session_id, snapshot->second);
        }
        if (entry.resident_prefix_blocks > 0) {
            append_to_evictable_lru(value.session_id, entry);
        }
        sync_reclaim_lru(value.session_id);
        erase_session_if_unowned(value.session_id);
    } else {
        blank_blocks_ += value.allocated_blocks;
    }
    allocations_.erase(allocation);
    validate_accounting();
    return value.allocated_blocks;
}

std::uint64_t ReplicaKVCacheManager::discard_session(SessionId session_id) {
    if (!prefix_cache_enabled_ || !session_id.valid()) {
        return 0;
    }
    auto position = sessions_.find(session_id);
    if (position == sessions_.end()) {
        discard_on_release_.erase(session_id);
        return 0;
    }
    SessionCacheEntry &entry = position->second;
    if (entry.active_request.valid()) {
        discard_on_release_.insert(session_id);
        return 0;
    }
    if (entry.resident_prefix_blocks == 0 && !has_kda_snapshot(session_id)) {
        throw ReplicaKVCacheError(
            "inactive discarded session is not evictable");
    }
    const std::uint64_t discarded = entry.resident_prefix_blocks;
    if (entry.resident_prefix_blocks > 0) {
        if (!entry.in_evictable_lru) {
            throw ReplicaKVCacheError(
                "inactive discarded session is not evictable");
        }
        remove_from_evictable_lru(entry);
    }
    if (discarded > resident_blocks_) {
        throw ReplicaKVCacheError(
            "discarded session exceeds resident accounting");
    }
    resident_blocks_ -= discarded;
    entry.resident_prefix_blocks = 0;
    blank_blocks_ += discarded;
    if (discarded > 0) {
        --sessions_with_nonzero_frontier_;
        stats_.evicted_blocks += discarded;
        ++stats_.evicted_sessions;
    }
    remove_kda_snapshot(session_id, false);
    erase_session_if_unowned(session_id);
    discard_on_release_.erase(session_id);
    validate_accounting();
    return discarded;
}

void ReplicaKVCacheManager::mark_blocks_computed(
    const entities::Request &request) {
    if (!prefix_cache_enabled_ || !request.session_id().valid()) {
        return;
    }
    const auto allocation = allocations_.find(request.id());
    if (allocation == allocations_.end()) {
        throw ReplicaKVCacheError(
            "computed request has no analytical KV allocation");
    }
    RequestKVAllocation &owned = allocation->second;
    auto session = sessions_.find(request.session_id());
    if (session == sessions_.end() ||
        session->second.active_request != request.id()) {
        throw ReplicaKVCacheError("computed request does not own its session");
    }
    const std::uint64_t complete_blocks = std::min(
        request.num_processed_tokens() / block_size_, owned.allocated_blocks);
    if (complete_blocks < owned.published_blocks) {
        throw ReplicaKVCacheError(
            "computed KV publication frontier moved backwards");
    }
    if (complete_blocks > session->second.resident_prefix_blocks) {
        if (session->second.resident_prefix_blocks == 0) {
            ++sessions_with_nonzero_frontier_;
        }
        resident_blocks_ +=
            complete_blocks - session->second.resident_prefix_blocks;
    }
    owned.published_blocks = complete_blocks;
    session->second.resident_prefix_blocks = complete_blocks;
    validate_accounting();
}

std::uint64_t
ReplicaKVCacheManager::allocated_blocks(RequestId request_id) const noexcept {
    const auto allocation = allocations_.find(request_id);
    return allocation == allocations_.end()
               ? 0
               : allocation->second.allocated_blocks;
}

std::uint64_t ReplicaKVCacheManager::gpu_cache_valid_prefix_blocks(
    SessionId session_id) const noexcept {
    const auto session = sessions_.find(session_id);
    if (session == sessions_.end()) {
        return 0;
    }
    std::uint64_t resident = session->second.resident_prefix_blocks;
    if (kda_snapshot_charge_blocks_ != 0) {
        const auto snapshot = kda_snapshots_.find(session_id);
        if (snapshot == kda_snapshots_.end()) {
            return 0;
        }
        resident = std::min(resident, snapshot->second.frontier_blocks);
    }
    return resident;
}

bool ReplicaKVCacheManager::session_has_active_request(
    SessionId session_id) const noexcept {
    if (!prefix_cache_enabled_ || !session_id.valid()) {
        return false;
    }
    const auto session = sessions_.find(session_id);
    return session != sessions_.end() && session->second.active_request.valid();
}

std::uint64_t ReplicaKVCacheManager::request_committed_blocks(
    RequestId request_id) const noexcept {
    const auto allocation = allocations_.find(request_id);
    return allocation == allocations_.end()
               ? 0
               : allocation->second.committed_blocks;
}

std::uint64_t ReplicaKVCacheManager::request_virtual_committed_blocks(
    RequestId request_id) const noexcept {
    const auto allocation = allocations_.find(request_id);
    if (allocation == allocations_.end() ||
        allocation->second.committed_blocks <
            allocation->second.allocated_blocks) {
        return 0;
    }
    return allocation->second.committed_blocks -
           allocation->second.allocated_blocks;
}

bool ReplicaKVCacheManager::full_sequence_fits_empty(
    std::uint64_t full_sequence_tokens) const {
    if (full_sequence_tokens == 0) {
        return false;
    }
    const std::uint64_t blocks = full_sequence_blocks(full_sequence_tokens);
    return required_capacity_fits(blocks, kda_snapshot_charge_blocks_,
                                  capacity_blocks_);
}

void ReplicaKVCacheManager::validate_accounting() const {
    if (kda_snapshot_charge_blocks_ > capacity_blocks_ ||
        kda_snapshot_occupied_blocks_ > capacity_blocks_ ||
        active_blocks_ > capacity_blocks_ - kda_snapshot_occupied_blocks_ ||
        virtual_committed_blocks_ >
            capacity_blocks_ - active_blocks_ - kda_snapshot_occupied_blocks_ ||
        blank_blocks_ >
            capacity_blocks_ - active_blocks_ - kda_snapshot_occupied_blocks_ ||
        evictable_blocks_ != capacity_blocks_ - active_blocks_ -
                                 kda_snapshot_occupied_blocks_ -
                                 blank_blocks_ ||
        resident_blocks_ > capacity_blocks_) {
        throw ReplicaKVCacheError(
            "analytical KV capacity partition is invalid");
    }
#ifndef NDEBUG
    std::uint64_t observed_active = 0;
    std::uint64_t observed_virtual = 0;
    for (const auto &[request_id, allocation] : allocations_) {
        static_cast<void>(request_id);
        if (allocation.committed_blocks < allocation.allocated_blocks ||
            allocation.published_blocks > allocation.allocated_blocks) {
            throw ReplicaKVCacheError(
                "request commitment is smaller than physical allocation");
        }
        if (observed_active > std::numeric_limits<std::uint64_t>::max() -
                                  allocation.allocated_blocks) {
            throw ReplicaKVCacheError("active allocation sum overflows uint64");
        }
        observed_active += allocation.allocated_blocks;
        const std::uint64_t future =
            allocation.committed_blocks - allocation.allocated_blocks;
        if (observed_virtual >
            std::numeric_limits<std::uint64_t>::max() - future) {
            throw ReplicaKVCacheError(
                "virtual commitment sum overflows uint64");
        }
        observed_virtual += future;
    }
    if (observed_active != active_blocks_ ||
        observed_virtual != virtual_committed_blocks_) {
        throw ReplicaKVCacheError(
            "request commitment totals diverged from KV accounting");
    }
    std::uint64_t observed_evictable = 0;
    std::uint64_t observed_resident = 0;
    std::uint64_t observed_nonzero_sessions = 0;
    std::size_t observed_sessions = 0;
    for (const auto &[session_id, entry] : sessions_) {
        static_cast<void>(session_id);
        if (entry.in_evictable_lru) {
            if (entry.active_request.valid() ||
                entry.resident_prefix_blocks == 0) {
                throw ReplicaKVCacheError(
                    "evictable session has invalid ownership");
            }
            observed_evictable += entry.resident_prefix_blocks;
            ++observed_sessions;
        } else if (!entry.active_request.valid() &&
                   kda_snapshots_.find(session_id) == kda_snapshots_.end()) {
            throw ReplicaKVCacheError(
                "resident session is neither active nor evictable");
        }
        observed_resident += entry.resident_prefix_blocks;
        observed_nonzero_sessions +=
            static_cast<std::uint64_t>(entry.resident_prefix_blocks > 0);
    }
    if (observed_evictable != evictable_blocks_ ||
        observed_resident != resident_blocks_ ||
        observed_sessions != evictable_lru_.size() ||
        sessions_with_nonzero_frontier_ != observed_nonzero_sessions) {
        throw ReplicaKVCacheError("analytical session LRU accounting diverged");
    }
    std::size_t observed_reclaim_sessions = 0;
    for (const auto &[session_id, entry] : sessions_) {
        const auto snapshot = kda_snapshots_.find(session_id);
        const bool should_be_reclaimable =
            !entry.active_request.valid() &&
            (entry.resident_prefix_blocks > 0 ||
             (snapshot != kda_snapshots_.end() && snapshot->second.in_lru));
        if (entry.in_reclaim_lru != should_be_reclaimable) {
            throw ReplicaKVCacheError(
                "unified session reclaimability diverged");
        }
        if (entry.in_reclaim_lru) {
            ++observed_reclaim_sessions;
        }
    }
    for (const SessionId session_id : reclaim_lru_) {
        const auto session = sessions_.find(session_id);
        if (session == sessions_.end() || !session->second.in_reclaim_lru ||
            session->second.active_request.valid()) {
            throw ReplicaKVCacheError("unified session reclaim LRU is corrupt");
        }
        const auto snapshot = kda_snapshots_.find(session_id);
        if (session->second.resident_prefix_blocks == 0 &&
            (snapshot == kda_snapshots_.end() || !snapshot->second.in_lru)) {
            throw ReplicaKVCacheError(
                "unified session reclaim LRU has an empty victim");
        }
    }
    if (observed_reclaim_sessions != reclaim_lru_.size()) {
        throw ReplicaKVCacheError(
            "unified session reclaim accounting diverged");
    }
    if (kda_snapshot_charge_blocks_ == 0 && !kda_snapshots_.empty()) {
        throw ReplicaKVCacheError(
            "KDA snapshots exist while snapshot policy is disabled");
    }
    std::uint64_t observed_snapshot_blocks = 0;
    std::size_t observed_snapshot_lru = 0;
    for (const auto &[session_id, snapshot] : kda_snapshots_) {
        const auto session = sessions_.find(session_id);
        if (session == sessions_.end()) {
            throw ReplicaKVCacheError("KDA snapshot has no session metadata");
        }
        if (snapshot.in_lru && session->second.active_request.valid()) {
            throw ReplicaKVCacheError("active KDA snapshot must remain pinned");
        }
        if (!snapshot.in_lru && !session->second.active_request.valid()) {
            throw ReplicaKVCacheError(
                "inactive KDA snapshot is missing from its LRU");
        }
        if (observed_snapshot_blocks >
            std::numeric_limits<std::uint64_t>::max() -
                kda_snapshot_charge_blocks_) {
            throw ReplicaKVCacheError(
                "KDA snapshot occupancy overflows uint64");
        }
        observed_snapshot_blocks += kda_snapshot_charge_blocks_;
    }
    for (const SessionId session_id : kda_snapshot_lru_) {
        const auto snapshot = kda_snapshots_.find(session_id);
        if (snapshot == kda_snapshots_.end() || !snapshot->second.in_lru ||
            snapshot->second.lru_position == kda_snapshot_lru_.end()) {
            throw ReplicaKVCacheError("KDA snapshot LRU is corrupt");
        }
        ++observed_snapshot_lru;
    }
    if (observed_snapshot_blocks != kda_snapshot_occupied_blocks_ ||
        observed_snapshot_lru != kda_snapshot_lru_.size()) {
        throw ReplicaKVCacheError("KDA snapshot accounting diverged");
    }
    for (const SessionId session_id : discard_on_release_) {
        const auto position = sessions_.find(session_id);
        if (position == sessions_.end() ||
            !position->second.active_request.valid()) {
            throw ReplicaKVCacheError(
                "deferred session discard has no active owner");
        }
    }
#endif
}

PrefixCacheDiagnostics ReplicaKVCacheManager::diagnostics() const {
    validate_accounting();
    PrefixCacheDiagnostics result{};
    result.capacity_blocks = capacity_blocks_;
    result.available_blocks = available_blocks();
    result.active_blocks = active_blocks_;
    result.evictable_blocks = evictable_blocks_;
    result.evictable_sessions =
        static_cast<std::uint64_t>(evictable_lru_.size());
    result.resident_blocks = resident_blocks_;
    result.sessions_with_nonzero_frontier = sessions_with_nonzero_frontier_;
    result.committed_blocks = committed_blocks();
    result.virtual_committed_blocks = virtual_committed_blocks_;
    result.available_commitment_blocks = available_commitment_blocks();
    result.kda_snapshot_occupied_blocks = kda_snapshot_occupied_blocks_;
    result.kda_snapshot_sessions =
        static_cast<std::uint64_t>(kda_snapshots_.size());
    result.kda_snapshot_evictable_sessions =
        static_cast<std::uint64_t>(kda_snapshot_lru_.size());
    return result;
}

} // namespace frontier::kv_cache
