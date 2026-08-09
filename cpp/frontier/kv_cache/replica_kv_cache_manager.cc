#include "frontier/kv_cache/replica_kv_cache_manager.h"

#include <algorithm>
#include <cmath>
#include <limits>

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
    return can_commit(request_id, full_sequence_tokens);
}

void ReplicaKVCacheManager::commit_virtual(RequestId request_id,
                                           SessionId session_id,
                                           std::uint64_t full_sequence_tokens) {
    if (!full_sequence_commitments_enabled_ || !request_id.valid() ||
        allocations_.find(request_id) != allocations_.end()) {
        throw ReplicaKVCacheError("invalid full-sequence commitment request");
    }
    const std::uint64_t committed = full_sequence_blocks(full_sequence_tokens);
    if (!can_commit_blocks(committed)) {
        throw ReplicaKVCacheError(
            "full-sequence commitment exceeds capacity or watermark");
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
            session->second.active_request = RequestId{};
            if (session->second.resident_prefix_blocks > 0) {
                append_to_evictable_lru(value.session_id, session->second);
            } else {
                sessions_.erase(session);
            }
        }
        discard_on_release_.erase(value.session_id);
    }
    virtual_committed_blocks_ -= value.committed_blocks;
    allocations_.erase(allocation);
    validate_accounting();
}

std::uint64_t ReplicaKVCacheManager::ceil_div(std::uint64_t numerator,
                                              std::uint64_t denominator) {
    if (denominator == 0) {
        throw ReplicaKVCacheError("KV division denominator is zero");
    }
    return numerator / denominator +
           static_cast<std::uint64_t>(numerator % denominator != 0);
}

std::uint64_t ReplicaKVCacheManager::full_sequence_blocks(
    std::uint64_t full_sequence_tokens) const {
    if (full_sequence_tokens == 0) {
        throw ReplicaKVCacheError(
            "full-sequence commitment requires positive token count");
    }
    return ceil_div(full_sequence_tokens, block_size_);
}

bool ReplicaKVCacheManager::can_commit_blocks(std::uint64_t blocks) const {
    if (!full_sequence_commitments_enabled_ || blocks == 0 ||
        blocks >
            std::numeric_limits<std::uint64_t>::max() - watermark_blocks_) {
        return false;
    }
    const std::uint64_t required = blocks + watermark_blocks_;
    return required <= available_commitment_blocks();
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
    result.cached_tokens = result.hit_blocks * block_size_;
    return result;
}

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
    if (position != sessions_.end() &&
        position->second.active_request.valid() &&
        position->second.active_request != request_id) {
        return false;
    }
    const std::uint64_t cached_blocks = cached_tokens / block_size_;
    const std::uint64_t required =
        ceil_div(cached_tokens + scheduled_tokens, block_size_);
    if (cached_blocks > resident || required < resident ||
        required > available_blocks()) {
        return false;
    }
    if (full_sequence_commitments_enabled_ && full_sequence_tokens > 0) {
        const std::uint64_t committed =
            full_sequence_blocks(full_sequence_tokens);
        if (committed < required || !can_commit_blocks(committed)) {
            return false;
        }
        // The full-sequence commitment already accounts for watermark
        // headroom.  Do not apply it a second time to this first chunk.
        return true;
    }
    return available_blocks() - required >= watermark_blocks_;
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

void ReplicaKVCacheManager::consume_available_blocks(std::uint64_t blocks) {
    if (blocks > available_blocks()) {
        throw ReplicaKVCacheError("analytical KV cache is exhausted");
    }
    const std::uint64_t blank = std::min(blocks, blank_blocks_);
    blank_blocks_ -= blank;
    blocks -= blank;
    while (blocks > 0) {
        if (evictable_lru_.empty()) {
            throw ReplicaKVCacheError(
                "reclaimable accounting has no evictable session");
        }
        const SessionId victim_id = evictable_lru_.front();
        auto victim = sessions_.find(victim_id);
        if (victim == sessions_.end() || !victim->second.in_evictable_lru ||
            victim->second.active_request.valid() ||
            victim->second.resident_prefix_blocks == 0) {
            throw ReplicaKVCacheError("evictable-session LRU is corrupt");
        }
        SessionCacheEntry &entry = victim->second;
        const std::uint64_t reclaimed =
            std::min(blocks, entry.resident_prefix_blocks);
        entry.resident_prefix_blocks -= reclaimed;
        evictable_blocks_ -= reclaimed;
        resident_blocks_ -= reclaimed;
        stats_.evicted_blocks += reclaimed;
        blocks -= reclaimed;
        if (entry.resident_prefix_blocks == 0) {
            evictable_lru_.pop_front();
            entry.in_evictable_lru = false;
            sessions_.erase(victim);
            --sessions_with_nonzero_frontier_;
            ++stats_.evicted_sessions;
        }
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
    SessionCacheEntry &session = sessions_[session_id];
    const std::uint64_t resident = session.resident_prefix_blocks;
    if (resident > 0) {
        remove_from_evictable_lru(session);
    }
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
    consume_available_blocks(required - resident);
    active_blocks_ += required;
    if (future > 0) {
        virtual_committed_blocks_ += future;
    }
    session.active_request = request_id;
    if (!allocations_
             .emplace(request_id, RequestKVAllocation{session_id, required,
                                                      resident, committed})
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
    if (position != sessions_.end() &&
        position->second.active_request.valid() &&
        position->second.active_request != request_id) {
        return false;
    }
    if (reusable_frontier_blocks < resident) {
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
    if (required < reusable_frontier_blocks || required < resident ||
        required < allocated || required - allocated > available_blocks()) {
        return false;
    }
    if (has_virtual_commitment) {
        if (allocation->second.committed_blocks < required) {
            return false;
        }
        return true;
    }
    if (full_sequence_commitments_enabled_ && full_sequence_tokens > 0) {
        const std::uint64_t committed =
            full_sequence_blocks(full_sequence_tokens);
        if (committed < required || !can_commit_blocks(committed)) {
            return false;
        }
        return true;
    }
    return available_blocks() - required >= watermark_blocks_;
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
    SessionCacheEntry &session = sessions_[session_id];
    const std::uint64_t resident = session.resident_prefix_blocks;
    const auto allocation = allocations_.find(request_id);
    const bool has_virtual_commitment = allocation != allocations_.end() &&
                                        full_sequence_commitments_enabled_ &&
                                        allocation->second.committed_blocks >
                                            allocation->second.allocated_blocks;
    const std::uint64_t already_allocated =
        has_virtual_commitment ? allocation->second.allocated_blocks : 0;
    if (resident > 0 && !has_virtual_commitment) {
        remove_from_evictable_lru(session);
    }
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
    consume_available_blocks(physical_additional);
    active_blocks_ += physical_additional;
    resident_blocks_ += reusable_frontier_blocks - resident;
    if (resident == 0 && reusable_frontier_blocks > 0) {
        ++sessions_with_nonzero_frontier_;
    }
    session.resident_prefix_blocks = reusable_frontier_blocks;
    session.active_request = request_id;
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

bool ReplicaKVCacheManager::can_reserve(
    RequestId request_id, std::uint64_t kv_accounted_tokens,
    std::uint64_t scheduled_tokens, std::uint64_t full_sequence_tokens) const {
    const std::uint64_t required = additional_blocks_required(
        request_id, kv_accounted_tokens, scheduled_tokens, false);
    if (required > available_blocks()) {
        return false;
    }
    const auto allocation = allocations_.find(request_id);
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
    return available_blocks() - required >= watermark_blocks_;
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
    consume_available_blocks(additional);
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
            sessions_.erase(session);
            allocations_.erase(allocation);
            validate_accounting();
            return value.allocated_blocks;
        }
        entry.resident_prefix_blocks = value.published_blocks;
        entry.active_request = RequestId{};
        blank_blocks_ += value.allocated_blocks - value.published_blocks;
        if (entry.resident_prefix_blocks > 0) {
            append_to_evictable_lru(value.session_id, entry);
        } else {
            sessions_.erase(session);
        }
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
    if (!entry.in_evictable_lru || entry.resident_prefix_blocks == 0) {
        throw ReplicaKVCacheError(
            "inactive discarded session is not evictable");
    }
    const std::uint64_t discarded = entry.resident_prefix_blocks;
    remove_from_evictable_lru(entry);
    if (discarded > resident_blocks_) {
        throw ReplicaKVCacheError(
            "discarded session exceeds resident accounting");
    }
    resident_blocks_ -= discarded;
    blank_blocks_ += discarded;
    --sessions_with_nonzero_frontier_;
    stats_.evicted_blocks += discarded;
    ++stats_.evicted_sessions;
    sessions_.erase(position);
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
    return session == sessions_.end() ? 0
                                      : session->second.resident_prefix_blocks;
}

bool ReplicaKVCacheManager::session_has_active_request(
    SessionId session_id) const noexcept {
    if (!prefix_cache_enabled_ || !session_id.valid()) {
        return false;
    }
    const auto session = sessions_.find(session_id);
    return session != sessions_.end() &&
           session->second.active_request.valid();
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
    return blocks <=
               std::numeric_limits<std::uint64_t>::max() - watermark_blocks_ &&
           blocks + watermark_blocks_ <= capacity_blocks_;
}

void ReplicaKVCacheManager::validate_accounting() const {
    if (active_blocks_ > capacity_blocks_ ||
        virtual_committed_blocks_ > capacity_blocks_ - active_blocks_ ||
        blank_blocks_ > capacity_blocks_ - active_blocks_ ||
        evictable_blocks_ !=
            capacity_blocks_ - active_blocks_ - blank_blocks_ ||
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
        } else if (!entry.active_request.valid()) {
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
    return result;
}

} // namespace frontier::kv_cache
