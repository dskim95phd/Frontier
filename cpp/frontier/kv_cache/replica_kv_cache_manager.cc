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

std::uint64_t ReplicaKVCacheManager::ceil_div(std::uint64_t numerator,
                                              std::uint64_t denominator) {
    if (denominator == 0) {
        throw ReplicaKVCacheError("KV division denominator is zero");
    }
    return numerator / denominator +
           static_cast<std::uint64_t>(numerator % denominator != 0);
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

bool ReplicaKVCacheManager::can_admit(RequestId request_id,
                                      SessionId session_id,
                                      std::uint64_t cached_tokens,
                                      std::uint64_t scheduled_tokens) const {
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
        position->second.active_request.valid()) {
        return false;
    }
    const std::uint64_t cached_blocks = cached_tokens / block_size_;
    const std::uint64_t required =
        ceil_div(cached_tokens + scheduled_tokens, block_size_);
    if (cached_blocks > resident || required < resident ||
        required > available_blocks()) {
        return false;
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
                                  std::uint64_t scheduled_tokens) {
    if (!can_admit(request_id, session_id, cached_tokens, scheduled_tokens)) {
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
    consume_available_blocks(required - resident);
    active_blocks_ += required;
    session.active_request = request_id;
    if (!allocations_
             .emplace(request_id,
                      RequestKVAllocation{session_id, required, resident})
             .second) {
        throw ReplicaKVCacheError("request already owns KV blocks");
    }
    validate_accounting();
}

bool ReplicaKVCacheManager::can_admit_tiered(
    RequestId request_id, SessionId session_id,
    std::uint64_t reusable_frontier_blocks,
    std::uint64_t scheduled_tokens) const {
    if (!prefix_cache_enabled_ || !session_id.valid() ||
        allocations_.find(request_id) != allocations_.end() ||
        scheduled_tokens == 0 ||
        reusable_frontier_blocks >
            std::numeric_limits<std::uint64_t>::max() / block_size_) {
        return false;
    }
    const auto position = sessions_.find(session_id);
    const std::uint64_t resident =
        position == sessions_.end() ? 0
                                    : position->second.resident_prefix_blocks;
    if (position != sessions_.end() &&
        position->second.active_request.valid()) {
        return false;
    }
    if (reusable_frontier_blocks < resident) {
        return false;
    }
    const std::uint64_t reusable_tokens = reusable_frontier_blocks * block_size_;
    if (reusable_tokens >
        std::numeric_limits<std::uint64_t>::max() - scheduled_tokens) {
        return false;
    }
    const std::uint64_t required =
        ceil_div(reusable_tokens + scheduled_tokens, block_size_);
    if (required < reusable_frontier_blocks || required < resident ||
        required > available_blocks()) {
        return false;
    }
    return available_blocks() - required >= watermark_blocks_;
}

void ReplicaKVCacheManager::admit_tiered(
    RequestId request_id, SessionId session_id,
    std::uint64_t reusable_frontier_blocks,
    std::uint64_t scheduled_tokens) {
    if (!can_admit_tiered(request_id, session_id, reusable_frontier_blocks,
                          scheduled_tokens)) {
        throw ReplicaKVCacheError(
            "tiered KV admission exceeds capacity or watermark");
    }
    SessionCacheEntry &session = sessions_[session_id];
    const std::uint64_t resident = session.resident_prefix_blocks;
    if (resident > 0) {
        remove_from_evictable_lru(session);
    }
    const std::uint64_t reusable_tokens = reusable_frontier_blocks * block_size_;
    const std::uint64_t required =
        ceil_div(reusable_tokens + scheduled_tokens, block_size_);
    consume_available_blocks(required - resident);
    active_blocks_ += required;
    resident_blocks_ += reusable_frontier_blocks - resident;
    if (resident == 0 && reusable_frontier_blocks > 0) {
        ++sessions_with_nonzero_frontier_;
    }
    session.resident_prefix_blocks = reusable_frontier_blocks;
    session.active_request = request_id;
    if (!allocations_
             .emplace(request_id,
                      RequestKVAllocation{session_id, required,
                                          reusable_frontier_blocks})
             .second) {
        throw ReplicaKVCacheError("tiered request already owns KV blocks");
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

bool ReplicaKVCacheManager::can_reserve(RequestId request_id,
                                        std::uint64_t kv_accounted_tokens,
                                        std::uint64_t scheduled_tokens) const {
    const std::uint64_t required = additional_blocks_required(
        request_id, kv_accounted_tokens, scheduled_tokens, false);
    if (required > available_blocks()) {
        return false;
    }
    if (allocations_.find(request_id) != allocations_.end()) {
        return true;
    }
    return available_blocks() - required >= watermark_blocks_;
}

void ReplicaKVCacheManager::reserve(RequestId request_id,
                                    std::uint64_t kv_accounted_tokens,
                                    std::uint64_t scheduled_tokens) {
    if (!can_reserve(request_id, kv_accounted_tokens, scheduled_tokens)) {
        throw ReplicaKVCacheError(
            "KV reservation exceeds capacity or watermark");
    }
    const std::uint64_t additional = additional_blocks_required(
        request_id, kv_accounted_tokens, scheduled_tokens, true);
    auto [position, inserted] =
        allocations_.try_emplace(request_id, RequestKVAllocation{});
    static_cast<void>(inserted);
    consume_available_blocks(additional);
    position->second.allocated_blocks += additional;
    active_blocks_ += additional;
    validate_accounting();
}

std::uint64_t ReplicaKVCacheManager::free(RequestId request_id) {
    const auto allocation = allocations_.find(request_id);
    if (allocation == allocations_.end()) {
        return 0;
    }
    const RequestKVAllocation value = allocation->second;
    if (value.allocated_blocks > active_blocks_ ||
        value.published_blocks > value.allocated_blocks) {
        throw ReplicaKVCacheError("request allocation accounting is corrupt");
    }
    active_blocks_ -= value.allocated_blocks;
    if (prefix_cache_enabled_ && value.session_id.valid()) {
        auto session = sessions_.find(value.session_id);
        if (session == sessions_.end() ||
            session->second.active_request != request_id ||
            session->second.in_evictable_lru) {
            throw ReplicaKVCacheError("active session allocation is corrupt");
        }
        SessionCacheEntry &entry = session->second;
        const bool discard =
            discard_on_release_.erase(value.session_id) != 0;
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

std::uint64_t
ReplicaKVCacheManager::discard_session(SessionId session_id) {
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

void ReplicaKVCacheManager::validate_accounting() const {
    if (active_blocks_ > capacity_blocks_ ||
        blank_blocks_ > capacity_blocks_ - active_blocks_ ||
        evictable_blocks_ !=
            capacity_blocks_ - active_blocks_ - blank_blocks_ ||
        resident_blocks_ > capacity_blocks_) {
        throw ReplicaKVCacheError(
            "analytical KV capacity partition is invalid");
    }
#ifndef NDEBUG
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
    return result;
}

} // namespace frontier::kv_cache
