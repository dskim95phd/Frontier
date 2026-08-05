#pragma once

#include <cstdint>
#include <list>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "frontier/config/config.h"
#include "frontier/core/ids.h"

namespace frontier::entities {
class Request;
}

namespace frontier::kv_cache {

class ReplicaKVCacheError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

struct PrefixLookupResult {
    std::uint64_t query_blocks = 0;
    std::uint64_t hit_blocks = 0;
    std::uint64_t cached_tokens = 0;
};

struct PrefixCacheStats {
    std::uint64_t successful_admissions = 0;
    std::uint64_t query_blocks = 0;
    std::uint64_t hit_blocks = 0;
    std::uint64_t evicted_blocks = 0;
    std::uint64_t evicted_sessions = 0;
};

struct PrefixCacheDiagnostics {
    std::uint64_t capacity_blocks = 0;
    std::uint64_t available_blocks = 0;
    std::uint64_t active_blocks = 0;
    std::uint64_t resident_blocks = 0;
    std::uint64_t evictable_blocks = 0;
    std::uint64_t evictable_sessions = 0;
    std::uint64_t sessions_with_nonzero_frontier = 0;
};

// Analytical, session-scoped append-only KV cache. Physical block identity is
// intentionally absent: capacity is partitioned into active, blank, and
// session-resident evictable block counts. Inactive sessions form an LRU;
// allocation reclaims the oldest session suffix in ranges.
class ReplicaKVCacheManager {
  public:
    ReplicaKVCacheManager(const config::SchedulerConfig &scheduler_config,
                          config::PrefixCacheConfig prefix_config,
                          bool cache_owner);

    [[nodiscard]] PrefixLookupResult
    lookup(const entities::Request &request) const;
    [[nodiscard]] bool can_admit(RequestId request_id, SessionId session_id,
                                 std::uint64_t cached_tokens,
                                 std::uint64_t scheduled_tokens) const;
    void admit(RequestId request_id, SessionId session_id,
               std::uint64_t cached_tokens, std::uint64_t scheduled_tokens);
    [[nodiscard]] bool can_admit_tiered(
        RequestId request_id, SessionId session_id,
        std::uint64_t reusable_frontier_blocks,
        std::uint64_t scheduled_tokens) const;
    void admit_tiered(RequestId request_id, SessionId session_id,
                      std::uint64_t reusable_frontier_blocks,
                      std::uint64_t scheduled_tokens);
    void record_successful_admission(std::uint64_t query_blocks,
                                     std::uint64_t hit_blocks);

    [[nodiscard]] bool can_reserve(RequestId request_id,
                                   std::uint64_t kv_accounted_tokens,
                                   std::uint64_t scheduled_tokens) const;
    void reserve(RequestId request_id, std::uint64_t kv_accounted_tokens,
                 std::uint64_t scheduled_tokens);
    [[nodiscard]] std::uint64_t free(RequestId request_id);
    // Drop every GPU-resident block for a session. If the session still has
    // an active request, the drop is deferred until that request releases its
    // allocation so routing can migrate without corrupting in-flight work.
    [[nodiscard]] std::uint64_t discard_session(SessionId session_id);
    void mark_blocks_computed(const entities::Request &request);

    [[nodiscard]] std::uint64_t
    allocated_blocks(RequestId request_id) const noexcept;
    [[nodiscard]] std::uint64_t total_allocated_blocks() const noexcept {
        return active_blocks_;
    }
    [[nodiscard]] std::uint64_t available_blocks() const noexcept {
        return capacity_blocks_ - active_blocks_;
    }
    [[nodiscard]] std::uint64_t capacity_blocks() const noexcept {
        return capacity_blocks_;
    }
    [[nodiscard]] std::uint64_t watermark_blocks() const noexcept {
        return watermark_blocks_;
    }
    [[nodiscard]] bool empty() const noexcept { return allocations_.empty(); }
    [[nodiscard]] std::size_t allocation_count() const noexcept {
        return allocations_.size();
    }
    [[nodiscard]] bool prefix_cache_enabled() const noexcept {
        return prefix_cache_enabled_;
    }
    [[nodiscard]] std::uint64_t block_size() const noexcept {
        return block_size_;
    }
    [[nodiscard]] std::uint64_t
    gpu_cache_valid_prefix_blocks(SessionId session_id) const noexcept;
    [[nodiscard]] const PrefixCacheStats &stats() const noexcept {
        return stats_;
    }
    [[nodiscard]] PrefixCacheDiagnostics diagnostics() const;

  private:
    struct SessionCacheEntry {
        std::uint64_t resident_prefix_blocks = 0;
        RequestId active_request;
        bool in_evictable_lru = false;
        std::list<SessionId>::iterator lru_position;
    };
    struct RequestKVAllocation {
        SessionId session_id;
        std::uint64_t allocated_blocks = 0;
        std::uint64_t published_blocks = 0;
    };

    [[nodiscard]] static std::uint64_t ceil_div(std::uint64_t numerator,
                                                std::uint64_t denominator);
    [[nodiscard]] std::uint64_t additional_blocks_required(
        RequestId request_id, std::uint64_t kv_accounted_tokens,
        std::uint64_t scheduled_tokens, bool for_materialization) const;
    void consume_available_blocks(std::uint64_t blocks);
    void remove_from_evictable_lru(SessionCacheEntry &entry);
    void append_to_evictable_lru(SessionId session_id,
                                 SessionCacheEntry &entry);
    void validate_accounting() const;

    std::uint64_t block_size_;
    std::uint64_t watermark_blocks_;
    bool prefix_cache_enabled_;
    std::uint64_t capacity_blocks_;
    std::uint64_t active_blocks_ = 0;
    std::uint64_t blank_blocks_;
    std::uint64_t evictable_blocks_ = 0;
    std::uint64_t resident_blocks_ = 0;
    std::uint64_t sessions_with_nonzero_frontier_ = 0;
    std::list<SessionId> evictable_lru_;
    std::unordered_map<RequestId, RequestKVAllocation, StrongIdHash<RequestId>>
        allocations_;
    std::unordered_map<SessionId, SessionCacheEntry, StrongIdHash<SessionId>>
        sessions_;
    std::unordered_set<SessionId, StrongIdHash<SessionId>>
        discard_on_release_;
    PrefixCacheStats stats_;
};

} // namespace frontier::kv_cache
