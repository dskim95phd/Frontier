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
    std::uint64_t evicted_kda_snapshots = 0;
    std::uint64_t evicted_kda_snapshot_blocks = 0;
};

struct PrefixCacheDiagnostics {
    std::uint64_t capacity_blocks = 0;
    std::uint64_t available_blocks = 0;
    std::uint64_t active_blocks = 0;
    std::uint64_t resident_blocks = 0;
    std::uint64_t evictable_blocks = 0;
    std::uint64_t evictable_sessions = 0;
    std::uint64_t sessions_with_nonzero_frontier = 0;
    // Logical KV capacity reserved by active physical blocks plus any
    // PREFILL-only virtual commitments.  The latter is future-only: blocks
    // already materialized for a request are counted in active_blocks.
    std::uint64_t committed_blocks = 0;
    std::uint64_t virtual_committed_blocks = 0;
    std::uint64_t available_commitment_blocks = 0;
    std::uint64_t kda_snapshot_occupied_blocks = 0;
    std::uint64_t kda_snapshot_sessions = 0;
    std::uint64_t kda_snapshot_evictable_sessions = 0;
};

// Transaction checkpoint for scheduler operations that temporarily replace
// a GPU KDA snapshot. `published` distinguishes a cold capacity reservation
// from a valid computed snapshot; both may legitimately have frontier zero.
struct KdaSnapshotCheckpoint {
    bool present = false;
    std::uint64_t frontier_blocks = 0;
    bool published = false;
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
                                 std::uint64_t scheduled_tokens,
                                 std::uint64_t full_sequence_tokens = 0) const;
    void admit(RequestId request_id, SessionId session_id,
               std::uint64_t cached_tokens, std::uint64_t scheduled_tokens,
               std::uint64_t full_sequence_tokens = 0);
    [[nodiscard]] bool
    can_admit_tiered(RequestId request_id, SessionId session_id,
                     std::uint64_t reusable_frontier_blocks,
                     std::uint64_t scheduled_tokens,
                     std::uint64_t full_sequence_tokens = 0) const;
    void admit_tiered(RequestId request_id, SessionId session_id,
                      std::uint64_t reusable_frontier_blocks,
                      std::uint64_t scheduled_tokens,
                      std::uint64_t full_sequence_tokens = 0);
    void record_successful_admission(std::uint64_t query_blocks,
                                     std::uint64_t hit_blocks);

    [[nodiscard]] bool
    can_reserve(RequestId request_id, std::uint64_t kv_accounted_tokens,
                std::uint64_t scheduled_tokens,
                std::uint64_t full_sequence_tokens = 0) const;
    void reserve(RequestId request_id, std::uint64_t kv_accounted_tokens,
                 std::uint64_t scheduled_tokens,
                 std::uint64_t full_sequence_tokens = 0);

    // PREFILL admission can reserve the complete prompt before an asynchronous
    // CPU/KV restore begins.  The commitment is future-only; materialization
    // moves its own blocks into the physical allocation atomically.
    void enable_full_sequence_commitments(bool enabled = true);
    [[nodiscard]] bool full_sequence_commitments_enabled() const noexcept {
        return full_sequence_commitments_enabled_;
    }
    [[nodiscard]] bool can_commit(RequestId request_id,
                                  std::uint64_t full_sequence_tokens) const;
    [[nodiscard]] bool can_commit(RequestId request_id, SessionId session_id,
                                  std::uint64_t full_sequence_tokens) const;
    void commit_virtual(RequestId request_id, SessionId session_id,
                        std::uint64_t full_sequence_tokens);
    void release_commitment(RequestId request_id);
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
    // Future-only blocks held by PREFILL full-sequence commitments.
    [[nodiscard]] std::uint64_t virtual_committed_blocks() const noexcept {
        return virtual_committed_blocks_;
    }
    [[nodiscard]] std::uint64_t committed_blocks() const noexcept {
        return active_blocks_ + virtual_committed_blocks_ +
               kda_snapshot_occupied_blocks_;
    }
    [[nodiscard]] std::uint64_t available_commitment_blocks() const noexcept {
        return capacity_blocks_ - committed_blocks();
    }

    // KDA snapshots are charged as one atomic group of KV-equivalent blocks
    // per session. A zero charge leaves the legacy GPU KV policy unchanged.
    void configure_kda_snapshot(std::uint64_t charged_blocks);
    [[nodiscard]] bool kda_snapshot_enabled() const noexcept {
        return kda_snapshot_charge_blocks_ != 0;
    }
    [[nodiscard]] std::uint64_t kda_snapshot_charge_blocks() const noexcept {
        return kda_snapshot_charge_blocks_;
    }
    // Publish or replace the one latest immutable snapshot for a session.
    // Active-session snapshots remain pinned until free() releases the request.
    [[nodiscard]] bool publish_kda_snapshot(SessionId session_id,
                                            std::uint64_t frontier_blocks);
    [[nodiscard]] bool has_kda_snapshot(SessionId session_id) const noexcept;
    [[nodiscard]] std::uint64_t
    kda_snapshot_frontier_blocks(SessionId session_id) const noexcept;
    [[nodiscard]] KdaSnapshotCheckpoint
    checkpoint_kda_snapshot(SessionId session_id) const noexcept;
    // Restore metadata captured while the same request still pins the
    // session. This is used to roll back a failed CPU-to-GPU restore before
    // releasing its full-sequence commitment.
    void restore_kda_snapshot(SessionId session_id,
                              KdaSnapshotCheckpoint checkpoint);
    [[nodiscard]] std::uint64_t kda_snapshot_occupied_blocks() const noexcept {
        return kda_snapshot_occupied_blocks_;
    }
    [[nodiscard]] std::uint64_t discard_kda_snapshot(SessionId session_id);
    [[nodiscard]] std::uint64_t
    request_committed_blocks(RequestId request_id) const noexcept;
    [[nodiscard]] std::uint64_t
    request_virtual_committed_blocks(RequestId request_id) const noexcept;
    [[nodiscard]] bool
    full_sequence_fits_empty(std::uint64_t full_sequence_tokens) const;
    [[nodiscard]] std::uint64_t available_blocks() const noexcept {
        return capacity_blocks_ - active_blocks_ -
               kda_snapshot_occupied_blocks_;
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
    [[nodiscard]] bool
    session_has_active_request(SessionId session_id) const noexcept;
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
        // Unified inactive-session recency used for capacity reclaim.  The
        // ordinary KV and KDA snapshot LRUs remain separate for diagnostics,
        // but eviction must compare their owning sessions in one order.
        bool in_reclaim_lru = false;
        std::list<SessionId>::iterator reclaim_lru_position;
    };
    struct RequestKVAllocation {
        SessionId session_id;
        std::uint64_t allocated_blocks = 0;
        std::uint64_t published_blocks = 0;
        // Total logical target for this request.  In PREFILL commitment mode
        // this may exceed allocated_blocks while the suffix is virtual.
        std::uint64_t committed_blocks = 0;
    };
    struct KdaSnapshotEntry {
        std::uint64_t frontier_blocks = 0;
        bool published = false;
        bool in_lru = false;
        std::list<SessionId>::iterator lru_position;
    };

    [[nodiscard]] static std::uint64_t ceil_div(std::uint64_t numerator,
                                                std::uint64_t denominator);
    [[nodiscard]] std::uint64_t additional_blocks_required(
        RequestId request_id, std::uint64_t kv_accounted_tokens,
        std::uint64_t scheduled_tokens, bool for_materialization) const;
    [[nodiscard]] std::uint64_t
    full_sequence_blocks(std::uint64_t full_sequence_tokens) const;
    [[nodiscard]] bool can_commit_blocks(
        std::uint64_t blocks,
        std::optional<SessionId> excluded_session = std::nullopt) const;
    [[nodiscard]] std::uint64_t additional_kda_snapshot_charge(
        std::optional<SessionId> session_id) const noexcept;
    [[nodiscard]] bool
    required_capacity_fits(std::uint64_t blocks, std::uint64_t extra_blocks,
                           std::uint64_t available_blocks) const noexcept;
    [[nodiscard]] std::uint64_t
    reclaimable_snapshot_blocks(std::optional<SessionId> excluded_session =
                                    std::nullopt) const noexcept;
    [[nodiscard]] std::uint64_t reclaimable_commitment_blocks(
        std::optional<SessionId> excluded_session = std::nullopt,
        std::uint64_t virtual_credit = 0) const noexcept;
    void consume_available_blocks(std::uint64_t blocks,
                                  std::uint64_t virtual_credit = 0);
    void reclaim_for_commitment(std::uint64_t blocks);
    [[nodiscard]] bool can_allocate_kda_snapshot() const;
    void reserve_kda_snapshot_capacity(SessionId session_id);
    [[nodiscard]] std::uint64_t
    effective_resident_frontier_blocks(SessionId session_id) const noexcept;
    void trim_resident_frontier(SessionId session_id,
                                std::uint64_t keep_blocks);
    void consume_kda_snapshot_capacity();
    void remove_kda_snapshot(SessionId session_id, bool count_eviction);
    void remove_kda_snapshot_from_lru(KdaSnapshotEntry &entry);
    void append_kda_snapshot_to_lru(SessionId session_id,
                                    KdaSnapshotEntry &entry);
    void erase_session_if_unowned(SessionId session_id);
    void remove_from_evictable_lru(SessionCacheEntry &entry);
    void append_to_evictable_lru(SessionId session_id,
                                 SessionCacheEntry &entry);
    void remove_from_reclaim_lru(SessionCacheEntry &entry);
    void append_to_reclaim_lru(SessionId session_id, SessionCacheEntry &entry);
    void sync_reclaim_lru(SessionId session_id);
    void touch_reclaim_lru(SessionId session_id);
    void validate_accounting() const;

    std::uint64_t block_size_;
    std::uint64_t watermark_blocks_;
    bool prefix_cache_enabled_;
    std::uint64_t capacity_blocks_;
    std::uint64_t active_blocks_ = 0;
    std::uint64_t virtual_committed_blocks_ = 0;
    std::uint64_t kda_snapshot_charge_blocks_ = 0;
    std::uint64_t kda_snapshot_occupied_blocks_ = 0;
    std::uint64_t blank_blocks_;
    std::uint64_t evictable_blocks_ = 0;
    std::uint64_t resident_blocks_ = 0;
    std::uint64_t sessions_with_nonzero_frontier_ = 0;
    std::list<SessionId> evictable_lru_;
    std::list<SessionId> reclaim_lru_;
    std::list<SessionId> kda_snapshot_lru_;
    std::unordered_map<RequestId, RequestKVAllocation, StrongIdHash<RequestId>>
        allocations_;
    std::unordered_map<SessionId, SessionCacheEntry, StrongIdHash<SessionId>>
        sessions_;
    std::unordered_map<SessionId, KdaSnapshotEntry, StrongIdHash<SessionId>>
        kda_snapshots_;
    std::unordered_set<SessionId, StrongIdHash<SessionId>> discard_on_release_;
    bool full_sequence_commitments_enabled_ = false;
    PrefixCacheStats stats_;
};

} // namespace frontier::kv_cache
