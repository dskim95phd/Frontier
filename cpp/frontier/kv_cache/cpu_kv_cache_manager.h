#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "frontier/config/config.h"
#include "frontier/core/event.h"
#include "frontier/core/ids.h"

namespace frontier::kv_cache {

class CpuKVCacheError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

enum class CpuBlockState { kReserved, kCommitted };
enum class CpuOffloadReservationState { kPending, kCommitted, kAborted };

struct CpuPrefixLookupResult {
    std::uint64_t query_blocks = 0;
    std::uint64_t hit_blocks = 0;
};

struct CpuOffloadReservationResult {
    CpuOffloadReservationId reservation_id;
    std::uint64_t desired_frontier_blocks = 0;
    std::uint64_t admitted_frontier_blocks = 0;
    std::uint64_t reserved_blocks = 0;
    bool skipped = false;
    bool truncated = false;

    [[nodiscard]] bool requires_transfer() const noexcept {
        return reservation_id.valid() && reserved_blocks > 0;
    }
};

struct CpuKVCacheStats {
    std::uint64_t successful_lookups = 0;
    std::uint64_t query_blocks = 0;
    std::uint64_t hit_blocks = 0;
    std::uint64_t committed_offloads = 0;
    std::uint64_t aborted_offloads = 0;
    std::uint64_t skipped_offloads = 0;
    std::uint64_t truncated_offloads = 0;
    std::uint64_t evicted_blocks = 0;
    std::uint64_t evicted_sessions = 0;
    std::uint64_t peak_resident_blocks = 0;
    std::uint64_t peak_reserved_blocks = 0;
    std::uint64_t stale_generation_completions = 0;
    std::uint64_t sessions_with_hits = 0;
};

struct CpuKVCacheDiagnostics {
    std::uint64_t capacity_blocks = 0;
    std::uint64_t resident_blocks = 0;
    std::uint64_t reserved_blocks = 0;
    std::uint64_t pinned_blocks = 0;
    std::uint64_t sessions = 0;
    std::uint64_t active_reservations = 0;
    std::uint64_t active_restore_leases = 0;
    std::uint64_t materialized_blocks = 0;
};

class CpuKVCacheManager {
  public:
    CpuKVCacheManager(
        std::uint64_t capacity_blocks,
        config::CpuKVCacheCapacityPressurePolicy pressure_policy);

    [[nodiscard]] CpuPrefixLookupResult
    lookup(SessionId session_id, std::uint64_t query_blocks) const noexcept;
    void record_successful_lookup(RequestId request_id,
                                  CpuPrefixLookupResult result,
                                  SessionId session_id = SessionId{});

    [[nodiscard]] CpuOffloadReservationResult reserve_offload(
        SessionId session_id, CpuOffloadGeneration generation,
        std::uint64_t desired_frontier_blocks, SimTime submitted_at);
    [[nodiscard]] bool commit_offload(CpuOffloadReservationId reservation_id,
                                      SimTime completed_at);
    [[nodiscard]] bool abort_offload(CpuOffloadReservationId reservation_id);

    [[nodiscard]] CpuRestoreLeaseId pin_restore(
        SessionId session_id, std::uint64_t begin_block,
        std::uint64_t end_block, SimTime started_at);
    [[nodiscard]] bool release_restore(CpuRestoreLeaseId lease_id, bool used,
                                       SimTime released_at);

    [[nodiscard]] std::uint64_t
    committed_frontier_blocks(SessionId session_id) const noexcept;
    [[nodiscard]] bool reservation_pending(
        CpuOffloadReservationId reservation_id) const noexcept;
    [[nodiscard]] bool lease_active(CpuRestoreLeaseId lease_id) const noexcept;
    [[nodiscard]] const CpuKVCacheStats &stats() const noexcept {
        return stats_;
    }
    [[nodiscard]] CpuKVCacheDiagnostics diagnostics() const;
    void validate_invariants() const;

  private:
    struct CpuBlock {
        CpuBlockId id;
        CpuBlockState state = CpuBlockState::kReserved;
        SessionId session_id;
        std::uint64_t logical_index = 0;
        std::uint64_t pin_count = 0;
        CpuOffloadReservationId reservation_id;
        CpuOffloadGeneration generation;
    };

    struct SessionState {
        // Materialized session blocks are always a contiguous prefix.  The
        // committed frontier may lag the reserved frontier when a suffix
        // reservation completes out of order.
        std::uint64_t committed_frontier_blocks = 0;
        std::uint64_t reserved_frontier_blocks = 0;
        SimTime last_access_time;
        SimTime last_commit_time;
        CpuOffloadGeneration latest_submitted_generation;
        CpuOffloadGeneration latest_committed_generation;
        std::unordered_map<std::uint64_t, CpuBlockId> blocks;
        std::unordered_set<CpuOffloadReservationId,
                           StrongIdHash<CpuOffloadReservationId>>
            active_reservations;
        std::uint64_t aggregate_restore_pins = 0;
    };

    struct OffloadReservation {
        CpuOffloadReservationId id;
        SessionId session_id;
        CpuOffloadGeneration generation;
        std::uint64_t desired_frontier_blocks = 0;
        std::uint64_t admitted_frontier_blocks = 0;
        std::uint64_t begin_block = 0;
        // Populated only while pending; terminal records retain compact
        // metadata for duplicate-completion idempotency.
        std::vector<CpuBlockId> block_ids;
        SimTime submitted_at;
        bool truncated = false;
        CpuOffloadReservationState state =
            CpuOffloadReservationState::kPending;
    };

    struct RestoreLease {
        CpuRestoreLeaseId id;
        SessionId session_id;
        // Populated only while active; release drops the backing allocation.
        std::vector<CpuBlockId> block_ids;
        SimTime started_at;
        bool released = false;
    };

    [[nodiscard]] CpuBlockId allocate_block_id();
    void free_block(CpuBlockId block_id);
    [[nodiscard]] std::uint64_t available_blocks() const noexcept;
    [[nodiscard]] std::uint64_t evict_for(
        std::uint64_t required, SessionId excluded_session);
    void advance_committed_frontier(SessionState &session);
    void erase_session_if_empty(SessionId session_id);
    void record_occupancy_peaks() noexcept;
    // Hot-path accounting checks intentionally avoid traversing every block,
    // session, reservation, and lease.  Full structural validation remains
    // available through validate_invariants() and diagnostics().
    void validate_local_invariants() const;

    std::uint64_t capacity_blocks_;
    config::CpuKVCacheCapacityPressurePolicy pressure_policy_;
    std::uint64_t resident_blocks_ = 0;
    std::uint64_t reserved_blocks_ = 0;
    std::uint64_t pinned_blocks_ = 0;
    std::uint64_t next_block_id_ = 0;
    std::uint64_t next_reservation_id_ = 0;
    std::uint64_t next_lease_id_ = 0;
    std::vector<CpuBlockId> recycled_block_ids_;
    std::unordered_map<CpuBlockId, CpuBlock, StrongIdHash<CpuBlockId>> blocks_;
    std::unordered_map<SessionId, SessionState, StrongIdHash<SessionId>>
        sessions_;
    std::unordered_map<CpuOffloadReservationId, OffloadReservation,
                       StrongIdHash<CpuOffloadReservationId>>
        reservations_;
    std::unordered_map<CpuRestoreLeaseId, RestoreLease,
                       StrongIdHash<CpuRestoreLeaseId>>
        leases_;
    std::unordered_set<RequestId, StrongIdHash<RequestId>>
        recorded_lookup_requests_;
    std::unordered_set<SessionId, StrongIdHash<SessionId>> hit_sessions_;
    CpuKVCacheStats stats_;
};

} // namespace frontier::kv_cache
