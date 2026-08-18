#pragma once

#include <cstdint>
#include <list>
#include <map>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "frontier/config/config.h"
#include "frontier/core/event.h"
#include "frontier/core/id_generator.h"
#include "frontier/core/ids.h"

namespace frontier::kv_cache {

class CpuKVCacheError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

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
    // A KDA state snapshot is transferred as one fixed-size immutable object.
    // The payload is charged in full for every update, including replacement
    // of an existing snapshot; it is intentionally separate from the normal
    // KV block reservation count above.
    std::uint64_t kda_snapshot_blocks = 0;
    std::uint64_t kda_snapshot_bytes = 0;
    bool skipped = false;
    bool truncated = false;

    [[nodiscard]] bool requires_transfer() const noexcept {
        return reservation_id.valid() &&
               (reserved_blocks > 0 || kda_snapshot_blocks > 0);
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
    std::uint64_t evicted_kda_snapshots = 0;
    std::uint64_t evicted_kda_snapshot_blocks = 0;
    std::uint64_t peak_resident_blocks = 0;
    std::uint64_t peak_reserved_blocks = 0;
    std::uint64_t stale_generation_completions = 0;
    std::uint64_t discarded_offload_completions = 0;
    std::uint64_t sessions_with_hits = 0;
};

struct CpuKVCacheDiagnostics {
    std::uint64_t capacity_blocks = 0;
    std::uint64_t resident_blocks = 0;
    std::uint64_t reserved_blocks = 0;
    std::uint64_t kda_snapshot_occupied_blocks = 0;
    std::uint64_t kda_snapshot_reserved_blocks = 0;
    std::uint64_t kda_snapshot_sessions = 0;
    std::uint64_t kda_snapshot_evictable_sessions = 0;
    std::uint64_t pinned_blocks = 0;
    std::uint64_t pinned_kda_snapshots = 0;
    std::uint64_t sessions = 0;
    std::uint64_t active_reservations = 0;
    std::uint64_t active_restore_leases = 0;
    std::uint64_t materialized_blocks = 0;
};

class CpuKVCacheManager {
  public:
    CpuKVCacheManager(std::uint64_t capacity_blocks,
                      config::CpuKVCacheCapacityPressurePolicy pressure_policy);

    // Configure the fixed CPU charge for one KDA recurrent-state snapshot.
    // A zero charge disables snapshot handling and preserves the legacy CPU
    // KV policy.  Bytes are optional for direct users; schedulers should pass
    // the resolved state footprint so transfer accounting can report it.
    void configure_kda_snapshot(std::uint64_t charged_blocks,
                                std::uint64_t charged_bytes = 0);
    [[nodiscard]] bool kda_snapshot_enabled() const noexcept {
        return kda_snapshot_charge_blocks_ != 0;
    }
    [[nodiscard]] std::uint64_t kda_snapshot_charge_blocks() const noexcept {
        return kda_snapshot_charge_blocks_;
    }
    [[nodiscard]] std::uint64_t kda_snapshot_charge_bytes() const noexcept {
        return kda_snapshot_charge_bytes_;
    }

    [[nodiscard]] CpuPrefixLookupResult
    lookup(SessionId session_id, std::uint64_t query_blocks) const noexcept;
    void record_successful_lookup(RequestId request_id,
                                  CpuPrefixLookupResult result,
                                  SessionId session_id = SessionId{});

    [[nodiscard]] CpuOffloadReservationResult
    reserve_offload(SessionId session_id, CpuOffloadGeneration generation,
                    std::uint64_t desired_frontier_blocks,
                    SimTime submitted_at);
    [[nodiscard]] bool commit_offload(CpuOffloadReservationId reservation_id,
                                      SimTime completed_at);
    [[nodiscard]] bool abort_offload(CpuOffloadReservationId reservation_id);
    [[nodiscard]] bool discard_session(SessionId session_id);
    [[nodiscard]] bool
    session_discard_pending(SessionId session_id) const noexcept;

    [[nodiscard]] bool has_kda_snapshot(SessionId session_id) const noexcept;
    [[nodiscard]] std::uint64_t
    kda_snapshot_frontier_blocks(SessionId session_id) const noexcept;
    [[nodiscard]] std::uint64_t kda_snapshot_occupied_blocks() const noexcept {
        return kda_snapshot_occupied_blocks_;
    }
    // Return the number of fixed-charge blocks removed.  Explicit discard is
    // not counted as an eviction in stats.
    [[nodiscard]] std::uint64_t discard_kda_snapshot(SessionId session_id);

    [[nodiscard]] CpuRestoreLeaseId pin_restore(SessionId session_id,
                                                std::uint64_t begin_block,
                                                std::uint64_t end_block,
                                                SimTime started_at);
    [[nodiscard]] bool release_restore(CpuRestoreLeaseId lease_id, bool used,
                                       SimTime released_at);
    [[nodiscard]] bool
    restore_includes_kda_snapshot(CpuRestoreLeaseId lease_id) const noexcept;
    [[nodiscard]] std::uint64_t
    restore_kda_snapshot_blocks(CpuRestoreLeaseId lease_id) const noexcept;
    [[nodiscard]] std::uint64_t
    restore_kda_snapshot_bytes(CpuRestoreLeaseId lease_id) const noexcept;

    [[nodiscard]] std::uint64_t
    committed_frontier_blocks(SessionId session_id) const noexcept;
    [[nodiscard]] bool
    reservation_pending(CpuOffloadReservationId reservation_id) const noexcept;
    [[nodiscard]] bool lease_active(CpuRestoreLeaseId lease_id) const noexcept;
    [[nodiscard]] std::size_t retained_reservation_count() const noexcept {
        return reservations_.size();
    }
    [[nodiscard]] std::size_t retained_restore_lease_count() const noexcept {
        return leases_.size();
    }
    [[nodiscard]] const CpuKVCacheStats &stats() const noexcept {
        return stats_;
    }
    [[nodiscard]] CpuKVCacheDiagnostics diagnostics() const;
    void validate_invariants() const;

  private:
    struct SessionState {
        // Ordinary CPU KV is an analytical contiguous prefix. Physical block
        // identity is intentionally absent: in-flight suffix reservations and
        // completed out-of-order ranges carry the only state that cannot be
        // represented by the two frontiers and aggregate counts below.
        std::uint64_t committed_frontier_blocks = 0;
        std::uint64_t reserved_frontier_blocks = 0;
        std::uint64_t committed_blocks = 0;
        std::uint64_t reserved_blocks = 0;
        SimTime last_access_time;
        SimTime last_commit_time;
        CpuOffloadGeneration latest_submitted_generation;
        CpuOffloadGeneration latest_committed_generation;
        std::map<std::uint64_t, std::uint64_t> committed_out_of_order_ranges;
        std::unordered_set<CpuOffloadReservationId,
                           StrongIdHash<CpuOffloadReservationId>>
            active_reservations;
        std::unordered_set<CpuRestoreLeaseId, StrongIdHash<CpuRestoreLeaseId>>
            active_restore_leases;
        std::uint64_t aggregate_restore_pins = 0;
        std::uint64_t distinct_pinned_blocks = 0;
        std::uint64_t aggregate_snapshot_pins = 0;
        bool discard_pending = false;
    };

    struct KdaSnapshot {
        std::uint64_t frontier_blocks = 0;
        CpuOffloadGeneration generation;
        bool in_lru = false;
        std::list<SessionId>::iterator lru_position;
    };

    struct OffloadReservation {
        CpuOffloadReservationId id;
        SessionId session_id;
        CpuOffloadGeneration generation;
        std::uint64_t desired_frontier_blocks = 0;
        std::uint64_t admitted_frontier_blocks = 0;
        std::uint64_t begin_block = 0;
        // Reservations exist only while pending. The ordinary KV suffix is
        // the half-open range [begin_block, admitted_frontier_blocks).
        SimTime submitted_at;
        bool includes_kda_snapshot = false;
        bool snapshot_slot_reserved = false;
        std::uint64_t kda_snapshot_frontier_blocks = 0;
        bool truncated = false;
        bool retired_completion_received = false;
        CpuOffloadReservationState state = CpuOffloadReservationState::kPending;
    };

    struct RestoreLease {
        CpuRestoreLeaseId id;
        SessionId session_id;
        // Leases exist only while active; release erases the record. The
        // pinned ordinary KV is the half-open range [begin_block, end_block).
        std::uint64_t begin_block = 0;
        std::uint64_t end_block = 0;
        SimTime started_at;
        bool includes_kda_snapshot = false;
        std::uint64_t kda_snapshot_blocks = 0;
        std::uint64_t kda_snapshot_bytes = 0;
        bool released = false;
    };

    [[nodiscard]] std::uint64_t available_blocks() const noexcept;
    [[nodiscard]] std::uint64_t evict_for(std::uint64_t required,
                                          SessionId excluded_session);
    void remove_kda_snapshot_from_lru(KdaSnapshot &snapshot);
    void append_kda_snapshot_to_lru(SessionId session_id,
                                    KdaSnapshot &snapshot);
    void remove_kda_snapshot(SessionId session_id, bool count_eviction);
    [[nodiscard]] bool can_evict_kda_snapshot(SessionId session_id) const;
    [[nodiscard]] bool reserve_kda_snapshot_slot(SessionId session_id);
    void release_kda_snapshot_slot(SessionId session_id);
    [[nodiscard]] bool
    session_has_pending_snapshot(SessionId session_id) const noexcept;
    void publish_committed_range(SessionState &session,
                                 std::uint64_t begin_block,
                                 std::uint64_t end_block);
    void recompute_distinct_pinned_blocks(SessionState &session);
    void erase_session_if_empty(SessionId session_id);
    void maybe_reap_discarded_session(SessionId session_id);
    void record_occupancy_peaks() noexcept;
    // Hot-path accounting checks intentionally avoid traversing every block,
    // session, reservation, and lease.  Full structural validation remains
    // available through validate_invariants() and diagnostics().
    void validate_local_invariants() const;

    std::uint64_t capacity_blocks_;
    config::CpuKVCacheCapacityPressurePolicy pressure_policy_;
    std::uint64_t resident_blocks_ = 0;
    std::uint64_t reserved_blocks_ = 0;
    std::uint64_t kda_snapshot_charge_blocks_ = 0;
    std::uint64_t kda_snapshot_charge_bytes_ = 0;
    std::uint64_t kda_snapshot_occupied_blocks_ = 0;
    std::uint64_t kda_snapshot_reserved_blocks_ = 0;
    std::uint64_t pinned_blocks_ = 0;
    std::uint64_t pinned_kda_snapshots_ = 0;
    CheckedIdGenerator<CpuOffloadReservationId> reservation_ids_;
    CheckedIdGenerator<CpuRestoreLeaseId> lease_ids_;
    std::unordered_map<SessionId, SessionState, StrongIdHash<SessionId>>
        sessions_;
    std::list<SessionId> kda_snapshot_lru_;
    std::unordered_map<SessionId, KdaSnapshot, StrongIdHash<SessionId>>
        kda_snapshots_;
    std::unordered_map<SessionId, CpuOffloadReservationId,
                       StrongIdHash<SessionId>>
        kda_snapshot_slot_owners_;
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
