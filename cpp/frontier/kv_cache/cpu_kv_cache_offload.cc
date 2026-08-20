#include "frontier/kv_cache/cpu_kv_cache_manager.h"

#include <algorithm>
#include <limits>
#include <tuple>
#include <vector>

#include "frontier/core/checked_math.h"

namespace frontier::kv_cache {

CpuOffloadReservationResult CpuKVCacheManager::reserve_offload(
    SessionId session_id, CpuOffloadGeneration generation,
    std::uint64_t desired_frontier_blocks, SimTime submitted_at) {
    if (!session_id.valid() || !generation.valid() || !submitted_at.valid()) {
        throw CpuKVCacheError("invalid CPU offload reservation identity/time");
    }
    auto existing = sessions_.find(session_id);
    if (existing != sessions_.end() && existing->second.discard_pending) {
        CpuOffloadReservationResult result{};
        result.desired_frontier_blocks = desired_frontier_blocks;
        result.skipped = true;
        ++stats_.skipped_offloads;
        return result;
    }
    if (existing != sessions_.end() &&
        existing->second.latest_submitted_generation.valid() &&
        generation <= existing->second.latest_submitted_generation) {
        throw CpuKVCacheError("CPU offload generation must be monotonic");
    }
    const std::uint64_t base = existing == sessions_.end()
                                   ? 0
                                   : existing->second.reserved_frontier_blocks;
    const auto snapshot_position = kda_snapshots_.find(session_id);
    const bool has_snapshot = snapshot_position != kda_snapshots_.end();
    // A snapshot update is needed for a new session or whenever its
    // block-level frontier changes.  Repeating the same frontier is a true
    // no-op and does not consume transfer bandwidth.
    const bool snapshot_update =
        kda_snapshot_charge_blocks_ != 0 &&
        (!has_snapshot ||
         snapshot_position->second.frontier_blocks != desired_frontier_blocks);
    const bool snapshot_slot_needed = snapshot_update && !has_snapshot &&
                                      !session_has_pending_snapshot(session_id);
    CpuOffloadReservationResult result{};
    result.desired_frontier_blocks = desired_frontier_blocks;
    result.admitted_frontier_blocks = std::min(desired_frontier_blocks, base);
    const std::uint64_t missing =
        desired_frontier_blocks > base ? desired_frontier_blocks - base : 0;
    if (desired_frontier_blocks == base && !snapshot_update) {
        // An identical ordinary-KV and KDA frontier is a true no-op, not a
        // capacity truncation. Preserve generation monotonicity without
        // creating a reservation or contaminating pressure metrics.
        if (existing != sessions_.end()) {
            existing->second.latest_submitted_generation = generation;
        }
        return result;
    }
    const std::uint64_t required_capacity = checked_math::add<CpuKVCacheError>(
        missing, snapshot_slot_needed ? kda_snapshot_charge_blocks_ : 0,
        "CPU offload capacity requirement overflows");

    if (pressure_policy_ ==
        config::CpuKVCacheCapacityPressurePolicy::kSkipOffload) {
        std::uint64_t evictable = available_blocks();
        for (const auto &[candidate_id, candidate] : sessions_) {
            if (candidate_id == session_id ||
                !candidate.active_reservations.empty() ||
                candidate.aggregate_restore_pins != 0 ||
                candidate.discard_pending) {
                continue;
            }
            const std::uint64_t count =
                candidate.committed_blocks + candidate.reserved_blocks;
            if (count > std::numeric_limits<std::uint64_t>::max() - evictable) {
                throw CpuKVCacheError("CPU evictable capacity overflows");
            }
            evictable += count;
            // A snapshot is the second, indivisible step in the same
            // session-level eviction chain.  It contributes to reclaimable
            // capacity only when its pins permit removal after ordinary KV
            // reaches zero.  A pinned snapshot does not prevent its
            // unpinned ordinary suffix from being reclaimed.
            const auto snapshot = kda_snapshots_.find(candidate_id);
            const bool snapshot_reclaimable =
                snapshot != kda_snapshots_.end() &&
                candidate.aggregate_snapshot_pins == 0 &&
                (candidate.committed_blocks != 0 ||
                 can_evict_kda_snapshot(candidate_id));
            if (kda_snapshot_charge_blocks_ != 0 && snapshot_reclaimable) {
                if (kda_snapshot_charge_blocks_ >
                    std::numeric_limits<std::uint64_t>::max() - evictable) {
                    throw CpuKVCacheError("CPU evictable capacity overflows");
                }
                evictable += kda_snapshot_charge_blocks_;
            }
        }
        if (evictable < required_capacity) {
            result.skipped = true;
            ++stats_.skipped_offloads;
            if (existing != sessions_.end()) {
                existing->second.latest_submitted_generation = generation;
            }
            return result;
        }
    }

    // Reserve a first snapshot slot before ordinary blocks.  This makes the
    // fixed-size object atomic: ordinary KV can be truncated, but a snapshot
    // is never admitted partially.
    if (snapshot_slot_needed) {
        if (!reserve_kda_snapshot_slot(session_id)) {
            result.truncated = true;
            ++stats_.truncated_offloads;
            if (existing != sessions_.end()) {
                existing->second.latest_submitted_generation = generation;
            }
            return result;
        }
    }

    static_cast<void>(evict_for(missing, session_id));
    const std::uint64_t admitted = std::min(missing, available_blocks());
    if (admitted == 0 && !snapshot_update) {
        result.truncated = true;
        ++stats_.truncated_offloads;
        if (existing != sessions_.end()) {
            existing->second.latest_submitted_generation = generation;
        }
        // No snapshot slot can have been reserved for this path, but keep the
        // cleanup explicit if future admission logic changes that invariant.
        release_kda_snapshot_slot(session_id);
        return result;
    }

    SessionState &session = sessions_[session_id];
    session.latest_submitted_generation = generation;
    const CpuOffloadReservationId reservation_id =
        reservation_ids_.next("CPU offload reservation ID space exhausted");
    OffloadReservation reservation{};
    reservation.id = reservation_id;
    reservation.session_id = session_id;
    reservation.generation = generation;
    reservation.desired_frontier_blocks = desired_frontier_blocks;
    reservation.admitted_frontier_blocks = base + admitted;
    reservation.begin_block = base;
    reservation.submitted_at = submitted_at;
    reservation.includes_kda_snapshot = snapshot_update;
    reservation.snapshot_slot_reserved = snapshot_slot_needed;
    reservation.kda_snapshot_frontier_blocks = desired_frontier_blocks;
    reservation.truncated = admitted < missing;
    if (admitted >
            std::numeric_limits<std::uint64_t>::max() - reserved_blocks_ ||
        admitted > std::numeric_limits<std::uint64_t>::max() -
                       session.reserved_blocks) {
        throw CpuKVCacheError("CPU reserved range accounting overflows");
    }
    reserved_blocks_ += admitted;
    session.reserved_blocks += admitted;
    if ((snapshot_update || admitted > 0) && has_snapshot &&
        snapshot_position->second.in_lru) {
        remove_kda_snapshot_from_lru(snapshot_position->second);
    }
    session.active_reservations.insert(reservation_id);
    session.reserved_frontier_blocks = base + admitted;
    if (!reservations_.emplace(reservation_id, std::move(reservation)).second) {
        throw CpuKVCacheError("duplicate CPU offload reservation");
    }
    if (snapshot_slot_needed &&
        !kda_snapshot_slot_owners_.emplace(session_id, reservation_id).second) {
        throw CpuKVCacheError("duplicate CPU KDA snapshot slot owner");
    }
    result.reservation_id = reservation_id;
    result.admitted_frontier_blocks = base + admitted;
    result.reserved_blocks = admitted;
    result.truncated = admitted < missing;
    if (snapshot_update) {
        result.kda_snapshot_blocks = kda_snapshot_charge_blocks_;
        result.kda_snapshot_bytes = kda_snapshot_charge_bytes_;
    }
    if (result.truncated) {
        ++stats_.truncated_offloads;
    }
    record_occupancy_peaks();
    validate_local_invariants();
    return result;
}

bool CpuKVCacheManager::commit_offload(CpuOffloadReservationId reservation_id,
                                       SimTime completed_at) {
    auto position = reservations_.find(reservation_id);
    if (position == reservations_.end()) {
        if (reservation_ids_.was_issued(reservation_id)) {
            ++stats_.stale_generation_completions;
            return false;
        }
        throw CpuKVCacheError("unknown CPU offload reservation");
    }
    OffloadReservation &reservation = position->second;
    if (reservation.state != CpuOffloadReservationState::kPending) {
        ++stats_.stale_generation_completions;
        return false;
    }
    if (!completed_at.valid() || completed_at < reservation.submitted_at) {
        throw CpuKVCacheError("invalid CPU offload completion time");
    }
    auto session_position = sessions_.find(reservation.session_id);
    if (session_position == sessions_.end()) {
        throw CpuKVCacheError("CPU offload session disappeared");
    }
    SessionState &session = session_position->second;
    if (session.discard_pending) {
        reservation.retired_completion_received = true;
        const SessionId session_id = reservation.session_id;
        ++stats_.discarded_offload_completions;
        maybe_reap_discarded_session(session_id);
        validate_local_invariants();
        return true;
    }
    const auto current_snapshot = kda_snapshots_.find(reservation.session_id);
    const bool stale_snapshot =
        current_snapshot != kda_snapshots_.end() &&
        current_snapshot->second.generation.valid() &&
        reservation.generation < current_snapshot->second.generation;
    if ((session.latest_committed_generation.valid() &&
         reservation.generation < session.latest_committed_generation) ||
        stale_snapshot) {
        ++stats_.stale_generation_completions;
    }
    if (reservation.admitted_frontier_blocks < reservation.begin_block) {
        throw CpuKVCacheError("CPU offload reservation range is inverted");
    }
    const std::uint64_t committed =
        reservation.admitted_frontier_blocks - reservation.begin_block;
    if (committed > reserved_blocks_ || committed > session.reserved_blocks ||
        committed >
            std::numeric_limits<std::uint64_t>::max() - resident_blocks_ ||
        committed > std::numeric_limits<std::uint64_t>::max() -
                        session.committed_blocks) {
        throw CpuKVCacheError("CPU offload range accounting diverged");
    }
    reserved_blocks_ -= committed;
    session.reserved_blocks -= committed;
    resident_blocks_ += committed;
    session.committed_blocks += committed;
    session.active_reservations.erase(reservation_id);
    session.last_commit_time = completed_at;
    session.last_access_time = completed_at;
    if (!session.latest_committed_generation.valid() ||
        reservation.generation > session.latest_committed_generation) {
        session.latest_committed_generation = reservation.generation;
    }
    if (reservation.includes_kda_snapshot && !stale_snapshot) {
        auto snapshot = kda_snapshots_.find(reservation.session_id);
        if (snapshot == kda_snapshots_.end()) {
            if (kda_snapshot_reserved_blocks_ < kda_snapshot_charge_blocks_) {
                throw CpuKVCacheError(
                    "CPU KDA snapshot commit lost its reserved slot");
            }
            const auto slot_owner =
                kda_snapshot_slot_owners_.find(reservation.session_id);
            if (slot_owner == kda_snapshot_slot_owners_.end()) {
                throw CpuKVCacheError(
                    "CPU KDA snapshot reserved slot has no owner");
            }
            const auto owner_reservation =
                reservations_.find(slot_owner->second);
            if (owner_reservation == reservations_.end() ||
                owner_reservation->second.session_id !=
                    reservation.session_id ||
                owner_reservation->second.state !=
                    CpuOffloadReservationState::kPending ||
                !owner_reservation->second.snapshot_slot_reserved) {
                throw CpuKVCacheError(
                    "CPU KDA snapshot reserved slot owner is invalid");
            }
            owner_reservation->second.snapshot_slot_reserved = false;
            kda_snapshot_slot_owners_.erase(slot_owner);
            kda_snapshot_reserved_blocks_ -= kda_snapshot_charge_blocks_;
            if (kda_snapshot_occupied_blocks_ >
                std::numeric_limits<std::uint64_t>::max() -
                    kda_snapshot_charge_blocks_) {
                throw CpuKVCacheError("CPU KDA snapshot occupancy overflows");
            }
            KdaSnapshot state{};
            state.frontier_blocks = reservation.kda_snapshot_frontier_blocks;
            state.generation = reservation.generation;
            auto inserted = kda_snapshots_.emplace(reservation.session_id,
                                                   std::move(state));
            if (!inserted.second) {
                throw CpuKVCacheError("CPU KDA snapshot commit raced");
            }
            kda_snapshot_occupied_blocks_ += kda_snapshot_charge_blocks_;
            snapshot = inserted.first;
        } else {
            if (snapshot->second.in_lru) {
                remove_kda_snapshot_from_lru(snapshot->second);
            }
            snapshot->second.frontier_blocks =
                reservation.kda_snapshot_frontier_blocks;
            snapshot->second.generation = reservation.generation;
        }
    }
    const auto committed_snapshot = kda_snapshots_.find(reservation.session_id);
    if (committed_snapshot != kda_snapshots_.end() &&
        session.committed_blocks == 0 && session.reserved_blocks == 0 &&
        session.active_reservations.empty() &&
        session.aggregate_restore_pins == 0 &&
        session.aggregate_snapshot_pins == 0 && !session.discard_pending &&
        !committed_snapshot->second.in_lru) {
        append_kda_snapshot_to_lru(reservation.session_id,
                                   committed_snapshot->second);
    }
    reservation.state = CpuOffloadReservationState::kCommitted;
    publish_committed_range(session, reservation.begin_block,
                            reservation.admitted_frontier_blocks);
    ++stats_.committed_offloads;
    record_occupancy_peaks();
    reservations_.erase(position);
    validate_local_invariants();
    return true;
}

bool CpuKVCacheManager::abort_offload(CpuOffloadReservationId reservation_id) {
    auto position = reservations_.find(reservation_id);
    if (position == reservations_.end()) {
        if (reservation_ids_.was_issued(reservation_id)) {
            return false;
        }
        throw CpuKVCacheError("unknown CPU offload reservation");
    }
    OffloadReservation &target = position->second;
    if (target.state != CpuOffloadReservationState::kPending) {
        return false;
    }
    auto session_position = sessions_.find(target.session_id);
    if (session_position == sessions_.end()) {
        throw CpuKVCacheError("CPU offload session disappeared");
    }
    SessionState &session = session_position->second;
    const std::uint64_t gap = target.begin_block;
    std::vector<CpuOffloadReservationId> aborted_reservations;
    for (auto &[id, reservation] : reservations_) {
        if (reservation.session_id == target.session_id &&
            reservation.state == CpuOffloadReservationState::kPending &&
            reservation.begin_block >= gap) {
            reservation.state = CpuOffloadReservationState::kAborted;
            session.active_reservations.erase(id);
            const std::uint64_t length =
                reservation.admitted_frontier_blocks - reservation.begin_block;
            if (length > reserved_blocks_ || length > session.reserved_blocks) {
                throw CpuKVCacheError(
                    "aborted CPU reservation accounting underflow");
            }
            reserved_blocks_ -= length;
            session.reserved_blocks -= length;
            aborted_reservations.push_back(id);
            ++stats_.aborted_offloads;
        }
    }
    auto committed = session.committed_out_of_order_ranges.lower_bound(gap);
    while (committed != session.committed_out_of_order_ranges.end()) {
        const std::uint64_t length = committed->second - committed->first;
        if (length > resident_blocks_ || length > session.committed_blocks) {
            throw CpuKVCacheError(
                "aborted committed CPU suffix accounting underflow");
        }
        resident_blocks_ -= length;
        session.committed_blocks -= length;
        committed = session.committed_out_of_order_ranges.erase(committed);
    }
    session.committed_frontier_blocks =
        std::min(session.committed_frontier_blocks, gap);
    session.reserved_frontier_blocks =
        std::min(session.reserved_frontier_blocks, gap);
    const SessionId session_id = target.session_id;
    const auto snapshot = kda_snapshots_.find(session_id);
    if (snapshot != kda_snapshots_.end() && session.committed_blocks == 0 &&
        session.reserved_blocks == 0 && session.active_reservations.empty() &&
        session.aggregate_restore_pins == 0 &&
        session.aggregate_snapshot_pins == 0 && !session.discard_pending &&
        !snapshot->second.in_lru) {
        append_kda_snapshot_to_lru(session_id, snapshot->second);
    }
    release_kda_snapshot_slot(session_id);
    erase_session_if_empty(session_id);
    for (const CpuOffloadReservationId id : aborted_reservations) {
        reservations_.erase(id);
    }
    validate_local_invariants();
    return true;
}

bool CpuKVCacheManager::discard_session(SessionId session_id) {
    if (!session_id.valid()) {
        return false;
    }
    auto position = sessions_.find(session_id);
    if (position == sessions_.end()) {
        return false;
    }
    position->second.discard_pending = true;
    maybe_reap_discarded_session(session_id);
    validate_local_invariants();
    return true;
}

bool CpuKVCacheManager::session_discard_pending(
    SessionId session_id) const noexcept {
    const auto position = sessions_.find(session_id);
    return position != sessions_.end() && position->second.discard_pending;
}

bool CpuKVCacheManager::has_kda_snapshot(SessionId session_id) const noexcept {
    return kda_snapshot_charge_blocks_ != 0 &&
           kda_snapshots_.find(session_id) != kda_snapshots_.end();
}

std::uint64_t CpuKVCacheManager::kda_snapshot_frontier_blocks(
    SessionId session_id) const noexcept {
    const auto position = kda_snapshots_.find(session_id);
    return position == kda_snapshots_.end() ? 0
                                            : position->second.frontier_blocks;
}

std::uint64_t CpuKVCacheManager::discard_kda_snapshot(SessionId session_id) {
    const auto session = sessions_.find(session_id);
    const auto snapshot = kda_snapshots_.find(session_id);
    if (snapshot == kda_snapshots_.end()) {
        return 0;
    }
    if (session != sessions_.end() &&
        (session->second.aggregate_snapshot_pins != 0 ||
         !session->second.active_reservations.empty())) {
        return 0;
    }
    const std::uint64_t charge = kda_snapshot_charge_blocks_;
    remove_kda_snapshot(session_id, false);
    validate_local_invariants();
    return charge;
}

} // namespace frontier::kv_cache
