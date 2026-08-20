#include "frontier/kv_cache/cpu_kv_cache_manager.h"

#include <algorithm>
#include <limits>
#include <tuple>
#include <vector>

#include "frontier/core/checked_math.h"

namespace frontier::kv_cache {

void CpuKVCacheManager::validate_local_invariants() const {
    if (kda_snapshot_charge_blocks_ > capacity_blocks_ ||
        resident_blocks_ > capacity_blocks_ ||
        reserved_blocks_ > capacity_blocks_ - resident_blocks_ ||
        kda_snapshot_occupied_blocks_ >
            capacity_blocks_ - resident_blocks_ - reserved_blocks_ ||
        kda_snapshot_reserved_blocks_ > capacity_blocks_ - resident_blocks_ -
                                            reserved_blocks_ -
                                            kda_snapshot_occupied_blocks_) {
        throw CpuKVCacheError("CPU block accounting exceeds capacity");
    }
    if (pinned_blocks_ > resident_blocks_) {
        throw CpuKVCacheError(
            "CPU pinned block accounting exceeds committed blocks");
    }
    if (kda_snapshot_charge_blocks_ == 0 &&
        (kda_snapshot_occupied_blocks_ != 0 ||
         kda_snapshot_reserved_blocks_ != 0 || !kda_snapshots_.empty())) {
        throw CpuKVCacheError(
            "CPU KDA snapshot state exists while snapshot policy is disabled");
    }
}

CpuKVCacheDiagnostics CpuKVCacheManager::diagnostics() const {
    validate_invariants();
    CpuKVCacheDiagnostics result{};
    result.capacity_blocks = capacity_blocks_;
    result.sessions = static_cast<std::uint64_t>(sessions_.size());
    result.resident_blocks = resident_blocks_;
    result.reserved_blocks = reserved_blocks_;
    result.kda_snapshot_occupied_blocks = kda_snapshot_occupied_blocks_;
    result.kda_snapshot_reserved_blocks = kda_snapshot_reserved_blocks_;
    result.kda_snapshot_sessions =
        static_cast<std::uint64_t>(kda_snapshots_.size());
    result.kda_snapshot_evictable_sessions =
        static_cast<std::uint64_t>(kda_snapshot_lru_.size());
    result.materialized_blocks = resident_blocks_ + reserved_blocks_;
    result.pinned_blocks = pinned_blocks_;
    result.pinned_kda_snapshots = pinned_kda_snapshots_;
    for (const auto &[id, reservation] : reservations_) {
        static_cast<void>(id);
        result.active_reservations += static_cast<std::uint64_t>(
            reservation.state == CpuOffloadReservationState::kPending);
    }
    for (const auto &[id, lease] : leases_) {
        static_cast<void>(id);
        result.active_restore_leases +=
            static_cast<std::uint64_t>(!lease.released);
    }
    return result;
}

void CpuKVCacheManager::validate_invariants() const {
    validate_local_invariants();
    if (kda_snapshot_charge_blocks_ != 0 &&
        kda_snapshots_.size() >
            capacity_blocks_ / kda_snapshot_charge_blocks_) {
        throw CpuKVCacheError("CPU resident plus reserved exceeds capacity");
    }
    std::uint64_t observed_resident_blocks = 0;
    std::uint64_t observed_reserved_blocks = 0;
    std::uint64_t observed_pinned_blocks = 0;
    std::uint64_t observed_snapshot_pins = 0;
    for (const auto &[session_id, session] : sessions_) {
        if (session.committed_blocks == 0 && session.reserved_blocks == 0 &&
            session.active_reservations.empty() &&
            session.active_restore_leases.empty() &&
            session.aggregate_restore_pins == 0 &&
            session.aggregate_snapshot_pins == 0 &&
            kda_snapshots_.find(session_id) == kda_snapshots_.end() &&
            !session.discard_pending) {
            throw CpuKVCacheError("empty CPU session metadata accumulated");
        }
        if (session.committed_frontier_blocks >
                session.reserved_frontier_blocks ||
            session.committed_blocks > session.reserved_frontier_blocks ||
            session.reserved_blocks >
                session.reserved_frontier_blocks - session.committed_blocks ||
            session.committed_blocks + session.reserved_blocks !=
                session.reserved_frontier_blocks) {
            throw CpuKVCacheError(
                "CPU analytical frontier accounting diverged");
        }

        std::vector<std::pair<std::uint64_t, std::uint64_t>> suffix_ranges;
        suffix_ranges.reserve(session.committed_out_of_order_ranges.size() +
                              session.active_reservations.size());
        std::uint64_t counted_committed = session.committed_frontier_blocks;
        for (const auto &[begin, end] : session.committed_out_of_order_ranges) {
            if (begin <= session.committed_frontier_blocks || begin >= end ||
                end > session.reserved_frontier_blocks) {
                throw CpuKVCacheError(
                    "CPU out-of-order committed range is invalid");
            }
            counted_committed += end - begin;
            suffix_ranges.emplace_back(begin, end);
        }
        if (counted_committed != session.committed_blocks) {
            throw CpuKVCacheError("CPU committed range count diverged");
        }
        std::uint64_t counted_reserved = 0;
        for (const CpuOffloadReservationId reservation_id :
             session.active_reservations) {
            const auto reservation = reservations_.find(reservation_id);
            if (reservation == reservations_.end() ||
                reservation->second.session_id != session_id ||
                reservation->second.state !=
                    CpuOffloadReservationState::kPending ||
                reservation->second.begin_block >
                    reservation->second.admitted_frontier_blocks) {
                throw CpuKVCacheError(
                    "CPU session has an invalid active reservation");
            }
            counted_reserved += reservation->second.admitted_frontier_blocks -
                                reservation->second.begin_block;
            if (reservation->second.begin_block !=
                reservation->second.admitted_frontier_blocks) {
                suffix_ranges.emplace_back(
                    reservation->second.begin_block,
                    reservation->second.admitted_frontier_blocks);
            }
        }
        if (counted_reserved != session.reserved_blocks) {
            throw CpuKVCacheError("CPU reserved range count diverged");
        }
        std::sort(suffix_ranges.begin(), suffix_ranges.end());
        std::uint64_t cursor = session.committed_frontier_blocks;
        for (const auto &[begin, end] : suffix_ranges) {
            if (begin != cursor || end < begin) {
                throw CpuKVCacheError("CPU session frontier contains a gap");
            }
            cursor = end;
        }
        if (cursor != session.reserved_frontier_blocks) {
            throw CpuKVCacheError("CPU session suffix coverage diverged");
        }

        std::uint64_t aggregate_pins = 0;
        std::uint64_t snapshot_pins = 0;
        std::vector<std::pair<std::uint64_t, std::uint64_t>> pin_ranges;
        pin_ranges.reserve(session.active_restore_leases.size());
        for (const CpuRestoreLeaseId lease_id : session.active_restore_leases) {
            const auto lease = leases_.find(lease_id);
            if (lease == leases_.end() || lease->second.released ||
                lease->second.session_id != session_id ||
                lease->second.begin_block > lease->second.end_block ||
                lease->second.end_block > session.committed_frontier_blocks) {
                throw CpuKVCacheError(
                    "CPU session has an invalid restore lease");
            }
            aggregate_pins +=
                lease->second.end_block - lease->second.begin_block;
            if (lease->second.begin_block != lease->second.end_block) {
                pin_ranges.emplace_back(lease->second.begin_block,
                                        lease->second.end_block);
            }
            snapshot_pins +=
                static_cast<std::uint64_t>(lease->second.includes_kda_snapshot);
            if (lease->second.includes_kda_snapshot &&
                (lease->second.kda_snapshot_blocks !=
                     kda_snapshot_charge_blocks_ ||
                 lease->second.kda_snapshot_bytes !=
                     kda_snapshot_charge_bytes_ ||
                 kda_snapshots_.find(session_id) == kda_snapshots_.end())) {
                throw CpuKVCacheError(
                    "active CPU restore lease has stale KDA payload");
            }
        }
        std::sort(pin_ranges.begin(), pin_ranges.end());
        std::uint64_t distinct_pins = 0;
        std::uint64_t pin_begin = 0;
        std::uint64_t pin_end = 0;
        bool have_pin = false;
        for (const auto &[begin, end] : pin_ranges) {
            if (!have_pin || begin > pin_end) {
                if (have_pin) {
                    distinct_pins += pin_end - pin_begin;
                }
                pin_begin = begin;
                pin_end = end;
                have_pin = true;
            } else {
                pin_end = std::max(pin_end, end);
            }
        }
        if (have_pin) {
            distinct_pins += pin_end - pin_begin;
        }
        if (aggregate_pins != session.aggregate_restore_pins ||
            distinct_pins != session.distinct_pinned_blocks ||
            snapshot_pins != session.aggregate_snapshot_pins) {
            throw CpuKVCacheError("CPU restore range accounting diverged");
        }

        observed_resident_blocks += session.committed_blocks;
        observed_reserved_blocks += session.reserved_blocks;
        observed_pinned_blocks += session.distinct_pinned_blocks;
        observed_snapshot_pins += session.aggregate_snapshot_pins;
    }
    if (observed_resident_blocks != resident_blocks_ ||
        observed_reserved_blocks != reserved_blocks_ ||
        observed_pinned_blocks != pinned_blocks_ ||
        observed_snapshot_pins != pinned_kda_snapshots_) {
        throw CpuKVCacheError("CPU analytical range accounting diverged");
    }

    for (const auto &[id, reservation] : reservations_) {
        if (reservation.state != CpuOffloadReservationState::kPending) {
            throw CpuKVCacheError("terminal CPU reservation was retained");
        }
        const auto session = sessions_.find(reservation.session_id);
        if (session == sessions_.end() ||
            session->second.active_reservations.find(id) ==
                session->second.active_reservations.end() ||
            reservation.begin_block > reservation.admitted_frontier_blocks ||
            reservation.admitted_frontier_blocks >
                session->second.reserved_frontier_blocks) {
            throw CpuKVCacheError("pending CPU reservation range is invalid");
        }
        if (reservation.includes_kda_snapshot &&
            kda_snapshot_charge_blocks_ == 0) {
            throw CpuKVCacheError(
                "pending CPU reservation carries disabled KDA snapshot");
        }
    }
    for (const auto &[id, lease] : leases_) {
        if (lease.released) {
            throw CpuKVCacheError("terminal CPU restore lease was retained");
        }
        const auto session = sessions_.find(lease.session_id);
        if (session == sessions_.end() ||
            session->second.active_restore_leases.find(id) ==
                session->second.active_restore_leases.end()) {
            throw CpuKVCacheError("active CPU restore lease is not owned");
        }
    }

    std::uint64_t observed_snapshot_blocks = 0;
    std::size_t observed_snapshot_lru = 0;
    std::uint64_t observed_snapshot_slots = 0;
    for (const auto &[session_id, snapshot] : kda_snapshots_) {
        const auto session = sessions_.find(session_id);
        if (session == sessions_.end()) {
            throw CpuKVCacheError("CPU KDA snapshot has no session metadata");
        }
        if (snapshot.in_lru) {
            if (session->second.committed_blocks != 0 ||
                session->second.reserved_blocks != 0 ||
                !session->second.active_reservations.empty() ||
                session->second.aggregate_restore_pins != 0 ||
                session->second.aggregate_snapshot_pins != 0 ||
                session->second.discard_pending) {
                throw CpuKVCacheError(
                    "CPU KDA snapshot LRU contains an active session");
            }
        }
        if (observed_snapshot_blocks >
            std::numeric_limits<std::uint64_t>::max() -
                kda_snapshot_charge_blocks_) {
            throw CpuKVCacheError("CPU KDA snapshot occupancy overflows");
        }
        observed_snapshot_blocks += kda_snapshot_charge_blocks_;
    }
    for (const SessionId session_id : kda_snapshot_lru_) {
        const auto snapshot = kda_snapshots_.find(session_id);
        if (snapshot == kda_snapshots_.end() || !snapshot->second.in_lru ||
            snapshot->second.lru_position == kda_snapshot_lru_.end()) {
            throw CpuKVCacheError("CPU KDA snapshot LRU is corrupt");
        }
        ++observed_snapshot_lru;
    }
    for (const auto &[id, reservation] : reservations_) {
        static_cast<void>(id);
        if (reservation.state == CpuOffloadReservationState::kPending &&
            reservation.snapshot_slot_reserved) {
            ++observed_snapshot_slots;
        }
    }
    if (kda_snapshot_slot_owners_.size() != observed_snapshot_slots) {
        throw CpuKVCacheError("CPU KDA snapshot slot ownership diverged");
    }
    for (const auto &[session_id, reservation_id] : kda_snapshot_slot_owners_) {
        const auto reservation = reservations_.find(reservation_id);
        if (reservation == reservations_.end() ||
            reservation->second.session_id != session_id ||
            reservation->second.state != CpuOffloadReservationState::kPending ||
            !reservation->second.snapshot_slot_reserved) {
            throw CpuKVCacheError("CPU KDA snapshot slot owner is invalid");
        }
    }
    // Multiple in-flight replacements share a single first-object slot; only
    // reservations that explicitly acquired the slot contribute here.
    if (observed_snapshot_slots >
        std::numeric_limits<std::uint64_t>::max() /
            std::max<std::uint64_t>(kda_snapshot_charge_blocks_, 1)) {
        throw CpuKVCacheError("CPU KDA snapshot reservation overflows");
    }
    const std::uint64_t expected_snapshot_reserved =
        observed_snapshot_slots * kda_snapshot_charge_blocks_;
    if (observed_snapshot_blocks != kda_snapshot_occupied_blocks_ ||
        observed_snapshot_lru != kda_snapshot_lru_.size() ||
        expected_snapshot_reserved != kda_snapshot_reserved_blocks_) {
        throw CpuKVCacheError("CPU KDA snapshot accounting diverged");
    }
}

} // namespace frontier::kv_cache
