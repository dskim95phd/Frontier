#include "frontier/kv_cache/cpu_kv_cache_manager.h"

#include <algorithm>
#include <limits>
#include <tuple>
#include <vector>

#include "frontier/core/checked_math.h"

namespace frontier::kv_cache {

CpuKVCacheManager::CpuKVCacheManager(
    std::uint64_t capacity_blocks,
    config::CpuKVCacheCapacityPressurePolicy pressure_policy)
    : capacity_blocks_(capacity_blocks), pressure_policy_(pressure_policy) {
    if (capacity_blocks_ == 0) {
        throw CpuKVCacheError("CPU KV cache capacity must be positive");
    }
}

void CpuKVCacheManager::configure_kda_snapshot(std::uint64_t charged_blocks,
                                               std::uint64_t charged_bytes) {
    if (charged_blocks > capacity_blocks_) {
        throw CpuKVCacheError("KDA snapshot charge exceeds CPU KV capacity");
    }
    if (charged_blocks == kda_snapshot_charge_blocks_ &&
        charged_bytes == kda_snapshot_charge_bytes_) {
        return;
    }
    if (!kda_snapshots_.empty() || kda_snapshot_reserved_blocks_ != 0) {
        throw CpuKVCacheError(
            "cannot reconfigure KDA snapshot charge while snapshots exist");
    }
    kda_snapshot_charge_blocks_ = charged_blocks;
    kda_snapshot_charge_bytes_ = charged_bytes;
}

CpuPrefixLookupResult
CpuKVCacheManager::lookup(SessionId session_id,
                          std::uint64_t query_blocks) const noexcept {
    CpuPrefixLookupResult result{query_blocks, 0};
    const auto position = sessions_.find(session_id);
    if (!session_id.valid() || position == sessions_.end() ||
        position->second.discard_pending) {
        return result;
    }
    result.hit_blocks =
        std::min(query_blocks, position->second.committed_frontier_blocks);
    if (kda_snapshot_charge_blocks_ != 0) {
        const auto snapshot = kda_snapshots_.find(session_id);
        if (snapshot == kda_snapshots_.end()) {
            result.hit_blocks = 0;
        } else {
            result.hit_blocks =
                std::min(result.hit_blocks, snapshot->second.frontier_blocks);
        }
    }
    return result;
}

void CpuKVCacheManager::record_successful_lookup(CpuPrefixLookupResult result,
                                                 SessionId session_id) {
    if (result.hit_blocks > result.query_blocks) {
        throw CpuKVCacheError("invalid CPU prefix lookup metrics");
    }
    ++stats_.successful_lookups;
    stats_.query_blocks += result.query_blocks;
    stats_.hit_blocks += result.hit_blocks;
    if (result.hit_blocks > 0 && session_id.valid() &&
        hit_sessions_.insert(session_id).second) {
        stats_.sessions_with_hits =
            static_cast<std::uint64_t>(hit_sessions_.size());
    }
}

std::uint64_t CpuKVCacheManager::available_blocks() const noexcept {
    const std::uint64_t ordinary = resident_blocks_ + reserved_blocks_;
    const std::uint64_t snapshots =
        kda_snapshot_occupied_blocks_ + kda_snapshot_reserved_blocks_;
    if (ordinary > capacity_blocks_ ||
        snapshots > capacity_blocks_ - ordinary) {
        return 0;
    }
    return capacity_blocks_ - ordinary - snapshots;
}

void CpuKVCacheManager::remove_kda_snapshot_from_lru(KdaSnapshot &snapshot) {
    if (!snapshot.in_lru) {
        throw CpuKVCacheError("KDA snapshot is not in its LRU");
    }
    kda_snapshot_lru_.erase(snapshot.lru_position);
    snapshot.in_lru = false;
}

void CpuKVCacheManager::append_kda_snapshot_to_lru(SessionId session_id,
                                                   KdaSnapshot &snapshot) {
    if (snapshot.in_lru) {
        throw CpuKVCacheError("KDA snapshot already appears in its LRU");
    }
    kda_snapshot_lru_.push_back(session_id);
    snapshot.lru_position = std::prev(kda_snapshot_lru_.end());
    snapshot.in_lru = true;
}

bool CpuKVCacheManager::session_has_pending_snapshot(
    SessionId session_id) const noexcept {
    const auto position = sessions_.find(session_id);
    if (position == sessions_.end()) {
        return false;
    }
    for (const CpuOffloadReservationId reservation_id :
         position->second.active_reservations) {
        const auto reservation = reservations_.find(reservation_id);
        if (reservation != reservations_.end() &&
            reservation->second.state == CpuOffloadReservationState::kPending &&
            reservation->second.includes_kda_snapshot) {
            return true;
        }
    }
    return false;
}

bool CpuKVCacheManager::can_evict_kda_snapshot(SessionId session_id) const {
    const auto snapshot = kda_snapshots_.find(session_id);
    const auto session = sessions_.find(session_id);
    return snapshot != kda_snapshots_.end() && session != sessions_.end() &&
           snapshot->second.in_lru && session->second.committed_blocks == 0 &&
           session->second.reserved_blocks == 0 &&
           session->second.active_reservations.empty() &&
           session->second.aggregate_restore_pins == 0 &&
           session->second.aggregate_snapshot_pins == 0 &&
           !session->second.discard_pending;
}

bool CpuKVCacheManager::reserve_kda_snapshot_slot(SessionId session_id) {
    if (kda_snapshot_charge_blocks_ == 0 ||
        kda_snapshots_.find(session_id) != kda_snapshots_.end() ||
        session_has_pending_snapshot(session_id)) {
        return false;
    }
    if (available_blocks() < kda_snapshot_charge_blocks_) {
        static_cast<void>(evict_for(kda_snapshot_charge_blocks_, session_id));
    }
    if (available_blocks() < kda_snapshot_charge_blocks_) {
        return false;
    }
    if (kda_snapshot_reserved_blocks_ >
        std::numeric_limits<std::uint64_t>::max() -
            kda_snapshot_charge_blocks_) {
        throw CpuKVCacheError("KDA snapshot reservation overflows uint64");
    }
    kda_snapshot_reserved_blocks_ += kda_snapshot_charge_blocks_;
    return true;
}

void CpuKVCacheManager::release_kda_snapshot_slot(SessionId session_id) {
    if (kda_snapshots_.find(session_id) != kda_snapshots_.end() ||
        session_has_pending_snapshot(session_id) ||
        kda_snapshot_reserved_blocks_ == 0) {
        return;
    }
    const auto owner = kda_snapshot_slot_owners_.find(session_id);
    if (owner == kda_snapshot_slot_owners_.end()) {
        return;
    }
    kda_snapshot_slot_owners_.erase(owner);
    if (kda_snapshot_reserved_blocks_ < kda_snapshot_charge_blocks_) {
        throw CpuKVCacheError("KDA snapshot reservation accounting underflow");
    }
    kda_snapshot_reserved_blocks_ -= kda_snapshot_charge_blocks_;
}

void CpuKVCacheManager::remove_kda_snapshot(SessionId session_id,
                                            bool count_eviction) {
    auto position = kda_snapshots_.find(session_id);
    if (position == kda_snapshots_.end()) {
        return;
    }
    KdaSnapshot &snapshot = position->second;
    if (snapshot.in_lru) {
        remove_kda_snapshot_from_lru(snapshot);
    }
    if (kda_snapshot_occupied_blocks_ < kda_snapshot_charge_blocks_) {
        throw CpuKVCacheError("KDA snapshot occupancy accounting underflow");
    }
    kda_snapshot_occupied_blocks_ -= kda_snapshot_charge_blocks_;
    if (count_eviction) {
        ++stats_.evicted_kda_snapshots;
        if (stats_.evicted_kda_snapshot_blocks >
            std::numeric_limits<std::uint64_t>::max() -
                kda_snapshot_charge_blocks_) {
            throw CpuKVCacheError("KDA snapshot eviction stats overflow");
        }
        stats_.evicted_kda_snapshot_blocks += kda_snapshot_charge_blocks_;
    }
    kda_snapshots_.erase(position);
    erase_session_if_empty(session_id);
}

std::uint64_t CpuKVCacheManager::evict_for(std::uint64_t required,
                                           SessionId excluded_session) {
    while (available_blocks() < required) {
        auto victim = sessions_.end();
        for (auto position = sessions_.begin(); position != sessions_.end();
             ++position) {
            const SessionState &candidate = position->second;
            if (position->first == excluded_session ||
                !candidate.active_reservations.empty() ||
                candidate.aggregate_restore_pins != 0 ||
                candidate.discard_pending) {
                continue;
            }
            const auto snapshot = kda_snapshots_.find(position->first);
            // A session contributes an eviction chain only when it owns
            // ordinary KV or an independently evictable KDA snapshot.  The
            // chain is consumed in that order below; selecting a victim from
            // the full session set is important because a zero-KV snapshot
            // can be older than a newer session's ordinary suffix.
            if (candidate.committed_blocks == 0 &&
                (snapshot == kda_snapshots_.end() ||
                 !can_evict_kda_snapshot(position->first))) {
                continue;
            }
            const auto key = std::make_tuple(
                candidate.last_access_time.seconds(),
                candidate.last_commit_time.seconds(), position->first.value());
            if (victim == sessions_.end() ||
                key < std::make_tuple(victim->second.last_access_time.seconds(),
                                      victim->second.last_commit_time.seconds(),
                                      victim->first.value())) {
                victim = position;
            }
        }
        if (victim == sessions_.end()) {
            break;
        }
        const SessionId victim_id = victim->first;
        SessionState &session = victim->second;
        const std::uint64_t need = required - available_blocks();
        if (session.reserved_blocks != 0 ||
            !session.committed_out_of_order_ranges.empty() ||
            session.committed_frontier_blocks != session.committed_blocks ||
            session.reserved_frontier_blocks != session.committed_blocks) {
            throw CpuKVCacheError(
                "evictable CPU session is not a settled contiguous prefix");
        }
        const std::uint64_t reclaim =
            std::min<std::uint64_t>(need, session.committed_blocks);
        if (reclaim > resident_blocks_) {
            throw CpuKVCacheError("CPU resident block accounting underflow");
        }
        session.committed_blocks -= reclaim;
        session.committed_frontier_blocks -= reclaim;
        session.reserved_frontier_blocks -= reclaim;
        resident_blocks_ -= reclaim;
        stats_.evicted_blocks += reclaim;
        if (reclaim != 0) {
            if (session.committed_blocks == 0) {
                const auto snapshot = kda_snapshots_.find(victim_id);
                if (snapshot != kda_snapshots_.end() &&
                    !snapshot->second.in_lru &&
                    session.active_reservations.empty() &&
                    session.aggregate_restore_pins == 0 &&
                    session.aggregate_snapshot_pins == 0 &&
                    !session.discard_pending) {
                    append_kda_snapshot_to_lru(victim_id, snapshot->second);
                }
                erase_session_if_empty(victim_id);
                ++stats_.evicted_sessions;
            }
        }

        // Keep reclaiming from the selected session before considering any
        // other session.  Once its ordinary suffix has reached zero, its KDA
        // snapshot is one indivisible next step in the same chain.  If the
        // ordinary suffix alone satisfied the request, leave that snapshot
        // in the LRU for a later pressure event; this preserves true
        // session-level LRU ordering across snapshot-only and KV-bearing
        // sessions.
        if (available_blocks() < required) {
            const auto current = sessions_.find(victim_id);
            if (current != sessions_.end() &&
                current->second.committed_blocks == 0 &&
                current->second.reserved_blocks == 0 &&
                can_evict_kda_snapshot(victim_id)) {
                remove_kda_snapshot(victim_id, true);
            }
        }
    }
    return available_blocks();
}

void CpuKVCacheManager::publish_committed_range(SessionState &session,
                                                std::uint64_t begin_block,
                                                std::uint64_t end_block) {
    if (begin_block > end_block ||
        end_block > session.reserved_frontier_blocks) {
        throw CpuKVCacheError("invalid committed CPU range");
    }
    if (begin_block == end_block) {
        return;
    }
    if (begin_block < session.committed_frontier_blocks) {
        throw CpuKVCacheError("committed CPU range overlaps published prefix");
    }
    if (begin_block != session.committed_frontier_blocks) {
        if (!session.committed_out_of_order_ranges
                 .emplace(begin_block, end_block)
                 .second) {
            throw CpuKVCacheError("duplicate out-of-order CPU range");
        }
        return;
    }
    session.committed_frontier_blocks = end_block;
    while (true) {
        const auto next = session.committed_out_of_order_ranges.find(
            session.committed_frontier_blocks);
        if (next == session.committed_out_of_order_ranges.end()) {
            break;
        }
        session.committed_frontier_blocks = next->second;
        session.committed_out_of_order_ranges.erase(next);
    }
}

void CpuKVCacheManager::recompute_distinct_pinned_blocks(
    SessionState &session) {
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
    ranges.reserve(session.active_restore_leases.size());
    for (const CpuRestoreLeaseId lease_id : session.active_restore_leases) {
        const auto lease = leases_.find(lease_id);
        if (lease == leases_.end() || lease->second.released) {
            throw CpuKVCacheError("CPU session references an inactive lease");
        }
        if (lease->second.begin_block != lease->second.end_block) {
            ranges.emplace_back(lease->second.begin_block,
                                lease->second.end_block);
        }
    }
    std::sort(ranges.begin(), ranges.end());
    std::uint64_t distinct = 0;
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
    bool have_range = false;
    for (const auto &[range_begin, range_end] : ranges) {
        if (!have_range || range_begin > end) {
            if (have_range) {
                distinct += end - begin;
            }
            begin = range_begin;
            end = range_end;
            have_range = true;
        } else {
            end = std::max(end, range_end);
        }
    }
    if (have_range) {
        distinct += end - begin;
    }
    if (session.distinct_pinned_blocks > pinned_blocks_) {
        throw CpuKVCacheError("CPU pinned block accounting underflow");
    }
    pinned_blocks_ -= session.distinct_pinned_blocks;
    pinned_blocks_ += distinct;
    session.distinct_pinned_blocks = distinct;
}

void CpuKVCacheManager::erase_session_if_empty(SessionId session_id) {
    const auto position = sessions_.find(session_id);
    if (position != sessions_.end() && position->second.committed_blocks == 0 &&
        position->second.reserved_blocks == 0 &&
        position->second.active_reservations.empty() &&
        position->second.active_restore_leases.empty() &&
        position->second.aggregate_restore_pins == 0 &&
        position->second.aggregate_snapshot_pins == 0 &&
        kda_snapshots_.find(session_id) == kda_snapshots_.end() &&
        !session_has_pending_snapshot(session_id)) {
        sessions_.erase(position);
    }
}

void CpuKVCacheManager::maybe_reap_discarded_session(SessionId session_id) {
    auto position = sessions_.find(session_id);
    if (position == sessions_.end() || !position->second.discard_pending ||
        position->second.aggregate_restore_pins != 0 ||
        position->second.aggregate_snapshot_pins != 0) {
        return;
    }
    SessionState &session = position->second;
    for (const CpuOffloadReservationId reservation_id :
         session.active_reservations) {
        const auto reservation = reservations_.find(reservation_id);
        if (reservation == reservations_.end() ||
            reservation->second.state != CpuOffloadReservationState::kPending) {
            throw CpuKVCacheError(
                "discarded CPU session has an invalid reservation");
        }
        if (!reservation->second.retired_completion_received) {
            return;
        }
    }

    const std::uint64_t discarded_resident = session.committed_blocks;
    const std::uint64_t discarded_reserved = session.reserved_blocks;
    if (discarded_resident > resident_blocks_ ||
        discarded_reserved > reserved_blocks_) {
        throw CpuKVCacheError("discarded CPU range accounting underflow");
    }
    resident_blocks_ -= discarded_resident;
    reserved_blocks_ -= discarded_reserved;
    std::vector<CpuOffloadReservationId> terminal_reservations{
        session.active_reservations.begin(), session.active_reservations.end()};
    for (const CpuOffloadReservationId reservation_id : terminal_reservations) {
        reservations_.at(reservation_id).state =
            CpuOffloadReservationState::kAborted;
    }
    const bool materialized = discarded_resident + discarded_reserved != 0;
    stats_.evicted_blocks += discarded_resident;
    stats_.evicted_sessions += static_cast<std::uint64_t>(materialized);
    session.active_reservations.clear();
    remove_kda_snapshot(session_id, false);
    // A first snapshot update can hold a reserved fixed-charge slot without
    // having published an object yet.  Once all retired completions drain,
    // release that slot before dropping the session metadata.
    release_kda_snapshot_slot(session_id);
    sessions_.erase(session_id);
    for (const CpuOffloadReservationId reservation_id : terminal_reservations) {
        reservations_.erase(reservation_id);
    }
}

void CpuKVCacheManager::record_occupancy_peaks() noexcept {
    stats_.peak_resident_blocks =
        std::max(stats_.peak_resident_blocks, resident_blocks_);
    stats_.peak_reserved_blocks =
        std::max(stats_.peak_reserved_blocks, reserved_blocks_);
}

} // namespace frontier::kv_cache
