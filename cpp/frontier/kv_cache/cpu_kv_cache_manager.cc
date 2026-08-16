#include "frontier/kv_cache/cpu_kv_cache_manager.h"

#include <algorithm>
#include <limits>
#include <tuple>

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

void CpuKVCacheManager::record_successful_lookup(RequestId request_id,
                                                 CpuPrefixLookupResult result,
                                                 SessionId session_id) {
    if (!request_id.valid() || result.hit_blocks > result.query_blocks) {
        throw CpuKVCacheError("invalid CPU prefix lookup metrics");
    }
    if (!recorded_lookup_requests_.insert(request_id).second) {
        return;
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

CpuBlockId CpuKVCacheManager::allocate_block_id() {
    if (!recycled_block_ids_.empty()) {
        const CpuBlockId id = recycled_block_ids_.back();
        recycled_block_ids_.pop_back();
        return id;
    }
    return block_ids_.next("CPU block ID space exhausted");
}

void CpuKVCacheManager::free_block(CpuBlockId block_id) {
    const auto position = blocks_.find(block_id);
    if (position == blocks_.end() || position->second.pin_count != 0) {
        throw CpuKVCacheError("cannot free missing or pinned CPU block");
    }
    if (position->second.state == CpuBlockState::kCommitted) {
        if (resident_blocks_ == 0) {
            throw CpuKVCacheError("CPU resident block accounting underflow");
        }
        --resident_blocks_;
    } else {
        if (reserved_blocks_ == 0) {
            throw CpuKVCacheError("CPU reserved block accounting underflow");
        }
        --reserved_blocks_;
    }
    blocks_.erase(position);
    recycled_block_ids_.push_back(block_id);
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
           snapshot->second.in_lru && session->second.blocks.empty() &&
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
            if (candidate.blocks.empty() &&
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
        const std::uint64_t reclaim = std::min<std::uint64_t>(
            need, static_cast<std::uint64_t>(session.blocks.size()));
        // Session block logical indices are materialized as a contiguous
        // prefix and eviction always removes its suffix.  Remove those keys
        // directly instead of searching the whole unordered map for the
        // maximum on every iteration (which made this path O(N^2)).
        const std::uint64_t suffix_end =
            static_cast<std::uint64_t>(session.blocks.size());
        for (std::uint64_t count = 0; count < reclaim; ++count) {
            if (session.blocks.empty() || suffix_end <= count) {
                throw CpuKVCacheError("CPU session suffix eviction underflow");
            }
            const std::uint64_t logical_index = suffix_end - count - 1;
            auto suffix = session.blocks.find(logical_index);
            if (suffix == session.blocks.end()) {
                throw CpuKVCacheError("CPU session suffix eviction gap");
            }
            const CpuBlockId block_id = suffix->second;
            session.blocks.erase(suffix);
            free_block(block_id);
            ++stats_.evicted_blocks;
        }
        if (suffix_end != 0) {
            const std::uint64_t remaining =
                static_cast<std::uint64_t>(session.blocks.size());
            session.committed_frontier_blocks =
                std::min(session.committed_frontier_blocks, remaining);
            session.reserved_frontier_blocks =
                std::min(session.reserved_frontier_blocks, remaining);
            if (session.blocks.empty()) {
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
            if (current != sessions_.end() && current->second.blocks.empty() &&
                can_evict_kda_snapshot(victim_id)) {
                remove_kda_snapshot(victim_id, true);
            }
        }
    }
    return available_blocks();
}

void CpuKVCacheManager::advance_committed_frontier(SessionState &session) {
    // Reservations append a contiguous suffix.  A commit can therefore only
    // expose new blocks starting at the current committed frontier; an
    // out-of-order suffix commit stops at the first still-reserved block.
    while (session.committed_frontier_blocks <
           session.reserved_frontier_blocks) {
        const auto owned =
            session.blocks.find(session.committed_frontier_blocks);
        if (owned == session.blocks.end()) {
            break;
        }
        const auto block = blocks_.find(owned->second);
        if (block == blocks_.end() ||
            block->second.state != CpuBlockState::kCommitted) {
            break;
        }
        ++session.committed_frontier_blocks;
    }
}

void CpuKVCacheManager::erase_session_if_empty(SessionId session_id) {
    const auto position = sessions_.find(session_id);
    if (position != sessions_.end() && position->second.blocks.empty() &&
        position->second.active_reservations.empty() &&
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

    std::uint64_t discarded_resident = 0;
    std::vector<CpuBlockId> discarded_blocks;
    discarded_blocks.reserve(session.blocks.size());
    for (const auto &[index, block_id] : session.blocks) {
        static_cast<void>(index);
        const auto block = blocks_.find(block_id);
        if (block == blocks_.end() || block->second.pin_count != 0) {
            throw CpuKVCacheError(
                "discarded CPU session retained an invalid block");
        }
        discarded_resident += static_cast<std::uint64_t>(
            block->second.state == CpuBlockState::kCommitted);
        discarded_blocks.push_back(block_id);
    }
    for (const CpuBlockId block_id : discarded_blocks) {
        free_block(block_id);
    }
    std::vector<CpuOffloadReservationId> terminal_reservations{
        session.active_reservations.begin(), session.active_reservations.end()};
    for (const CpuOffloadReservationId reservation_id :
         terminal_reservations) {
        reservations_.at(reservation_id).state =
            CpuOffloadReservationState::kAborted;
    }
    const bool materialized = !session.blocks.empty();
    stats_.evicted_blocks += discarded_resident;
    stats_.evicted_sessions += static_cast<std::uint64_t>(materialized);
    session.active_reservations.clear();
    remove_kda_snapshot(session_id, false);
    // A first snapshot update can hold a reserved fixed-charge slot without
    // having published an object yet.  Once all retired completions drain,
    // release that slot before dropping the session metadata.
    release_kda_snapshot_slot(session_id);
    sessions_.erase(position);
    for (const CpuOffloadReservationId reservation_id :
         terminal_reservations) {
        reservations_.erase(reservation_id);
    }
}

void CpuKVCacheManager::record_occupancy_peaks() noexcept {
    stats_.peak_resident_blocks =
        std::max(stats_.peak_resident_blocks, resident_blocks_);
    stats_.peak_reserved_blocks =
        std::max(stats_.peak_reserved_blocks, reserved_blocks_);
}

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
    const std::uint64_t required_capacity =
        checked_math::add<CpuKVCacheError>(
            missing,
            snapshot_slot_needed ? kda_snapshot_charge_blocks_ : 0,
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
                static_cast<std::uint64_t>(candidate.blocks.size());
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
                (!candidate.blocks.empty() ||
                 can_evict_kda_snapshot(candidate_id));
            if (kda_snapshot_charge_blocks_ != 0 &&
                snapshot_reclaimable) {
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
    reservation.block_ids.reserve(static_cast<std::size_t>(admitted));
    for (std::uint64_t offset = 0; offset < admitted; ++offset) {
        const std::uint64_t logical_index = base + offset;
        const CpuBlockId block_id = allocate_block_id();
        CpuBlock block{};
        block.id = block_id;
        block.session_id = session_id;
        block.logical_index = logical_index;
        block.reservation_id = reservation_id;
        block.generation = generation;
        if (!blocks_.emplace(block_id, block).second ||
            !session.blocks.emplace(logical_index, block_id).second) {
            throw CpuKVCacheError("duplicate CPU block materialization");
        }
        ++reserved_blocks_;
        reservation.block_ids.push_back(block_id);
    }
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
    for (const CpuBlockId block_id : reservation.block_ids) {
        auto block = blocks_.find(block_id);
        if (block == blocks_.end() ||
            block->second.state != CpuBlockState::kReserved ||
            block->second.reservation_id != reservation_id) {
            throw CpuKVCacheError("CPU offload reservation blocks diverged");
        }
        block->second.state = CpuBlockState::kCommitted;
        block->second.reservation_id = CpuOffloadReservationId{};
        if (reserved_blocks_ == 0) {
            throw CpuKVCacheError("CPU reserved block accounting underflow");
        }
        --reserved_blocks_;
        ++resident_blocks_;
    }
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
            kda_snapshot_reserved_blocks_ -= kda_snapshot_charge_blocks_;
            const auto slot_owner =
                kda_snapshot_slot_owners_.find(reservation.session_id);
            if (slot_owner == kda_snapshot_slot_owners_.end()) {
                throw CpuKVCacheError(
                    "CPU KDA snapshot reserved slot has no owner");
            }
            kda_snapshot_slot_owners_.erase(slot_owner);
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
    if (committed_snapshot != kda_snapshots_.end() && session.blocks.empty() &&
        session.active_reservations.empty() &&
        session.aggregate_restore_pins == 0 &&
        session.aggregate_snapshot_pins == 0 && !session.discard_pending &&
        !committed_snapshot->second.in_lru) {
        append_kda_snapshot_to_lru(reservation.session_id,
                                   committed_snapshot->second);
    }
    reservation.state = CpuOffloadReservationState::kCommitted;
    advance_committed_frontier(session);
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
            aborted_reservations.push_back(id);
            ++stats_.aborted_offloads;
        }
    }
    std::vector<std::pair<std::uint64_t, CpuBlockId>> suffix;
    for (const auto &[index, block_id] : session.blocks) {
        if (index >= gap) {
            suffix.emplace_back(index, block_id);
        }
    }
    for (const auto &[index, block_id] : suffix) {
        session.blocks.erase(index);
        free_block(block_id);
    }
    session.committed_frontier_blocks =
        std::min(session.committed_frontier_blocks, gap);
    session.reserved_frontier_blocks =
        std::min(session.reserved_frontier_blocks, gap);
    const SessionId session_id = target.session_id;
    const auto snapshot = kda_snapshots_.find(session_id);
    if (snapshot != kda_snapshots_.end() && session.blocks.empty() &&
        session.active_reservations.empty() &&
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

CpuRestoreLeaseId CpuKVCacheManager::pin_restore(SessionId session_id,
                                                 std::uint64_t begin_block,
                                                 std::uint64_t end_block,
                                                 SimTime started_at) {
    if (!session_id.valid() || !started_at.valid() ||
        begin_block > end_block) {
        throw CpuKVCacheError("invalid CPU restore lease range/time");
    }
    auto position = sessions_.find(session_id);
    if (position == sessions_.end() || position->second.discard_pending ||
        end_block > position->second.committed_frontier_blocks) {
        throw CpuKVCacheError("CPU restore range is not committed");
    }
    SessionState &session = position->second;
    const auto snapshot = kda_snapshots_.find(session_id);
    const bool include_snapshot = kda_snapshot_charge_blocks_ != 0;
    if (begin_block == end_block &&
        (!include_snapshot || snapshot == kda_snapshots_.end())) {
        throw CpuKVCacheError(
            "empty CPU restore range requires a KDA snapshot");
    }
    if (include_snapshot && (snapshot == kda_snapshots_.end() ||
                             end_block > snapshot->second.frontier_blocks)) {
        throw CpuKVCacheError(
            "CPU restore range has no matching KDA snapshot frontier");
    }
    RestoreLease lease{};
    lease.id = lease_ids_.next("CPU restore lease ID space exhausted");
    lease.session_id = session_id;
    lease.started_at = started_at;
    lease.includes_kda_snapshot = include_snapshot;
    if (include_snapshot) {
        if (snapshot->second.in_lru) {
            remove_kda_snapshot_from_lru(snapshot->second);
        }
        lease.kda_snapshot_blocks = kda_snapshot_charge_blocks_;
        lease.kda_snapshot_bytes = kda_snapshot_charge_bytes_;
        ++session.aggregate_snapshot_pins;
        ++pinned_kda_snapshots_;
    }
    lease.block_ids.reserve(static_cast<std::size_t>(end_block - begin_block));
    for (std::uint64_t index = begin_block; index < end_block; ++index) {
        const auto owned = session.blocks.find(index);
        if (owned == session.blocks.end()) {
            for (const CpuBlockId pinned : lease.block_ids) {
                CpuBlock &block = blocks_.at(pinned);
                if (block.pin_count == 1) {
                    if (pinned_blocks_ == 0) {
                        throw CpuKVCacheError(
                            "CPU pinned block accounting underflow");
                    }
                    --pinned_blocks_;
                }
                --block.pin_count;
                --session.aggregate_restore_pins;
            }
            if (lease.includes_kda_snapshot) {
                if (session.aggregate_snapshot_pins == 0 ||
                    pinned_kda_snapshots_ == 0) {
                    throw CpuKVCacheError(
                        "CPU KDA snapshot pin accounting underflow");
                }
                --session.aggregate_snapshot_pins;
                --pinned_kda_snapshots_;
            }
            throw CpuKVCacheError("CPU restore range contains a gap");
        }
        CpuBlock &block = blocks_.at(owned->second);
        if (block.pin_count == 0) {
            ++pinned_blocks_;
        }
        ++block.pin_count;
        ++session.aggregate_restore_pins;
        lease.block_ids.push_back(block.id);
    }
    const CpuRestoreLeaseId lease_id = lease.id;
    leases_.emplace(lease_id, std::move(lease));
    validate_local_invariants();
    return lease_id;
}

bool CpuKVCacheManager::release_restore(CpuRestoreLeaseId lease_id, bool used,
                                        SimTime released_at) {
    auto position = leases_.find(lease_id);
    if (position == leases_.end()) {
        if (lease_ids_.was_issued(lease_id)) {
            return false;
        }
        throw CpuKVCacheError("unknown CPU restore lease");
    }
    RestoreLease &lease = position->second;
    if (lease.released) {
        return false;
    }
    if (!released_at.valid() || released_at < lease.started_at) {
        throw CpuKVCacheError("invalid CPU restore release time");
    }
    auto session_position = sessions_.find(lease.session_id);
    if (session_position == sessions_.end()) {
        throw CpuKVCacheError("pinned CPU restore session disappeared");
    }
    SessionState &session = session_position->second;
    for (const CpuBlockId block_id : lease.block_ids) {
        CpuBlock &block = blocks_.at(block_id);
        if (block.pin_count == 0 || session.aggregate_restore_pins == 0) {
            throw CpuKVCacheError("CPU restore pin accounting underflow");
        }
        if (block.pin_count == 1) {
            if (pinned_blocks_ == 0) {
                throw CpuKVCacheError("CPU pinned block accounting underflow");
            }
            --pinned_blocks_;
        }
        --block.pin_count;
        --session.aggregate_restore_pins;
    }
    if (lease.includes_kda_snapshot) {
        if (session.aggregate_snapshot_pins == 0 ||
            pinned_kda_snapshots_ == 0) {
            throw CpuKVCacheError("CPU KDA snapshot pin accounting underflow");
        }
        --session.aggregate_snapshot_pins;
        --pinned_kda_snapshots_;
        const auto snapshot = kda_snapshots_.find(lease.session_id);
        if (!session.discard_pending && snapshot != kda_snapshots_.end() &&
            session.blocks.empty() && session.active_reservations.empty() &&
            session.aggregate_restore_pins == 0 &&
            session.aggregate_snapshot_pins == 0 &&
            !snapshot->second.in_lru) {
            append_kda_snapshot_to_lru(lease.session_id, snapshot->second);
        }
    }
    if (used && !session.discard_pending) {
        session.last_access_time = released_at;
    }
    lease.released = true;
    const SessionId session_id = lease.session_id;
    leases_.erase(position);
    maybe_reap_discarded_session(session_id);
    validate_local_invariants();
    return true;
}

bool CpuKVCacheManager::restore_includes_kda_snapshot(
    CpuRestoreLeaseId lease_id) const noexcept {
    const auto position = leases_.find(lease_id);
    return position != leases_.end() && position->second.includes_kda_snapshot;
}

std::uint64_t CpuKVCacheManager::restore_kda_snapshot_blocks(
    CpuRestoreLeaseId lease_id) const noexcept {
    const auto position = leases_.find(lease_id);
    return position == leases_.end() ? 0 : position->second.kda_snapshot_blocks;
}

std::uint64_t CpuKVCacheManager::restore_kda_snapshot_bytes(
    CpuRestoreLeaseId lease_id) const noexcept {
    const auto position = leases_.find(lease_id);
    return position == leases_.end() ? 0 : position->second.kda_snapshot_bytes;
}

std::uint64_t CpuKVCacheManager::committed_frontier_blocks(
    SessionId session_id) const noexcept {
    const auto position = sessions_.find(session_id);
    return position == sessions_.end() || position->second.discard_pending
               ? 0
               : position->second.committed_frontier_blocks;
}

bool CpuKVCacheManager::reservation_pending(
    CpuOffloadReservationId reservation_id) const noexcept {
    const auto position = reservations_.find(reservation_id);
    return position != reservations_.end() &&
           position->second.state == CpuOffloadReservationState::kPending;
}

bool CpuKVCacheManager::lease_active(
    CpuRestoreLeaseId lease_id) const noexcept {
    const auto position = leases_.find(lease_id);
    return position != leases_.end() && !position->second.released;
}

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
    if (resident_blocks_ + reserved_blocks_ != blocks_.size()) {
        throw CpuKVCacheError("CPU block accounting diverged");
    }
    if (pinned_blocks_ > resident_blocks_ + reserved_blocks_) {
        throw CpuKVCacheError(
            "CPU pinned block accounting exceeds materialized blocks");
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
    if (blocks_.size() > capacity_blocks_ ||
        kda_snapshot_charge_blocks_ != 0 &&
            kda_snapshots_.size() >
                capacity_blocks_ / kda_snapshot_charge_blocks_) {
        throw CpuKVCacheError("CPU resident plus reserved exceeds capacity");
    }
    std::uint64_t observed_pins = 0;
    std::uint64_t session_pins = 0;
    std::uint64_t observed_snapshot_pins = 0;
    std::uint64_t observed_resident_blocks = 0;
    std::uint64_t observed_reserved_blocks = 0;
    std::uint64_t observed_pinned_blocks = 0;
    for (const auto &[session_id, session] : sessions_) {
        if (session.blocks.empty() && session.active_reservations.empty() &&
            session.aggregate_restore_pins == 0 &&
            session.aggregate_snapshot_pins == 0 &&
            kda_snapshots_.find(session_id) == kda_snapshots_.end() &&
            !session.discard_pending) {
            throw CpuKVCacheError("empty CPU session metadata accumulated");
        }
        std::uint64_t committed = 0;
        while (true) {
            const auto owned = session.blocks.find(committed);
            if (owned == session.blocks.end()) {
                break;
            }
            const auto block = blocks_.find(owned->second);
            if (block == blocks_.end() ||
                block->second.state != CpuBlockState::kCommitted) {
                break;
            }
            ++committed;
        }
        std::uint64_t reserved = committed;
        while (session.blocks.find(reserved) != session.blocks.end()) {
            ++reserved;
        }
        if (committed != session.committed_frontier_blocks ||
            reserved != session.reserved_frontier_blocks) {
            throw CpuKVCacheError("CPU session frontier contains a gap");
        }
        for (const auto &[index, block_id] : session.blocks) {
            const auto block = blocks_.find(block_id);
            if (block == blocks_.end() ||
                block->second.session_id != session_id ||
                block->second.logical_index != index) {
                throw CpuKVCacheError("CPU session block ownership diverged");
            }
            session_pins += block->second.pin_count;
        }
        if (session_pins < observed_pins ||
            session_pins - observed_pins != session.aggregate_restore_pins) {
            throw CpuKVCacheError("CPU aggregate restore pins diverged");
        }
        observed_snapshot_pins += session.aggregate_snapshot_pins;
        observed_pins = session_pins;
    }
    for (const auto &[block_id, block] : blocks_) {
        if (block.state == CpuBlockState::kCommitted) {
            ++observed_resident_blocks;
        } else {
            ++observed_reserved_blocks;
        }
        if (block.pin_count > 0) {
            ++observed_pinned_blocks;
        }
        const auto session = sessions_.find(block.session_id);
        if (session == sessions_.end()) {
            throw CpuKVCacheError("materialized CPU block has no session");
        }
        const auto owned = session->second.blocks.find(block.logical_index);
        if (owned == session->second.blocks.end() ||
            owned->second != block_id) {
            throw CpuKVCacheError("materialized CPU block is orphaned");
        }
    }
    if (observed_resident_blocks != resident_blocks_ ||
        observed_reserved_blocks != reserved_blocks_ ||
        observed_pinned_blocks != pinned_blocks_) {
        throw CpuKVCacheError("CPU block state accounting diverged");
    }
    std::uint64_t active_leases = 0;
    std::uint64_t lease_pins = 0;
    std::uint64_t lease_snapshot_pins = 0;
    for (const auto &[id, lease] : leases_) {
        static_cast<void>(id);
        if (!lease.released) {
            ++active_leases;
            lease_pins += static_cast<std::uint64_t>(lease.block_ids.size());
            lease_snapshot_pins +=
                static_cast<std::uint64_t>(lease.includes_kda_snapshot);
            if (lease.includes_kda_snapshot &&
                (lease.kda_snapshot_blocks != kda_snapshot_charge_blocks_ ||
                 lease.kda_snapshot_bytes != kda_snapshot_charge_bytes_)) {
                throw CpuKVCacheError(
                    "active CPU restore lease has stale KDA payload");
            }
            if (lease.includes_kda_snapshot &&
                kda_snapshots_.find(lease.session_id) == kda_snapshots_.end()) {
                throw CpuKVCacheError(
                    "active CPU restore lease lost its KDA snapshot");
            }
        } else if (!lease.block_ids.empty()) {
            throw CpuKVCacheError(
                "terminal CPU restore lease retained block metadata");
        }
    }
    static_cast<void>(active_leases);
    if (lease_pins != observed_pins) {
        throw CpuKVCacheError("CPU restore leases do not own every pin");
    }
    if (lease_snapshot_pins != observed_snapshot_pins ||
        lease_snapshot_pins != pinned_kda_snapshots_) {
        throw CpuKVCacheError("CPU KDA snapshot pins diverged");
    }
    for (const auto &[id, reservation] : reservations_) {
        if (reservation.state != CpuOffloadReservationState::kPending) {
            throw CpuKVCacheError("terminal CPU reservation was retained");
        }
        const auto session = sessions_.find(reservation.session_id);
        if (session == sessions_.end() ||
            session->second.active_reservations.find(id) ==
                session->second.active_reservations.end()) {
            throw CpuKVCacheError("pending CPU reservation is not active");
        }
        for (const CpuBlockId block_id : reservation.block_ids) {
            const auto block = blocks_.find(block_id);
            if (block == blocks_.end() ||
                block->second.state != CpuBlockState::kReserved ||
                block->second.reservation_id != id) {
                throw CpuKVCacheError("pending CPU reservation lost a block");
            }
        }
        if (reservation.includes_kda_snapshot &&
            kda_snapshot_charge_blocks_ == 0) {
            throw CpuKVCacheError(
                "pending CPU reservation carries disabled KDA snapshot");
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
            if (!session->second.blocks.empty() ||
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
    for (const auto &[session_id, reservation_id] :
         kda_snapshot_slot_owners_) {
        const auto reservation = reservations_.find(reservation_id);
        if (reservation == reservations_.end() ||
            reservation->second.session_id != session_id ||
            reservation->second.state !=
                CpuOffloadReservationState::kPending ||
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
