#include "frontier/kv_cache/cpu_kv_cache_manager.h"

#include <algorithm>
#include <limits>
#include <tuple>

namespace frontier::kv_cache {

CpuKVCacheManager::CpuKVCacheManager(
    std::uint64_t capacity_blocks,
    config::CpuKVCacheCapacityPressurePolicy pressure_policy)
    : capacity_blocks_(capacity_blocks), pressure_policy_(pressure_policy) {
    if (capacity_blocks_ == 0) {
        throw CpuKVCacheError("CPU KV cache capacity must be positive");
    }
}

CpuPrefixLookupResult
CpuKVCacheManager::lookup(SessionId session_id,
                          std::uint64_t query_blocks) const noexcept {
    CpuPrefixLookupResult result{query_blocks, 0};
    const auto position = sessions_.find(session_id);
    if (!session_id.valid() || position == sessions_.end()) {
        return result;
    }
    result.hit_blocks = std::min(
        query_blocks, position->second.committed_frontier_blocks);
    return result;
}

void CpuKVCacheManager::record_successful_lookup(
    RequestId request_id, CpuPrefixLookupResult result, SessionId session_id) {
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
    if (next_block_id_ >
        static_cast<std::uint64_t>(
            std::numeric_limits<CpuBlockId::ValueType>::max())) {
        throw CpuKVCacheError("CPU block ID space exhausted");
    }
    return CpuBlockId{next_block_id_++};
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
    return capacity_blocks_ - resident_blocks_ - reserved_blocks_;
}

std::uint64_t CpuKVCacheManager::evict_for(
    std::uint64_t required, SessionId excluded_session) {
    while (available_blocks() < required) {
        auto victim = sessions_.end();
        for (auto position = sessions_.begin(); position != sessions_.end();
             ++position) {
            const SessionState &candidate = position->second;
            if (position->first == excluded_session ||
                candidate.blocks.empty() ||
                !candidate.active_reservations.empty() ||
                candidate.aggregate_restore_pins != 0) {
                continue;
            }
            const auto key = std::make_tuple(
                candidate.last_access_time.seconds(),
                candidate.last_commit_time.seconds(), position->first.value());
            if (victim == sessions_.end() ||
                key < std::make_tuple(
                          victim->second.last_access_time.seconds(),
                          victim->second.last_commit_time.seconds(),
                          victim->first.value())) {
                victim = position;
            }
        }
        if (victim == sessions_.end()) {
            break;
        }
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
        const std::uint64_t remaining = suffix_end - reclaim;
        session.committed_frontier_blocks =
            std::min(session.committed_frontier_blocks, remaining);
        session.reserved_frontier_blocks =
            std::min(session.reserved_frontier_blocks, remaining);
        if (session.blocks.empty()) {
            sessions_.erase(victim);
            ++stats_.evicted_sessions;
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
        position->second.aggregate_restore_pins == 0) {
        sessions_.erase(position);
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
    if (existing != sessions_.end() &&
        existing->second.latest_submitted_generation.valid() &&
        generation <= existing->second.latest_submitted_generation) {
        throw CpuKVCacheError("CPU offload generation must be monotonic");
    }
    const std::uint64_t base =
        existing == sessions_.end()
            ? 0
            : existing->second.reserved_frontier_blocks;
    CpuOffloadReservationResult result{};
    result.desired_frontier_blocks = desired_frontier_blocks;
    result.admitted_frontier_blocks =
        std::min(desired_frontier_blocks, base);
    if (desired_frontier_blocks <= base) {
        if (existing != sessions_.end()) {
            existing->second.latest_submitted_generation = generation;
        }
        return result;
    }
    const std::uint64_t missing = desired_frontier_blocks - base;
    if (pressure_policy_ ==
        config::CpuKVCacheCapacityPressurePolicy::kSkipOffload) {
        std::uint64_t evictable = available_blocks();
        for (const auto &[candidate_id, candidate] : sessions_) {
            if (candidate_id != session_id &&
                candidate.active_reservations.empty() &&
                candidate.aggregate_restore_pins == 0) {
                const auto count =
                    static_cast<std::uint64_t>(candidate.blocks.size());
                if (count >
                    std::numeric_limits<std::uint64_t>::max() - evictable) {
                    throw CpuKVCacheError("CPU evictable capacity overflows");
                }
                evictable += count;
            }
        }
        if (evictable < missing) {
            result.skipped = true;
            ++stats_.skipped_offloads;
            if (existing != sessions_.end()) {
                existing->second.latest_submitted_generation = generation;
            }
            return result;
        }
    }

    static_cast<void>(evict_for(missing, session_id));
    const std::uint64_t admitted = std::min(missing, available_blocks());
    if (admitted == 0) {
        result.truncated = true;
        ++stats_.truncated_offloads;
        if (existing != sessions_.end()) {
            existing->second.latest_submitted_generation = generation;
        }
        return result;
    }

    SessionState &session = sessions_[session_id];
    session.latest_submitted_generation = generation;
    const CpuOffloadReservationId reservation_id{next_reservation_id_++};
    OffloadReservation reservation{};
    reservation.id = reservation_id;
    reservation.session_id = session_id;
    reservation.generation = generation;
    reservation.desired_frontier_blocks = desired_frontier_blocks;
    reservation.admitted_frontier_blocks = base + admitted;
    reservation.begin_block = base;
    reservation.submitted_at = submitted_at;
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
    session.active_reservations.insert(reservation_id);
    session.reserved_frontier_blocks = base + admitted;
    reservations_.emplace(reservation_id, std::move(reservation));
    result.reservation_id = reservation_id;
    result.admitted_frontier_blocks = base + admitted;
    result.reserved_blocks = admitted;
    result.truncated = admitted < missing;
    if (result.truncated) {
        ++stats_.truncated_offloads;
    }
    record_occupancy_peaks();
    validate_local_invariants();
    return result;
}

bool CpuKVCacheManager::commit_offload(
    CpuOffloadReservationId reservation_id, SimTime completed_at) {
    auto position = reservations_.find(reservation_id);
    if (position == reservations_.end()) {
        throw CpuKVCacheError("unknown CPU offload reservation");
    }
    OffloadReservation &reservation = position->second;
    if (reservation.state != CpuOffloadReservationState::kPending) {
        ++stats_.stale_generation_completions;
        return false;
    }
    if (!completed_at.valid() ||
        completed_at < reservation.submitted_at) {
        throw CpuKVCacheError("invalid CPU offload completion time");
    }
    auto session_position = sessions_.find(reservation.session_id);
    if (session_position == sessions_.end()) {
        throw CpuKVCacheError("CPU offload session disappeared");
    }
    SessionState &session = session_position->second;
    if (session.latest_committed_generation.valid() &&
        reservation.generation < session.latest_committed_generation) {
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
    reservation.state = CpuOffloadReservationState::kCommitted;
    // Terminal reservations remain as a compact idempotency record, but do
    // not retain the potentially large block-id allocation.
    std::vector<CpuBlockId>{}.swap(reservation.block_ids);
    advance_committed_frontier(session);
    ++stats_.committed_offloads;
    record_occupancy_peaks();
    validate_local_invariants();
    return true;
}

bool CpuKVCacheManager::abort_offload(
    CpuOffloadReservationId reservation_id) {
    auto position = reservations_.find(reservation_id);
    if (position == reservations_.end()) {
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
    for (auto &[id, reservation] : reservations_) {
        if (reservation.session_id == target.session_id &&
            reservation.state == CpuOffloadReservationState::kPending &&
            reservation.begin_block >= gap) {
            reservation.state = CpuOffloadReservationState::kAborted;
            std::vector<CpuBlockId>{}.swap(reservation.block_ids);
            session.active_reservations.erase(id);
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
    erase_session_if_empty(session_id);
    validate_local_invariants();
    return true;
}

CpuRestoreLeaseId CpuKVCacheManager::pin_restore(
    SessionId session_id, std::uint64_t begin_block,
    std::uint64_t end_block, SimTime started_at) {
    if (!session_id.valid() || !started_at.valid() ||
        begin_block >= end_block) {
        throw CpuKVCacheError("invalid CPU restore lease range/time");
    }
    auto position = sessions_.find(session_id);
    if (position == sessions_.end() ||
        end_block > position->second.committed_frontier_blocks) {
        throw CpuKVCacheError("CPU restore range is not committed");
    }
    SessionState &session = position->second;
    RestoreLease lease{};
    lease.id = CpuRestoreLeaseId{next_lease_id_++};
    lease.session_id = session_id;
    lease.started_at = started_at;
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
    if (used) {
        session.last_access_time = released_at;
    }
    lease.released = true;
    // Keep only the terminal lease metadata needed for duplicate-release
    // idempotency; release the vector's backing allocation as well.
    std::vector<CpuBlockId>{}.swap(lease.block_ids);
    validate_local_invariants();
    return true;
}

std::uint64_t CpuKVCacheManager::committed_frontier_blocks(
    SessionId session_id) const noexcept {
    const auto position = sessions_.find(session_id);
    return position == sessions_.end()
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
    if (resident_blocks_ > capacity_blocks_ ||
        reserved_blocks_ > capacity_blocks_ - resident_blocks_) {
        throw CpuKVCacheError("CPU block accounting exceeds capacity");
    }
    if (resident_blocks_ + reserved_blocks_ != blocks_.size()) {
        throw CpuKVCacheError("CPU block accounting diverged");
    }
    if (pinned_blocks_ > resident_blocks_ + reserved_blocks_) {
        throw CpuKVCacheError(
            "CPU pinned block accounting exceeds materialized blocks");
    }
}

CpuKVCacheDiagnostics CpuKVCacheManager::diagnostics() const {
    validate_invariants();
    CpuKVCacheDiagnostics result{};
    result.capacity_blocks = capacity_blocks_;
    result.sessions = static_cast<std::uint64_t>(sessions_.size());
    result.resident_blocks = resident_blocks_;
    result.reserved_blocks = reserved_blocks_;
    result.materialized_blocks = resident_blocks_ + reserved_blocks_;
    result.pinned_blocks = pinned_blocks_;
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
    if (blocks_.size() > capacity_blocks_) {
        throw CpuKVCacheError("CPU resident plus reserved exceeds capacity");
    }
    std::uint64_t observed_pins = 0;
    std::uint64_t session_pins = 0;
    std::uint64_t observed_resident_blocks = 0;
    std::uint64_t observed_reserved_blocks = 0;
    std::uint64_t observed_pinned_blocks = 0;
    for (const auto &[session_id, session] : sessions_) {
        if (session.blocks.empty() && session.active_reservations.empty() &&
            session.aggregate_restore_pins == 0) {
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
            if (block == blocks_.end() || block->second.session_id != session_id ||
                block->second.logical_index != index) {
                throw CpuKVCacheError("CPU session block ownership diverged");
            }
            session_pins += block->second.pin_count;
        }
        if (session_pins < observed_pins ||
            session_pins - observed_pins != session.aggregate_restore_pins) {
            throw CpuKVCacheError("CPU aggregate restore pins diverged");
        }
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
        if (owned == session->second.blocks.end() || owned->second != block_id) {
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
    for (const auto &[id, lease] : leases_) {
        static_cast<void>(id);
        if (!lease.released) {
            ++active_leases;
            lease_pins += static_cast<std::uint64_t>(lease.block_ids.size());
        } else if (!lease.block_ids.empty()) {
            throw CpuKVCacheError(
                "terminal CPU restore lease retained block metadata");
        }
    }
    static_cast<void>(active_leases);
    if (lease_pins != observed_pins) {
        throw CpuKVCacheError("CPU restore leases do not own every pin");
    }
    for (const auto &[id, reservation] : reservations_) {
        if (reservation.state != CpuOffloadReservationState::kPending) {
            if (!reservation.block_ids.empty()) {
                throw CpuKVCacheError(
                    "terminal CPU reservation retained block metadata");
            }
            continue;
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
    }
}

} // namespace frontier::kv_cache
