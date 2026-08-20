#include "frontier/kv_cache/replica_kv_cache_manager.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "frontier/core/checked_math.h"
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

void ReplicaKVCacheManager::configure_kda_snapshot(
    std::uint64_t charged_blocks) {
    if (charged_blocks > capacity_blocks_) {
        throw ReplicaKVCacheError(
            "KDA snapshot charge exceeds GPU KV cache capacity");
    }
    if (charged_blocks == kda_snapshot_charge_blocks_) {
        return;
    }
    if (!kda_snapshots_.empty()) {
        throw ReplicaKVCacheError(
            "cannot reconfigure KDA snapshot charge while snapshots exist");
    }
    kda_snapshot_charge_blocks_ = charged_blocks;
    validate_accounting();
}

void ReplicaKVCacheManager::remove_kda_snapshot_from_lru(
    KdaSnapshotEntry &entry) {
    if (!entry.in_lru) {
        throw ReplicaKVCacheError("KDA snapshot is not in its LRU");
    }
    kda_snapshot_lru_.erase(entry.lru_position);
    entry.in_lru = false;
}

void ReplicaKVCacheManager::append_kda_snapshot_to_lru(
    SessionId session_id, KdaSnapshotEntry &entry) {
    if (entry.in_lru) {
        throw ReplicaKVCacheError("KDA snapshot already appears in its LRU");
    }
    kda_snapshot_lru_.push_back(session_id);
    entry.lru_position = std::prev(kda_snapshot_lru_.end());
    entry.in_lru = true;
}

void ReplicaKVCacheManager::remove_from_reclaim_lru(SessionCacheEntry &entry) {
    if (!entry.in_reclaim_lru) {
        return;
    }
    reclaim_lru_.erase(entry.reclaim_lru_position);
    entry.in_reclaim_lru = false;
}

void ReplicaKVCacheManager::append_to_reclaim_lru(SessionId session_id,
                                                  SessionCacheEntry &entry) {
    if (entry.in_reclaim_lru) {
        throw ReplicaKVCacheError("session already appears in reclaim LRU");
    }
    reclaim_lru_.push_back(session_id);
    entry.reclaim_lru_position = std::prev(reclaim_lru_.end());
    entry.in_reclaim_lru = true;
}

void ReplicaKVCacheManager::sync_reclaim_lru(SessionId session_id) {
    const auto position = sessions_.find(session_id);
    if (position == sessions_.end()) {
        return;
    }
    SessionCacheEntry &entry = position->second;
    const auto snapshot = kda_snapshots_.find(session_id);
    const bool inactive = !entry.active_request.valid();
    const bool has_reclaimable_state =
        inactive &&
        (entry.resident_prefix_blocks > 0 ||
         (snapshot != kda_snapshots_.end() && snapshot->second.in_lru));
    if (has_reclaimable_state) {
        if (!entry.in_reclaim_lru) {
            append_to_reclaim_lru(session_id, entry);
        }
    } else {
        remove_from_reclaim_lru(entry);
    }
}

void ReplicaKVCacheManager::touch_reclaim_lru(SessionId session_id) {
    const auto position = sessions_.find(session_id);
    if (position == sessions_.end()) {
        return;
    }
    SessionCacheEntry &entry = position->second;
    remove_from_reclaim_lru(entry);
    sync_reclaim_lru(session_id);
}

std::uint64_t ReplicaKVCacheManager::reclaimable_snapshot_blocks(
    std::optional<SessionId> excluded_session) const noexcept {
    if (kda_snapshot_charge_blocks_ == 0) {
        return 0;
    }
    std::uint64_t result = 0;
    for (const SessionId session_id : kda_snapshot_lru_) {
        if (excluded_session.has_value() && session_id == *excluded_session) {
            continue;
        }
        if (result > std::numeric_limits<std::uint64_t>::max() -
                         kda_snapshot_charge_blocks_) {
            return std::numeric_limits<std::uint64_t>::max();
        }
        result += kda_snapshot_charge_blocks_;
    }
    return result;
}

std::uint64_t ReplicaKVCacheManager::reclaimable_commitment_blocks(
    std::optional<SessionId> excluded_session,
    std::uint64_t virtual_credit) const noexcept {
    // available_commitment_blocks() excludes all future-only reservations.
    // A continuation may spend only its own virtual suffix; add that suffix
    // back, then include inactive KDA groups that the unified reclaim LRU can
    // release.  Saturate on overflow so a precheck cannot wrap around.
    std::uint64_t result = available_commitment_blocks();
    if (virtual_credit > std::numeric_limits<std::uint64_t>::max() - result) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    result += virtual_credit;
    const std::uint64_t snapshots =
        reclaimable_snapshot_blocks(excluded_session);
    if (snapshots > std::numeric_limits<std::uint64_t>::max() - result) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return result + snapshots;
}

bool ReplicaKVCacheManager::can_allocate_kda_snapshot() const {
    if (kda_snapshot_charge_blocks_ == 0) {
        return false;
    }
    return reclaimable_commitment_blocks() >= kda_snapshot_charge_blocks_;
}

void ReplicaKVCacheManager::reserve_kda_snapshot_capacity(
    SessionId session_id) {
    if (kda_snapshot_charge_blocks_ == 0 ||
        kda_snapshots_.find(session_id) != kda_snapshots_.end()) {
        return;
    }
    if (!session_id.valid()) {
        throw ReplicaKVCacheError(
            "KDA snapshot reservation requires a valid session");
    }
    // Frontier zero is a capacity reservation for the initial recurrent
    // state.  It cannot produce a prefix hit, but it guarantees that the first
    // real snapshot publication only updates an already charged atomic group.
    if (!publish_kda_snapshot(session_id, 0)) {
        throw ReplicaKVCacheError(
            "KDA snapshot capacity was not reserved during admission");
    }
    // publish_kda_snapshot() is also the public materialization API and marks
    // even frontier zero as valid. Undo only that semantic bit for this
    // internal cold-capacity reservation.
    kda_snapshots_.at(session_id).published = false;
}

std::uint64_t ReplicaKVCacheManager::effective_resident_frontier_blocks(
    SessionId session_id) const noexcept {
    const auto session = sessions_.find(session_id);
    if (session == sessions_.end()) {
        return 0;
    }
    const std::uint64_t resident = session->second.resident_prefix_blocks;
    if (kda_snapshot_charge_blocks_ == 0) {
        return resident;
    }
    const auto snapshot = kda_snapshots_.find(session_id);
    return snapshot == kda_snapshots_.end()
               ? 0
               : std::min(resident, snapshot->second.frontier_blocks);
}

void ReplicaKVCacheManager::trim_resident_frontier(SessionId session_id,
                                                   std::uint64_t keep_blocks) {
    auto position = sessions_.find(session_id);
    if (position == sessions_.end()) {
        return;
    }
    SessionCacheEntry &entry = position->second;
    if (keep_blocks > entry.resident_prefix_blocks) {
        throw ReplicaKVCacheError("KDA frontier trim exceeds resident KV");
    }
    const std::uint64_t excess = entry.resident_prefix_blocks - keep_blocks;
    if (excess == 0) {
        return;
    }
    if (entry.active_request.valid()) {
        throw ReplicaKVCacheError(
            "cannot trim a resident KV frontier owned by an active request");
    }
    if (!entry.in_evictable_lru) {
        throw ReplicaKVCacheError("resident KV frontier is not evictable");
    }
    remove_from_evictable_lru(entry);
    if (excess > resident_blocks_ ||
        blank_blocks_ > std::numeric_limits<std::uint64_t>::max() - excess) {
        throw ReplicaKVCacheError("resident KV frontier accounting underflow");
    }
    entry.resident_prefix_blocks = keep_blocks;
    resident_blocks_ -= excess;
    blank_blocks_ += excess;
    stats_.evicted_blocks += excess;
    if (keep_blocks > 0) {
        append_to_evictable_lru(session_id, entry);
    }
}

void ReplicaKVCacheManager::consume_kda_snapshot_capacity() {
    const std::uint64_t charge = kda_snapshot_charge_blocks_;
    if (charge == 0 || !can_allocate_kda_snapshot()) {
        throw ReplicaKVCacheError(
            "KDA snapshot occupancy exceeds reclaimable GPU capacity");
    }
    // Snapshot publication uses the same unified session reclaim chain as an
    // ordinary allocation.  A victim's resident suffix is trimmed first;
    // when that suffix reaches zero, its inactive snapshot is removed as one
    // atomic group before the next session can be considered.  Any group
    // surplus remains blank capacity for the newly published snapshot.
    consume_available_blocks(charge);
    if (kda_snapshot_occupied_blocks_ >
        std::numeric_limits<std::uint64_t>::max() - charge) {
        throw ReplicaKVCacheError("KDA snapshot occupancy overflows uint64");
    }
    kda_snapshot_occupied_blocks_ += charge;
}

void ReplicaKVCacheManager::erase_session_if_unowned(SessionId session_id) {
    const auto position = sessions_.find(session_id);
    if (position == sessions_.end() ||
        position->second.active_request.valid() ||
        position->second.resident_prefix_blocks != 0 ||
        kda_snapshots_.find(session_id) != kda_snapshots_.end()) {
        return;
    }
    remove_from_reclaim_lru(position->second);
    sessions_.erase(position);
    discard_on_release_.erase(session_id);
}

void ReplicaKVCacheManager::remove_kda_snapshot(SessionId session_id,
                                                bool count_eviction) {
    auto position = kda_snapshots_.find(session_id);
    if (position == kda_snapshots_.end()) {
        return;
    }
    KdaSnapshotEntry &snapshot = position->second;
    // Active snapshots are pinned and therefore absent from the LRU.  The
    // explicit bit is authoritative for inactive snapshots.
    if (snapshot.in_lru) {
        remove_kda_snapshot_from_lru(snapshot);
    }
    if (kda_snapshot_occupied_blocks_ < kda_snapshot_charge_blocks_ ||
        blank_blocks_ > std::numeric_limits<std::uint64_t>::max() -
                            kda_snapshot_charge_blocks_) {
        throw ReplicaKVCacheError(
            "KDA snapshot occupancy accounting underflow");
    }
    kda_snapshot_occupied_blocks_ -= kda_snapshot_charge_blocks_;
    blank_blocks_ += kda_snapshot_charge_blocks_;
    if (count_eviction) {
        ++stats_.evicted_kda_snapshots;
        stats_.evicted_kda_snapshot_blocks += kda_snapshot_charge_blocks_;
    }
    kda_snapshots_.erase(position);
    sync_reclaim_lru(session_id);
    erase_session_if_unowned(session_id);
}

bool ReplicaKVCacheManager::publish_kda_snapshot(
    SessionId session_id, std::uint64_t frontier_blocks) {
    if (kda_snapshot_charge_blocks_ == 0 || !session_id.valid()) {
        throw ReplicaKVCacheError("KDA snapshots are not configured");
    }
    auto existing = kda_snapshots_.find(session_id);
    if (existing != kda_snapshots_.end()) {
        existing->second.frontier_blocks = frontier_blocks;
        existing->second.published = true;
        const auto session = sessions_.find(session_id);
        if (session != sessions_.end() &&
            session->second.active_request.valid()) {
            if (existing->second.in_lru) {
                remove_kda_snapshot_from_lru(existing->second);
            }
        } else {
            if (existing->second.in_lru) {
                remove_kda_snapshot_from_lru(existing->second);
            }
            append_kda_snapshot_to_lru(session_id, existing->second);
            touch_reclaim_lru(session_id);
        }
        validate_accounting();
        return true;
    }
    if (!can_allocate_kda_snapshot()) {
        return false;
    }
    consume_kda_snapshot_capacity();
    SessionCacheEntry &session = sessions_[session_id];
    KdaSnapshotEntry snapshot{};
    snapshot.frontier_blocks = frontier_blocks;
    snapshot.published = true;
    const bool active = session.active_request.valid();
    auto inserted = kda_snapshots_.emplace(session_id, std::move(snapshot));
    if (!inserted.second) {
        throw ReplicaKVCacheError("KDA snapshot replacement raced allocation");
    }
    if (!active) {
        append_kda_snapshot_to_lru(session_id, inserted.first->second);
        touch_reclaim_lru(session_id);
    }
    validate_accounting();
    return true;
}

bool ReplicaKVCacheManager::has_kda_snapshot(
    SessionId session_id) const noexcept {
    return kda_snapshot_charge_blocks_ != 0 &&
           kda_snapshots_.find(session_id) != kda_snapshots_.end();
}

std::uint64_t ReplicaKVCacheManager::kda_snapshot_frontier_blocks(
    SessionId session_id) const noexcept {
    const auto position = kda_snapshots_.find(session_id);
    return position == kda_snapshots_.end() ? 0
                                            : position->second.frontier_blocks;
}

KdaSnapshotCheckpoint ReplicaKVCacheManager::checkpoint_kda_snapshot(
    SessionId session_id) const noexcept {
    const auto position = kda_snapshots_.find(session_id);
    if (position == kda_snapshots_.end()) {
        return {};
    }
    return KdaSnapshotCheckpoint{true, position->second.frontier_blocks,
                                 position->second.published};
}

void ReplicaKVCacheManager::restore_kda_snapshot(
    SessionId session_id, KdaSnapshotCheckpoint checkpoint) {
    auto current = kda_snapshots_.find(session_id);
    if (!checkpoint.present) {
        if (current != kda_snapshots_.end()) {
            remove_kda_snapshot(session_id, false);
        }
        validate_accounting();
        return;
    }
    if (current == kda_snapshots_.end()) {
        throw ReplicaKVCacheError(
            "cannot restore a missing GPU KDA snapshot checkpoint");
    }
    current->second.frontier_blocks = checkpoint.frontier_blocks;
    current->second.published = checkpoint.published;
    validate_accounting();
}

std::uint64_t
ReplicaKVCacheManager::discard_kda_snapshot(SessionId session_id) {
    if (!has_kda_snapshot(session_id)) {
        return 0;
    }
    remove_kda_snapshot(session_id, false);
    validate_accounting();
    return kda_snapshot_charge_blocks_;
}

void ReplicaKVCacheManager::enable_full_sequence_commitments(bool enabled) {
    if (!enabled && virtual_committed_blocks_ != 0) {
        throw ReplicaKVCacheError("cannot disable full-sequence commitments "
                                  "while reservations exist");
    }
    full_sequence_commitments_enabled_ = enabled;
    validate_accounting();
}

bool ReplicaKVCacheManager::can_commit(
    RequestId request_id, std::uint64_t full_sequence_tokens) const {
    if (!full_sequence_commitments_enabled_ || !request_id.valid() ||
        allocations_.find(request_id) != allocations_.end()) {
        return false;
    }
    const std::uint64_t blocks = full_sequence_blocks(full_sequence_tokens);
    return can_commit_blocks(blocks);
}

bool ReplicaKVCacheManager::can_commit(
    RequestId request_id, SessionId session_id,
    std::uint64_t full_sequence_tokens) const {
    if (prefix_cache_enabled_ && session_id.valid()) {
        const auto session = sessions_.find(session_id);
        if (session != sessions_.end() &&
            session->second.active_request.valid() &&
            session->second.active_request != request_id) {
            return false;
        }
    }
    if (!full_sequence_commitments_enabled_ || !request_id.valid() ||
        allocations_.find(request_id) != allocations_.end()) {
        return false;
    }
    const std::uint64_t blocks = full_sequence_blocks(full_sequence_tokens);
    return can_commit_blocks(blocks, session_id.valid()
                                         ? std::optional<SessionId>{session_id}
                                         : std::nullopt);
}

void ReplicaKVCacheManager::commit_virtual(RequestId request_id,
                                           SessionId session_id,
                                           std::uint64_t full_sequence_tokens) {
    if (!full_sequence_commitments_enabled_ || !request_id.valid() ||
        allocations_.find(request_id) != allocations_.end()) {
        throw ReplicaKVCacheError("invalid full-sequence commitment request");
    }
    const std::uint64_t committed = full_sequence_blocks(full_sequence_tokens);

    // Complete every request/session validation before reserving a cold KDA
    // snapshot. Snapshot reservation is a mutating operation and may reclaim
    // inactive sessions; a later validation failure cannot be repaired via
    // release_commitment() because no request allocation exists yet.
    if (kda_snapshot_charge_blocks_ != 0 && !session_id.valid()) {
        throw ReplicaKVCacheError(
            "KDA snapshot reservation requires a valid session");
    }
    std::uint64_t resident = 0;
    if (prefix_cache_enabled_ && session_id.valid()) {
        const auto position = sessions_.find(session_id);
        if (position != sessions_.end()) {
            if (position->second.active_request.valid() &&
                position->second.active_request != request_id) {
                throw ReplicaKVCacheError(
                    "session already has an active KV commitment");
            }
            resident = position->second.resident_prefix_blocks;
        }
    }
    if (resident > committed) {
        throw ReplicaKVCacheError(
            "resident prefix exceeds full-sequence commitment");
    }
    const std::uint64_t future = committed - resident;
    if (future >
        std::numeric_limits<std::uint64_t>::max() - virtual_committed_blocks_) {
        throw ReplicaKVCacheError("virtual commitment count overflows uint64");
    }
    if (!can_commit_blocks(committed, session_id.valid()
                                          ? std::optional<SessionId>{session_id}
                                          : std::nullopt)) {
        throw ReplicaKVCacheError(
            "full-sequence commitment exceeds capacity or watermark");
    }
    reserve_kda_snapshot_capacity(session_id);

    // Pin any GPU-resident prefix before the asynchronous restore starts.  A
    // resident cache suffix is physical capacity already in the cache, so
    // converting it to an active allocation preserves the partition while
    // preventing another request from evicting it during the restore.
    if (prefix_cache_enabled_ && session_id.valid()) {
        SessionCacheEntry &entry = sessions_[session_id];
        resident = entry.resident_prefix_blocks;
        if (resident > 0 && entry.in_evictable_lru) {
            remove_from_evictable_lru(entry);
            active_blocks_ += resident;
        }
        entry.active_request = request_id;
        const auto snapshot = kda_snapshots_.find(session_id);
        if (snapshot != kda_snapshots_.end() && snapshot->second.in_lru) {
            remove_kda_snapshot_from_lru(snapshot->second);
        }
        remove_from_reclaim_lru(entry);
    }

    // A virtual commitment consumes logical capacity without materializing a
    // physical block.  If reclaimable inactive state was needed to satisfy
    // the precheck, evict it now.  consume_available_blocks may consume a
    // blank slot as part of that operation; restore that consumption because
    // the slot remains reserved by the new virtual suffix.
    const std::uint64_t before_blank = blank_blocks_;
    const std::uint64_t strict_available = available_commitment_blocks();
    const std::uint64_t watermark_required =
        future > std::numeric_limits<std::uint64_t>::max() - watermark_blocks_
            ? std::numeric_limits<std::uint64_t>::max()
            : future + watermark_blocks_;
    if (watermark_required > strict_available) {
        const std::uint64_t deficit = watermark_required - strict_available;
        consume_available_blocks(deficit);
        if (blank_blocks_ < before_blank) {
            blank_blocks_ += before_blank - blank_blocks_;
        }
    }

    RequestKVAllocation allocation{};
    allocation.session_id = session_id;
    allocation.allocated_blocks = resident;
    allocation.published_blocks = resident;
    allocation.committed_blocks = committed;
    if (!allocations_.emplace(request_id, allocation).second) {
        throw ReplicaKVCacheError("request already owns KV commitment");
    }
    virtual_committed_blocks_ += future;
    validate_accounting();
}

void ReplicaKVCacheManager::release_commitment(RequestId request_id) {
    const auto allocation = allocations_.find(request_id);
    if (allocation == allocations_.end()) {
        return;
    }
    // If a GPU prefix was pinned while a restore was pending, free() both
    // unpins those physical blocks and releases the remaining future suffix.
    if (allocation->second.allocated_blocks > 0) {
        static_cast<void>(free(request_id));
        return;
    }
    const RequestKVAllocation value = allocation->second;
    if (value.committed_blocks < value.allocated_blocks ||
        value.committed_blocks - value.allocated_blocks >
            virtual_committed_blocks_) {
        throw ReplicaKVCacheError("invalid virtual commitment accounting");
    }
    if (prefix_cache_enabled_ && value.session_id.valid()) {
        auto session = sessions_.find(value.session_id);
        if (session != sessions_.end()) {
            if (session->second.active_request != request_id) {
                throw ReplicaKVCacheError(
                    "virtual commitment does not own its session");
            }
            const auto reserved_snapshot =
                kda_snapshots_.find(value.session_id);
            if (reserved_snapshot != kda_snapshots_.end() &&
                !reserved_snapshot->second.published) {
                // A restore/admission rollback occurred before any reusable
                // KDA frontier was published.  Release the cold reservation
                // instead of retaining a useless snapshot-only session.
                remove_kda_snapshot(value.session_id, false);
            }
            session->second.active_request = RequestId{};
            const auto snapshot = kda_snapshots_.find(value.session_id);
            if (snapshot != kda_snapshots_.end()) {
                if (snapshot->second.in_lru) {
                    remove_kda_snapshot_from_lru(snapshot->second);
                }
                append_kda_snapshot_to_lru(value.session_id, snapshot->second);
            }
            if (session->second.resident_prefix_blocks > 0) {
                append_to_evictable_lru(value.session_id, session->second);
            }
            sync_reclaim_lru(value.session_id);
            erase_session_if_unowned(value.session_id);
        }
        discard_on_release_.erase(value.session_id);
    }
    virtual_committed_blocks_ -= value.committed_blocks;
    allocations_.erase(allocation);
    validate_accounting();
}

std::uint64_t ReplicaKVCacheManager::ceil_div(std::uint64_t numerator,
                                              std::uint64_t denominator) {
    return checked_math::ceil_div<ReplicaKVCacheError>(
        numerator, denominator, "KV division denominator is zero");
}

std::uint64_t ReplicaKVCacheManager::full_sequence_blocks(
    std::uint64_t full_sequence_tokens) const {
    if (full_sequence_tokens == 0) {
        throw ReplicaKVCacheError(
            "full-sequence commitment requires positive token count");
    }
    return ceil_div(full_sequence_tokens, block_size_);
}

bool ReplicaKVCacheManager::can_commit_blocks(
    std::uint64_t blocks, std::optional<SessionId> excluded_session) const {
    if (!full_sequence_commitments_enabled_ || blocks == 0) {
        return false;
    }
    return required_capacity_fits(
        blocks, additional_kda_snapshot_charge(excluded_session),
        reclaimable_commitment_blocks(excluded_session));
}

std::uint64_t ReplicaKVCacheManager::additional_kda_snapshot_charge(
    std::optional<SessionId> session_id) const noexcept {
    if (kda_snapshot_charge_blocks_ == 0 || !session_id.has_value() ||
        !session_id->valid() ||
        kda_snapshots_.find(*session_id) != kda_snapshots_.end()) {
        return 0;
    }
    return kda_snapshot_charge_blocks_;
}

bool ReplicaKVCacheManager::required_capacity_fits(
    std::uint64_t blocks, std::uint64_t extra_blocks,
    std::uint64_t available_blocks) const noexcept {
    if (blocks == 0 || blocks > std::numeric_limits<std::uint64_t>::max() -
                                    watermark_blocks_) {
        return false;
    }
    const std::uint64_t with_watermark = blocks + watermark_blocks_;
    return extra_blocks <=
               std::numeric_limits<std::uint64_t>::max() - with_watermark &&
           with_watermark + extra_blocks <= available_blocks;
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
    if (kda_snapshot_charge_blocks_ != 0) {
        const auto snapshot = kda_snapshots_.find(request.session_id());
        if (snapshot == kda_snapshots_.end()) {
            result.hit_blocks = 0;
        } else {
            // A newer immutable snapshot may be valid for a shorter GPU KV
            // frontier; never expose blocks beyond that KDA frontier.
            result.hit_blocks =
                std::min(result.hit_blocks, snapshot->second.frontier_blocks);
        }
    }
    result.cached_tokens = result.hit_blocks * block_size_;
    return result;
}

bool ReplicaKVCacheManager::can_admit(
    RequestId request_id, SessionId session_id, std::uint64_t cached_tokens,
    std::uint64_t scheduled_tokens, std::uint64_t full_sequence_tokens) const {
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
    const std::uint64_t effective_resident =
        kda_snapshot_charge_blocks_ == 0
            ? resident
            : effective_resident_frontier_blocks(session_id);
    if (position != sessions_.end() &&
        position->second.active_request.valid() &&
        position->second.active_request != request_id) {
        return false;
    }
    const std::uint64_t cached_blocks = cached_tokens / block_size_;
    const std::uint64_t required =
        ceil_div(cached_tokens + scheduled_tokens, block_size_);
    const std::uint64_t snapshot_charge =
        additional_kda_snapshot_charge(std::optional<SessionId>{session_id});
    if (cached_blocks > effective_resident || required < effective_resident ||
        !required_capacity_fits(required, snapshot_charge,
                                reclaimable_commitment_blocks(session_id))) {
        return false;
    }
    if (full_sequence_commitments_enabled_ && full_sequence_tokens > 0) {
        const std::uint64_t committed =
            full_sequence_blocks(full_sequence_tokens);
        if (committed < required || !can_commit_blocks(committed, session_id)) {
            return false;
        }
        // The full-sequence commitment already accounts for watermark
        // headroom.  Do not apply it a second time to this first chunk.
        return true;
    }
    return true;
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

void ReplicaKVCacheManager::consume_available_blocks(
    std::uint64_t blocks, std::uint64_t virtual_credit) {
    const std::uint64_t reclaimable =
        reclaimable_commitment_blocks(std::nullopt, virtual_credit);
    if (blocks > reclaimable) {
        throw ReplicaKVCacheError("analytical KV cache is exhausted");
    }

    // Future-only commitments are overlaid on blank/ordinary capacity.  A
    // continuation may spend its own virtual suffix (virtual_credit), while
    // all other virtual reservations remain protected.  Consume the usable
    // blank portion first; the unified LRU below then reclaims session state.
    const std::uint64_t ordinary_budget =
        available_commitment_blocks() >
                std::numeric_limits<std::uint64_t>::max() - virtual_credit
            ? std::numeric_limits<std::uint64_t>::max()
            : available_commitment_blocks() + virtual_credit;
    const std::uint64_t blank =
        std::min(blocks, std::min(blank_blocks_, ordinary_budget));
    blank_blocks_ -= blank;
    blocks -= blank;

    while (blocks > 0) {
        if (reclaim_lru_.empty()) {
            throw ReplicaKVCacheError(
                "reclaimable accounting has no evictable session");
        }
        const SessionId victim_id = reclaim_lru_.front();
        auto victim = sessions_.find(victim_id);
        if (victim == sessions_.end() || !victim->second.in_reclaim_lru ||
            victim->second.active_request.valid() ||
            (victim->second.resident_prefix_blocks == 0 &&
             kda_snapshots_.find(victim_id) == kda_snapshots_.end())) {
            throw ReplicaKVCacheError("unified reclaim LRU is corrupt");
        }

        SessionCacheEntry &entry = victim->second;
        if (entry.resident_prefix_blocks > 0) {
            // Ordinary KV is always the first leg of a session's reclaim
            // chain.  Partial suffix eviction leaves this session at the
            // unified LRU front so subsequent pressure continues the same
            // victim before considering a newer session.
            const std::uint64_t reclaimed =
                std::min(blocks, entry.resident_prefix_blocks);
            entry.resident_prefix_blocks -= reclaimed;
            evictable_blocks_ -= reclaimed;
            resident_blocks_ -= reclaimed;
            stats_.evicted_blocks += reclaimed;
            blocks -= reclaimed;
            if (entry.resident_prefix_blocks > 0) {
                continue;
            }

            // The ordinary LRU node is now empty.  Remove it before touching
            // the snapshot so all diagnostic counters remain synchronized.
            evictable_lru_.erase(entry.lru_position);
            entry.in_evictable_lru = false;
            --sessions_with_nonzero_frontier_;
            ++stats_.evicted_sessions;

            // Once the resident suffix reaches zero, evict this same
            // session's inactive snapshot atomically.  A larger snapshot
            // charge may leave surplus blank capacity after satisfying the
            // current request; never advance to the next session for that
            // surplus.
            const bool has_snapshot =
                kda_snapshots_.find(victim_id) != kda_snapshots_.end();
            if (has_snapshot && blocks > 0) {
                const std::uint64_t charge = kda_snapshot_charge_blocks_;
                remove_kda_snapshot(victim_id, true);
                const std::uint64_t used = std::min(blocks, charge);
                if (used > blank_blocks_) {
                    throw ReplicaKVCacheError(
                        "KDA snapshot reclaim accounting underflow");
                }
                blank_blocks_ -= used;
                blocks -= used;
            } else {
                // Exact-fit ordinary reclaim leaves an inactive snapshot
                // snapshot-only victim at the unified LRU front.  It is
                // retired on the next pressure event, before any newer
                // session's ordinary KV is considered.
                sync_reclaim_lru(victim_id);
                if (!has_snapshot) {
                    erase_session_if_unowned(victim_id);
                }
            }
            continue;
        }

        // Snapshot-only sessions are still full reclaim victims and must be
        // selected by the same unified recency order as ordinary sessions.
        const std::uint64_t charge = kda_snapshot_charge_blocks_;
        if (charge == 0 || !has_kda_snapshot(victim_id)) {
            throw ReplicaKVCacheError("snapshot-only reclaim LRU is corrupt");
        }
        remove_kda_snapshot(victim_id, true);
        const std::uint64_t used = std::min(blocks, charge);
        if (used > blank_blocks_) {
            throw ReplicaKVCacheError(
                "KDA snapshot reclaim accounting underflow");
        }
        blank_blocks_ -= used;
        blocks -= used;
    }
}

void ReplicaKVCacheManager::admit(RequestId request_id, SessionId session_id,
                                  std::uint64_t cached_tokens,
                                  std::uint64_t scheduled_tokens,
                                  std::uint64_t full_sequence_tokens) {
    if (!can_admit(request_id, session_id, cached_tokens, scheduled_tokens,
                   full_sequence_tokens)) {
        throw ReplicaKVCacheError(
            "KV prefix admission exceeds capacity or watermark");
    }
    reserve_kda_snapshot_capacity(session_id);
    SessionCacheEntry &session = sessions_[session_id];
    const std::uint64_t resident = session.resident_prefix_blocks;
    const std::uint64_t effective_resident =
        kda_snapshot_charge_blocks_ == 0
            ? resident
            : effective_resident_frontier_blocks(session_id);
    if (resident > effective_resident) {
        trim_resident_frontier(session_id, effective_resident);
    }
    const std::uint64_t reusable_resident = effective_resident;
    if (reusable_resident > 0) {
        remove_from_evictable_lru(session);
    }
    const auto snapshot = kda_snapshots_.find(session_id);
    if (snapshot != kda_snapshots_.end() && snapshot->second.in_lru) {
        remove_kda_snapshot_from_lru(snapshot->second);
    }
    // Pin the target session before reclaiming capacity so the unified LRU
    // cannot evict its own resident range or snapshot while materializing.
    session.active_request = request_id;
    remove_from_reclaim_lru(session);
    const std::uint64_t required =
        ceil_div(cached_tokens + scheduled_tokens, block_size_);
    const std::uint64_t committed =
        full_sequence_commitments_enabled_ && full_sequence_tokens > 0
            ? full_sequence_blocks(full_sequence_tokens)
            : required;
    if (committed < required) {
        throw ReplicaKVCacheError(
            "full-sequence commitment is smaller than materialization");
    }
    const std::uint64_t future = committed - required;
    if (future >
        std::numeric_limits<std::uint64_t>::max() - virtual_committed_blocks_) {
        throw ReplicaKVCacheError("invalid prefix commitment accounting");
    }
    consume_available_blocks(required - reusable_resident);
    active_blocks_ += required;
    if (future > 0) {
        virtual_committed_blocks_ += future;
    }
    if (!allocations_
             .emplace(request_id,
                      RequestKVAllocation{session_id, required,
                                          reusable_resident, committed})
             .second) {
        throw ReplicaKVCacheError("request already owns KV blocks");
    }
    validate_accounting();
}

bool ReplicaKVCacheManager::can_admit_tiered(
    RequestId request_id, SessionId session_id,
    std::uint64_t reusable_frontier_blocks, std::uint64_t scheduled_tokens,
    std::uint64_t full_sequence_tokens) const {
    if (!prefix_cache_enabled_ || !session_id.valid() ||
        scheduled_tokens == 0 ||
        reusable_frontier_blocks >
            std::numeric_limits<std::uint64_t>::max() / block_size_) {
        return false;
    }
    const auto allocation = allocations_.find(request_id);
    const bool has_virtual_commitment = allocation != allocations_.end() &&
                                        full_sequence_commitments_enabled_ &&
                                        allocation->second.committed_blocks >
                                            allocation->second.allocated_blocks;
    if (allocation != allocations_.end() && !has_virtual_commitment) {
        return false;
    }
    const auto position = sessions_.find(session_id);
    const std::uint64_t resident =
        position == sessions_.end() ? 0
                                    : position->second.resident_prefix_blocks;
    const std::uint64_t effective_resident =
        kda_snapshot_charge_blocks_ == 0
            ? resident
            : effective_resident_frontier_blocks(session_id);
    if (position != sessions_.end() &&
        position->second.active_request.valid() &&
        position->second.active_request != request_id) {
        return false;
    }
    if (reusable_frontier_blocks < effective_resident) {
        return false;
    }
    const std::uint64_t reusable_tokens =
        reusable_frontier_blocks * block_size_;
    if (reusable_tokens >
        std::numeric_limits<std::uint64_t>::max() - scheduled_tokens) {
        return false;
    }
    const std::uint64_t required =
        ceil_div(reusable_tokens + scheduled_tokens, block_size_);
    const std::uint64_t allocated =
        has_virtual_commitment ? allocation->second.allocated_blocks : 0;
    const std::uint64_t virtual_credit =
        has_virtual_commitment ? allocation->second.committed_blocks - allocated
                               : 0;
    const std::uint64_t reclaimable =
        reclaimable_commitment_blocks(session_id, virtual_credit);
    const std::uint64_t snapshot_charge =
        additional_kda_snapshot_charge(std::optional<SessionId>{session_id});
    if (required < reusable_frontier_blocks || required < effective_resident ||
        required < allocated) {
        return false;
    }
    if (has_virtual_commitment) {
        if (snapshot_charge != 0 || required - allocated > reclaimable ||
            allocation->second.committed_blocks < required) {
            return false;
        }
        return true;
    }
    if (!required_capacity_fits(required - allocated, snapshot_charge,
                                reclaimable)) {
        return false;
    }
    if (full_sequence_commitments_enabled_ && full_sequence_tokens > 0) {
        const std::uint64_t committed =
            full_sequence_blocks(full_sequence_tokens);
        if (committed < required || !can_commit_blocks(committed, session_id)) {
            return false;
        }
        return true;
    }
    return true;
}

void ReplicaKVCacheManager::admit_tiered(RequestId request_id,
                                         SessionId session_id,
                                         std::uint64_t reusable_frontier_blocks,
                                         std::uint64_t scheduled_tokens,
                                         std::uint64_t full_sequence_tokens) {
    if (!can_admit_tiered(request_id, session_id, reusable_frontier_blocks,
                          scheduled_tokens, full_sequence_tokens)) {
        throw ReplicaKVCacheError(
            "tiered KV admission exceeds capacity or watermark");
    }
    reserve_kda_snapshot_capacity(session_id);
    SessionCacheEntry &session = sessions_[session_id];
    const std::uint64_t resident = session.resident_prefix_blocks;
    const auto allocation = allocations_.find(request_id);
    const bool has_virtual_commitment = allocation != allocations_.end() &&
                                        full_sequence_commitments_enabled_ &&
                                        allocation->second.committed_blocks >
                                            allocation->second.allocated_blocks;
    const std::uint64_t effective_resident =
        kda_snapshot_charge_blocks_ == 0
            ? resident
            : effective_resident_frontier_blocks(session_id);
    if (resident > effective_resident) {
        if (!has_virtual_commitment) {
            trim_resident_frontier(session_id, effective_resident);
        } else {
            const std::uint64_t excess = resident - effective_resident;
            RequestKVAllocation &owned = allocation->second;
            if (owned.allocated_blocks < excess || active_blocks_ < excess ||
                resident_blocks_ < excess ||
                blank_blocks_ >
                    std::numeric_limits<std::uint64_t>::max() - excess ||
                virtual_committed_blocks_ >
                    std::numeric_limits<std::uint64_t>::max() - excess) {
                throw ReplicaKVCacheError(
                    "active KDA frontier trim accounting underflow");
            }
            owned.allocated_blocks -= excess;
            owned.published_blocks =
                std::min(owned.published_blocks, effective_resident);
            active_blocks_ -= excess;
            resident_blocks_ -= excess;
            blank_blocks_ += excess;
            virtual_committed_blocks_ += excess;
            session.resident_prefix_blocks = effective_resident;
        }
    }
    const std::uint64_t reusable_resident = effective_resident;
    const std::uint64_t already_allocated =
        has_virtual_commitment ? allocation->second.allocated_blocks : 0;
    const std::uint64_t virtual_credit =
        has_virtual_commitment
            ? allocation->second.committed_blocks - already_allocated
            : 0;
    if (reusable_resident > 0 && !has_virtual_commitment) {
        remove_from_evictable_lru(session);
    }
    const auto snapshot = kda_snapshots_.find(session_id);
    if (snapshot != kda_snapshots_.end() && snapshot->second.in_lru) {
        remove_kda_snapshot_from_lru(snapshot->second);
    }
    // Pin the target before reclaiming any physical continuation blocks.
    session.active_request = request_id;
    remove_from_reclaim_lru(session);
    const std::uint64_t reusable_tokens =
        reusable_frontier_blocks * block_size_;
    const std::uint64_t required =
        ceil_div(reusable_tokens + scheduled_tokens, block_size_);
    const std::uint64_t physical_additional = required - already_allocated;
    const std::uint64_t committed =
        full_sequence_commitments_enabled_ && full_sequence_tokens > 0
            ? full_sequence_blocks(full_sequence_tokens)
            : (has_virtual_commitment ? allocation->second.committed_blocks
                                      : required);
    if (committed < required) {
        throw ReplicaKVCacheError(
            "full-sequence commitment is smaller than materialization");
    }
    const std::uint64_t future = committed - required;
    if (has_virtual_commitment) {
        if (allocation->second.committed_blocks < already_allocated ||
            allocation->second.committed_blocks - already_allocated <
                physical_additional ||
            physical_additional > virtual_committed_blocks_) {
            throw ReplicaKVCacheError("invalid staged commitment accounting");
        }
    } else if (future > std::numeric_limits<std::uint64_t>::max() -
                            virtual_committed_blocks_) {
        throw ReplicaKVCacheError("invalid staged commitment accounting");
    }
    consume_available_blocks(physical_additional, virtual_credit);
    active_blocks_ += physical_additional;
    resident_blocks_ += reusable_frontier_blocks - reusable_resident;
    if (reusable_resident == 0 && reusable_frontier_blocks > 0) {
        ++sessions_with_nonzero_frontier_;
    }
    session.resident_prefix_blocks = reusable_frontier_blocks;
    if (has_virtual_commitment) {
        RequestKVAllocation &owned = allocation->second;
        owned.allocated_blocks = required;
        owned.published_blocks = reusable_frontier_blocks;
        if (physical_additional > virtual_committed_blocks_) {
            throw ReplicaKVCacheError("staged commitment underflow");
        }
        virtual_committed_blocks_ -= physical_additional;
    } else {
        if (!allocations_
                 .emplace(request_id,
                          RequestKVAllocation{session_id, required,
                                              reusable_frontier_blocks,
                                              committed})
                 .second) {
            throw ReplicaKVCacheError("tiered request already owns KV blocks");
        }
        virtual_committed_blocks_ += future;
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

bool ReplicaKVCacheManager::can_reserve(
    RequestId request_id, std::uint64_t kv_accounted_tokens,
    std::uint64_t scheduled_tokens, std::uint64_t full_sequence_tokens) const {
    const std::uint64_t required = additional_blocks_required(
        request_id, kv_accounted_tokens, scheduled_tokens, false);
    const auto allocation = allocations_.find(request_id);
    const std::uint64_t virtual_credit =
        allocation != allocations_.end() &&
                allocation->second.committed_blocks >
                    allocation->second.allocated_blocks
            ? allocation->second.committed_blocks -
                  allocation->second.allocated_blocks
            : 0;
    const std::uint64_t reclaimable =
        reclaimable_commitment_blocks(std::nullopt, virtual_credit);
    if (required > reclaimable) {
        return false;
    }
    if (allocation != allocations_.end()) {
        if (full_sequence_commitments_enabled_ && full_sequence_tokens > 0 &&
            allocation->second.committed_blocks <
                full_sequence_blocks(full_sequence_tokens)) {
            return false;
        }
        return true;
    }
    if (full_sequence_commitments_enabled_ && full_sequence_tokens > 0) {
        if (!can_commit(request_id, full_sequence_tokens)) {
            return false;
        }
        // Full-sequence admission already reserves watermark headroom.  The
        // first physical chunk must consume only its own reservation.
        return true;
    }
    return reclaimable - required >= watermark_blocks_;
}

void ReplicaKVCacheManager::reserve(RequestId request_id,
                                    std::uint64_t kv_accounted_tokens,
                                    std::uint64_t scheduled_tokens,
                                    std::uint64_t full_sequence_tokens) {
    if (!can_reserve(request_id, kv_accounted_tokens, scheduled_tokens,
                     full_sequence_tokens)) {
        throw ReplicaKVCacheError(
            "KV reservation exceeds capacity or watermark");
    }
    const std::uint64_t additional = additional_blocks_required(
        request_id, kv_accounted_tokens, scheduled_tokens, true);
    auto position = allocations_.find(request_id);
    const bool inserted = position == allocations_.end();
    const std::uint64_t old_allocated =
        inserted ? 0 : position->second.allocated_blocks;
    const std::uint64_t old_committed =
        inserted ? 0 : position->second.committed_blocks;
    const bool had_virtual_suffix = !inserted && old_committed > old_allocated;
    const std::uint64_t committed = [&]() {
        if (inserted) {
            return full_sequence_commitments_enabled_ &&
                           full_sequence_tokens > 0
                       ? full_sequence_blocks(full_sequence_tokens)
                       : additional;
        }
        const std::uint64_t old = old_committed;
        if (had_virtual_suffix) {
            // A previously committed request is materializing its own
            // reserved suffix.  Do not grow the target on continuation calls.
            if (full_sequence_commitments_enabled_ &&
                full_sequence_tokens > 0) {
                return std::max(old,
                                full_sequence_blocks(full_sequence_tokens));
            }
            return old;
        }
        if (full_sequence_commitments_enabled_ && full_sequence_tokens > 0) {
            return std::max(old, full_sequence_blocks(full_sequence_tokens));
        }
        // Legacy/materialized allocations have no future suffix to preserve;
        // continuation materialization extends their logical target by the
        // newly allocated physical blocks.
        if (old > std::numeric_limits<std::uint64_t>::max() - additional) {
            throw ReplicaKVCacheError("KV commitment count overflows uint64");
        }
        return old + additional;
    }();
    if (committed < old_allocated || committed - old_allocated < additional) {
        throw ReplicaKVCacheError(
            "KV materialization exceeds full-sequence commitment");
    }
    const std::uint64_t future_before = committed - old_allocated;
    const std::uint64_t future_after = future_before - additional;
    if (inserted) {
        if (future_after > std::numeric_limits<std::uint64_t>::max() -
                               virtual_committed_blocks_) {
            throw ReplicaKVCacheError(
                "virtual commitment count overflows uint64");
        }
    } else if (had_virtual_suffix && additional > virtual_committed_blocks_) {
        throw ReplicaKVCacheError(
            "virtual commitment underflows on materialization");
    }
    consume_available_blocks(
        additional, had_virtual_suffix ? old_committed - old_allocated : 0);
    active_blocks_ += additional;
    if (inserted) {
        const RequestKVAllocation allocation{SessionId{}, additional, 0,
                                             committed};
        position = allocations_.emplace(request_id, allocation).first;
        virtual_committed_blocks_ += future_after;
    } else {
        position->second.allocated_blocks += additional;
        position->second.committed_blocks = committed;
        if (had_virtual_suffix) {
            virtual_committed_blocks_ -= additional;
        }
    }
    validate_accounting();
}

std::uint64_t ReplicaKVCacheManager::free(RequestId request_id) {
    const auto allocation = allocations_.find(request_id);
    if (allocation == allocations_.end()) {
        return 0;
    }
    const RequestKVAllocation value = allocation->second;
    if (value.allocated_blocks > active_blocks_ ||
        value.published_blocks > value.allocated_blocks ||
        value.committed_blocks < value.allocated_blocks ||
        value.committed_blocks - value.allocated_blocks >
            virtual_committed_blocks_) {
        throw ReplicaKVCacheError("request allocation accounting is corrupt");
    }
    active_blocks_ -= value.allocated_blocks;
    virtual_committed_blocks_ -=
        value.committed_blocks - value.allocated_blocks;
    if (prefix_cache_enabled_ && value.session_id.valid()) {
        auto session = sessions_.find(value.session_id);
        if (session == sessions_.end() ||
            session->second.active_request != request_id ||
            session->second.in_evictable_lru) {
            throw ReplicaKVCacheError("active session allocation is corrupt");
        }
        SessionCacheEntry &entry = session->second;
        const bool discard = discard_on_release_.erase(value.session_id) != 0;
        if (discard) {
            // Explicit migration discard retires both ordinary GPU KV and the
            // pinned KDA snapshot before dropping session metadata.
            remove_kda_snapshot(value.session_id, false);
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
            remove_from_reclaim_lru(entry);
            sessions_.erase(session);
            allocations_.erase(allocation);
            validate_accounting();
            return value.allocated_blocks;
        }
        entry.resident_prefix_blocks = value.published_blocks;
        const auto reserved_snapshot = kda_snapshots_.find(value.session_id);
        if (reserved_snapshot != kda_snapshots_.end() &&
            !reserved_snapshot->second.published) {
            remove_kda_snapshot(value.session_id, false);
        }
        entry.active_request = RequestId{};
        blank_blocks_ += value.allocated_blocks - value.published_blocks;
        const auto snapshot = kda_snapshots_.find(value.session_id);
        if (snapshot != kda_snapshots_.end()) {
            if (snapshot->second.in_lru) {
                remove_kda_snapshot_from_lru(snapshot->second);
            }
            append_kda_snapshot_to_lru(value.session_id, snapshot->second);
        }
        if (entry.resident_prefix_blocks > 0) {
            append_to_evictable_lru(value.session_id, entry);
        }
        sync_reclaim_lru(value.session_id);
        erase_session_if_unowned(value.session_id);
    } else {
        blank_blocks_ += value.allocated_blocks;
    }
    allocations_.erase(allocation);
    validate_accounting();
    return value.allocated_blocks;
}

std::uint64_t ReplicaKVCacheManager::discard_session(SessionId session_id) {
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
    if (entry.resident_prefix_blocks == 0 && !has_kda_snapshot(session_id)) {
        throw ReplicaKVCacheError(
            "inactive discarded session is not evictable");
    }
    const std::uint64_t discarded = entry.resident_prefix_blocks;
    if (entry.resident_prefix_blocks > 0) {
        if (!entry.in_evictable_lru) {
            throw ReplicaKVCacheError(
                "inactive discarded session is not evictable");
        }
        remove_from_evictable_lru(entry);
    }
    if (discarded > resident_blocks_) {
        throw ReplicaKVCacheError(
            "discarded session exceeds resident accounting");
    }
    resident_blocks_ -= discarded;
    entry.resident_prefix_blocks = 0;
    blank_blocks_ += discarded;
    if (discarded > 0) {
        --sessions_with_nonzero_frontier_;
        stats_.evicted_blocks += discarded;
        ++stats_.evicted_sessions;
    }
    remove_kda_snapshot(session_id, false);
    erase_session_if_unowned(session_id);
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
    if (session == sessions_.end()) {
        return 0;
    }
    std::uint64_t resident = session->second.resident_prefix_blocks;
    if (kda_snapshot_charge_blocks_ != 0) {
        const auto snapshot = kda_snapshots_.find(session_id);
        if (snapshot == kda_snapshots_.end()) {
            return 0;
        }
        resident = std::min(resident, snapshot->second.frontier_blocks);
    }
    return resident;
}

bool ReplicaKVCacheManager::session_has_active_request(
    SessionId session_id) const noexcept {
    if (!prefix_cache_enabled_ || !session_id.valid()) {
        return false;
    }
    const auto session = sessions_.find(session_id);
    return session != sessions_.end() && session->second.active_request.valid();
}

std::uint64_t ReplicaKVCacheManager::request_committed_blocks(
    RequestId request_id) const noexcept {
    const auto allocation = allocations_.find(request_id);
    return allocation == allocations_.end()
               ? 0
               : allocation->second.committed_blocks;
}

std::uint64_t ReplicaKVCacheManager::request_virtual_committed_blocks(
    RequestId request_id) const noexcept {
    const auto allocation = allocations_.find(request_id);
    if (allocation == allocations_.end() ||
        allocation->second.committed_blocks <
            allocation->second.allocated_blocks) {
        return 0;
    }
    return allocation->second.committed_blocks -
           allocation->second.allocated_blocks;
}

bool ReplicaKVCacheManager::full_sequence_fits_empty(
    std::uint64_t full_sequence_tokens) const {
    if (full_sequence_tokens == 0) {
        return false;
    }
    const std::uint64_t blocks = full_sequence_blocks(full_sequence_tokens);
    return required_capacity_fits(blocks, kda_snapshot_charge_blocks_,
                                  capacity_blocks_);
}

void ReplicaKVCacheManager::validate_accounting() const {
    if (kda_snapshot_charge_blocks_ > capacity_blocks_ ||
        kda_snapshot_occupied_blocks_ > capacity_blocks_ ||
        active_blocks_ > capacity_blocks_ - kda_snapshot_occupied_blocks_ ||
        virtual_committed_blocks_ >
            capacity_blocks_ - active_blocks_ - kda_snapshot_occupied_blocks_ ||
        blank_blocks_ >
            capacity_blocks_ - active_blocks_ - kda_snapshot_occupied_blocks_ ||
        evictable_blocks_ != capacity_blocks_ - active_blocks_ -
                                 kda_snapshot_occupied_blocks_ -
                                 blank_blocks_ ||
        resident_blocks_ > capacity_blocks_) {
        throw ReplicaKVCacheError(
            "analytical KV capacity partition is invalid");
    }
#ifndef NDEBUG
    std::uint64_t observed_active = 0;
    std::uint64_t observed_virtual = 0;
    for (const auto &[request_id, allocation] : allocations_) {
        static_cast<void>(request_id);
        if (allocation.committed_blocks < allocation.allocated_blocks ||
            allocation.published_blocks > allocation.allocated_blocks) {
            throw ReplicaKVCacheError(
                "request commitment is smaller than physical allocation");
        }
        if (observed_active > std::numeric_limits<std::uint64_t>::max() -
                                  allocation.allocated_blocks) {
            throw ReplicaKVCacheError("active allocation sum overflows uint64");
        }
        observed_active += allocation.allocated_blocks;
        const std::uint64_t future =
            allocation.committed_blocks - allocation.allocated_blocks;
        if (observed_virtual >
            std::numeric_limits<std::uint64_t>::max() - future) {
            throw ReplicaKVCacheError(
                "virtual commitment sum overflows uint64");
        }
        observed_virtual += future;
    }
    if (observed_active != active_blocks_ ||
        observed_virtual != virtual_committed_blocks_) {
        throw ReplicaKVCacheError(
            "request commitment totals diverged from KV accounting");
    }
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
        } else if (!entry.active_request.valid() &&
                   kda_snapshots_.find(session_id) == kda_snapshots_.end()) {
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
    std::size_t observed_reclaim_sessions = 0;
    for (const auto &[session_id, entry] : sessions_) {
        const auto snapshot = kda_snapshots_.find(session_id);
        const bool should_be_reclaimable =
            !entry.active_request.valid() &&
            (entry.resident_prefix_blocks > 0 ||
             (snapshot != kda_snapshots_.end() && snapshot->second.in_lru));
        if (entry.in_reclaim_lru != should_be_reclaimable) {
            throw ReplicaKVCacheError(
                "unified session reclaimability diverged");
        }
        if (entry.in_reclaim_lru) {
            ++observed_reclaim_sessions;
        }
    }
    for (const SessionId session_id : reclaim_lru_) {
        const auto session = sessions_.find(session_id);
        if (session == sessions_.end() || !session->second.in_reclaim_lru ||
            session->second.active_request.valid()) {
            throw ReplicaKVCacheError("unified session reclaim LRU is corrupt");
        }
        const auto snapshot = kda_snapshots_.find(session_id);
        if (session->second.resident_prefix_blocks == 0 &&
            (snapshot == kda_snapshots_.end() || !snapshot->second.in_lru)) {
            throw ReplicaKVCacheError(
                "unified session reclaim LRU has an empty victim");
        }
    }
    if (observed_reclaim_sessions != reclaim_lru_.size()) {
        throw ReplicaKVCacheError(
            "unified session reclaim accounting diverged");
    }
    if (kda_snapshot_charge_blocks_ == 0 && !kda_snapshots_.empty()) {
        throw ReplicaKVCacheError(
            "KDA snapshots exist while snapshot policy is disabled");
    }
    std::uint64_t observed_snapshot_blocks = 0;
    std::size_t observed_snapshot_lru = 0;
    for (const auto &[session_id, snapshot] : kda_snapshots_) {
        const auto session = sessions_.find(session_id);
        if (session == sessions_.end()) {
            throw ReplicaKVCacheError("KDA snapshot has no session metadata");
        }
        if (snapshot.in_lru && session->second.active_request.valid()) {
            throw ReplicaKVCacheError("active KDA snapshot must remain pinned");
        }
        if (!snapshot.in_lru && !session->second.active_request.valid()) {
            throw ReplicaKVCacheError(
                "inactive KDA snapshot is missing from its LRU");
        }
        if (observed_snapshot_blocks >
            std::numeric_limits<std::uint64_t>::max() -
                kda_snapshot_charge_blocks_) {
            throw ReplicaKVCacheError(
                "KDA snapshot occupancy overflows uint64");
        }
        observed_snapshot_blocks += kda_snapshot_charge_blocks_;
    }
    for (const SessionId session_id : kda_snapshot_lru_) {
        const auto snapshot = kda_snapshots_.find(session_id);
        if (snapshot == kda_snapshots_.end() || !snapshot->second.in_lru ||
            snapshot->second.lru_position == kda_snapshot_lru_.end()) {
            throw ReplicaKVCacheError("KDA snapshot LRU is corrupt");
        }
        ++observed_snapshot_lru;
    }
    if (observed_snapshot_blocks != kda_snapshot_occupied_blocks_ ||
        observed_snapshot_lru != kda_snapshot_lru_.size()) {
        throw ReplicaKVCacheError("KDA snapshot accounting diverged");
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
    result.committed_blocks = committed_blocks();
    result.virtual_committed_blocks = virtual_committed_blocks_;
    result.available_commitment_blocks = available_commitment_blocks();
    result.kda_snapshot_occupied_blocks = kda_snapshot_occupied_blocks_;
    result.kda_snapshot_sessions =
        static_cast<std::uint64_t>(kda_snapshots_.size());
    result.kda_snapshot_evictable_sessions =
        static_cast<std::uint64_t>(kda_snapshot_lru_.size());
    return result;
}

} // namespace frontier::kv_cache
