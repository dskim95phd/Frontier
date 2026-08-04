#include "frontier/entities/cpu_kv_cache_transfer_info.h"

#include <algorithm>
#include <cmath>

namespace frontier::entities {
namespace {

void validate_timing(
    const cpu_kv_cache_transfer::CpuTransferTiming &timing) {
    if (!timing.submitted_at.valid() || !timing.started_at.valid() ||
        !timing.completed_at.valid() ||
        timing.started_at < timing.submitted_at ||
        timing.completed_at < timing.started_at ||
        !std::isfinite(timing.queue_time_ms) || timing.queue_time_ms < 0.0 ||
        !std::isfinite(timing.service_time_ms) ||
        timing.service_time_ms < 0.0) {
        throw CpuKVCacheTransferError("invalid analytical CPU transfer timing");
    }
}

void mark_started(CpuKVCacheTransferState &state, SimTime actual,
                  SimTime expected) {
    if (state != CpuKVCacheTransferState::kPending || actual != expected) {
        throw CpuKVCacheTransferError("invalid CPU transfer start event");
    }
    state = CpuKVCacheTransferState::kInFlight;
}

void mark_completed(CpuKVCacheTransferState &state, SimTime actual,
                    SimTime expected) {
    if (state != CpuKVCacheTransferState::kInFlight || actual != expected) {
        throw CpuKVCacheTransferError("invalid CPU transfer end event");
    }
    state = CpuKVCacheTransferState::kCompleted;
}

void cancel(CpuKVCacheTransferState &state) {
    if (state == CpuKVCacheTransferState::kCompleted) {
        throw CpuKVCacheTransferError("completed CPU transfer cannot cancel");
    }
    state = CpuKVCacheTransferState::kCancelled;
}

} // namespace

CpuKVCacheOffloadInfo::CpuKVCacheOffloadInfo(
    CpuKvTransferId transfer_id, RequestId request_id, ReplicaId replica_id,
    DataParallelId dp_id, CpuOffloadReservationId reservation_id,
    cpu_kv_cache_transfer::CpuTransferTiming timing,
    std::uint64_t desired_frontier_blocks, CpuOffloadGeneration generation)
    : transfer_id_(transfer_id), request_id_(request_id),
      replica_id_(replica_id), dp_id_(dp_id), reservation_id_(reservation_id),
      timing_(timing), desired_frontier_blocks_(desired_frontier_blocks),
      generation_(generation) {
    if (!transfer_id_.valid() || !request_id_.valid() ||
        !replica_id_.valid() || !dp_id_.valid() || !reservation_id_.valid() ||
        !generation_.valid() || desired_frontier_blocks_ == 0 ||
        timing_.direction !=
            cpu_kv_cache_transfer::CpuTransferDirection::kD2H ||
        timing_.size_bytes == 0) {
        throw CpuKVCacheTransferError("invalid CPU offload operation");
    }
    validate_timing(timing_);
}

void CpuKVCacheOffloadInfo::mark_started(SimTime time) {
    entities::mark_started(state_, time, timing_.started_at);
}

void CpuKVCacheOffloadInfo::mark_completed(SimTime time) {
    entities::mark_completed(state_, time, timing_.completed_at);
}

void CpuKVCacheOffloadInfo::cancel() { entities::cancel(state_); }

void CpuKVCacheOffloadInfo::set_decode_transfer_completed_at(SimTime time) {
    if (!time.valid() || decode_transfer_completed_at_.valid()) {
        throw CpuKVCacheTransferError(
            "invalid PREFILL-to-DECODE completion timestamp");
    }
    decode_transfer_completed_at_ = time;
}

double CpuKVCacheOffloadInfo::attributable_source_hold_ms() const noexcept {
    if (state_ != CpuKVCacheTransferState::kCompleted ||
        !decode_transfer_completed_at_.valid()) {
        return 0.0;
    }
    return std::max(0.0, (timing_.completed_at.seconds() -
                          decode_transfer_completed_at_.seconds()) *
                             1e3);
}

CpuKVCacheRestoreInfo::CpuKVCacheRestoreInfo(
    CpuKvTransferId transfer_id, RequestId request_id, ReplicaId replica_id,
    DataParallelId dp_id, CpuRestoreLeaseId lease_id, TieredPrefixPlan plan,
    cpu_kv_cache_transfer::CpuTransferTiming timing,
    Generation request_generation)
    : transfer_id_(transfer_id), request_id_(request_id),
      replica_id_(replica_id), dp_id_(dp_id), lease_id_(lease_id), plan_(plan),
      timing_(timing), request_generation_(request_generation) {
    if (!transfer_id_.valid() || !request_id_.valid() ||
        !replica_id_.valid() || !dp_id_.valid() || !lease_id_.valid() ||
        !request_generation_.valid() || plan_.cpu_begin_block >=
                                              plan_.cpu_end_block ||
        plan_.cpu_end_block > plan_.hit_frontier_blocks ||
        timing_.direction !=
            cpu_kv_cache_transfer::CpuTransferDirection::kH2D ||
        timing_.size_bytes == 0) {
        throw CpuKVCacheTransferError("invalid CPU restore operation");
    }
    validate_timing(timing_);
}

void CpuKVCacheRestoreInfo::mark_started(SimTime time) {
    entities::mark_started(state_, time, timing_.started_at);
}

void CpuKVCacheRestoreInfo::mark_completed(SimTime time) {
    entities::mark_completed(state_, time, timing_.completed_at);
}

void CpuKVCacheRestoreInfo::cancel() { entities::cancel(state_); }

} // namespace frontier::entities
