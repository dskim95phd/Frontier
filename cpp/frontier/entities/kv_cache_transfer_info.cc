#include "frontier/entities/kv_cache_transfer_info.h"

#include <cmath>

namespace frontier::entities {

KVCacheTransferInfo::KVCacheTransferInfo(
    TransferId transfer_id, RequestId request_id, BatchId source_batch_id,
    ReplicaId source_replica_id, DataParallelId source_dp_id,
    std::uint64_t size_bytes, double predicted_time_ms,
    Generation source_generation)
    : transfer_id_(transfer_id), request_id_(request_id),
      source_batch_id_(source_batch_id), source_replica_id_(source_replica_id),
      source_dp_id_(source_dp_id), size_bytes_(size_bytes),
      predicted_time_ms_(predicted_time_ms),
      source_generation_(source_generation) {
    if (size_bytes_ == 0 || !std::isfinite(predicted_time_ms_) ||
        predicted_time_ms_ < 0.0) {
        throw KVCacheTransferError(
            "KV transfer requires positive bytes and finite nonnegative time");
    }
}

void KVCacheTransferInfo::mark_started(SimTime time) {
    if (state_ != KVCacheTransferState::kPending ||
        !std::isfinite(time.seconds()) || time.seconds() < 0.0) {
        throw KVCacheTransferError("invalid KV transfer start");
    }
    started_at_ = time;
    state_ = KVCacheTransferState::kInFlight;
}

void KVCacheTransferInfo::mark_completed(SimTime time) {
    if (state_ != KVCacheTransferState::kInFlight || !started_at_.valid() ||
        !std::isfinite(time.seconds()) ||
        time.seconds() < started_at_.seconds()) {
        throw KVCacheTransferError("invalid KV transfer completion");
    }
    completed_at_ = time;
    state_ = KVCacheTransferState::kCompleted;
}

} // namespace frontier::entities
