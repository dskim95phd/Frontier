#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>

#include "frontier/core/cluster_type.h"
#include "frontier/core/event.h"
#include "frontier/core/ids.h"

namespace frontier::entities {

enum class KVCacheTransferState : std::uint8_t {
  kPending,
  kInFlight,
  kCompleted,
};

class KVCacheTransferError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class KVCacheTransferInfo {
 public:
  KVCacheTransferInfo(
      TransferId transfer_id,
      RequestId request_id,
      BatchId source_batch_id,
      ReplicaId source_replica_id,
      DataParallelId source_dp_id,
      std::uint64_t size_bytes,
      double predicted_time_ms,
      Generation source_generation);

  [[nodiscard]] TransferId id() const noexcept {
    return transfer_id_;
  }
  [[nodiscard]] RequestId request_id() const noexcept {
    return request_id_;
  }
  [[nodiscard]] BatchId source_batch_id() const noexcept {
    return source_batch_id_;
  }
  [[nodiscard]] ClusterType source_cluster_type() const noexcept {
    return ClusterType::kPrefill;
  }
  [[nodiscard]] ClusterType target_cluster_type() const noexcept {
    return ClusterType::kDecode;
  }
  [[nodiscard]] ReplicaId source_replica_id() const noexcept {
    return source_replica_id_;
  }
  [[nodiscard]] DataParallelId source_dp_id() const noexcept {
    return source_dp_id_;
  }
  [[nodiscard]] std::uint64_t size_bytes() const noexcept {
    return size_bytes_;
  }
  [[nodiscard]] double predicted_time_ms() const noexcept {
    return predicted_time_ms_;
  }
  [[nodiscard]] Generation source_generation() const noexcept {
    return source_generation_;
  }
  [[nodiscard]] KVCacheTransferState state() const noexcept {
    return state_;
  }
  [[nodiscard]] const std::optional<SimTime>& started_at()
      const noexcept {
    return started_at_;
  }
  [[nodiscard]] const std::optional<SimTime>& completed_at()
      const noexcept {
    return completed_at_;
  }

  void mark_started(SimTime time);
  void mark_completed(SimTime time);

 private:
  TransferId transfer_id_;
  RequestId request_id_;
  BatchId source_batch_id_;
  ReplicaId source_replica_id_;
  DataParallelId source_dp_id_;
  std::uint64_t size_bytes_;
  double predicted_time_ms_;
  Generation source_generation_;
  KVCacheTransferState state_ = KVCacheTransferState::kPending;
  std::optional<SimTime> started_at_;
  std::optional<SimTime> completed_at_;
};

}  // namespace frontier::entities
