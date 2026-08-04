#pragma once

#include <cstdint>
#include <stdexcept>

#include "frontier/core/event.h"
#include "frontier/cpu_kv_cache_transfer/analytical_transfer.h"

namespace frontier::entities {

class CpuKVCacheTransferError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

enum class CpuKVCacheTransferState {
    kPending,
    kInFlight,
    kCompleted,
    kCancelled,
};

struct TieredPrefixPlan {
    std::uint64_t query_blocks = 0;
    std::uint64_t cpu_query_blocks = 0;
    std::uint64_t gpu_hit_frontier_blocks = 0;
    std::uint64_t cpu_begin_block = 0;
    std::uint64_t cpu_end_block = 0;
    std::uint64_t hit_frontier_blocks = 0;
    std::uint64_t block_size = 0;
    std::uint64_t prompt_tokens = 0;
};

class CpuKVCacheOffloadInfo {
  public:
    CpuKVCacheOffloadInfo(
        CpuKvTransferId transfer_id, RequestId request_id,
        ReplicaId replica_id, DataParallelId dp_id,
        CpuOffloadReservationId reservation_id,
        cpu_kv_cache_transfer::CpuTransferTiming timing,
        std::uint64_t desired_frontier_blocks,
        CpuOffloadGeneration generation);

    void mark_started(SimTime time);
    void mark_completed(SimTime time);
    void cancel();
    void set_decode_transfer_completed_at(SimTime time);

    [[nodiscard]] CpuKvTransferId transfer_id() const noexcept {
        return transfer_id_;
    }
    [[nodiscard]] RequestId request_id() const noexcept { return request_id_; }
    [[nodiscard]] ReplicaId replica_id() const noexcept { return replica_id_; }
    [[nodiscard]] DataParallelId dp_id() const noexcept { return dp_id_; }
    [[nodiscard]] CpuOffloadReservationId reservation_id() const noexcept {
        return reservation_id_;
    }
    [[nodiscard]] const cpu_kv_cache_transfer::CpuTransferTiming &
    timing() const noexcept {
        return timing_;
    }
    [[nodiscard]] std::uint64_t desired_frontier_blocks() const noexcept {
        return desired_frontier_blocks_;
    }
    [[nodiscard]] CpuOffloadGeneration generation() const noexcept {
        return generation_;
    }
    [[nodiscard]] CpuKVCacheTransferState state() const noexcept {
        return state_;
    }
    [[nodiscard]] SimTime decode_transfer_completed_at() const noexcept {
        return decode_transfer_completed_at_;
    }
    [[nodiscard]] double attributable_source_hold_ms() const noexcept;

  private:
    CpuKvTransferId transfer_id_;
    RequestId request_id_;
    ReplicaId replica_id_;
    DataParallelId dp_id_;
    CpuOffloadReservationId reservation_id_;
    cpu_kv_cache_transfer::CpuTransferTiming timing_;
    std::uint64_t desired_frontier_blocks_;
    CpuOffloadGeneration generation_;
    CpuKVCacheTransferState state_ = CpuKVCacheTransferState::kPending;
    SimTime decode_transfer_completed_at_;
};

class CpuKVCacheRestoreInfo {
  public:
    CpuKVCacheRestoreInfo(
        CpuKvTransferId transfer_id, RequestId request_id,
        ReplicaId replica_id, DataParallelId dp_id,
        CpuRestoreLeaseId lease_id, TieredPrefixPlan plan,
        cpu_kv_cache_transfer::CpuTransferTiming timing,
        Generation request_generation);

    void mark_started(SimTime time);
    void mark_completed(SimTime time);
    void cancel();

    [[nodiscard]] CpuKvTransferId transfer_id() const noexcept {
        return transfer_id_;
    }
    [[nodiscard]] RequestId request_id() const noexcept { return request_id_; }
    [[nodiscard]] ReplicaId replica_id() const noexcept { return replica_id_; }
    [[nodiscard]] DataParallelId dp_id() const noexcept { return dp_id_; }
    [[nodiscard]] CpuRestoreLeaseId lease_id() const noexcept {
        return lease_id_;
    }
    [[nodiscard]] const TieredPrefixPlan &plan() const noexcept { return plan_; }
    [[nodiscard]] const cpu_kv_cache_transfer::CpuTransferTiming &
    timing() const noexcept {
        return timing_;
    }
    [[nodiscard]] Generation request_generation() const noexcept {
        return request_generation_;
    }
    [[nodiscard]] CpuKVCacheTransferState state() const noexcept {
        return state_;
    }

  private:
    CpuKvTransferId transfer_id_;
    RequestId request_id_;
    ReplicaId replica_id_;
    DataParallelId dp_id_;
    CpuRestoreLeaseId lease_id_;
    TieredPrefixPlan plan_;
    cpu_kv_cache_transfer::CpuTransferTiming timing_;
    Generation request_generation_;
    CpuKVCacheTransferState state_ = CpuKVCacheTransferState::kPending;
};

struct StagedCpuKVCacheRestore {
    RequestId request_id;
    SessionId session_id;
    ReplicaId replica_id;
    DataParallelId dp_id;
    std::uint64_t cpu_begin_block = 0;
    std::uint64_t cpu_end_block = 0;
    std::uint64_t lookup_gpu_frontier_blocks = 0;
    std::uint64_t lookup_cpu_frontier_blocks = 0;
    std::uint64_t query_blocks = 0;
    std::uint64_t block_size = 0;
    std::uint64_t prompt_tokens = 0;
    cpu_kv_cache_transfer::CpuTransferTiming timing;
};

} // namespace frontier::entities
