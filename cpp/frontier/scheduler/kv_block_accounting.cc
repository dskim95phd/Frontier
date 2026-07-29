#include "frontier/scheduler/kv_block_accounting.h"

#include <cmath>
#include <limits>
#include <string>

namespace frontier::scheduler {

KvBlockAccounting::KvBlockAccounting(
    const config::SchedulerConfig& config)
    : block_size_(config.block_size),
      num_blocks_(config.num_blocks),
      watermark_blocks_(static_cast<std::uint64_t>(
          config.watermark_blocks_fraction *
          static_cast<double>(config.num_blocks))) {
  if (block_size_ == 0 || num_blocks_ == 0) {
    throw KvBlockAccountingError(
        "KV block size and capacity must be positive");
  }
  if (!std::isfinite(config.watermark_blocks_fraction) ||
      config.watermark_blocks_fraction < 0.0 ||
      config.watermark_blocks_fraction >= 1.0) {
    throw KvBlockAccountingError(
        "KV watermark fraction must be finite and in [0, 1)");
  }
}

std::uint64_t KvBlockAccounting::ceil_div(
    std::uint64_t numerator,
    std::uint64_t denominator) {
  if (denominator == 0) {
    throw KvBlockAccountingError("KV division denominator is zero");
  }
  return numerator / denominator +
      static_cast<std::uint64_t>(numerator % denominator != 0);
}

std::uint64_t KvBlockAccounting::additional_blocks_required(
    RequestId request_id,
    std::uint64_t kv_accounted_tokens,
    std::uint64_t scheduled_tokens,
    bool for_materialization) const {
  if (scheduled_tokens == 0) {
    throw KvBlockAccountingError(
        "KV reservation requires positive scheduled tokens");
  }
  if (kv_accounted_tokens >
      std::numeric_limits<std::uint64_t>::max() - scheduled_tokens) {
    throw KvBlockAccountingError("KV token reservation overflows uint64");
  }

  const auto allocation = allocations_.find(request_id);
  if (allocation == allocations_.end()) {
    // Python checks admission against the whole resulting frontier but
    // materializes only the blocks needed by this iteration's scheduled
    // tokens.
    const std::uint64_t tokens =
        for_materialization
        ? scheduled_tokens
        : kv_accounted_tokens + scheduled_tokens;
    return ceil_div(tokens, block_size_);
  }

  if (allocation->second >
      std::numeric_limits<std::uint64_t>::max() / block_size_) {
    throw KvBlockAccountingError("KV reserved token count overflows uint64");
  }
  const std::uint64_t reserved_tokens =
      allocation->second * block_size_;
  const std::uint64_t required_tokens =
      kv_accounted_tokens + scheduled_tokens;
  if (required_tokens <= reserved_tokens) {
    return 0;
  }
  return ceil_div(required_tokens - reserved_tokens, block_size_);
}

bool KvBlockAccounting::can_reserve(
    RequestId request_id,
    std::uint64_t kv_accounted_tokens,
    std::uint64_t scheduled_tokens) const {
  const std::uint64_t required = additional_blocks_required(
      request_id,
      kv_accounted_tokens,
      scheduled_tokens,
      false);
  if (required > available_blocks()) {
    return false;
  }
  if (allocations_.contains(request_id)) {
    return true;
  }
  return available_blocks() - required >= watermark_blocks_;
}

void KvBlockAccounting::reserve(
    RequestId request_id,
    std::uint64_t kv_accounted_tokens,
    std::uint64_t scheduled_tokens) {
  if (!can_reserve(
          request_id,
          kv_accounted_tokens,
          scheduled_tokens)) {
    throw KvBlockAccountingError(
        "KV reservation exceeds capacity or watermark");
  }
  const std::uint64_t additional = additional_blocks_required(
      request_id,
      kv_accounted_tokens,
      scheduled_tokens,
      true);
  if (additional > available_blocks()) {
    throw KvBlockAccountingError(
        "KV materialization exceeds available blocks");
  }
  allocations_[request_id] += additional;
  total_allocated_blocks_ += additional;
}

std::uint64_t KvBlockAccounting::free(RequestId request_id) {
  const auto allocation = allocations_.find(request_id);
  if (allocation == allocations_.end()) {
    return 0;
  }
  const std::uint64_t freed = allocation->second;
  if (freed > total_allocated_blocks_) {
    throw KvBlockAccountingError("KV allocation total underflow");
  }
  total_allocated_blocks_ -= freed;
  allocations_.erase(allocation);
  return freed;
}

std::uint64_t KvBlockAccounting::allocated_blocks(
    RequestId request_id) const noexcept {
  const auto allocation = allocations_.find(request_id);
  return allocation == allocations_.end() ? 0 : allocation->second;
}

}  // namespace frontier::scheduler
