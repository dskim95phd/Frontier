#pragma once

#include <cstdint>
#include <stdexcept>
#include <unordered_map>

#include "frontier/config/config.h"
#include "frontier/core/ids.h"

namespace frontier::scheduler {

class KvBlockAccountingError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

class KvBlockAccounting {
  public:
    explicit KvBlockAccounting(const config::SchedulerConfig &config);

    [[nodiscard]] bool can_reserve(RequestId request_id,
                                   std::uint64_t kv_accounted_tokens,
                                   std::uint64_t scheduled_tokens) const;
    void reserve(RequestId request_id, std::uint64_t kv_accounted_tokens,
                 std::uint64_t scheduled_tokens);
    [[nodiscard]] std::uint64_t free(RequestId request_id);

    [[nodiscard]] std::uint64_t
    allocated_blocks(RequestId request_id) const noexcept;
    [[nodiscard]] std::uint64_t total_allocated_blocks() const noexcept {
        return total_allocated_blocks_;
    }
    [[nodiscard]] std::uint64_t available_blocks() const noexcept {
        return num_blocks_ - total_allocated_blocks_;
    }
    [[nodiscard]] std::uint64_t capacity_blocks() const noexcept {
        return num_blocks_;
    }
    [[nodiscard]] std::uint64_t watermark_blocks() const noexcept {
        return watermark_blocks_;
    }
    [[nodiscard]] bool empty() const noexcept { return allocations_.empty(); }
    [[nodiscard]] std::size_t allocation_count() const noexcept {
        return allocations_.size();
    }

  private:
    [[nodiscard]] static std::uint64_t ceil_div(std::uint64_t numerator,
                                                std::uint64_t denominator);
    [[nodiscard]] std::uint64_t additional_blocks_required(
        RequestId request_id, std::uint64_t kv_accounted_tokens,
        std::uint64_t scheduled_tokens, bool for_materialization) const;

    std::uint64_t block_size_;
    std::uint64_t num_blocks_;
    std::uint64_t watermark_blocks_;
    std::uint64_t total_allocated_blocks_ = 0;
    std::unordered_map<RequestId, std::uint64_t, StrongIdHash<RequestId>>
        allocations_;
};

} // namespace frontier::scheduler
