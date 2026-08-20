#pragma once

#include <cstdint>

#include "frontier/entities/cpu_kv_cache_transfer_info.h"

namespace frontier::scheduler {

[[nodiscard]] entities::TieredPrefixPlan build_contiguous_tiered_prefix_plan(
    std::uint64_t query_blocks, std::uint64_t gpu_frontier_blocks,
    std::uint64_t cpu_frontier_blocks, std::uint64_t block_size,
    std::uint64_t prompt_tokens);

[[nodiscard]] std::uint64_t revalidate_contiguous_tiered_prefix_frontier(
    std::uint64_t current_gpu_frontier_blocks,
    const entities::StagedCpuKVCacheRestore &staged);

} // namespace frontier::scheduler
