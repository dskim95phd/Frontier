#include "frontier/scheduler/replica_scheduler/tiered_prefix_plan.h"

#include <algorithm>

#include "frontier/scheduler/replica_scheduler/base_replica_scheduler.h"

namespace frontier::scheduler {

entities::TieredPrefixPlan build_contiguous_tiered_prefix_plan(
    std::uint64_t query_blocks, std::uint64_t gpu_frontier_blocks,
    std::uint64_t cpu_frontier_blocks, std::uint64_t block_size,
    std::uint64_t prompt_tokens) {
    if (block_size == 0 || query_blocks != prompt_tokens / block_size) {
        throw SchedulerError("invalid contiguous tiered prefix dimensions");
    }
    entities::TieredPrefixPlan plan{};
    plan.query_blocks = query_blocks;
    plan.gpu_hit_frontier_blocks = std::min(gpu_frontier_blocks, query_blocks);
    const std::uint64_t cpu_frontier =
        std::min(cpu_frontier_blocks, query_blocks);
    plan.cpu_query_blocks = query_blocks > plan.gpu_hit_frontier_blocks
                                ? query_blocks - plan.gpu_hit_frontier_blocks
                                : 0;
    plan.hit_frontier_blocks =
        std::max(plan.gpu_hit_frontier_blocks, cpu_frontier);
    if (prompt_tokens % block_size == 0 &&
        plan.hit_frontier_blocks == query_blocks && query_blocks > 0) {
        --plan.hit_frontier_blocks;
    }
    plan.cpu_begin_block =
        std::min(plan.gpu_hit_frontier_blocks, plan.hit_frontier_blocks);
    plan.cpu_end_block = std::min(cpu_frontier, plan.hit_frontier_blocks);
    if (plan.cpu_end_block < plan.cpu_begin_block) {
        plan.cpu_end_block = plan.cpu_begin_block;
    }
    plan.block_size = block_size;
    plan.prompt_tokens = prompt_tokens;
    return plan;
}

std::uint64_t revalidate_contiguous_tiered_prefix_frontier(
    std::uint64_t current_gpu_frontier_blocks,
    const entities::StagedCpuKVCacheRestore &staged) {
    if (staged.block_size == 0 ||
        staged.query_blocks != staged.prompt_tokens / staged.block_size ||
        staged.cpu_begin_block > staged.cpu_end_block) {
        throw SchedulerError("invalid staged CPU prefix dimensions");
    }
    std::uint64_t reusable =
        std::min(current_gpu_frontier_blocks, staged.query_blocks);
    if (reusable >= staged.cpu_begin_block) {
        reusable = std::max(
            reusable, std::min(staged.cpu_end_block, staged.query_blocks));
    }
    if (staged.prompt_tokens % staged.block_size == 0 &&
        reusable == staged.query_blocks && reusable > 0) {
        --reusable;
    }
    return reusable;
}

} // namespace frontier::scheduler
