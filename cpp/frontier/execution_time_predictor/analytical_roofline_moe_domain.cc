#include "frontier/execution_time_predictor/analytical_roofline_execution_time_predictor.h"

#include <cstddef>

namespace frontier::execution_time_predictor::detail {

ExpertParallelDomain::ExpertParallelDomain(std::uint64_t total_experts,
                                           std::uint64_t expert_parallel_size)
    : total_experts_(total_experts),
      expert_parallel_size_(expert_parallel_size) {
    if (total_experts_ == 0 || expert_parallel_size_ == 0 ||
        total_experts_ % expert_parallel_size_ != 0) {
        throw ParallelDomainError(
            "expert count must be positive and divisible by EP size");
    }
}

ExpertRange ExpertParallelDomain::expert_range(std::uint64_t lane) const {
    if (lane >= expert_parallel_size_) {
        throw ParallelDomainError("EP lane is outside the parallel domain");
    }
    const std::uint64_t width = experts_per_lane();
    return ExpertRange{lane * width, (lane + 1) * width};
}

std::uint64_t ExpertParallelDomain::owner(std::uint64_t expert_id) const {
    if (expert_id >= total_experts_) {
        throw ParallelDomainError("expert ID is outside the parallel domain");
    }
    return expert_id / experts_per_lane();
}

std::vector<std::vector<std::uint64_t>> ExpertParallelDomain::partition(
    const std::vector<std::uint64_t> &global_counts) const {
    if (global_counts.size() != static_cast<std::size_t>(total_experts_)) {
        throw ParallelDomainError(
            "global expert allocation has the wrong size");
    }

    std::vector<std::vector<std::uint64_t>> lanes;
    lanes.reserve(static_cast<std::size_t>(expert_parallel_size_));
    for (std::uint64_t lane = 0; lane < expert_parallel_size_; ++lane) {
        const ExpertRange range = expert_range(lane);
        lanes.emplace_back(
            global_counts.begin() + static_cast<std::ptrdiff_t>(range.begin),
            global_counts.begin() + static_cast<std::ptrdiff_t>(range.end));
    }
    return lanes;
}

} // namespace frontier::execution_time_predictor::detail
