#pragma once

#include <cstdint>
#include <optional>

#include "frontier/config/config.h"

namespace frontier::simulator::simulator_detail {

struct GpuKvPhysicalBlockLayout {
    std::uint64_t max_rank_bytes = 0;
    std::uint64_t pipeline_bytes = 0;
};

[[nodiscard]] GpuKvPhysicalBlockLayout
gpu_kv_physical_block_layout(const config::ClusterRuntimeConfig &runtime,
                             const config::SimulationConfig &config);
[[nodiscard]] std::optional<std::uint64_t>
total_hbm_bytes_per_gpu(const config::ClusterRuntimeConfig &runtime);

} // namespace frontier::simulator::simulator_detail
