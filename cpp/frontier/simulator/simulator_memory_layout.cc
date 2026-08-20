#include "frontier/simulator/simulator_memory_layout.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <string>
#include <string_view>

#include "frontier/core/precision.h"
#include "frontier/kv_cache_transfer/analytical_transfer.h"
#include "frontier/simulator/simulator.h"

namespace frontier::simulator::simulator_detail {

double precision_dtype_size_bytes(std::string_view precision) {
    const std::optional<Precision> parsed = parse_precision(precision);
    if (parsed.has_value()) {
        return storage_bytes_per_element(*parsed);
    }
    throw SimulationError("unsupported precision: " + std::string{precision});
}

double kv_cache_dtype_size_bytes(const config::ClusterRuntimeConfig &runtime,
                                 const config::SimulationConfig &config) {
    if (config.system_architecture ==
        config::SystemArchitecture::kPdDisaggregation) {
        return config.pdd().kv_cache_transfer.kv_cache_dtype_size_bytes;
    }
    if (runtime.execution_model.type !=
        config::ExecutionModelType::kAnalytical) {
        // Fixed-latency configs do not carry a precision declaration.  The
        // historical simulator contract uses BF16-sized KV by default.
        return 2.0;
    }
    return precision_dtype_size_bytes(
        runtime.execution_model.analytical.kv_cache_precision());
}

GpuKvPhysicalBlockLayout
gpu_kv_physical_block_layout(const config::ClusterRuntimeConfig &runtime,
                             const config::SimulationConfig &config) {
    const double dtype_bytes = kv_cache_dtype_size_bytes(runtime, config);
    GpuKvPhysicalBlockLayout result{};
    const std::uint64_t dcp = runtime.parallelism.decode_context_parallel_size;
    const std::uint64_t tp = runtime.parallelism.tensor_parallel_size;
    if (dcp == 0 || tp == 0 || tp % dcp != 0) {
        throw SimulationError("invalid GPU KV physical parallelism layout");
    }
    const std::uint64_t mla_copies = tp / dcp;
    const auto add_pipeline_bytes = [&](std::uint64_t stage_bytes) {
        if (stage_bytes >
            std::numeric_limits<std::uint64_t>::max() - result.pipeline_bytes) {
            throw SimulationError("GPU KV pipeline byte layout overflows");
        }
        result.pipeline_bytes += stage_bytes;
    };

    if (!runtime.gpu_memory.pipeline_stage_memory_profiles.empty()) {
        for (const config::PipelineStageMemoryProfile &profile :
             runtime.gpu_memory.pipeline_stage_memory_profiles) {
            std::uint64_t stage_bytes = 0;
            for (const std::uint64_t rank_bytes :
                 profile.kv_bytes_per_block_by_rank) {
                result.max_rank_bytes =
                    std::max(result.max_rank_bytes, rank_bytes);
                if (rank_bytes >
                    std::numeric_limits<std::uint64_t>::max() - stage_bytes) {
                    throw SimulationError("GPU KV stage byte layout overflows");
                }
                stage_bytes += rank_bytes;
            }
            if (runtime.model.use_mla) {
                if (stage_bytes >
                    std::numeric_limits<std::uint64_t>::max() / mla_copies) {
                    throw SimulationError(
                        "GPU KV stage replication bytes overflow");
                }
                stage_bytes *= mla_copies;
            }
            add_pipeline_bytes(stage_bytes);
        }
    } else {
        for (std::uint64_t stage = 0;
             stage < runtime.parallelism.pipeline_parallel_size; ++stage) {
            const config::PipelineStageLayerRange layers =
                config::pipeline_stage_layer_range(runtime.model.num_layers,
                                                   runtime.parallelism, stage);
            for (std::uint64_t rank = 0; rank < dcp; ++rank) {
                result.max_rank_bytes = std::max(
                    result.max_rank_bytes,
                    kv_cache_transfer::
                        model_kv_cache_size_bytes_stage_rank_local(
                            runtime.scheduler.block_size, runtime.model,
                            dtype_bytes, layers, dcp, rank));
            }
            add_pipeline_bytes(
                kv_cache_transfer::model_kv_cache_size_bytes_stage_physical(
                    runtime.scheduler.block_size, runtime.model, dtype_bytes,
                    layers, tp, dcp));
        }
    }
    if (result.max_rank_bytes == 0 || result.pipeline_bytes == 0 ||
        result.pipeline_bytes < result.max_rank_bytes) {
        throw SimulationError("GPU KV physical block layout is empty");
    }
    return result;
}

std::optional<std::uint64_t>
total_hbm_bytes_per_gpu(const config::ClusterRuntimeConfig &runtime) {
    // HBM is an explicit part of the runtime contract.  Use the configured
    // capacity for every execution model/device instead of inferring a
    // nominal device size from the analytical device name.  Keep a nullopt
    // fallback for callers that construct legacy in-memory configs without a
    // capacity; parsed configs reject that shape before simulation starts.
    if (runtime.gpu_memory.capacity_bytes_per_gpu != 0) {
        return runtime.gpu_memory.capacity_bytes_per_gpu;
    }
    return std::nullopt;
}

} // namespace frontier::simulator::simulator_detail
