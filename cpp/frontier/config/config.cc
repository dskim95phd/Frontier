#include "frontier/config/config.h"

namespace frontier::config {

std::string_view to_string(SimulationMode mode) noexcept {
    switch (mode) {
    case SimulationMode::kOffline:
        return "offline";
    case SimulationMode::kOnline:
        return "online";
    }
    return "unknown";
}

std::string_view to_string(SystemArchitecture architecture) noexcept {
    switch (architecture) {
    case SystemArchitecture::kCoLocation:
        return "co-location";
    case SystemArchitecture::kPdDisaggregation:
        return "pd-disaggregation";
    }
    return "unknown";
}

std::string_view to_string(PrefixCachingKeyMode key_mode) noexcept {
    switch (key_mode) {
    case PrefixCachingKeyMode::kSession:
        return "session";
    }
    return "unknown";
}

std::string_view to_string(SchedulerType type) noexcept {
    switch (type) {
    case SchedulerType::kVllmV1:
        return "vllm_v1";
    }
    return "unknown";
}

std::string_view to_string(SchedulingPolicy policy) noexcept {
    switch (policy) {
    case SchedulingPolicy::kFcfs:
        return "fcfs";
    }
    return "unknown";
}

std::string_view to_string(ClusterSchedulerType type) noexcept {
    switch (type) {
    case ClusterSchedulerType::kRoundRobin:
        return "round_robin";
    case ClusterSchedulerType::kStickyRoundRobin:
        return "sticky_round_robin";
    case ClusterSchedulerType::kVllmQueueAware:
        return "vllm_queue_aware";
    case ClusterSchedulerType::kKvAware:
        return "kv_aware";
    case ClusterSchedulerType::kCacheAware:
        return "cache_aware";
    }
    return "unknown";
}

std::string_view to_string(ModelKind kind) noexcept {
    switch (kind) {
    case ModelKind::kDense:
        return "dense";
    case ModelKind::kMoe:
        return "moe";
    }
    return "unknown";
}

std::string_view to_string(MoeRoutingMode mode) noexcept {
    switch (mode) {
    case MoeRoutingMode::kSimulation:
        return "simulation";
    case MoeRoutingMode::kUniformLegacy:
        return "uniform_legacy";
    case MoeRoutingMode::kUniformRandom:
        return "uniform_random";
    }
    return "unknown";
}

std::string_view to_string(MoeRoutingLayerScope layer_scope) noexcept {
    switch (layer_scope) {
    case MoeRoutingLayerScope::kShared:
        return "shared";
    case MoeRoutingLayerScope::kPerLayer:
        return "per_layer";
    }
    return "unknown";
}

MoeRoutingLayerScope
default_moe_routing_layer_scope(MoeRoutingMode mode,
                                MoeRoutingDistribution distribution) noexcept {
    // uniform_legacy spreads tokens evenly without consulting the seed at all,
    // and simulation/balanced weights every expert identically, so both
    // produced one assignment for every layer before the field existed.
    if (mode == MoeRoutingMode::kUniformLegacy) {
        return MoeRoutingLayerScope::kShared;
    }
    if (mode == MoeRoutingMode::kSimulation &&
        distribution == MoeRoutingDistribution::kBalanced) {
        return MoeRoutingLayerScope::kShared;
    }
    return MoeRoutingLayerScope::kPerLayer;
}

std::string_view to_string(MoeRoutingDistribution distribution) noexcept {
    switch (distribution) {
    case MoeRoutingDistribution::kBalanced:
        return "balanced";
    case MoeRoutingDistribution::kRandom:
        return "random";
    case MoeRoutingDistribution::kSkewed:
        return "skewed";
    case MoeRoutingDistribution::kZipf:
        return "zipf";
    }
    return "unknown";
}

std::string_view to_string(ExecutionModelType type) noexcept {
    switch (type) {
    case ExecutionModelType::kFixed:
        return "fixed";
    case ExecutionModelType::kAnalytical:
        return "analytical";
    }
    return "unknown";
}

std::string_view to_string(CpuKVCacheEvictionPolicy policy) noexcept {
    switch (policy) {
    case CpuKVCacheEvictionPolicy::kSessionLruSuffix:
        return "session_lru_suffix";
    }
    return "unknown";
}

std::string_view to_string(CpuKVCacheCapacityPressurePolicy policy) noexcept {
    switch (policy) {
    case CpuKVCacheCapacityPressurePolicy::kPrefixFit:
        return "prefix_fit";
    case CpuKVCacheCapacityPressurePolicy::kSkipOffload:
        return "skip_offload";
    }
    return "unknown";
}

std::string_view to_string(CpuKVCacheTransferConcurrency concurrency) noexcept {
    switch (concurrency) {
    case CpuKVCacheTransferConcurrency::kFullDuplexSerialized:
        return "full_duplex_serialized";
    }
    return "unknown";
}

ClusterRuntimeConfig &SimulationConfig::cluster() {
    auto *value = std::get_if<ClusterRuntimeConfig>(&runtime);
    if (value == nullptr) {
        throw ConfigError("simulation config has no single cluster runtime");
    }
    return *value;
}

const ClusterRuntimeConfig &SimulationConfig::cluster() const {
    const auto *value = std::get_if<ClusterRuntimeConfig>(&runtime);
    if (value == nullptr) {
        throw ConfigError("simulation config has no single cluster runtime");
    }
    return *value;
}

PddRuntimeConfig &SimulationConfig::pdd() {
    auto *value = std::get_if<PddRuntimeConfig>(&runtime);
    if (value == nullptr) {
        throw ConfigError("simulation config is not a PDD config");
    }
    return *value;
}

const PddRuntimeConfig &SimulationConfig::pdd() const {
    const auto *value = std::get_if<PddRuntimeConfig>(&runtime);
    if (value == nullptr) {
        throw ConfigError("simulation config is not a PDD config");
    }
    return *value;
}

} // namespace frontier::config
