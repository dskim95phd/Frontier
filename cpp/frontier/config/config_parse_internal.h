#pragma once

#include <cstdint>
#include <initializer_list>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "frontier/config/config.h"

namespace frontier::config::parse_detail {

using Json = nlohmann::json;

void require_object(const Json &value, std::string_view context);
void require_exact_keys(const Json &object,
                        std::initializer_list<std::string_view> required_keys,
                        std::string_view context);
void require_keys(const Json &object,
                  std::initializer_list<std::string_view> required_keys,
                  std::initializer_list<std::string_view> optional_keys,
                  std::string_view context);
[[nodiscard]] std::string require_string(const Json &object,
                                         std::string_view field,
                                         std::string_view context);
[[nodiscard]] bool require_bool(const Json &object, std::string_view field,
                                std::string_view context);
[[nodiscard]] int require_int(const Json &object, std::string_view field,
                              std::string_view context);
[[nodiscard]] std::uint64_t require_uint64(const Json &object,
                                           std::string_view field,
                                           std::string_view context);
[[nodiscard]] std::vector<std::uint64_t>
require_uint64_array(const Json &object, std::string_view field,
                     std::string_view context);
[[nodiscard]] double require_finite_number(const Json &object,
                                           std::string_view field,
                                           std::string_view context);
[[nodiscard]] bool is_blank(std::string_view value);

[[nodiscard]] SimulationMode parse_simulation_mode(std::string_view value);
[[nodiscard]] SystemArchitecture
parse_system_architecture(std::string_view value);
[[nodiscard]] PrefixCachingKeyMode
parse_prefix_key_mode(std::string_view value);
[[nodiscard]] SchedulerType parse_scheduler_type(std::string_view value);
[[nodiscard]] SchedulingPolicy parse_scheduling_policy(std::string_view value);
[[nodiscard]] ClusterSchedulerType
parse_cluster_scheduler_type(std::string_view value);
[[nodiscard]] MoeRoutingMode parse_moe_routing_mode(std::string_view value);
[[nodiscard]] MoeRoutingDistribution
parse_moe_routing_distribution(std::string_view value);
[[nodiscard]] MoeRoutingLayerScope
parse_moe_routing_layer_scope(std::string_view value);

[[nodiscard]] ModelConfig parse_model(const Json &cluster,
                                      std::string_view context);
[[nodiscard]] MoeRoutingConfig parse_moe_routing(const Json &root);
[[nodiscard]] PrefixCacheConfig parse_prefix_cache(const Json &root);
[[nodiscard]] CpuKVCacheConfig parse_cpu_kv_cache(const Json &root);
[[nodiscard]] SchedulerConfig parse_scheduler(const Json &root);
[[nodiscard]] GpuMemoryConfig parse_gpu_memory(const Json &cluster,
                                               std::string_view context);
[[nodiscard]] ParallelismConfig parse_parallelism(const Json &root,
                                                  const ModelConfig &model);
[[nodiscard]] ClusterSchedulerConfig parse_cluster_scheduler(const Json &root);

[[nodiscard]] double
analytical_precision_size_bytes(std::string_view precision);
[[nodiscard]] ExecutionModelConfig
parse_execution_model(const Json &root, const ParallelismConfig &parallelism,
                      const ModelConfig &model);

} // namespace frontier::config::parse_detail
