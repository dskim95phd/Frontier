#include <cmath>
#include <initializer_list>
#include <limits>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "frontier/config/config_parse_internal.h"
#include "frontier/core/precision.h"

namespace frontier::config::parse_detail {

void require_object(const Json &value, std::string_view context) {
    if (!value.is_object()) {
        throw ConfigError(std::string{context} + " must be a JSON object");
    }
}

void require_exact_keys(const Json &object,
                        std::initializer_list<std::string_view> required_keys,
                        std::string_view context) {
    require_object(object, context);

    std::unordered_set<std::string> allowed;
    allowed.reserve(required_keys.size());
    for (const std::string_view key : required_keys) {
        allowed.emplace(key);
        if (!object.contains(key)) {
            throw ConfigError(std::string{context} +
                              " is missing required field '" +
                              std::string{key} + "'");
        }
    }

    for (const auto &[key, value] : object.items()) {
        static_cast<void>(value);
        if (allowed.find(key) == allowed.end()) {
            throw ConfigError(std::string{context} +
                              " contains unknown field '" + key + "'");
        }
    }
}

void require_keys(const Json &object,
                  std::initializer_list<std::string_view> required_keys,
                  std::initializer_list<std::string_view> optional_keys,
                  std::string_view context) {
    require_object(object, context);
    std::unordered_set<std::string> allowed;
    allowed.reserve(required_keys.size() + optional_keys.size());
    for (const std::string_view key : required_keys) {
        allowed.emplace(key);
        if (!object.contains(key)) {
            throw ConfigError(std::string{context} +
                              " is missing required field '" +
                              std::string{key} + "'");
        }
    }
    for (const std::string_view key : optional_keys) {
        allowed.emplace(key);
    }
    for (const auto &[key, value] : object.items()) {
        static_cast<void>(value);
        if (allowed.find(key) == allowed.end()) {
            throw ConfigError(std::string{context} +
                              " contains unknown field '" + key + "'");
        }
    }
}

std::string require_string(const Json &object, std::string_view field,
                           std::string_view context) {
    const Json &value = object.at(field);
    if (!value.is_string()) {
        throw ConfigError(std::string{context} + "." + std::string{field} +
                          " must be a string");
    }
    return value.get<std::string>();
}

bool require_bool(const Json &object, std::string_view field,
                  std::string_view context) {
    const Json &value = object.at(field);
    if (!value.is_boolean()) {
        throw ConfigError(std::string{context} + "." + std::string{field} +
                          " must be a boolean");
    }
    return value.get<bool>();
}

int require_int(const Json &object, std::string_view field,
                std::string_view context) {
    const Json &value = object.at(field);
    if (!value.is_number_integer()) {
        throw ConfigError(std::string{context} + "." + std::string{field} +
                          " must be an integer");
    }
    const auto throw_out_of_range = [&context, &field] {
        throw ConfigError(std::string{context} + "." + std::string{field} +
                          " is outside the supported integer range");
    };

    if (value.is_number_unsigned()) {
        const Json::number_unsigned_t number =
            value.get<Json::number_unsigned_t>();
        if (number > static_cast<Json::number_unsigned_t>(
                         std::numeric_limits<int>::max())) {
            throw_out_of_range();
        }
        return static_cast<int>(number);
    }

    const Json::number_integer_t number = value.get<Json::number_integer_t>();
    if (number < static_cast<Json::number_integer_t>(
                     std::numeric_limits<int>::min()) ||
        number > static_cast<Json::number_integer_t>(
                     std::numeric_limits<int>::max())) {
        throw_out_of_range();
    }
    return static_cast<int>(number);
}

std::uint64_t require_uint64(const Json &object, std::string_view field,
                             std::string_view context) {
    const Json &value = object.at(field);
    if (!value.is_number_integer() || value.is_number_float()) {
        throw ConfigError(std::string{context} + "." + std::string{field} +
                          " must be a nonnegative integer");
    }
    try {
        if (value.is_number_unsigned()) {
            return value.get<std::uint64_t>();
        }
        const std::int64_t parsed = value.get<std::int64_t>();
        if (parsed < 0) {
            throw ConfigError(std::string{context} + "." + std::string{field} +
                              " must be nonnegative");
        }
        return static_cast<std::uint64_t>(parsed);
    } catch (const ConfigError &) {
        throw;
    } catch (const Json::exception &) {
        throw ConfigError(std::string{context} + "." + std::string{field} +
                          " is outside the supported integer range");
    }
}

std::vector<std::uint64_t> require_uint64_array(const Json &object,
                                                std::string_view field,
                                                std::string_view context) {
    const Json &value = object.at(field);
    if (!value.is_array()) {
        throw ConfigError(std::string{context} + "." + std::string{field} +
                          " must be an array");
    }
    std::vector<std::uint64_t> result;
    result.reserve(value.size());
    for (std::size_t index = 0; index < value.size(); ++index) {
        const Json &element = value[index];
        const std::string element_context = std::string{context} + "." +
                                            std::string{field} + "[" +
                                            std::to_string(index) + "]";
        if (!element.is_number_integer() || element.is_number_float()) {
            throw ConfigError(element_context +
                              " must be a nonnegative integer");
        }
        try {
            if (element.is_number_unsigned()) {
                result.push_back(element.get<std::uint64_t>());
            } else {
                const std::int64_t parsed = element.get<std::int64_t>();
                if (parsed < 0) {
                    throw ConfigError(element_context + " must be nonnegative");
                }
                result.push_back(static_cast<std::uint64_t>(parsed));
            }
        } catch (const ConfigError &) {
            throw;
        } catch (const Json::exception &) {
            throw ConfigError(element_context +
                              " is outside the supported integer range");
        }
    }
    return result;
}

double require_finite_number(const Json &object, std::string_view field,
                             std::string_view context) {
    const Json &value = object.at(field);
    if (!value.is_number()) {
        throw ConfigError(std::string{context} + "." + std::string{field} +
                          " must be numeric");
    }
    double parsed = 0.0;
    try {
        parsed = value.get<double>();
    } catch (const Json::exception &) {
        throw ConfigError(std::string{context} + "." + std::string{field} +
                          " is outside the supported numeric range");
    }
    if (!std::isfinite(parsed)) {
        throw ConfigError(std::string{context} + "." + std::string{field} +
                          " must be finite");
    }
    return parsed;
}

bool is_blank(std::string_view value) {
    return value.find_first_not_of(" \t\r\n") == std::string_view::npos;
}

SimulationMode parse_simulation_mode(std::string_view value) {
    if (value == "offline") {
        return SimulationMode::kOffline;
    }
    if (value == "online") {
        return SimulationMode::kOnline;
    }
    throw ConfigError(
        "config.simulation_mode must be 'offline' or 'online', got '" +
        std::string{value} + "'");
}

SystemArchitecture parse_system_architecture(std::string_view value) {
    if (value == "co-location") {
        return SystemArchitecture::kCoLocation;
    }
    if (value == "pd-disaggregation") {
        return SystemArchitecture::kPdDisaggregation;
    }
    throw ConfigError("config.system_architecture must be 'co-location' or "
                      "'pd-disaggregation', got '" +
                      std::string{value} + "'");
}

PrefixCachingKeyMode parse_prefix_key_mode(std::string_view value) {
    if (value == "session") {
        return PrefixCachingKeyMode::kSession;
    }
    if (value == "block_hash") {
        throw ConfigError(
            "config.prefix_cache.key_mode='block_hash' is outside the C++ "
            "port; only 'session' is supported");
    }
    throw ConfigError("config.prefix_cache.key_mode must be 'session', got '" +
                      std::string{value} + "'");
}

SchedulerType parse_scheduler_type(std::string_view value) {
    if (value == "vllm_v1") {
        return SchedulerType::kVllmV1;
    }
    throw ConfigError("config.scheduler.type must be 'vllm_v1', got '" +
                      std::string{value} + "'");
}

SchedulingPolicy parse_scheduling_policy(std::string_view value) {
    if (value == "fcfs") {
        return SchedulingPolicy::kFcfs;
    }
    throw ConfigError(
        "config.scheduler.scheduling_policy must be 'fcfs', got '" +
        std::string{value} + "'");
}

ClusterSchedulerType parse_cluster_scheduler_type(std::string_view value) {
    if (value == "round_robin") {
        return ClusterSchedulerType::kRoundRobin;
    }
    if (value == "sticky_round_robin") {
        return ClusterSchedulerType::kStickyRoundRobin;
    }
    if (value == "vllm_queue_aware") {
        return ClusterSchedulerType::kVllmQueueAware;
    }
    if (value == "kv_aware") {
        return ClusterSchedulerType::kKvAware;
    }
    if (value == "cache_aware") {
        return ClusterSchedulerType::kCacheAware;
    }
    throw ConfigError(
        "config.cluster_scheduler.type must be 'round_robin', "
        "'sticky_round_robin', 'vllm_queue_aware', 'kv_aware', or "
        "'cache_aware', got '" +
        std::string{value} + "'");
}

MoeRoutingMode parse_moe_routing_mode(std::string_view value) {
    if (value == "simulation") {
        return MoeRoutingMode::kSimulation;
    }
    if (value == "uniform_legacy") {
        return MoeRoutingMode::kUniformLegacy;
    }
    if (value == "uniform_random") {
        return MoeRoutingMode::kUniformRandom;
    }
    throw ConfigError("config.moe_routing.mode must be 'simulation', "
                      "'uniform_legacy', or 'uniform_random', got '" +
                      std::string{value} + "'");
}

MoeRoutingDistribution parse_moe_routing_distribution(std::string_view value) {
    if (value == "balanced") {
        return MoeRoutingDistribution::kBalanced;
    }
    if (value == "random") {
        return MoeRoutingDistribution::kRandom;
    }
    if (value == "skewed") {
        return MoeRoutingDistribution::kSkewed;
    }
    if (value == "zipf") {
        return MoeRoutingDistribution::kZipf;
    }
    throw ConfigError("config.moe_routing.distribution must be 'balanced', "
                      "'random', 'skewed', or 'zipf', got '" +
                      std::string{value} + "'");
}

MoeRoutingLayerScope parse_moe_routing_layer_scope(std::string_view value) {
    if (value == "shared") {
        return MoeRoutingLayerScope::kShared;
    }
    if (value == "per_layer") {
        return MoeRoutingLayerScope::kPerLayer;
    }
    throw ConfigError("config.moe_routing.layer_scope must be 'shared' or "
                      "'per_layer', got '" +
                      std::string{value} + "'");
}

} // namespace frontier::config::parse_detail
