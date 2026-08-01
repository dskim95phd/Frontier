#include "frontier/config/config.h"

#include <cmath>
#include <initializer_list>
#include <limits>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace frontier::config {
namespace {

using Json = nlohmann::json;

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
    throw ConfigError(
        "config.cluster_scheduler.type must be 'round_robin', got '" +
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

ModelConfig parse_model(const Json &cluster, std::string_view context) {
    ModelConfig parsed = load_model_config(
        require_string(cluster, "model_name", context));
    if (cluster.contains("total_expert_num")) {
        parsed.total_expert_num =
            require_uint64(cluster, "total_expert_num", context);
    }
    if (cluster.contains("router_topk")) {
        parsed.router_topk = require_uint64(cluster, "router_topk", context);
    }
    if (!parsed.is_moe()) {
        if (parsed.total_expert_num != 1 || parsed.router_topk != 1) {
            throw ConfigError(std::string{context} +
                              " dense model requires total_expert_num=1 and "
                              "router_topk=1");
        }
        return parsed;
    }
    if (parsed.total_expert_num < 2 ||
        parsed.total_expert_num > parsed.num_experts) {
        throw ConfigError(std::string{context} +
                          ".total_expert_num must be in [2, model "
                          "num_experts]");
    }
    if (parsed.router_topk == 0 ||
        parsed.router_topk > parsed.total_expert_num) {
        throw ConfigError(std::string{context} +
                          ".router_topk must be in [1, total_expert_num]");
    }
    return parsed;
}

MoeRoutingConfig parse_moe_routing(const Json &root) {
    const Json &routing = root.at("moe_routing");
    require_exact_keys(routing, {"mode", "distribution", "seed"},
                       "config.moe_routing");
    return [&]() {
        MoeRoutingConfig value{};
        value.mode = parse_moe_routing_mode(
            require_string(routing, "mode", "config.moe_routing"));
        value.distribution = parse_moe_routing_distribution(
            require_string(routing, "distribution", "config.moe_routing"));
        value.seed = require_uint64(routing, "seed", "config.moe_routing");
        return value;
    }();
}

PrefixCacheConfig parse_prefix_cache(const Json &root) {
    const Json &prefix_cache = root.at("prefix_cache");
    require_exact_keys(prefix_cache, {"enabled", "key_mode"},
                       "config.prefix_cache");
    return [&]() {
        PrefixCacheConfig value{};
        value.enabled =
            require_bool(prefix_cache, "enabled", "config.prefix_cache");
        value.key_mode = parse_prefix_key_mode(
            require_string(prefix_cache, "key_mode", "config.prefix_cache"));
        return value;
    }();
}

SchedulerConfig parse_scheduler(const Json &root) {
    const Json &scheduler = root.at("scheduler");
    require_exact_keys(scheduler,
                       {
                           "type",
                           "scheduling_policy",
                           "batch_size_cap",
                           "max_tokens_in_batch",
                           "enable_preemption",
                           "enable_chunked_prefill",
                           "long_prefill_token_threshold",
                           "block_size",
                           "num_blocks",
                           "watermark_blocks_fraction",
                           "num_preallocate_tokens",
                       },
                       "config.scheduler");

    SchedulerConfig parsed = [&]() {
        SchedulerConfig value{};
        value.type = parse_scheduler_type(
            require_string(scheduler, "type", "config.scheduler"));
        value.scheduling_policy = parse_scheduling_policy(
            require_string(scheduler, "scheduling_policy", "config.scheduler"));
        value.batch_size_cap =
            require_uint64(scheduler, "batch_size_cap", "config.scheduler");
        value.max_tokens_in_batch = require_uint64(
            scheduler, "max_tokens_in_batch", "config.scheduler");
        value.enable_preemption =
            require_bool(scheduler, "enable_preemption", "config.scheduler");
        value.enable_chunked_prefill = require_bool(
            scheduler, "enable_chunked_prefill", "config.scheduler");
        value.long_prefill_token_threshold = require_uint64(
            scheduler, "long_prefill_token_threshold", "config.scheduler");
        value.block_size =
            require_uint64(scheduler, "block_size", "config.scheduler");
        value.num_blocks =
            require_uint64(scheduler, "num_blocks", "config.scheduler");
        value.watermark_blocks_fraction = require_finite_number(
            scheduler, "watermark_blocks_fraction", "config.scheduler");
        value.num_preallocate_tokens = require_uint64(
            scheduler, "num_preallocate_tokens", "config.scheduler");
        return value;
    }();

    if (parsed.batch_size_cap == 0) {
        throw ConfigError("config.scheduler.batch_size_cap must be positive");
    }
    if (parsed.max_tokens_in_batch == 0) {
        throw ConfigError(
            "config.scheduler.max_tokens_in_batch must be positive");
    }
    if (parsed.block_size == 0) {
        throw ConfigError("config.scheduler.block_size must be positive");
    }
    if (parsed.num_blocks == 0) {
        throw ConfigError("config.scheduler.num_blocks must be positive");
    }
    if (parsed.watermark_blocks_fraction < 0.0 ||
        parsed.watermark_blocks_fraction >= 1.0) {
        throw ConfigError(
            "config.scheduler.watermark_blocks_fraction must be in [0, 1)");
    }
    if (parsed.long_prefill_token_threshold > 0 &&
        !parsed.enable_chunked_prefill) {
        throw ConfigError(
            "config.scheduler.long_prefill_token_threshold > 0 requires "
            "enable_chunked_prefill=true");
    }
    return parsed;
}

ParallelismConfig parse_parallelism(const Json &root,
                                    const ModelConfig &model) {
    const Json &parallelism = root.at("parallelism");
    require_exact_keys(parallelism,
                       {
                           "num_replicas",
                           "tensor_parallel_size",
                           "pipeline_parallel_size",
                           "data_parallel_size",
                           "moe_tensor_parallel_size",
                           "moe_expert_parallel_size",
                       },
                       "config.parallelism");
    ParallelismConfig parsed = [&]() {
        ParallelismConfig value{};
        value.num_replicas =
            require_uint64(parallelism, "num_replicas", "config.parallelism");
        value.tensor_parallel_size = require_uint64(
            parallelism, "tensor_parallel_size", "config.parallelism");
        value.pipeline_parallel_size = require_uint64(
            parallelism, "pipeline_parallel_size", "config.parallelism");
        value.data_parallel_size = require_uint64(
            parallelism, "data_parallel_size", "config.parallelism");
        value.moe_tensor_parallel_size = require_uint64(
            parallelism, "moe_tensor_parallel_size", "config.parallelism");
        value.moe_expert_parallel_size = require_uint64(
            parallelism, "moe_expert_parallel_size", "config.parallelism");
        return value;
    }();
    if (parsed.num_replicas == 0 || parsed.tensor_parallel_size == 0 ||
        parsed.pipeline_parallel_size == 0 || parsed.data_parallel_size == 0 ||
        parsed.moe_tensor_parallel_size == 0 ||
        parsed.moe_expert_parallel_size == 0) {
        throw ConfigError("all config.parallelism dimensions must be positive");
    }
    if (parsed.tensor_parallel_size != 1 && parsed.tensor_parallel_size != 2 &&
        parsed.tensor_parallel_size != 4 && parsed.tensor_parallel_size != 8) {
        throw ConfigError("config.parallelism.tensor_parallel_size must be one "
                          "of 1, 2, 4, 8");
    }
    if (model.num_layers % parsed.pipeline_parallel_size != 0) {
        throw ConfigError("model num_layers must be divisible by "
                          "config.parallelism.pipeline_parallel_size");
    }
    if (model.hidden_size % parsed.tensor_parallel_size != 0 ||
        model.num_query_heads % parsed.tensor_parallel_size != 0) {
        throw ConfigError(
            "model hidden size and attention heads must be divisible by "
            "config.parallelism.tensor_parallel_size");
    }
    if (model.is_moe()) {
        if (parsed.attention_parallel_size() != parsed.moe_parallel_size()) {
            throw ConfigError(
                "MoE shared parallel domain requires "
                "tensor_parallel_size*data_parallel_size == "
                "moe_tensor_parallel_size*moe_expert_parallel_size");
        }
        if (model.total_expert_num % parsed.moe_expert_parallel_size !=
            0) {
            throw ConfigError("total_expert_num must be divisible by "
                              "moe_expert_parallel_size");
        }
        if (model.intermediate_size % parsed.moe_tensor_parallel_size != 0) {
            throw ConfigError("MoE intermediate size must be divisible by "
                              "moe_tensor_parallel_size");
        }
    } else if (parsed.moe_tensor_parallel_size != 1 ||
               parsed.moe_expert_parallel_size != 1) {
        throw ConfigError("dense models require MoE TP=1 and EP=1");
    }
    const auto checked_multiply = [](std::uint64_t left, std::uint64_t right,
                                     std::string_view context) {
        if (right != 0 &&
            left > std::numeric_limits<std::uint64_t>::max() / right) {
            throw ConfigError(std::string{context} + " overflows uint64");
        }
        return left * right;
    };
    std::uint64_t accelerators =
        checked_multiply(parsed.num_replicas, parsed.data_parallel_size,
                         "config.parallelism accelerator count");
    accelerators = checked_multiply(accelerators, parsed.pipeline_parallel_size,
                                    "config.parallelism accelerator count");
    accelerators = checked_multiply(accelerators, parsed.tensor_parallel_size,
                                    "config.parallelism accelerator count");
    static_cast<void>(accelerators);
    return parsed;
}

ClusterSchedulerConfig parse_cluster_scheduler(const Json &root) {
    const Json &scheduler = root.at("cluster_scheduler");
    require_exact_keys(scheduler, {"type"}, "config.cluster_scheduler");
    return [&]() {
        ClusterSchedulerConfig value{};
        value.type = parse_cluster_scheduler_type(
            require_string(scheduler, "type", "config.cluster_scheduler"));
        return value;
    }();
}

std::vector<double> require_finite_number_array(const Json &object,
                                                std::string_view field,
                                                std::string_view context) {
    const Json &value = object.at(field);
    if (!value.is_array()) {
        throw ConfigError(std::string{context} + "." + std::string{field} +
                          " must be an array");
    }
    std::vector<double> result;
    result.reserve(value.size());
    for (std::size_t index = 0; index < value.size(); ++index) {
        const Json &element = value[index];
        if (!element.is_number()) {
            throw ConfigError(std::string{context} + "." + std::string{field} +
                              "[" + std::to_string(index) +
                              "] must be numeric");
        }
        double parsed = 0.0;
        try {
            parsed = element.get<double>();
        } catch (const Json::exception &) {
            throw ConfigError(std::string{context} + "." + std::string{field} +
                              "[" + std::to_string(index) +
                              "] is outside the supported numeric range");
        }
        if (!std::isfinite(parsed) || parsed < 0.0) {
            throw ConfigError(std::string{context} + "." + std::string{field} +
                              "[" + std::to_string(index) +
                              "] must be finite and nonnegative");
        }
        result.push_back(parsed);
    }
    return result;
}

ExecutionModelConfig parse_execution_model(const Json &root,
                                           const ParallelismConfig &parallelism,
                                           const ModelConfig &model) {
    const Json &execution = root.at("execution_model");
    require_object(execution, "config.execution_model");
    if (!execution.contains("type")) {
        throw ConfigError(
            "config.execution_model is missing required field 'type'");
    }
    const std::string type =
        require_string(execution, "type", "config.execution_model");

    if (type == "fixed") {
        require_exact_keys(execution, {"type", "stage_latencies_ms"},
                           "config.execution_model");
        std::vector<double> stage_latencies = require_finite_number_array(
            execution, "stage_latencies_ms", "config.execution_model");
        if (stage_latencies.size() != parallelism.pipeline_parallel_size) {
            throw ConfigError(
                "config.execution_model.stage_latencies_ms length must equal "
                "config.parallelism.pipeline_parallel_size");
        }
        return [&]() {
            ExecutionModelConfig value{};
            value.type = ExecutionModelType::kFixed;
            value.fixed = [&]() {
                FixedExecutionModelConfig value{};
                value.batch_latency_ms = stage_latencies.front();
                value.stage_latencies_ms = std::move(stage_latencies);
                return value;
            }();
            value.analytical = AnalyticalExecutionModelConfig{};
            return value;
        }();
    }

    if (type == "analytical") {
        require_exact_keys(execution,
                           {
                               "type",
                               "device",
                               "precision",
                               "network_bandwidth_gbps",
                               "network_latency_us",
                               "intra_node_bandwidth_gbps",
                           },
                           "config.execution_model");
        AnalyticalExecutionModelConfig analytical = [&]() {
            AnalyticalExecutionModelConfig value{};
            value.device =
                require_string(execution, "device", "config.execution_model");
            value.precision = require_string(execution, "precision",
                                             "config.execution_model");
            value.tensor_parallel_size = parallelism.tensor_parallel_size;
            value.network_bandwidth_gbps = require_finite_number(
                execution, "network_bandwidth_gbps", "config.execution_model");
            value.network_latency_us = require_finite_number(
                execution, "network_latency_us", "config.execution_model");
            value.intra_node_bandwidth_gbps =
                require_finite_number(execution, "intra_node_bandwidth_gbps",
                                      "config.execution_model");
            return value;
        }();
        if (analytical.device != "rubin" ||
            (analytical.precision != "fp16" &&
             analytical.precision != "bf16")) {
            throw ConfigError(
                "analytical execution requires Rubin with fp16 or bf16 "
                "precision");
        }
        if (!model.attention.execution_enabled ||
            model.attention.memory_layout ==
                attention::AttentionMemoryLayout::kFrozenDsa) {
            throw ConfigError(
                "analytical execution does not support frozen DSA attention");
        }
        if (analytical.network_bandwidth_gbps <= 0.0 ||
            analytical.intra_node_bandwidth_gbps <= 0.0 ||
            analytical.network_latency_us < 0.0) {
            throw ConfigError(
                "analytical bandwidths must be positive and latency must be "
                "nonnegative");
        }
        return [&]() {
            ExecutionModelConfig value{};
            value.type = ExecutionModelType::kAnalytical;
            value.fixed = FixedExecutionModelConfig{};
            value.analytical = std::move(analytical);
            return value;
        }();
    }

    throw ConfigError(
        "config.execution_model.type must be 'fixed' or 'analytical', got '" +
        type + "'");
}

ClusterRuntimeConfig parse_cluster_runtime(const Json &clusters,
                                           std::string_view name) {
    const std::string context = "config.clusters." + std::string{name};
    if (!clusters.contains(name)) {
        throw ConfigError("config.clusters is missing required field '" +
                          std::string{name} + "'");
    }
    const Json &cluster = clusters.at(name);
    require_keys(cluster,
                 {
                     "parallelism",
                     "scheduler",
                     "execution_model",
                     "model_name",
                     "moe_routing",
                 },
                 {"total_expert_num", "router_topk"}, context);
    const ModelConfig model = parse_model(cluster, context);
    const ParallelismConfig parallelism = parse_parallelism(cluster, model);
    return [&]() {
        ClusterRuntimeConfig value{};
        value.parallelism = parallelism;
        value.scheduler = parse_scheduler(cluster);
        value.execution_model =
            parse_execution_model(cluster, parallelism, model);
        value.model = model;
        value.moe_routing = parse_moe_routing(cluster);
        return value;
    }();
}

PddClustersConfig parse_pdd_clusters(const Json &root) {
    const Json &clusters = root.at("clusters");
    require_exact_keys(clusters, {"prefill", "decode"}, "config.clusters");
    PddClustersConfig parsed = [&]() {
        PddClustersConfig value{};
        value.prefill = parse_cluster_runtime(clusters, "prefill");
        value.decode = parse_cluster_runtime(clusters, "decode");
        return value;
    }();
    if (parsed.prefill.model != parsed.decode.model) {
        throw ConfigError(
            "PDD PREFILL and DECODE must use the same model and expert "
            "contract");
    }
    return parsed;
}

KvCacheTransferConfig parse_kv_cache_transfer(const Json &root) {
    const Json &transfer = root.at("kv_cache_transfer");
    require_exact_keys(transfer,
                       {
                           "type",
                           "network_bandwidth_gbps",
                           "network_latency_ms",
                           "kv_cache_dtype_size_bytes",
                           "enable_compression",
                       },
                       "config.kv_cache_transfer");
    const std::string type =
        require_string(transfer, "type", "config.kv_cache_transfer");
    if (type != "analytical") {
        throw ConfigError("config.kv_cache_transfer.type must be 'analytical'");
    }
    KvCacheTransferConfig parsed = [&]() {
        KvCacheTransferConfig value{};
        value.network_bandwidth_gbps = require_finite_number(
            transfer, "network_bandwidth_gbps", "config.kv_cache_transfer");
        value.network_latency_ms = require_finite_number(
            transfer, "network_latency_ms", "config.kv_cache_transfer");
        value.kv_cache_dtype_size_bytes = require_finite_number(
            transfer, "kv_cache_dtype_size_bytes", "config.kv_cache_transfer");
        value.enable_compression = require_bool(transfer, "enable_compression",
                                                "config.kv_cache_transfer");
        return value;
    }();
    if (parsed.network_bandwidth_gbps <= 0.0 ||
        parsed.network_latency_ms < 0.0 ||
        parsed.kv_cache_dtype_size_bytes <= 0.0) {
        throw ConfigError(
            "config.kv_cache_transfer requires positive bandwidth/dtype and "
            "nonnegative latency");
    }
    if (parsed.enable_compression) {
        throw ConfigError("config.kv_cache_transfer.enable_compression=true is "
                          "outside Step 3");
    }
    return parsed;
}

struct CommonConfigFields {
    int schema_version;
    std::string run_id;
    SimulationMode simulation_mode;
    SystemArchitecture system_architecture;
    bool enable_parallel_clusters;
    PrefixCacheConfig prefix_cache;
};

void require_schema_version(int schema_version) {
    if (schema_version == kSchemaVersion) {
        return;
    }
    throw ConfigError("unsupported config schema_version=" +
                      std::to_string(schema_version) + "; expected 1");
}

CommonConfigFields parse_common_fields(const Json &root) {
    if (!root.contains("schema_version")) {
        throw ConfigError("config is missing required field 'schema_version'");
    }
    const int schema_version = require_int(root, "schema_version", "config");
    require_schema_version(schema_version);

    std::string run_id = require_string(root, "run_id", "config");
    if (run_id.empty() || is_blank(run_id)) {
        throw ConfigError("config.run_id must not be empty");
    }
    const bool enable_parallel_clusters =
        require_bool(root, "enable_parallel_clusters", "config");
    if (enable_parallel_clusters) {
        throw ConfigError(
            "config.enable_parallel_clusters=true is outside the C++ port");
    }
    return [&]() {
        CommonConfigFields value{};
        value.schema_version = schema_version;
        value.run_id = std::move(run_id);
        value.simulation_mode = parse_simulation_mode(
            require_string(root, "simulation_mode", "config"));
        value.system_architecture = parse_system_architecture(
            require_string(root, "system_architecture", "config"));
        value.enable_parallel_clusters = enable_parallel_clusters;
        value.prefix_cache = parse_prefix_cache(root);
        return value;
    }();
}

SimulationConfig make_pdd_config(const Json &root, CommonConfigFields common) {
    require_exact_keys(root,
                       {
                           "schema_version",
                           "run_id",
                           "simulation_mode",
                           "system_architecture",
                           "enable_parallel_clusters",
                           "prefix_cache",
                           "cluster_scheduler",
                           "clusters",
                           "kv_cache_transfer",
                       },
                       "config");
    if (common.system_architecture != SystemArchitecture::kPdDisaggregation) {
        throw ConfigError(
            "PDD config requires system_architecture='pd-disaggregation'");
    }
    if (common.prefix_cache.enabled) {
        throw ConfigError("PDD config requires prefix_cache.enabled=false; "
                          "session prefix caching begins in Step 4");
    }
    return [&]() {
        SimulationConfig value{};
        value.schema_version = common.schema_version;
        value.run_id = std::move(common.run_id);
        value.simulation_mode = common.simulation_mode;
        value.system_architecture = common.system_architecture;
        value.enable_parallel_clusters = common.enable_parallel_clusters;
        value.prefix_cache = common.prefix_cache;
        value.cluster_scheduler = parse_cluster_scheduler(root);
        value.runtime = [&]() {
            PddRuntimeConfig value{};
            value.clusters = parse_pdd_clusters(root);
            value.kv_cache_transfer = parse_kv_cache_transfer(root);
            return value;
        }();
        return value;
    }();
}

SimulationConfig make_single_cluster_config(const Json &root,
                                            CommonConfigFields common) {
    require_exact_keys(root,
                       {
                           "schema_version",
                           "run_id",
                           "simulation_mode",
                           "system_architecture",
                           "enable_parallel_clusters",
                           "prefix_cache",
                           "cluster_scheduler",
                           "clusters",
                       },
                       "config");
    if (common.system_architecture != SystemArchitecture::kCoLocation) {
        throw ConfigError("single-cluster config requires "
                          "system_architecture='co-location'");
    }
    if (common.prefix_cache.enabled) {
        throw ConfigError(
            "single-cluster config requires prefix_cache.enabled=false; "
            "session prefix caching begins in Step 4");
    }

    const Json &clusters = root.at("clusters");
    require_exact_keys(clusters, {"monolithic"}, "config.clusters");

    return [&]() {
        SimulationConfig value{};
        value.schema_version = common.schema_version;
        value.run_id = std::move(common.run_id);
        value.simulation_mode = common.simulation_mode;
        value.system_architecture = common.system_architecture;
        value.enable_parallel_clusters = common.enable_parallel_clusters;
        value.prefix_cache = common.prefix_cache;
        value.cluster_scheduler = parse_cluster_scheduler(root);
        value.runtime = parse_cluster_runtime(clusters, "monolithic");
        return value;
    }();
}

} // namespace

SimulationConfig parse_simulation_config_json(std::string_view json_text) {
    Json root;
    try {
        root = Json::parse(json_text);
    } catch (const Json::parse_error &error) {
        throw ConfigError("invalid config JSON: " + std::string{error.what()});
    }
    require_object(root, "config");
    CommonConfigFields common = parse_common_fields(root);
    if (common.system_architecture == SystemArchitecture::kPdDisaggregation) {
        return make_pdd_config(root, std::move(common));
    }
    return make_single_cluster_config(root, std::move(common));
}

} // namespace frontier::config
