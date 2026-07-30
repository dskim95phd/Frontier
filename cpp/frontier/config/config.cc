#include "frontier/config/config.h"

#include <cmath>
#include <initializer_list>
#include <limits>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace frontier::config {
namespace {

using Json = nlohmann::json;
using OrderedJson = nlohmann::ordered_json;

void require_object(const Json& value, std::string_view context) {
  if (!value.is_object()) {
    throw ConfigError(std::string{context} + " must be a JSON object");
  }
}

void require_exact_keys(
    const Json& object,
    std::initializer_list<std::string_view> required_keys,
    std::string_view context) {
  require_object(object, context);

  std::unordered_set<std::string> allowed;
  allowed.reserve(required_keys.size());
  for (const std::string_view key : required_keys) {
    allowed.emplace(key);
    if (!object.contains(key)) {
      throw ConfigError(
          std::string{context} + " is missing required field '" +
          std::string{key} + "'");
    }
  }

  for (const auto& [key, value] : object.items()) {
    static_cast<void>(value);
    if (!allowed.contains(key)) {
      throw ConfigError(
          std::string{context} + " contains unknown field '" + key + "'");
    }
  }
}

std::string require_string(
    const Json& object,
    std::string_view field,
    std::string_view context) {
  const Json& value = object.at(field);
  if (!value.is_string()) {
    throw ConfigError(
        std::string{context} + "." + std::string{field} +
        " must be a string");
  }
  return value.get<std::string>();
}

bool require_bool(
    const Json& object,
    std::string_view field,
    std::string_view context) {
  const Json& value = object.at(field);
  if (!value.is_boolean()) {
    throw ConfigError(
        std::string{context} + "." + std::string{field} +
        " must be a boolean");
  }
  return value.get<bool>();
}

int require_int(
    const Json& object,
    std::string_view field,
    std::string_view context) {
  const Json& value = object.at(field);
  if (!value.is_number_integer()) {
    throw ConfigError(
        std::string{context} + "." + std::string{field} +
        " must be an integer");
  }
  const auto throw_out_of_range = [&context, &field] {
    throw ConfigError(
        std::string{context} + "." + std::string{field} +
        " is outside the supported integer range");
  };

  if (value.is_number_unsigned()) {
    const Json::number_unsigned_t number =
        value.get<Json::number_unsigned_t>();
    if (number >
        static_cast<Json::number_unsigned_t>(
            std::numeric_limits<int>::max())) {
      throw_out_of_range();
    }
    return static_cast<int>(number);
  }

  const Json::number_integer_t number =
      value.get<Json::number_integer_t>();
  if (number <
          static_cast<Json::number_integer_t>(
              std::numeric_limits<int>::min()) ||
      number >
          static_cast<Json::number_integer_t>(
              std::numeric_limits<int>::max())) {
    throw_out_of_range();
  }
  return static_cast<int>(number);
}

std::uint64_t require_uint64(
    const Json& object,
    std::string_view field,
    std::string_view context) {
  const Json& value = object.at(field);
  if (!value.is_number_integer() || value.is_number_float()) {
    throw ConfigError(
        std::string{context} + "." + std::string{field} +
        " must be a nonnegative integer");
  }
  try {
    if (value.is_number_unsigned()) {
      return value.get<std::uint64_t>();
    }
    const std::int64_t parsed = value.get<std::int64_t>();
    if (parsed < 0) {
      throw ConfigError(
          std::string{context} + "." + std::string{field} +
          " must be nonnegative");
    }
    return static_cast<std::uint64_t>(parsed);
  } catch (const ConfigError&) {
    throw;
  } catch (const Json::exception&) {
    throw ConfigError(
        std::string{context} + "." + std::string{field} +
        " is outside the supported integer range");
  }
}

double require_finite_number(
    const Json& object,
    std::string_view field,
    std::string_view context) {
  const Json& value = object.at(field);
  if (!value.is_number()) {
    throw ConfigError(
        std::string{context} + "." + std::string{field} +
        " must be numeric");
  }
  double parsed = 0.0;
  try {
    parsed = value.get<double>();
  } catch (const Json::exception&) {
    throw ConfigError(
        std::string{context} + "." + std::string{field} +
        " is outside the supported numeric range");
  }
  if (!std::isfinite(parsed)) {
    throw ConfigError(
        std::string{context} + "." + std::string{field} +
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
  throw ConfigError(
      "config.system_architecture must be 'co-location' or "
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
  throw ConfigError(
      "config.prefix_cache.key_mode must be 'session', got '" +
      std::string{value} + "'");
}

SchedulerType parse_scheduler_type(std::string_view value) {
  if (value == "vllm_v1") {
    return SchedulerType::kVllmV1;
  }
  throw ConfigError(
      "config.scheduler.type must be 'vllm_v1', got '" +
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

ClusterSchedulerType parse_cluster_scheduler_type(
    std::string_view value) {
  if (value == "round_robin") {
    return ClusterSchedulerType::kRoundRobin;
  }
  throw ConfigError(
      "config.cluster_scheduler.type must be 'round_robin', got '" +
      std::string{value} + "'");
}

PrefixCacheConfig parse_prefix_cache(const Json& root) {
  const Json& prefix_cache = root.at("prefix_cache");
  require_exact_keys(
      prefix_cache,
      {"enabled", "key_mode"},
      "config.prefix_cache");
  return PrefixCacheConfig{
      .enabled = require_bool(
          prefix_cache,
          "enabled",
          "config.prefix_cache"),
      .key_mode = parse_prefix_key_mode(
          require_string(
              prefix_cache,
              "key_mode",
              "config.prefix_cache")),
  };
}

SchedulerConfig parse_scheduler(const Json& root) {
  const Json& scheduler = root.at("scheduler");
  require_exact_keys(
      scheduler,
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

  SchedulerConfig parsed{
      .type = parse_scheduler_type(
          require_string(scheduler, "type", "config.scheduler")),
      .scheduling_policy = parse_scheduling_policy(require_string(
          scheduler,
          "scheduling_policy",
          "config.scheduler")),
      .batch_size_cap =
          require_uint64(scheduler, "batch_size_cap", "config.scheduler"),
      .max_tokens_in_batch = require_uint64(
          scheduler,
          "max_tokens_in_batch",
          "config.scheduler"),
      .enable_preemption =
          require_bool(scheduler, "enable_preemption", "config.scheduler"),
      .enable_chunked_prefill = require_bool(
          scheduler,
          "enable_chunked_prefill",
          "config.scheduler"),
      .long_prefill_token_threshold = require_uint64(
          scheduler,
          "long_prefill_token_threshold",
          "config.scheduler"),
      .block_size =
          require_uint64(scheduler, "block_size", "config.scheduler"),
      .num_blocks =
          require_uint64(scheduler, "num_blocks", "config.scheduler"),
      .watermark_blocks_fraction = require_finite_number(
          scheduler,
          "watermark_blocks_fraction",
          "config.scheduler"),
      .num_preallocate_tokens = require_uint64(
          scheduler,
          "num_preallocate_tokens",
          "config.scheduler"),
  };

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

ParallelismConfig parse_parallelism(const Json& root) {
  const Json& parallelism = root.at("parallelism");
  require_exact_keys(
      parallelism,
      {
          "num_replicas",
          "tensor_parallel_size",
          "pipeline_parallel_size",
          "data_parallel_size",
      },
      "config.parallelism");
  ParallelismConfig parsed{
      .num_replicas = require_uint64(
          parallelism, "num_replicas", "config.parallelism"),
      .tensor_parallel_size = require_uint64(
          parallelism, "tensor_parallel_size", "config.parallelism"),
      .pipeline_parallel_size = require_uint64(
          parallelism, "pipeline_parallel_size", "config.parallelism"),
      .data_parallel_size = require_uint64(
          parallelism, "data_parallel_size", "config.parallelism"),
  };
  if (parsed.num_replicas == 0 ||
      parsed.tensor_parallel_size == 0 ||
      parsed.pipeline_parallel_size == 0 ||
      parsed.data_parallel_size == 0) {
    throw ConfigError(
        "all config.parallelism dimensions must be positive");
  }
  if (parsed.tensor_parallel_size != 1 &&
      parsed.tensor_parallel_size != 2 &&
      parsed.tensor_parallel_size != 4 &&
      parsed.tensor_parallel_size != 8) {
    throw ConfigError(
        "config.parallelism.tensor_parallel_size must be one of 1, 2, 4, 8");
  }
  if (32 % parsed.pipeline_parallel_size != 0) {
    throw ConfigError(
        "Llama-2-7B num_layers=32 must be divisible by "
        "config.parallelism.pipeline_parallel_size");
  }
  if (4'096 % parsed.tensor_parallel_size != 0 ||
      32 % parsed.tensor_parallel_size != 0) {
    throw ConfigError(
        "Llama-2-7B hidden size and attention heads must be divisible by "
        "config.parallelism.tensor_parallel_size");
  }
  const auto checked_multiply = [](std::uint64_t left,
                                   std::uint64_t right,
                                   std::string_view context) {
    if (right != 0 &&
        left > std::numeric_limits<std::uint64_t>::max() / right) {
      throw ConfigError(
          std::string{context} + " overflows uint64");
    }
    return left * right;
  };
  std::uint64_t accelerators = checked_multiply(
      parsed.num_replicas,
      parsed.data_parallel_size,
      "config.parallelism accelerator count");
  accelerators = checked_multiply(
      accelerators,
      parsed.pipeline_parallel_size,
      "config.parallelism accelerator count");
  accelerators = checked_multiply(
      accelerators,
      parsed.tensor_parallel_size,
      "config.parallelism accelerator count");
  if (accelerators > 72) {
    throw ConfigError(
        "config.parallelism requires more than the supported 72 Rubin "
        "accelerators");
  }
  return parsed;
}

ClusterSchedulerConfig parse_cluster_scheduler(const Json& root) {
  const Json& scheduler = root.at("cluster_scheduler");
  require_exact_keys(
      scheduler, {"type"}, "config.cluster_scheduler");
  return ClusterSchedulerConfig{
      .type = parse_cluster_scheduler_type(require_string(
          scheduler, "type", "config.cluster_scheduler")),
  };
}

std::vector<double> require_finite_number_array(
    const Json& object,
    std::string_view field,
    std::string_view context) {
  const Json& value = object.at(field);
  if (!value.is_array()) {
    throw ConfigError(
        std::string{context} + "." + std::string{field} +
        " must be an array");
  }
  std::vector<double> result;
  result.reserve(value.size());
  for (std::size_t index = 0; index < value.size(); ++index) {
    const Json& element = value[index];
    if (!element.is_number()) {
      throw ConfigError(
          std::string{context} + "." + std::string{field} + "[" +
          std::to_string(index) + "] must be numeric");
    }
    double parsed = 0.0;
    try {
      parsed = element.get<double>();
    } catch (const Json::exception&) {
      throw ConfigError(
          std::string{context} + "." + std::string{field} + "[" +
          std::to_string(index) +
          "] is outside the supported numeric range");
    }
    if (!std::isfinite(parsed) || parsed < 0.0) {
      throw ConfigError(
          std::string{context} + "." + std::string{field} + "[" +
          std::to_string(index) +
          "] must be finite and nonnegative");
    }
    result.push_back(parsed);
  }
  return result;
}

ExecutionModelConfig parse_execution_model(
    const Json& root,
    int schema_version,
    const std::optional<ParallelismConfig>& parallelism) {
  const Json& execution = root.at("execution_model");
  require_object(execution, "config.execution_model");
  if (!execution.contains("type")) {
    throw ConfigError(
        "config.execution_model is missing required field 'type'");
  }
  const std::string type =
      require_string(execution, "type", "config.execution_model");

  if (type == "fixed") {
    if (schema_version == kParallelSchemaVersion) {
      require_exact_keys(
          execution,
          {"type", "stage_latencies_ms"},
          "config.execution_model");
      std::vector<double> stage_latencies =
          require_finite_number_array(
              execution,
              "stage_latencies_ms",
              "config.execution_model");
      if (!parallelism.has_value() ||
          stage_latencies.size() !=
              parallelism->pipeline_parallel_size) {
        throw ConfigError(
            "config.execution_model.stage_latencies_ms length must equal "
            "config.parallelism.pipeline_parallel_size");
      }
      return ExecutionModelConfig{
          .type = ExecutionModelType::kFixed,
          .fixed = FixedExecutionModelConfig{
              .batch_latency_ms = stage_latencies.front(),
              .stage_latencies_ms = std::move(stage_latencies),
          },
          .analytical = AnalyticalExecutionModelConfig{},
      };
    }
    require_exact_keys(
        execution,
        {"type", "batch_latency_ms"},
        "config.execution_model");
    const double latency = require_finite_number(
        execution,
        "batch_latency_ms",
        "config.execution_model");
    if (latency < 0.0) {
      throw ConfigError(
          "config.execution_model.batch_latency_ms must be nonnegative");
    }
    return ExecutionModelConfig{
        .type = ExecutionModelType::kFixed,
        .fixed = FixedExecutionModelConfig{
            .batch_latency_ms = latency,
            .stage_latencies_ms = {},
        },
        .analytical = AnalyticalExecutionModelConfig{},
    };
  }

  if (type == "analytical") {
    if (schema_version == kParallelSchemaVersion) {
      require_exact_keys(
          execution,
          {
              "type",
              "device",
              "model",
              "precision",
              "num_layers",
              "network_bandwidth_gbps",
              "network_latency_us",
              "intra_node_bandwidth_gbps",
          },
          "config.execution_model");
    } else {
      require_exact_keys(
          execution,
          {
              "type",
              "device",
              "model",
              "precision",
              "tensor_parallel_size",
              "num_layers",
              "network_bandwidth_gbps",
              "network_latency_us",
              "intra_node_bandwidth_gbps",
          },
          "config.execution_model");
    }
    AnalyticalExecutionModelConfig analytical{
        .device =
            require_string(execution, "device", "config.execution_model"),
        .model =
            require_string(execution, "model", "config.execution_model"),
        .precision =
            require_string(execution, "precision", "config.execution_model"),
        .tensor_parallel_size =
            schema_version == kParallelSchemaVersion
                ? parallelism->tensor_parallel_size
                : require_uint64(
                      execution,
                      "tensor_parallel_size",
                      "config.execution_model"),
        .num_layers = require_uint64(
            execution,
            "num_layers",
            "config.execution_model"),
        .network_bandwidth_gbps = require_finite_number(
            execution,
            "network_bandwidth_gbps",
            "config.execution_model"),
        .network_latency_us = require_finite_number(
            execution,
            "network_latency_us",
            "config.execution_model"),
        .intra_node_bandwidth_gbps = require_finite_number(
            execution,
            "intra_node_bandwidth_gbps",
            "config.execution_model"),
    };
    if (analytical.device != "rubin" ||
        analytical.model != "llama2-7b" ||
        analytical.precision != "fp16" ||
        analytical.num_layers != 32) {
      throw ConfigError(
          "analytical execution supports only "
          "rubin/llama2-7b/fp16/32-layers");
    }
    if (schema_version == kSchedulerSchemaVersion &&
        analytical.tensor_parallel_size != 8) {
      throw ConfigError(
          "schema v2 analytical execution supports only TP8");
    }
    if (analytical.network_bandwidth_gbps <= 0.0 ||
        analytical.intra_node_bandwidth_gbps <= 0.0 ||
        analytical.network_latency_us < 0.0) {
      throw ConfigError(
          "analytical bandwidths must be positive and latency must be "
          "nonnegative");
    }
    return ExecutionModelConfig{
        .type = ExecutionModelType::kAnalytical,
        .fixed = FixedExecutionModelConfig{},
        .analytical = std::move(analytical),
    };
  }

  throw ConfigError(
      "config.execution_model.type must be 'fixed' or 'analytical', got '" +
      type + "'");
}

ClusterRuntimeConfig parse_cluster_runtime(
    const Json& clusters,
    std::string_view name) {
  const std::string context =
      "config.clusters." + std::string{name};
  if (!clusters.contains(name)) {
    throw ConfigError(
        "config.clusters is missing required field '" +
        std::string{name} + "'");
  }
  const Json& cluster = clusters.at(name);
  require_exact_keys(
      cluster,
      {"parallelism", "scheduler", "execution_model"},
      context);
  const ParallelismConfig parallelism =
      parse_parallelism(cluster);
  return ClusterRuntimeConfig{
      .parallelism = parallelism,
      .scheduler = parse_scheduler(cluster),
      .execution_model = parse_execution_model(
          cluster,
          kParallelSchemaVersion,
          parallelism),
  };
}

PddClustersConfig parse_pdd_clusters(const Json& root) {
  const Json& clusters = root.at("clusters");
  require_exact_keys(
      clusters,
      {"prefill", "decode"},
      "config.clusters");
  return PddClustersConfig{
      .prefill = parse_cluster_runtime(clusters, "prefill"),
      .decode = parse_cluster_runtime(clusters, "decode"),
  };
}

KvCacheTransferConfig parse_kv_cache_transfer(const Json& root) {
  const Json& transfer = root.at("kv_cache_transfer");
  require_exact_keys(
      transfer,
      {
          "type",
          "network_bandwidth_gbps",
          "network_latency_ms",
          "kv_cache_dtype_size_bytes",
          "enable_compression",
      },
      "config.kv_cache_transfer");
  const std::string type = require_string(
      transfer, "type", "config.kv_cache_transfer");
  if (type != "analytical") {
    throw ConfigError(
        "config.kv_cache_transfer.type must be 'analytical'");
  }
  KvCacheTransferConfig parsed{
      .network_bandwidth_gbps = require_finite_number(
          transfer,
          "network_bandwidth_gbps",
          "config.kv_cache_transfer"),
      .network_latency_ms = require_finite_number(
          transfer,
          "network_latency_ms",
          "config.kv_cache_transfer"),
      .kv_cache_dtype_size_bytes = require_finite_number(
          transfer,
          "kv_cache_dtype_size_bytes",
          "config.kv_cache_transfer"),
      .enable_compression = require_bool(
          transfer,
          "enable_compression",
          "config.kv_cache_transfer"),
  };
  if (parsed.network_bandwidth_gbps <= 0.0 ||
      parsed.network_latency_ms < 0.0 ||
      parsed.kv_cache_dtype_size_bytes <= 0.0) {
    throw ConfigError(
        "config.kv_cache_transfer requires positive bandwidth/dtype and "
        "nonnegative latency");
  }
  if (parsed.enable_compression) {
    throw ConfigError(
        "config.kv_cache_transfer.enable_compression=true is outside Step 3");
  }
  return parsed;
}

void validate_common_for_serialization(const SimulationConfig& config) {
  if (config.schema_version != kFoundationSchemaVersion &&
      config.schema_version != kSchedulerSchemaVersion &&
      config.schema_version != kParallelSchemaVersion &&
      config.schema_version != kPddSchemaVersion) {
    throw ConfigError(
        "cannot serialize unsupported config schema_version=" +
        std::to_string(config.schema_version));
  }
  if (config.run_id.empty() || is_blank(config.run_id)) {
    throw ConfigError("config.run_id must not be empty");
  }
  if (config.enable_parallel_clusters) {
    throw ConfigError(
        "config.enable_parallel_clusters=true is outside the C++ port");
  }
}

OrderedJson serialize_prefix_cache(const PrefixCacheConfig& config) {
  return OrderedJson::object({
      {"enabled", config.enabled},
      {"key_mode", to_string(config.key_mode)},
  });
}

OrderedJson serialize_scheduler(const SchedulerConfig& scheduler) {
  return OrderedJson::object({
      {"type", to_string(scheduler.type)},
      {"scheduling_policy", to_string(scheduler.scheduling_policy)},
      {"batch_size_cap", scheduler.batch_size_cap},
      {"max_tokens_in_batch", scheduler.max_tokens_in_batch},
      {"enable_preemption", scheduler.enable_preemption},
      {"enable_chunked_prefill", scheduler.enable_chunked_prefill},
      {
          "long_prefill_token_threshold",
          scheduler.long_prefill_token_threshold,
      },
      {"block_size", scheduler.block_size},
      {"num_blocks", scheduler.num_blocks},
      {
          "watermark_blocks_fraction",
          scheduler.watermark_blocks_fraction,
      },
      {"num_preallocate_tokens", scheduler.num_preallocate_tokens},
  });
}

OrderedJson serialize_parallelism(
    const ParallelismConfig& parallelism) {
  return OrderedJson::object({
      {"num_replicas", parallelism.num_replicas},
      {"tensor_parallel_size", parallelism.tensor_parallel_size},
      {"pipeline_parallel_size", parallelism.pipeline_parallel_size},
      {"data_parallel_size", parallelism.data_parallel_size},
  });
}

OrderedJson serialize_cluster_scheduler(
    const ClusterSchedulerConfig& scheduler) {
  return OrderedJson::object({
      {"type", to_string(scheduler.type)},
  });
}

OrderedJson serialize_execution_model(
    const ExecutionModelConfig& execution,
    int schema_version) {
  if (execution.type == ExecutionModelType::kFixed) {
    if (schema_version == kParallelSchemaVersion) {
      return OrderedJson::object({
          {"type", to_string(execution.type)},
          {"stage_latencies_ms", execution.fixed.stage_latencies_ms},
      });
    }
    return OrderedJson::object({
        {"type", to_string(execution.type)},
        {"batch_latency_ms", execution.fixed.batch_latency_ms},
    });
  }
  OrderedJson result = OrderedJson::object({
      {"type", to_string(execution.type)},
      {"device", execution.analytical.device},
      {"model", execution.analytical.model},
      {"precision", execution.analytical.precision},
      {"num_layers", execution.analytical.num_layers},
      {
          "network_bandwidth_gbps",
          execution.analytical.network_bandwidth_gbps,
      },
      {"network_latency_us", execution.analytical.network_latency_us},
      {
          "intra_node_bandwidth_gbps",
          execution.analytical.intra_node_bandwidth_gbps,
      },
  });
  if (schema_version == kSchedulerSchemaVersion) {
    result["tensor_parallel_size"] =
        execution.analytical.tensor_parallel_size;
    OrderedJson ordered = OrderedJson::object();
    ordered["type"] = result["type"];
    ordered["device"] = result["device"];
    ordered["model"] = result["model"];
    ordered["precision"] = result["precision"];
    ordered["tensor_parallel_size"] =
        result["tensor_parallel_size"];
    ordered["num_layers"] = result["num_layers"];
    ordered["network_bandwidth_gbps"] =
        result["network_bandwidth_gbps"];
    ordered["network_latency_us"] = result["network_latency_us"];
    ordered["intra_node_bandwidth_gbps"] =
        result["intra_node_bandwidth_gbps"];
    return ordered;
  }
  return result;
}

OrderedJson serialize_cluster_runtime(
    const ClusterRuntimeConfig& cluster) {
  return OrderedJson::object({
      {"parallelism", serialize_parallelism(cluster.parallelism)},
      {"scheduler", serialize_scheduler(cluster.scheduler)},
      {
          "execution_model",
          serialize_execution_model(
              cluster.execution_model,
              kParallelSchemaVersion),
      },
  });
}

OrderedJson serialize_pdd_clusters(
    const PddClustersConfig& clusters) {
  return OrderedJson::object({
      {"prefill", serialize_cluster_runtime(clusters.prefill)},
      {"decode", serialize_cluster_runtime(clusters.decode)},
  });
}

OrderedJson serialize_kv_cache_transfer(
    const KvCacheTransferConfig& transfer) {
  return OrderedJson::object({
      {"type", "analytical"},
      {"network_bandwidth_gbps", transfer.network_bandwidth_gbps},
      {"network_latency_ms", transfer.network_latency_ms},
      {
          "kv_cache_dtype_size_bytes",
          transfer.kv_cache_dtype_size_bytes,
      },
      {"enable_compression", transfer.enable_compression},
  });
}

}  // namespace

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

SimulationConfig parse_simulation_config_json(std::string_view json_text) {
  Json root;
  try {
    root = Json::parse(json_text);
  } catch (const Json::parse_error& error) {
    throw ConfigError("invalid config JSON: " + std::string{error.what()});
  }
  require_object(root, "config");
  if (!root.contains("schema_version")) {
    throw ConfigError("config is missing required field 'schema_version'");
  }
  const int schema_version =
      require_int(root, "schema_version", "config");

  if (schema_version == kFoundationSchemaVersion) {
    require_exact_keys(
        root,
        {
            "schema_version",
            "run_id",
            "simulation_mode",
            "system_architecture",
            "enable_parallel_clusters",
            "prefix_cache",
        },
        "config");
  } else if (schema_version == kSchedulerSchemaVersion) {
    require_exact_keys(
        root,
        {
            "schema_version",
            "run_id",
            "simulation_mode",
            "system_architecture",
            "enable_parallel_clusters",
            "prefix_cache",
            "scheduler",
            "execution_model",
        },
        "config");
  } else if (schema_version == kParallelSchemaVersion) {
    require_exact_keys(
        root,
        {
            "schema_version",
            "run_id",
            "simulation_mode",
            "system_architecture",
            "enable_parallel_clusters",
            "parallelism",
            "cluster_scheduler",
            "prefix_cache",
            "scheduler",
            "execution_model",
        },
        "config");
  } else if (schema_version == kPddSchemaVersion) {
    require_exact_keys(
        root,
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
  } else {
    throw ConfigError(
        "unsupported config schema_version=" +
        std::to_string(schema_version) + "; expected 1, 2, 3, or 4");
  }

  std::string run_id = require_string(root, "run_id", "config");
  if (run_id.empty() || is_blank(run_id)) {
    throw ConfigError("config.run_id must not be empty");
  }
  const SimulationMode simulation_mode = parse_simulation_mode(
      require_string(root, "simulation_mode", "config"));
  const SystemArchitecture system_architecture = parse_system_architecture(
      require_string(root, "system_architecture", "config"));
  const bool enable_parallel_clusters =
      require_bool(root, "enable_parallel_clusters", "config");
  if (enable_parallel_clusters) {
    throw ConfigError(
        "config.enable_parallel_clusters=true is outside the C++ port");
  }
  const PrefixCacheConfig prefix_cache = parse_prefix_cache(root);

  if (schema_version == kFoundationSchemaVersion) {
    return SimulationConfig{
        .schema_version = schema_version,
        .run_id = std::move(run_id),
        .simulation_mode = simulation_mode,
        .system_architecture = system_architecture,
        .enable_parallel_clusters = enable_parallel_clusters,
        .prefix_cache = prefix_cache,
        .parallelism = std::nullopt,
        .cluster_scheduler = std::nullopt,
        .scheduler = std::nullopt,
        .execution_model = std::nullopt,
        .clusters = std::nullopt,
        .kv_cache_transfer = std::nullopt,
    };
  }

  if (schema_version == kPddSchemaVersion) {
    if (system_architecture !=
        SystemArchitecture::kPdDisaggregation) {
      throw ConfigError(
          "schema v4 requires "
          "system_architecture='pd-disaggregation'");
    }
    if (prefix_cache.enabled) {
      throw ConfigError(
          "schema v4 requires prefix_cache.enabled=false; "
          "session prefix caching begins in Step 4");
    }
    return SimulationConfig{
        .schema_version = schema_version,
        .run_id = std::move(run_id),
        .simulation_mode = simulation_mode,
        .system_architecture = system_architecture,
        .enable_parallel_clusters = enable_parallel_clusters,
        .prefix_cache = prefix_cache,
        .parallelism = std::nullopt,
        .cluster_scheduler = parse_cluster_scheduler(root),
        .scheduler = std::nullopt,
        .execution_model = std::nullopt,
        .clusters = parse_pdd_clusters(root),
        .kv_cache_transfer = parse_kv_cache_transfer(root),
    };
  }

  if (system_architecture != SystemArchitecture::kCoLocation) {
    throw ConfigError(
        "scheduler schemas support only system_architecture='co-location'; "
        "sequential PDD begins in Step 3");
  }
  if (prefix_cache.enabled) {
    throw ConfigError(
        "scheduler schemas require prefix_cache.enabled=false; "
        "session prefix caching begins in Step 4");
  }

  std::optional<ParallelismConfig> parallelism;
  std::optional<ClusterSchedulerConfig> cluster_scheduler;
  if (schema_version == kParallelSchemaVersion) {
    parallelism = parse_parallelism(root);
    cluster_scheduler = parse_cluster_scheduler(root);
  }

  return SimulationConfig{
      .schema_version = schema_version,
      .run_id = std::move(run_id),
      .simulation_mode = simulation_mode,
      .system_architecture = system_architecture,
      .enable_parallel_clusters = enable_parallel_clusters,
      .prefix_cache = prefix_cache,
      .parallelism = parallelism,
      .cluster_scheduler = cluster_scheduler,
      .scheduler = parse_scheduler(root),
      .execution_model = parse_execution_model(
          root, schema_version, parallelism),
      .clusters = std::nullopt,
      .kv_cache_transfer = std::nullopt,
  };
}

std::string serialize_simulation_config_json(
    const SimulationConfig& config) {
  validate_common_for_serialization(config);

  OrderedJson root = OrderedJson::object();
  root["schema_version"] = config.schema_version;
  root["run_id"] = config.run_id;
  root["simulation_mode"] = to_string(config.simulation_mode);
  root["system_architecture"] = to_string(config.system_architecture);
  root["enable_parallel_clusters"] = config.enable_parallel_clusters;
  if (config.schema_version == kPddSchemaVersion) {
    if (config.system_architecture !=
            SystemArchitecture::kPdDisaggregation ||
        config.prefix_cache.enabled ||
        !config.cluster_scheduler.has_value() ||
        !config.clusters.has_value() ||
        !config.kv_cache_transfer.has_value() ||
        config.parallelism.has_value() ||
        config.scheduler.has_value() ||
        config.execution_model.has_value()) {
      throw ConfigError(
          "schema v4 requires sequential PDD cluster and transfer configs");
    }
    root["prefix_cache"] =
        serialize_prefix_cache(config.prefix_cache);
    root["cluster_scheduler"] =
        serialize_cluster_scheduler(config.cluster_scheduler.value());
    root["clusters"] =
        serialize_pdd_clusters(config.clusters.value());
    root["kv_cache_transfer"] =
        serialize_kv_cache_transfer(config.kv_cache_transfer.value());
    return root.dump(2) + '\n';
  }
  if (config.schema_version == kParallelSchemaVersion) {
    if (!config.parallelism.has_value() ||
        !config.cluster_scheduler.has_value()) {
      throw ConfigError(
          "schema v3 requires parallelism and cluster_scheduler");
    }
    root["parallelism"] =
        serialize_parallelism(config.parallelism.value());
    root["cluster_scheduler"] =
        serialize_cluster_scheduler(config.cluster_scheduler.value());
  }
  root["prefix_cache"] = serialize_prefix_cache(config.prefix_cache);

  if (config.schema_version == kFoundationSchemaVersion) {
    if (config.parallelism.has_value() ||
        config.cluster_scheduler.has_value() ||
        config.scheduler.has_value() ||
        config.execution_model.has_value() ||
        config.clusters.has_value() ||
        config.kv_cache_transfer.has_value()) {
      throw ConfigError(
          "schema v1 must not contain parallel or scheduler fields");
    }
  } else {
    if (!config.scheduler.has_value() ||
        !config.execution_model.has_value() ||
        config.clusters.has_value() ||
        config.kv_cache_transfer.has_value()) {
      throw ConfigError(
          "scheduler schemas require scheduler and execution_model");
    }
    if (config.system_architecture != SystemArchitecture::kCoLocation ||
        config.prefix_cache.enabled) {
      throw ConfigError(
          "scheduler schemas require co-location with prefix caching disabled");
    }
    if (config.schema_version == kSchedulerSchemaVersion &&
        (config.parallelism.has_value() ||
         config.cluster_scheduler.has_value())) {
      throw ConfigError(
          "schema v2 must not contain parallelism or cluster_scheduler");
    }
    root["scheduler"] = serialize_scheduler(config.scheduler.value());
    root["execution_model"] =
        serialize_execution_model(
            config.execution_model.value(), config.schema_version);
  }
  return root.dump(2) + '\n';
}

}  // namespace frontier::config
