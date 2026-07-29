#include "frontier/config/config.h"

#include <initializer_list>
#include <string>
#include <unordered_set>

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
  try {
    return value.get<int>();
  } catch (const Json::exception&) {
    throw ConfigError(
        std::string{context} + "." + std::string{field} +
        " is outside the supported integer range");
  }
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
        "config.prefix_cache.key_mode='block_hash' is outside the C++ MVP; "
        "only 'session' is supported");
  }
  throw ConfigError(
      "config.prefix_cache.key_mode must be 'session', got '" +
      std::string{value} + "'");
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

SimulationConfig parse_simulation_config_json(std::string_view json_text) {
  Json root;
  try {
    root = Json::parse(json_text);
  } catch (const Json::parse_error& error) {
    throw ConfigError("invalid config JSON: " + std::string{error.what()});
  }

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

  const int schema_version =
      require_int(root, "schema_version", "config");
  if (schema_version != kSchemaVersion) {
    throw ConfigError(
        "unsupported config schema_version=" +
        std::to_string(schema_version) + "; expected " +
        std::to_string(kSchemaVersion));
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
        "config.enable_parallel_clusters=true is outside the C++ MVP");
  }

  const Json& prefix_cache = root.at("prefix_cache");
  require_exact_keys(
      prefix_cache,
      {"enabled", "key_mode"},
      "config.prefix_cache");
  const PrefixCacheConfig parsed_prefix_cache{
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

  return SimulationConfig{
      .schema_version = schema_version,
      .run_id = std::move(run_id),
      .simulation_mode = simulation_mode,
      .system_architecture = system_architecture,
      .enable_parallel_clusters = enable_parallel_clusters,
      .prefix_cache = parsed_prefix_cache,
  };
}

std::string serialize_simulation_config_json(
    const SimulationConfig& config) {
  if (config.schema_version != kSchemaVersion) {
    throw ConfigError(
        "cannot serialize unsupported config schema_version=" +
        std::to_string(config.schema_version));
  }
  if (config.run_id.empty() || is_blank(config.run_id)) {
    throw ConfigError("config.run_id must not be empty");
  }
  if (config.enable_parallel_clusters) {
    throw ConfigError(
        "config.enable_parallel_clusters=true is outside the C++ MVP");
  }

  OrderedJson root = OrderedJson::object();
  root["schema_version"] = config.schema_version;
  root["run_id"] = config.run_id;
  root["simulation_mode"] = to_string(config.simulation_mode);
  root["system_architecture"] = to_string(config.system_architecture);
  root["enable_parallel_clusters"] = config.enable_parallel_clusters;
  root["prefix_cache"] = OrderedJson::object({
      {"enabled", config.prefix_cache.enabled},
      {"key_mode", to_string(config.prefix_cache.key_mode)},
  });
  return root.dump(2) + '\n';
}

}  // namespace frontier::config
