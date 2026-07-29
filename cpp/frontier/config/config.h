#pragma once

#include <stdexcept>
#include <string>
#include <string_view>

namespace frontier::config {

inline constexpr int kSchemaVersion = 1;

enum class SimulationMode {
  kOffline,
  kOnline,
};

enum class SystemArchitecture {
  kCoLocation,
  kPdDisaggregation,
};

enum class PrefixCachingKeyMode {
  kSession,
};

struct PrefixCacheConfig {
  bool enabled;
  PrefixCachingKeyMode key_mode;

  friend bool operator==(
      const PrefixCacheConfig&,
      const PrefixCacheConfig&) = default;
};

struct SimulationConfig {
  int schema_version;
  std::string run_id;
  SimulationMode simulation_mode;
  SystemArchitecture system_architecture;
  bool enable_parallel_clusters;
  PrefixCacheConfig prefix_cache;

  friend bool operator==(
      const SimulationConfig&,
      const SimulationConfig&) = default;
};

class ConfigError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

[[nodiscard]] std::string_view to_string(SimulationMode mode) noexcept;
[[nodiscard]] std::string_view to_string(
    SystemArchitecture architecture) noexcept;
[[nodiscard]] std::string_view to_string(
    PrefixCachingKeyMode key_mode) noexcept;

[[nodiscard]] SimulationConfig parse_simulation_config_json(
    std::string_view json_text);
[[nodiscard]] std::string serialize_simulation_config_json(
    const SimulationConfig& config);

}  // namespace frontier::config
