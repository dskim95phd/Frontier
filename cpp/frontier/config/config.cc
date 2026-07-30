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

ClusterRuntimeConfig& SimulationConfig::cluster() {
  auto* value = std::get_if<ClusterRuntimeConfig>(&runtime);
  if (value == nullptr) {
    throw ConfigError("simulation config has no single cluster runtime");
  }
  return *value;
}

const ClusterRuntimeConfig& SimulationConfig::cluster() const {
  const auto* value =
      std::get_if<ClusterRuntimeConfig>(&runtime);
  if (value == nullptr) {
    throw ConfigError("simulation config has no single cluster runtime");
  }
  return *value;
}

PddRuntimeConfig& SimulationConfig::pdd() {
  auto* value = std::get_if<PddRuntimeConfig>(&runtime);
  if (value == nullptr) {
    throw ConfigError("simulation config is not a PDD config");
  }
  return *value;
}

const PddRuntimeConfig& SimulationConfig::pdd() const {
  const auto* value = std::get_if<PddRuntimeConfig>(&runtime);
  if (value == nullptr) {
    throw ConfigError("simulation config is not a PDD config");
  }
  return *value;
}

}  // namespace frontier::config
