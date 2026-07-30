#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

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

enum class SchedulerType {
  kVllmV1,
};

enum class SchedulingPolicy {
  kFcfs,
};

enum class ClusterSchedulerType {
  kRoundRobin,
};

struct ParallelismConfig {
  std::uint64_t num_replicas = 1;
  std::uint64_t tensor_parallel_size = 1;
  std::uint64_t pipeline_parallel_size = 1;
  std::uint64_t data_parallel_size = 1;

  friend bool operator==(
      const ParallelismConfig&,
      const ParallelismConfig&) = default;
};

struct ClusterSchedulerConfig {
  ClusterSchedulerType type = ClusterSchedulerType::kRoundRobin;

  friend bool operator==(
      const ClusterSchedulerConfig&,
      const ClusterSchedulerConfig&) = default;
};

struct SchedulerConfig {
  SchedulerType type = SchedulerType::kVllmV1;
  SchedulingPolicy scheduling_policy = SchedulingPolicy::kFcfs;
  std::uint64_t batch_size_cap = 1;
  std::uint64_t max_tokens_in_batch = 1;
  bool enable_preemption = false;
  bool enable_chunked_prefill = false;
  std::uint64_t long_prefill_token_threshold = 0;
  std::uint64_t block_size = 16;
  std::uint64_t num_blocks = 1;
  double watermark_blocks_fraction = 0.0;
  std::uint64_t num_preallocate_tokens = 0;

  friend bool operator==(
      const SchedulerConfig&,
      const SchedulerConfig&) = default;
};

enum class ExecutionModelType {
  kFixed,
  kAnalytical,
};

struct FixedExecutionModelConfig {
  double batch_latency_ms = 1.0;
  std::vector<double> stage_latencies_ms;

  friend bool operator==(
      const FixedExecutionModelConfig&,
      const FixedExecutionModelConfig&) = default;
};

struct AnalyticalExecutionModelConfig {
  std::string device = "rubin";
  std::string model = "llama2-7b";
  std::string precision = "fp16";
  std::uint64_t tensor_parallel_size = 8;
  std::uint64_t num_layers = 32;
  double network_bandwidth_gbps = 400.0;
  double network_latency_us = 1.0;
  double intra_node_bandwidth_gbps = 14'400.0;

  friend bool operator==(
      const AnalyticalExecutionModelConfig&,
      const AnalyticalExecutionModelConfig&) = default;
};

struct ExecutionModelConfig {
  ExecutionModelType type = ExecutionModelType::kFixed;
  FixedExecutionModelConfig fixed;
  AnalyticalExecutionModelConfig analytical;

  friend bool operator==(
      const ExecutionModelConfig&,
      const ExecutionModelConfig&) = default;
};

struct ClusterRuntimeConfig {
  ParallelismConfig parallelism;
  SchedulerConfig scheduler;
  ExecutionModelConfig execution_model;

  friend bool operator==(
      const ClusterRuntimeConfig&,
      const ClusterRuntimeConfig&) = default;
};

struct PddClustersConfig {
  ClusterRuntimeConfig prefill;
  ClusterRuntimeConfig decode;

  friend bool operator==(
      const PddClustersConfig&,
      const PddClustersConfig&) = default;
};

struct KvCacheTransferConfig {
  double network_bandwidth_gbps = 100.0;
  double network_latency_ms = 0.1;
  double kv_cache_dtype_size_bytes = 2.0;
  bool enable_compression = false;

  friend bool operator==(
      const KvCacheTransferConfig&,
      const KvCacheTransferConfig&) = default;
};

struct PddRuntimeConfig {
  PddClustersConfig clusters;
  KvCacheTransferConfig kv_cache_transfer;

  friend bool operator==(
      const PddRuntimeConfig&,
      const PddRuntimeConfig&) = default;
};

using RuntimeConfig = std::variant<
    ClusterRuntimeConfig,
    PddRuntimeConfig>;

class ConfigError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

struct SimulationConfig {
  int schema_version;
  std::string run_id;
  SimulationMode simulation_mode;
  SystemArchitecture system_architecture;
  bool enable_parallel_clusters;
  PrefixCacheConfig prefix_cache;
  ClusterSchedulerConfig cluster_scheduler;
  RuntimeConfig runtime;

  [[nodiscard]] ClusterRuntimeConfig& cluster();
  [[nodiscard]] const ClusterRuntimeConfig& cluster() const;
  [[nodiscard]] PddRuntimeConfig& pdd();
  [[nodiscard]] const PddRuntimeConfig& pdd() const;

  friend bool operator==(
      const SimulationConfig&,
      const SimulationConfig&) = default;
};

[[nodiscard]] std::string_view to_string(SimulationMode mode) noexcept;
[[nodiscard]] std::string_view to_string(
    SystemArchitecture architecture) noexcept;
[[nodiscard]] std::string_view to_string(
    PrefixCachingKeyMode key_mode) noexcept;
[[nodiscard]] std::string_view to_string(SchedulerType type) noexcept;
[[nodiscard]] std::string_view to_string(
    SchedulingPolicy policy) noexcept;
[[nodiscard]] std::string_view to_string(
    ClusterSchedulerType type) noexcept;
[[nodiscard]] std::string_view to_string(
    ExecutionModelType type) noexcept;

[[nodiscard]] SimulationConfig parse_simulation_config_json(
    std::string_view json_text);
[[nodiscard]] std::string serialize_simulation_config_json(
    const SimulationConfig& config);

}  // namespace frontier::config
