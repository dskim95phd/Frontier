#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <variant>
#include <vector>

#include "frontier/attention/ops.h"

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

    friend bool operator==(const PrefixCacheConfig &lhs,
                           const PrefixCacheConfig &rhs) {
        return std::tie(lhs.enabled, lhs.key_mode) ==
               std::tie(rhs.enabled, rhs.key_mode);
    }
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

enum class ModelKind {
    kDense,
    kMoe,
};

struct ModelConfig {
    std::string name = "meta-llama/Llama-2-7b-hf";
    std::string model_type = "llama";
    ModelKind kind = ModelKind::kDense;
    std::uint64_t num_layers = 32;
    std::uint64_t hidden_size = 4'096;
    std::uint64_t intermediate_size = 11'008;
    std::uint64_t num_query_heads = 32;
    std::uint64_t num_kv_heads = 32;
    std::uint64_t head_dim = 128;
    bool gated_mlp = true;
    bool fused_add_norm = true;
    std::uint64_t num_experts = 1;
    std::uint64_t num_experts_per_token = 1;
    std::uint64_t total_expert_num = 1;
    std::uint64_t router_topk = 1;
    bool use_mla = false;
    bool use_mfa = false;
    std::uint64_t q_lora_rank = 0;
    std::uint64_t kv_lora_rank = 0;
    std::uint64_t qk_nope_head_dim = 0;
    std::uint64_t qk_rope_head_dim = 0;
    std::uint64_t qk_head_dim = 0;
    std::uint64_t v_head_dim = 0;
    std::uint64_t share_q_dim = 0;
    bool has_dsa_marker = false;
    std::vector<std::string> exotic_attention_fields;
    attention::AttentionFamilyBinding attention;

    [[nodiscard]] bool is_moe() const noexcept {
        return kind == ModelKind::kMoe;
    }

    [[nodiscard]] std::uint64_t runtime_num_kv_heads() const noexcept {
        return attention.memory_layout ==
                       attention::AttentionMemoryLayout::kLatentMla
                   ? 1
                   : num_kv_heads;
    }

    [[nodiscard]] std::uint64_t runtime_head_size() const noexcept {
        return attention.memory_layout ==
                       attention::AttentionMemoryLayout::kLatentMla
                   ? kv_lora_rank + qk_rope_head_dim
                   : head_dim;
    }

    [[nodiscard]] std::uint64_t kv_factor() const noexcept {
        return attention.memory_layout ==
                       attention::AttentionMemoryLayout::kLatentMla
                   ? 1
                   : 2;
    }

    friend bool operator==(const ModelConfig &lhs, const ModelConfig &rhs) {
        return std::tie(lhs.name, lhs.model_type, lhs.kind, lhs.num_layers,
                        lhs.hidden_size,
                        lhs.intermediate_size, lhs.num_query_heads,
                        lhs.num_kv_heads, lhs.head_dim, lhs.gated_mlp,
                        lhs.fused_add_norm, lhs.num_experts,
                        lhs.num_experts_per_token, lhs.total_expert_num,
                        lhs.router_topk, lhs.use_mla, lhs.use_mfa,
                        lhs.q_lora_rank, lhs.kv_lora_rank,
                        lhs.qk_nope_head_dim, lhs.qk_rope_head_dim,
                        lhs.qk_head_dim, lhs.v_head_dim, lhs.share_q_dim,
                        lhs.has_dsa_marker, lhs.exotic_attention_fields,
                        lhs.attention) ==
               std::tie(rhs.name, rhs.model_type, rhs.kind, rhs.num_layers,
                        rhs.hidden_size,
                        rhs.intermediate_size, rhs.num_query_heads,
                        rhs.num_kv_heads, rhs.head_dim, rhs.gated_mlp,
                        rhs.fused_add_norm, rhs.num_experts,
                        rhs.num_experts_per_token, rhs.total_expert_num,
                        rhs.router_topk, rhs.use_mla, rhs.use_mfa,
                        rhs.q_lora_rank, rhs.kv_lora_rank,
                        rhs.qk_nope_head_dim, rhs.qk_rope_head_dim,
                        rhs.qk_head_dim, rhs.v_head_dim, rhs.share_q_dim,
                        rhs.has_dsa_marker, rhs.exotic_attention_fields,
                        rhs.attention);
    }
    friend bool operator!=(const ModelConfig &lhs, const ModelConfig &rhs) {
        return !(lhs == rhs);
    }
};

enum class MoeRoutingMode {
    kSimulation,
    kUniformLegacy,
    kUniformRandom,
};

enum class MoeRoutingDistribution {
    kBalanced,
    kRandom,
    kSkewed,
    kZipf,
};

struct MoeRoutingConfig {
    MoeRoutingMode mode = MoeRoutingMode::kSimulation;
    MoeRoutingDistribution distribution = MoeRoutingDistribution::kBalanced;
    std::uint64_t seed = 42;

    friend bool operator==(const MoeRoutingConfig &lhs,
                           const MoeRoutingConfig &rhs) {
        return std::tie(lhs.mode, lhs.distribution, lhs.seed) ==
               std::tie(rhs.mode, rhs.distribution, rhs.seed);
    }
};

struct ParallelismConfig {
    std::uint64_t num_replicas = 1;
    std::uint64_t tensor_parallel_size = 1;
    std::uint64_t pipeline_parallel_size = 1;
    std::uint64_t data_parallel_size = 1;
    std::uint64_t moe_tensor_parallel_size = 1;
    std::uint64_t moe_expert_parallel_size = 1;

    [[nodiscard]] std::uint64_t attention_parallel_size() const noexcept {
        return tensor_parallel_size * data_parallel_size;
    }
    [[nodiscard]] std::uint64_t moe_parallel_size() const noexcept {
        return moe_tensor_parallel_size * moe_expert_parallel_size;
    }

    friend bool operator==(const ParallelismConfig &lhs,
                           const ParallelismConfig &rhs) {
        return std::tie(lhs.num_replicas, lhs.tensor_parallel_size,
                        lhs.pipeline_parallel_size, lhs.data_parallel_size,
                        lhs.moe_tensor_parallel_size,
                        lhs.moe_expert_parallel_size) ==
               std::tie(rhs.num_replicas, rhs.tensor_parallel_size,
                        rhs.pipeline_parallel_size, rhs.data_parallel_size,
                        rhs.moe_tensor_parallel_size,
                        rhs.moe_expert_parallel_size);
    }
};

struct ClusterSchedulerConfig {
    ClusterSchedulerType type = ClusterSchedulerType::kRoundRobin;

    friend bool operator==(const ClusterSchedulerConfig &lhs,
                           const ClusterSchedulerConfig &rhs) {
        return lhs.type == rhs.type;
    }
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

    friend bool operator==(const SchedulerConfig &lhs,
                           const SchedulerConfig &rhs) {
        return std::tie(lhs.type, lhs.scheduling_policy, lhs.batch_size_cap,
                        lhs.max_tokens_in_batch, lhs.enable_preemption,
                        lhs.enable_chunked_prefill,
                        lhs.long_prefill_token_threshold, lhs.block_size,
                        lhs.num_blocks, lhs.watermark_blocks_fraction,
                        lhs.num_preallocate_tokens) ==
               std::tie(rhs.type, rhs.scheduling_policy, rhs.batch_size_cap,
                        rhs.max_tokens_in_batch, rhs.enable_preemption,
                        rhs.enable_chunked_prefill,
                        rhs.long_prefill_token_threshold, rhs.block_size,
                        rhs.num_blocks, rhs.watermark_blocks_fraction,
                        rhs.num_preallocate_tokens);
    }
};

enum class ExecutionModelType {
    kFixed,
    kAnalytical,
};

struct FixedExecutionModelConfig {
    double batch_latency_ms = 1.0;
    std::vector<double> stage_latencies_ms;

    friend bool operator==(const FixedExecutionModelConfig &lhs,
                           const FixedExecutionModelConfig &rhs) {
        return std::tie(lhs.batch_latency_ms, lhs.stage_latencies_ms) ==
               std::tie(rhs.batch_latency_ms, rhs.stage_latencies_ms);
    }
};

struct AnalyticalExecutionModelConfig {
    std::string device = "rubin";
    std::string precision = "fp16";
    std::uint64_t tensor_parallel_size = 8;
    double network_bandwidth_gbps = 400.0;
    double network_latency_us = 1.0;
    double intra_node_bandwidth_gbps = 14'400.0;

    friend bool operator==(const AnalyticalExecutionModelConfig &lhs,
                           const AnalyticalExecutionModelConfig &rhs) {
        return std::tie(lhs.device, lhs.precision, lhs.tensor_parallel_size,
                        lhs.network_bandwidth_gbps, lhs.network_latency_us,
                        lhs.intra_node_bandwidth_gbps) ==
               std::tie(rhs.device, rhs.precision, rhs.tensor_parallel_size,
                        rhs.network_bandwidth_gbps, rhs.network_latency_us,
                        rhs.intra_node_bandwidth_gbps);
    }
};

struct ExecutionModelConfig {
    ExecutionModelType type = ExecutionModelType::kFixed;
    FixedExecutionModelConfig fixed;
    AnalyticalExecutionModelConfig analytical;

    friend bool operator==(const ExecutionModelConfig &lhs,
                           const ExecutionModelConfig &rhs) {
        return std::tie(lhs.type, lhs.fixed, lhs.analytical) ==
               std::tie(rhs.type, rhs.fixed, rhs.analytical);
    }
};

struct ClusterRuntimeConfig {
    ParallelismConfig parallelism;
    SchedulerConfig scheduler;
    ExecutionModelConfig execution_model;
    ModelConfig model;
    MoeRoutingConfig moe_routing;

    friend bool operator==(const ClusterRuntimeConfig &lhs,
                           const ClusterRuntimeConfig &rhs) {
        return std::tie(lhs.parallelism, lhs.scheduler, lhs.execution_model,
                        lhs.model, lhs.moe_routing) ==
               std::tie(rhs.parallelism, rhs.scheduler, rhs.execution_model,
                        rhs.model, rhs.moe_routing);
    }
};

struct PddClustersConfig {
    ClusterRuntimeConfig prefill;
    ClusterRuntimeConfig decode;

    friend bool operator==(const PddClustersConfig &lhs,
                           const PddClustersConfig &rhs) {
        return std::tie(lhs.prefill, lhs.decode) ==
               std::tie(rhs.prefill, rhs.decode);
    }
};

struct KvCacheTransferConfig {
    double network_bandwidth_gbps = 100.0;
    double network_latency_ms = 0.1;
    double kv_cache_dtype_size_bytes = 2.0;
    bool enable_compression = false;

    friend bool operator==(const KvCacheTransferConfig &lhs,
                           const KvCacheTransferConfig &rhs) {
        return std::tie(lhs.network_bandwidth_gbps, lhs.network_latency_ms,
                        lhs.kv_cache_dtype_size_bytes,
                        lhs.enable_compression) ==
               std::tie(rhs.network_bandwidth_gbps, rhs.network_latency_ms,
                        rhs.kv_cache_dtype_size_bytes, rhs.enable_compression);
    }
};

struct PddRuntimeConfig {
    PddClustersConfig clusters;
    KvCacheTransferConfig kv_cache_transfer;

    friend bool operator==(const PddRuntimeConfig &lhs,
                           const PddRuntimeConfig &rhs) {
        return std::tie(lhs.clusters, lhs.kv_cache_transfer) ==
               std::tie(rhs.clusters, rhs.kv_cache_transfer);
    }
};

using RuntimeConfig = std::variant<ClusterRuntimeConfig, PddRuntimeConfig>;

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

    [[nodiscard]] ClusterRuntimeConfig &cluster();
    [[nodiscard]] const ClusterRuntimeConfig &cluster() const;
    [[nodiscard]] PddRuntimeConfig &pdd();
    [[nodiscard]] const PddRuntimeConfig &pdd() const;

    friend bool operator==(const SimulationConfig &lhs,
                           const SimulationConfig &rhs) {
        return std::tie(lhs.schema_version, lhs.run_id, lhs.simulation_mode,
                        lhs.system_architecture, lhs.enable_parallel_clusters,
                        lhs.prefix_cache, lhs.cluster_scheduler, lhs.runtime) ==
               std::tie(rhs.schema_version, rhs.run_id, rhs.simulation_mode,
                        rhs.system_architecture, rhs.enable_parallel_clusters,
                        rhs.prefix_cache, rhs.cluster_scheduler, rhs.runtime);
    }
};

[[nodiscard]] std::string_view to_string(SimulationMode mode) noexcept;
[[nodiscard]] std::string_view
to_string(SystemArchitecture architecture) noexcept;
[[nodiscard]] std::string_view
to_string(PrefixCachingKeyMode key_mode) noexcept;
[[nodiscard]] std::string_view to_string(SchedulerType type) noexcept;
[[nodiscard]] std::string_view to_string(SchedulingPolicy policy) noexcept;
[[nodiscard]] std::string_view to_string(ClusterSchedulerType type) noexcept;
[[nodiscard]] std::string_view to_string(ModelKind kind) noexcept;
[[nodiscard]] std::string_view to_string(MoeRoutingMode mode) noexcept;
[[nodiscard]] std::string_view
to_string(MoeRoutingDistribution distribution) noexcept;
[[nodiscard]] std::string_view to_string(ExecutionModelType type) noexcept;

[[nodiscard]] SimulationConfig
parse_simulation_config_json(std::string_view json_text);

[[nodiscard]] ModelConfig load_model_config(std::string_view model_name);
[[nodiscard]] std::string
serialize_simulation_config_json(const SimulationConfig &config);

} // namespace frontier::config
