#pragma once

#include <cstdint>
#include <optional>
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
    bool enabled = false;
    PrefixCachingKeyMode key_mode = PrefixCachingKeyMode::kSession;

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
    kStickyRoundRobin,
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
    std::uint64_t dense_intermediate_size = 11'008;
    std::uint64_t moe_intermediate_size = 11'008;
    std::uint64_t num_query_heads = 32;
    std::uint64_t num_kv_heads = 32;
    std::uint64_t head_dim = 128;
    bool gated_mlp = true;
    bool fused_add_norm = true;
    std::uint64_t num_experts = 1;
    std::uint64_t num_experts_per_token = 1;
    std::uint64_t total_expert_num = 1;
    std::uint64_t router_topk = 1;
    std::uint64_t num_shared_experts = 0;
    std::uint64_t first_k_dense_replace = 0;
    std::uint64_t moe_layer_freq = 1;
    std::uint64_t vocab_size = 32'000;
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

    [[nodiscard]] bool is_moe_layer(std::uint64_t layer) const noexcept {
        return is_moe() && moe_layer_freq > 0 &&
               layer >= first_k_dense_replace &&
               (layer - first_k_dense_replace) % moe_layer_freq == 0;
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
                        lhs.hidden_size, lhs.intermediate_size,
                        lhs.dense_intermediate_size, lhs.moe_intermediate_size,
                        lhs.num_query_heads, lhs.num_kv_heads, lhs.head_dim,
                        lhs.gated_mlp, lhs.fused_add_norm, lhs.num_experts,
                        lhs.num_experts_per_token, lhs.total_expert_num,
                        lhs.router_topk, lhs.num_shared_experts,
                        lhs.first_k_dense_replace, lhs.moe_layer_freq,
                        lhs.vocab_size, lhs.use_mla, lhs.use_mfa,
                        lhs.q_lora_rank, lhs.kv_lora_rank, lhs.qk_nope_head_dim,
                        lhs.qk_rope_head_dim, lhs.qk_head_dim, lhs.v_head_dim,
                        lhs.share_q_dim, lhs.has_dsa_marker,
                        lhs.exotic_attention_fields, lhs.attention) ==
               std::tie(rhs.name, rhs.model_type, rhs.kind, rhs.num_layers,
                        rhs.hidden_size, rhs.intermediate_size,
                        rhs.dense_intermediate_size, rhs.moe_intermediate_size,
                        rhs.num_query_heads, rhs.num_kv_heads, rhs.head_dim,
                        rhs.gated_mlp, rhs.fused_add_norm, rhs.num_experts,
                        rhs.num_experts_per_token, rhs.total_expert_num,
                        rhs.router_topk, rhs.num_shared_experts,
                        rhs.first_k_dense_replace, rhs.moe_layer_freq,
                        rhs.vocab_size, rhs.use_mla, rhs.use_mfa,
                        rhs.q_lora_rank, rhs.kv_lora_rank, rhs.qk_nope_head_dim,
                        rhs.qk_rope_head_dim, rhs.qk_head_dim, rhs.v_head_dim,
                        rhs.share_q_dim, rhs.has_dsa_marker,
                        rhs.exotic_attention_fields, rhs.attention);
    }
    friend bool operator!=(const ModelConfig &lhs, const ModelConfig &rhs) {
        return !(lhs == rhs);
    }
};

struct PipelineStageLayerRange {
    std::uint64_t begin = 0;
    std::uint64_t end = 0;

    [[nodiscard]] std::uint64_t size() const noexcept { return end - begin; }
};

[[nodiscard]] inline PipelineStageLayerRange
pipeline_stage_layer_range(std::uint64_t num_layers,
                           std::uint64_t pipeline_parallel_size,
                           std::uint64_t stage) {
    if (pipeline_parallel_size == 0 || pipeline_parallel_size > num_layers ||
        stage >= pipeline_parallel_size) {
        throw std::invalid_argument("invalid pipeline layer partition");
    }
    const std::uint64_t base = num_layers / pipeline_parallel_size;
    const std::uint64_t remainder = num_layers % pipeline_parallel_size;
    const std::uint64_t begin =
        stage * base + (stage < remainder ? stage : remainder);
    const std::uint64_t size = base + (stage < remainder ? 1 : 0);
    return PipelineStageLayerRange{begin, begin + size};
}

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

struct OperatorPrecisionConfig {
    // Empty fields inherit AnalyticalExecutionModelConfig::precision.
    std::string attention;
    std::string dense;
    std::string moe_expert;
    std::string moe_router;
    std::string kv_cache;
    std::string communication;
    std::string attention_weight;
    std::string attention_activation;
    std::string dense_weight;
    std::string dense_activation;
    std::string moe_expert_weight;
    std::string moe_expert_activation;
    std::string moe_router_weight;
    std::string moe_router_activation;
    std::string lm_head;
    std::string lm_head_weight;
    std::string lm_head_activation;

    [[nodiscard]] bool empty() const noexcept {
        return attention.empty() && dense.empty() && moe_expert.empty() &&
               moe_router.empty() && kv_cache.empty() &&
               communication.empty() && attention_weight.empty() &&
               attention_activation.empty() && dense_weight.empty() &&
               dense_activation.empty() && moe_expert_weight.empty() &&
               moe_expert_activation.empty() && moe_router_weight.empty() &&
               moe_router_activation.empty() && lm_head.empty() &&
               lm_head_weight.empty() && lm_head_activation.empty();
    }

    friend bool operator==(const OperatorPrecisionConfig &lhs,
                           const OperatorPrecisionConfig &rhs) {
        return std::tie(lhs.attention, lhs.dense, lhs.moe_expert,
                        lhs.moe_router, lhs.kv_cache, lhs.communication,
                        lhs.attention_weight, lhs.attention_activation,
                        lhs.dense_weight, lhs.dense_activation,
                        lhs.moe_expert_weight, lhs.moe_expert_activation,
                        lhs.moe_router_weight, lhs.moe_router_activation,
                        lhs.lm_head, lhs.lm_head_weight,
                        lhs.lm_head_activation) ==
               std::tie(rhs.attention, rhs.dense, rhs.moe_expert,
                        rhs.moe_router, rhs.kv_cache, rhs.communication,
                        rhs.attention_weight, rhs.attention_activation,
                        rhs.dense_weight, rhs.dense_activation,
                        rhs.moe_expert_weight, rhs.moe_expert_activation,
                        rhs.moe_router_weight, rhs.moe_router_activation,
                        rhs.lm_head, rhs.lm_head_weight,
                        rhs.lm_head_activation);
    }
};

struct AnalyticalDeviceOverrides {
    std::optional<double> hbm_bandwidth_tbps;
    std::optional<double> fp32_tflops;
    std::optional<double> fp16_tflops;
    std::optional<double> fp8_tflops;
    std::optional<double> fp4_tflops;

    [[nodiscard]] bool empty() const noexcept {
        return !hbm_bandwidth_tbps.has_value() && !fp32_tflops.has_value() &&
               !fp16_tflops.has_value() && !fp8_tflops.has_value() &&
               !fp4_tflops.has_value();
    }

    [[nodiscard]] bool complete() const noexcept {
        return hbm_bandwidth_tbps.has_value() && fp32_tflops.has_value() &&
               fp16_tflops.has_value() && fp8_tflops.has_value() &&
               fp4_tflops.has_value();
    }

    friend bool operator==(const AnalyticalDeviceOverrides &lhs,
                           const AnalyticalDeviceOverrides &rhs) {
        return std::tie(lhs.hbm_bandwidth_tbps, lhs.fp32_tflops,
                        lhs.fp16_tflops, lhs.fp8_tflops, lhs.fp4_tflops) ==
               std::tie(rhs.hbm_bandwidth_tbps, rhs.fp32_tflops,
                        rhs.fp16_tflops, rhs.fp8_tflops, rhs.fp4_tflops);
    }
};

struct AnalyticalExecutionModelConfig {
    std::string device = "rubin";
    AnalyticalDeviceOverrides device_overrides;
    std::string precision = "fp16";
    OperatorPrecisionConfig operator_precisions;
    // "detailed" predicts and emits synchronization events one MoE layer at a
    // time.
    // "first_layer_scaled" emits the first layer normally and models the
    // remaining identical layers as one accumulated delay.
    std::string moe_layer_event_mode = "detailed";
    std::uint64_t tensor_parallel_size = 8;
    double network_bandwidth_gbps = 400.0;
    double network_latency_us = 1.0;
    double intra_node_bandwidth_gbps = 14'400.0;

    friend bool operator==(const AnalyticalExecutionModelConfig &lhs,
                           const AnalyticalExecutionModelConfig &rhs) {
        return std::tie(lhs.device, lhs.device_overrides, lhs.precision,
                        lhs.operator_precisions, lhs.moe_layer_event_mode,
                        lhs.tensor_parallel_size, lhs.network_bandwidth_gbps,
                        lhs.network_latency_us,
                        lhs.intra_node_bandwidth_gbps) ==
               std::tie(rhs.device, rhs.device_overrides, rhs.precision,
                        rhs.operator_precisions, rhs.moe_layer_event_mode,
                        rhs.tensor_parallel_size, rhs.network_bandwidth_gbps,
                        rhs.network_latency_us, rhs.intra_node_bandwidth_gbps);
    }

    [[nodiscard]] const std::string &attention_precision() const noexcept {
        return operator_precisions.attention.empty()
                   ? precision
                   : operator_precisions.attention;
    }
    [[nodiscard]] const std::string &dense_precision() const noexcept {
        return operator_precisions.dense.empty() ? precision
                                                 : operator_precisions.dense;
    }
    [[nodiscard]] const std::string &moe_expert_precision() const noexcept {
        return operator_precisions.moe_expert.empty()
                   ? precision
                   : operator_precisions.moe_expert;
    }
    [[nodiscard]] const std::string &moe_router_precision() const noexcept {
        return operator_precisions.moe_router.empty()
                   ? precision
                   : operator_precisions.moe_router;
    }
    [[nodiscard]] const std::string &kv_cache_precision() const noexcept {
        return operator_precisions.kv_cache.empty()
                   ? precision
                   : operator_precisions.kv_cache;
    }
    [[nodiscard]] const std::string &communication_precision() const noexcept {
        return operator_precisions.communication.empty()
                   ? precision
                   : operator_precisions.communication;
    }
    [[nodiscard]] const std::string &
    attention_weight_precision() const noexcept {
        return operator_precisions.attention_weight.empty()
                   ? attention_precision()
                   : operator_precisions.attention_weight;
    }
    [[nodiscard]] const std::string &
    attention_activation_precision() const noexcept {
        return operator_precisions.attention_activation.empty()
                   ? attention_precision()
                   : operator_precisions.attention_activation;
    }
    [[nodiscard]] const std::string &dense_weight_precision() const noexcept {
        return operator_precisions.dense_weight.empty()
                   ? dense_precision()
                   : operator_precisions.dense_weight;
    }
    [[nodiscard]] const std::string &
    dense_activation_precision() const noexcept {
        return operator_precisions.dense_activation.empty()
                   ? dense_precision()
                   : operator_precisions.dense_activation;
    }
    [[nodiscard]] const std::string &
    moe_expert_weight_precision() const noexcept {
        return operator_precisions.moe_expert_weight.empty()
                   ? moe_expert_precision()
                   : operator_precisions.moe_expert_weight;
    }
    [[nodiscard]] const std::string &
    moe_expert_activation_precision() const noexcept {
        return operator_precisions.moe_expert_activation.empty()
                   ? moe_expert_precision()
                   : operator_precisions.moe_expert_activation;
    }
    [[nodiscard]] const std::string &
    moe_router_weight_precision() const noexcept {
        return operator_precisions.moe_router_weight.empty()
                   ? moe_router_precision()
                   : operator_precisions.moe_router_weight;
    }
    [[nodiscard]] const std::string &
    moe_router_activation_precision() const noexcept {
        return operator_precisions.moe_router_activation.empty()
                   ? moe_router_precision()
                   : operator_precisions.moe_router_activation;
    }
    [[nodiscard]] const std::string &lm_head_precision() const noexcept {
        return operator_precisions.lm_head.empty()
                   ? dense_precision()
                   : operator_precisions.lm_head;
    }
    [[nodiscard]] const std::string &lm_head_weight_precision() const noexcept {
        return operator_precisions.lm_head_weight.empty()
                   ? lm_head_precision()
                   : operator_precisions.lm_head_weight;
    }
    [[nodiscard]] const std::string &
    lm_head_activation_precision() const noexcept {
        return operator_precisions.lm_head_activation.empty()
                   ? lm_head_precision()
                   : operator_precisions.lm_head_activation;
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
