#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <variant>
#include <vector>

#include "frontier/attention/ops.h"
#include "frontier/core/cluster_type.h"
#include "frontier/core/ids.h"

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
    // Select the target with the fewest observable running + waiting
    // requests.  This mirrors vLLM's queue-aware placement contract without
    // consulting workload fields that are not visible to a live scheduler.
    kVllmQueueAware,
    // Select the target with the smallest observable KV footprint.  The
    // scheduler uses allocated blocks plus queued/current-frontier blocks.
    kKvAware,
    // Prefer the session's target while enough of its actual GPU prefix is
    // resident, but abandon affinity when load is imbalanced or the hit ratio
    // is too low.
    kCacheAware,
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
    // K3's Stable LatentMoE exports both the routed expert width and the
    // conventional moe_intermediate_size.  Keep the former when present so
    // execution/memory consumers can choose the faithful latent path.
    std::uint64_t routed_expert_hidden_size = 0;
    bool latent_moe_use_norm = false;
    std::uint64_t attn_res_block_size = 0;
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
    // MLA variants are explicit model metadata. Kimi K3's full-attention
    // layers use both options; legacy MLA checkpoints leave them disabled.
    bool mla_use_output_gate = false;
    bool mla_use_nope = false;
    bool use_mfa = false;
    std::uint64_t q_lora_rank = 0;
    std::uint64_t kv_lora_rank = 0;
    std::uint64_t qk_nope_head_dim = 0;
    std::uint64_t qk_rope_head_dim = 0;
    std::uint64_t qk_head_dim = 0;
    std::uint64_t v_head_dim = 0;
    std::uint64_t share_q_dim = 0;
    // Kimi-Linear/Kimi-K3 hybrid attention metadata.  The official K3
    // checkpoint stores these values below text_config.linear_attn_config;
    // keeping them on ModelConfig lets consumers account for the recurrent
    // state without having to reopen the source JSON.
    std::uint64_t num_kda_layers = 0;
    std::uint64_t num_mla_layers = 0;
    std::uint64_t kda_num_heads = 0;
    std::uint64_t kda_num_k_heads = 0;
    std::uint64_t kda_num_v_heads = 0;
    std::uint64_t kda_key_head_dim = 0;
    std::uint64_t kda_value_head_dim = 0;
    std::uint64_t kda_head_dim = 0;
    std::uint64_t kda_short_conv_kernel_size = 0;
    std::uint64_t kda_conv_state_dim = 0;
    // Layer ids are zero based, unlike the one-based ids in the Hugging Face
    // linear_attn_config arrays.  They are sorted and contain an exact
    // partition of the decoder layers whenever hybrid metadata is present.
    std::vector<std::uint64_t> kda_layer_indices;
    std::vector<std::uint64_t> mla_layer_indices;
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

    [[nodiscard]] bool has_kda() const noexcept {
        return num_kda_layers != 0 || !kda_layer_indices.empty();
    }

    // The current KDA weight-memory and execution models implement the
    // symmetric topology published for Kimi K3. Keep this predicate explicit
    // so every entry point can reject unsupported generalized Kimi-Linear
    // shapes instead of silently applying the K3 formulas to them.
    [[nodiscard]] bool kda_topology_is_symmetric() const noexcept {
        return !has_kda() || (kda_num_heads == kda_num_k_heads &&
                              kda_num_heads == kda_num_v_heads &&
                              kda_head_dim == kda_key_head_dim &&
                              kda_head_dim == kda_value_head_dim);
    }

    [[nodiscard]] bool is_kda_layer(std::uint64_t layer) const noexcept {
        return std::binary_search(kda_layer_indices.begin(),
                                  kda_layer_indices.end(), layer);
    }

    [[nodiscard]] bool is_mla_layer(std::uint64_t layer) const noexcept {
        if (!mla_layer_indices.empty()) {
            return std::binary_search(mla_layer_indices.begin(),
                                      mla_layer_indices.end(), layer);
        }
        // Legacy flat MLA assets (including Kimi K2) do not carry a per-layer
        // list, so use_mla retains its historical all-layers semantics.
        return use_mla && !has_kda();
    }

    [[nodiscard]] bool has_hybrid_attention() const noexcept {
        return has_kda() && num_mla_layers != 0;
    }

    [[nodiscard]] bool has_latent_moe() const noexcept {
        return routed_expert_hidden_size != 0 || latent_moe_use_norm;
    }

    [[nodiscard]] std::uint64_t
    kda_recurrent_state_elements_per_layer() const noexcept {
        return kda_num_v_heads == 0 || kda_key_head_dim == 0 ||
                       kda_value_head_dim == 0
                   ? 0
                   : kda_num_v_heads * kda_key_head_dim * kda_value_head_dim;
    }

    [[nodiscard]] std::uint64_t
    kda_conv_state_elements_per_layer() const noexcept {
        return kda_conv_state_dim == 0 || kda_short_conv_kernel_size <= 1
                   ? 0
                   : kda_conv_state_dim * (kda_short_conv_kernel_size - 1);
    }

    [[nodiscard]] std::uint64_t kda_state_elements_per_layer() const noexcept {
        return kda_recurrent_state_elements_per_layer() +
               kda_conv_state_elements_per_layer();
    }

    [[nodiscard]] std::uint64_t kda_state_snapshot_bytes(
        std::uint64_t element_size_bytes = 2) const noexcept {
        return num_kda_layers * kda_state_elements_per_layer() *
               element_size_bytes;
    }

    friend bool operator==(const ModelConfig &lhs, const ModelConfig &rhs) {
        return std::tie(lhs.name, lhs.model_type, lhs.kind, lhs.num_layers,
                        lhs.hidden_size, lhs.intermediate_size,
                        lhs.dense_intermediate_size, lhs.moe_intermediate_size,
                        lhs.routed_expert_hidden_size, lhs.latent_moe_use_norm,
                        lhs.attn_res_block_size, lhs.num_query_heads,
                        lhs.num_kv_heads, lhs.head_dim, lhs.gated_mlp,
                        lhs.fused_add_norm, lhs.num_experts,
                        lhs.num_experts_per_token, lhs.total_expert_num,
                        lhs.router_topk, lhs.num_shared_experts,
                        lhs.first_k_dense_replace, lhs.moe_layer_freq,
                        lhs.vocab_size, lhs.use_mla, lhs.mla_use_output_gate,
                        lhs.mla_use_nope, lhs.use_mfa, lhs.q_lora_rank,
                        lhs.kv_lora_rank, lhs.qk_nope_head_dim,
                        lhs.qk_rope_head_dim, lhs.qk_head_dim, lhs.v_head_dim,
                        lhs.share_q_dim, lhs.num_kda_layers, lhs.num_mla_layers,
                        lhs.kda_num_heads, lhs.kda_num_k_heads,
                        lhs.kda_num_v_heads, lhs.kda_key_head_dim,
                        lhs.kda_value_head_dim, lhs.kda_head_dim,
                        lhs.kda_short_conv_kernel_size, lhs.kda_conv_state_dim,
                        lhs.kda_layer_indices, lhs.mla_layer_indices,
                        lhs.has_dsa_marker, lhs.exotic_attention_fields,
                        lhs.attention) ==
               std::tie(rhs.name, rhs.model_type, rhs.kind, rhs.num_layers,
                        rhs.hidden_size, rhs.intermediate_size,
                        rhs.dense_intermediate_size, rhs.moe_intermediate_size,
                        rhs.routed_expert_hidden_size, rhs.latent_moe_use_norm,
                        rhs.attn_res_block_size, rhs.num_query_heads,
                        rhs.num_kv_heads, rhs.head_dim, rhs.gated_mlp,
                        rhs.fused_add_norm, rhs.num_experts,
                        rhs.num_experts_per_token, rhs.total_expert_num,
                        rhs.router_topk, rhs.num_shared_experts,
                        rhs.first_k_dense_replace, rhs.moe_layer_freq,
                        rhs.vocab_size, rhs.use_mla, rhs.mla_use_output_gate,
                        rhs.mla_use_nope, rhs.use_mfa, rhs.q_lora_rank,
                        rhs.kv_lora_rank, rhs.qk_nope_head_dim,
                        rhs.qk_rope_head_dim, rhs.qk_head_dim, rhs.v_head_dim,
                        rhs.share_q_dim, rhs.num_kda_layers, rhs.num_mla_layers,
                        rhs.kda_num_heads, rhs.kda_num_k_heads,
                        rhs.kda_num_v_heads, rhs.kda_key_head_dim,
                        rhs.kda_value_head_dim, rhs.kda_head_dim,
                        rhs.kda_short_conv_kernel_size, rhs.kda_conv_state_dim,
                        rhs.kda_layer_indices, rhs.mla_layer_indices,
                        rhs.has_dsa_marker, rhs.exotic_attention_fields,
                        rhs.attention);
    }
    friend bool operator!=(const ModelConfig &lhs, const ModelConfig &rhs) {
        return !(lhs == rhs);
    }
};

struct PipelineStageLayerRange {
    std::uint64_t begin = 0;
    std::uint64_t end = 0;

    [[nodiscard]] std::uint64_t size() const noexcept { return end - begin; }

    friend bool operator==(const PipelineStageLayerRange &lhs,
                           const PipelineStageLayerRange &rhs) noexcept {
        return lhs.begin == rhs.begin && lhs.end == rhs.end;
    }
    friend bool operator!=(const PipelineStageLayerRange &lhs,
                           const PipelineStageLayerRange &rhs) noexcept {
        return !(lhs == rhs);
    }
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

// Static attention-family identity used by PP stage signatures.  This is
// intentionally independent from the execution-time predictor's family enum:
// it describes model structure, not a prediction policy.
enum class AttentionFamily {
    kStandard,
    kMla,
    kKda,
};

struct LayerStaticSignature {
    AttentionFamily attention_family = AttentionFamily::kStandard;
    bool is_moe = false;
    bool has_dense_mlp = true;

    friend bool operator==(const LayerStaticSignature &lhs,
                           const LayerStaticSignature &rhs) noexcept {
        return std::tie(lhs.attention_family, lhs.is_moe, lhs.has_dense_mlp) ==
               std::tie(rhs.attention_family, rhs.is_moe, rhs.has_dense_mlp);
    }
    friend bool operator!=(const LayerStaticSignature &lhs,
                           const LayerStaticSignature &rhs) noexcept {
        return !(lhs == rhs);
    }
};

struct StageTimingSignature {
    std::vector<LayerStaticSignature> ordered_layers;
    bool owns_input_embedding = false;
    bool owns_final_norm = false;
    bool owns_lm_head = false;
    bool emits_pp_send = false;

    friend bool operator==(const StageTimingSignature &lhs,
                           const StageTimingSignature &rhs) noexcept {
        return std::tie(lhs.ordered_layers, lhs.owns_input_embedding,
                        lhs.owns_final_norm, lhs.owns_lm_head,
                        lhs.emits_pp_send) ==
               std::tie(rhs.ordered_layers, rhs.owns_input_embedding,
                        rhs.owns_final_norm, rhs.owns_lm_head,
                        rhs.emits_pp_send);
    }
    friend bool operator!=(const StageTimingSignature &lhs,
                           const StageTimingSignature &rhs) noexcept {
        return !(lhs == rhs);
    }
};

struct StageMemorySignature {
    std::uint64_t resident_weight_bytes = 0;
    std::uint64_t reserved_bytes = 0;
    std::vector<std::uint64_t> kv_bytes_per_block_by_rank;
    std::vector<std::uint64_t> kda_snapshot_bytes_by_rank;

    friend bool operator==(const StageMemorySignature &lhs,
                           const StageMemorySignature &rhs) noexcept {
        return std::tie(lhs.resident_weight_bytes, lhs.reserved_bytes,
                        lhs.kv_bytes_per_block_by_rank,
                        lhs.kda_snapshot_bytes_by_rank) ==
               std::tie(rhs.resident_weight_bytes, rhs.reserved_bytes,
                        rhs.kv_bytes_per_block_by_rank,
                        rhs.kda_snapshot_bytes_by_rank);
    }
    friend bool operator!=(const StageMemorySignature &lhs,
                           const StageMemorySignature &rhs) noexcept {
        return !(lhs == rhs);
    }
};

struct PipelineStageMemoryProfile {
    StageId stage_id;
    PipelineStageLayerRange layers;
    std::uint64_t resident_weight_bytes = 0;
    std::uint64_t reserved_bytes = 0;
    std::uint64_t free_bytes = 0;
    std::vector<std::uint64_t> kv_bytes_per_block_by_rank;
    std::vector<std::uint64_t> kda_snapshot_bytes_by_rank;

    [[nodiscard]] StageMemorySignature memory_signature() const {
        return StageMemorySignature{resident_weight_bytes, reserved_bytes,
                                    kv_bytes_per_block_by_rank,
                                    kda_snapshot_bytes_by_rank};
    }

    friend bool operator==(const PipelineStageMemoryProfile &lhs,
                           const PipelineStageMemoryProfile &rhs) noexcept {
        return std::tie(lhs.stage_id, lhs.layers, lhs.resident_weight_bytes,
                        lhs.reserved_bytes, lhs.free_bytes,
                        lhs.kv_bytes_per_block_by_rank,
                        lhs.kda_snapshot_bytes_by_rank) ==
               std::tie(rhs.stage_id, rhs.layers, rhs.resident_weight_bytes,
                        rhs.reserved_bytes, rhs.free_bytes,
                        rhs.kv_bytes_per_block_by_rank,
                        rhs.kda_snapshot_bytes_by_rank);
    }
    friend bool operator!=(const PipelineStageMemoryProfile &lhs,
                           const PipelineStageMemoryProfile &rhs) noexcept {
        return !(lhs == rhs);
    }
};

// First-occurrence-order group IDs make diagnostics and tests stable without
// relying on hash iteration order.  Group signatures are operation-specific:
// timing equivalence and memory equivalence are intentionally independent.
struct PipelineStageGroupCatalogue {
    std::vector<StageTimingSignature> timing_groups;
    std::vector<StageMemorySignature> memory_groups;
    std::vector<std::uint32_t> stage_to_timing_group;
    std::vector<std::uint32_t> stage_to_memory_group;
    std::vector<std::uint64_t> timing_group_multiplicity;
    std::vector<std::uint64_t> memory_group_multiplicity;

    friend bool operator==(const PipelineStageGroupCatalogue &lhs,
                           const PipelineStageGroupCatalogue &rhs) noexcept {
        return std::tie(lhs.timing_groups, lhs.memory_groups,
                        lhs.stage_to_timing_group, lhs.stage_to_memory_group,
                        lhs.timing_group_multiplicity,
                        lhs.memory_group_multiplicity) ==
               std::tie(rhs.timing_groups, rhs.memory_groups,
                        rhs.stage_to_timing_group, rhs.stage_to_memory_group,
                        rhs.timing_group_multiplicity,
                        rhs.memory_group_multiplicity);
    }
    friend bool operator!=(const PipelineStageGroupCatalogue &lhs,
                           const PipelineStageGroupCatalogue &rhs) noexcept {
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

// Whether every MoE layer routes the batch's tokens to the same experts, or
// each layer draws its own assignment. This is orthogonal to `distribution`,
// which decides the shape of one assignment rather than how many are drawn.
//
// "shared" routes once per batch and charges every layer the same per-expert
// token counts, matching a model whose routers agree. It also lets the
// predictor evaluate routing and the expert roofline a single time and reuse
// the result for the remaining layers, which is what makes the compressed
// `moe_layer_event_mode` values exact rather than approximate.
//
// "per_layer" re-draws the assignment for each layer, seeded by that layer's
// index, so imbalance decorrelates across depth. It forces per-layer routing
// and per-layer expert predictions in detailed mode; first_layer_scaled
// deliberately overrides it with one batch-shared representative draw.
enum class MoeRoutingLayerScope {
    kShared,
    kPerLayer,
};

struct MoeRoutingConfig {
    MoeRoutingMode mode = MoeRoutingMode::kSimulation;
    MoeRoutingDistribution distribution = MoeRoutingDistribution::kBalanced;
    std::uint64_t seed = 42;
    // Defaults to the scope that matches this struct's own default mode and
    // distribution, so a default-constructed value stays self-consistent.
    // Code that selects a layer-dependent distribution in C++ must set this
    // as well; JSON configs that omit the key get
    // default_moe_routing_layer_scope() applied during parsing.
    MoeRoutingLayerScope layer_scope = MoeRoutingLayerScope::kShared;

    friend bool operator==(const MoeRoutingConfig &lhs,
                           const MoeRoutingConfig &rhs) {
        return std::tie(lhs.mode, lhs.distribution, lhs.seed,
                        lhs.layer_scope) ==
               std::tie(rhs.mode, rhs.distribution, rhs.seed, rhs.layer_scope);
    }
};

// Layer scope predates its config field: a distribution that ignores the
// per-layer seed produced one shared assignment, and every other combination
// re-drew per layer. Configs that omit `layer_scope` keep exactly that
// behavior. Note that "skewed" and "zipf" are also seed-independent, so
// selecting "shared" for them changes no number -- it only lets the predictor
// stop recomputing an assignment it already has.
[[nodiscard]] MoeRoutingLayerScope
default_moe_routing_layer_scope(MoeRoutingMode mode,
                                MoeRoutingDistribution distribution) noexcept;

struct ParallelismConfig {
    std::uint64_t num_replicas = 1;
    std::uint64_t tensor_parallel_size = 1;
    std::uint64_t pipeline_parallel_size = 1;
    std::uint64_t data_parallel_size = 1;
    std::uint64_t moe_tensor_parallel_size = 1;
    std::uint64_t moe_expert_parallel_size = 1;
    // DCP reuses ranks inside each TP group and shards KV entries along the
    // token axis. It therefore does not contribute to accelerator count.
    // Keep extensions after the legacy fields so positional aggregate
    // initialization remains source-compatible.
    std::uint64_t decode_context_parallel_size = 1;
    // When enabled, each pipeline stage owns its execution resources
    // exclusively. This is an opt-in topology contract; the default keeps
    // legacy parallelism behavior unchanged.
    bool pipeline_exclusive = false;
    // Optional exact number of transformer layers assigned to each physical
    // pipeline stage. An empty vector keeps the legacy near-even contiguous
    // partition. When set, it must contain pipeline_parallel_size positive
    // entries whose sum equals the model layer count.
    std::vector<std::uint64_t> pipeline_stage_layer_counts;

    [[nodiscard]] std::uint64_t attention_parallel_size() const noexcept {
        return tensor_parallel_size * data_parallel_size;
    }
    [[nodiscard]] std::uint64_t moe_parallel_size() const noexcept {
        return moe_tensor_parallel_size * moe_expert_parallel_size;
    }

    friend bool operator==(const ParallelismConfig &lhs,
                           const ParallelismConfig &rhs) {
        return std::tie(
                   lhs.num_replicas, lhs.tensor_parallel_size,
                   lhs.pipeline_parallel_size, lhs.data_parallel_size,
                   lhs.moe_tensor_parallel_size, lhs.moe_expert_parallel_size,
                   lhs.decode_context_parallel_size, lhs.pipeline_exclusive,
                   lhs.pipeline_stage_layer_counts) ==
               std::tie(
                   rhs.num_replicas, rhs.tensor_parallel_size,
                   rhs.pipeline_parallel_size, rhs.data_parallel_size,
                   rhs.moe_tensor_parallel_size, rhs.moe_expert_parallel_size,
                   rhs.decode_context_parallel_size, rhs.pipeline_exclusive,
                   rhs.pipeline_stage_layer_counts);
    }
};

[[nodiscard]] inline PipelineStageLayerRange
pipeline_stage_layer_range(std::uint64_t num_layers,
                           const ParallelismConfig &parallelism,
                           std::uint64_t stage) {
    if (parallelism.pipeline_stage_layer_counts.empty()) {
        return pipeline_stage_layer_range(
            num_layers, parallelism.pipeline_parallel_size, stage);
    }
    if (parallelism.pipeline_parallel_size == 0 ||
        parallelism.pipeline_stage_layer_counts.size() !=
            parallelism.pipeline_parallel_size ||
        stage >= parallelism.pipeline_parallel_size) {
        throw std::invalid_argument(
            "invalid explicit pipeline layer partition");
    }
    std::uint64_t assigned_layers = 0;
    for (const std::uint64_t count : parallelism.pipeline_stage_layer_counts) {
        if (count == 0 || count > num_layers - assigned_layers) {
            throw std::invalid_argument(
                "invalid explicit pipeline layer partition");
        }
        assigned_layers += count;
    }
    if (assigned_layers != num_layers) {
        throw std::invalid_argument(
            "invalid explicit pipeline layer partition");
    }
    std::uint64_t begin = 0;
    for (std::uint64_t index = 0; index < stage; ++index) {
        begin += parallelism.pipeline_stage_layer_counts[index];
    }
    return PipelineStageLayerRange{
        begin, begin + parallelism.pipeline_stage_layer_counts[stage]};
}

struct ClusterSchedulerConfig {
    ClusterSchedulerType type = ClusterSchedulerType::kRoundRobin;
    // PDD-only stage overrides.  A missing override intentionally falls back
    // to `type` so every existing co-location/PDD config remains valid.
    std::optional<ClusterSchedulerType> prefill_type;
    std::optional<ClusterSchedulerType> decode_type;
    double cache_threshold = 0.5;
    std::uint64_t balance_abs_threshold = 32;
    double balance_rel_threshold = 1.1;

    [[nodiscard]] ClusterSchedulerType
    type_for_cluster(::frontier::ClusterType cluster_type) const noexcept {
        if (cluster_type == ::frontier::ClusterType::kPrefill &&
            prefill_type.has_value()) {
            return *prefill_type;
        }
        if (cluster_type == ::frontier::ClusterType::kDecode &&
            decode_type.has_value()) {
            return *decode_type;
        }
        return type;
    }

    friend bool operator==(const ClusterSchedulerConfig &lhs,
                           const ClusterSchedulerConfig &rhs) {
        return std::tie(lhs.type, lhs.prefill_type, lhs.decode_type,
                        lhs.cache_threshold, lhs.balance_abs_threshold,
                        lhs.balance_rel_threshold) ==
               std::tie(rhs.type, rhs.prefill_type, rhs.decode_type,
                        rhs.cache_threshold, rhs.balance_abs_threshold,
                        rhs.balance_rel_threshold);
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
    // Controls how pipeline-stage causality is represented in the DES.
    // "exact" retains one arrival/schedule/end chain per stage;
    // "collapsed" fuses safe pipeline transitions while preserving the
    // stage-local resource calendar. Unsupported synchronized paths fall
    // back to the exact chain.
    //
    // The two modes produce identical timings and identical request and batch
    // records; only the event count differs. That holds because the analytical
    // model derives a batch's attention inputs entirely from the batch's own
    // RequestBatchSnapshot, so a stage's predicted work does not depend on when
    // the stage runs. Collapsed mode predicts every stage at stage-zero entry
    // and therefore relies on exactly that property; see
    // docs/design/kimi-k3-support.md 12.3 for the divergence this replaced.
    std::string pipeline_event_mode = "exact";
    // Derived per-session KDA recurrent-state footprint, expressed in GPU
    // scheduler blocks.  It is populated by memory resolution and omitted
    // from user-authored JSON unless a caller explicitly normalizes it.
    std::uint64_t kda_snapshot_blocks_per_session = 0;

    friend bool operator==(const SchedulerConfig &lhs,
                           const SchedulerConfig &rhs) {
        return std::tie(lhs.type, lhs.scheduling_policy, lhs.batch_size_cap,
                        lhs.max_tokens_in_batch, lhs.enable_preemption,
                        lhs.enable_chunked_prefill,
                        lhs.long_prefill_token_threshold, lhs.block_size,
                        lhs.num_blocks, lhs.watermark_blocks_fraction,
                        lhs.num_preallocate_tokens, lhs.pipeline_event_mode,
                        lhs.kda_snapshot_blocks_per_session) ==
               std::tie(rhs.type, rhs.scheduling_policy, rhs.batch_size_cap,
                        rhs.max_tokens_in_batch, rhs.enable_preemption,
                        rhs.enable_chunked_prefill,
                        rhs.long_prefill_token_threshold, rhs.block_size,
                        rhs.num_blocks, rhs.watermark_blocks_fraction,
                        rhs.num_preallocate_tokens, rhs.pipeline_event_mode,
                        rhs.kda_snapshot_blocks_per_session);
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
    // Canonical router weight storage precision.  `moe_router_weight` is
    // retained as a legacy fallback for configs written before router
    // compute/storage precisions were split.
    std::string router_weight_storage;
    std::string lm_head;
    std::string lm_head_weight;
    std::string lm_head_activation;
    // K3-native mixed-precision overrides.  Empty values are optional and
    // inherit the legacy family/suffix fields through the accessors below.
    std::string routed_expert_weight;
    std::string routed_expert_activation;
    // Stable LatentMoE's dense down/up projections sit outside the routed
    // expert bank and may use a different dtype.  Empty values retain the
    // historical routed-expert precision for backward compatibility.
    std::string latent_moe_projection_weight;
    std::string latent_moe_projection_activation;
    std::string shared_expert_weight;
    std::string shared_expert_activation;
    std::string dense_mlp_weight;
    std::string dense_mlp_activation;
    std::string router_compute;
    std::string kda_snapshot;

    [[nodiscard]] bool empty() const noexcept {
        return attention.empty() && dense.empty() && moe_expert.empty() &&
               moe_router.empty() && kv_cache.empty() &&
               communication.empty() && attention_weight.empty() &&
               attention_activation.empty() && dense_weight.empty() &&
               dense_activation.empty() && moe_expert_weight.empty() &&
               moe_expert_activation.empty() && moe_router_weight.empty() &&
               moe_router_activation.empty() && lm_head.empty() &&
               lm_head_weight.empty() && lm_head_activation.empty() &&
               router_weight_storage.empty() && routed_expert_weight.empty() &&
               routed_expert_activation.empty() &&
               latent_moe_projection_weight.empty() &&
               latent_moe_projection_activation.empty() &&
               shared_expert_weight.empty() &&
               shared_expert_activation.empty() && dense_mlp_weight.empty() &&
               dense_mlp_activation.empty() && router_compute.empty() &&
               kda_snapshot.empty();
    }

    friend bool operator==(const OperatorPrecisionConfig &lhs,
                           const OperatorPrecisionConfig &rhs) {
        return std::tie(lhs.attention, lhs.dense, lhs.moe_expert,
                        lhs.moe_router, lhs.kv_cache, lhs.communication,
                        lhs.attention_weight, lhs.attention_activation,
                        lhs.dense_weight, lhs.dense_activation,
                        lhs.moe_expert_weight, lhs.moe_expert_activation,
                        lhs.moe_router_weight, lhs.moe_router_activation,
                        lhs.router_weight_storage, lhs.lm_head,
                        lhs.lm_head_weight, lhs.lm_head_activation,
                        lhs.routed_expert_weight, lhs.routed_expert_activation,
                        lhs.latent_moe_projection_weight,
                        lhs.latent_moe_projection_activation,
                        lhs.shared_expert_weight, lhs.shared_expert_activation,
                        lhs.dense_mlp_weight, lhs.dense_mlp_activation,
                        lhs.router_compute, lhs.kda_snapshot) ==
               std::tie(rhs.attention, rhs.dense, rhs.moe_expert,
                        rhs.moe_router, rhs.kv_cache, rhs.communication,
                        rhs.attention_weight, rhs.attention_activation,
                        rhs.dense_weight, rhs.dense_activation,
                        rhs.moe_expert_weight, rhs.moe_expert_activation,
                        rhs.moe_router_weight, rhs.moe_router_activation,
                        rhs.router_weight_storage, rhs.lm_head,
                        rhs.lm_head_weight, rhs.lm_head_activation,
                        rhs.routed_expert_weight, rhs.routed_expert_activation,
                        rhs.latent_moe_projection_weight,
                        rhs.latent_moe_projection_activation,
                        rhs.shared_expert_weight, rhs.shared_expert_activation,
                        rhs.dense_mlp_weight, rhs.dense_mlp_activation,
                        rhs.router_compute, rhs.kda_snapshot);
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
    // Selects an explicit operator-efficiency/fusion profile.  "generic"
    // preserves the historical roofline constants.  K3 profiles are opt-in
    // because they describe backend-specific Blackwell kernel stacks. Rubin
    // use is an explicit forward projection of those public priors.
    std::string kernel_profile = "generic";
    // "detailed" predicts and emits synchronization events one MoE layer at a
    // time.
    // "first_layer_scaled" emits the first MoE layer normally, uses one
    // batch-shared routing draw regardless of moe_routing.layer_scope, reuses
    // its expert path, and accumulates attention delays by implementation
    // family.
    // "stage_group_scaled" additionally exposes canonical PP stage groups,
    // uses the same family-aware compression only for layer-invariant routing,
    // and otherwise falls back to exact per-layer prediction.
    std::string moe_layer_event_mode = "detailed";
    // "generic" uses the configured collective backend. The SM100 profile is
    // an opt-in public-prior model for fused MegaMoE dispatch/expert/combine;
    // on Rubin it is a forward projection rather than a calibrated profile.
    std::string moe_communication_backend = "generic";
    // Optional absolute MegaMoE sensitivity/calibration overrides. They are
    // valid only with k3_deepgemm_megamoe; absent values use profile priors.
    std::optional<double> mega_moe_tail_io_fraction;
    std::optional<double> mega_moe_wave_exposure;
    std::optional<double> mega_moe_cluster_task_latency_us;
    std::uint64_t tensor_parallel_size = 8;
    double network_bandwidth_gbps = 400.0;
    double network_latency_us = 1.0;
    double intra_node_bandwidth_gbps = 14'400.0;

    friend bool operator==(const AnalyticalExecutionModelConfig &lhs,
                           const AnalyticalExecutionModelConfig &rhs) {
        return std::tie(
                   lhs.device, lhs.device_overrides, lhs.precision,
                   lhs.operator_precisions, lhs.kernel_profile,
                   lhs.moe_layer_event_mode, lhs.moe_communication_backend,
                   lhs.mega_moe_tail_io_fraction, lhs.mega_moe_wave_exposure,
                   lhs.mega_moe_cluster_task_latency_us,
                   lhs.tensor_parallel_size, lhs.network_bandwidth_gbps,
                   lhs.network_latency_us, lhs.intra_node_bandwidth_gbps) ==
               std::tie(rhs.device, rhs.device_overrides, rhs.precision,
                        rhs.operator_precisions, rhs.kernel_profile,
                        rhs.moe_layer_event_mode, rhs.moe_communication_backend,
                        rhs.mega_moe_tail_io_fraction,
                        rhs.mega_moe_wave_exposure,
                        rhs.mega_moe_cluster_task_latency_us,
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
        // Keep the historical getter as an alias.  New configs should use
        // router_weight_storage_precision() so the storage dtype cannot be
        // confused with the FP32 router compute dtype.
        return router_weight_storage_precision();
    }
    [[nodiscard]] const std::string &
    router_weight_storage_precision() const noexcept {
        if (!operator_precisions.router_weight_storage.empty()) {
            return operator_precisions.router_weight_storage;
        }
        // Legacy configs used moe_router_weight for the router's resident
        // weight dtype.  Preserve that fallback when no canonical field is
        // present.
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

    // K3-native precision families retain the legacy operator-specific
    // fields as fallbacks.  The older *_weight/*_activation overrides are
    // checked before their unsuffixed family so existing configs preserve
    // their most specific setting.
    [[nodiscard]] const std::string &
    routed_expert_weight_precision() const noexcept {
        return operator_precisions.routed_expert_weight.empty()
                   ? moe_expert_weight_precision()
                   : operator_precisions.routed_expert_weight;
    }
    [[nodiscard]] const std::string &
    routed_expert_activation_precision() const noexcept {
        return operator_precisions.routed_expert_activation.empty()
                   ? moe_expert_activation_precision()
                   : operator_precisions.routed_expert_activation;
    }
    [[nodiscard]] const std::string &
    latent_moe_projection_weight_precision() const noexcept {
        return operator_precisions.latent_moe_projection_weight.empty()
                   ? routed_expert_weight_precision()
                   : operator_precisions.latent_moe_projection_weight;
    }
    [[nodiscard]] const std::string &
    latent_moe_projection_activation_precision() const noexcept {
        return operator_precisions.latent_moe_projection_activation.empty()
                   ? routed_expert_activation_precision()
                   : operator_precisions.latent_moe_projection_activation;
    }
    [[nodiscard]] const std::string &
    shared_expert_weight_precision() const noexcept {
        return operator_precisions.shared_expert_weight.empty()
                   ? moe_expert_weight_precision()
                   : operator_precisions.shared_expert_weight;
    }
    [[nodiscard]] const std::string &
    shared_expert_activation_precision() const noexcept {
        return operator_precisions.shared_expert_activation.empty()
                   ? moe_expert_activation_precision()
                   : operator_precisions.shared_expert_activation;
    }
    [[nodiscard]] const std::string &
    dense_mlp_weight_precision() const noexcept {
        return operator_precisions.dense_mlp_weight.empty()
                   ? dense_weight_precision()
                   : operator_precisions.dense_mlp_weight;
    }
    [[nodiscard]] const std::string &
    dense_mlp_activation_precision() const noexcept {
        return operator_precisions.dense_mlp_activation.empty()
                   ? dense_activation_precision()
                   : operator_precisions.dense_mlp_activation;
    }
    [[nodiscard]] const std::string &router_compute_precision() const noexcept {
        if (!operator_precisions.router_compute.empty()) {
            return operator_precisions.router_compute;
        }
        // A legacy router weight override was historically also used as the
        // router compute dtype.  Keep that fallback for old configs while
        // allowing the canonical router_weight_storage field to be
        // independent.  K3 native defaults install an explicit FP32 value.
        return operator_precisions.moe_router_weight.empty()
                   ? moe_router_precision()
                   : operator_precisions.moe_router_weight;
    }
    [[nodiscard]] const std::string &kda_snapshot_precision() const noexcept {
        // This is an execution-config override only. A model-aware resolver
        // installs K3's native BF16 snapshot default before memory sizing.
        return operator_precisions.kda_snapshot.empty()
                   ? precision
                   : operator_precisions.kda_snapshot;
    }
};

// Fill model-native operator defaults without overwriting explicit modern or
// legacy precision overrides. Kimi K3 uses its published mixed-precision
// execution policy; other models are unchanged.
void apply_model_native_precision_defaults(
    AnalyticalExecutionModelConfig &execution, const ModelConfig &model);

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

struct GpuMemoryConfig {
    bool auto_calculate_num_blocks = false;
    // Required for every runtime cluster.  Zero denotes an unbound in-memory
    // object and is rejected by parsing and resolve_gpu_memory_config().
    std::uint64_t capacity_bytes_per_gpu = 0;
    double runtime_reserve_fraction = 0.0;
    std::uint64_t runtime_reserve_bytes = 0;
    double weight_overhead_fraction = 0.0;

    // Materialized diagnostics. These are emitted in normalized configs and
    // make the automatic block calculation auditable and reproducible.
    std::uint64_t model_weight_bytes_per_gpu = 0;
    std::uint64_t kv_cache_budget_bytes_per_gpu = 0;
    std::uint64_t kv_cache_bytes_per_block = 0;

    // Exact PP stage/rank-local memory diagnostics.  These are derived fields
    // populated by resolve_gpu_memory_config; they are deliberately omitted
    // from the value-equality contract and legacy JSON surface so older
    // normalized configs continue to round-trip.  Consumers that need physical
    // occupancy should use these profiles instead of reconstructing a full
    // model footprint from kv_cache_bytes_per_block.
    std::vector<PipelineStageMemoryProfile> pipeline_stage_memory_profiles;
    PipelineStageGroupCatalogue pipeline_stage_group_catalogue;
    std::uint64_t ordinary_kv_capacity_blocks = 0;
    std::uint64_t ordinary_kv_limiting_stage = 0;
    std::uint64_t ordinary_kv_limiting_rank = 0;
    std::uint64_t kda_snapshot_limiting_stage = 0;
    std::uint64_t kda_snapshot_limiting_rank = 0;

    friend bool operator==(const GpuMemoryConfig &lhs,
                           const GpuMemoryConfig &rhs) {
        return std::tie(lhs.auto_calculate_num_blocks,
                        lhs.capacity_bytes_per_gpu,
                        lhs.runtime_reserve_fraction, lhs.runtime_reserve_bytes,
                        lhs.weight_overhead_fraction) ==
               std::tie(rhs.auto_calculate_num_blocks,
                        rhs.capacity_bytes_per_gpu,
                        rhs.runtime_reserve_fraction, rhs.runtime_reserve_bytes,
                        rhs.weight_overhead_fraction);
    }
};

struct ClusterRuntimeConfig {
    ParallelismConfig parallelism;
    SchedulerConfig scheduler;
    ExecutionModelConfig execution_model;
    GpuMemoryConfig gpu_memory;
    ModelConfig model;
    MoeRoutingConfig moe_routing;

    friend bool operator==(const ClusterRuntimeConfig &lhs,
                           const ClusterRuntimeConfig &rhs) {
        return std::tie(lhs.parallelism, lhs.scheduler, lhs.execution_model,
                        lhs.gpu_memory, lhs.model, lhs.moe_routing) ==
               std::tie(rhs.parallelism, rhs.scheduler, rhs.execution_model,
                        rhs.gpu_memory, rhs.model, rhs.moe_routing);
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

enum class CpuKVCacheEvictionPolicy { kSessionLruSuffix };

enum class CpuKVCacheCapacityPressurePolicy { kPrefixFit, kSkipOffload };

enum class CpuKVCacheTransferConcurrency { kFullDuplexSerialized };

struct CpuKVCacheConfig {
    bool enabled = false;
    std::uint64_t capacity_bytes = 0;
    bool static_slice_per_gpu = false;
    std::uint64_t capacity_bytes_per_gpu = 750'000'000'000ULL;
    double dram_bandwidth_gbps_per_gpu = 4'800.0;
    double c2c_bandwidth_gbps_per_gpu = 3'600.0;
    double write_bandwidth_gbps = 64.0;
    double write_latency_ms = 0.01;
    double read_bandwidth_gbps = 64.0;
    double read_latency_ms = 0.01;
    CpuKVCacheEvictionPolicy eviction_policy =
        CpuKVCacheEvictionPolicy::kSessionLruSuffix;
    CpuKVCacheCapacityPressurePolicy capacity_pressure_policy =
        CpuKVCacheCapacityPressurePolicy::kPrefixFit;
    CpuKVCacheTransferConcurrency transfer_concurrency =
        CpuKVCacheTransferConcurrency::kFullDuplexSerialized;

    friend bool operator==(const CpuKVCacheConfig &lhs,
                           const CpuKVCacheConfig &rhs) {
        return std::tie(
                   lhs.enabled, lhs.capacity_bytes, lhs.static_slice_per_gpu,
                   lhs.capacity_bytes_per_gpu, lhs.dram_bandwidth_gbps_per_gpu,
                   lhs.c2c_bandwidth_gbps_per_gpu, lhs.write_bandwidth_gbps,
                   lhs.write_latency_ms, lhs.read_bandwidth_gbps,
                   lhs.read_latency_ms, lhs.eviction_policy,
                   lhs.capacity_pressure_policy, lhs.transfer_concurrency) ==
               std::tie(
                   rhs.enabled, rhs.capacity_bytes, rhs.static_slice_per_gpu,
                   rhs.capacity_bytes_per_gpu, rhs.dram_bandwidth_gbps_per_gpu,
                   rhs.c2c_bandwidth_gbps_per_gpu, rhs.write_bandwidth_gbps,
                   rhs.write_latency_ms, rhs.read_bandwidth_gbps,
                   rhs.read_latency_ms, rhs.eviction_policy,
                   rhs.capacity_pressure_policy, rhs.transfer_concurrency);
    }
};

struct ResolvedCpuKVCacheTargetConfig {
    bool enabled = false;
    std::uint64_t capacity_bytes = 0;
    std::uint64_t capacity_blocks = 0;
    std::uint64_t bytes_per_block = 0;
    // Hybrid KDA models keep one latest recurrent-state snapshot per cached
    // session. The CPU manager charges it as one indivisible group while
    // transfer accounting uses the exact byte size.
    std::uint64_t kda_snapshot_bytes = 0;
    std::uint64_t kda_snapshot_blocks = 0;
    double d2h_bandwidth_gbps = 0.0;
    double d2h_latency_ms = 0.0;
    double h2d_bandwidth_gbps = 0.0;
    double h2d_latency_ms = 0.0;
    CpuKVCacheCapacityPressurePolicy capacity_pressure_policy =
        CpuKVCacheCapacityPressurePolicy::kPrefixFit;
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
    CpuKVCacheConfig cpu_kv_cache;
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
                        lhs.prefix_cache, lhs.cpu_kv_cache,
                        lhs.cluster_scheduler, lhs.runtime) ==
               std::tie(rhs.schema_version, rhs.run_id, rhs.simulation_mode,
                        rhs.system_architecture, rhs.enable_parallel_clusters,
                        rhs.prefix_cache, rhs.cpu_kv_cache,
                        rhs.cluster_scheduler, rhs.runtime);
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
to_string(MoeRoutingLayerScope layer_scope) noexcept;
[[nodiscard]] std::string_view
to_string(MoeRoutingDistribution distribution) noexcept;
[[nodiscard]] std::string_view to_string(ExecutionModelType type) noexcept;
[[nodiscard]] std::string_view
to_string(CpuKVCacheEvictionPolicy policy) noexcept;
[[nodiscard]] std::string_view
to_string(CpuKVCacheCapacityPressurePolicy policy) noexcept;
[[nodiscard]] std::string_view
to_string(CpuKVCacheTransferConcurrency concurrency) noexcept;

[[nodiscard]] ResolvedCpuKVCacheTargetConfig
resolve_cpu_kv_cache_target(const SimulationConfig &config);

// Resolve the common KDA recurrent-state snapshot element size used by a
// sequential PDD transfer. Fixed execution follows the native BF16 snapshot;
// analytical PREFILL/DECODE clusters must agree on the exact precision when
// the model exposes KDA layers. A mixed analytical/fixed pair is therefore
// valid only when the analytical side also uses BF16.
[[nodiscard]] double
resolve_pdd_kda_snapshot_dtype_size_bytes(const PddClustersConfig &clusters);

// Build the exact static PP memory view for one logical (replica, DP) target.
// The returned profiles contain one entry per physical pipeline stage and one
// rank-local KV/KDA byte value per decode-context-parallel shard.  CPU/PDD
// aggregate transfer helpers intentionally remain separate from this view.
[[nodiscard]] std::vector<PipelineStageMemoryProfile>
build_pipeline_stage_memory_profiles(const ClusterRuntimeConfig &cluster);

// Build the canonical execution signature for one physical PP stage. Both
// configuration/metrics grouping and predictor caching must use this helper.
[[nodiscard]] StageTimingSignature
build_pipeline_stage_timing_signature(const ModelConfig &model,
                                      const ParallelismConfig &parallelism,
                                      std::uint64_t stage);

// Build deterministic first-occurrence-order timing and memory groups for a
// profile set.  `profiles` must be ordered by stage id and cover every PP
// stage in `cluster.parallelism.pipeline_parallel_size`.
[[nodiscard]] PipelineStageGroupCatalogue build_pipeline_stage_group_catalogue(
    const ClusterRuntimeConfig &cluster,
    const std::vector<PipelineStageMemoryProfile> &profiles);

// Resolve the shared logical KV-block capacity from real stage/rank physical
// profiles.  Ranks with zero KV bytes do not constrain ordinary KV capacity,
// but remain part of the profile set for KDA snapshot admission.
[[nodiscard]] std::uint64_t resolve_pipeline_logical_kv_capacity(
    const std::vector<PipelineStageMemoryProfile> &profiles,
    std::uint64_t *limiting_stage = nullptr,
    std::uint64_t *limiting_rank = nullptr);

// Derive the conservative scalar logical charge for one resident KDA
// snapshot.  This includes KDA-only stages with B == 0 and checks that every
// nonzero snapshot shard fits in its stage/rank's free HBM budget.
[[nodiscard]] std::uint64_t resolve_pipeline_kda_snapshot_charge(
    const std::vector<PipelineStageMemoryProfile> &profiles,
    std::uint64_t logical_kv_capacity, std::uint64_t *limiting_stage = nullptr,
    std::uint64_t *limiting_rank = nullptr);

// Resolve per-rank model weight storage and the remaining rank-local KV block
// capacity. Every caller must bind a positive gpu_memory capacity. Manual
// configs retain their explicit scheduler.num_blocks but are checked against
// exact PP stage/rank profiles; an explicit value on an automatic config is
// treated as a normalized-config consistency check.
void resolve_gpu_memory_config(
    ClusterRuntimeConfig &cluster,
    std::optional<std::uint64_t> explicit_num_blocks = std::nullopt);

[[nodiscard]] SimulationConfig
parse_simulation_config_json(std::string_view json_text);

[[nodiscard]] ModelConfig load_model_config(std::string_view model_name);
[[nodiscard]] std::string
serialize_simulation_config_json(const SimulationConfig &config);

} // namespace frontier::config
