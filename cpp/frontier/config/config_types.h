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

} // namespace frontier::config
