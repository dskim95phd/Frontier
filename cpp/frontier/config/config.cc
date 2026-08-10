#include "frontier/config/config.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

#include "frontier/kv_cache_transfer/analytical_transfer.h"

namespace frontier::config {

namespace {

double storage_bytes_per_element(std::string_view precision) {
    if (precision == "fp32") {
        return 4.0;
    }
    if (precision == "fp16" || precision == "bf16") {
        return 2.0;
    }
    if (precision == "fp8" || precision == "int8") {
        return 1.0;
    }
    if (precision == "mxfp8") {
        return 1.0 + 1.0 / 32.0;
    }
    if (precision == "fp4" || precision == "int4") {
        return 0.5;
    }
    if (precision == "mxfp4") {
        return 0.5 + 1.0 / 32.0;
    }
    throw ConfigError("unsupported storage precision: " +
                      std::string{precision});
}

std::uint64_t ceil_div(std::uint64_t numerator, std::uint64_t denominator) {
    if (denominator == 0) {
        throw ConfigError("GPU memory parallelism divisor must be positive");
    }
    return numerator / denominator +
           static_cast<std::uint64_t>(numerator % denominator != 0);
}

void add_weight_bytes(long double &total, std::uint64_t rows,
                      std::uint64_t columns, double bytes_per_element,
                      std::uint64_t copies = 1) {
    total += static_cast<long double>(rows) *
             static_cast<long double>(columns) *
             static_cast<long double>(copies) *
             static_cast<long double>(bytes_per_element);
    if (!std::isfinite(total) ||
        total > static_cast<long double>(
                    std::numeric_limits<std::uint64_t>::max())) {
        throw ConfigError("per-GPU model weight storage overflows uint64");
    }
}

void add_attention_weight_bytes(long double &total, const ModelConfig &model,
                                const ParallelismConfig &parallelism,
                                double bytes_per_element, std::uint64_t layer) {
    const std::uint64_t tp = parallelism.tensor_parallel_size;
    const std::uint64_t local_query_heads = ceil_div(model.num_query_heads, tp);
    const std::uint64_t local_kv_heads = ceil_div(model.num_kv_heads, tp);
    if (model.is_kda_layer(layer)) {
        const std::uint64_t local_kda_heads = ceil_div(model.kda_num_heads, tp);
        const std::uint64_t projection_size =
            local_kda_heads * model.kda_head_dim;
        // Q/K/V projections and their three causal short convolutions.
        add_weight_bytes(total, model.hidden_size, projection_size,
                         bytes_per_element, 3);
        add_weight_bytes(total, projection_size,
                         model.kda_short_conv_kernel_size, bytes_per_element,
                         3);
        // Decay/beta projections, the full-rank output gate used by K3, and
        // the final output projection.
        add_weight_bytes(total, model.hidden_size, model.kda_head_dim,
                         bytes_per_element);
        add_weight_bytes(total, model.kda_head_dim, projection_size,
                         bytes_per_element);
        add_weight_bytes(total, model.hidden_size, local_kda_heads,
                         bytes_per_element);
        add_weight_bytes(total, model.hidden_size, projection_size,
                         bytes_per_element);
        add_weight_bytes(total, projection_size, model.hidden_size,
                         bytes_per_element);
        // A_log, dt_bias, and the gated RMSNorm scale are small but persistent
        // parameters and belong in the automatic memory budget.
        add_weight_bytes(total, 1, local_kda_heads, sizeof(float));
        add_weight_bytes(total, 1, projection_size, sizeof(float));
        add_weight_bytes(total, 1, model.kda_head_dim, bytes_per_element);
        return;
    }
    if (model.use_mla) {
        if (model.q_lora_rank == 0) {
            add_weight_bytes(total, model.hidden_size,
                             local_query_heads * model.qk_head_dim,
                             bytes_per_element);
        } else {
            // MLA's low-rank input projection is replicated, while the
            // head-facing projection is attention-TP sharded.
            add_weight_bytes(total, model.hidden_size, model.q_lora_rank,
                             bytes_per_element);
            add_weight_bytes(total, model.q_lora_rank,
                             local_query_heads * model.qk_head_dim,
                             bytes_per_element);
        }
        add_weight_bytes(total, model.hidden_size,
                         model.kv_lora_rank + model.qk_rope_head_dim,
                         bytes_per_element);
        add_weight_bytes(total, model.kv_lora_rank,
                         local_query_heads *
                             (model.qk_nope_head_dim + model.v_head_dim),
                         bytes_per_element);
        if (model.mla_use_output_gate) {
            // Gated MLA's output gate is a hidden-size to value-head output
            // projection. It follows attention TP, so only local query heads
            // are resident on this GPU.
            add_weight_bytes(total, model.hidden_size,
                             local_query_heads * model.v_head_dim,
                             bytes_per_element);
        }
        add_weight_bytes(total, local_query_heads * model.v_head_dim,
                         model.hidden_size, bytes_per_element);
        return;
    }
    if (model.use_mfa) {
        add_weight_bytes(total, model.hidden_size,
                         model.share_q_dim +
                             2 * local_kv_heads * model.head_dim,
                         bytes_per_element);
        add_weight_bytes(total, model.share_q_dim,
                         local_query_heads * model.head_dim, bytes_per_element);
        add_weight_bytes(total, local_query_heads * model.head_dim,
                         model.hidden_size, bytes_per_element);
        return;
    }
    add_weight_bytes(total, model.hidden_size,
                     (local_query_heads + 2 * local_kv_heads) * model.head_dim,
                     bytes_per_element);
    add_weight_bytes(total, ceil_div(model.hidden_size, tp), model.hidden_size,
                     bytes_per_element);
}

void add_mlp_weight_bytes(long double &total, std::uint64_t hidden_size,
                          std::uint64_t local_intermediate_size, bool gated,
                          double bytes_per_element, std::uint64_t copies = 1) {
    add_weight_bytes(total, hidden_size, local_intermediate_size,
                     bytes_per_element, copies * (gated ? 2 : 1));
    add_weight_bytes(total, local_intermediate_size, hidden_size,
                     bytes_per_element, copies);
}

std::uint64_t
model_weight_bytes_per_gpu(const ModelConfig &model,
                           const ParallelismConfig &parallelism,
                           const AnalyticalExecutionModelConfig &execution,
                           double weight_overhead_fraction) {
    if (!model.kda_topology_is_symmetric()) {
        throw ConfigError(
            "automatic GPU weight memory currently supports only symmetric "
            "KDA Q/K/V topology");
    }
    const double attention_bytes =
        storage_bytes_per_element(execution.attention_weight_precision());
    const double dense_bytes =
        storage_bytes_per_element(execution.dense_weight_precision());
    const double dense_mlp_bytes =
        storage_bytes_per_element(execution.dense_mlp_weight_precision());
    const double routed_expert_bytes = storage_bytes_per_element(
        execution.routed_expert_weight_precision());
    const double latent_moe_projection_bytes = storage_bytes_per_element(
        execution.latent_moe_projection_weight_precision());
    const double shared_expert_bytes = storage_bytes_per_element(
        execution.shared_expert_weight_precision());
    // Router weights remain resident in their storage dtype.  The router
    // compute dtype (normally FP32 for K3) controls the GEMM roofline, not the
    // automatic GPU HBM model; casts are assumed fused into the router kernel.
    const double router_bytes = storage_bytes_per_element(
        execution.router_weight_storage_precision());
    const double lm_head_bytes =
        storage_bytes_per_element(execution.lm_head_weight_precision());
    std::uint64_t maximum = 0;
    for (std::uint64_t stage = 0; stage < parallelism.pipeline_parallel_size;
         ++stage) {
        long double total = 0.0L;
        const PipelineStageLayerRange layers = pipeline_stage_layer_range(
            model.num_layers, parallelism.pipeline_parallel_size, stage);
        if (stage == 0) {
            add_weight_bytes(
                total,
                ceil_div(model.vocab_size, parallelism.tensor_parallel_size),
                model.hidden_size, lm_head_bytes);
        }
        if (stage + 1 == parallelism.pipeline_parallel_size) {
            // Conservatively model untied input embeddings and LM head. The
            // model contract currently has no tie_word_embeddings field.
            add_weight_bytes(
                total,
                ceil_div(model.vocab_size, parallelism.tensor_parallel_size),
                model.hidden_size, lm_head_bytes);
        }
        for (std::uint64_t layer = layers.begin; layer < layers.end; ++layer) {
            add_attention_weight_bytes(total, model, parallelism,
                                       attention_bytes, layer);
            // Attention and post-attention norms are replicated.
            add_weight_bytes(total, 2, model.hidden_size, dense_bytes);
            if (!model.is_moe_layer(layer)) {
                add_mlp_weight_bytes(total, model.hidden_size,
                                     ceil_div(model.dense_intermediate_size,
                                              parallelism.tensor_parallel_size),
                                     model.gated_mlp, dense_mlp_bytes);
                continue;
            }
            add_weight_bytes(total, model.hidden_size, model.total_expert_num,
                             router_bytes);
            const std::uint64_t local_intermediate =
                ceil_div(model.moe_intermediate_size,
                         parallelism.moe_tensor_parallel_size);
            const std::uint64_t local_routed_experts = ceil_div(
                model.total_expert_num, parallelism.moe_expert_parallel_size);
            const std::uint64_t routed_hidden =
                model.has_latent_moe() ? model.routed_expert_hidden_size
                                       : model.hidden_size;
            add_mlp_weight_bytes(total, routed_hidden, local_intermediate,
                                 model.gated_mlp, routed_expert_bytes,
                                 local_routed_experts);
            if (model.has_latent_moe()) {
                add_weight_bytes(total, model.hidden_size, routed_hidden,
                                 latent_moe_projection_bytes);
                add_weight_bytes(total, routed_hidden, model.hidden_size,
                                 latent_moe_projection_bytes);
                if (model.latent_moe_use_norm) {
                    add_weight_bytes(total, 1, routed_hidden, dense_bytes);
                }
            }
            // Shared experts are replicated across EP lanes and sharded only
            // by the MoE tensor-parallel dimension.
            add_mlp_weight_bytes(total, model.hidden_size, local_intermediate,
                                 model.gated_mlp, shared_expert_bytes,
                                 model.num_shared_experts);
        }
        // Final RMSNorm lives on the last pipeline stage.
        if (stage + 1 == parallelism.pipeline_parallel_size) {
            add_weight_bytes(total, 1, model.hidden_size, dense_bytes);
        }
        total *= 1.0L + static_cast<long double>(weight_overhead_fraction);
        if (!std::isfinite(total) ||
            total > static_cast<long double>(
                        std::numeric_limits<std::uint64_t>::max())) {
            throw ConfigError("per-GPU model weight storage overflows uint64");
        }
        maximum =
            std::max(maximum, static_cast<std::uint64_t>(std::ceil(total)));
    }
    return maximum;
}

void resolve_kda_snapshot_charge(ClusterRuntimeConfig &cluster,
                                 std::uint64_t kv_cache_bytes_per_block) {
    cluster.scheduler.kda_snapshot_blocks_per_session = 0;
    if (!cluster.model.has_kda()) {
        return;
    }
    if (kv_cache_bytes_per_block == 0) {
        throw ConfigError(
            "KDA snapshot charge requires a positive KV block byte size");
    }
    try {
        const std::uint64_t snapshot_bytes =
            kv_cache_transfer::model_kda_state_snapshot_size_bytes_rank_local(
                cluster.model,
                cluster.execution_model.type == ExecutionModelType::kAnalytical
                    ? storage_bytes_per_element(
                          cluster.execution_model.analytical
                              .kda_snapshot_precision())
                    : 4.0,
                cluster.parallelism.tensor_parallel_size);
        cluster.scheduler.kda_snapshot_blocks_per_session =
            ceil_div(snapshot_bytes, kv_cache_bytes_per_block);
    } catch (const kv_cache_transfer::TransferModelError &error) {
        throw ConfigError(std::string{"invalid KDA snapshot layout: "} +
                          error.what());
    }
}

std::uint64_t
derive_rank_local_kv_block_bytes(const ClusterRuntimeConfig &cluster,
                                 double kv_cache_dtype_size_bytes) {
    try {
        std::uint64_t maximum_block_bytes = 0;
        for (std::uint64_t rank = 0;
             rank < cluster.parallelism.decode_context_parallel_size; ++rank) {
            maximum_block_bytes = std::max(
                maximum_block_bytes,
                kv_cache_transfer::model_kv_cache_size_bytes_rank_local(
                    cluster.scheduler.block_size, cluster.model,
                    kv_cache_dtype_size_bytes,
                    cluster.parallelism.decode_context_parallel_size, rank));
        }
        return maximum_block_bytes;
    } catch (const kv_cache_transfer::TransferModelError &error) {
        throw ConfigError(std::string{"invalid automatic GPU KV layout: "} +
                          error.what());
    }
}

} // namespace

double resolve_pdd_kda_snapshot_dtype_size_bytes(
    const PddClustersConfig &clusters) {
    // Non-KDA models never append a recurrent-state payload to a PDD KV
    // transfer. Keep the default explicit for callers that construct a
    // predictor uniformly for all model families.
    const bool prefill_has_kda = clusters.prefill.model.has_kda();
    const bool decode_has_kda = clusters.decode.model.has_kda();
    if (prefill_has_kda != decode_has_kda) {
        throw ConfigError(
            "PDD PREFILL and DECODE must agree on KDA snapshot support");
    }
    if (!prefill_has_kda) {
        return 4.0;
    }

    struct SnapshotPrecision {
        std::string name;
        double size_bytes;
    };
    const auto resolve = [](const ClusterRuntimeConfig &cluster) {
        if (cluster.execution_model.type != ExecutionModelType::kAnalytical) {
            // Fixed execution has no operator precision declaration. The
            // runtime contract follows Kimi K3's native BF16 snapshot.
            return SnapshotPrecision{"bf16", 2.0};
        }
        const std::string &precision =
            cluster.execution_model.analytical.kda_snapshot_precision();
        return SnapshotPrecision{precision,
                                 storage_bytes_per_element(precision)};
    };

    const SnapshotPrecision prefill = resolve(clusters.prefill);
    const SnapshotPrecision decode = resolve(clusters.decode);
    const bool prefill_analytical =
        clusters.prefill.execution_model.type == ExecutionModelType::kAnalytical;
    const bool decode_analytical =
        clusters.decode.execution_model.type == ExecutionModelType::kAnalytical;

    if (prefill_analytical && decode_analytical) {
        if (prefill.name != decode.name) {
            throw ConfigError(
                "PDD PREFILL and DECODE must use the same KDA snapshot "
                "precision");
        }
        return prefill.size_bytes;
    }

    if (prefill_analytical != decode_analytical) {
        const SnapshotPrecision &analytical =
            prefill_analytical ? prefill : decode;
        if (analytical.name != "bf16") {
            throw ConfigError(
                "PDD analytical KDA snapshot precision must be bf16 when "
                "paired with fixed execution");
        }
    }
    return 2.0;
}

void apply_model_native_precision_defaults(
    AnalyticalExecutionModelConfig &execution, const ModelConfig &model) {
    if (model.model_type != "kimi_k3") {
        return;
    }
    OperatorPrecisionConfig &operators = execution.operator_precisions;
    const auto set_if_family_unset = [](std::string &target,
                                        const std::string &specific_fallback,
                                        const std::string &family_fallback,
                                        std::string_view native) {
        if (target.empty() && specific_fallback.empty() &&
            family_fallback.empty()) {
            target = native;
        }
    };

    set_if_family_unset(operators.attention_weight,
                        operators.attention_weight, operators.attention,
                        "bf16");
    set_if_family_unset(operators.attention_activation,
                        operators.attention_activation, operators.attention,
                        "bf16");
    set_if_family_unset(operators.dense_mlp_weight, operators.dense_weight,
                        operators.dense, "bf16");
    set_if_family_unset(operators.dense_mlp_activation,
                        operators.dense_activation, operators.dense, "bf16");
    if (operators.dense.empty()) {
        operators.dense = "bf16";
    }
    set_if_family_unset(operators.routed_expert_weight,
                        operators.moe_expert_weight, operators.moe_expert,
                        "mxfp4");
    set_if_family_unset(operators.routed_expert_activation,
                        operators.moe_expert_activation, operators.moe_expert,
                        "mxfp8");
    // K3 checkpoint tensors routed_expert_down_proj/up_proj are BF16. They
    // are dense Stable LatentMoE projections, not members of the quantized
    // routed-expert bank. Explicit projection overrides remain authoritative.
    if (operators.latent_moe_projection_weight.empty()) {
        operators.latent_moe_projection_weight = "bf16";
    }
    if (operators.latent_moe_projection_activation.empty()) {
        operators.latent_moe_projection_activation = "bf16";
    }
    set_if_family_unset(operators.shared_expert_weight,
                        operators.moe_expert_weight, operators.moe_expert,
                        "bf16");
    set_if_family_unset(operators.shared_expert_activation,
                        operators.moe_expert_activation, operators.moe_expert,
                        "bf16");
    // K3's router inputs/weights are stored in BF16 and converted to FP32 in
    // the forward kernel.  Keep explicit legacy family/weight overrides
    // authoritative, but do not let a canonical storage override suppress
    // the native FP32 compute default.
    if (operators.router_weight_storage.empty() &&
        operators.moe_router_weight.empty() && operators.moe_router.empty()) {
        operators.router_weight_storage = "bf16";
    }
    if (operators.moe_router_activation.empty() &&
        operators.moe_router.empty()) {
        operators.moe_router_activation = "bf16";
    }
    if (operators.router_compute.empty() && operators.moe_router.empty() &&
        operators.moe_router_weight.empty()) {
        operators.router_compute = "fp32";
    }
    if (operators.lm_head.empty() && operators.lm_head_weight.empty() &&
        operators.lm_head_activation.empty()) {
        operators.lm_head = "bf16";
    }
    if (operators.kv_cache.empty()) {
        operators.kv_cache = "fp8";
    }
    if (operators.kda_snapshot.empty()) {
        operators.kda_snapshot = "bf16";
    }
}

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
    case ClusterSchedulerType::kStickyRoundRobin:
        return "sticky_round_robin";
    case ClusterSchedulerType::kVllmQueueAware:
        return "vllm_queue_aware";
    case ClusterSchedulerType::kKvAware:
        return "kv_aware";
    case ClusterSchedulerType::kCacheAware:
        return "cache_aware";
    }
    return "unknown";
}

std::string_view to_string(ModelKind kind) noexcept {
    switch (kind) {
    case ModelKind::kDense:
        return "dense";
    case ModelKind::kMoe:
        return "moe";
    }
    return "unknown";
}

std::string_view to_string(MoeRoutingMode mode) noexcept {
    switch (mode) {
    case MoeRoutingMode::kSimulation:
        return "simulation";
    case MoeRoutingMode::kUniformLegacy:
        return "uniform_legacy";
    case MoeRoutingMode::kUniformRandom:
        return "uniform_random";
    }
    return "unknown";
}

std::string_view to_string(MoeRoutingDistribution distribution) noexcept {
    switch (distribution) {
    case MoeRoutingDistribution::kBalanced:
        return "balanced";
    case MoeRoutingDistribution::kRandom:
        return "random";
    case MoeRoutingDistribution::kSkewed:
        return "skewed";
    case MoeRoutingDistribution::kZipf:
        return "zipf";
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

std::string_view to_string(CpuKVCacheEvictionPolicy policy) noexcept {
    switch (policy) {
    case CpuKVCacheEvictionPolicy::kSessionLruSuffix:
        return "session_lru_suffix";
    }
    return "unknown";
}

std::string_view to_string(CpuKVCacheCapacityPressurePolicy policy) noexcept {
    switch (policy) {
    case CpuKVCacheCapacityPressurePolicy::kPrefixFit:
        return "prefix_fit";
    case CpuKVCacheCapacityPressurePolicy::kSkipOffload:
        return "skip_offload";
    }
    return "unknown";
}

std::string_view to_string(CpuKVCacheTransferConcurrency concurrency) noexcept {
    switch (concurrency) {
    case CpuKVCacheTransferConcurrency::kFullDuplexSerialized:
        return "full_duplex_serialized";
    }
    return "unknown";
}

void resolve_gpu_memory_config(
    ClusterRuntimeConfig &cluster,
    std::optional<std::uint64_t> explicit_num_blocks) {
    if (cluster.execution_model.type == ExecutionModelType::kAnalytical) {
        apply_model_native_precision_defaults(
            cluster.execution_model.analytical, cluster.model);
    }
    GpuMemoryConfig &memory = cluster.gpu_memory;
    if (!memory.auto_calculate_num_blocks) {
        if (!explicit_num_blocks.has_value() ||
            explicit_num_blocks.value() == 0) {
            throw ConfigError(
                "manual GPU memory config requires scheduler.num_blocks");
        }
        cluster.scheduler.num_blocks = explicit_num_blocks.value();
        const double kv_bytes =
            cluster.execution_model.type == ExecutionModelType::kAnalytical
                ? storage_bytes_per_element(
                      cluster.execution_model.analytical.kv_cache_precision())
                : 2.0;
        resolve_kda_snapshot_charge(
            cluster, derive_rank_local_kv_block_bytes(cluster, kv_bytes));
        return;
    }
    if (cluster.execution_model.type != ExecutionModelType::kAnalytical) {
        throw ConfigError(
            "automatic GPU block calculation requires analytical execution");
    }
    if (memory.capacity_bytes_per_gpu == 0) {
        throw ConfigError("gpu_memory.capacity_bytes_per_gpu must be positive");
    }
    if (!std::isfinite(memory.runtime_reserve_fraction) ||
        memory.runtime_reserve_fraction < 0.0 ||
        memory.runtime_reserve_fraction >= 1.0) {
        throw ConfigError(
            "gpu_memory.runtime_reserve_fraction must be in [0, 1)");
    }
    if (!std::isfinite(memory.weight_overhead_fraction) ||
        memory.weight_overhead_fraction < 0.0) {
        throw ConfigError(
            "gpu_memory.weight_overhead_fraction must be nonnegative");
    }
    memory.model_weight_bytes_per_gpu = model_weight_bytes_per_gpu(
        cluster.model, cluster.parallelism, cluster.execution_model.analytical,
        memory.weight_overhead_fraction);
    const long double fractional_reserve =
        static_cast<long double>(memory.capacity_bytes_per_gpu) *
        static_cast<long double>(memory.runtime_reserve_fraction);
    if (!std::isfinite(fractional_reserve) ||
        fractional_reserve > static_cast<long double>(
                                 std::numeric_limits<std::uint64_t>::max())) {
        throw ConfigError("GPU runtime reserve overflows uint64");
    }
    const std::uint64_t reserve_from_fraction =
        static_cast<std::uint64_t>(std::ceil(fractional_reserve));
    if (memory.runtime_reserve_bytes > memory.capacity_bytes_per_gpu ||
        reserve_from_fraction >
            memory.capacity_bytes_per_gpu - memory.runtime_reserve_bytes) {
        throw ConfigError("GPU runtime reserve exceeds device capacity");
    }
    const std::uint64_t total_reserve =
        memory.runtime_reserve_bytes + reserve_from_fraction;
    if (memory.model_weight_bytes_per_gpu >
        memory.capacity_bytes_per_gpu - total_reserve) {
        throw ConfigError(
            "model weights and runtime reserve exceed GPU capacity");
    }
    memory.kv_cache_budget_bytes_per_gpu = memory.capacity_bytes_per_gpu -
                                           total_reserve -
                                           memory.model_weight_bytes_per_gpu;
    memory.kv_cache_bytes_per_block = derive_rank_local_kv_block_bytes(
        cluster, storage_bytes_per_element(
                     cluster.execution_model.analytical.kv_cache_precision()));
    if (memory.kv_cache_bytes_per_block == 0) {
        throw ConfigError("automatic GPU KV block size must be positive");
    }
    const std::uint64_t calculated_blocks =
        memory.kv_cache_budget_bytes_per_gpu / memory.kv_cache_bytes_per_block;
    if (calculated_blocks == 0) {
        throw ConfigError(
            "remaining GPU KV budget cannot hold one cache block");
    }
    if (explicit_num_blocks.has_value() &&
        explicit_num_blocks.value() != calculated_blocks) {
        throw ConfigError(
            "scheduler.num_blocks does not match automatic GPU memory "
            "calculation");
    }
    cluster.scheduler.num_blocks = calculated_blocks;
    resolve_kda_snapshot_charge(cluster, memory.kv_cache_bytes_per_block);
}

ResolvedCpuKVCacheTargetConfig
resolve_cpu_kv_cache_target(const SimulationConfig &config) {
    ResolvedCpuKVCacheTargetConfig result{};
    result.enabled = config.cpu_kv_cache.enabled;
    if (!result.enabled) {
        return result;
    }
    if (config.system_architecture != SystemArchitecture::kPdDisaggregation) {
        throw ConfigError("CPU KV cache requires sequential PDD");
    }
    const auto &prefill = config.pdd().clusters.prefill;
    if (prefill.parallelism.decode_context_parallel_size != 1) {
        throw ConfigError("PDD PREFILL decode_context_parallel_size must be 1");
    }
    const auto &cpu = config.cpu_kv_cache;
    result.capacity_pressure_policy = cpu.capacity_pressure_policy;
    if (prefill.parallelism.tensor_parallel_size >
        std::numeric_limits<std::uint64_t>::max() /
            prefill.parallelism.pipeline_parallel_size) {
        throw ConfigError("CPU KV cache physical slice count overflows uint64");
    }
    const std::uint64_t slices = prefill.parallelism.tensor_parallel_size *
                                 prefill.parallelism.pipeline_parallel_size;
    if (slices == 0) {
        throw ConfigError("CPU KV cache target has no physical GPU slices");
    }
    if (cpu.static_slice_per_gpu) {
        if (cpu.capacity_bytes_per_gpu >
            std::numeric_limits<std::uint64_t>::max() / slices) {
            throw ConfigError(
                "CPU KV cache resolved capacity overflows uint64");
        }
        result.capacity_bytes = cpu.capacity_bytes_per_gpu * slices;
        const double bandwidth = std::min(cpu.dram_bandwidth_gbps_per_gpu,
                                          cpu.c2c_bandwidth_gbps_per_gpu) *
                                 static_cast<double>(slices);
        if (!std::isfinite(bandwidth)) {
            throw ConfigError("CPU KV cache resolved bandwidth overflows");
        }
        result.d2h_bandwidth_gbps = bandwidth;
        result.h2d_bandwidth_gbps = bandwidth;
    } else {
        result.capacity_bytes = cpu.capacity_bytes;
        result.d2h_bandwidth_gbps = cpu.write_bandwidth_gbps;
        result.h2d_bandwidth_gbps = cpu.read_bandwidth_gbps;
    }
    result.d2h_latency_ms = cpu.write_latency_ms;
    result.h2d_latency_ms = cpu.read_latency_ms;
    try {
        result.bytes_per_block =
            kv_cache_transfer::model_kv_cache_size_bytes_target_physical(
                prefill.scheduler.block_size, prefill.model,
                config.pdd().kv_cache_transfer.kv_cache_dtype_size_bytes,
                prefill.parallelism.tensor_parallel_size,
                prefill.parallelism.decode_context_parallel_size);
    } catch (const kv_cache_transfer::TransferModelError &error) {
        throw ConfigError(std::string{"invalid CPU KV cache block layout: "} +
                          error.what());
    }
    if (result.bytes_per_block == 0) {
        throw ConfigError("CPU KV cache bytes per block must be positive");
    }
    result.capacity_blocks = result.capacity_bytes / result.bytes_per_block;
    if (result.capacity_blocks == 0) {
        throw ConfigError(
            "CPU KV cache capacity must hold at least one logical KV block");
    }
    if (prefill.model.has_kda()) {
        try {
            result.kda_snapshot_bytes =
                kv_cache_transfer::model_kda_state_snapshot_size_bytes(
                    prefill.model,
                    prefill.execution_model.type ==
                            ExecutionModelType::kAnalytical
                        ? storage_bytes_per_element(
                              prefill.execution_model.analytical
                                  .kda_snapshot_precision())
                        : 2.0);
        } catch (const kv_cache_transfer::TransferModelError &error) {
            throw ConfigError(std::string{"invalid CPU KDA snapshot layout: "} +
                              error.what());
        }
        result.kda_snapshot_blocks =
            ceil_div(result.kda_snapshot_bytes, result.bytes_per_block);
        if (result.kda_snapshot_blocks >= result.capacity_blocks) {
            throw ConfigError(
                "CPU KV cache must hold one atomic KDA snapshot plus at "
                "least one logical KV block");
        }
    }
    return result;
}

ClusterRuntimeConfig &SimulationConfig::cluster() {
    auto *value = std::get_if<ClusterRuntimeConfig>(&runtime);
    if (value == nullptr) {
        throw ConfigError("simulation config has no single cluster runtime");
    }
    return *value;
}

const ClusterRuntimeConfig &SimulationConfig::cluster() const {
    const auto *value = std::get_if<ClusterRuntimeConfig>(&runtime);
    if (value == nullptr) {
        throw ConfigError("simulation config has no single cluster runtime");
    }
    return *value;
}

PddRuntimeConfig &SimulationConfig::pdd() {
    auto *value = std::get_if<PddRuntimeConfig>(&runtime);
    if (value == nullptr) {
        throw ConfigError("simulation config is not a PDD config");
    }
    return *value;
}

const PddRuntimeConfig &SimulationConfig::pdd() const {
    const auto *value = std::get_if<PddRuntimeConfig>(&runtime);
    if (value == nullptr) {
        throw ConfigError("simulation config is not a PDD config");
    }
    return *value;
}

} // namespace frontier::config
