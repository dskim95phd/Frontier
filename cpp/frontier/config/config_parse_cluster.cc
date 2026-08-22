#include <cmath>
#include <initializer_list>
#include <limits>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "frontier/config/config_parse_internal.h"
#include "frontier/core/precision.h"

namespace frontier::config::parse_detail {

ModelConfig parse_model(const Json &cluster, std::string_view context) {
    ModelConfig parsed =
        load_model_config(require_string(cluster, "model_name", context));
    if (cluster.contains("total_expert_num")) {
        parsed.total_expert_num =
            require_uint64(cluster, "total_expert_num", context);
    }
    if (cluster.contains("router_topk")) {
        parsed.router_topk = require_uint64(cluster, "router_topk", context);
    }
    if (cluster.contains("first_k_dense_replace")) {
        parsed.first_k_dense_replace =
            require_uint64(cluster, "first_k_dense_replace", context);
    }
    if (cluster.contains("num_shared_experts")) {
        parsed.num_shared_experts =
            require_uint64(cluster, "num_shared_experts", context);
    }
    if (!parsed.is_moe()) {
        if (parsed.total_expert_num != 1 || parsed.router_topk != 1) {
            throw ConfigError(std::string{context} +
                              " dense model requires total_expert_num=1 and "
                              "router_topk=1");
        }
        return parsed;
    }
    if (parsed.total_expert_num < 2 ||
        parsed.total_expert_num > parsed.num_experts) {
        throw ConfigError(std::string{context} +
                          ".total_expert_num must be in [2, model "
                          "num_experts]");
    }
    if (parsed.router_topk == 0 ||
        parsed.router_topk > parsed.total_expert_num) {
        throw ConfigError(std::string{context} +
                          ".router_topk must be in [1, total_expert_num]");
    }
    if (parsed.first_k_dense_replace > parsed.num_layers) {
        throw ConfigError(std::string{context} +
                          ".first_k_dense_replace must not exceed model "
                          "num_layers");
    }
    return parsed;
}

MoeRoutingConfig parse_moe_routing(const Json &root) {
    const Json &routing = root.at("moe_routing");
    require_keys(routing, {"mode", "distribution", "seed"}, {"layer_scope"},
                 "config.moe_routing");
    return [&]() {
        MoeRoutingConfig value{};
        value.mode = parse_moe_routing_mode(
            require_string(routing, "mode", "config.moe_routing"));
        value.distribution = parse_moe_routing_distribution(
            require_string(routing, "distribution", "config.moe_routing"));
        value.seed = require_uint64(routing, "seed", "config.moe_routing");
        // Omitting the field keeps whatever the mode and distribution implied
        // before it existed, so configs written against the older schema keep
        // producing identical numbers.
        value.layer_scope =
            routing.contains("layer_scope")
                ? parse_moe_routing_layer_scope(require_string(
                      routing, "layer_scope", "config.moe_routing"))
                : default_moe_routing_layer_scope(value.mode,
                                                  value.distribution);
        return value;
    }();
}

PrefixCacheConfig parse_prefix_cache(const Json &root) {
    const Json &prefix_cache = root.at("prefix_cache");
    require_exact_keys(prefix_cache, {"enabled", "key_mode"},
                       "config.prefix_cache");
    return [&]() {
        PrefixCacheConfig value{};
        value.enabled =
            require_bool(prefix_cache, "enabled", "config.prefix_cache");
        value.key_mode = parse_prefix_key_mode(
            require_string(prefix_cache, "key_mode", "config.prefix_cache"));
        return value;
    }();
}

CpuKVCacheConfig parse_cpu_kv_cache(const Json &root) {
    if (!root.contains("cpu_kv_cache")) {
        return CpuKVCacheConfig{};
    }
    const Json &cpu = root.at("cpu_kv_cache");
    require_exact_keys(cpu,
                       {
                           "enabled",
                           "capacity_bytes",
                           "static_slice_per_gpu",
                           "capacity_bytes_per_gpu",
                           "dram_bandwidth_gbps_per_gpu",
                           "c2c_bandwidth_gbps_per_gpu",
                           "write_bandwidth_gbps",
                           "write_latency_ms",
                           "read_bandwidth_gbps",
                           "read_latency_ms",
                           "eviction_policy",
                           "capacity_pressure_policy",
                           "transfer_concurrency",
                       },
                       "config.cpu_kv_cache");
    CpuKVCacheConfig result{};
    result.enabled = require_bool(cpu, "enabled", "config.cpu_kv_cache");
    result.capacity_bytes =
        require_uint64(cpu, "capacity_bytes", "config.cpu_kv_cache");
    result.static_slice_per_gpu =
        require_bool(cpu, "static_slice_per_gpu", "config.cpu_kv_cache");
    result.capacity_bytes_per_gpu =
        require_uint64(cpu, "capacity_bytes_per_gpu", "config.cpu_kv_cache");
    result.dram_bandwidth_gbps_per_gpu = require_finite_number(
        cpu, "dram_bandwidth_gbps_per_gpu", "config.cpu_kv_cache");
    result.c2c_bandwidth_gbps_per_gpu = require_finite_number(
        cpu, "c2c_bandwidth_gbps_per_gpu", "config.cpu_kv_cache");
    result.write_bandwidth_gbps = require_finite_number(
        cpu, "write_bandwidth_gbps", "config.cpu_kv_cache");
    result.write_latency_ms =
        require_finite_number(cpu, "write_latency_ms", "config.cpu_kv_cache");
    result.read_bandwidth_gbps = require_finite_number(
        cpu, "read_bandwidth_gbps", "config.cpu_kv_cache");
    result.read_latency_ms =
        require_finite_number(cpu, "read_latency_ms", "config.cpu_kv_cache");

    const std::string eviction =
        require_string(cpu, "eviction_policy", "config.cpu_kv_cache");
    if (eviction != "session_lru_suffix") {
        throw ConfigError("config.cpu_kv_cache.eviction_policy must be "
                          "'session_lru_suffix'");
    }
    const std::string pressure =
        require_string(cpu, "capacity_pressure_policy", "config.cpu_kv_cache");
    if (pressure == "prefix_fit") {
        result.capacity_pressure_policy =
            CpuKVCacheCapacityPressurePolicy::kPrefixFit;
    } else if (pressure == "skip_offload") {
        result.capacity_pressure_policy =
            CpuKVCacheCapacityPressurePolicy::kSkipOffload;
    } else {
        throw ConfigError("config.cpu_kv_cache.capacity_pressure_policy must "
                          "be 'prefix_fit' or 'skip_offload'");
    }
    const std::string concurrency =
        require_string(cpu, "transfer_concurrency", "config.cpu_kv_cache");
    if (concurrency != "full_duplex_serialized") {
        throw ConfigError("config.cpu_kv_cache.transfer_concurrency must be "
                          "'full_duplex_serialized'");
    }
    if (result.capacity_bytes_per_gpu == 0 ||
        result.dram_bandwidth_gbps_per_gpu <= 0.0 ||
        result.c2c_bandwidth_gbps_per_gpu <= 0.0 ||
        result.write_bandwidth_gbps <= 0.0 ||
        result.read_bandwidth_gbps <= 0.0 || result.write_latency_ms < 0.0 ||
        result.read_latency_ms < 0.0) {
        throw ConfigError(
            "config.cpu_kv_cache requires positive capacities/bandwidths and "
            "nonnegative latencies");
    }
    if (result.enabled && !result.static_slice_per_gpu &&
        result.capacity_bytes == 0) {
        throw ConfigError("enabled direct CPU KV cache capacity must be "
                          "positive");
    }
    return result;
}

SchedulerConfig parse_scheduler(const Json &root) {
    const Json &scheduler = root.at("scheduler");
    require_keys(scheduler,
                 {
                     "type",
                     "scheduling_policy",
                     "batch_size_cap",
                     "max_tokens_in_batch",
                     "enable_preemption",
                     "enable_chunked_prefill",
                     "long_prefill_token_threshold",
                     "block_size",
                     "watermark_blocks_fraction",
                     "num_preallocate_tokens",
                 },
                 {"num_blocks"}, "config.scheduler");

    SchedulerConfig parsed = [&]() {
        SchedulerConfig value{};
        value.type = parse_scheduler_type(
            require_string(scheduler, "type", "config.scheduler"));
        value.scheduling_policy = parse_scheduling_policy(
            require_string(scheduler, "scheduling_policy", "config.scheduler"));
        value.batch_size_cap =
            require_uint64(scheduler, "batch_size_cap", "config.scheduler");
        value.max_tokens_in_batch = require_uint64(
            scheduler, "max_tokens_in_batch", "config.scheduler");
        value.enable_preemption =
            require_bool(scheduler, "enable_preemption", "config.scheduler");
        value.enable_chunked_prefill = require_bool(
            scheduler, "enable_chunked_prefill", "config.scheduler");
        value.long_prefill_token_threshold = require_uint64(
            scheduler, "long_prefill_token_threshold", "config.scheduler");
        value.block_size =
            require_uint64(scheduler, "block_size", "config.scheduler");
        if (scheduler.contains("num_blocks")) {
            value.num_blocks =
                require_uint64(scheduler, "num_blocks", "config.scheduler");
        }
        value.watermark_blocks_fraction = require_finite_number(
            scheduler, "watermark_blocks_fraction", "config.scheduler");
        value.num_preallocate_tokens = require_uint64(
            scheduler, "num_preallocate_tokens", "config.scheduler");
        return value;
    }();

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
    if (scheduler.contains("num_blocks") && parsed.num_blocks == 0) {
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

GpuMemoryConfig parse_gpu_memory(const Json &cluster,
                                 std::string_view context) {
    if (!cluster.contains("gpu_memory")) {
        throw ConfigError(std::string{context} +
                          ".gpu_memory is required; provide positive "
                          "capacity_bytes_per_gpu");
    }
    const Json &memory = cluster.at("gpu_memory");
    const std::string memory_context = std::string{context} + ".gpu_memory";
    require_keys(memory, {"capacity_bytes_per_gpu"},
                 {"auto_calculate_num_blocks", "runtime_reserve_fraction",
                  "runtime_reserve_bytes", "weight_overhead_fraction",
                  "model_weight_bytes_per_gpu", "kv_cache_budget_bytes_per_gpu",
                  "kv_cache_bytes_per_block"},
                 memory_context);
    GpuMemoryConfig parsed{};
    // A scheduler.num_blocks field denotes an explicit/manual capacity unless
    // the serialized gpu_memory object carries the explicit auto flag.  The
    // latter is emitted by the normalizer so round-trips preserve the mode.
    parsed.auto_calculate_num_blocks =
        memory.contains("auto_calculate_num_blocks")
            ? require_bool(memory, "auto_calculate_num_blocks", memory_context)
            : !cluster.at("scheduler").contains("num_blocks");
    parsed.capacity_bytes_per_gpu =
        require_uint64(memory, "capacity_bytes_per_gpu", memory_context);
    if (parsed.capacity_bytes_per_gpu == 0) {
        throw ConfigError(memory_context +
                          ".capacity_bytes_per_gpu must be positive");
    }
    parsed.runtime_reserve_fraction =
        memory.contains("runtime_reserve_fraction")
            ? require_finite_number(memory, "runtime_reserve_fraction",
                                    memory_context)
            : 0.1;
    if (memory.contains("runtime_reserve_bytes")) {
        parsed.runtime_reserve_bytes =
            require_uint64(memory, "runtime_reserve_bytes", memory_context);
    }
    if (memory.contains("weight_overhead_fraction")) {
        parsed.weight_overhead_fraction = require_finite_number(
            memory, "weight_overhead_fraction", memory_context);
    }
    return parsed;
}

ParallelismConfig parse_parallelism(const Json &root,
                                    const ModelConfig &model) {
    const Json &parallelism = root.at("parallelism");
    require_keys(parallelism,
                 {
                     "num_replicas",
                     "tensor_parallel_size",
                     "pipeline_parallel_size",
                     "data_parallel_size",
                     "moe_tensor_parallel_size",
                     "moe_expert_parallel_size",
                 },
                 {"decode_context_parallel_size", "pipeline_exclusive",
                  "pipeline_stage_layer_counts"},
                 "config.parallelism");
    ParallelismConfig parsed = [&]() {
        ParallelismConfig value{};
        value.num_replicas =
            require_uint64(parallelism, "num_replicas", "config.parallelism");
        value.tensor_parallel_size = require_uint64(
            parallelism, "tensor_parallel_size", "config.parallelism");
        if (parallelism.contains("decode_context_parallel_size")) {
            value.decode_context_parallel_size =
                require_uint64(parallelism, "decode_context_parallel_size",
                               "config.parallelism");
        }
        if (parallelism.contains("pipeline_exclusive")) {
            value.pipeline_exclusive = require_bool(
                parallelism, "pipeline_exclusive", "config.parallelism");
        }
        value.pipeline_parallel_size = require_uint64(
            parallelism, "pipeline_parallel_size", "config.parallelism");
        if (parallelism.contains("pipeline_stage_layer_counts")) {
            value.pipeline_stage_layer_counts =
                require_uint64_array(parallelism, "pipeline_stage_layer_counts",
                                     "config.parallelism");
        }
        value.data_parallel_size = require_uint64(
            parallelism, "data_parallel_size", "config.parallelism");
        value.moe_tensor_parallel_size = require_uint64(
            parallelism, "moe_tensor_parallel_size", "config.parallelism");
        value.moe_expert_parallel_size = require_uint64(
            parallelism, "moe_expert_parallel_size", "config.parallelism");
        return value;
    }();
    if (parsed.num_replicas == 0 || parsed.tensor_parallel_size == 0 ||
        parsed.decode_context_parallel_size == 0 ||
        parsed.pipeline_parallel_size == 0 || parsed.data_parallel_size == 0 ||
        parsed.moe_tensor_parallel_size == 0 ||
        parsed.moe_expert_parallel_size == 0) {
        throw ConfigError("all config.parallelism dimensions must be positive");
    }
    if (parsed.pipeline_exclusive) {
        if (parsed.pipeline_parallel_size <= 1) {
            throw ConfigError("config.parallelism.pipeline_exclusive requires "
                              "pipeline_parallel_size > 1");
        }
        if (parsed.data_parallel_size != 1) {
            throw ConfigError("config.parallelism.pipeline_exclusive requires "
                              "data_parallel_size == 1");
        }
        if (parsed.moe_expert_parallel_size != 1) {
            throw ConfigError("config.parallelism.pipeline_exclusive requires "
                              "moe_expert_parallel_size == 1");
        }
        if (parsed.moe_tensor_parallel_size != parsed.tensor_parallel_size) {
            throw ConfigError(
                "config.parallelism.pipeline_exclusive requires "
                "moe_tensor_parallel_size == tensor_parallel_size");
        }
    }
    if (parsed.tensor_parallel_size != 1 && parsed.tensor_parallel_size != 2 &&
        parsed.tensor_parallel_size != 4 && parsed.tensor_parallel_size != 8) {
        throw ConfigError("config.parallelism.tensor_parallel_size must be one "
                          "of 1, 2, 4, 8");
    }
    if (parsed.tensor_parallel_size % parsed.decode_context_parallel_size !=
        0) {
        throw ConfigError(
            "config.parallelism.tensor_parallel_size must be divisible by "
            "decode_context_parallel_size");
    }
    if (parsed.decode_context_parallel_size > 1 && !model.use_mla) {
        throw ConfigError(
            "config.parallelism.decode_context_parallel_size > 1 is "
            "currently supported only for MLA models");
    }
    if (parsed.pipeline_parallel_size > model.num_layers) {
        throw ConfigError("config.parallelism.pipeline_parallel_size must not "
                          "exceed model num_layers");
    }
    if (parallelism.contains("pipeline_stage_layer_counts")) {
        if (parsed.pipeline_stage_layer_counts.size() !=
            parsed.pipeline_parallel_size) {
            throw ConfigError(
                "config.parallelism.pipeline_stage_layer_counts must contain "
                "exactly pipeline_parallel_size entries");
        }
        std::uint64_t assigned_layers = 0;
        for (std::size_t stage = 0;
             stage < parsed.pipeline_stage_layer_counts.size(); ++stage) {
            const std::uint64_t count =
                parsed.pipeline_stage_layer_counts[stage];
            if (count == 0) {
                throw ConfigError(
                    "config.parallelism.pipeline_stage_layer_counts[" +
                    std::to_string(stage) + "] must be positive");
            }
            if (count > model.num_layers - assigned_layers) {
                throw ConfigError(
                    "config.parallelism.pipeline_stage_layer_counts must sum "
                    "to model num_layers");
            }
            assigned_layers += count;
        }
        if (assigned_layers != model.num_layers) {
            throw ConfigError(
                "config.parallelism.pipeline_stage_layer_counts must sum to "
                "model num_layers");
        }
    }
    if (model.hidden_size % parsed.tensor_parallel_size != 0 ||
        model.num_query_heads % parsed.tensor_parallel_size != 0) {
        throw ConfigError(
            "model hidden size and attention heads must be divisible by "
            "config.parallelism.tensor_parallel_size");
    }
    if (model.is_moe()) {
        if (parsed.attention_parallel_size() != parsed.moe_parallel_size()) {
            throw ConfigError(
                "MoE shared parallel domain requires "
                "tensor_parallel_size*data_parallel_size == "
                "moe_tensor_parallel_size*moe_expert_parallel_size");
        }
        if (model.total_expert_num % parsed.moe_expert_parallel_size != 0) {
            throw ConfigError("total_expert_num must be divisible by "
                              "moe_expert_parallel_size");
        }
        if (model.moe_intermediate_size % parsed.moe_tensor_parallel_size !=
            0) {
            throw ConfigError("MoE intermediate size must be divisible by "
                              "moe_tensor_parallel_size");
        }
    } else if (parsed.moe_tensor_parallel_size != 1 ||
               parsed.moe_expert_parallel_size != 1) {
        throw ConfigError("dense models require MoE TP=1 and EP=1");
    }
    const auto checked_multiply = [](std::uint64_t left, std::uint64_t right,
                                     std::string_view context) {
        if (right != 0 &&
            left > std::numeric_limits<std::uint64_t>::max() / right) {
            throw ConfigError(std::string{context} + " overflows uint64");
        }
        return left * right;
    };
    std::uint64_t accelerators =
        checked_multiply(parsed.num_replicas, parsed.data_parallel_size,
                         "config.parallelism accelerator count");
    accelerators = checked_multiply(accelerators, parsed.pipeline_parallel_size,
                                    "config.parallelism accelerator count");
    accelerators = checked_multiply(accelerators, parsed.tensor_parallel_size,
                                    "config.parallelism accelerator count");
    static_cast<void>(accelerators);
    return parsed;
}

ClusterSchedulerConfig parse_cluster_scheduler(const Json &root) {
    const Json &scheduler = root.at("cluster_scheduler");
    require_keys(scheduler, {"type"},
                 {"prefill_type", "decode_type", "cache_threshold",
                  "balance_abs_threshold", "balance_rel_threshold"},
                 "config.cluster_scheduler");
    return [&]() {
        ClusterSchedulerConfig value{};
        value.type = parse_cluster_scheduler_type(
            require_string(scheduler, "type", "config.cluster_scheduler"));
        if (scheduler.contains("prefill_type")) {
            value.prefill_type = parse_cluster_scheduler_type(require_string(
                scheduler, "prefill_type", "config.cluster_scheduler"));
        }
        if (scheduler.contains("decode_type")) {
            value.decode_type = parse_cluster_scheduler_type(require_string(
                scheduler, "decode_type", "config.cluster_scheduler"));
        }
        if (scheduler.contains("cache_threshold")) {
            value.cache_threshold = require_finite_number(
                scheduler, "cache_threshold", "config.cluster_scheduler");
        }
        if (scheduler.contains("balance_abs_threshold")) {
            value.balance_abs_threshold = require_uint64(
                scheduler, "balance_abs_threshold", "config.cluster_scheduler");
        }
        if (scheduler.contains("balance_rel_threshold")) {
            value.balance_rel_threshold = require_finite_number(
                scheduler, "balance_rel_threshold", "config.cluster_scheduler");
        }
        if (!std::isfinite(value.cache_threshold) ||
            value.cache_threshold < 0.0 || value.cache_threshold > 1.0) {
            throw ConfigError(
                "config.cluster_scheduler.cache_threshold must be finite "
                "and in [0, 1]");
        }
        if (!std::isfinite(value.balance_rel_threshold) ||
            value.balance_rel_threshold < 1.0) {
            throw ConfigError(
                "config.cluster_scheduler.balance_rel_threshold must be "
                "finite and >= 1");
        }
        return value;
    }();
}

} // namespace frontier::config::parse_detail
