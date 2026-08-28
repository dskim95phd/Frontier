#include "frontier/config/config.h"

#include <cmath>
#include <string>

#include <nlohmann/json.hpp>

namespace frontier::config {
namespace {

using OrderedJson = nlohmann::ordered_json;

bool is_blank(std::string_view value) {
    return value.find_first_not_of(" \t\r\n") == std::string_view::npos;
}

void validate_common_for_serialization(const SimulationConfig &config) {
    if (config.schema_version != kSchemaVersion) {
        throw ConfigError(
            "cannot serialize unsupported config schema_version=" +
            std::to_string(config.schema_version));
    }
    if (config.run_id.empty() || is_blank(config.run_id)) {
        throw ConfigError("config.run_id must not be empty");
    }
    if (config.prefill_only.has_value() &&
        config.system_architecture != SystemArchitecture::kPdDisaggregation) {
        throw ConfigError(
            "config.prefill_only is supported only for pd-disaggregation");
    }
    if (config.prefill_only.has_value() &&
        (!std::isfinite(config.prefill_only->decode_tokens_per_second) ||
         config.prefill_only->decode_tokens_per_second <= 0.0)) {
        throw ConfigError(
            "config.prefill_only.decode_tokens_per_second must be finite "
            "and positive");
    }
}

OrderedJson serialize_prefix_cache(const PrefixCacheConfig &config) {
    return OrderedJson::object({
        {"enabled", config.enabled},
        {"key_mode", to_string(config.key_mode)},
    });
}

OrderedJson serialize_cpu_kv_cache(const CpuKVCacheConfig &config) {
    return OrderedJson::object({
        {"enabled", config.enabled},
        {"capacity_bytes", config.capacity_bytes},
        {"static_slice_per_gpu", config.static_slice_per_gpu},
        {"capacity_bytes_per_gpu", config.capacity_bytes_per_gpu},
        {"dram_bandwidth_gbps_per_gpu", config.dram_bandwidth_gbps_per_gpu},
        {"c2c_bandwidth_gbps_per_gpu", config.c2c_bandwidth_gbps_per_gpu},
        {"write_bandwidth_gbps", config.write_bandwidth_gbps},
        {"write_latency_ms", config.write_latency_ms},
        {"read_bandwidth_gbps", config.read_bandwidth_gbps},
        {"read_latency_ms", config.read_latency_ms},
        {"eviction_policy", to_string(config.eviction_policy)},
        {"capacity_pressure_policy",
         to_string(config.capacity_pressure_policy)},
        {"transfer_concurrency", to_string(config.transfer_concurrency)},
    });
}

OrderedJson serialize_scheduler(const SchedulerConfig &scheduler) {
    return OrderedJson::object({
        {"type", to_string(scheduler.type)},
        {"scheduling_policy", to_string(scheduler.scheduling_policy)},
        {"batch_size_cap", scheduler.batch_size_cap},
        {"max_tokens_in_batch", scheduler.max_tokens_in_batch},
        {"enable_preemption", scheduler.enable_preemption},
        {"enable_chunked_prefill", scheduler.enable_chunked_prefill},
        {
            "long_prefill_token_threshold",
            scheduler.long_prefill_token_threshold,
        },
        {"block_size", scheduler.block_size},
        {"num_blocks", scheduler.num_blocks},
        {
            "watermark_blocks_fraction",
            scheduler.watermark_blocks_fraction,
        },
        {"num_preallocate_tokens", scheduler.num_preallocate_tokens},
    });
}

OrderedJson serialize_parallelism(const ParallelismConfig &parallelism) {
    OrderedJson serialized = OrderedJson::object({
        {"num_replicas", parallelism.num_replicas},
        {"tensor_parallel_size", parallelism.tensor_parallel_size},
        {
            "decode_context_parallel_size",
            parallelism.decode_context_parallel_size,
        },
        {"pipeline_parallel_size", parallelism.pipeline_parallel_size},
        {"data_parallel_size", parallelism.data_parallel_size},
        {
            "moe_tensor_parallel_size",
            parallelism.moe_tensor_parallel_size,
        },
        {
            "moe_expert_parallel_size",
            parallelism.moe_expert_parallel_size,
        },
        {"pipeline_exclusive", parallelism.pipeline_exclusive},
    });
    if (!parallelism.pipeline_stage_layer_counts.empty()) {
        serialized["pipeline_stage_layer_counts"] =
            parallelism.pipeline_stage_layer_counts;
    }
    return serialized;
}

OrderedJson serialize_moe_routing(const MoeRoutingConfig &routing) {
    return OrderedJson::object({
        {"mode", to_string(routing.mode)},
        {"distribution", to_string(routing.distribution)},
        {"seed", routing.seed},
        {"layer_scope", to_string(routing.layer_scope)},
    });
}

OrderedJson
serialize_cluster_scheduler(const ClusterSchedulerConfig &scheduler) {
    OrderedJson result = OrderedJson::object({
        {"type", to_string(scheduler.type)},
        {"cache_threshold", scheduler.cache_threshold},
        {"balance_abs_threshold", scheduler.balance_abs_threshold},
        {"balance_rel_threshold", scheduler.balance_rel_threshold},
    });
    if (scheduler.prefill_type.has_value()) {
        result["prefill_type"] = to_string(*scheduler.prefill_type);
    }
    if (scheduler.decode_type.has_value()) {
        result["decode_type"] = to_string(*scheduler.decode_type);
    }
    return result;
}

OrderedJson serialize_execution_model(const ExecutionModelConfig &execution) {
    if (execution.type == ExecutionModelType::kFixed) {
        return OrderedJson::object({
            {"type", to_string(execution.type)},
            {"stage_latencies_ms", execution.fixed.stage_latencies_ms},
        });
    }
    OrderedJson result = OrderedJson::object({
        {"type", to_string(execution.type)},
        {"device", execution.analytical.device},
        {"precision", execution.analytical.precision},
        {"moe_layer_event_mode", execution.analytical.moe_layer_event_mode},
        {"moe_communication_backend",
         execution.analytical.moe_communication_backend},
        {
            "network_bandwidth_gbps",
            execution.analytical.network_bandwidth_gbps,
        },
        {"network_latency_us", execution.analytical.network_latency_us},
        {
            "intra_node_bandwidth_gbps",
            execution.analytical.intra_node_bandwidth_gbps,
        },
    });
    const AnalyticalDeviceOverrides &device_overrides =
        execution.analytical.device_overrides;
    if (!device_overrides.empty()) {
        OrderedJson values = OrderedJson::object();
        const auto add = [&values](std::string_view name,
                                   const std::optional<double> &value) {
            if (value.has_value()) {
                values[std::string{name}] = value.value();
            }
        };
        add("hbm_bandwidth_tbps", device_overrides.hbm_bandwidth_tbps);
        add("fp32_tflops", device_overrides.fp32_tflops);
        add("fp16_tflops", device_overrides.fp16_tflops);
        add("fp8_tflops", device_overrides.fp8_tflops);
        add("fp4_tflops", device_overrides.fp4_tflops);
        result["device_overrides"] = std::move(values);
    }
    const OperatorPrecisionConfig &operators =
        execution.analytical.operator_precisions;
    if (!operators.empty()) {
        OrderedJson values = OrderedJson::object();
        const auto add = [&values](std::string_view name,
                                   const std::string &precision) {
            if (!precision.empty()) {
                values[std::string{name}] = precision;
            }
        };
        add("attention", operators.attention);
        add("attention_core", operators.attention_core);
        add("dense", operators.dense);
        add("moe_expert", operators.moe_expert);
        add("moe_router", operators.moe_router);
        add("kv_cache", operators.kv_cache);
        add("communication", operators.communication);
        add("attention_weight", operators.attention_weight);
        add("attention_activation", operators.attention_activation);
        add("dense_weight", operators.dense_weight);
        add("dense_activation", operators.dense_activation);
        add("moe_expert_weight", operators.moe_expert_weight);
        add("moe_expert_activation", operators.moe_expert_activation);
        add("moe_router_weight", operators.moe_router_weight);
        add("moe_router_activation", operators.moe_router_activation);
        add("router_weight_storage", operators.router_weight_storage);
        add("lm_head", operators.lm_head);
        add("lm_head_weight", operators.lm_head_weight);
        add("lm_head_activation", operators.lm_head_activation);
        add("routed_expert_weight", operators.routed_expert_weight);
        add("routed_expert_activation", operators.routed_expert_activation);
        add("latent_moe_projection_weight",
            operators.latent_moe_projection_weight);
        add("latent_moe_projection_activation",
            operators.latent_moe_projection_activation);
        add("shared_expert_weight", operators.shared_expert_weight);
        add("shared_expert_activation", operators.shared_expert_activation);
        add("dense_mlp_weight", operators.dense_mlp_weight);
        add("dense_mlp_activation", operators.dense_mlp_activation);
        add("router_compute", operators.router_compute);
        add("kda_snapshot", operators.kda_snapshot);
        result["operator_precisions"] = std::move(values);
    }
    if (execution.analytical.kernel_profile != "generic") {
        result["kernel_profile"] = execution.analytical.kernel_profile;
    }
    if (execution.analytical.mega_moe_tail_io_fraction.has_value()) {
        result["mega_moe_tail_io_fraction"] =
            *execution.analytical.mega_moe_tail_io_fraction;
    }
    if (execution.analytical.mega_moe_wave_exposure.has_value()) {
        result["mega_moe_wave_exposure"] =
            *execution.analytical.mega_moe_wave_exposure;
    }
    return result;
}

OrderedJson serialize_gpu_memory(const GpuMemoryConfig &memory) {
    return OrderedJson::object({
        {"auto_calculate_num_blocks", memory.auto_calculate_num_blocks},
        {"capacity_bytes_per_gpu", memory.capacity_bytes_per_gpu},
        {"runtime_reserve_fraction", memory.runtime_reserve_fraction},
        {"runtime_reserve_bytes", memory.runtime_reserve_bytes},
        {"weight_overhead_fraction", memory.weight_overhead_fraction},
        {"model_weight_bytes_per_gpu", memory.model_weight_bytes_per_gpu},
        {"kv_cache_budget_bytes_per_gpu", memory.kv_cache_budget_bytes_per_gpu},
        {"kv_cache_bytes_per_block", memory.kv_cache_bytes_per_block},
    });
}

OrderedJson serialize_cluster_runtime(const ClusterRuntimeConfig &cluster) {
    if (cluster.gpu_memory.capacity_bytes_per_gpu == 0) {
        throw ConfigError(
            "gpu_memory.capacity_bytes_per_gpu must be positive before "
            "serialization");
    }
    OrderedJson result = OrderedJson::object({
        {"parallelism", serialize_parallelism(cluster.parallelism)},
        {"scheduler", serialize_scheduler(cluster.scheduler)},
        {
            "execution_model",
            serialize_execution_model(cluster.execution_model),
        },
        {"model_name", cluster.model.name},
        {"total_expert_num", cluster.model.total_expert_num},
        {"router_topk", cluster.model.router_topk},
        {"first_k_dense_replace", cluster.model.first_k_dense_replace},
        {"num_shared_experts", cluster.model.num_shared_experts},
        {"moe_routing", serialize_moe_routing(cluster.moe_routing)},
    });
    // HBM capacity is part of every runtime cluster contract, including
    // fixed/manual scheduler configurations.  Keep the explicit auto flag so
    // normalized JSON can round-trip manual num_blocks without ambiguity.
    result["gpu_memory"] = serialize_gpu_memory(cluster.gpu_memory);
    return result;
}

OrderedJson serialize_pdd_clusters(const PddClustersConfig &clusters) {
    return OrderedJson::object({
        {"prefill", serialize_cluster_runtime(clusters.prefill)},
        {"decode", serialize_cluster_runtime(clusters.decode)},
    });
}

OrderedJson serialize_single_cluster(const ClusterRuntimeConfig &cluster) {
    return OrderedJson::object({
        {"monolithic", serialize_cluster_runtime(cluster)},
    });
}

OrderedJson serialize_kv_cache_transfer(const KvCacheTransferConfig &transfer) {
    return OrderedJson::object({
        {"type", "analytical"},
        {"network_bandwidth_gbps", transfer.network_bandwidth_gbps},
        {"network_latency_ms", transfer.network_latency_ms},
        {
            "kv_cache_dtype_size_bytes",
            transfer.kv_cache_dtype_size_bytes,
        },
        {"enable_compression", transfer.enable_compression},
    });
}

} // namespace

std::string serialize_simulation_config_json(const SimulationConfig &config) {
    validate_common_for_serialization(config);

    OrderedJson root = OrderedJson::object();
    root["schema_version"] = config.schema_version;
    root["run_id"] = config.run_id;
    root["simulation_mode"] = to_string(config.simulation_mode);
    root["system_architecture"] = to_string(config.system_architecture);
    root["prefix_cache"] = serialize_prefix_cache(config.prefix_cache);
    root["cpu_kv_cache"] = serialize_cpu_kv_cache(config.cpu_kv_cache);
    root["cluster_scheduler"] =
        serialize_cluster_scheduler(config.cluster_scheduler);

    if (config.system_architecture == SystemArchitecture::kPdDisaggregation) {
        if (config.system_architecture !=
                SystemArchitecture::kPdDisaggregation ||
            !std::holds_alternative<PddRuntimeConfig>(config.runtime)) {
            throw ConfigError(
                "PDD config requires sequential cluster and transfer configs");
        }
        const PddRuntimeConfig &runtime = config.pdd();
        root["clusters"] = serialize_pdd_clusters(runtime.clusters);
        root["kv_cache_transfer"] =
            serialize_kv_cache_transfer(runtime.kv_cache_transfer);
        if (config.prefill_only.has_value()) {
            root["prefill_only"] = OrderedJson::object({
                {"decode_tokens_per_second",
                 config.prefill_only->decode_tokens_per_second},
            });
        }
        return root.dump(2) + '\n';
    }
    if (!std::holds_alternative<ClusterRuntimeConfig>(config.runtime)) {
        throw ConfigError("co-location config requires a monolithic cluster");
    }
    if (config.system_architecture != SystemArchitecture::kCoLocation) {
        throw ConfigError("co-location config requires a monolithic cluster");
    }
    root["clusters"] = serialize_single_cluster(config.cluster());
    return root.dump(2) + '\n';
}

} // namespace frontier::config
