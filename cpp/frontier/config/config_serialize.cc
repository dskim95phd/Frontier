#include "frontier/config/config.h"

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
    if (config.enable_parallel_clusters) {
        throw ConfigError(
            "config.enable_parallel_clusters=true is outside the C++ port");
    }
}

OrderedJson serialize_prefix_cache(const PrefixCacheConfig &config) {
    return OrderedJson::object({
        {"enabled", config.enabled},
        {"key_mode", to_string(config.key_mode)},
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
    return OrderedJson::object({
        {"num_replicas", parallelism.num_replicas},
        {"tensor_parallel_size", parallelism.tensor_parallel_size},
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
    });
}

OrderedJson serialize_moe_routing(const MoeRoutingConfig &routing) {
    return OrderedJson::object({
        {"mode", to_string(routing.mode)},
        {"distribution", to_string(routing.distribution)},
        {"seed", routing.seed},
    });
}

OrderedJson
serialize_cluster_scheduler(const ClusterSchedulerConfig &scheduler) {
    return OrderedJson::object({
        {"type", to_string(scheduler.type)},
    });
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
        add("dense", operators.dense);
        add("moe_expert", operators.moe_expert);
        add("moe_router", operators.moe_router);
        add("kv_cache", operators.kv_cache);
        add("communication", operators.communication);
        result["operator_precisions"] = std::move(values);
    }
    return result;
}

OrderedJson serialize_cluster_runtime(const ClusterRuntimeConfig &cluster) {
    return OrderedJson::object({
        {"parallelism", serialize_parallelism(cluster.parallelism)},
        {"scheduler", serialize_scheduler(cluster.scheduler)},
        {
            "execution_model",
            serialize_execution_model(cluster.execution_model),
        },
        {"model_name", cluster.model.name},
        {"total_expert_num", cluster.model.total_expert_num},
        {"router_topk", cluster.model.router_topk},
        {"moe_routing", serialize_moe_routing(cluster.moe_routing)},
    });
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
    root["enable_parallel_clusters"] = config.enable_parallel_clusters;
    root["prefix_cache"] = serialize_prefix_cache(config.prefix_cache);
    root["cluster_scheduler"] =
        serialize_cluster_scheduler(config.cluster_scheduler);

    if (config.system_architecture == SystemArchitecture::kPdDisaggregation) {
        if (config.system_architecture !=
                SystemArchitecture::kPdDisaggregation ||
            config.prefix_cache.enabled ||
            !std::holds_alternative<PddRuntimeConfig>(config.runtime)) {
            throw ConfigError(
                "PDD config requires sequential cluster and transfer configs");
        }
        const PddRuntimeConfig &runtime = config.pdd();
        root["clusters"] = serialize_pdd_clusters(runtime.clusters);
        root["kv_cache_transfer"] =
            serialize_kv_cache_transfer(runtime.kv_cache_transfer);
        return root.dump(2) + '\n';
    }
    if (!std::holds_alternative<ClusterRuntimeConfig>(config.runtime)) {
        throw ConfigError("co-location config requires a monolithic cluster");
    }
    if (config.system_architecture != SystemArchitecture::kCoLocation ||
        config.prefix_cache.enabled) {
        throw ConfigError(
            "co-location config requires prefix caching disabled");
    }
    root["clusters"] = serialize_single_cluster(config.cluster());
    return root.dump(2) + '\n';
}

} // namespace frontier::config
