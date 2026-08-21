#include "frontier/config/config.h"

#include <cmath>
#include <initializer_list>
#include <limits>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "frontier/config/config_parse_internal.h"
#include "frontier/core/precision.h"

namespace frontier::config {
namespace {

using Json = nlohmann::json;
using namespace parse_detail;

ClusterRuntimeConfig parse_cluster_runtime(const Json &clusters,
                                           std::string_view name) {
    const std::string context = "config.clusters." + std::string{name};
    if (!clusters.contains(name)) {
        throw ConfigError("config.clusters is missing required field '" +
                          std::string{name} + "'");
    }
    const Json &cluster = clusters.at(name);
    require_keys(cluster,
                 {
                     "parallelism",
                     "scheduler",
                     "execution_model",
                     "model_name",
                     "moe_routing",
                     "gpu_memory",
                 },
                 {"total_expert_num", "router_topk", "first_k_dense_replace",
                  "num_shared_experts"},
                 context);
    const ModelConfig model = parse_model(cluster, context);
    const ParallelismConfig parallelism = parse_parallelism(cluster, model);
    const GpuMemoryConfig gpu_memory = parse_gpu_memory(cluster, context);
    const std::optional<std::uint64_t> explicit_num_blocks =
        cluster.at("scheduler").contains("num_blocks")
            ? std::optional<std::uint64_t>{require_uint64(
                  cluster.at("scheduler"), "num_blocks", "config.scheduler")}
            : std::nullopt;
    return [&]() {
        ClusterRuntimeConfig value{};
        value.parallelism = parallelism;
        value.scheduler = parse_scheduler(cluster);
        value.execution_model =
            parse_execution_model(cluster, parallelism, model);
        value.gpu_memory = gpu_memory;
        value.model = model;
        value.moe_routing = parse_moe_routing(cluster);
        resolve_gpu_memory_config(value, explicit_num_blocks);
        return value;
    }();
}

PddClustersConfig parse_pdd_clusters(const Json &root) {
    const Json &clusters = root.at("clusters");
    require_exact_keys(clusters, {"prefill", "decode"}, "config.clusters");
    PddClustersConfig parsed = [&]() {
        PddClustersConfig value{};
        value.prefill = parse_cluster_runtime(clusters, "prefill");
        value.decode = parse_cluster_runtime(clusters, "decode");
        return value;
    }();
    if (parsed.prefill.parallelism.decode_context_parallel_size != 1) {
        throw ConfigError("config.clusters.prefill.parallelism."
                          "decode_context_parallel_size must be 1 for "
                          "pd-disaggregation");
    }
    if (parsed.prefill.model != parsed.decode.model) {
        throw ConfigError(
            "PDD PREFILL and DECODE must use the same model and expert "
            "contract");
    }
    return parsed;
}

KvCacheTransferConfig parse_kv_cache_transfer(const Json &root) {
    const Json &transfer = root.at("kv_cache_transfer");
    require_exact_keys(transfer,
                       {
                           "type",
                           "network_bandwidth_gbps",
                           "network_latency_ms",
                           "kv_cache_dtype_size_bytes",
                           "enable_compression",
                       },
                       "config.kv_cache_transfer");
    const std::string type =
        require_string(transfer, "type", "config.kv_cache_transfer");
    if (type != "analytical") {
        throw ConfigError("config.kv_cache_transfer.type must be 'analytical'");
    }
    KvCacheTransferConfig parsed = [&]() {
        KvCacheTransferConfig value{};
        value.network_bandwidth_gbps = require_finite_number(
            transfer, "network_bandwidth_gbps", "config.kv_cache_transfer");
        value.network_latency_ms = require_finite_number(
            transfer, "network_latency_ms", "config.kv_cache_transfer");
        value.kv_cache_dtype_size_bytes = require_finite_number(
            transfer, "kv_cache_dtype_size_bytes", "config.kv_cache_transfer");
        value.enable_compression = require_bool(transfer, "enable_compression",
                                                "config.kv_cache_transfer");
        return value;
    }();
    if (parsed.network_bandwidth_gbps <= 0.0 ||
        parsed.network_latency_ms < 0.0 ||
        parsed.kv_cache_dtype_size_bytes <= 0.0) {
        throw ConfigError(
            "config.kv_cache_transfer requires positive bandwidth/dtype and "
            "nonnegative latency");
    }
    if (parsed.enable_compression) {
        throw ConfigError("config.kv_cache_transfer.enable_compression=true is "
                          "outside Step 3");
    }
    return parsed;
}

std::optional<PrefillOnlyConfig> parse_prefill_only(const Json &root) {
    if (!root.contains("prefill_only")) {
        return std::nullopt;
    }
    const Json &prefill_only = root.at("prefill_only");
    require_exact_keys(prefill_only, {"decode_tokens_per_second"},
                       "config.prefill_only");
    PrefillOnlyConfig parsed{};
    parsed.decode_tokens_per_second = require_finite_number(
        prefill_only, "decode_tokens_per_second", "config.prefill_only");
    if (parsed.decode_tokens_per_second <= 0.0) {
        throw ConfigError(
            "config.prefill_only.decode_tokens_per_second must be positive");
    }
    return parsed;
}

struct CommonConfigFields {
    int schema_version;
    std::string run_id;
    SimulationMode simulation_mode;
    SystemArchitecture system_architecture;
    bool enable_parallel_clusters;
    PrefixCacheConfig prefix_cache;
    CpuKVCacheConfig cpu_kv_cache;
};

void require_schema_version(int schema_version) {
    if (schema_version == kSchemaVersion) {
        return;
    }
    throw ConfigError("unsupported config schema_version=" +
                      std::to_string(schema_version) + "; expected 1");
}

CommonConfigFields parse_common_fields(const Json &root) {
    if (!root.contains("schema_version")) {
        throw ConfigError("config is missing required field 'schema_version'");
    }
    const int schema_version = require_int(root, "schema_version", "config");
    require_schema_version(schema_version);

    std::string run_id = require_string(root, "run_id", "config");
    if (run_id.empty() || is_blank(run_id)) {
        throw ConfigError("config.run_id must not be empty");
    }
    const bool enable_parallel_clusters =
        require_bool(root, "enable_parallel_clusters", "config");
    if (enable_parallel_clusters) {
        throw ConfigError(
            "config.enable_parallel_clusters=true is outside the C++ port");
    }
    const SimulationMode simulation_mode = parse_simulation_mode(
        require_string(root, "simulation_mode", "config"));
    return [&]() {
        CommonConfigFields value{};
        value.schema_version = schema_version;
        value.run_id = std::move(run_id);
        value.simulation_mode = simulation_mode;
        value.system_architecture = parse_system_architecture(
            require_string(root, "system_architecture", "config"));
        value.enable_parallel_clusters = enable_parallel_clusters;
        value.prefix_cache = parse_prefix_cache(root);
        value.cpu_kv_cache = parse_cpu_kv_cache(root);
        return value;
    }();
}

void validate_kda_prefix_capacity(const ClusterRuntimeConfig &cluster,
                                  bool prefix_cache_enabled,
                                  std::string_view context) {
    if (!prefix_cache_enabled || !cluster.model.has_kda()) {
        return;
    }
    const std::uint64_t charge =
        cluster.scheduler.kda_snapshot_blocks_per_session;
    if (charge == 0 || charge > cluster.scheduler.num_blocks) {
        throw ConfigError(std::string{context} +
                          " GPU cache cannot hold one atomic KDA snapshot");
    }
}

SimulationConfig make_pdd_config(const Json &root, CommonConfigFields common) {
    require_keys(root,
                 {
                     "schema_version",
                     "run_id",
                     "simulation_mode",
                     "system_architecture",
                     "enable_parallel_clusters",
                     "prefix_cache",
                     "cluster_scheduler",
                     "clusters",
                     "kv_cache_transfer",
                 },
                 {"cpu_kv_cache", "prefill_only"}, "config");
    if (common.system_architecture != SystemArchitecture::kPdDisaggregation) {
        throw ConfigError(
            "PDD config requires system_architecture='pd-disaggregation'");
    }
    PddRuntimeConfig runtime{};
    runtime.clusters = parse_pdd_clusters(root);
    runtime.kv_cache_transfer = parse_kv_cache_transfer(root);
    const auto &prefill_execution = runtime.clusters.prefill.execution_model;
    const auto &decode_execution = runtime.clusters.decode.execution_model;
    if (prefill_execution.type == ExecutionModelType::kAnalytical &&
        decode_execution.type == ExecutionModelType::kAnalytical) {
        const std::string &prefill_kv =
            prefill_execution.analytical.kv_cache_precision();
        const std::string &decode_kv =
            decode_execution.analytical.kv_cache_precision();
        if (prefill_kv != decode_kv) {
            throw ConfigError(
                "PDD PREFILL and DECODE must use the same KV-cache precision");
        }
        const double expected_bytes =
            analytical_precision_size_bytes(prefill_kv);
        if (runtime.kv_cache_transfer.kv_cache_dtype_size_bytes !=
            expected_bytes) {
            throw ConfigError(
                "config.kv_cache_transfer.kv_cache_dtype_size_bytes must "
                "match the analytical KV-cache precision");
        }
    }
    // KDA recurrent-state snapshots are part of the PDD payload too. Unlike
    // KV cache bytes, the snapshot dtype is an operator precision and must
    // match exactly (e.g. BF16 and FP16 are not interchangeable despite both
    // occupying two bytes). Fixed execution follows the native BF16 snapshot;
    // mixed analytical/fixed PDD is accepted only for analytical BF16.
    static_cast<void>(
        resolve_pdd_kda_snapshot_dtype_size_bytes(runtime.clusters));
    return [&]() {
        SimulationConfig value{};
        value.schema_version = common.schema_version;
        value.run_id = std::move(common.run_id);
        value.simulation_mode = common.simulation_mode;
        value.system_architecture = common.system_architecture;
        value.enable_parallel_clusters = common.enable_parallel_clusters;
        value.prefix_cache = common.prefix_cache;
        value.cpu_kv_cache = common.cpu_kv_cache;
        value.cluster_scheduler = parse_cluster_scheduler(root);
        value.runtime = std::move(runtime);
        value.prefill_only = parse_prefill_only(root);
        validate_kda_prefix_capacity(value.pdd().clusters.prefill,
                                     value.prefix_cache.enabled,
                                     "config.clusters.prefill");
        if (value.cpu_kv_cache.enabled) {
            if (!value.prefix_cache.enabled) {
                throw ConfigError(
                    "CPU KV cache requires prefix_cache.enabled=true");
            }
            const ClusterSchedulerType prefill_scheduler =
                value.cluster_scheduler.type_for_cluster(ClusterType::kPrefill);
            if (prefill_scheduler != ClusterSchedulerType::kStickyRoundRobin &&
                prefill_scheduler != ClusterSchedulerType::kCacheAware) {
                throw ConfigError("CPU KV cache requires "
                                  "PREFILL cluster scheduler to be "
                                  "'sticky_round_robin' or 'cache_aware'");
            }
            static_cast<void>(resolve_cpu_kv_cache_target(value));
        }
        return value;
    }();
}

SimulationConfig make_single_cluster_config(const Json &root,
                                            CommonConfigFields common) {
    require_keys(root,
                 {
                     "schema_version",
                     "run_id",
                     "simulation_mode",
                     "system_architecture",
                     "enable_parallel_clusters",
                     "prefix_cache",
                     "cluster_scheduler",
                     "clusters",
                 },
                 {"cpu_kv_cache", "prefill_only"}, "config");
    if (common.system_architecture != SystemArchitecture::kCoLocation) {
        throw ConfigError("single-cluster config requires "
                          "system_architecture='co-location'");
    }
    if (root.contains("prefill_only")) {
        throw ConfigError(
            "config.prefill_only is supported only for pd-disaggregation");
    }
    const Json &clusters = root.at("clusters");
    require_exact_keys(clusters, {"monolithic"}, "config.clusters");

    return [&]() {
        SimulationConfig value{};
        value.schema_version = common.schema_version;
        value.run_id = std::move(common.run_id);
        value.simulation_mode = common.simulation_mode;
        value.system_architecture = common.system_architecture;
        value.enable_parallel_clusters = common.enable_parallel_clusters;
        value.prefix_cache = common.prefix_cache;
        value.cpu_kv_cache = common.cpu_kv_cache;
        value.cluster_scheduler = parse_cluster_scheduler(root);
        value.runtime = parse_cluster_runtime(clusters, "monolithic");
        validate_kda_prefix_capacity(value.cluster(),
                                     value.prefix_cache.enabled,
                                     "config.clusters.monolithic");
        if (value.cpu_kv_cache.enabled) {
            throw ConfigError(
                "CPU KV cache is supported only for sequential PDD");
        }
        return value;
    }();
}

} // namespace

SimulationConfig parse_simulation_config_json(std::string_view json_text) {
    Json root;
    try {
        root = Json::parse(json_text);
    } catch (const Json::parse_error &error) {
        throw ConfigError("invalid config JSON: " + std::string{error.what()});
    }
    require_object(root, "config");
    CommonConfigFields common = parse_common_fields(root);
    if (common.system_architecture == SystemArchitecture::kPdDisaggregation) {
        return make_pdd_config(root, std::move(common));
    }
    return make_single_cluster_config(root, std::move(common));
}

} // namespace frontier::config
