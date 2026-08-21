#include "frontier/config/config.h"

#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "frontier/config/config_parse_internal.h"
#include "frontier/core/precision.h"
#include "frontier/core/runtime_paths.h"

namespace frontier::config {
namespace {

using Json = nlohmann::json;
using namespace parse_detail;

constexpr int kModularSchemaVersion = 2;
constexpr int kAssetSchemaVersion = 1;

Json read_json_file(const std::filesystem::path &path,
                    std::string_view description) {
    std::ifstream input{path, std::ios::binary};
    if (!input) {
        throw ConfigError("failed to open " + std::string{description} +
                          ": " + path.string());
    }
    try {
        return Json::parse(input);
    } catch (const Json::exception &error) {
        throw ConfigError("failed to parse " + std::string{description} +
                          " " + path.string() + ": " + error.what());
    }
}

bool valid_asset_name(std::string_view name) {
    if (name.empty() || name == "." || name == "..") {
        return false;
    }
    for (const unsigned char ch : name) {
        if (!std::isalnum(ch) && ch != '-' && ch != '_' && ch != '.') {
            return false;
        }
    }
    return true;
}

void append_asset_root(std::vector<std::filesystem::path> &roots,
                       std::filesystem::path root) {
    if (root.empty()) {
        return;
    }
    for (const auto &existing : roots) {
        if (existing == root) {
            return;
        }
    }
    roots.push_back(std::move(root));
}

std::vector<std::filesystem::path>
asset_roots(const std::filesystem::path &scenario_path) {
    std::vector<std::filesystem::path> roots;
    const std::filesystem::path scenario_dir =
        std::filesystem::absolute(scenario_path).parent_path();
    append_asset_root(roots, scenario_dir / "assets");
    append_asset_root(roots, scenario_dir);
    append_asset_root(roots, scenario_dir.parent_path());

    if (const char *configured = std::getenv("FRONTIER_CONFIG_ASSET_DIR");
        configured != nullptr && *configured != '\0') {
        append_asset_root(roots, configured);
    }
    if (const auto executable_dir = core::executable_directory();
        executable_dir.has_value()) {
        append_asset_root(roots, *executable_dir / ".." / "share" /
                                     "frontier");
        append_asset_root(roots,
                          *executable_dir / "share" / "frontier");
        append_asset_root(roots,
                          *executable_dir / ".." / "data" / "config");
    }
    append_asset_root(roots, "data/config");
    append_asset_root(roots, "../data/config");
    append_asset_root(roots, "../../data/config");
    return roots;
}

struct AssetDocument {
    Json json;
    std::filesystem::path path;
};

AssetDocument load_asset(const std::vector<std::filesystem::path> &roots,
                         std::string_view kind, std::string_view name) {
    if (!valid_asset_name(name)) {
        throw ConfigError("invalid " + std::string{kind} + " asset name '" +
                          std::string{name} + "'");
    }
    std::vector<std::filesystem::path> searched;
    for (const auto &root : roots) {
        const std::filesystem::path candidate =
            root / kind / (std::string{name} + ".json");
        searched.push_back(candidate);
        if (std::filesystem::is_regular_file(candidate)) {
            AssetDocument result{read_json_file(candidate, "config asset"),
                                 candidate};
            require_object(result.json,
                           std::string{kind} + " asset " + candidate.string());
            // The kind-specific parser validates identity and remaining keys.
            return result;
        }
    }
    std::ostringstream message;
    message << "unknown " << kind << " asset '" << name << "' (searched";
    for (const auto &path : searched) {
        message << ' ' << path.string();
    }
    message << ')';
    throw ConfigError(message.str());
}

void validate_asset_identity(const Json &asset,
                             const std::filesystem::path &path,
                             std::string_view expected_name,
                             std::string_view kind) {
    const std::string context =
        std::string{kind} + " asset " + path.string();
    const int version =
        require_int(asset, "asset_schema_version", context);
    if (version != kAssetSchemaVersion) {
        throw ConfigError(context + " has unsupported asset_schema_version=" +
                          std::to_string(version) + "; expected 1");
    }
    const std::string actual_name = require_string(asset, "name", context);
    if (actual_name != expected_name) {
        throw ConfigError(context + ".name is '" + actual_name +
                          "', expected referenced name '" +
                          std::string{expected_name} + "'");
    }
}

struct GpuAsset {
    Json memory;
    std::optional<Json> analytical;
};

GpuAsset parse_gpu_asset(const AssetDocument &document,
                         std::string_view expected_name) {
    const std::string context = "gpu asset " + document.path.string();
    require_keys(document.json,
                 {"asset_schema_version", "name", "memory"},
                 {"analytical"}, context);
    validate_asset_identity(document.json, document.path, expected_name,
                            "gpu");

    const Json &memory = document.json.at("memory");
    require_keys(memory, {"capacity_bytes_per_gpu"},
                 {"auto_calculate_num_blocks", "runtime_reserve_fraction",
                  "runtime_reserve_bytes", "weight_overhead_fraction"},
                 context + ".memory");
    if (require_uint64(memory, "capacity_bytes_per_gpu",
                       context + ".memory") == 0) {
        throw ConfigError(context +
                          ".memory.capacity_bytes_per_gpu must be positive");
    }

    GpuAsset result{memory, std::nullopt};
    if (document.json.contains("analytical")) {
        const Json &analytical = document.json.at("analytical");
        require_keys(analytical, {"device"}, {"device_overrides"},
                     context + ".analytical");
        const std::string device =
            require_string(analytical, "device", context + ".analytical");
        if (device != "rubin" && device != "gb300" && device != "custom") {
            throw ConfigError(context +
                              ".analytical.device must be 'rubin', 'gb300', "
                              "or 'custom'");
        }
        if (analytical.contains("device_overrides")) {
            const Json &overrides = analytical.at("device_overrides");
            require_keys(overrides, {},
                         {"hbm_bandwidth_tbps", "fp32_tflops",
                          "fp16_tflops", "fp8_tflops", "fp4_tflops"},
                         context + ".analytical.device_overrides");
            for (const std::string_view field :
                 {"hbm_bandwidth_tbps", "fp32_tflops", "fp16_tflops",
                  "fp8_tflops", "fp4_tflops"}) {
                if (overrides.contains(field) &&
                    require_finite_number(
                        overrides, field,
                        context + ".analytical.device_overrides") <= 0.0) {
                    throw ConfigError(
                        context + ".analytical.device_overrides." +
                        std::string{field} + " must be positive");
                }
            }
            if (device == "custom" && overrides.size() != 5) {
                throw ConfigError(context +
                                  ".analytical custom device requires all "
                                  "five device_overrides fields");
            }
        } else if (device == "custom") {
            throw ConfigError(context +
                              ".analytical custom device requires "
                              "device_overrides");
        }
        result.analytical = analytical;
    }
    return result;
}

struct ClusterAsset {
    std::uint64_t gpu_count;
    Json gpu_memory;
    std::optional<Json> gpu_analytical;
    Json network;
};

ClusterAsset parse_cluster_asset(
    const AssetDocument &document, std::string_view expected_name,
    const std::vector<std::filesystem::path> &roots) {
    const std::string context = "cluster asset " + document.path.string();
    require_keys(document.json,
                 {"asset_schema_version", "name", "gpu", "network"},
                 {"gpu_memory"}, context);
    validate_asset_identity(document.json, document.path, expected_name,
                            "cluster");
    const Json &gpu_ref = document.json.at("gpu");
    require_exact_keys(gpu_ref, {"profile", "count"}, context + ".gpu");
    const std::string gpu_name =
        require_string(gpu_ref, "profile", context + ".gpu");
    const std::uint64_t gpu_count =
        require_uint64(gpu_ref, "count", context + ".gpu");
    if (gpu_count == 0) {
        throw ConfigError(context + ".gpu.count must be positive");
    }

    const Json &network = document.json.at("network");
    require_exact_keys(network,
                       {"network_bandwidth_gbps", "network_latency_us",
                        "intra_node_bandwidth_gbps"},
                       context + ".network");
    if (require_finite_number(network, "network_bandwidth_gbps",
                              context + ".network") <= 0.0 ||
        require_finite_number(network, "network_latency_us",
                              context + ".network") < 0.0 ||
        require_finite_number(network, "intra_node_bandwidth_gbps",
                              context + ".network") <= 0.0) {
        throw ConfigError(context +
                          ".network requires positive bandwidths and "
                          "nonnegative latency");
    }
    const AssetDocument gpu_document = load_asset(roots, "gpus", gpu_name);
    GpuAsset gpu = parse_gpu_asset(gpu_document, gpu_name);
    if (document.json.contains("gpu_memory")) {
        const Json &overrides = document.json.at("gpu_memory");
        require_keys(overrides, {},
                     {"auto_calculate_num_blocks", "runtime_reserve_fraction",
                      "runtime_reserve_bytes", "weight_overhead_fraction"},
                     context + ".gpu_memory");
        for (const auto &[field, value] : overrides.items()) {
            gpu.memory[field] = value;
        }
    }
    return ClusterAsset{gpu_count, std::move(gpu.memory),
                        std::move(gpu.analytical), network};
}

void validate_precision_field(const Json &object, std::string_view field,
                              std::string_view context) {
    const std::string value = require_string(object, field, context);
    if (!parse_precision(value).has_value()) {
        throw ConfigError(std::string{context} + "." + std::string{field} +
                          " has unsupported precision '" + value + "'");
    }
}

Json parse_precision_profile(const AssetDocument &document,
                             std::string_view expected_name) {
    const std::string context =
        "precision profile asset " + document.path.string();
    require_keys(document.json,
                 {"asset_schema_version", "name", "precision"},
                 {"operator_precisions"}, context);
    validate_asset_identity(document.json, document.path, expected_name,
                            "precision profile");
    validate_precision_field(document.json, "precision", context);

    Json result{{"precision", document.json.at("precision")}};
    if (!document.json.contains("operator_precisions")) {
        return result;
    }
    const Json &operators = document.json.at("operator_precisions");
    require_keys(
        operators, {},
        {"attention", "dense", "moe_expert", "moe_router", "kv_cache",
         "communication", "attention_weight", "attention_activation",
         "dense_weight", "dense_activation", "moe_expert_weight",
         "moe_expert_activation", "moe_router_weight",
         "moe_router_activation", "router_weight_storage", "lm_head",
         "lm_head_weight", "lm_head_activation", "routed_expert_weight",
         "routed_expert_activation", "latent_moe_projection_weight",
         "latent_moe_projection_activation", "shared_expert_weight",
         "shared_expert_activation", "dense_mlp_weight",
         "dense_mlp_activation", "router_compute", "kda_snapshot"},
        context + ".operator_precisions");
    for (const auto &item : operators.items()) {
        validate_precision_field(operators, item.key(),
                                 context + ".operator_precisions");
    }
    result["operator_precisions"] = operators;
    return result;
}

Json compose_analytical_execution(
    const Json &input, std::string_view context,
    const std::vector<std::filesystem::path> &roots) {
    require_keys(
        input, {"type"},
        {"precision_profile", "precision", "operator_precisions",
         "kernel_profile", "moe_layer_event_mode",
         "moe_communication_backend", "mega_moe_tail_io_fraction",
         "mega_moe_wave_exposure", "mega_moe_cluster_task_latency_us"},
        context);

    Json result{{"type", "analytical"}};
    if (input.contains("precision_profile")) {
        const std::string profile =
            require_string(input, "precision_profile", context);
        const AssetDocument document =
            load_asset(roots, "precision_profiles", profile);
        const Json defaults = parse_precision_profile(document, profile);
        result["precision"] = defaults.at("precision");
        if (defaults.contains("operator_precisions")) {
            result["operator_precisions"] =
                defaults.at("operator_precisions");
        }
    }

    for (const auto &[field, value] : input.items()) {
        if (field == "precision_profile" || field == "operator_precisions") {
            continue;
        }
        result[field] = value;
    }
    if (input.contains("operator_precisions")) {
        const Json &overrides = input.at("operator_precisions");
        require_object(overrides, std::string{context} +
                                       ".operator_precisions");
        Json merged = result.contains("operator_precisions")
                          ? result.at("operator_precisions")
                          : Json::object();
        for (const auto &[field, value] : overrides.items()) {
            merged[field] = value;
        }
        result["operator_precisions"] = std::move(merged);
    }
    if (!result.contains("precision")) {
        throw ConfigError(std::string{context} +
                          " requires precision or precision_profile");
    }
    return result;
}

std::uint64_t checked_multiply(std::uint64_t left, std::uint64_t right,
                               std::string_view context) {
    if (right != 0 &&
        left > std::numeric_limits<std::uint64_t>::max() / right) {
        throw ConfigError(std::string{context} + " overflows uint64");
    }
    return left * right;
}

std::uint64_t required_gpu_count(const Json &parallelism,
                                 std::string_view context) {
    require_keys(parallelism,
                 {"num_replicas", "tensor_parallel_size",
                  "pipeline_parallel_size", "data_parallel_size",
                  "moe_tensor_parallel_size", "moe_expert_parallel_size"},
                 {"decode_context_parallel_size", "pipeline_exclusive",
                  "pipeline_stage_layer_counts"},
                 context);
    std::uint64_t count = require_uint64(parallelism, "num_replicas", context);
    count = checked_multiply(
        count, require_uint64(parallelism, "data_parallel_size", context),
        context);
    count = checked_multiply(
        count, require_uint64(parallelism, "pipeline_parallel_size", context),
        context);
    count = checked_multiply(
        count, require_uint64(parallelism, "tensor_parallel_size", context),
        context);
    return count;
}

Json default_moe_routing() {
    return Json{{"mode", "simulation"},
                {"distribution", "balanced"},
                {"seed", 42}};
}

Json compose_cluster_runtime(
    const Json &scenario_cluster, std::string_view role,
    std::string_view model_name,
    const std::vector<std::filesystem::path> &roots) {
    const std::string context =
        "config.clusters." + std::string{role};
    require_keys(scenario_cluster,
                 {"profile", "parallelism", "scheduler", "execution_model"},
                 {"moe_routing", "total_expert_num", "router_topk",
                  "first_k_dense_replace", "num_shared_experts"},
                 context);
    const std::string profile =
        require_string(scenario_cluster, "profile", context);
    const AssetDocument cluster_document =
        load_asset(roots, "clusters", profile);
    const ClusterAsset cluster =
        parse_cluster_asset(cluster_document, profile, roots);

    const Json &parallelism = scenario_cluster.at("parallelism");
    const std::uint64_t required =
        required_gpu_count(parallelism, context + ".parallelism");
    if (required != cluster.gpu_count) {
        throw ConfigError(context + " profile '" + profile + "' provides " +
                          std::to_string(cluster.gpu_count) +
                          " GPUs but parallelism requires " +
                          std::to_string(required) +
                          " (num_replicas*data_parallel_size*"
                          "pipeline_parallel_size*tensor_parallel_size)");
    }

    Json execution = scenario_cluster.at("execution_model");
    require_object(execution, context + ".execution_model");
    const std::string execution_type =
        require_string(execution, "type", context + ".execution_model");
    if (execution_type == "analytical") {
        execution = compose_analytical_execution(
            execution, context + ".execution_model", roots);
        if (!cluster.gpu_analytical.has_value()) {
            throw ConfigError(context + " uses analytical execution but GPU "
                              "profile has no analytical device contract");
        }
        execution["device"] = cluster.gpu_analytical->at("device");
        if (cluster.gpu_analytical->contains("device_overrides")) {
            execution["device_overrides"] =
                cluster.gpu_analytical->at("device_overrides");
        }
        execution["network_bandwidth_gbps"] =
            cluster.network.at("network_bandwidth_gbps");
        execution["network_latency_us"] =
            cluster.network.at("network_latency_us");
        execution["intra_node_bandwidth_gbps"] =
            cluster.network.at("intra_node_bandwidth_gbps");
    } else if (execution_type == "fixed") {
        require_exact_keys(execution, {"type", "stage_latencies_ms"},
                           context + ".execution_model");
    } else {
        throw ConfigError(context +
                          ".execution_model.type must be 'fixed' or "
                          "'analytical'");
    }

    Json result{{"parallelism", parallelism},
                {"scheduler", scenario_cluster.at("scheduler")},
                {"gpu_memory", cluster.gpu_memory},
                {"execution_model", std::move(execution)},
                {"model_name", model_name},
                {"moe_routing",
                 scenario_cluster.contains("moe_routing")
                     ? scenario_cluster.at("moe_routing")
                     : default_moe_routing()}};
    for (const std::string_view field :
         {"total_expert_num", "router_topk", "first_k_dense_replace",
          "num_shared_experts"}) {
        if (scenario_cluster.contains(field)) {
            result[std::string{field}] = scenario_cluster.at(field);
        }
    }
    return result;
}

Json compose_kv_cache_transfer(
    const Json &input, const std::vector<std::filesystem::path> &roots) {
    require_exact_keys(input,
                       {"link", "kv_cache_dtype_size_bytes",
                        "enable_compression"},
                       "config.kv_cache_transfer");
    const std::string link_name =
        require_string(input, "link", "config.kv_cache_transfer");
    const AssetDocument document = load_asset(roots, "links", link_name);
    const std::string context = "link asset " + document.path.string();
    require_exact_keys(document.json,
                       {"asset_schema_version", "name",
                        "network_bandwidth_gbps", "network_latency_ms"},
                       context);
    validate_asset_identity(document.json, document.path, link_name, "link");
    return Json{{"type", "analytical"},
                {"network_bandwidth_gbps",
                 document.json.at("network_bandwidth_gbps")},
                {"network_latency_ms",
                 document.json.at("network_latency_ms")},
                {"kv_cache_dtype_size_bytes",
                 input.at("kv_cache_dtype_size_bytes")},
                {"enable_compression", input.at("enable_compression")}};
}

Json compose_v2(const Json &scenario,
                const std::filesystem::path &scenario_path) {
    require_keys(scenario,
                 {"schema_version", "run_id", "simulation_mode",
                  "system_architecture", "model", "clusters"},
                 {"prefix_cache", "cpu_kv_cache", "cluster_scheduler",
                  "kv_cache_transfer", "prefill_only"},
                 "config");
    const int version = require_int(scenario, "schema_version", "config");
    if (version != kModularSchemaVersion) {
        throw ConfigError("modular config requires schema_version=2");
    }
    const std::string architecture =
        require_string(scenario, "system_architecture", "config");
    const std::string model_name =
        require_string(scenario, "model", "config");
    const auto roots = asset_roots(scenario_path);

    Json result{{"schema_version", kSchemaVersion},
                {"run_id", scenario.at("run_id")},
                {"simulation_mode", scenario.at("simulation_mode")},
                {"system_architecture", architecture},
                {"enable_parallel_clusters", false},
                {"prefix_cache",
                 scenario.contains("prefix_cache")
                     ? scenario.at("prefix_cache")
                     : Json{{"enabled", false}, {"key_mode", "session"}}},
                {"cluster_scheduler",
                 scenario.contains("cluster_scheduler")
                     ? scenario.at("cluster_scheduler")
                     : Json{{"type", "round_robin"}}}};
    if (scenario.contains("cpu_kv_cache")) {
        result["cpu_kv_cache"] = scenario.at("cpu_kv_cache");
    }

    const Json &clusters = scenario.at("clusters");
    if (architecture == "co-location") {
        require_exact_keys(clusters, {"monolithic"}, "config.clusters");
        if (scenario.contains("kv_cache_transfer")) {
            throw ConfigError(
                "co-location config must not contain kv_cache_transfer");
        }
        if (scenario.contains("prefill_only")) {
            throw ConfigError(
                "config.prefill_only is supported only for "
                "pd-disaggregation");
        }
        result["clusters"] =
            Json{{"monolithic",
                  compose_cluster_runtime(clusters.at("monolithic"),
                                          "monolithic", model_name, roots)}};
    } else if (architecture == "pd-disaggregation") {
        require_exact_keys(clusters, {"prefill", "decode"},
                           "config.clusters");
        if (!scenario.contains("kv_cache_transfer")) {
            throw ConfigError(
                "PDD config is missing required field 'kv_cache_transfer'");
        }
        result["clusters"] =
            Json{{"prefill",
                  compose_cluster_runtime(clusters.at("prefill"), "prefill",
                                          model_name, roots)},
                 {"decode",
                  compose_cluster_runtime(clusters.at("decode"), "decode",
                                          model_name, roots)}};
        result["kv_cache_transfer"] =
            compose_kv_cache_transfer(scenario.at("kv_cache_transfer"),
                                      roots);
        if (scenario.contains("prefill_only")) {
            result["prefill_only"] = scenario.at("prefill_only");
        }
    } else {
        throw ConfigError("config.system_architecture must be 'co-location' "
                          "or 'pd-disaggregation'");
    }
    return result;
}

} // namespace

SimulationConfig
load_simulation_config_file(const std::filesystem::path &path) {
    const Json root = read_json_file(path, "config");
    require_object(root, "config");
    if (!root.contains("schema_version")) {
        throw ConfigError("config is missing required field 'schema_version'");
    }
    const int version = require_int(root, "schema_version", "config");
    if (version == kSchemaVersion) {
        return parse_simulation_config_json(root.dump());
    }
    if (version == kModularSchemaVersion) {
        return parse_simulation_config_json(compose_v2(root, path).dump());
    }
    throw ConfigError("unsupported config schema_version=" +
                      std::to_string(version) + "; expected 1 or 2");
}

} // namespace frontier::config
