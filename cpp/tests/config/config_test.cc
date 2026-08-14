#include "frontier/config/config.h"
#include "tests/test_support.h"

#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>

#include "frontier/attention/ops.h"
#include "frontier/kv_cache_transfer/analytical_transfer.h"

#ifndef FRONTIER_TEST_FIXTURE_DIR
#error "FRONTIER_TEST_FIXTURE_DIR must be defined for contract tests"
#endif

namespace {

using frontier::ClusterType;
using frontier::config::ClusterSchedulerType;
using frontier::config::ConfigError;
using frontier::config::ExecutionModelType;
using frontier::config::kSchemaVersion;
using frontier::config::parse_simulation_config_json;
using frontier::config::PddRuntimeConfig;
using frontier::config::resolve_pdd_kda_snapshot_dtype_size_bytes;
using frontier::config::serialize_simulation_config_json;
using frontier::config::SystemArchitecture;
using frontier::test::expect;
using frontier::test::expect_throws;
using frontier::test::read_text_file;

std::filesystem::path fixture(std::string_view name) {
    return std::filesystem::path{FRONTIER_TEST_FIXTURE_DIR} / "config" / name;
}

frontier::config::SimulationConfig load(std::string_view name) {
    return parse_simulation_config_json(read_text_file(fixture(name)));
}

class ScopedModelConfigDirectory {
  public:
    explicit ScopedModelConfigDirectory(const std::filesystem::path &path) {
#ifdef _WIN32
        char *previous = nullptr;
        std::size_t previous_size = 0;
        if (_dupenv_s(&previous, &previous_size,
                      "FRONTIER_MODEL_CONFIG_DIR") == 0 &&
            previous != nullptr) {
            had_previous_ = true;
            previous_ = previous;
        }
        std::free(previous);
#else
        if (const char *value = std::getenv("FRONTIER_MODEL_CONFIG_DIR");
            value != nullptr) {
            had_previous_ = true;
            previous_ = value;
        }
#endif
#ifdef _WIN32
        if (_putenv_s("FRONTIER_MODEL_CONFIG_DIR", path.string().c_str()) !=
            0) {
            throw std::runtime_error("failed to set model fixture directory");
        }
#else
        if (setenv("FRONTIER_MODEL_CONFIG_DIR", path.string().c_str(), 1) !=
            0) {
            throw std::runtime_error("failed to set model fixture directory");
        }
#endif
    }

    ScopedModelConfigDirectory(const ScopedModelConfigDirectory &) = delete;
    ScopedModelConfigDirectory &
    operator=(const ScopedModelConfigDirectory &) = delete;

    ~ScopedModelConfigDirectory() {
#ifdef _WIN32
        static_cast<void>(_putenv_s("FRONTIER_MODEL_CONFIG_DIR",
                                    had_previous_ ? previous_.c_str() : ""));
#else
        if (had_previous_) {
            static_cast<void>(
                setenv("FRONTIER_MODEL_CONFIG_DIR", previous_.c_str(), 1));
        } else {
            static_cast<void>(unsetenv("FRONTIER_MODEL_CONFIG_DIR"));
        }
#endif
    }

  private:
    bool had_previous_ = false;
    std::string previous_;
};

void test_colocation_contract_round_trip() {
    auto config = load("fixed_parallel_colocation.json");
    expect(config.schema_version == kSchemaVersion,
           "the current schema version must parse");
    expect(config.system_architecture == SystemArchitecture::kCoLocation,
           "co-location architecture must parse");
    expect(config.cluster().parallelism.num_replicas == 2 &&
               config.cluster().parallelism.data_parallel_size == 2 &&
               !config.cluster().parallelism.pipeline_exclusive,
           "monolithic cluster topology must parse");
    expect(config.cluster().execution_model.type == ExecutionModelType::kFixed,
           "fixed execution model must parse");
    expect(parse_simulation_config_json(
               serialize_simulation_config_json(config)) == config,
           "co-location config must round-trip deterministically");

    config.cluster().scheduler.pipeline_event_mode = "collapsed";
    const auto collapsed = parse_simulation_config_json(
        serialize_simulation_config_json(config));
    expect(collapsed.cluster().scheduler.pipeline_event_mode == "collapsed" &&
               collapsed == config,
           "collapsed pipeline event mode must round-trip");

    std::string invalid = serialize_simulation_config_json(config);
    const std::string valid_mode = "\"pipeline_event_mode\": \"collapsed\"";
    const auto position = invalid.find(valid_mode);
    expect(position != std::string::npos,
           "serialized scheduler must expose pipeline_event_mode");
    invalid.replace(position, valid_mode.size(),
                    "\"pipeline_event_mode\": \"invalid\"");
    expect_throws<ConfigError>(
        [&invalid] {
            static_cast<void>(parse_simulation_config_json(invalid));
        },
        "invalid pipeline event mode must fail fast");
}

void test_pdd_contract_round_trip() {
    const auto config = load("fixed_sequential_pdd.json");
    expect(config.schema_version == kSchemaVersion &&
               config.system_architecture ==
                   SystemArchitecture::kPdDisaggregation &&
               std::holds_alternative<PddRuntimeConfig>(config.runtime),
           "PDD clusters and transfer config must parse");
    expect(parse_simulation_config_json(
               serialize_simulation_config_json(config)) == config,
           "PDD config must round-trip deterministically");
}

void test_stage_specific_cluster_scheduler_overrides() {
    auto config = load("fixed_sequential_pdd.json");
    expect(config.cluster_scheduler.type_for_cluster(ClusterType::kPrefill) ==
                   ClusterSchedulerType::kRoundRobin &&
               config.cluster_scheduler.type_for_cluster(
                   ClusterType::kDecode) == ClusterSchedulerType::kRoundRobin,
           "legacy cluster scheduler type must fall back for both PDD stages");

    config.cluster_scheduler.type = ClusterSchedulerType::kStickyRoundRobin;
    config.cluster_scheduler.prefill_type =
        ClusterSchedulerType::kStickyRoundRobin;
    config.cluster_scheduler.decode_type =
        ClusterSchedulerType::kVllmQueueAware;
    config.cluster_scheduler.cache_threshold = 0.625;
    config.cluster_scheduler.balance_abs_threshold = 17;
    config.cluster_scheduler.balance_rel_threshold = 1.25;
    const auto parsed =
        parse_simulation_config_json(serialize_simulation_config_json(config));
    expect(
        parsed.cluster_scheduler.prefill_type.has_value() &&
            parsed.cluster_scheduler.decode_type.has_value() &&
            parsed.cluster_scheduler.type_for_cluster(ClusterType::kPrefill) ==
                ClusterSchedulerType::kStickyRoundRobin &&
            parsed.cluster_scheduler.type_for_cluster(ClusterType::kDecode) ==
                ClusterSchedulerType::kVllmQueueAware &&
            parsed.cluster_scheduler.cache_threshold == 0.625 &&
            parsed.cluster_scheduler.balance_abs_threshold == 17 &&
            parsed.cluster_scheduler.balance_rel_threshold == 1.25,
        "PDD stage scheduler overrides must parse and round-trip");

    auto colocation = load("fixed_parallel_colocation.json");
    colocation.cluster_scheduler.type = ClusterSchedulerType::kRoundRobin;
    colocation.cluster_scheduler.prefill_type = ClusterSchedulerType::kKvAware;
    colocation.cluster_scheduler.decode_type =
        ClusterSchedulerType::kVllmQueueAware;
    const auto parsed_colocation = parse_simulation_config_json(
        serialize_simulation_config_json(colocation));
    expect(parsed_colocation.cluster_scheduler.type_for_cluster(
               ClusterType::kMonolithic) == ClusterSchedulerType::kRoundRobin,
           "stage overrides must not change co-location legacy routing");
}

void test_analytical_contract_round_trip() {
    auto config = load("analytical_parallel_colocation.json");
    expect(config.cluster().execution_model.type ==
               ExecutionModelType::kAnalytical,
           "analytical execution model must parse");
    expect(config.cluster().execution_model.analytical.tensor_parallel_size ==
               4,
           "analytical TP must come from cluster parallelism");

    std::string unsupported = serialize_simulation_config_json(config);
    unsupported.replace(unsupported.find("\"rubin\""),
                        std::string{"\"rubin\""}.size(), "\"a100\"");
    expect_throws<ConfigError>(
        [&unsupported] {
            static_cast<void>(parse_simulation_config_json(unsupported));
        },
        "unsupported analytical hardware must fail fast");

    auto &analytical = config.cluster().execution_model.analytical;
    analytical.device = "gb300";
    analytical.moe_layer_event_mode = "first_layer_scaled";
    analytical.device_overrides.hbm_bandwidth_tbps = 7.5;
    analytical.device_overrides.fp8_tflops = 4'750.0;
    const auto gb300 =
        parse_simulation_config_json(serialize_simulation_config_json(config));
    expect(
        gb300 == config &&
            gb300.cluster()
                    .execution_model.analytical.device_overrides
                    .hbm_bandwidth_tbps == 7.5 &&
            gb300.cluster()
                    .execution_model.analytical.device_overrides.fp8_tflops ==
                4'750.0,
        "GB300 preset and partial device overrides must round-trip");

    std::string invalid_event_mode = serialize_simulation_config_json(config);
    const std::string valid_event_mode = "\"first_layer_scaled\"";
    invalid_event_mode.replace(invalid_event_mode.find(valid_event_mode),
                               valid_event_mode.size(), "\"invalid\"");
    expect_throws<ConfigError>(
        [&invalid_event_mode] {
            static_cast<void>(parse_simulation_config_json(invalid_event_mode));
        },
        "unsupported MoE layer event modes must fail fast");

    analytical.device = "custom";
    const std::string incomplete_custom =
        serialize_simulation_config_json(config);
    expect_throws<ConfigError>(
        [&incomplete_custom] {
            static_cast<void>(parse_simulation_config_json(incomplete_custom));
        },
        "custom devices must provide every hardware ceiling");

    analytical.device_overrides.fp32_tflops = 80.0;
    analytical.device_overrides.fp16_tflops = 2'000.0;
    analytical.device_overrides.fp4_tflops = 12'000.0;
    expect(parse_simulation_config_json(
               serialize_simulation_config_json(config)) == config,
           "complete custom analytical devices must round-trip");
}

void test_operator_precision_contract_round_trip() {
    auto config = load("analytical_parallel_colocation.json");
    auto &analytical = config.cluster().execution_model.analytical;
    analytical.operator_precisions.attention = "fp8";
    analytical.operator_precisions.dense = "fp8";
    analytical.operator_precisions.moe_expert = "fp4";
    analytical.operator_precisions.moe_router = "fp8";
    analytical.operator_precisions.kv_cache = "fp8";
    analytical.operator_precisions.communication = "fp8";
    analytical.operator_precisions.attention_weight = "fp8";
    analytical.operator_precisions.attention_activation = "bf16";
    analytical.operator_precisions.moe_expert_weight = "fp4";
    analytical.operator_precisions.moe_expert_activation = "fp8";
    analytical.operator_precisions.lm_head_weight = "fp8";
    analytical.operator_precisions.lm_head_activation = "bf16";
    analytical.operator_precisions.routed_expert_weight = "mxfp4";
    analytical.operator_precisions.routed_expert_activation = "mxfp8";
    analytical.operator_precisions.latent_moe_projection_weight = "bf16";
    analytical.operator_precisions.latent_moe_projection_activation = "fp8";
    analytical.operator_precisions.shared_expert_weight = "mxfp4";
    analytical.operator_precisions.shared_expert_activation = "mxfp8";
    analytical.operator_precisions.dense_mlp_weight = "mxfp8";
    analytical.operator_precisions.dense_mlp_activation = "bf16";
    analytical.operator_precisions.router_weight_storage = "bf16";
    analytical.operator_precisions.router_compute = "mxfp8";
    analytical.operator_precisions.kda_snapshot = "mxfp8";

    const std::string serialized = serialize_simulation_config_json(config);
    const auto parsed = parse_simulation_config_json(serialized);
    const auto &resolved = parsed.cluster().execution_model.analytical;
    expect(parsed == config && resolved.attention_precision() == "fp8" &&
               resolved.dense_precision() == "fp8" &&
               resolved.moe_expert_precision() == "fp4" &&
               resolved.moe_router_precision() == "fp8" &&
               resolved.kv_cache_precision() == "fp8" &&
               resolved.communication_precision() == "fp8" &&
               resolved.attention_activation_precision() == "bf16" &&
               resolved.moe_expert_weight_precision() == "fp4" &&
               resolved.moe_expert_activation_precision() == "fp8" &&
               resolved.lm_head_weight_precision() == "fp8" &&
               resolved.lm_head_activation_precision() == "bf16" &&
               resolved.routed_expert_weight_precision() == "mxfp4" &&
               resolved.routed_expert_activation_precision() == "mxfp8" &&
               resolved.latent_moe_projection_weight_precision() == "bf16" &&
               resolved.latent_moe_projection_activation_precision() ==
                   "fp8" &&
               resolved.shared_expert_weight_precision() == "mxfp4" &&
               resolved.shared_expert_activation_precision() == "mxfp8" &&
               resolved.dense_mlp_weight_precision() == "mxfp8" &&
               resolved.dense_mlp_activation_precision() == "bf16" &&
               resolved.router_weight_storage_precision() == "bf16" &&
               resolved.moe_router_weight_precision() == "bf16" &&
               resolved.router_compute_precision() == "mxfp8" &&
               resolved.kda_snapshot_precision() == "mxfp8",
           "operator precision overrides must round-trip and resolve");

    auto legacy = config;
    auto &legacy_precisions =
        legacy.cluster().execution_model.analytical.operator_precisions;
    legacy_precisions.routed_expert_weight.clear();
    legacy_precisions.routed_expert_activation.clear();
    legacy_precisions.latent_moe_projection_weight.clear();
    legacy_precisions.latent_moe_projection_activation.clear();
    legacy_precisions.shared_expert_weight.clear();
    legacy_precisions.shared_expert_activation.clear();
    legacy_precisions.dense_mlp_weight.clear();
    legacy_precisions.dense_mlp_activation.clear();
    legacy_precisions.router_weight_storage.clear();
    legacy_precisions.moe_router_weight = "fp8";
    legacy_precisions.router_compute.clear();
    legacy_precisions.kda_snapshot.clear();
    const auto legacy_resolved =
        parse_simulation_config_json(serialize_simulation_config_json(legacy))
            .cluster()
            .execution_model.analytical;
    expect(legacy_resolved.routed_expert_weight_precision() == "fp4" &&
               legacy_resolved.routed_expert_activation_precision() == "fp8" &&
               legacy_resolved.latent_moe_projection_weight_precision() ==
                   "fp4" &&
               legacy_resolved.latent_moe_projection_activation_precision() ==
                   "fp8" &&
               legacy_resolved.shared_expert_weight_precision() == "fp4" &&
               legacy_resolved.shared_expert_activation_precision() == "fp8" &&
               legacy_resolved.dense_mlp_weight_precision() == "fp8" &&
               legacy_resolved.dense_mlp_activation_precision() == "fp8" &&
               legacy_resolved.router_weight_storage_precision() == "fp8" &&
               legacy_resolved.router_compute_precision() == "fp8" &&
               legacy_resolved.kda_snapshot_precision() == "fp16",
           "K3-native precision getters must retain legacy family fallbacks");

    std::string invalid = serialized;
    const std::string valid_precision = "\"moe_expert\": \"fp4\"";
    const auto position = invalid.find(valid_precision);
    expect(position != std::string::npos,
           "serialized operator precision must be present");
    invalid.replace(position, valid_precision.size(),
                    "\"moe_expert\": \"fp3\"");
    expect_throws<ConfigError>(
        [&invalid] {
            static_cast<void>(parse_simulation_config_json(invalid));
        },
        "unsupported operator precision must fail fast");

    std::string invalid_native = serialized;
    const std::string valid_native_precision =
        "\"router_compute\": \"mxfp8\"";
    const auto native_position = invalid_native.find(valid_native_precision);
    expect(native_position != std::string::npos,
           "serialized K3-native precision must be present");
    invalid_native.replace(native_position, valid_native_precision.size(),
                           "\"router_compute\": \"mxfp16\"");
    expect_throws<ConfigError>(
        [&invalid_native] {
            static_cast<void>(parse_simulation_config_json(invalid_native));
        },
        "unsupported K3-native precision must fail fast");

    std::string invalid_router_storage = serialized;
    const std::string valid_router_storage =
        "\"router_weight_storage\": \"bf16\"";
    const auto router_storage_position =
        invalid_router_storage.find(valid_router_storage);
    expect(router_storage_position != std::string::npos,
           "serialized router storage precision must be present");
    invalid_router_storage.replace(router_storage_position,
                                   valid_router_storage.size(),
                                   "\"router_weight_storage\": \"fp19\"");
    expect_throws<ConfigError>(
        [&invalid_router_storage] {
            static_cast<void>(
                parse_simulation_config_json(invalid_router_storage));
        },
        "unsupported router storage precision must fail fast");

    std::string unknown = serialized;
    const std::string object_marker = "\"operator_precisions\": {";
    const auto object_position = unknown.find(object_marker);
    expect(object_position != std::string::npos,
           "serialized operator precision object must be present");
    unknown.insert(object_position + object_marker.size(),
                   "\n          \"unknown_operator\": \"fp8\",");
    expect_throws<ConfigError>(
        [&unknown] {
            static_cast<void>(parse_simulation_config_json(unknown));
        },
        "unknown operator precision keys must fail fast");
}

void test_gpu_memory_auto_block_calculation() {
    auto config = load("analytical_parallel_colocation.json");
    auto &cluster = config.cluster();
    cluster.gpu_memory.auto_calculate_num_blocks = true;
    cluster.gpu_memory.capacity_bytes_per_gpu = 80'000'000'000ULL;
    cluster.gpu_memory.runtime_reserve_fraction = 0.10;
    frontier::config::resolve_gpu_memory_config(cluster);

    const auto expected_stage_capacity =
        frontier::config::resolve_pipeline_logical_kv_capacity(
            cluster.gpu_memory.pipeline_stage_memory_profiles);
    expect(cluster.gpu_memory.model_weight_bytes_per_gpu > 0 &&
               cluster.gpu_memory.kv_cache_budget_bytes_per_gpu > 0 &&
               cluster.gpu_memory.kv_cache_bytes_per_block > 0 &&
               cluster.scheduler.num_blocks == expected_stage_capacity &&
               cluster.gpu_memory.ordinary_kv_capacity_blocks ==
                   expected_stage_capacity,
            "GPU capacity must resolve stage-local profiles and logical "
            "blocks");

    std::string without_manual_blocks =
        serialize_simulation_config_json(config);
    const std::string num_blocks =
        "\"num_blocks\": " + std::to_string(cluster.scheduler.num_blocks);
    const auto value_position = without_manual_blocks.find(num_blocks);
    expect(value_position != std::string::npos,
           "normalized auto-memory config must expose resolved blocks");
    const auto line_begin = without_manual_blocks.rfind('\n', value_position);
    const auto line_end = without_manual_blocks.find('\n', value_position);
    expect(line_begin != std::string::npos && line_end != std::string::npos,
           "resolved block field must occupy one JSON line");
    without_manual_blocks.erase(line_begin + 1, line_end - line_begin);
    const auto parsed = parse_simulation_config_json(without_manual_blocks);
    expect(parsed == config,
           "auto-memory input may omit scheduler.num_blocks and must "
           "round-trip deterministically");

    auto smaller = config;
    smaller.cluster().gpu_memory.capacity_bytes_per_gpu = 60'000'000'000ULL;
    frontier::config::resolve_gpu_memory_config(smaller.cluster());
    expect(smaller.cluster().scheduler.num_blocks <
               cluster.scheduler.num_blocks,
           "smaller HBM capacity must produce fewer KV blocks");

    auto quantized = config;
    auto &precisions =
        quantized.cluster().execution_model.analytical.operator_precisions;
    precisions.attention_weight = "fp8";
    precisions.dense_weight = "fp8";
    precisions.lm_head_weight = "fp8";
    frontier::config::resolve_gpu_memory_config(quantized.cluster());
    expect(quantized.cluster().gpu_memory.model_weight_bytes_per_gpu <
                   cluster.gpu_memory.model_weight_bytes_per_gpu &&
               quantized.cluster().gpu_memory.kv_cache_bytes_per_block ==
                   cluster.gpu_memory.kv_cache_bytes_per_block &&
               quantized.cluster().scheduler.num_blocks >
                   cluster.scheduler.num_blocks,
           "weight-only quantization must free HBM for additional KV blocks");

    auto kimi_prefill = config.cluster();
    kimi_prefill.model =
        frontier::config::load_model_config("moonshotai/Kimi-K2-Instruct");
    kimi_prefill.parallelism.tensor_parallel_size = 1;
    kimi_prefill.parallelism.decode_context_parallel_size = 1;
    kimi_prefill.parallelism.pipeline_parallel_size = 1;
    kimi_prefill.parallelism.data_parallel_size = 8;
    kimi_prefill.parallelism.moe_tensor_parallel_size = 1;
    kimi_prefill.parallelism.moe_expert_parallel_size = 8;
    kimi_prefill.execution_model.analytical.tensor_parallel_size = 1;
    kimi_prefill.execution_model.analytical.precision = "fp8";
    kimi_prefill.execution_model.analytical.operator_precisions = {};
    kimi_prefill.execution_model.analytical.operator_precisions
        .moe_expert_weight = "fp4";
    kimi_prefill.gpu_memory.capacity_bytes_per_gpu = 288'000'000'000ULL;
    kimi_prefill.gpu_memory.runtime_reserve_fraction = 0.10;
    resolve_gpu_memory_config(kimi_prefill);

    auto kimi_decode = kimi_prefill;
    kimi_decode.parallelism.tensor_parallel_size = 4;
    kimi_decode.parallelism.decode_context_parallel_size = 4;
    kimi_decode.parallelism.data_parallel_size = 6;
    kimi_decode.parallelism.moe_expert_parallel_size = 24;
    kimi_decode.execution_model.analytical.tensor_parallel_size = 4;
    resolve_gpu_memory_config(kimi_decode);
    expect(kimi_prefill.gpu_memory.model_weight_bytes_per_gpu >
                   kimi_decode.gpu_memory.model_weight_bytes_per_gpu &&
               kimi_prefill.gpu_memory.kv_cache_bytes_per_block ==
                   4 * kimi_decode.gpu_memory.kv_cache_bytes_per_block &&
               kimi_prefill.scheduler.num_blocks <
                   kimi_decode.scheduler.num_blocks,
           "Kimi K2 TP/EP and MLA DCP must change rank-local weights and KV "
           "capacity");
}

void test_gpu_memory_capacity_is_required_and_mode_round_trips() {
    const auto fixed = load("fixed_parallel_colocation.json");
    const auto fixed_json = serialize_simulation_config_json(fixed);
    expect(!fixed.cluster().gpu_memory.auto_calculate_num_blocks,
           "explicit scheduler.num_blocks must select manual GPU memory mode");
    expect(fixed_json.find("\"auto_calculate_num_blocks\": false") !=
               std::string::npos,
           "serialized manual GPU memory must carry an explicit auto flag");
    expect(parse_simulation_config_json(fixed_json) == fixed,
           "manual GPU memory mode must round-trip with required HBM input");

    auto missing = fixed_json;
    const std::string capacity_line = "\"capacity_bytes_per_gpu\":";
    const auto capacity_position = missing.find(capacity_line);
    expect(capacity_position != std::string::npos,
           "serialized GPU memory must expose HBM capacity");
    const auto line_begin = missing.rfind('\n', capacity_position);
    const auto line_end = missing.find('\n', capacity_position);
    expect(line_begin != std::string::npos && line_end != std::string::npos,
           "serialized HBM capacity must occupy one JSON line");
    missing.erase(line_begin + 1, line_end - line_begin);
    expect_throws<ConfigError>(
        [&missing] {
            static_cast<void>(parse_simulation_config_json(missing));
        },
        "every cluster must provide gpu_memory.capacity_bytes_per_gpu");

    auto zero = fixed_json;
    const auto zero_position = zero.find(capacity_line);
    const auto zero_value_begin = zero_position + capacity_line.size();
    const auto zero_value_end = zero.find(',', zero_value_begin);
    expect(zero_value_end != std::string::npos,
           "serialized HBM capacity must be comma-terminated");
    zero.replace(zero_value_begin, zero_value_end - zero_value_begin, " 0");
    expect_throws<ConfigError>(
        [&zero] {
            static_cast<void>(parse_simulation_config_json(zero));
        },
        "GPU HBM capacity must be strictly positive");

    auto unset = fixed.cluster();
    unset.gpu_memory.capacity_bytes_per_gpu = 0;
    expect_throws<ConfigError>(
        [&unset] {
            frontier::config::resolve_gpu_memory_config(
                unset, unset.scheduler.num_blocks);
        },
        "in-memory GPU memory resolution must reject an unset HBM capacity");

    auto too_small = fixed.cluster();
    too_small.gpu_memory.capacity_bytes_per_gpu = 1;
    too_small.gpu_memory.runtime_reserve_fraction = 0.0;
    expect_throws<ConfigError>(
        [&too_small] {
            frontier::config::resolve_gpu_memory_config(
                too_small, too_small.scheduler.num_blocks);
        },
        "manual blocks must fit the exact stage-local HBM capacity");

    auto analytical = load("analytical_parallel_colocation.json");
    analytical.cluster().gpu_memory.auto_calculate_num_blocks = true;
    frontier::config::resolve_gpu_memory_config(analytical.cluster());
    const auto analytical_json = serialize_simulation_config_json(analytical);
    expect(analytical.cluster().gpu_memory.auto_calculate_num_blocks,
           "analytical configs without explicit blocks must select auto mode");
    expect(analytical_json.find("\"auto_calculate_num_blocks\": true") !=
               std::string::npos,
           "serialized automatic GPU memory must carry an explicit auto flag");
    expect(parse_simulation_config_json(analytical_json) == analytical,
           "automatic GPU memory mode must round-trip with required HBM input");
}

void test_kimi_k3_gated_mla_weight_memory() {
    auto config = load("analytical_parallel_colocation.json");
    auto &cluster = config.cluster();
    cluster.model =
        frontier::config::load_model_config("moonshotai/Kimi-K3");
    cluster.gpu_memory.auto_calculate_num_blocks = true;
    cluster.gpu_memory.capacity_bytes_per_gpu = 10'000'000'000'000'000ULL;
    cluster.gpu_memory.runtime_reserve_fraction = 0.10;
    frontier::config::resolve_gpu_memory_config(cluster);
    const auto gated_weight_bytes =
        cluster.gpu_memory.model_weight_bytes_per_gpu;

    auto ungated = cluster;
    ungated.model.mla_use_output_gate = false;
    frontier::config::resolve_gpu_memory_config(ungated);
    expect(cluster.model.mla_use_nope &&
               gated_weight_bytes >
                   ungated.gpu_memory.model_weight_bytes_per_gpu,
           "K3 gated MLA must add output-gate weights to automatic HBM "
           "accounting while preserving no-PE metadata");

    auto without_attn_res = cluster;
    without_attn_res.model.attn_res_block_size = 0;
    frontier::config::resolve_gpu_memory_config(without_attn_res);
    expect(gated_weight_bytes ==
               without_attn_res.gpu_memory.model_weight_bytes_per_gpu,
           "AttnRes metadata must not add automatic GPU weight memory");

    auto bf16_router = cluster;
    bf16_router.execution_model.analytical.operator_precisions
        .router_weight_storage = "bf16";
    bf16_router.execution_model.analytical.operator_precisions.router_compute =
        "fp32";
    frontier::config::resolve_gpu_memory_config(bf16_router);
    auto fp32_router = bf16_router;
    fp32_router.execution_model.analytical.operator_precisions
        .router_weight_storage = "fp32";
    frontier::config::resolve_gpu_memory_config(fp32_router);
    expect(fp32_router.gpu_memory.model_weight_bytes_per_gpu >
               bf16_router.gpu_memory.model_weight_bytes_per_gpu,
           "router compute precision must not inflate resident BF16 weight "
           "storage");
}

void test_kimi_k3_latent_moe_projection_weight_memory() {
    auto config = load("analytical_parallel_colocation.json");
    auto &cluster = config.cluster();
    cluster.model =
        frontier::config::load_model_config("moonshotai/Kimi-K3");
    cluster.parallelism.pipeline_parallel_size = 1;
    cluster.gpu_memory.auto_calculate_num_blocks = true;
    cluster.gpu_memory.capacity_bytes_per_gpu = 10'000'000'000'000'000ULL;
    frontier::config::apply_model_native_precision_defaults(
        cluster.execution_model.analytical, cluster.model);
    frontier::config::resolve_gpu_memory_config(cluster);

    auto quantized_projection = cluster;
    quantized_projection.execution_model.analytical.operator_precisions
        .latent_moe_projection_weight = "mxfp4";
    frontier::config::resolve_gpu_memory_config(quantized_projection);
    constexpr std::uint64_t kExpectedBf16MinusMxfp4Bytes =
        92ULL * 2ULL * 7'168ULL * 3'584ULL * 47ULL / 32ULL;
    expect(cluster.gpu_memory.model_weight_bytes_per_gpu -
                   quantized_projection.gpu_memory.model_weight_bytes_per_gpu ==
               kExpectedBf16MinusMxfp4Bytes,
           "K3 PP1 must account for both BF16 Stable LatentMoE projections "
           "independently from MXFP4 routed experts");

    auto dense = load("analytical_parallel_colocation.json").cluster();
    dense.parallelism.pipeline_parallel_size = 1;
    dense.gpu_memory.auto_calculate_num_blocks = true;
    dense.gpu_memory.capacity_bytes_per_gpu = 10'000'000'000'000'000ULL;
    frontier::config::resolve_gpu_memory_config(dense);
    const std::uint64_t baseline = dense.gpu_memory.model_weight_bytes_per_gpu;
    dense.execution_model.analytical.operator_precisions
        .latent_moe_projection_weight = "fp32";
    dense.execution_model.analytical.operator_precisions
        .latent_moe_projection_activation = "fp32";
    frontier::config::resolve_gpu_memory_config(dense);
    expect(dense.gpu_memory.model_weight_bytes_per_gpu == baseline,
           "non-latent models must ignore Stable LatentMoE projection "
           "precision overrides in automatic HBM accounting");
}

void test_kimi_k3_tp8_dcp8_gpu_memory_layout() {
    auto config = load("analytical_parallel_colocation.json");
    auto replicated = config.cluster();
    replicated.model =
        frontier::config::load_model_config("moonshotai/Kimi-K3");
    replicated.parallelism.tensor_parallel_size = 8;
    replicated.parallelism.decode_context_parallel_size = 1;
    replicated.parallelism.pipeline_parallel_size = 1;
    replicated.parallelism.data_parallel_size = 1;
    replicated.parallelism.moe_tensor_parallel_size = 1;
    replicated.parallelism.moe_expert_parallel_size = 8;
    replicated.execution_model.analytical.tensor_parallel_size = 8;
    replicated.gpu_memory.auto_calculate_num_blocks = true;
    replicated.gpu_memory.capacity_bytes_per_gpu =
        10'000'000'000'000'000ULL;
    frontier::config::resolve_gpu_memory_config(replicated);

    auto dcp = replicated;
    dcp.parallelism.decode_context_parallel_size = 8;
    frontier::config::resolve_gpu_memory_config(dcp);

    constexpr std::uint64_t kRankLocalSnapshotBytes = 29'039'616ULL;
    const auto charged_snapshot_bytes = [](const auto &cluster) {
        return cluster.scheduler.kda_snapshot_blocks_per_session *
               cluster.gpu_memory.kv_cache_bytes_per_block;
    };
    expect(dcp.gpu_memory.model_weight_bytes_per_gpu ==
                   replicated.gpu_memory.model_weight_bytes_per_gpu &&
               replicated.gpu_memory.kv_cache_bytes_per_block ==
                   8 * dcp.gpu_memory.kv_cache_bytes_per_block &&
               dcp.scheduler.num_blocks > replicated.scheduler.num_blocks,
           "K3 TP8/DCP8 must preserve TP-sharded weights while sharding MLA "
           "KV capacity eight ways");
    for (const auto *cluster : {&replicated, &dcp}) {
        const std::uint64_t charged = charged_snapshot_bytes(*cluster);
        expect(charged >= kRankLocalSnapshotBytes &&
                   charged - kRankLocalSnapshotBytes <
                       cluster->gpu_memory.kv_cache_bytes_per_block,
               "K3 snapshot charge must preserve one TP8 shard within one "
               "KV-block rounding unit");
    }
}

void test_pdd_kv_precision_matches_transfer_dtype() {
    auto config = load("fixed_sequential_pdd.json");
    auto &runtime = config.pdd();
    for (auto *cluster :
         {&runtime.clusters.prefill, &runtime.clusters.decode}) {
        cluster->execution_model.type = ExecutionModelType::kAnalytical;
        cluster->execution_model.analytical.tensor_parallel_size =
            cluster->parallelism.tensor_parallel_size;
        cluster->execution_model.analytical.operator_precisions.kv_cache =
            "fp8";
    }
    runtime.kv_cache_transfer.kv_cache_dtype_size_bytes = 1.0;
    const auto parsed =
        parse_simulation_config_json(serialize_simulation_config_json(config));
    expect(parsed.pdd().clusters.prefill.execution_model.analytical
                       .kv_cache_precision() == "fp8" &&
               parsed.pdd()
                       .clusters.decode.execution_model.analytical
                       .kv_cache_precision() == "fp8" &&
               parsed.pdd().kv_cache_transfer.kv_cache_dtype_size_bytes == 1.0,
           "PDD FP8 KV precision must accept one-byte transfer elements");

    runtime.kv_cache_transfer.kv_cache_dtype_size_bytes = 2.0;
    const std::string mismatched = serialize_simulation_config_json(config);
    expect_throws<ConfigError>(
        [&mismatched] {
            static_cast<void>(parse_simulation_config_json(mismatched));
        },
        "PDD KV precision and transfer dtype mismatch must fail fast");
}

void test_pdd_kda_snapshot_precision_contract() {
    auto config = load("fixed_sequential_pdd.json");
    const auto fixed_prefill_execution =
        config.pdd().clusters.prefill.execution_model;
    const auto fixed_decode_execution =
        config.pdd().clusters.decode.execution_model;
    const auto k3 = frontier::config::load_model_config("moonshotai/Kimi-K3");
    config.pdd().clusters.prefill.model = k3;
    config.pdd().clusters.decode.model = k3;

    // Fixed execution follows the native BF16 KDA snapshot on both sides.
    expect(resolve_pdd_kda_snapshot_dtype_size_bytes(config.pdd().clusters) ==
               2.0,
           "fixed PDD KDA snapshots must resolve to BF16-sized payloads");

    // Analytical BF16 and FP16 are both two bytes, but they are distinct
    // state contracts and must not be silently accepted as interchangeable.
    const auto analytical =
        load("analytical_parallel_colocation.json")
            .cluster()
            .execution_model;
    config.pdd().clusters.prefill.execution_model = analytical;
    config.pdd().clusters.decode.execution_model = analytical;
    config.pdd().clusters.prefill.execution_model.analytical
        .operator_precisions.kda_snapshot = "bf16";
    config.pdd().clusters.decode.execution_model.analytical
        .operator_precisions.kda_snapshot = "bf16";
    expect(resolve_pdd_kda_snapshot_dtype_size_bytes(config.pdd().clusters) ==
               2.0,
           "matching analytical BF16 KDA snapshots must resolve to two bytes");

    config.pdd().clusters.decode.execution_model.analytical
        .operator_precisions.kda_snapshot = "fp16";
    expect_throws<ConfigError>(
        [&config] {
            static_cast<void>(resolve_pdd_kda_snapshot_dtype_size_bytes(
                config.pdd().clusters));
        },
        "analytical BF16 and FP16 KDA snapshots must fail despite equal size");

    // A fixed/analytical pair is valid only when the analytical side uses the
    // fixed runtime's implicit BF16 snapshot representation.
    config.pdd().clusters.decode.execution_model = fixed_decode_execution;
    expect(resolve_pdd_kda_snapshot_dtype_size_bytes(config.pdd().clusters) ==
               2.0,
           "mixed fixed/analytical BF16 KDA snapshots must pass");
    config.pdd().clusters.prefill.execution_model.analytical
        .operator_precisions.kda_snapshot = "fp32";
    expect_throws<ConfigError>(
        [&config] {
            static_cast<void>(resolve_pdd_kda_snapshot_dtype_size_bytes(
                config.pdd().clusters));
        },
        "mixed fixed/analytical FP32 KDA snapshots must fail");

    // The helper also protects in-memory callers from asymmetric model state,
    // even though the JSON parser already enforces the full model contract.
    config.pdd().clusters.prefill.execution_model = fixed_prefill_execution;
    config.pdd().clusters.decode.execution_model = fixed_decode_execution;
    config.pdd().clusters.decode.model =
        frontier::config::load_model_config("meta-llama/Llama-2-7b-hf");
    expect_throws<ConfigError>(
        [&config] {
            static_cast<void>(resolve_pdd_kda_snapshot_dtype_size_bytes(
                config.pdd().clusters));
        },
        "PDD KDA snapshot resolver must reject asymmetric model state");
}

void test_analytical_attention_family_configs_parse() {
    const auto with_model = [](std::string model_name, std::uint64_t experts,
                               std::uint64_t topk) {
        std::string text =
            read_text_file(fixture("analytical_parallel_colocation.json"));
        const auto replace_once = [&text](std::string_view from,
                                          std::string replacement) {
            const auto position = text.find(from);
            if (position == std::string::npos) {
                throw std::runtime_error("analytical fixture mutation failed");
            }
            text.replace(position, from.size(), std::move(replacement));
        };
        replace_once("\"pipeline_parallel_size\": 2",
                     "\"pipeline_parallel_size\": 1");
        replace_once("\"moe_expert_parallel_size\": 1",
                     "\"moe_expert_parallel_size\": 8");
        replace_once("\"meta-llama/Llama-2-7b-hf\"", "\"" + model_name + "\"");
        replace_once("\"total_expert_num\": 1",
                     "\"total_expert_num\": " + std::to_string(experts));
        replace_once("\"router_topk\": 1",
                     "\"router_topk\": " + std::to_string(topk));
        return parse_simulation_config_json(text);
    };

    const auto kimi = with_model("moonshotai/Kimi-K2-Instruct", 384, 8);
    expect(kimi.cluster().model.attention.memory_layout ==
               frontier::attention::AttentionMemoryLayout::kLatentMla,
           "analytical config must accept latent MLA models");

    const auto step = with_model("step-moe-noquant-small", 24, 3);
    expect(step.cluster().model.use_mfa &&
               step.cluster().model.attention.memory_layout ==
                   frontier::attention::AttentionMemoryLayout::kDenseKv,
           "analytical config must accept Step3Text MFA models");
}

void test_legacy_schemas_and_shapes_are_rejected() {
    std::string old_version =
        read_text_file(fixture("fixed_parallel_colocation.json"));
    old_version.replace(old_version.find("\"schema_version\": 1"),
                        std::string{"\"schema_version\": 1"}.size(),
                        "\"schema_version\": 3");
    expect_throws<ConfigError>(
        [&old_version] {
            static_cast<void>(parse_simulation_config_json(old_version));
        },
        "legacy schema versions must be rejected");

    constexpr std::string_view old_flat_shape = R"json({
    "schema_version": 1,
    "run_id": "legacy",
    "simulation_mode": "offline",
    "system_architecture": "co-location",
    "enable_parallel_clusters": false,
    "prefix_cache": {"enabled": false, "key_mode": "session"}
  })json";
    expect_throws<ConfigError>(
        [old_flat_shape] {
            static_cast<void>(parse_simulation_config_json(old_flat_shape));
        },
        "the former foundation shape must be rejected");
}

void test_invalid_surfaces_are_rejected() {
    auto pdd = load("fixed_sequential_pdd.json");
    pdd.enable_parallel_clusters = true;
    expect_throws<ConfigError>(
        [&pdd] { static_cast<void>(serialize_simulation_config_json(pdd)); },
        "parallel PDD clusters must be rejected");

    auto colocation = load("fixed_parallel_colocation.json");
    colocation.prefix_cache.enabled = true;
    colocation.cluster_scheduler.type =
        frontier::config::ClusterSchedulerType::kStickyRoundRobin;
    expect(parse_simulation_config_json(
               serialize_simulation_config_json(colocation)) == colocation,
           "session prefix caching must round-trip in Step 4");

    std::string priority = serialize_simulation_config_json(
        load("fixed_parallel_colocation.json"));
    priority.replace(priority.find("\"fcfs\""), std::string{"\"fcfs\""}.size(),
                     "\"priority\"");
    expect_throws<ConfigError>(
        [&priority] {
            static_cast<void>(parse_simulation_config_json(priority));
        },
        "unsupported scheduling policies must be rejected");
}

void test_schema_version_range_is_checked_before_conversion() {
    const std::string valid =
        read_text_file(fixture("fixed_parallel_colocation.json"));
    for (const std::string_view out_of_range : {"4294967298", "-4294967295"}) {
        std::string config = valid;
        config.replace(config.find("\"schema_version\": 1"),
                       std::string{"\"schema_version\": 1"}.size(),
                       "\"schema_version\": " + std::string{out_of_range});
        expect_throws<ConfigError>(
            [&config] {
                static_cast<void>(parse_simulation_config_json(config));
            },
            "out-of-range schema versions must fail before conversion");
    }
}

void test_moe_contract_and_invalid_topologies() {
    const auto moe = load("analytical_moe_ep4_colocation.json");
    expect(moe.cluster().model.is_moe() &&
               moe.cluster().model.num_experts == 16 &&
               moe.cluster().model.total_expert_num == 8 &&
               moe.cluster().model.router_topk == 2 &&
               moe.cluster().parallelism.attention_parallel_size() == 4 &&
               moe.cluster().parallelism.moe_parallel_size() == 4,
           "Phi MoE model and shared parallel domain must normalize");
    expect(parse_simulation_config_json(
               serialize_simulation_config_json(moe)) == moe,
           "MoE normalized schema must round trip exactly");

    auto bad_shared_domain = moe;
    bad_shared_domain.cluster().parallelism.moe_expert_parallel_size = 2;
    expect_throws<ConfigError>(
        [&bad_shared_domain] {
            static_cast<void>(parse_simulation_config_json(
                serialize_simulation_config_json(bad_shared_domain)));
        },
        "attention and MoE physical domains must match");

    auto bad_expert_partition = moe;
    bad_expert_partition.cluster().model.total_expert_num = 6;
    expect_throws<ConfigError>(
        [&bad_expert_partition] {
            static_cast<void>(parse_simulation_config_json(
                serialize_simulation_config_json(bad_expert_partition)));
        },
        "runtime experts must divide evenly over EP lanes");

    auto bad_topk = moe;
    bad_topk.cluster().model.router_topk = 9;
    expect_throws<ConfigError>(
        [&bad_topk] {
            static_cast<void>(parse_simulation_config_json(
                serialize_simulation_config_json(bad_topk)));
        },
        "router top-k cannot exceed runtime expert count");

    auto mismatched_pdd = load("fixed_moe_sequential_pdd.json");
    mismatched_pdd.pdd().clusters.decode.model.router_topk = 1;
    expect_throws<ConfigError>(
        [&mismatched_pdd] {
            static_cast<void>(parse_simulation_config_json(
                serialize_simulation_config_json(mismatched_pdd)));
        },
        "PDD clusters must share the same MoE model contract");
}

void test_pipeline_exclusive_contract() {
    auto exclusive = load("analytical_moe_ep4_colocation.json");
    auto &parallelism = exclusive.cluster().parallelism;
    parallelism.pipeline_exclusive = true;
    parallelism.data_parallel_size = 1;
    parallelism.moe_tensor_parallel_size = parallelism.tensor_parallel_size;
    parallelism.moe_expert_parallel_size = 1;
    const auto round_tripped = parse_simulation_config_json(
        serialize_simulation_config_json(exclusive));
    expect(round_tripped == exclusive &&
               round_tripped.cluster().parallelism.pipeline_exclusive,
           "pipeline-exclusive parallelism must round-trip deterministically");

    auto bad_pipeline = exclusive;
    bad_pipeline.cluster().parallelism.pipeline_parallel_size = 1;
    expect_throws<ConfigError>(
        [&bad_pipeline] {
            static_cast<void>(parse_simulation_config_json(
                serialize_simulation_config_json(bad_pipeline)));
        },
        "pipeline-exclusive mode requires multiple pipeline stages");

    auto bad_data = exclusive;
    bad_data.cluster().parallelism.data_parallel_size = 2;
    expect_throws<ConfigError>(
        [&bad_data] {
            static_cast<void>(parse_simulation_config_json(
                serialize_simulation_config_json(bad_data)));
        },
        "pipeline-exclusive mode forbids data parallelism");

    auto bad_expert = exclusive;
    bad_expert.cluster().parallelism.moe_expert_parallel_size = 2;
    expect_throws<ConfigError>(
        [&bad_expert] {
            static_cast<void>(parse_simulation_config_json(
                serialize_simulation_config_json(bad_expert)));
        },
        "pipeline-exclusive mode forbids expert parallelism");

    auto bad_moe_tensor = exclusive;
    bad_moe_tensor.cluster().parallelism.moe_tensor_parallel_size = 1;
    expect_throws<ConfigError>(
        [&bad_moe_tensor] {
            static_cast<void>(parse_simulation_config_json(
                serialize_simulation_config_json(bad_moe_tensor)));
        },
        "pipeline-exclusive mode requires matching MoE and attention TP");
}

void test_model_registry_and_attention_binding() {
    const auto llama =
        frontier::config::load_model_config("meta-llama/Llama-2-7b-hf");
    expect(!llama.is_moe() && llama.num_layers == 32 &&
               llama.hidden_size == 4'096 &&
               llama.attention.variant ==
                   frontier::attention::AttentionVariant::kMha &&
               llama.runtime_num_kv_heads() == 32 && llama.kv_factor() == 2,
           "registered Llama model must bind dense MHA semantics");

    const auto llama3 =
        frontier::config::load_model_config("meta-llama/Meta-Llama-3-8B");
    expect(llama3.num_layers == 32 && llama3.num_kv_heads == 8 &&
               llama3.attention.variant ==
                   frontier::attention::AttentionVariant::kGqa,
           "Python-registered model names must resolve without a JSON asset");

    const auto phi =
        frontier::config::load_model_config("Phi-tiny-MoE-instruct");
    expect(phi.is_moe() && phi.num_experts == 16 &&
               phi.num_experts_per_token == 2 &&
               phi.attention.variant ==
                   frontier::attention::AttentionVariant::kGqa,
           "HF-style model JSON must load MoE and GQA structure");

    const auto deepseek = frontier::config::load_model_config("deepseek-v3");
    expect(deepseek.is_moe() && deepseek.use_mla &&
               deepseek.attention.memory_layout ==
                   frontier::attention::AttentionMemoryLayout::kLatentMla &&
               deepseek.runtime_num_kv_heads() == 1 &&
               deepseek.runtime_head_size() == 576 && deepseek.kv_factor() == 1,
           "MLA model assets must bind latent cache semantics");

    const auto kimi =
        frontier::config::load_model_config("moonshotai/Kimi-K2-Instruct");
    expect(kimi.is_moe() && kimi.num_experts == 384 &&
               kimi.num_experts_per_token == 8 && kimi.use_mla &&
               kimi.runtime_num_kv_heads() == 1 &&
               kimi.runtime_head_size() == 576 && kimi.num_layers == 61 &&
               kimi.first_k_dense_replace == 1 && !kimi.is_moe_layer(0) &&
               kimi.is_moe_layer(1) && kimi.dense_intermediate_size == 18'432 &&
               kimi.moe_intermediate_size == 2'048 &&
               kimi.num_shared_experts == 1 && kimi.vocab_size == 163'840,
           "Kimi K2 asset must expose dense-prefix, shared-expert, LM-head, "
           "and MLA fields");

    const auto stage0 =
        frontier::config::pipeline_stage_layer_range(kimi.num_layers, 4, 0);
    const auto stage1 =
        frontier::config::pipeline_stage_layer_range(kimi.num_layers, 4, 1);
    const auto stage3 =
        frontier::config::pipeline_stage_layer_range(kimi.num_layers, 4, 3);
    expect(stage0.begin == 0 && stage0.end == 16 && stage1.begin == 16 &&
               stage1.end == 31 && stage3.begin == 46 && stage3.end == 61,
           "61 Kimi K2 layers must partition over PP4 as 16/15/15/15");

    const auto step = frontier::config::load_model_config("step-moe");
    expect(step.is_moe() && step.use_mfa && step.share_q_dim == 2'048 &&
               step.attention.variant ==
                   frontier::attention::AttentionVariant::kMqa,
           "MFA model assets must validate shared-Q dense-KV structure");
}

void test_model_runtime_overrides_are_optional() {
    std::string text =
        read_text_file(fixture("fixed_parallel_colocation.json"));
    const auto erase_field = [&text](std::string_view field) {
        const std::string marker = "\"" + std::string{field} + "\"";
        const std::size_t marker_pos = text.find(marker);
        expect(marker_pos != std::string::npos,
               "optional model field must exist in source fixture");
        const std::size_t line_start = text.rfind('\n', marker_pos);
        const std::size_t line_end = text.find('\n', marker_pos);
        expect(line_start != std::string::npos && line_end != std::string::npos,
               "optional model field must occupy one JSON line");
        text.erase(line_start + 1, line_end - line_start);
    };
    erase_field("total_expert_num");
    erase_field("router_topk");

    const auto parsed = parse_simulation_config_json(text);
    expect(parsed.cluster().model.total_expert_num == 1 &&
               parsed.cluster().model.router_topk == 1,
           "dense model runtime values must default from the selected model");

    expect_throws<ConfigError>(
        [] {
            static_cast<void>(
                frontier::config::load_model_config("not-a-real/model"));
        },
        "unknown model names without matching JSON assets must fail fast");
}

void test_kimi_k3_nested_model_asset_and_aliases() {
    const auto k3 = frontier::config::load_model_config("moonshotai/Kimi-K3");
    expect(k3.model_type == "kimi_k3" && k3.is_moe() && k3.num_layers == 93 &&
               k3.hidden_size == 7'168 && k3.num_experts == 896 &&
               k3.num_experts_per_token == 16 && k3.num_shared_experts == 2 &&
               k3.first_k_dense_replace == 1,
           "nested Kimi K3 text_config must expose decoder and MoE fields");
    expect(k3.use_mla && k3.mla_use_nope && k3.mla_use_output_gate &&
               k3.has_kda() && k3.has_hybrid_attention() &&
               k3.num_kda_layers == 69 && k3.num_mla_layers == 24 &&
               k3.kda_layer_indices.size() == 69 &&
               k3.mla_layer_indices.size() == 24 && k3.is_kda_layer(0) &&
               !k3.is_kda_layer(3) && k3.is_mla_layer(3) && k3.is_mla_layer(92),
           "Kimi K3 hybrid metadata must preserve the official layer split");
    expect(
        k3.kda_num_heads == 96 && k3.kda_num_k_heads == 96 &&
            k3.kda_num_v_heads == 96 && k3.kda_head_dim == 128 &&
            k3.kda_key_head_dim == 128 && k3.kda_value_head_dim == 128 &&
            k3.kda_topology_is_symmetric() &&
            k3.kda_short_conv_kernel_size == 4 &&
            k3.kda_conv_state_dim == 36'864 &&
            k3.kda_recurrent_state_elements_per_layer() == 1'572'864 &&
            k3.kda_conv_state_elements_per_layer() == 110'592 &&
            k3.kda_state_elements_per_layer() == 1'683'456 &&
            k3.kda_state_snapshot_bytes() == 232'316'928,
        "Kimi K3 KDA state dimensions must match the recurrent/conv contract");
    expect(k3.routed_expert_hidden_size == 3'584 && k3.latent_moe_use_norm &&
               k3.attn_res_block_size == 12 && k3.has_latent_moe(),
           "Kimi K3 latent-MoE and attention-residual metadata must parse");

    const auto alias = frontier::config::load_model_config("Kimi-K3");
    expect(alias.name == "Kimi-K3" && alias.model_type == k3.model_type &&
               alias.kda_layer_indices == k3.kda_layer_indices &&
               alias.mla_layer_indices == k3.mla_layer_indices,
           "Kimi K3 short aliases must resolve the canonical nested asset");

    const auto k2 =
        frontier::config::load_model_config("moonshotai/Kimi-K2-Instruct");
    expect(!k2.has_kda() && !k2.mla_use_nope && !k2.mla_use_output_gate &&
               k2.num_mla_layers == k2.num_layers &&
               k2.is_mla_layer(0),
           "legacy flat Kimi K2 must retain all-layer MLA semantics");

    std::string runtime =
        read_text_file(fixture("fixed_moe_local_colocation.json"));
    const std::string old_name = "\"Phi-tiny-MoE-instruct\"";
    const std::size_t name_position = runtime.find(old_name);
    expect(name_position != std::string::npos,
           "MoE fixture must expose a replaceable model name");
    runtime.replace(name_position, old_name.size(), "\"moonshotai/Kimi-K3\"");
    const auto parsed = parse_simulation_config_json(runtime);
    const auto expected_runtime_snapshot_charge =
        frontier::config::resolve_pipeline_kda_snapshot_charge(
            parsed.cluster().gpu_memory.pipeline_stage_memory_profiles,
            parsed.cluster().scheduler.num_blocks);
    expect(parsed.cluster().model.has_kda() &&
               parsed.cluster().scheduler.kda_snapshot_blocks_per_session ==
                   expected_runtime_snapshot_charge &&
               expected_runtime_snapshot_charge > 0,
           "K3 runtime must derive one bounded physical per-session snapshot "
           "charge");

    std::string analytical_runtime =
        read_text_file(fixture("analytical_moe_ep4_colocation.json"));
    const std::size_t analytical_name_position =
        analytical_runtime.find(old_name);
    expect(analytical_name_position != std::string::npos,
           "analytical MoE fixture must expose a replaceable model name");
    analytical_runtime.replace(analytical_name_position, old_name.size(),
                               "\"moonshotai/Kimi-K3\"");
    const auto native = parse_simulation_config_json(analytical_runtime);
    const auto &precision =
        native.cluster().execution_model.analytical.operator_precisions;
    expect(precision.routed_expert_weight == "mxfp4" &&
               precision.routed_expert_activation == "mxfp8" &&
               precision.latent_moe_projection_weight == "bf16" &&
               precision.latent_moe_projection_activation == "bf16" &&
               precision.shared_expert_weight == "bf16" &&
               precision.shared_expert_activation == "bf16" &&
               precision.dense_mlp_weight == "bf16" &&
               precision.dense_mlp_activation == "bf16" &&
               precision.router_weight_storage == "bf16" &&
               precision.moe_router_activation == "bf16" &&
               precision.router_compute == "fp32" &&
               precision.lm_head == "bf16" &&
               precision.kv_cache == "fp8" &&
               precision.kda_snapshot == "bf16",
           "K3 analytical runtime must install the native mixed-precision "
           "policy");

    auto overridden = native;
    auto &overridden_execution =
        overridden.cluster().execution_model.analytical;
    overridden_execution.operator_precisions.routed_expert_weight = "int4";
    overridden_execution.operator_precisions.latent_moe_projection_weight =
        "fp8";
    overridden_execution.operator_precisions.shared_expert_weight = "mxfp4";
    overridden_execution.operator_precisions.router_weight_storage = "fp8";
    overridden_execution.operator_precisions.kv_cache = "bf16";
    overridden_execution.operator_precisions.kda_snapshot = "fp32";
    frontier::config::apply_model_native_precision_defaults(
        overridden_execution, overridden.cluster().model);
    expect(overridden_execution.routed_expert_weight_precision() == "int4" &&
               overridden_execution
                       .latent_moe_projection_weight_precision() == "fp8" &&
               overridden_execution.shared_expert_weight_precision() ==
                   "mxfp4" &&
               overridden_execution.router_weight_storage_precision() ==
                   "fp8" &&
               overridden_execution.kv_cache_precision() == "bf16" &&
               overridden_execution.kda_snapshot_precision() == "fp32" &&
               overridden_execution.router_compute_precision() == "fp32",
           "explicit K3 precision overrides must win while unspecified "
           "families retain native defaults");
    frontier::config::resolve_gpu_memory_config(
        overridden.cluster(), overridden.cluster().scheduler.num_blocks);
    const auto expected_overridden_snapshot_charge =
        frontier::config::resolve_pipeline_kda_snapshot_charge(
            overridden.cluster().gpu_memory.pipeline_stage_memory_profiles,
            overridden.cluster().scheduler.num_blocks);
    expect(overridden.cluster().scheduler.kda_snapshot_blocks_per_session ==
               expected_overridden_snapshot_charge &&
               expected_overridden_snapshot_charge > 0,
           "explicit BF16 KV and FP32 KDA snapshot overrides must preserve "
           "the physical snapshot charge");

    const std::string disabled_prefix = "\"enabled\": false";
    const std::size_t prefix_position = runtime.find(disabled_prefix);
    expect(prefix_position != std::string::npos,
           "MoE fixture must expose disabled prefix caching");
    runtime.replace(prefix_position, disabled_prefix.size(),
                    "\"enabled\": true");
    const auto prefix_enabled = parse_simulation_config_json(runtime);
    expect(prefix_enabled.cluster().scheduler
                   .kda_snapshot_blocks_per_session > 0 &&
               prefix_enabled.cluster().scheduler
                       .kda_snapshot_blocks_per_session <=
                   prefix_enabled.cluster().scheduler.num_blocks,
           "K3 prefix caching must use the normalized physical snapshot "
           "charge when the HBM snapshot fits");
}

void test_explicit_hybrid_attention_metadata_is_strict() {
    const ScopedModelConfigDirectory model_fixtures{
        std::filesystem::path{FRONTIER_TEST_FIXTURE_DIR} / "models"};

    expect_throws<ConfigError>(
        [] {
            static_cast<void>(frontier::config::load_model_config(
                "test/hybrid-empty-arrays"));
        },
        "explicitly empty KDA and MLA arrays must not fall back to all-MLA");
    expect_throws<ConfigError>(
        [] {
            static_cast<void>(frontier::config::load_model_config(
                "test/hybrid-explicit-empty-counterpart"));
        },
        "an explicitly empty counterpart must not be inferred as absent");
    expect_throws<ConfigError>(
        [] {
            static_cast<void>(frontier::config::load_model_config(
                "test/hybrid-explicit-zero-count"));
        },
        "an explicit zero hybrid count must be checked against layer arrays");
    expect_throws<ConfigError>(
        [] {
            static_cast<void>(frontier::config::load_model_config(
                "test/hybrid-asymmetric-heads"));
        },
        "KDA models with asymmetric Q/K/V head counts must fail fast");
    expect_throws<ConfigError>(
        [] {
            static_cast<void>(frontier::config::load_model_config(
                "test/hybrid-asymmetric-dimensions"));
        },
        "KDA models with asymmetric Q/K/V head dimensions must fail fast");

    const auto legacy =
        frontier::config::load_model_config("moonshotai/Kimi-K2-Instruct");
    expect(!legacy.has_kda() && legacy.num_mla_layers == legacy.num_layers &&
               legacy.is_mla_layer(0),
           "metadata-free legacy MLA assets must retain all-layer fallback");
}

void test_cluster_parallelism_is_not_limited_to_one_nvl72_domain() {
    std::string text =
        read_text_file(fixture("analytical_parallel_colocation.json"));
    const std::string original = "\"data_parallel_size\": 2";
    const std::size_t position = text.find(original);
    expect(position != std::string::npos,
           "analytical fixture must expose data parallelism");
    text.replace(position, original.size(), "\"data_parallel_size\": 16");

    const auto parsed = parse_simulation_config_json(text);
    const auto &parallelism = parsed.cluster().parallelism;
    expect(parallelism.num_replicas * parallelism.tensor_parallel_size *
                   parallelism.pipeline_parallel_size *
                   parallelism.data_parallel_size ==
               128,
           "cluster parallelism above 72 accelerators must parse");
}

void test_cpu_kv_cache_contract_and_resolution() {
    auto config = load("fixed_sequential_pdd.json");
    expect(!config.cpu_kv_cache.enabled,
           "omitted CPU KV cache config must default to disabled");
    const std::string disabled = serialize_simulation_config_json(config);
    expect(disabled.find("\"cpu_kv_cache\"") != std::string::npos &&
               parse_simulation_config_json(disabled) == config,
           "normalized config must emit the disabled CPU KV cache object");

    config.prefix_cache.enabled = true;
    config.cluster_scheduler.type = ClusterSchedulerType::kStickyRoundRobin;
    config.cpu_kv_cache.enabled = true;
    config.cpu_kv_cache.capacity_bytes = 1'000'000'000ULL;
    config.cpu_kv_cache.write_bandwidth_gbps = 80.0;
    config.cpu_kv_cache.read_bandwidth_gbps = 40.0;
    const auto parsed =
        parse_simulation_config_json(serialize_simulation_config_json(config));
    const auto resolved = frontier::config::resolve_cpu_kv_cache_target(parsed);
    const auto expected_block =
        frontier::kv_cache_transfer::model_kv_cache_size_bytes(
            parsed.pdd().clusters.prefill.scheduler.block_size,
            parsed.pdd().clusters.prefill.model,
            parsed.pdd().kv_cache_transfer.kv_cache_dtype_size_bytes);
    expect(parsed == config && resolved.enabled &&
               resolved.capacity_bytes == 1'000'000'000ULL &&
               resolved.bytes_per_block == expected_block &&
               resolved.capacity_blocks == 1'000'000'000ULL / expected_block &&
               resolved.d2h_bandwidth_gbps == 80.0 &&
               resolved.h2d_bandwidth_gbps == 40.0,
           "direct CPU KV cache config must round-trip and resolve by model");

    auto cache_aware = config;
    cache_aware.cluster_scheduler.prefill_type =
        ClusterSchedulerType::kCacheAware;
    cache_aware.cluster_scheduler.decode_type =
        ClusterSchedulerType::kVllmQueueAware;
    expect(parse_simulation_config_json(
               serialize_simulation_config_json(cache_aware)) == cache_aware,
           "CPU KV cache must allow actual-cache-aware PREFILL routing");

    auto kimi_target = parsed;
    kimi_target.cpu_kv_cache.capacity_bytes = 10'000'000ULL;
    kimi_target.pdd().clusters.prefill.model =
        frontier::config::load_model_config("moonshotai/Kimi-K2-Instruct");
    kimi_target.pdd().clusters.prefill.parallelism.tensor_parallel_size = 4;
    kimi_target.pdd().clusters.prefill.scheduler.block_size = 16;
    kimi_target.pdd().kv_cache_transfer.kv_cache_dtype_size_bytes = 1.0;
    const auto kimi_resolved =
        frontier::config::resolve_cpu_kv_cache_target(kimi_target);
    expect(kimi_resolved.bytes_per_block == 2'498'560ULL &&
               kimi_resolved.capacity_blocks == 10'000'000ULL / 2'498'560ULL,
           "CPU KV cache must use TP-replicated MLA target bytes per block");

    auto k3_target = parsed;
    k3_target.cpu_kv_cache.capacity_bytes = 1'000'000'000ULL;
    k3_target.pdd().clusters.prefill.model =
        frontier::config::load_model_config("moonshotai/Kimi-K3");
    k3_target.pdd().clusters.prefill.parallelism.tensor_parallel_size = 4;
    k3_target.pdd().clusters.prefill.scheduler.block_size = 16;
    k3_target.pdd().kv_cache_transfer.kv_cache_dtype_size_bytes = 1.0;
    const auto k3_resolved =
        frontier::config::resolve_cpu_kv_cache_target(k3_target);
    expect(k3_resolved.kda_snapshot_bytes == 232'316'928ULL &&
               k3_resolved.kda_snapshot_blocks ==
                   (232'316'928ULL + k3_resolved.bytes_per_block - 1) /
                       k3_resolved.bytes_per_block &&
               k3_resolved.capacity_blocks > k3_resolved.kda_snapshot_blocks,
           "K3 CPU tier must reserve one full atomic BF16 KDA snapshot");

    k3_target.cpu_kv_cache.capacity_bytes = k3_resolved.kda_snapshot_bytes;
    expect_throws<ConfigError>(
        [&k3_target] {
            static_cast<void>(
                frontier::config::resolve_cpu_kv_cache_target(k3_target));
        },
        "K3 CPU tier must leave capacity for at least one MLA KV block");

    kimi_target.pdd()
        .clusters.prefill.parallelism.decode_context_parallel_size = 4;
    expect_throws<ConfigError>(
        [&kimi_target] {
            static_cast<void>(
                frontier::config::resolve_cpu_kv_cache_target(kimi_target));
        },
        "PDD PREFILL must reject DCP greater than one");
    expect_throws<ConfigError>(
        [&kimi_target] {
            static_cast<void>(parse_simulation_config_json(
                serialize_simulation_config_json(kimi_target)));
        },
        "PDD parser must force PREFILL DCP to one");
    kimi_target.pdd()
        .clusters.prefill.parallelism.decode_context_parallel_size = 1;
    kimi_target.pdd().clusters.decode.model =
        kimi_target.pdd().clusters.prefill.model;
    kimi_target.pdd().clusters.decode.parallelism.tensor_parallel_size = 4;
    kimi_target.pdd().clusters.decode.parallelism.decode_context_parallel_size =
        4;
    kimi_target.pdd().clusters.prefill.parallelism.moe_expert_parallel_size = 4;
    kimi_target.pdd().clusters.decode.parallelism.moe_expert_parallel_size = 4;
    const auto kimi_dcp_round_trip = parse_simulation_config_json(
        serialize_simulation_config_json(kimi_target));
    expect(
        kimi_dcp_round_trip.pdd()
                    .clusters.prefill.parallelism
                    .decode_context_parallel_size == 1 &&
            kimi_dcp_round_trip.pdd()
                    .clusters.decode.parallelism.decode_context_parallel_size ==
                4,
        "PDD must keep PREFILL DCP1 and round-trip DECODE DCP");

    kimi_target.pdd().clusters.decode.parallelism.decode_context_parallel_size =
        3;
    expect_throws<ConfigError>(
        [&kimi_target] {
            static_cast<void>(parse_simulation_config_json(
                serialize_simulation_config_json(kimi_target)));
        },
        "MLA DCP size must divide TP size");

    kimi_target.pdd().clusters.decode.parallelism.decode_context_parallel_size =
        0;
    expect_throws<ConfigError>(
        [&kimi_target] {
            static_cast<void>(parse_simulation_config_json(
                serialize_simulation_config_json(kimi_target)));
        },
        "MLA DCP size must be positive");

    auto dense_dcp = parsed;
    dense_dcp.pdd().clusters.decode.parallelism.tensor_parallel_size = 2;
    dense_dcp.pdd().clusters.decode.parallelism.decode_context_parallel_size =
        2;
    expect_throws<ConfigError>(
        [&dense_dcp] {
            static_cast<void>(parse_simulation_config_json(
                serialize_simulation_config_json(dense_dcp)));
        },
        "DCP must reject non-MLA models");

    config.cpu_kv_cache.static_slice_per_gpu = true;
    config.cpu_kv_cache.capacity_bytes_per_gpu = 2'000'000'000ULL;
    config.cpu_kv_cache.dram_bandwidth_gbps_per_gpu = 500.0;
    config.cpu_kv_cache.c2c_bandwidth_gbps_per_gpu = 300.0;
    config.pdd().clusters.prefill.parallelism.tensor_parallel_size = 2;
    config.pdd().clusters.prefill.parallelism.pipeline_parallel_size = 2;
    config.pdd().clusters.prefill.execution_model.fixed.stage_latencies_ms = {
        1.0, 1.0};
    const auto static_parsed =
        parse_simulation_config_json(serialize_simulation_config_json(config));
    const auto static_resolved =
        frontier::config::resolve_cpu_kv_cache_target(static_parsed);
    expect(static_resolved.capacity_bytes == 8'000'000'000ULL &&
               static_resolved.d2h_bandwidth_gbps == 1'200.0 &&
               static_resolved.h2d_bandwidth_gbps == 1'200.0,
           "static per-GPU CPU slices must scale by PREFILL TP times PP");
}

void test_cpu_kv_cache_invalid_combinations() {
    auto config = load("fixed_sequential_pdd.json");
    config.cpu_kv_cache.enabled = true;
    config.cpu_kv_cache.capacity_bytes = 1'000'000'000ULL;
    expect_throws<ConfigError>(
        [&config] {
            static_cast<void>(parse_simulation_config_json(
                serialize_simulation_config_json(config)));
        },
        "CPU KV cache must require session prefix caching");

    config.prefix_cache.enabled = true;
    expect_throws<ConfigError>(
        [&config] {
            static_cast<void>(parse_simulation_config_json(
                serialize_simulation_config_json(config)));
        },
        "CPU KV cache must require sticky target affinity");

    config.cluster_scheduler.type = ClusterSchedulerType::kStickyRoundRobin;
    config.cpu_kv_cache.capacity_bytes = 1;
    expect_throws<ConfigError>(
        [&config] {
            static_cast<void>(parse_simulation_config_json(
                serialize_simulation_config_json(config)));
        },
        "CPU capacity smaller than one logical block must fail");

    config.cpu_kv_cache.capacity_bytes = 1'000'000'000ULL;
    std::string unknown = serialize_simulation_config_json(config);
    const std::string policy = "\"prefix_fit\"";
    unknown.replace(unknown.find(policy), policy.size(), "\"unknown\"");
    expect_throws<ConfigError>(
        [&unknown] {
            static_cast<void>(parse_simulation_config_json(unknown));
        },
        "unknown CPU capacity policies must fail");

    auto colocation = load("fixed_parallel_colocation.json");
    colocation.cpu_kv_cache.enabled = true;
    colocation.cpu_kv_cache.capacity_bytes = 1'000'000'000ULL;
    expect_throws<ConfigError>(
        [&colocation] {
            static_cast<void>(parse_simulation_config_json(
                serialize_simulation_config_json(colocation)));
        },
        "co-location CPU tiering must fail fast");
}

void test_pipeline_stage_profiles_k3_partitions_and_groups() {
    const auto base = load("analytical_parallel_colocation.json").cluster();
    const auto k3 = frontier::config::load_model_config("moonshotai/Kimi-K3");
    for (const std::uint64_t pp : {2ULL, 4ULL, 24ULL}) {
        auto cluster = base;
        cluster.model = k3;
        cluster.parallelism.pipeline_parallel_size = pp;
        cluster.gpu_memory.capacity_bytes_per_gpu =
            10'000'000'000'000'000ULL;
        cluster.gpu_memory.auto_calculate_num_blocks = true;

        const auto first =
            frontier::config::build_pipeline_stage_memory_profiles(cluster);
        const auto second =
            frontier::config::build_pipeline_stage_memory_profiles(cluster);
        expect(first == second && first.size() == pp,
               "K3 PP stage profiles must be deterministic");

        std::uint64_t kda_layers = 0;
        std::uint64_t mla_layers = 0;
        for (const auto &profile : first) {
            expect(profile.layers.begin < profile.layers.end &&
                       profile.layers.end <= k3.num_layers,
                   "K3 stage profile ranges must be nonempty and bounded");
            for (std::uint64_t layer = profile.layers.begin;
                 layer < profile.layers.end; ++layer) {
                kda_layers += static_cast<std::uint64_t>(
                    k3.is_kda_layer(layer));
                mla_layers += static_cast<std::uint64_t>(
                    k3.is_mla_layer(layer));
            }
        }
        expect(kda_layers == 69 && mla_layers == 24,
               "K3 PP stage profiles must preserve 69 KDA and 24 MLA layers");

        const auto groups = frontier::config::build_pipeline_stage_group_catalogue(
            cluster, first);
        const auto groups_again =
            frontier::config::build_pipeline_stage_group_catalogue(cluster, second);
        expect(groups == groups_again &&
                   groups.stage_to_timing_group.size() == pp &&
                   groups.stage_to_memory_group.size() == pp,
               "stage timing/memory groups must be deterministic");
        expect(std::all_of(
                   groups.timing_groups.begin(), groups.timing_groups.end(),
                   [](const auto &group) {
                       return std::all_of(
                           group.ordered_layers.begin(),
                           group.ordered_layers.end(), [](const auto &layer) {
                               return !layer.is_moe || layer.has_dense_mlp;
                           });
                   }),
               "K3 timing signatures must retain shared-expert MLP work");

        frontier::config::resolve_gpu_memory_config(cluster);
        const auto expected_capacity =
            frontier::config::resolve_pipeline_logical_kv_capacity(
                cluster.gpu_memory.pipeline_stage_memory_profiles);
        expect(cluster.scheduler.num_blocks == expected_capacity &&
                   cluster.scheduler.kda_snapshot_blocks_per_session > 0 &&
                   cluster.gpu_memory.pipeline_stage_memory_profiles.size() ==
                       pp,
               "K3 PP automatic memory must install exact N and KDA charge");
    }
}

void test_pipeline_stage_profile_capacity_and_snapshot_charge() {
    using frontier::config::PipelineStageMemoryProfile;

    PipelineStageMemoryProfile kda_only{};
    kda_only.stage_id = frontier::StageId{0};
    kda_only.layers = {0, 1};
    kda_only.free_bytes = 100;
    kda_only.kv_bytes_per_block_by_rank = {0};
    kda_only.kda_snapshot_bytes_by_rank = {50};

    PipelineStageMemoryProfile kv_stage{};
    kv_stage.stage_id = frontier::StageId{1};
    kv_stage.layers = {1, 2};
    kv_stage.free_bytes = 1'000;
    kv_stage.kv_bytes_per_block_by_rank = {10};
    kv_stage.kda_snapshot_bytes_by_rank = {0};

    const std::vector<PipelineStageMemoryProfile> profiles = {kda_only,
                                                               kv_stage};
    std::uint64_t limiting_stage = 0;
    std::uint64_t limiting_rank = 0;
    const auto capacity =
        frontier::config::resolve_pipeline_logical_kv_capacity(
            profiles, &limiting_stage, &limiting_rank);
    expect(capacity == 100 && limiting_stage == 1 && limiting_rank == 0,
           "ordinary KV capacity must ignore B==0 KDA-only ranks");
    const auto charge = frontier::config::resolve_pipeline_kda_snapshot_charge(
        profiles, capacity, &limiting_stage, &limiting_rank);
    expect(charge == 50 && limiting_stage == 0 && limiting_rank == 0,
           "KDA-only stage must contribute normalized snapshot charge");
    expect_throws<ConfigError>(
        [&profiles] {
            auto invalid = profiles;
            invalid[0].free_bytes = 49;
            static_cast<void>(
                frontier::config::resolve_pipeline_kda_snapshot_charge(
                    invalid, 100));
        },
        "KDA snapshot larger than stage free HBM must fail");
}

void test_pipeline_stage_profile_pp1_compatibility_and_uneven_ranges() {
    auto cluster = load("analytical_parallel_colocation.json").cluster();
    cluster.parallelism.pipeline_parallel_size = 1;
    cluster.gpu_memory.capacity_bytes_per_gpu =
        10'000'000'000'000'000ULL;
    cluster.gpu_memory.auto_calculate_num_blocks = true;
    const auto profiles =
        frontier::config::build_pipeline_stage_memory_profiles(cluster);
    expect(profiles.size() == 1 && profiles.front().layers.begin == 0 &&
               profiles.front().layers.end == cluster.model.num_layers,
           "PP1 must retain the full model range in one stage");
    const auto legacy_block =
        frontier::kv_cache_transfer::model_kv_cache_size_bytes_rank_local(
            cluster.scheduler.block_size, cluster.model,
            2.0, cluster.parallelism.decode_context_parallel_size, 0);
    expect(profiles.front().kv_bytes_per_block_by_rank.front() == legacy_block,
           "PP1 stage-local KV bytes must match the legacy full-model helper");

    const auto kimi =
        frontier::config::load_model_config("moonshotai/Kimi-K2-Instruct");
    for (std::uint64_t stage = 0; stage < 4; ++stage) {
        const auto range = frontier::config::pipeline_stage_layer_range(
            kimi.num_layers, 4, stage);
        expect(range.end > range.begin,
               "uneven PP partitions must assign every Kimi K2 stage layers");
    }
    const auto first = frontier::config::pipeline_stage_layer_range(
        kimi.num_layers, 4, 0);
    const auto second = frontier::config::pipeline_stage_layer_range(
        kimi.num_layers, 4, 1);
    expect(first.size() == 16 && second.size() == 15,
           "uneven Kimi K2 PP4 partition must preserve 16/15 stage sizes");
}

} // namespace

int main() {
    int failures = 0;
    failures += frontier::test::run("co-location contract round trip",
                                    test_colocation_contract_round_trip);
    failures += frontier::test::run("PDD contract round trip",
                                    test_pdd_contract_round_trip);
    failures +=
        frontier::test::run("stage-specific cluster scheduler overrides",
                            test_stage_specific_cluster_scheduler_overrides);
    failures += frontier::test::run("analytical contract round trip",
                                    test_analytical_contract_round_trip);
    failures +=
        frontier::test::run("operator precision contract round trip",
                            test_operator_precision_contract_round_trip);
    failures += frontier::test::run("GPU memory auto block calculation",
                                    test_gpu_memory_auto_block_calculation);
    failures += frontier::test::run(
        "GPU memory capacity is required and mode round trips",
        test_gpu_memory_capacity_is_required_and_mode_round_trips);
    failures += frontier::test::run("Kimi K3 gated MLA weight memory",
                                    test_kimi_k3_gated_mla_weight_memory);
    failures += frontier::test::run(
        "Kimi K3 latent MoE projection weight memory",
        test_kimi_k3_latent_moe_projection_weight_memory);
    failures += frontier::test::run(
        "Kimi K3 TP8 DCP8 GPU memory layout",
        test_kimi_k3_tp8_dcp8_gpu_memory_layout);
    failures +=
        frontier::test::run("PDD KV precision matches transfer dtype",
                            test_pdd_kv_precision_matches_transfer_dtype);
    failures += frontier::test::run(
        "PDD KDA snapshot precision contract",
        test_pdd_kda_snapshot_precision_contract);
    failures +=
        frontier::test::run("analytical attention family configs parse",
                            test_analytical_attention_family_configs_parse);
    failures +=
        frontier::test::run("legacy schemas and shapes are rejected",
                            test_legacy_schemas_and_shapes_are_rejected);
    failures += frontier::test::run("invalid surfaces are rejected",
                                    test_invalid_surfaces_are_rejected);
    failures += frontier::test::run(
        "schema range is checked before conversion",
        test_schema_version_range_is_checked_before_conversion);
    failures += frontier::test::run("MoE contract and invalid topologies",
                                    test_moe_contract_and_invalid_topologies);
    failures += frontier::test::run(
        "pipeline-exclusive parallelism contract",
        test_pipeline_exclusive_contract);
    failures += frontier::test::run("model registry and attention binding",
                                    test_model_registry_and_attention_binding);
    failures +=
        frontier::test::run("nested Kimi K3 model asset and aliases",
                            test_kimi_k3_nested_model_asset_and_aliases);
    failures += frontier::test::run(
        "explicit hybrid attention metadata is strict",
        test_explicit_hybrid_attention_metadata_is_strict);
    failures += frontier::test::run("model runtime overrides are optional",
                                    test_model_runtime_overrides_are_optional);
    failures += frontier::test::run(
        "cluster parallelism above NVL72 parses",
        test_cluster_parallelism_is_not_limited_to_one_nvl72_domain);
    failures += frontier::test::run("CPU KV cache contract and resolution",
                                    test_cpu_kv_cache_contract_and_resolution);
    failures += frontier::test::run("CPU KV cache invalid combinations",
                                    test_cpu_kv_cache_invalid_combinations);
    failures += frontier::test::run(
        "PP stage profiles K3 partitions and groups",
        test_pipeline_stage_profiles_k3_partitions_and_groups);
    failures += frontier::test::run(
        "PP stage profile capacity and snapshot charge",
        test_pipeline_stage_profile_capacity_and_snapshot_charge);
    failures += frontier::test::run(
        "PP stage profile PP1 compatibility and uneven ranges",
        test_pipeline_stage_profile_pp1_compatibility_and_uneven_ranges);
    return failures == 0 ? 0 : 1;
}
