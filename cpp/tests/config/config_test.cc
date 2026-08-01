#include "frontier/config/config.h"
#include "tests/test_support.h"

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>

#include "frontier/attention/ops.h"

#ifndef FRONTIER_TEST_FIXTURE_DIR
#error "FRONTIER_TEST_FIXTURE_DIR must be defined for contract tests"
#endif

namespace {

using frontier::config::ConfigError;
using frontier::config::ExecutionModelType;
using frontier::config::kSchemaVersion;
using frontier::config::parse_simulation_config_json;
using frontier::config::PddRuntimeConfig;
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

void test_colocation_contract_round_trip() {
    const auto config = load("fixed_parallel_colocation.json");
    expect(config.schema_version == kSchemaVersion,
           "the current schema version must parse");
    expect(config.system_architecture == SystemArchitecture::kCoLocation,
           "co-location architecture must parse");
    expect(config.cluster().parallelism.num_replicas == 2 &&
               config.cluster().parallelism.data_parallel_size == 2,
           "monolithic cluster topology must parse");
    expect(config.cluster().execution_model.type == ExecutionModelType::kFixed,
           "fixed execution model must parse");
    expect(parse_simulation_config_json(
               serialize_simulation_config_json(config)) == config,
           "co-location config must round-trip deterministically");
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

void test_analytical_contract_round_trip() {
    const auto config = load("analytical_parallel_colocation.json");
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

    const std::string serialized = serialize_simulation_config_json(config);
    const auto parsed = parse_simulation_config_json(serialized);
    const auto &resolved = parsed.cluster().execution_model.analytical;
    expect(parsed == config && resolved.attention_precision() == "fp8" &&
               resolved.dense_precision() == "fp8" &&
               resolved.moe_expert_precision() == "fp4" &&
               resolved.moe_router_precision() == "fp8" &&
               resolved.kv_cache_precision() == "fp8" &&
               resolved.communication_precision() == "fp8",
           "operator precision overrides must round-trip and resolve");

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
    expect_throws<ConfigError>(
        [&colocation] {
            static_cast<void>(serialize_simulation_config_json(colocation));
        },
        "prefix caching must remain deferred");

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
               kimi.runtime_head_size() == 576,
           "Kimi K2 asset must expose Python-equivalent MoE and MLA fields");

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

} // namespace

int main() {
    int failures = 0;
    failures += frontier::test::run("co-location contract round trip",
                                    test_colocation_contract_round_trip);
    failures += frontier::test::run("PDD contract round trip",
                                    test_pdd_contract_round_trip);
    failures += frontier::test::run("analytical contract round trip",
                                    test_analytical_contract_round_trip);
    failures +=
        frontier::test::run("operator precision contract round trip",
                            test_operator_precision_contract_round_trip);
    failures +=
        frontier::test::run("PDD KV precision matches transfer dtype",
                            test_pdd_kv_precision_matches_transfer_dtype);
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
    failures += frontier::test::run("model registry and attention binding",
                                    test_model_registry_and_attention_binding);
    failures += frontier::test::run("model runtime overrides are optional",
                                    test_model_runtime_overrides_are_optional);
    failures += frontier::test::run(
        "cluster parallelism above NVL72 parses",
        test_cluster_parallelism_is_not_limited_to_one_nvl72_domain);
    return failures == 0 ? 0 : 1;
}
