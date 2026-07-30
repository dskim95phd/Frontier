#include "frontier/config/config.h"
#include "tests/test_support.h"

#include <filesystem>
#include <string>

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
               moe.cluster().model.model_num_experts == 16 &&
               moe.cluster().model.runtime_total_experts == 8 &&
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
    bad_expert_partition.cluster().model.runtime_total_experts = 6;
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
        frontier::test::run("legacy schemas and shapes are rejected",
                            test_legacy_schemas_and_shapes_are_rejected);
    failures += frontier::test::run("invalid surfaces are rejected",
                                    test_invalid_surfaces_are_rejected);
    failures += frontier::test::run(
        "schema range is checked before conversion",
        test_schema_version_range_is_checked_before_conversion);
    failures += frontier::test::run("MoE contract and invalid topologies",
                                    test_moe_contract_and_invalid_topologies);
    return failures == 0 ? 0 : 1;
}
