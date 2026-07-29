#include "frontier/config/config.h"
#include "tests/test_support.h"

#include <filesystem>
#include <string>

#ifndef FRONTIER_TEST_FIXTURE_DIR
#error "FRONTIER_TEST_FIXTURE_DIR must be defined for contract tests"
#endif

namespace {

using frontier::config::ConfigError;
using frontier::config::SimulationMode;
using frontier::config::SystemArchitecture;
using frontier::config::parse_simulation_config_json;
using frontier::config::serialize_simulation_config_json;
using frontier::test::expect;
using frontier::test::expect_throws;
using frontier::test::read_text_file;

constexpr std::string_view kValidConfig = R"json(
{
  "schema_version": 1,
  "run_id": "step1-contract",
  "simulation_mode": "offline",
  "system_architecture": "co-location",
  "enable_parallel_clusters": false,
  "prefix_cache": {
    "enabled": true,
    "key_mode": "session"
  }
}
)json";

void test_valid_config_round_trip() {
  const auto config = parse_simulation_config_json(kValidConfig);
  expect(config.schema_version == 1, "schema version must parse");
  expect(config.run_id == "step1-contract", "run_id must parse");
  expect(
      config.simulation_mode == SimulationMode::kOffline,
      "simulation mode must parse");
  expect(
      config.system_architecture == SystemArchitecture::kCoLocation,
      "system architecture must parse");
  expect(config.prefix_cache.enabled, "prefix cache flag must parse");

  const std::string serialized =
      serialize_simulation_config_json(config);
  expect(
      parse_simulation_config_json(serialized) == config,
      "serialized config must round-trip");
  expect(
      serialized.find("\"schema_version\"") <
          serialized.find("\"run_id\""),
      "serialized config field order must be stable");
}

void test_checked_in_config_fixture() {
  const std::filesystem::path fixture =
      std::filesystem::path{FRONTIER_TEST_FIXTURE_DIR} /
      "config/minimal_colocation.json";
  const auto config =
      parse_simulation_config_json(read_text_file(fixture));
  expect(
      config.run_id == "step1-minimal-colocation",
      "checked-in config fixture must remain valid");
}

void test_sequential_pdd_is_accepted() {
  std::string config{kValidConfig};
  const std::string old_value = "\"co-location\"";
  const std::string new_value = "\"pd-disaggregation\"";
  config.replace(config.find(old_value), old_value.size(), new_value);
  const auto parsed = parse_simulation_config_json(config);
  expect(
      parsed.system_architecture ==
          SystemArchitecture::kPdDisaggregation,
      "sequential PDD must be part of the schema");
}

void test_malformed_and_incomplete_configs_are_rejected() {
  expect_throws<ConfigError>(
      [] { static_cast<void>(parse_simulation_config_json("{")); },
      "malformed JSON must be rejected");
  expect_throws<ConfigError>(
      [] {
        static_cast<void>(parse_simulation_config_json(
            R"({"schema_version": 1})"));
      },
      "missing fields must be rejected");
  expect_throws<ConfigError>(
      [] {
        static_cast<void>(parse_simulation_config_json(R"json(
          {
            "schema_version": 1,
            "run_id": "x",
            "simulation_mode": "offline",
            "system_architecture": "co-location",
            "enable_parallel_clusters": false,
            "prefix_cache": {
              "enabled": false,
              "key_mode": "session"
            },
            "unexpected": true
          }
        )json"));
      },
      "unknown fields must be rejected");
}

void test_unsupported_schema_and_features_are_rejected() {
  std::string schema{kValidConfig};
  schema.replace(
      schema.find("\"schema_version\": 1"),
      std::string{"\"schema_version\": 1"}.size(),
      "\"schema_version\": 2");
  expect_throws<ConfigError>(
      [&schema] {
        static_cast<void>(parse_simulation_config_json(schema));
      },
      "unknown schema versions must be rejected");

  std::string block_hash{kValidConfig};
  block_hash.replace(
      block_hash.find("\"session\""),
      std::string{"\"session\""}.size(),
      "\"block_hash\"");
  expect_throws<ConfigError>(
      [&block_hash] {
        static_cast<void>(parse_simulation_config_json(block_hash));
      },
      "block-hash key mode must be rejected");

  std::string parallel{kValidConfig};
  parallel.replace(
      parallel.find("\"enable_parallel_clusters\": false"),
      std::string{"\"enable_parallel_clusters\": false"}.size(),
      "\"enable_parallel_clusters\": true");
  expect_throws<ConfigError>(
      [&parallel] {
        static_cast<void>(parse_simulation_config_json(parallel));
      },
      "parallel clusters must be rejected");
}

}  // namespace

int main() {
  int failures = 0;
  failures += frontier::test::run(
      "valid config round trip",
      test_valid_config_round_trip);
  failures += frontier::test::run(
      "checked-in config fixture",
      test_checked_in_config_fixture);
  failures += frontier::test::run(
      "sequential PDD is accepted",
      test_sequential_pdd_is_accepted);
  failures += frontier::test::run(
      "malformed and incomplete configs are rejected",
      test_malformed_and_incomplete_configs_are_rejected);
  failures += frontier::test::run(
      "unsupported schema and features are rejected",
      test_unsupported_schema_and_features_are_rejected);
  return failures == 0 ? 0 : 1;
}
