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

constexpr std::string_view kValidSchedulerConfig = R"json(
{
  "schema_version": 2,
  "run_id": "step2-contract",
  "simulation_mode": "online",
  "system_architecture": "co-location",
  "enable_parallel_clusters": false,
  "prefix_cache": {
    "enabled": false,
    "key_mode": "session"
  },
  "scheduler": {
    "type": "vllm_v1",
    "scheduling_policy": "fcfs",
    "batch_size_cap": 8,
    "max_tokens_in_batch": 128,
    "enable_preemption": true,
    "enable_chunked_prefill": true,
    "long_prefill_token_threshold": 64,
    "block_size": 16,
    "num_blocks": 128,
    "watermark_blocks_fraction": 0.01,
    "num_preallocate_tokens": 0
  },
  "execution_model": {
    "type": "fixed",
    "batch_latency_ms": 1.0
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

void test_checked_in_step2_configs() {
  const std::filesystem::path fixture_root{
      FRONTIER_TEST_FIXTURE_DIR};
  const auto fixed = parse_simulation_config_json(read_text_file(
      fixture_root / "config/step2_fixed_colocation.json"));
  const auto analytical = parse_simulation_config_json(read_text_file(
      fixture_root / "config/step2_analytical_colocation.json"));
  expect(
      fixed.execution_model->type ==
          frontier::config::ExecutionModelType::kFixed,
      "fixed Step 2 fixture must select fixed timing");
  expect(
      analytical.execution_model->type ==
          frontier::config::ExecutionModelType::kAnalytical,
      "analytical Step 2 fixture must select analytical timing");
  expect(
      parse_simulation_config_json(
          serialize_simulation_config_json(analytical)) == analytical,
      "analytical schema v2 must round-trip");

  std::string unsupported_analytical =
      serialize_simulation_config_json(analytical);
  unsupported_analytical.replace(
      unsupported_analytical.find("\"rubin\""),
      std::string{"\"rubin\""}.size(),
      "\"a100\"");
  expect_throws<ConfigError>(
      [&unsupported_analytical] {
        static_cast<void>(parse_simulation_config_json(
            unsupported_analytical));
      },
      "unsupported analytical hardware must fail fast");
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

void test_scheduler_config_v2_round_trip() {
  const auto config =
      parse_simulation_config_json(kValidSchedulerConfig);
  expect(config.schema_version == 2, "scheduler schema must parse");
  expect(config.scheduler.has_value(), "scheduler config is required");
  expect(
      config.scheduler->enable_preemption &&
          config.scheduler->enable_chunked_prefill,
      "scheduler behavior flags must parse");
  expect(
      config.execution_model.has_value() &&
          config.execution_model->fixed.batch_latency_ms == 1.0,
      "fixed execution model must parse");
  expect(
      parse_simulation_config_json(
          serialize_simulation_config_json(config)) == config,
      "schema v2 must round-trip deterministically");
}

void test_scheduler_config_rejects_deferred_surfaces() {
  std::string prefix{kValidSchedulerConfig};
  prefix.replace(
      prefix.find("\"enabled\": false"),
      std::string{"\"enabled\": false"}.size(),
      "\"enabled\": true");
  expect_throws<ConfigError>(
      [&prefix] {
        static_cast<void>(
            parse_simulation_config_json(prefix));
      },
      "Step 2 prefix caching must be rejected");

  std::string priority{kValidSchedulerConfig};
  priority.replace(
      priority.find("\"fcfs\""),
      std::string{"\"fcfs\""}.size(),
      "\"priority\"");
  expect_throws<ConfigError>(
      [&priority] {
        static_cast<void>(
            parse_simulation_config_json(priority));
      },
      "Step 2 priority scheduling must be rejected");

  std::string pdd{kValidSchedulerConfig};
  pdd.replace(
      pdd.find("\"co-location\""),
      std::string{"\"co-location\""}.size(),
      "\"pd-disaggregation\"");
  expect_throws<ConfigError>(
      [&pdd] {
        static_cast<void>(parse_simulation_config_json(pdd));
      },
      "Step 2 PDD must be rejected");

  std::string parallel{kValidSchedulerConfig};
  parallel.replace(
      parallel.find("\"enable_parallel_clusters\": false"),
      std::string{"\"enable_parallel_clusters\": false"}.size(),
      "\"enable_parallel_clusters\": true");
  expect_throws<ConfigError>(
      [&parallel] {
        static_cast<void>(parse_simulation_config_json(parallel));
      },
      "Step 2 parallel clusters must be rejected");

  std::string block_hash{kValidSchedulerConfig};
  block_hash.replace(
      block_hash.find("\"session\""),
      std::string{"\"session\""}.size(),
      "\"block_hash\"");
  expect_throws<ConfigError>(
      [&block_hash] {
        static_cast<void>(parse_simulation_config_json(block_hash));
      },
      "Step 2 block-hash key mode must be rejected");
}

void test_scheduler_capacity_validation() {
  std::string zero_batch{kValidSchedulerConfig};
  zero_batch.replace(
      zero_batch.find("\"batch_size_cap\": 8"),
      std::string{"\"batch_size_cap\": 8"}.size(),
      "\"batch_size_cap\": 0");
  expect_throws<ConfigError>(
      [&zero_batch] {
        static_cast<void>(
            parse_simulation_config_json(zero_batch));
      },
      "zero batch size must be rejected");

  std::string invalid_watermark{kValidSchedulerConfig};
  invalid_watermark.replace(
      invalid_watermark.find(
          "\"watermark_blocks_fraction\": 0.01"),
      std::string{
          "\"watermark_blocks_fraction\": 0.01"}.size(),
      "\"watermark_blocks_fraction\": 1.0");
  expect_throws<ConfigError>(
      [&invalid_watermark] {
        static_cast<void>(
            parse_simulation_config_json(invalid_watermark));
      },
      "watermark outside [0, 1) must be rejected");

  std::string threshold_without_chunking{kValidSchedulerConfig};
  threshold_without_chunking.replace(
      threshold_without_chunking.find(
          "\"enable_chunked_prefill\": true"),
      std::string{
          "\"enable_chunked_prefill\": true"}.size(),
      "\"enable_chunked_prefill\": false");
  expect_throws<ConfigError>(
      [&threshold_without_chunking] {
        static_cast<void>(parse_simulation_config_json(
            threshold_without_chunking));
      },
      "long-prefill threshold must require chunking");
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
      "\"schema_version\": 3");
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

void test_schema_version_range_is_checked_before_conversion() {
  for (const std::string_view out_of_range_version :
       {"4294967298", "-4294967295"}) {
    std::string schema{kValidConfig};
    schema.replace(
        schema.find("\"schema_version\": 1"),
        std::string{"\"schema_version\": 1"}.size(),
        "\"schema_version\": " +
            std::string{out_of_range_version});
    expect_throws<ConfigError>(
        [&schema] {
          static_cast<void>(parse_simulation_config_json(schema));
        },
        "out-of-range schema versions must be rejected before conversion");
  }
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
      "checked-in Step 2 configs",
      test_checked_in_step2_configs);
  failures += frontier::test::run(
      "sequential PDD is accepted",
      test_sequential_pdd_is_accepted);
  failures += frontier::test::run(
      "scheduler config v2 round trip",
      test_scheduler_config_v2_round_trip);
  failures += frontier::test::run(
      "scheduler config rejects deferred surfaces",
      test_scheduler_config_rejects_deferred_surfaces);
  failures += frontier::test::run(
      "scheduler capacity validation",
      test_scheduler_capacity_validation);
  failures += frontier::test::run(
      "malformed and incomplete configs are rejected",
      test_malformed_and_incomplete_configs_are_rejected);
  failures += frontier::test::run(
      "unsupported schema and features are rejected",
      test_unsupported_schema_and_features_are_rejected);
  failures += frontier::test::run(
      "schema version range is checked before conversion",
      test_schema_version_range_is_checked_before_conversion);
  return failures == 0 ? 0 : 1;
}
