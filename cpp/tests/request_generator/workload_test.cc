#include "frontier/config/config.h"
#include "frontier/request_generator/workload.h"
#include "tests/test_support.h"

#include <filesystem>
#include <string>
#include <string_view>

#ifndef FRONTIER_TEST_FIXTURE_DIR
#error "FRONTIER_TEST_FIXTURE_DIR must be defined for contract tests"
#endif

namespace {

using frontier::config::parse_simulation_config_json;
using frontier::request_generator::WorkloadError;
using frontier::request_generator::parse_workload_csv;
using frontier::request_generator::serialize_workload_csv;
using frontier::request_generator::validate_workload_for_config;
using frontier::test::expect;
using frontier::test::expect_throws;
using frontier::test::read_text_file;

auto load_config() {
  const std::filesystem::path fixture_root{
      FRONTIER_TEST_FIXTURE_DIR};
  return parse_simulation_config_json(read_text_file(
      fixture_root /
      "config/fixed_parallel_colocation.json"));
}

void test_valid_workload_round_trip() {
  const auto requests = parse_workload_csv(
      "arrived_at,num_prefill_tokens,num_decode_tokens,session_id,"
      "session_turn_index\n"
      "0,32,8,7,0\n"
      "1.5,16,4,7,1\n"
      "2,8,2,,\n");

  expect(requests.size() == 3, "all workload rows must parse");
  expect(requests[0].request_id.value() == 0, "request IDs start at zero");
  expect(requests[1].request_id.value() == 1, "request IDs follow row order");
  expect(
      requests[1].arrived_at.seconds() == 1.5,
      "arrival time must parse in seconds");
  expect(
      requests[1].session_id.value() == 7,
      "session ID must parse");
  expect(
      !requests[2].session_id.valid(),
      "empty optional session ID must remain absent");

  const std::string serialized = serialize_workload_csv(requests);
  expect(
      parse_workload_csv(serialized) == requests,
      "serialized workload must round-trip");
}

void test_checked_in_workload_fixture() {
  const std::filesystem::path fixture_root{
      FRONTIER_TEST_FIXTURE_DIR};
  auto config = load_config();
  config.prefix_cache.enabled = true;
  const auto requests = parse_workload_csv(
      read_text_file(
          fixture_root / "workloads/session_prefix.csv"));
  validate_workload_for_config(requests, config);
  expect(
      requests.size() == 3,
      "checked-in workload fixture must remain valid");
}

void test_header_contract_is_strict() {
  expect_throws<WorkloadError>(
      [] {
        static_cast<void>(parse_workload_csv(
            "arrived_at,num_prefill_tokens\n0,1\n"));
      },
      "missing required columns must be rejected");
  expect_throws<WorkloadError>(
      [] {
        static_cast<void>(parse_workload_csv(
            "arrived_at,num_prefill_tokens,num_decode_tokens,arrived_at\n"
            "0,1,1,0\n"));
      },
      "duplicate columns must be rejected");
  expect_throws<WorkloadError>(
      [] {
        static_cast<void>(parse_workload_csv(
            "arrived_at,num_prefill_tokens,num_decode_tokens,unknown\n"
            "0,1,1,0\n"));
      },
      "unknown columns must be rejected");
  expect_throws<WorkloadError>(
      [] {
        static_cast<void>(parse_workload_csv(
            "arrived_at,num_prefill_tokens,num_decode_tokens,"
            "block_hash_ids\n"
            "0,1,1,123\n"));
      },
      "block_hash_ids must be rejected");
}

void test_invalid_values_are_rejected() {
  for (const std::string_view row : {
           "-1,32,8,7\n",
           "nan,32,8,7\n",
           "0,0,8,7\n",
           "0,-1,8,7\n",
           "0,32,0,7\n",
           "0,1.5,8,7\n",
       }) {
    expect_throws<WorkloadError>(
        [row] {
          static_cast<void>(parse_workload_csv(
              std::string{
                  "arrived_at,num_prefill_tokens,num_decode_tokens,"
                  "session_id\n"} +
              std::string{row}));
        },
        "invalid workload values must be rejected");
  }

  expect_throws<WorkloadError>(
      [] {
        static_cast<void>(parse_workload_csv(
            "arrived_at,num_prefill_tokens,num_decode_tokens,session_id,"
            "session_turn_index\n"
            "0,32,8,,1\n"));
      },
      "turn index without session ID must be rejected");
}

void test_session_prefix_sequence_validation() {
  auto config = load_config();
  config.prefix_cache.enabled = true;
  const auto valid = parse_workload_csv(
      "arrived_at,num_prefill_tokens,num_decode_tokens,session_id,"
      "session_turn_index\n"
      "0,32,8,7,0\n"
      "1,16,8,9,0\n"
      "2,16,8,7,1\n");
  validate_workload_for_config(valid, config);

  const auto missing_session = parse_workload_csv(
      "arrived_at,num_prefill_tokens,num_decode_tokens\n"
      "0,32,8\n");
  expect_throws<WorkloadError>(
      [&missing_session, &config] {
        validate_workload_for_config(missing_session, config);
      },
      "prefix caching must require session IDs");

  const auto decreasing_arrival = parse_workload_csv(
      "arrived_at,num_prefill_tokens,num_decode_tokens,session_id\n"
      "10,32,8,7\n"
      "5,16,8,7\n");
  expect_throws<WorkloadError>(
      [&decreasing_arrival, &config] {
        validate_workload_for_config(decreasing_arrival, config);
      },
      "session arrivals must be nondecreasing");

  const auto duplicate_turn = parse_workload_csv(
      "arrived_at,num_prefill_tokens,num_decode_tokens,session_id,"
      "session_turn_index\n"
      "0,32,8,7,1\n"
      "1,16,8,7,1\n");
  expect_throws<WorkloadError>(
      [&duplicate_turn, &config] {
        validate_workload_for_config(duplicate_turn, config);
      },
      "session turn indices must be strictly increasing");
}

}  // namespace

int main() {
  int failures = 0;
  failures += frontier::test::run(
      "valid workload round trip",
      test_valid_workload_round_trip);
  failures += frontier::test::run(
      "checked-in workload fixture",
      test_checked_in_workload_fixture);
  failures += frontier::test::run(
      "header contract is strict",
      test_header_contract_is_strict);
  failures += frontier::test::run(
      "invalid values are rejected",
      test_invalid_values_are_rejected);
  failures += frontier::test::run(
      "session prefix sequence validation",
      test_session_prefix_sequence_validation);
  return failures == 0 ? 0 : 1;
}
