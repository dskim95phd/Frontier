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
using frontier::request_generator::materialize_workload_for_config;
using frontier::request_generator::parse_workload_csv;
using frontier::request_generator::serialize_workload_csv;
using frontier::request_generator::validate_workload_for_config;
using frontier::request_generator::WorkloadError;
using frontier::test::expect;
using frontier::test::expect_throws;
using frontier::test::read_text_file;

auto load_config() {
    const std::filesystem::path fixture_root{FRONTIER_TEST_FIXTURE_DIR};
    return parse_simulation_config_json(
        read_text_file(fixture_root / "config/fixed_parallel_colocation.json"));
}

void test_valid_workload_round_trip() {
    const auto requests =
        parse_workload_csv("session_start_at,think_time,num_prefill_tokens,num_"
                           "decode_tokens,session_id,"
                           "session_turn_index\n"
                           "0,0,32,8,7,0\n"
                           ",1.5,16,4,7,1\n"
                           "2,0,8,2,,\n");

    expect(requests.size() == 3, "all workload rows must parse");
    expect(requests[0].request_id.value() == 0, "request IDs start at zero");
    expect(requests[1].request_id.value() == 1, "request IDs follow row order");
    expect(!requests[1].session_start_at.valid() &&
               requests[1].think_time.seconds() == 1.5,
           "successor think time must parse in seconds");
    expect(requests[1].session_id.value() == 7, "session ID must parse");
    expect(!requests[2].session_id.valid(),
           "empty optional session ID must remain absent");

    const std::string serialized = serialize_workload_csv(requests);
    expect(parse_workload_csv(serialized) == requests,
           "serialized workload must round-trip");
}

void test_checked_in_workload_fixture() {
    const std::filesystem::path fixture_root{FRONTIER_TEST_FIXTURE_DIR};
    auto config = load_config();
    config.prefix_cache.enabled = true;
    const auto requests = parse_workload_csv(
        read_text_file(fixture_root / "workloads/session_prefix.csv"));
    validate_workload_for_config(requests, config);
    expect(requests.size() == 3,
           "checked-in workload fixture must remain valid");
}

void test_header_contract_is_strict() {
    expect_throws<WorkloadError>(
        [] {
            static_cast<void>(parse_workload_csv(
                "arrived_at,num_prefill_tokens,num_decode_tokens\n0,1,1\n"));
        },
        "legacy absolute request arrivals must be rejected");
    expect_throws<WorkloadError>(
        [] {
            static_cast<void>(parse_workload_csv(
                "session_start_at,think_time,num_prefill_tokens\n0,0,1\n"));
        },
        "missing required columns must be rejected");
    expect_throws<WorkloadError>(
        [] {
            static_cast<void>(parse_workload_csv(
                "session_start_at,think_time,num_prefill_tokens,"
                "num_decode_tokens,think_time\n0,0,1,1,0\n"));
        },
        "duplicate columns must be rejected");
    expect_throws<WorkloadError>(
        [] {
            static_cast<void>(parse_workload_csv(
                "session_start_at,think_time,num_prefill_tokens,"
                "num_decode_tokens,unknown\n0,0,1,1,0\n"));
        },
        "unknown columns must be rejected");
    expect_throws<WorkloadError>(
        [] {
            static_cast<void>(
                parse_workload_csv("session_start_at,think_time,num_prefill_"
                                   "tokens,num_decode_tokens,"
                                   "block_hash_ids\n"
                                   "0,0,1,1,123\n"));
        },
        "block_hash_ids must be rejected");
}

void test_invalid_values_are_rejected() {
    for (const std::string_view row : {
             "-1,0,32,8,7\n",
             "nan,0,32,8,7\n",
             "0,-1,32,8,7\n",
             "0,0,0,8,7\n",
             "0,0,-1,8,7\n",
             "0,0,32,0,7\n",
             "0,0,1.5,8,7\n",
         }) {
        expect_throws<WorkloadError>(
            [row] {
                static_cast<void>(parse_workload_csv(
                    std::string{
                        "session_start_at,think_time,num_prefill_tokens,"
                        "num_decode_tokens,"
                        "session_id\n"} +
                    std::string{row}));
            },
            "invalid workload values must be rejected");
    }

    expect_throws<WorkloadError>(
        [] {
            static_cast<void>(
                parse_workload_csv("session_start_at,think_time,num_prefill_"
                                   "tokens,num_decode_tokens,session_id,"
                                   "session_turn_index\n"
                                   "0,0,32,8,,1\n"));
        },
        "turn index without session ID must be rejected");
}

void test_session_prefix_sequence_validation() {
    auto config = load_config();
    config.prefix_cache.enabled = true;
    const auto valid =
        parse_workload_csv("session_start_at,think_time,num_prefill_tokens,num_"
                           "decode_tokens,session_id,"
                           "session_turn_index\n"
                           "0,0,32,8,7,0\n"
                           "1,0,16,8,9,0\n"
                           ",2,16,8,7,1\n");
    validate_workload_for_config(valid, config);

    const auto missing_session =
        parse_workload_csv("session_start_at,think_time,num_prefill_tokens,"
                           "num_decode_tokens\n0,0,32,8\n");
    expect_throws<WorkloadError>(
        [&missing_session, &config] {
            validate_workload_for_config(missing_session, config);
        },
        "prefix caching must require session IDs");

    expect_throws<WorkloadError>(
        [] {
            static_cast<void>(parse_workload_csv(
                "session_start_at,think_time,num_prefill_tokens,"
                "num_decode_tokens,session_id\n"
                "10,0,32,8,7\n5,1,16,8,7\n"));
        },
        "successor turns must omit session_start_at");

    expect_throws<WorkloadError>(
        [] {
            static_cast<void>(parse_workload_csv(
                "session_start_at,think_time,num_prefill_tokens,"
                "num_decode_tokens,session_id,session_turn_index\n"
                "0,0,32,8,7,1\n,1,16,8,7,1\n"));
        },
        "session turn indices must be strictly increasing");
}

void test_session_prefix_materialization_is_session_local() {
    auto config = load_config();
    config.prefix_cache.enabled = true;
    const auto raw =
        parse_workload_csv("session_start_at,think_time,num_prefill_tokens,num_"
                           "decode_tokens,session_id,"
                           "session_turn_index\n"
                           "0,0,32,8,7,0\n"
                           "1,0,24,4,9,0\n"
                           ",2,8,2,7,1\n"
                           ",3,4,2,9,1\n");
    const auto materialized = materialize_workload_for_config(raw, config);
    expect(materialized[0].num_prefill_tokens == 32 &&
               materialized[1].num_prefill_tokens == 24 &&
               materialized[2].num_prefill_tokens == 48 &&
               materialized[3].num_prefill_tokens == 32,
           "incremental prefill lengths must accumulate independently per "
           "session");
    expect(raw[2].num_prefill_tokens == 8,
           "materialization must not mutate raw workload rows");

    const auto overflowing = parse_workload_csv(
        "session_start_at,think_time,num_prefill_tokens,num_decode_tokens,"
        "session_id\n0,0,18446744073709551614,1,7\n"
        ",1,1,1,7\n");
    expect_throws<WorkloadError>(
        [&] {
            static_cast<void>(
                materialize_workload_for_config(overflowing, config));
        },
        "materialized session context overflow must be rejected");
}

} // namespace

int main() {
    int failures = 0;
    failures += frontier::test::run("valid workload round trip",
                                    test_valid_workload_round_trip);
    failures += frontier::test::run("checked-in workload fixture",
                                    test_checked_in_workload_fixture);
    failures += frontier::test::run("header contract is strict",
                                    test_header_contract_is_strict);
    failures += frontier::test::run("invalid values are rejected",
                                    test_invalid_values_are_rejected);
    failures += frontier::test::run("session prefix sequence validation",
                                    test_session_prefix_sequence_validation);
    failures += frontier::test::run(
        "session prefix materialization is session local",
        test_session_prefix_materialization_is_session_local);
    return failures == 0 ? 0 : 1;
}
