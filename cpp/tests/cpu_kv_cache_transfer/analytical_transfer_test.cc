#include "frontier/cpu_kv_cache_transfer/analytical_transfer.h"
#include "frontier/core/event_queue.h"
#include "frontier/entities/cpu_kv_cache_transfer_info.h"
#include "frontier/metrics/output_contract.h"
#include "tests/test_support.h"

#include <cmath>
#include <limits>

namespace {

using frontier::CpuKVCacheOffloadEndPayload;
using frontier::CpuKVCacheOffloadStartPayload;
using frontier::CpuKvTransferId;
using frontier::EventQueue;
using frontier::EventType;
using frontier::SimTime;
using frontier::cpu_kv_cache_transfer::AnalyticalCpuKVCacheTransferEngine;
using frontier::cpu_kv_cache_transfer::CpuTransferDirection;
using frontier::cpu_kv_cache_transfer::CpuTransferEngineConfig;
using frontier::cpu_kv_cache_transfer::CpuTransferModelError;
using frontier::test::expect;
using frontier::test::expect_throws;

CpuTransferEngineConfig config() {
    CpuTransferEngineConfig result{};
    result.d2h_bandwidth_gbps = 1.0;
    result.d2h_latency_ms = 0.25;
    result.h2d_bandwidth_gbps = 2.0;
    result.h2d_latency_ms = 0.5;
    return result;
}

void test_directional_serialization_and_full_duplex() {
    AnalyticalCpuKVCacheTransferEngine engine{config()};
    const auto d2h_first = engine.schedule(
        CpuTransferDirection::kD2H, 125'000'000, SimTime::from_seconds(0.0));
    const auto d2h_second = engine.schedule(
        CpuTransferDirection::kD2H, 125'000'000, SimTime::from_seconds(0.0));
    const auto h2d = engine.schedule(
        CpuTransferDirection::kH2D, 250'000'000, SimTime::from_seconds(0.0));
    expect(d2h_second.started_at == d2h_first.completed_at &&
               d2h_second.queue_time_ms > 0.0,
           "D2H transfers must serialize and expose queue time");
    expect(h2d.started_at == SimTime::from_seconds(0.0) &&
               h2d.completed_at > d2h_first.started_at,
           "H2D must overlap an independently queued D2H transfer");

    const auto h2d_second = engine.schedule(
        CpuTransferDirection::kH2D, 1, SimTime::from_seconds(0.0));
    expect(h2d_second.started_at == h2d.completed_at &&
               h2d_second.queue_time_ms > 0.0,
           "H2D transfers must serialize independently");
}

void test_zero_bytes_validation_and_large_values() {
    AnalyticalCpuKVCacheTransferEngine engine{config()};
    const auto zero = engine.schedule(CpuTransferDirection::kD2H, 0,
                                      SimTime::from_seconds(3.0));
    expect(zero.service_time_ms == 0.25 &&
               zero.completed_at > zero.started_at,
           "zero-byte transfer contract must charge fixed latency");
    const auto large = engine.schedule(
        CpuTransferDirection::kH2D,
        std::numeric_limits<std::uint64_t>::max(),
        SimTime::from_seconds(1.0e100));
    expect(large.started_at.valid() && large.completed_at.valid() &&
               std::isfinite(large.service_time_ms),
           "large finite timestamps and sizes must not overflow");

    auto invalid = config();
    invalid.d2h_bandwidth_gbps =
        std::numeric_limits<double>::quiet_NaN();
    expect_throws<CpuTransferModelError>(
        [&invalid] {
            static_cast<void>(AnalyticalCpuKVCacheTransferEngine{invalid});
        },
        "nonfinite CPU bandwidth must fail fast");
    invalid = config();
    invalid.h2d_latency_ms = -1.0;
    expect_throws<CpuTransferModelError>(
        [&invalid] {
            static_cast<void>(AnalyticalCpuKVCacheTransferEngine{invalid});
        },
        "negative CPU latency must fail fast");
}

void test_typed_events_and_names() {
    EventQueue queue;
    CpuKVCacheOffloadStartPayload start{};
    start.transfer_id = CpuKvTransferId{2};
    const auto first = queue.push(SimTime::from_seconds(1.0), start);
    CpuKVCacheOffloadEndPayload end{};
    end.transfer_id = CpuKvTransferId{2};
    const auto second = queue.push(SimTime::from_seconds(1.0), end);
    expect(queue.pop().sequence == first && queue.pop().sequence == second,
           "equal-time CPU events must preserve creation sequence");
    expect(frontier::metrics::to_string(
               EventType::kCpuKvCacheOffloadStart) ==
               "cpu_kv_cache_offload_start" &&
               frontier::metrics::to_string(
                   EventType::kCpuKvCacheOffloadEnd) ==
                   "cpu_kv_cache_offload_end" &&
               frontier::metrics::to_string(
                   EventType::kCpuKvCacheRestoreStart) ==
                   "cpu_kv_cache_restore_start" &&
               frontier::metrics::to_string(
                   EventType::kCpuKvCacheRestoreEnd) ==
                   "cpu_kv_cache_restore_end",
           "all four CPU event types must have stable output names");
}

} // namespace

int main() {
    int failures = 0;
    failures += frontier::test::run(
        "CPU transfer directional serialization and full duplex",
        test_directional_serialization_and_full_duplex);
    failures += frontier::test::run(
        "CPU transfer zero bytes, validation, and large values",
        test_zero_bytes_validation_and_large_values);
    failures += frontier::test::run("CPU typed events and names",
                                    test_typed_events_and_names);
    return failures == 0 ? 0 : 1;
}
