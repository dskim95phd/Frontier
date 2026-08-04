#include "frontier/cpu_kv_cache_transfer/analytical_transfer.h"

#include <algorithm>
#include <cmath>

namespace frontier::cpu_kv_cache_transfer {

AnalyticalCpuKVCacheTransferEngine::AnalyticalCpuKVCacheTransferEngine(
    CpuTransferEngineConfig config)
    : config_(config) {
    if (!std::isfinite(config_.d2h_bandwidth_gbps) ||
        config_.d2h_bandwidth_gbps <= 0.0 ||
        !std::isfinite(config_.h2d_bandwidth_gbps) ||
        config_.h2d_bandwidth_gbps <= 0.0 ||
        !std::isfinite(config_.d2h_latency_ms) ||
        config_.d2h_latency_ms < 0.0 ||
        !std::isfinite(config_.h2d_latency_ms) ||
        config_.h2d_latency_ms < 0.0) {
        throw CpuTransferModelError(
            "CPU transfer bandwidths must be positive and latencies "
            "nonnegative");
    }
}

CpuTransferTiming AnalyticalCpuKVCacheTransferEngine::schedule(
    CpuTransferDirection direction, std::uint64_t size_bytes,
    SimTime submitted_at) {
    if (!submitted_at.valid()) {
        throw CpuTransferModelError("CPU transfer submission time is invalid");
    }
    SimTime &available = direction == CpuTransferDirection::kD2H
                             ? next_d2h_available_at_
                             : next_h2d_available_at_;
    const double bandwidth = direction == CpuTransferDirection::kD2H
                                 ? config_.d2h_bandwidth_gbps
                                 : config_.h2d_bandwidth_gbps;
    const double latency = direction == CpuTransferDirection::kD2H
                               ? config_.d2h_latency_ms
                               : config_.h2d_latency_ms;
    const double start_seconds =
        std::max(submitted_at.seconds(), available.seconds());
    const double bytes_per_ms = bandwidth * 1e9 / 8.0 / 1e3;
    const double service_ms =
        latency + static_cast<double>(size_bytes) / bytes_per_ms;
    const double end_seconds = start_seconds + service_ms * 1e-3;
    const double queue_ms =
        (start_seconds - submitted_at.seconds()) * 1e3;
    if (!std::isfinite(start_seconds) || !std::isfinite(service_ms) ||
        !std::isfinite(end_seconds) || !std::isfinite(queue_ms) ||
        service_ms < 0.0 || queue_ms < 0.0) {
        throw CpuTransferModelError("CPU transfer timing overflowed");
    }
    available = SimTime::from_seconds(end_seconds);
    CpuTransferTiming result{};
    result.direction = direction;
    result.size_bytes = size_bytes;
    result.submitted_at = submitted_at;
    result.started_at = SimTime::from_seconds(start_seconds);
    result.completed_at = available;
    result.queue_time_ms = queue_ms;
    result.service_time_ms = service_ms;
    return result;
}

} // namespace frontier::cpu_kv_cache_transfer
