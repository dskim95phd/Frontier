#pragma once

#include <cstdint>
#include <stdexcept>

#include "frontier/core/event.h"

namespace frontier::cpu_kv_cache_transfer {

class CpuTransferModelError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

enum class CpuTransferDirection { kD2H, kH2D };

struct CpuTransferTiming {
    CpuTransferDirection direction = CpuTransferDirection::kD2H;
    std::uint64_t size_bytes = 0;
    SimTime submitted_at;
    SimTime started_at;
    SimTime completed_at;
    double queue_time_ms = 0.0;
    double service_time_ms = 0.0;
};

struct CpuTransferEngineConfig {
    double d2h_bandwidth_gbps = 0.0;
    double d2h_latency_ms = 0.0;
    double h2d_bandwidth_gbps = 0.0;
    double h2d_latency_ms = 0.0;
};

class AnalyticalCpuKVCacheTransferEngine {
  public:
    explicit AnalyticalCpuKVCacheTransferEngine(CpuTransferEngineConfig config);

    [[nodiscard]] CpuTransferTiming schedule(CpuTransferDirection direction,
                                             std::uint64_t size_bytes,
                                             SimTime submitted_at);
    [[nodiscard]] SimTime next_d2h_available_at() const noexcept {
        return next_d2h_available_at_;
    }
    [[nodiscard]] SimTime next_h2d_available_at() const noexcept {
        return next_h2d_available_at_;
    }

  private:
    CpuTransferEngineConfig config_;
    SimTime next_d2h_available_at_ = SimTime::from_seconds(0.0);
    SimTime next_h2d_available_at_ = SimTime::from_seconds(0.0);
};

} // namespace frontier::cpu_kv_cache_transfer
