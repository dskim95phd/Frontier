#pragma once

#include <cstdint>
#include <stdexcept>
#include <string_view>

#include "frontier/config/config.h"
#include "frontier/core/precision.h"

namespace frontier::execution_time_predictor::detail {

using Precision = frontier::Precision;

[[nodiscard]] Precision precision_from_string(std::string_view precision);

enum class Bottleneck {
    kNone,
    kLaunch,
    kCompute,
    kHbm,
};

struct DeviceCeilings {
    double hbm_bandwidth_tbps;
    double fp32_tflops;
    double fp16_tflops;
    double fp8_tflops;
    double fp4_tflops;

    [[nodiscard]] static constexpr DeviceCeilings rubin() noexcept {
        return DeviceCeilings{
            22.0, 130.0, 4'000.0, 17'500.0, 35'000.0,
        };
    }

    [[nodiscard]] static constexpr DeviceCeilings gb300() noexcept {
        return DeviceCeilings{
            8.0, 83.33333333333333, 2'500.0, 5'000.0, 15'000.0,
        };
    }

    [[nodiscard]] static DeviceCeilings
    from_config(const config::AnalyticalExecutionModelConfig &config);
};

struct KernelWork {
    double flops;
    double hbm_bytes;
};

struct Efficiency {
    double compute;
    double memory;
    double overlap_penalty;
};

struct RooflineResult {
    double compute_time_ms;
    double memory_time_ms;
    double launch_time_ms;
    double predicted_time_ms;
    Bottleneck bottleneck;
};

struct AnalyticalConfig {
    Efficiency large_gemm{0.65, 0.75, 0.10};
    Efficiency small_gemm{0.25, 0.60, 0.50};
    Efficiency prefill_attention{0.55, 0.65, 0.20};
    Efficiency decode_attention{0.20, 0.60, 0.50};
    Efficiency streaming{0.20, 0.75, 0.0};
    Efficiency moe{0.45, 0.65, 0.30};
    Efficiency routing{0.15, 0.55, 0.75};
    double kernel_launch_latency_us = 5.0;
    std::uint64_t small_gemm_token_threshold = 128;
};

class AnalyticalModelError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

[[nodiscard]] double bytes_per_element(Precision precision) noexcept;
[[nodiscard]] double peak_tflops(const DeviceCeilings &device,
                                 Precision precision);
[[nodiscard]] RooflineResult predict_roofline(const DeviceCeilings &device,
                                              Precision precision,
                                              const KernelWork &work,
                                              const Efficiency &efficiency,
                                              double kernel_launch_latency_us);
[[nodiscard]] KernelWork gemm_work(std::uint64_t m, std::uint64_t k,
                                   std::uint64_t n, double element_bytes,
                                   std::uint64_t weight_multiplier = 1);
[[nodiscard]] KernelWork gemm_work(std::uint64_t m, std::uint64_t k,
                                   std::uint64_t n, double weight_element_bytes,
                                   double activation_element_bytes,
                                   std::uint64_t weight_multiplier);
[[nodiscard]] KernelWork gemm_work(std::uint64_t m, std::uint64_t k,
                                   std::uint64_t n, double weight_element_bytes,
                                   double activation_element_bytes,
                                   double output_element_bytes,
                                   std::uint64_t weight_multiplier);
[[nodiscard]] KernelWork streaming_work(double elements_read,
                                        double elements_written, double flops,
                                        double element_bytes);

} // namespace frontier::execution_time_predictor::detail
