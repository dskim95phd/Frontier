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
    // The public SM100 MegaMoE profile converts the destination lane's exact
    // per-expert token histogram into padded M blocks and two-CTA cluster
    // waves. Portable profiles leave this disabled and retain the historical
    // aggregate roofline contract.
    bool mega_moe_geometry_enabled = false;
    std::uint64_t mega_moe_sm_count = 160;
    std::uint64_t mega_moe_block_n = 128;
    std::uint64_t mega_moe_cluster_size = 2;
    // Fraction of tail-row activation traffic that reaches HBM. The public
    // dynamic-tail kernel masks residual rows, while tensor-core work still
    // executes at the full M-block shape.
    double mega_moe_tail_io_fraction = 0.0;
    // Exposure of a partially occupied final cluster wave. One is the
    // geometry-derived prior; zero disables the wave correction.
    double mega_moe_wave_exposure = 1.0;
    // Residual scheduler/pipeline cost per two-CTA cluster task. The
    // roofline already charges arithmetic and bulk HBM traffic; this is the
    // grid-size-dependent remainder of a RaMP-style wave model.
    double mega_moe_cluster_task_latency_us = 0.0;
    // Device-generation scaling for the public one-sided A2A prior. Payload
    // bandwidth follows per-GPU NVLink peak; fixed startup remains separate
    // because link bandwidth alone does not establish a latency improvement.
    double mega_moe_a2a_bandwidth_scale = 1.0;
    double mega_moe_a2a_startup_scale = 1.0;
    // Fraction of the shorter A2A-vs-expert path left exposed.  The generic
    // public prior retains 0.35; the synchronized DeepGEMM profile uses 1.0.
    double moe_a2a_overlap_residual = 0.35;
};

[[nodiscard]] AnalyticalConfig
analytical_config_from_profile(std::string_view profile,
                               std::string_view device = "gb300");

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
