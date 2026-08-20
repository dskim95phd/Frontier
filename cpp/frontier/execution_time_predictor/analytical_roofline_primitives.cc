#include "frontier/execution_time_predictor/analytical_roofline_primitives.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <utility>

namespace frontier::execution_time_predictor::detail {

AnalyticalConfig analytical_config_from_profile(std::string_view profile,
                                                std::string_view device) {
    AnalyticalConfig result{};
    if (device != "rubin" && device != "gb300" && device != "custom") {
        throw AnalyticalModelError("unknown analytical profile device: " +
                                   std::string{device});
    }
    if (profile == "generic") {
        // Keep portable kernel efficiencies. Device resource scaling below
        // remains independent because the separately selected MegaMoE
        // communication backend may be used with this kernel profile.
    } else if (profile == "k3_sglang_mxfp4") {
        // Blackwell K3 local-MoE profile: official W4A8 SiTU cubins, fused
        // routing/finalize, small-M GEMMs, and lower launch count.  Keep the
        // ceilings physical; only achieved efficiencies and launch overhead
        // differ from the portable generic path.
        result.large_gemm = Efficiency{0.70, 0.80, 0.075};
        result.small_gemm = Efficiency{0.315, 0.69, 0.375};
        result.streaming = Efficiency{0.26, 0.825, 0.0};
        result.moe = Efficiency{0.525, 0.75, 0.225};
        result.routing = Efficiency{0.195, 0.625, 0.625};
        result.kernel_launch_latency_us = 3.75;
    } else if (profile == "k3_deepgemm_megamoe") {
        // Large DP-attention + EP K3 deployments use a synchronized
        // MegaMoE/DeepGEMM critical path.  Public communication measurements
        // include mandatory dispatch-tail and combine-head barriers, so do
        // not hide the shorter A2A path.  Expert execution uses the public
        // SM100 block-M policy and exact destination-lane histogram instead
        // of a workload-fitted constant slowdown.
        result.mega_moe_geometry_enabled = true;
        // RaMP's public H200 wave staircase implies about 0.1--0.2 us per
        // CTA over this grid range. MegaMoE uses two-CTA clusters and GB300
        // has higher compute/HBM ceilings, so retain only a conservative
        // residual after the ordinary roofline contribution. The coefficient
        // is calibrated on the three LMSYS DP4/EP32 points; DP2/EP16 is kept
        // as a topology holdout rather than participating in the fit.
        // A 2x2 factorial refit of tail IO and wave exposure found that the
        // former wave term was degenerate with this coefficient. Full padded
        // activation IO plus no separate wave multiplier preserves the DP4
        // calibration while reducing the DP2 topology holdout error.
        result.mega_moe_tail_io_fraction = 1.0;
        result.mega_moe_wave_exposure = 0.0;
        result.mega_moe_cluster_task_latency_us = 0.06325;
        // The measured wide-EP path includes a synchronized dispatch-tail /
        // combine-head barrier. Rubin exposes mechanisms that may improve
        // this dependency, but no K3 MegaMoE measurement quantifies the
        // overlap. Keep the measured residual instead of folding an
        // optimistic software assumption into hardware resource scaling.
        result.moe_a2a_overlap_residual = 1.0;
    } else {
        throw AnalyticalModelError("unknown analytical kernel profile: " +
                                   std::string{profile});
    }

    if (device == "rubin") {
        // Device resources are intentionally independent of kernel_profile.
        // The 2x payload scale is consumed only by the explicitly selected
        // sm100_megamoe_public communication backend; generic collectives
        // ignore it.
        result.mega_moe_sm_count = 224;
        result.mega_moe_a2a_bandwidth_scale = 2.0;
        // Keep the GB300-calibrated per-cluster residual unchanged. It
        // represents small-grid scheduling/pipeline latency, not bulk
        // arithmetic throughput, and no public Rubin measurement provides a
        // generational latency ratio. The former 5/3 throughput-per-SM scale
        // remains available as an explicit optimistic config sensitivity.
    }
    return result;
}

DeviceCeilings DeviceCeilings::from_config(
    const config::AnalyticalExecutionModelConfig &config) {
    DeviceCeilings result{};
    if (config.device == "rubin") {
        result = rubin();
    } else if (config.device == "gb300") {
        result = gb300();
    } else if (config.device != "custom") {
        throw AnalyticalModelError("unknown analytical device preset: " +
                                   config.device);
    }

    const config::AnalyticalDeviceOverrides &overrides =
        config.device_overrides;
    const auto apply = [](const std::optional<double> &override_value,
                          double &destination) {
        if (override_value.has_value()) {
            destination = override_value.value();
        }
    };
    apply(overrides.hbm_bandwidth_tbps, result.hbm_bandwidth_tbps);
    apply(overrides.fp32_tflops, result.fp32_tflops);
    apply(overrides.fp16_tflops, result.fp16_tflops);
    apply(overrides.fp8_tflops, result.fp8_tflops);
    apply(overrides.fp4_tflops, result.fp4_tflops);

    for (const auto &[value, name] : {
             std::pair{result.hbm_bandwidth_tbps, "HBM bandwidth"},
             std::pair{result.fp32_tflops, "FP32 ceiling"},
             std::pair{result.fp16_tflops, "FP16 ceiling"},
             std::pair{result.fp8_tflops, "FP8 ceiling"},
             std::pair{result.fp4_tflops, "FP4 ceiling"},
         }) {
        if (!std::isfinite(value) || value <= 0.0) {
            throw AnalyticalModelError(std::string{name} +
                                       " must be finite and positive");
        }
    }
    return result;
}

namespace {

void require_finite_nonnegative(double value, const char *field) {
    if (!std::isfinite(value) || value < 0.0) {
        throw AnalyticalModelError(std::string{field} +
                                   " must be finite and nonnegative");
    }
}

void validate_efficiency(const Efficiency &efficiency) {
    for (const auto &[value, name] : {
             std::pair{efficiency.compute, "compute efficiency"},
             std::pair{efficiency.memory, "memory efficiency"},
         }) {
        if (!std::isfinite(value) || value <= 0.0 || value > 1.0) {
            throw AnalyticalModelError(std::string{name} +
                                       " must satisfy 0 < value <= 1");
        }
    }
    if (!std::isfinite(efficiency.overlap_penalty) ||
        efficiency.overlap_penalty < 0.0 || efficiency.overlap_penalty > 1.0) {
        throw AnalyticalModelError(
            "overlap penalty must satisfy 0 <= value <= 1");
    }
}

} // namespace

Precision precision_from_string(std::string_view precision) {
    const std::optional<Precision> parsed = parse_precision(precision);
    if (parsed.has_value()) {
        return *parsed;
    }
    throw AnalyticalModelError("unsupported analytical precision: " +
                               std::string{precision});
}

double bytes_per_element(Precision precision) noexcept {
    return storage_bytes_per_element(precision);
}

double peak_tflops(const DeviceCeilings &device, Precision precision) {
    double peak = 0.0;
    switch (precision) {
    case Precision::kFp32:
        peak = device.fp32_tflops;
        break;
    case Precision::kFp16:
    case Precision::kBf16:
        peak = device.fp16_tflops;
        break;
    case Precision::kFp8:
    case Precision::kMxFp8:
    case Precision::kInt8:
        peak = device.fp8_tflops;
        break;
    case Precision::kFp4:
    case Precision::kMxFp4:
    case Precision::kInt4:
        peak = device.fp4_tflops;
        break;
    }
    if (!std::isfinite(peak) || peak <= 0.0) {
        throw AnalyticalModelError(
            "device compute ceiling must be finite and positive");
    }
    return peak;
}

RooflineResult predict_roofline(const DeviceCeilings &device,
                                Precision precision, const KernelWork &work,
                                const Efficiency &efficiency,
                                double kernel_launch_latency_us) {
    require_finite_nonnegative(work.flops, "FLOPs");
    require_finite_nonnegative(work.hbm_bytes, "HBM bytes");
    validate_efficiency(efficiency);
    require_finite_nonnegative(kernel_launch_latency_us,
                               "kernel launch latency");
    if (!std::isfinite(device.hbm_bandwidth_tbps) ||
        device.hbm_bandwidth_tbps <= 0.0) {
        throw AnalyticalModelError("HBM bandwidth must be finite and positive");
    }

    if (work.flops == 0.0 && work.hbm_bytes == 0.0) {
        return [&]() {
            RooflineResult value{};
            value.compute_time_ms = 0.0;
            value.memory_time_ms = 0.0;
            value.launch_time_ms = 0.0;
            value.predicted_time_ms = 0.0;
            value.bottleneck = Bottleneck::kNone;
            return value;
        }();
    }

    const double compute_time_ms =
        work.flops /
        (peak_tflops(device, precision) * 1e12 * efficiency.compute) * 1e3;
    const double memory_time_ms =
        work.hbm_bytes /
        (device.hbm_bandwidth_tbps * 1e12 * efficiency.memory) * 1e3;
    const double launch_time_ms = kernel_launch_latency_us / 1e3;
    const double maximum = std::max(compute_time_ms, memory_time_ms);
    const double minimum = std::min(compute_time_ms, memory_time_ms);
    const double predicted_time_ms =
        launch_time_ms + maximum + efficiency.overlap_penalty * minimum;

    Bottleneck bottleneck = Bottleneck::kHbm;
    if (launch_time_ms >= maximum) {
        bottleneck = Bottleneck::kLaunch;
    } else if (compute_time_ms >= memory_time_ms) {
        bottleneck = Bottleneck::kCompute;
    }

    return [&]() {
        RooflineResult value{};
        value.compute_time_ms = compute_time_ms;
        value.memory_time_ms = memory_time_ms;
        value.launch_time_ms = launch_time_ms;
        value.predicted_time_ms = predicted_time_ms;
        value.bottleneck = bottleneck;
        return value;
    }();
}

KernelWork gemm_work(std::uint64_t m, std::uint64_t k, std::uint64_t n,
                     double element_bytes, std::uint64_t weight_multiplier) {
    return gemm_work(m, k, n, element_bytes, element_bytes, weight_multiplier);
}

KernelWork gemm_work(std::uint64_t m, std::uint64_t k, std::uint64_t n,
                     double weight_element_bytes,
                     double activation_element_bytes,
                     std::uint64_t weight_multiplier) {
    return gemm_work(m, k, n, weight_element_bytes, activation_element_bytes,
                     activation_element_bytes, weight_multiplier);
}

KernelWork gemm_work(std::uint64_t m, std::uint64_t k, std::uint64_t n,
                     double weight_element_bytes,
                     double activation_element_bytes,
                     double output_element_bytes,
                     std::uint64_t weight_multiplier) {
    require_finite_nonnegative(weight_element_bytes, "weight element bytes");
    require_finite_nonnegative(activation_element_bytes,
                               "activation element bytes");
    require_finite_nonnegative(output_element_bytes, "output element bytes");
    if (weight_element_bytes == 0.0 || activation_element_bytes == 0.0 ||
        output_element_bytes == 0.0) {
        throw AnalyticalModelError("GEMM element bytes must be positive");
    }
    if (weight_multiplier == 0) {
        throw AnalyticalModelError("weight multiplier must be positive");
    }
    if (m == 0 || k == 0 || n == 0) {
        return [&]() {
            KernelWork value{};
            value.flops = 0.0;
            value.hbm_bytes = 0.0;
            return value;
        }();
    }

    const double resolved_m = static_cast<double>(m);
    const double resolved_k = static_cast<double>(k);
    const double resolved_n = static_cast<double>(n);
    const double multiplier = static_cast<double>(weight_multiplier);
    return [&]() {
        KernelWork value{};
        value.flops = 2.0 * resolved_m * resolved_k * resolved_n * multiplier;
        value.hbm_bytes =
            activation_element_bytes * resolved_m * resolved_k +
            weight_element_bytes * multiplier * resolved_k * resolved_n +
            output_element_bytes * multiplier * resolved_m * resolved_n;
        return value;
    }();
}

KernelWork streaming_work(double elements_read, double elements_written,
                          double flops, double element_bytes) {
    require_finite_nonnegative(elements_read, "elements read");
    require_finite_nonnegative(elements_written, "elements written");
    require_finite_nonnegative(flops, "streaming FLOPs");
    require_finite_nonnegative(element_bytes, "element bytes");
    if (element_bytes == 0.0) {
        throw AnalyticalModelError("element bytes must be positive");
    }
    return [&]() {
        KernelWork value{};
        value.flops = flops;
        value.hbm_bytes = (elements_read + elements_written) * element_bytes;
        return value;
    }();
}

} // namespace frontier::execution_time_predictor::detail
