#include "frontier/execution_time_predictor/analytical_roofline_primitives.h"

#include <algorithm>
#include <cmath>
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
    } else if (profile == "k3_flashkda_prefill") {
        // Isolated ablation: retain portable GEMM/MoE assumptions and change
        // only KDA prefill to the released two-kernel FlashKDA schedule.
        result.use_flashkda_prefill_two_stage = true;
    } else if (profile == "k3_sglang_mxfp4") {
        // Blackwell K3 local-MoE profile: official W4A8 SiTU cubins, fused
        // routing/finalize, small-M GEMMs, and lower launch count.  Keep the
        // ceilings physical; only achieved efficiencies and launch overhead
        // differ from the portable generic path.
        result.gemm.floor = Efficiency{0.315, 0.69, 0.375};
        result.gemm.ceiling = Efficiency{0.70, 0.80, 0.075};
        // Concatenating the 896-wide gate to the 3584-wide latent down
        // projection increases N by 4480/3584. Scale achieved throughput by
        // the same geometry ratio so the gate slice is nearly free, matching
        // the published kernel intent without fitting a latency constant.
        result.latent_moe_front_floor =
            Efficiency{0.39375, 0.8625, 0.375};
        result.streaming = Efficiency{0.26, 0.825, 0.0};
        result.moe = Efficiency{0.525, 0.75, 0.225};
        result.routing = Efficiency{0.195, 0.625, 0.625};
        result.kernel_launch_latency_us = 3.75;
        result.fuse_latent_moe_front = true;
        result.fuse_shared_expert_moe_front = true;
        result.fuse_route_quant = true;
        result.overlap_shared_routed_moe = true;
        result.use_tp8_latent_up_gemm_allgather = true;
        result.fuse_kda_decode_chain = true;
        result.overlap_kda_aux_projections = true;
        result.use_flashkda_prefill_two_stage = true;
        result.overlap_mla_output_gate = true;
    } else if (profile == "k3_deepgemm_megamoe") {
        // Large DP-attention + EP K3 deployments use a synchronized
        // MegaMoE/DeepGEMM critical path.  Public communication measurements
        // include mandatory dispatch-tail and combine-head barriers, so do
        // not hide the shorter A2A path.  Expert execution uses the public
        // SM100 block-M policy and exact destination-lane histogram instead
        // of a workload-fitted constant slowdown.
        result.mega_moe_geometry_enabled = true;
        result.fuse_latent_moe_front = true;
        result.fuse_route_quant = true;
        result.overlap_shared_routed_moe = true;
        result.ep_shared_routed_overlap_fraction = 0.5;
        result.fuse_kda_decode_chain = true;
        result.overlap_kda_aux_projections = true;
        result.use_flashkda_prefill_two_stage = true;
        result.overlap_mla_output_gate = true;
        result.latent_moe_front_floor = Efficiency{0.3125, 0.75, 0.50};
        // Full padded activation IO is retained, but no additional wave or
        // per-cluster residual is charged. The old 0.06325 us coefficient was
        // fitted while router FP32 logits incorrectly forced the router GEMM
        // onto the FP32 CUDA-core ceiling and while dispatch/combine shared
        // one dtype. A precision-correct refit selected the zero boundary on
        // the three LMSYS DP4/EP32 points; DP2/EP16 remained the holdout.
        result.mega_moe_tail_io_fraction = 1.0;
        result.mega_moe_wave_exposure = 0.0;
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
        result.gemm.sm_count = 224;
        result.mega_moe_a2a_bandwidth_scale = 2.0;
        // GEMM and MegaMoE wave exposure both use Rubin's larger SM count.
        // No fitted per-grid-task latency is carried across generations.
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

Efficiency gemm_efficiency_for_shape(const GemmEfficiencyCurve &curve,
                                     std::uint64_t m, std::uint64_t k,
                                     std::uint64_t n,
                                     std::uint64_t independent_matrices,
                                     const Efficiency *floor_override) {
    validate_efficiency(curve.floor);
    validate_efficiency(curve.ceiling);
    const Efficiency &floor =
        floor_override == nullptr ? curve.floor : *floor_override;
    validate_efficiency(floor);
    if (!std::isfinite(curve.m_saturation_rows) ||
        curve.m_saturation_rows <= 0.0 || curve.tile_m == 0 ||
        curve.tile_n == 0 || curve.tile_k == 0 || curve.sm_count == 0 ||
        independent_matrices == 0 ||
        !std::isfinite(curve.grid_saturation_weight) ||
        !std::isfinite(curve.k_saturation_weight) ||
        curve.grid_saturation_weight < 0.0 ||
        curve.k_saturation_weight < 0.0 ||
        curve.grid_saturation_weight + curve.k_saturation_weight > 1.0) {
        throw AnalyticalModelError("invalid GEMM efficiency curve");
    }
    if (m == 0 || k == 0 || n == 0) {
        return floor;
    }

    // DeepGEMM and CUTLASS select multiple block-M shapes around 32/64/128.
    // A shifted exponential represents the useful-row fill of those choices
    // without inventing a discontinuity at any one token count.  M=1 pins the
    // measured GEMV-like floor; roughly two 64-row SM100 MMA tiles reach 86%
    // of the M-axis envelope.
    const double m_progress =
        1.0 - std::exp(-(static_cast<double>(m) - 1.0) /
                       curve.m_saturation_rows);

    const auto ceil_div = [](std::uint64_t value, std::uint64_t divisor) {
        return value / divisor + static_cast<std::uint64_t>(value % divisor != 0);
    };
    const long double grid_tiles =
        static_cast<long double>(ceil_div(m, curve.tile_m)) *
        static_cast<long double>(ceil_div(n, curve.tile_n)) *
        static_cast<long double>(independent_matrices);
    const double grid_progress = std::min(
        1.0, static_cast<double>(grid_tiles /
                                 static_cast<long double>(curve.sm_count)));
    const double k_tiles = static_cast<double>(k) /
                           static_cast<double>(curve.tile_k);
    const double k_progress = 1.0 - std::exp(-k_tiles / 4.0);
    const double base_weight =
        1.0 - curve.grid_saturation_weight - curve.k_saturation_weight;
    const double shape_factor = base_weight +
                                curve.grid_saturation_weight * grid_progress +
                                curve.k_saturation_weight * k_progress;
    const double progress = std::clamp(m_progress * shape_factor, 0.0, 1.0);

    const auto interpolate = [progress](double low, double high) {
        return low + (high - low) * progress;
    };
    return Efficiency{
        interpolate(floor.compute, curve.ceiling.compute),
        interpolate(floor.memory, curve.ceiling.memory),
        interpolate(floor.overlap_penalty,
                    curve.ceiling.overlap_penalty),
    };
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
