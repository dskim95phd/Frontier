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

// Smooth achieved-efficiency envelope for Tensor Core GEMMs.  SM100 kernels
// choose among several MMA/block shapes, so there is no architecture-level
// token count at which every GEMM changes regime.  The curve instead ramps
// with M while applying smaller N-grid and K-pipeline saturation factors.
struct GemmEfficiencyCurve {
    Efficiency floor{0.25, 0.60, 0.50};
    Efficiency ceiling{0.65, 0.75, 0.10};
    double m_saturation_rows = 64.0;
    std::uint64_t tile_m = 64;
    std::uint64_t tile_n = 128;
    std::uint64_t tile_k = 128;
    std::uint64_t sm_count = 160;
    double grid_saturation_weight = 0.15;
    double k_saturation_weight = 0.10;
};

struct RooflineResult {
    double compute_time_ms;
    double memory_time_ms;
    double launch_time_ms;
    double predicted_time_ms;
    Bottleneck bottleneck;
};

struct AnalyticalConfig {
    GemmEfficiencyCurve gemm{};
    Efficiency latent_moe_front_floor{0.25, 0.60, 0.50};
    Efficiency prefill_attention{0.55, 0.65, 0.20};
    Efficiency decode_attention{0.20, 0.60, 0.50};
    Efficiency streaming{0.20, 0.75, 0.0};
    Efficiency moe{0.45, 0.65, 0.30};
    Efficiency routing{0.15, 0.55, 0.75};
    double kernel_launch_latency_us = 5.0;
    // K3's Day-0 path concatenates the BF16 router gate and LatentMoE down
    // projection weights. They share one input read and one GEMM launch.
    // Portable profiles retain the historical separate-kernel contract.
    bool fuse_latent_moe_front = false;
    // K3 decode-only kernel graph switches. These are kept separate from the
    // hardware ceilings and from numerical precision: an MXFP4 checkpoint may
    // still be executed by a portable fallback kernel, while the SGLang Day-0
    // path uses these fused schedules on SM100.
    bool fuse_shared_expert_moe_front = false;
    bool fuse_route_quant = false;
    bool overlap_shared_routed_moe = false;
    // EP side streams contend for HBM and achieve only partial overlap. The
    // current SGLang GB300 measurement reports a 4-5% end-to-end gain rather
    // than the full max(shared, routed) ideal.
    double ep_shared_routed_overlap_fraction = 0.0;
    bool use_tp8_latent_up_gemm_allgather = false;
    bool fuse_kda_decode_chain = false;
    bool overlap_kda_aux_projections = false;
    // FlashKDA prefill is a two-kernel pipeline. K1 is parallel over
    // sequence chunks and materializes a compact per-chunk workspace; K2 is
    // parallel over sequence/head and carries the recurrent state through the
    // chunk scan. The portable profile retains the conservative per-token
    // state-traffic roofline.
    bool use_flashkda_prefill_two_stage = false;
    std::uint64_t flashkda_chunk_tokens = 16;
    std::uint64_t flashkda_sm_count = 160;
    std::uint64_t flashkda_workspace_bytes_per_chunk_head = 13'824;
    // Fitted only to the four public GB200 variable-length/batched scheduling
    // points after fixing the single-sequence H64/H96 anchors. It represents
    // the exposed longest-sequence tail after the head/chunk waves.
    double flashkda_serial_tail_exposure = 0.823;
    bool overlap_mla_output_gate = false;
    std::uint64_t tp8_latent_up_gemm_allgather_max_tokens = 12;
    double tp8_latent_up_gemm_allgather_bandwidth_tbps = 1.8;
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
[[nodiscard]] Efficiency gemm_efficiency_for_shape(
    const GemmEfficiencyCurve &curve, std::uint64_t m, std::uint64_t k,
    std::uint64_t n, std::uint64_t independent_matrices = 1,
    const Efficiency *floor_override = nullptr);
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
