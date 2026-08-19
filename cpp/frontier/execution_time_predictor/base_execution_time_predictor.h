#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "frontier/core/ids.h"
#include "frontier/entities/batch.h"
#include "frontier/entities/execution_time.h"
#include "frontier/entities/request.h"

namespace frontier::execution_time_predictor {

struct MoEGroupedGemmGeometryDiagnostic {
    bool enabled = false;
    std::uint64_t block_m = 0;
    std::uint64_t routed_m_blocks = 0;
    std::uint64_t shared_m_blocks = 0;
    std::uint64_t routed_padded_tokens = 0;
    std::uint64_t shared_padded_tokens = 0;
    std::uint64_t up_cluster_tasks = 0;
    std::uint64_t down_cluster_tasks = 0;
    double up_wave_utilization = 1.0;
    double down_wave_utilization = 1.0;
    double up_cluster_task_overhead_ms = 0.0;
    double down_cluster_task_overhead_ms = 0.0;
};

struct MoERoutingDiagnostic {
    LayerId layer_id;
    std::uint64_t model_layer_id = 0;
    double pre_moe_compute_ms = 0.0;
    // Attention TP/DCP communication that must complete before this layer's
    // routing/expert phase can begin.  MoE decode scheduling is decomposed
    // into separate DES events, so keeping this distinct from compute avoids
    // dropping the communication from wall-clock time while preserving the
    // ExecutionTime component breakdown.
    double pre_moe_tp_communication_ms = 0.0;
    std::uint64_t input_tokens = 0;
    std::uint64_t routed_tokens = 0;
    std::vector<std::uint64_t> global_expert_tokens;
    std::vector<std::vector<std::uint64_t>> lane_expert_tokens;
    std::vector<std::uint64_t> lane_routed_tokens;
    std::vector<std::uint64_t> lane_active_experts;
    std::vector<std::uint64_t> lane_unique_tokens;
    std::vector<double> lane_times_ms;
    // Routed-expert/shuffle-only lane time. Source-local router, latent,
    // shared-expert, and finalize work remains in lane_times_ms.
    std::vector<double> routed_lane_times_ms;
    // Lane-aligned non-routed contribution emitted directly by the predictor.
    // Preserving lane identity avoids max(full)-max(routed) subtraction.
    std::vector<double> source_local_lane_times_ms;
    // Lane-invariant portion of source_local_lane_times_ms used by the
    // fused dispatch/expert/combine overlap model.
    double shared_expert_path_ms = 0.0;
    std::uint64_t critical_lane = 0;
    double critical_lane_time_ms = 0.0;
    double raw_ep_dispatch_ms = 0.0;
    double raw_ep_combine_ms = 0.0;
    double exposed_ep_dispatch_ms = 0.0;
    double exposed_ep_combine_ms = 0.0;
    // Destination-group critical routed lane. Disabled for local-only paths.
    MoEGroupedGemmGeometryDiagnostic grouped_gemm_geometry;
};

enum class ScaledMoEAttentionFamily {
    kStandard,
    kMla,
    kKda,
};

// A first-layer-scaled prediction retains one detailed MoE routing/expert
// event, but attention can have multiple implementations within the same
// pipeline stage (for example Kimi K3's KDA/MLA schedule).  Each entry
// represents the remaining logical layers of one attention family.
struct ScaledMoEAttentionGroup {
    ScaledMoEAttentionFamily family = ScaledMoEAttentionFamily::kStandard;
    std::uint64_t layer_count = 0;
    double pre_moe_compute_ms_per_layer = 0.0;
    double pre_moe_tp_communication_ms_per_layer = 0.0;
};

struct ExecutionTimePrediction {
    double duration_ms = 0.0;
    entities::ExecutionTime execution_time;
    std::vector<std::pair<std::string, double>> diagnostics;
    std::vector<MoERoutingDiagnostic> moe_routing;
    std::uint64_t logical_moe_layer_count = 0;
    std::vector<ScaledMoEAttentionGroup> scaled_moe_attention_groups;
    // Retained as the first detailed MoE layer's attention time for output
    // compatibility.  Schedulers use scaled_moe_attention_groups for the
    // remaining layers.
    double repeated_moe_layer_pre_compute_ms = 0.0;
    double moe_suffix_compute_ms = 0.0;
    // TP communication belonging to dense layers after the final MoE layer.
    // Decode's decomposed scheduler must wait for it at stage completion.
    double moe_suffix_tp_communication_ms = 0.0;
    bool lazy_moe_layer_prediction = false;
    bool scaled_moe_layer_prediction = false;
};

struct MoEGroupLayerInput {
    LayerId layer_id;
    std::uint64_t model_layer_id = 0;
    std::uint64_t input_tokens = 0;
    std::uint64_t routed_tokens = 0;
    std::vector<std::uint64_t> global_expert_tokens;
    // Exact local token count for every active attention-DP source. The sum is
    // input_tokens, but source-local operators must never consume that sum.
    std::vector<std::uint64_t> source_input_tokens;
    // One already-predicted shared-expert path per active DP source. This is
    // carried with source_input_tokens to avoid recomputing full MoE layers
    // at every destination-group barrier.
    std::vector<double> source_shared_expert_path_ms;
    // One row per active attention-DP source and one column per destination
    // EP lane. Column sums are the receiver-side loads that bound A2A time.
    std::vector<std::vector<std::uint64_t>> source_lane_routed_tokens;
    std::vector<std::vector<std::uint64_t>> source_lane_unique_tokens;
    // Predictors without a token-sensitive group model may retain their
    // existing behavior by returning this lane-wise sum.
    std::vector<double> fallback_lane_times_ms;
};

struct MoEGroupLayerPrediction {
    std::vector<double> lane_times_ms;
    // Analytical group predictions contain destination routed work only.
    // The scheduler composes this with the slowest source-local path. Fixed
    // predictors retain their historical full-lane fallback and leave false.
    bool lane_times_are_routed_only = false;
    std::uint64_t critical_lane = 0;
    double critical_lane_time_ms = 0.0;
    bool has_source_aware_ep_communication = false;
    std::vector<std::uint64_t> destination_lane_routed_tokens;
    std::vector<std::uint64_t> destination_lane_unique_tokens;
    double raw_ep_dispatch_ms = 0.0;
    double raw_ep_combine_ms = 0.0;
    double ep_dispatch_ms = 0.0;
    double ep_combine_ms = 0.0;
    MoEGroupedGemmGeometryDiagnostic grouped_gemm_geometry;
};

class ExecutionTimePredictorError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

// Stable scheduler-facing predictor contract. Concrete predictors own their
// modeling policy; schedulers depend only on this interface.
class BaseExecutionTimePredictor {
  public:
    virtual ~BaseExecutionTimePredictor() = default;

    [[nodiscard]] virtual ExecutionTimePrediction
    predict_stage_execution_time(const entities::Batch &batch,
                                 const entities::RequestCollection &requests,
                                 StageId stage_id) const = 0;

    [[nodiscard]] virtual MoEGroupLayerPrediction
    predict_moe_group_layer(const MoEGroupLayerInput &input) const = 0;

    // Predictors with batch-scoped memoization release it alongside the batch
    // entity. Predictors without such state intentionally keep this a no-op.
    virtual void release_batch_timing_cache(BatchId) const {}

    // Output modes that omit detailed traces can disable diagnostic payload
    // construction at the source. Runtime routing data remains unaffected.
    virtual void set_detailed_diagnostics_enabled(bool) const noexcept {}

    [[nodiscard]] virtual bool supports_lazy_moe_prediction() const noexcept {
        return false;
    }

    [[nodiscard]] virtual ExecutionTimePrediction
    prepare_moe_stage_execution(const entities::Batch &batch,
                                const entities::RequestCollection &requests,
                                StageId stage_id) const {
        return predict_stage_execution_time(batch, requests, stage_id);
    }

    [[nodiscard]] virtual ExecutionTimePrediction
    predict_moe_layer_execution(const entities::Batch &,
                                const entities::RequestCollection &, StageId,
                                std::uint64_t) const {
        throw ExecutionTimePredictorError(
            "execution predictor does not support lazy MoE layers");
    }
};

using ExecutionTimePredictorPtr =
    std::shared_ptr<const BaseExecutionTimePredictor>;

} // namespace frontier::execution_time_predictor
