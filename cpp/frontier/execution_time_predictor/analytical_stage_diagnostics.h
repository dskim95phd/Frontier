#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "frontier/config/config.h"
#include "frontier/entities/execution_time.h"
#include "frontier/execution_time_predictor/analytical_attention_model.h"

namespace frontier::execution_time_predictor::detail {

struct StageTimingCacheDiagnostics {
    bool enabled = false;
    bool hit = false;
    std::uint64_t hits_total = 0;
    std::uint64_t misses_total = 0;
    std::uint64_t unique_templates_total = 0;
    std::uint64_t entries = 0;
};

// Read-only data needed to render the analytical stage diagnostics. Keeping
// this separate from prediction makes the hot path and the output contract
// independently reviewable.
struct StageDiagnosticsInput {
    const config::AnalyticalExecutionModelConfig &config;
    const config::ParallelismConfig &parallelism;
    const config::ModelConfig &model;
    const config::PipelineStageGroupCatalogue &timing_catalogue;
    const DenseBatch &batch;
    const DenseOperatorPrecisions &precisions;
    const std::vector<DenseLayerTimes> &layer_times;
    const config::PipelineStageLayerRange &stage_layers;
    const entities::ExecutionTime &execution_time;
    std::uint64_t stage = 0;
    std::uint32_t timing_group_id = 0;
    std::uint64_t timing_group_multiplicity = 0;
    StageTimingCacheDiagnostics timing_cache;
    bool stage_group_requested = false;
    bool stage_group_active = false;
    bool routing_is_layer_invariant = false;
    std::uint64_t lm_head_tokens = 0;
    double communication_element_bytes = 0.0;
    double allreduce_ms = 0.0;
    double dcp_attention_communication_ms = 0.0;
    double first_layer_compute_ms = 0.0;
    double first_tp_layer_ms = 0.0;
    double dense_compute_ms = 0.0;
    double tp_communication_ms = 0.0;
    double pp_communication_ms = 0.0;
    double duration_ms = 0.0;
};

[[nodiscard]] std::vector<std::pair<std::string, double>>
build_stage_diagnostics(const StageDiagnosticsInput &input);

} // namespace frontier::execution_time_predictor::detail
