#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "frontier/cc_backend/base_cc_backend.h"
#include "frontier/config/config.h"
#include "frontier/entities/execution_time.h"
#include "frontier/execution_time_predictor/analytical_attention_model.h"
#include "frontier/execution_time_predictor/analytical_moe_model.h"
#include "frontier/execution_time_predictor/base_execution_time_predictor.h"

namespace frontier::execution_time_predictor::internal {

struct MoEStagePrediction {
    entities::ExecutionTime execution_time;
    std::vector<std::pair<std::string, double>> diagnostics;
    std::vector<MoERoutingDiagnostic> routing_diagnostics;
    std::uint64_t logical_moe_layer_count = 0;
    std::vector<ScaledMoEAttentionGroup> scaled_attention_groups;
    double repeated_moe_layer_pre_compute_ms = 0.0;
    double suffix_compute_ms = 0.0;
    double suffix_tp_communication_ms = 0.0;
};

struct MoEStageContext {
    ClusterType cluster_type;
    const detail::DenseBatch &dense_batch;
    const config::AnalyticalExecutionModelConfig &config;
    const detail::DeviceCeilings &device;
    const config::ParallelismConfig &parallelism;
    const config::ModelConfig &model;
    const config::MoeRoutingConfig &routing;
    const cc_backend::BaseCCBackend &communication_backend;
    config::PipelineStageLayerRange stage_layers;
    const std::vector<detail::DenseLayerTimes> &layer_times;
    double allreduce_ms;
    double dcp_attention_communication_ms;
    double communication_element_bytes;
    entities::ExecutionTime base_execution_time;
    const detail::MoELanePrediction *reusable_moe_lane_prediction = nullptr;
    const detail::MoECommunicationTime *reusable_moe_communication = nullptr;
    const detail::RoutingAllocation *reusable_routing_allocation = nullptr;
    bool detailed_diagnostics_enabled = true;

    [[nodiscard]] const detail::DenseLayerTimes &
    layer_time(std::uint64_t model_layer) const noexcept;
    [[nodiscard]] double
    dense_layer_compute_ms(std::uint64_t model_layer) const noexcept;
    [[nodiscard]] double
    attention_compute_ms(std::uint64_t model_layer) const noexcept;
    [[nodiscard]] bool is_mla_layer(std::uint64_t model_layer) const noexcept;
    [[nodiscard]] double tp_layer_ms(std::uint64_t model_layer) const noexcept;
    [[nodiscard]] double
    attention_communication_ms(std::uint64_t model_layer) const noexcept;
};

[[nodiscard]] detail::MoEModel
make_moe_model(const config::ModelConfig &model,
               const config::ParallelismConfig &parallelism);
[[nodiscard]] detail::MoEOperatorPrecisions make_moe_operator_precisions(
    const config::AnalyticalExecutionModelConfig &config);
[[nodiscard]] std::uint64_t activation_payload_bytes(std::uint64_t tokens,
                                                     std::uint64_t hidden_size,
                                                     double element_bytes);
[[nodiscard]] std::vector<double>
lane_times_ms(const detail::MoELanePrediction &prediction);
[[nodiscard]] bool
routing_is_layer_invariant(const config::MoeRoutingConfig &routing) noexcept;
[[nodiscard]] bool
routing_is_batch_shared(const config::AnalyticalExecutionModelConfig &config,
                        const config::MoeRoutingConfig &routing) noexcept;
[[nodiscard]] bool
has_contiguous_moe_suffix(const config::ModelConfig &model,
                          const config::PipelineStageLayerRange &layers);
[[nodiscard]] MoEStagePrediction
predict_selected_moe_layer_execution(const MoEStageContext &context,
                                     std::uint64_t selected_moe_layer);
[[nodiscard]] MoEStagePrediction
predict_moe_stage_execution(const MoEStageContext &context);

} // namespace frontier::execution_time_predictor::internal
