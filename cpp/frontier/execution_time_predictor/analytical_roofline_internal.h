#pragma once

#include <memory>

#include "frontier/cc_backend/base_cc_backend.h"
#include "frontier/config/config.h"
#include "frontier/entities/batch.h"
#include "frontier/entities/request.h"
#include "frontier/execution_time_predictor/analytical_attention_model.h"
#include "frontier/execution_time_predictor/base_execution_time_predictor.h"

namespace frontier::execution_time_predictor::roofline_detail {

// Shared data conversion and validation used by the scheduler-facing
// predictor facade and its cache/prediction implementation modules.
struct StageBatchInfo {
    detail::DenseBatch dense_batch;
    std::uint64_t lm_head_tokens = 0;
};

[[nodiscard]] StageBatchInfo
build_stage_batch_info(const entities::Batch &batch,
                       const entities::RequestCollection &requests);
[[nodiscard]] detail::DenseModel
make_dense_model(const config::ModelConfig &model,
                 const config::ParallelismConfig &parallelism);
[[nodiscard]] detail::DenseOperatorPrecisions make_dense_operator_precisions(
    const config::AnalyticalExecutionModelConfig &config);
[[nodiscard]] ExecutionTimePrediction materialize_cached_prediction(
    const ExecutionTimePrediction &cached, const config::ModelConfig &model,
    const config::PipelineStageLayerRange &stage_layers);
void validate_predictor_configuration(
    const config::AnalyticalExecutionModelConfig &config,
    const config::ParallelismConfig &parallelism,
    const config::ModelConfig &model,
    const std::shared_ptr<const cc_backend::BaseCCBackend> &backend);

} // namespace frontier::execution_time_predictor::roofline_detail
