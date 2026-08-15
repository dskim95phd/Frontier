#pragma once

#include "frontier/config/config.h"
#include "frontier/execution_time_predictor/base_execution_time_predictor.h"

namespace frontier::execution_time_predictor {

class FixedExecutionTimePredictor final : public BaseExecutionTimePredictor {
  public:
    explicit FixedExecutionTimePredictor(
        config::FixedExecutionModelConfig config);
    FixedExecutionTimePredictor(config::FixedExecutionModelConfig config,
                                config::ParallelismConfig parallelism,
                                config::ModelConfig model,
                                config::MoeRoutingConfig routing = {});

    [[nodiscard]] ExecutionTimePrediction
    predict_stage_execution_time(const entities::Batch &batch,
                                 const std::vector<entities::Request> &requests,
                                 StageId stage_id) const override;
    [[nodiscard]] MoEGroupLayerPrediction
    predict_moe_group_layer(const MoEGroupLayerInput &input) const override;
    void
    set_detailed_diagnostics_enabled(bool enabled) const noexcept override {
        detailed_diagnostics_enabled_ = enabled;
    }

  private:
    config::FixedExecutionModelConfig config_;
    config::ParallelismConfig parallelism_;
    config::ModelConfig model_;
    config::MoeRoutingConfig routing_;
    mutable bool detailed_diagnostics_enabled_ = true;
};

} // namespace frontier::execution_time_predictor
