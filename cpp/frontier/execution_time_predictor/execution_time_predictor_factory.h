#pragma once

#include <memory>
#include <optional>

#include "frontier/config/config.h"
#include "frontier/execution_time_predictor/base_execution_time_predictor.h"

namespace frontier::cc_backend {
class BaseCCBackend;
}

namespace frontier::execution_time_predictor {

[[nodiscard]] ExecutionTimePredictorPtr make_execution_time_predictor(
    const config::ExecutionModelConfig &config,
    const std::optional<config::ParallelismConfig> &parallelism = std::nullopt,
    const std::optional<config::ModelConfig> &model = std::nullopt,
    const std::optional<config::MoeRoutingConfig> &routing = std::nullopt,
    std::shared_ptr<const cc_backend::BaseCCBackend> communication_backend =
        nullptr);

} // namespace frontier::execution_time_predictor
