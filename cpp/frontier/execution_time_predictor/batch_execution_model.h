#pragma once

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "frontier/config/config.h"
#include "frontier/entities/batch.h"
#include "frontier/entities/execution_time.h"
#include "frontier/entities/request.h"

namespace frontier::execution_time_predictor {

struct BatchExecutionPrediction {
  double duration_ms;
  entities::ExecutionTime execution_time;
  std::vector<std::pair<std::string, double>> diagnostics;
};

class BatchExecutionModelError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class BatchExecutionModel {
 public:
  virtual ~BatchExecutionModel() = default;

  [[nodiscard]] virtual BatchExecutionPrediction predict(
      const entities::Batch& batch,
      const std::vector<entities::Request>& requests,
      StageId stage_id) const = 0;
  [[nodiscard]] virtual std::unique_ptr<BatchExecutionModel>
  clone() const = 0;
};

class FixedBatchExecutionModel final : public BatchExecutionModel {
 public:
  explicit FixedBatchExecutionModel(
      config::FixedExecutionModelConfig config);

  [[nodiscard]] BatchExecutionPrediction predict(
      const entities::Batch& batch,
      const std::vector<entities::Request>& requests,
      StageId stage_id) const override;
  [[nodiscard]] std::unique_ptr<BatchExecutionModel>
  clone() const override;

 private:
  config::FixedExecutionModelConfig config_;
};

class AnalyticalBatchExecutionModel final : public BatchExecutionModel {
 public:
  explicit AnalyticalBatchExecutionModel(
      config::AnalyticalExecutionModelConfig config);
  AnalyticalBatchExecutionModel(
      config::AnalyticalExecutionModelConfig config,
      config::ParallelismConfig parallelism);

  [[nodiscard]] BatchExecutionPrediction predict(
      const entities::Batch& batch,
      const std::vector<entities::Request>& requests,
      StageId stage_id) const override;
  [[nodiscard]] std::unique_ptr<BatchExecutionModel>
  clone() const override;

 private:
  config::AnalyticalExecutionModelConfig config_;
  config::ParallelismConfig parallelism_;
};

[[nodiscard]] std::unique_ptr<BatchExecutionModel>
make_batch_execution_model(
    const config::ExecutionModelConfig& config,
    const std::optional<config::ParallelismConfig>& parallelism =
        std::nullopt);

}  // namespace frontier::execution_time_predictor
