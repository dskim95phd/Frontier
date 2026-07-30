#pragma once

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "frontier/cc_backend/base_cc_backend.h"
#include "frontier/config/config.h"
#include "frontier/entities/batch.h"
#include "frontier/entities/execution_time.h"
#include "frontier/entities/request.h"

namespace frontier::execution_time_predictor {

struct MoERoutingDiagnostic {
  LayerId layer_id;
  std::uint64_t input_tokens = 0;
  std::uint64_t routed_tokens = 0;
  std::vector<std::uint64_t> global_expert_tokens;
  std::vector<std::vector<std::uint64_t>> lane_expert_tokens;
  std::vector<double> lane_times_ms;
  std::uint64_t critical_lane = 0;
  double critical_lane_time_ms = 0.0;
};

struct BatchExecutionPrediction {
  double duration_ms;
  entities::ExecutionTime execution_time;
  std::vector<std::pair<std::string, double>> diagnostics;
  std::vector<MoERoutingDiagnostic> moe_routing;
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
};

class FixedBatchExecutionModel final : public BatchExecutionModel {
 public:
  explicit FixedBatchExecutionModel(
      config::FixedExecutionModelConfig config);
  FixedBatchExecutionModel(
      config::FixedExecutionModelConfig config,
      config::ParallelismConfig parallelism,
      config::ModelConfig model,
      config::MoeRoutingConfig routing = {});

  [[nodiscard]] BatchExecutionPrediction predict(
      const entities::Batch& batch,
      const std::vector<entities::Request>& requests,
      StageId stage_id) const override;

 private:
  config::FixedExecutionModelConfig config_;
  config::ParallelismConfig parallelism_;
  config::ModelConfig model_;
  config::MoeRoutingConfig routing_;
};

class AnalyticalBatchExecutionModel final : public BatchExecutionModel {
 public:
  explicit AnalyticalBatchExecutionModel(
      config::AnalyticalExecutionModelConfig config);
  AnalyticalBatchExecutionModel(
      config::AnalyticalExecutionModelConfig config,
      config::ParallelismConfig parallelism);
  AnalyticalBatchExecutionModel(
      config::AnalyticalExecutionModelConfig config,
      config::ParallelismConfig parallelism,
      config::ModelConfig model,
      config::MoeRoutingConfig routing);
  AnalyticalBatchExecutionModel(
      config::AnalyticalExecutionModelConfig config,
      config::ParallelismConfig parallelism,
      config::ModelConfig model,
      config::MoeRoutingConfig routing,
      std::shared_ptr<const cc_backend::BaseCCBackend>
          communication_backend);

  [[nodiscard]] BatchExecutionPrediction predict(
      const entities::Batch& batch,
      const std::vector<entities::Request>& requests,
      StageId stage_id) const override;

 private:
  config::AnalyticalExecutionModelConfig config_;
  config::ParallelismConfig parallelism_;
  config::ModelConfig model_;
  config::MoeRoutingConfig routing_;
  std::shared_ptr<const cc_backend::BaseCCBackend>
      communication_backend_;
};

[[nodiscard]] std::shared_ptr<const BatchExecutionModel>
make_batch_execution_model(
    const config::ExecutionModelConfig& config,
    const std::optional<config::ParallelismConfig>& parallelism =
        std::nullopt,
    const std::optional<config::ModelConfig>& model = std::nullopt,
    const std::optional<config::MoeRoutingConfig>& routing = std::nullopt,
    std::shared_ptr<const cc_backend::BaseCCBackend>
        communication_backend = nullptr);

}  // namespace frontier::execution_time_predictor
