#pragma once

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

struct ExecutionTimePrediction {
    double duration_ms;
    entities::ExecutionTime execution_time;
    std::vector<std::pair<std::string, double>> diagnostics;
    std::vector<MoERoutingDiagnostic> moe_routing;
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
                                 const std::vector<entities::Request> &requests,
                                 StageId stage_id) const = 0;
};

using ExecutionTimePredictorPtr =
    std::shared_ptr<const BaseExecutionTimePredictor>;

} // namespace frontier::execution_time_predictor
