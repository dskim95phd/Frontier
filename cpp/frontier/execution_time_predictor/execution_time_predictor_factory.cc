#include "frontier/execution_time_predictor/execution_time_predictor_factory.h"

#include <utility>

#include "frontier/cc_backend/analytical_model.h"
#include "frontier/execution_time_predictor/analytical_roofline_execution_time_predictor.h"
#include "frontier/execution_time_predictor/fixed_execution_time_predictor.h"

namespace frontier::execution_time_predictor {

ExecutionTimePredictorPtr make_execution_time_predictor(
    const config::ExecutionModelConfig &config,
    const std::optional<config::ParallelismConfig> &parallelism,
    const std::optional<config::ModelConfig> &model,
    const std::optional<config::MoeRoutingConfig> &routing,
    std::shared_ptr<const cc_backend::BaseCCBackend> communication_backend) {
    switch (config.type) {
    case config::ExecutionModelType::kFixed:
        return std::make_shared<FixedExecutionTimePredictor>(
            config.fixed, parallelism.value_or(config::ParallelismConfig{}),
            model.value_or(config::ModelConfig{}),
            routing.value_or(config::MoeRoutingConfig{}));
    case config::ExecutionModelType::kAnalytical:
        break;
    }

    config::ParallelismConfig resolved =
        parallelism.value_or(config::ParallelismConfig{});
    if (!parallelism.has_value()) {
        resolved.tensor_parallel_size = config.analytical.tensor_parallel_size;
    }
    return std::make_shared<AnalyticalRooflineExecutionTimePredictor>(
        config.analytical, resolved, model.value_or(config::ModelConfig{}),
        routing.value_or(config::MoeRoutingConfig{}),
        communication_backend != nullptr
            ? std::move(communication_backend)
            : cc_backend::make_analytical_cc_backend([&]() {
                  cc_backend::AnalyticalCommunicationConfig value{};
                  value.network_bandwidth_gbps =
                      config.analytical.network_bandwidth_gbps;
                  value.latency_us = config.analytical.network_latency_us;
                  value.intra_node_bandwidth_gbps =
                      config.analytical.intra_node_bandwidth_gbps;
                  return value;
              }()));
}

} // namespace frontier::execution_time_predictor
