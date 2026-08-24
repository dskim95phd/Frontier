#include "frontier/execution_time_predictor/analytical_roofline_execution_time_predictor.h"

#include <optional>
#include <utility>

#include "frontier/cc_backend/analytical_model.h"
#include "frontier/execution_time_predictor/analytical_roofline_internal.h"

namespace frontier::execution_time_predictor {
using roofline_detail::validate_predictor_configuration;

AnalyticalRooflineExecutionTimePredictor::
    AnalyticalRooflineExecutionTimePredictor(
        config::AnalyticalExecutionModelConfig config)
    : AnalyticalRooflineExecutionTimePredictor(
          config,
          [&]() {
              config::ParallelismConfig value{};
              value.tensor_parallel_size = config.tensor_parallel_size;
              return value;
          }(),
          config::ModelConfig{}, config::MoeRoutingConfig{}) {}

AnalyticalRooflineExecutionTimePredictor::
    AnalyticalRooflineExecutionTimePredictor(
        config::AnalyticalExecutionModelConfig config,
        config::ParallelismConfig parallelism)
    : AnalyticalRooflineExecutionTimePredictor(std::move(config), parallelism,
                                               config::ModelConfig{},
                                               config::MoeRoutingConfig{}) {}

AnalyticalRooflineExecutionTimePredictor::
    AnalyticalRooflineExecutionTimePredictor(
        config::AnalyticalExecutionModelConfig config,
        config::ParallelismConfig parallelism, config::ModelConfig model,
        config::MoeRoutingConfig routing)
    : AnalyticalRooflineExecutionTimePredictor(
          config, std::move(parallelism), std::move(model), std::move(routing),
          cc_backend::make_analytical_cc_backend([&]() {
              cc_backend::AnalyticalCommunicationConfig value{};
              value.network_bandwidth_gbps = config.network_bandwidth_gbps;
              value.latency_us = config.network_latency_us;
              value.intra_node_bandwidth_gbps =
                  config.intra_node_bandwidth_gbps;
              return value;
          }())) {}

AnalyticalRooflineExecutionTimePredictor::
    AnalyticalRooflineExecutionTimePredictor(
        config::AnalyticalExecutionModelConfig config,
        config::ParallelismConfig parallelism, config::ModelConfig model,
        config::MoeRoutingConfig routing,
        std::shared_ptr<const cc_backend::BaseCCBackend> communication_backend)
    : config_(std::move(config)),
      device_(detail::DeviceCeilings::from_config(config_)),
      analytical_(detail::analytical_config_from_profile(config_.kernel_profile,
                                                         config_.device)),
      parallelism_(parallelism), model_(std::move(model)), routing_(routing),
      communication_backend_(std::move(communication_backend)) {
    if (config_.mega_moe_tail_io_fraction.has_value()) {
        analytical_.mega_moe_tail_io_fraction =
            *config_.mega_moe_tail_io_fraction;
    }
    if (config_.mega_moe_wave_exposure.has_value()) {
        analytical_.mega_moe_wave_exposure = *config_.mega_moe_wave_exposure;
    }
    config::apply_model_native_precision_defaults(config_, model_);
    if (parallelism_.tensor_parallel_size == 0) {
        parallelism_.tensor_parallel_size = config_.tensor_parallel_size;
    }
    validate_predictor_configuration(config_, parallelism_, model_,
                                     communication_backend_);
    build_stage_timing_groups();
}

ExecutionTimePrediction
AnalyticalRooflineExecutionTimePredictor::predict_stage_execution_time(
    const entities::Batch &batch, const entities::RequestCollection &requests,
    StageId stage_id) const {
    return predict_execution(batch, requests, stage_id, std::nullopt);
}

} // namespace frontier::execution_time_predictor
