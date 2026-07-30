#include "frontier/execution_time_predictor/fixed_execution_time_predictor.h"

#include <algorithm>
#include <cmath>
#include <utility>

#include "frontier/execution_time_predictor/analytical_roofline_execution_time_predictor.h"

namespace frontier::execution_time_predictor {

FixedExecutionTimePredictor::FixedExecutionTimePredictor(
    config::FixedExecutionModelConfig config)
    : FixedExecutionTimePredictor(std::move(config),
                                  config::ParallelismConfig{},
                                  config::ModelConfig{}) {}

FixedExecutionTimePredictor::FixedExecutionTimePredictor(
    config::FixedExecutionModelConfig config,
    config::ParallelismConfig parallelism, config::ModelConfig model,
    config::MoeRoutingConfig routing)
    : config_(std::move(config)), parallelism_(parallelism),
      model_(std::move(model)), routing_(routing) {
    if (!std::isfinite(config_.batch_latency_ms) ||
        config_.batch_latency_ms < 0.0) {
        throw ExecutionTimePredictorError(
            "fixed batch latency must be finite and nonnegative");
    }
    for (const double latency : config_.stage_latencies_ms) {
        if (!std::isfinite(latency) || latency < 0.0) {
            throw ExecutionTimePredictorError(
                "fixed stage latency must be finite and nonnegative");
        }
    }
}

ExecutionTimePrediction
FixedExecutionTimePredictor::predict_stage_execution_time(
    const entities::Batch &batch,
    const std::vector<entities::Request> &requests, StageId stage_id) const {
    static_cast<void>(requests);
    double latency = config_.batch_latency_ms;
    if (!config_.stage_latencies_ms.empty()) {
        if (!stage_id.valid() ||
            stage_id.index() >= config_.stage_latencies_ms.size()) {
            throw ExecutionTimePredictorError(
                "fixed stage latency does not cover stage ID");
        }
        latency = config_.stage_latencies_ms.at(stage_id.index());
    }
    entities::ExecutionTime execution_time = [&]() {
        entities::ExecutionTime value{};
        value.dense_compute_ms = latency;
        return value;
    }();
    if (model_.is_moe()) {
        const std::uint64_t layers_per_stage =
            model_.num_layers /
            std::max<std::uint64_t>(1, parallelism_.pipeline_parallel_size);
        const double component =
            latency * static_cast<double>(layers_per_stage);
        execution_time = [&]() {
            entities::ExecutionTime value{};
            value.dense_compute_ms = component * 7.0;
            value.tp_communication_ms =
                parallelism_.tensor_parallel_size > 1 ? component : 0.0;
            value.pp_communication_ms =
                stage_id.index() + 1 < parallelism_.pipeline_parallel_size
                    ? latency
                    : 0.0;
            value.moe_gating_linear_ms = component * 0.5;
            value.moe_gating_routing_topk_ms = component * 0.5;
            value.moe_grouped_gemm_ms = component;
            value.moe_shuffling_ms = component;
            value.moe_post_attention_norm_ms = component * 2.0;
            value.moe_tp_communication_ms =
                parallelism_.moe_tensor_parallel_size > 1 ? component : 0.0;
            value.ep_dispatch_ms =
                parallelism_.moe_expert_parallel_size > 1 ? component : 0.0;
            value.ep_combine_ms =
                parallelism_.moe_expert_parallel_size > 1 ? component : 0.0;
            value.dp_input_communication_ms =
                parallelism_.data_parallel_size > 1 ? component : 0.0;
            value.dp_output_communication_ms =
                parallelism_.data_parallel_size > 1 ? component : 0.0;
            return value;
        }();
    }

    std::vector<MoERoutingDiagnostic> routing_diagnostics;
    if (model_.is_moe()) {
        const std::uint64_t layers_per_stage =
            model_.num_layers / parallelism_.pipeline_parallel_size;
        routing_diagnostics.reserve(static_cast<std::size_t>(layers_per_stage));
        for (std::uint64_t local_layer = 0; local_layer < layers_per_stage;
             ++local_layer) {
            const detail::RoutingAllocation allocation = detail::route_tokens(
                batch.total_scheduled_tokens(), model_.router_topk,
                model_.runtime_total_experts,
                parallelism_.moe_expert_parallel_size, routing_, local_layer);
            routing_diagnostics.push_back([&]() {
                MoERoutingDiagnostic value{};
                value.layer_id = LayerId{local_layer};
                value.input_tokens = allocation.input_tokens;
                value.routed_tokens = allocation.routed_tokens;
                value.global_expert_tokens = allocation.global_expert_tokens;
                value.lane_expert_tokens = allocation.lane_expert_tokens;
                value.lane_times_ms = std::vector<double>(
                    static_cast<std::size_t>(
                        parallelism_.moe_expert_parallel_size),
                    latency);
                value.critical_lane = 0;
                value.critical_lane_time_ms = latency;
                return value;
            }());
        }
    }
    return [&]() {
        ExecutionTimePrediction value{};
        value.duration_ms = execution_time.total_ms();
        value.execution_time = execution_time;
        value.diagnostics = {
            {"fixed_stage_latency_ms", latency},
            {"stage_duration_ms", execution_time.total_ms()},
        };
        value.moe_routing = std::move(routing_diagnostics);
        return value;
    }();
}

} // namespace frontier::execution_time_predictor
