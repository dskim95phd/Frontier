#include "frontier/execution_time_predictor/analytical_roofline_execution_time_predictor.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

#include "frontier/cc_backend/analytical_model.h"

namespace frontier::execution_time_predictor {
namespace {

const entities::Request &
get_request(const std::vector<entities::Request> &requests,
            RequestId request_id) {
    if (!request_id.valid() || request_id.index() >= requests.size()) {
        throw BatchExecutionModelError(
            "batch execution references an unknown request");
    }
    const entities::Request &request = requests.at(request_id.index());
    if (request.id() != request_id) {
        throw BatchExecutionModelError(
            "batch execution request arena invariant failed");
    }
    return request;
}

} // namespace

FixedBatchExecutionModel::FixedBatchExecutionModel(
    config::FixedExecutionModelConfig config)
    : FixedBatchExecutionModel(std::move(config), config::ParallelismConfig{},
                               config::ModelConfig{}) {}

FixedBatchExecutionModel::FixedBatchExecutionModel(
    config::FixedExecutionModelConfig config,
    config::ParallelismConfig parallelism, config::ModelConfig model,
    config::MoeRoutingConfig routing)
    : config_(std::move(config)), parallelism_(parallelism),
      model_(std::move(model)), routing_(routing) {
    if (!std::isfinite(config_.batch_latency_ms) ||
        config_.batch_latency_ms < 0.0) {
        throw BatchExecutionModelError(
            "fixed batch latency must be finite and nonnegative");
    }
    for (const double latency : config_.stage_latencies_ms) {
        if (!std::isfinite(latency) || latency < 0.0) {
            throw BatchExecutionModelError(
                "fixed stage latency must be finite and nonnegative");
        }
    }
}

BatchExecutionPrediction FixedBatchExecutionModel::predict(
    const entities::Batch &batch,
    const std::vector<entities::Request> &requests, StageId stage_id) const {
    static_cast<void>(requests);
    double latency = config_.batch_latency_ms;
    if (!config_.stage_latencies_ms.empty()) {
        if (!stage_id.valid() ||
            stage_id.index() >= config_.stage_latencies_ms.size()) {
            throw BatchExecutionModelError(
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
            // Phi dummy timing has one MoE norm and one FFN residual add per
            // layer. They share this normalized post-attention bucket.
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
        BatchExecutionPrediction value{};
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
    : config_(std::move(config)), parallelism_(parallelism),
      model_(std::move(model)), routing_(routing),
      communication_backend_(std::move(communication_backend)) {
    if (parallelism_.tensor_parallel_size == 0) {
        parallelism_.tensor_parallel_size = config_.tensor_parallel_size;
    }
    if (communication_backend_ == nullptr || config_.device != "rubin" ||
        config_.model != model_.name ||
        (config_.precision != "fp16" && config_.precision != "bf16") ||
        config_.num_layers != model_.num_layers ||
        parallelism_.tensor_parallel_size == 0 ||
        parallelism_.pipeline_parallel_size == 0 ||
        config_.num_layers % parallelism_.pipeline_parallel_size != 0) {
        throw BatchExecutionModelError(
            "unsupported analytical model configuration");
    }
}

BatchExecutionPrediction AnalyticalRooflineExecutionTimePredictor::predict(
    const entities::Batch &batch,
    const std::vector<entities::Request> &requests, StageId stage_id) const {
    if (!stage_id.valid() ||
        stage_id.index() >= parallelism_.pipeline_parallel_size) {
        throw BatchExecutionModelError(
            "analytical stage ID exceeds pipeline size");
    }

    detail::DenseBatch dense_batch = [&]() {
        detail::DenseBatch value{};
        value.total_tokens = batch.total_scheduled_tokens();
        value.prefill_requests = {};
        value.decode_requests = {};
        return value;
    }();
    for (const entities::RequestBatchSnapshot &snapshot : batch.requests()) {
        const entities::Request &request =
            get_request(requests, snapshot.request_id);
        // Production Python evaluates analytical attention from the request's
        // mutable state when each PP stage starts, rather than from the batch
        // creation snapshot. Earlier overlapping stages may have made
        // additional progress visible by then.
        const detail::AttentionRequestSlice slice = [&]() {
            detail::AttentionRequestSlice value{};
            value.query_tokens = snapshot.scheduled_tokens;
            value.past_context = request.num_processed_tokens();
            return value;
        }();
        if (!request.is_prefill_complete()) {
            dense_batch.prefill_requests.push_back(slice);
        } else {
            dense_batch.decode_requests.push_back(slice);
        }
    }

    const detail::DenseLayerTimes layer = detail::predict_dense_layer(
        detail::DeviceCeilings::rubin(), detail::AnalyticalConfig{},
        [&]() {
            detail::DenseModel value{};
            value.hidden_size = model_.hidden_size;
            value.intermediate_size = model_.intermediate_size;
            value.num_query_heads = model_.num_query_heads;
            value.num_kv_heads = model_.num_kv_heads;
            value.head_dim = model_.head_dim;
            value.tensor_parallel_size = parallelism_.tensor_parallel_size;
            value.gated_mlp = model_.gated_mlp;
            value.fused_add_norm = model_.fused_add_norm;
            return value;
        }(),
        dense_batch,
        config_.precision == "bf16" ? detail::Precision::kBf16
                                    : detail::Precision::kFp16);

    if (dense_batch.total_tokens > std::numeric_limits<std::uint64_t>::max() /
                                       (model_.hidden_size * 2ULL)) {
        throw BatchExecutionModelError(
            "analytical communication byte count overflows uint64");
    }
    const std::uint64_t activation_bytes =
        dense_batch.total_tokens * model_.hidden_size * 2ULL;
    const double allreduce_ms =
        parallelism_.tensor_parallel_size > 1
            ? communication_backend_->allreduce_ms(
                  activation_bytes, parallelism_.tensor_parallel_size, true)
            : 0.0;
    const double dense_layer_compute_ms = layer.total_ms();
    const double attention_layer_compute_ms =
        layer.attention_pre_projection_ms + layer.attention_post_projection_ms +
        layer.rope_ms + layer.kv_cache_save_ms + layer.attention_norm_ms +
        layer.prefill_attention_ms + layer.decode_attention_ms;
    const double tp_layer_ms = 2.0 * allreduce_ms;
    const std::uint64_t layers_per_stage =
        config_.num_layers / parallelism_.pipeline_parallel_size;
    double dense_compute_ms =
        static_cast<double>(layers_per_stage) * dense_layer_compute_ms;
    double tp_communication_ms =
        static_cast<double>(layers_per_stage) * tp_layer_ms;
    const double pp_communication_ms =
        stage_id.index() + 1 < parallelism_.pipeline_parallel_size
            ? communication_backend_->point_to_point_ms(activation_bytes, true)
            : 0.0;
    entities::ExecutionTime execution_time = [&]() {
        entities::ExecutionTime value{};
        value.dense_compute_ms = dense_compute_ms;
        value.tp_communication_ms = tp_communication_ms;
        value.pp_communication_ms = pp_communication_ms;
        return value;
    }();
    std::vector<std::pair<std::string, double>> moe_diagnostics;
    std::vector<MoERoutingDiagnostic> routing_diagnostics;
    if (model_.is_moe()) {
        execution_time.dense_compute_ms =
            static_cast<double>(layers_per_stage) * attention_layer_compute_ms;
        execution_time.tp_communication_ms =
            static_cast<double>(layers_per_stage) * allreduce_ms;
        execution_time.moe_gating_linear_ms = 0.0;
        execution_time.moe_gating_routing_topk_ms = 0.0;
        execution_time.moe_grouped_gemm_ms = 0.0;
        execution_time.moe_shuffling_ms = 0.0;
        execution_time.moe_post_attention_norm_ms = 0.0;

        const detail::MoEModel moe_model = [&]() {
            detail::MoEModel value{};
            value.hidden_size = model_.hidden_size;
            value.intermediate_size = model_.intermediate_size;
            value.model_num_experts = model_.model_num_experts;
            value.moe_tensor_parallel_size =
                parallelism_.moe_tensor_parallel_size;
            value.gated_mlp = model_.gated_mlp;
            value.fused_add_norm = model_.fused_add_norm;
            return value;
        }();
        const detail::Precision precision = config_.precision == "bf16"
                                                ? detail::Precision::kBf16
                                                : detail::Precision::kFp16;
        for (std::uint64_t local_layer = 0; local_layer < layers_per_stage;
             ++local_layer) {
            const detail::RoutingAllocation allocation = detail::route_tokens(
                dense_batch.total_tokens, model_.router_topk,
                model_.runtime_total_experts,
                parallelism_.moe_expert_parallel_size, routing_, local_layer);
            const detail::MoELanePrediction lane_prediction =
                detail::predict_moe_lanes(
                    detail::DeviceCeilings::rubin(), detail::AnalyticalConfig{},
                    moe_model, allocation, model_.router_topk, precision);
            std::vector<double> lane_times_ms;
            lane_times_ms.reserve(lane_prediction.lane_times.size());
            for (const detail::MoELayerTime &lane :
                 lane_prediction.lane_times) {
                lane_times_ms.push_back(lane.total_ms());
            }
            routing_diagnostics.push_back([&]() {
                MoERoutingDiagnostic value{};
                value.layer_id = LayerId{local_layer};
                value.input_tokens = allocation.input_tokens;
                value.routed_tokens = allocation.routed_tokens;
                value.global_expert_tokens = allocation.global_expert_tokens;
                value.lane_expert_tokens = allocation.lane_expert_tokens;
                value.lane_times_ms = std::move(lane_times_ms);
                value.critical_lane = lane_prediction.critical_lane;
                value.critical_lane_time_ms =
                    lane_prediction.critical_lane_time_ms;
                return value;
            }());
            const detail::MoELayerTime &critical =
                lane_prediction.lane_times.at(
                    static_cast<std::size_t>(lane_prediction.critical_lane));
            execution_time.moe_gating_linear_ms += critical.gating_linear_ms;
            execution_time.moe_gating_routing_topk_ms +=
                critical.gating_routing_topk_ms;
            execution_time.moe_grouped_gemm_ms +=
                critical.grouped_up_projection_ms +
                critical.grouped_down_projection_ms;
            execution_time.moe_shuffling_ms += critical.shuffling_ms;
            execution_time.moe_post_attention_norm_ms +=
                critical.post_attention_norm_ms + 2.0 * layer.residual_add_ms;
            moe_diagnostics.emplace_back(
                "layer_" + std::to_string(local_layer) + "_critical_lane",
                static_cast<double>(lane_prediction.critical_lane));
            moe_diagnostics.emplace_back(
                "layer_" + std::to_string(local_layer) + "_critical_lane_ms",
                lane_prediction.critical_lane_time_ms);
        }
        const detail::MoECommunicationTime communication_time =
            detail::predict_moe_communication(
                *communication_backend_, dense_batch.total_tokens,
                model_.hidden_size,
                dense_batch.total_tokens * model_.router_topk,
                parallelism_.tensor_parallel_size,
                parallelism_.moe_tensor_parallel_size,
                parallelism_.moe_expert_parallel_size,
                parallelism_.data_parallel_size, false, 2.0);
        execution_time.moe_tp_communication_ms =
            static_cast<double>(layers_per_stage) *
            communication_time.moe_tp_ms;
        execution_time.ep_dispatch_ms = static_cast<double>(layers_per_stage) *
                                        communication_time.ep_dispatch_ms;
        execution_time.ep_combine_ms = static_cast<double>(layers_per_stage) *
                                       communication_time.ep_combine_ms;
        if (batch.cluster_type() == ClusterType::kDecode) {
            execution_time.dp_input_communication_ms =
                static_cast<double>(layers_per_stage) *
                communication_time.dp_input_ms;
            execution_time.dp_output_communication_ms =
                static_cast<double>(layers_per_stage) *
                communication_time.dp_output_ms;
        }
    }
    const double duration_ms = execution_time.total_ms();
    if (!std::isfinite(duration_ms) || duration_ms < 0.0) {
        throw BatchExecutionModelError("analytical stage duration is invalid");
    }

    BatchExecutionPrediction result = [&]() {
        BatchExecutionPrediction value{};
        value.duration_ms = duration_ms;
        value.execution_time = execution_time;
        value.diagnostics = {
            {"total_tokens", static_cast<double>(dense_batch.total_tokens)},
            {
                "prefill_request_count",
                static_cast<double>(dense_batch.prefill_requests.size()),
            },
            {
                "decode_request_count",
                static_cast<double>(dense_batch.decode_requests.size()),
            },
            {"dense_layer_compute_ms", dense_layer_compute_ms},
            {"tp_allreduce_ms", allreduce_ms},
            {
                "dense_layer_total_ms",
                dense_layer_compute_ms + tp_layer_ms,
            },
            {
                "num_layers",
                static_cast<double>(layers_per_stage),
            },
            {"dense_compute_ms", dense_compute_ms},
            {"tp_communication_ms", tp_communication_ms},
            {"pp_communication_ms", pp_communication_ms},
            {"stage_duration_ms", duration_ms},
            {"batch_duration_ms", duration_ms},
        };
        value.moe_routing = std::move(routing_diagnostics);
        return value;
    }();
    result.diagnostics.insert(result.diagnostics.end(), moe_diagnostics.begin(),
                              moe_diagnostics.end());
    return result;
}

std::shared_ptr<const BatchExecutionModel> make_batch_execution_model(
    const config::ExecutionModelConfig &config,
    const std::optional<config::ParallelismConfig> &parallelism,
    const std::optional<config::ModelConfig> &model,
    const std::optional<config::MoeRoutingConfig> &routing,
    std::shared_ptr<const cc_backend::BaseCCBackend> communication_backend) {
    if (config.type == config::ExecutionModelType::kFixed) {
        return std::make_shared<FixedBatchExecutionModel>(
            config.fixed, parallelism.value_or(config::ParallelismConfig{}),
            model.value_or(config::ModelConfig{}),
            routing.value_or(config::MoeRoutingConfig{}));
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
