#include "frontier/execution_time_predictor/batch_execution_model.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

#include "frontier/cc_backend/analytical_model.h"
#include "frontier/execution_time_predictor/analytical_model.h"
#include "frontier/moe/analytical_model.h"
#include "frontier/moe/routing.h"

namespace frontier::execution_time_predictor {
namespace {

const entities::Request& get_request(
    const std::vector<entities::Request>& requests,
    RequestId request_id) {
  if (!request_id.valid() ||
      request_id.index() >= requests.size()) {
    throw BatchExecutionModelError(
        "batch execution references an unknown request");
  }
  const entities::Request& request =
      requests.at(request_id.index());
  if (request.id() != request_id) {
    throw BatchExecutionModelError(
        "batch execution request arena invariant failed");
  }
  return request;
}

}  // namespace

FixedBatchExecutionModel::FixedBatchExecutionModel(
    config::FixedExecutionModelConfig config)
    : FixedBatchExecutionModel(
          std::move(config),
          config::ParallelismConfig{},
          config::ModelConfig{}) {}

FixedBatchExecutionModel::FixedBatchExecutionModel(
    config::FixedExecutionModelConfig config,
    config::ParallelismConfig parallelism,
    config::ModelConfig model,
    config::MoeRoutingConfig routing)
    : config_(std::move(config)),
      parallelism_(parallelism),
      model_(std::move(model)),
      routing_(routing) {
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
    const entities::Batch& batch,
    const std::vector<entities::Request>& requests,
    StageId stage_id) const {
  static_cast<void>(requests);
  double latency = config_.batch_latency_ms;
  if (!config_.stage_latencies_ms.empty()) {
    if (!stage_id.valid() ||
        stage_id.index() >= config_.stage_latencies_ms.size()) {
      throw BatchExecutionModelError(
          "fixed stage latency does not cover stage ID");
    }
    latency = config_.stage_latencies_ms.at(
        stage_id.index());
  }
  entities::ExecutionTime execution_time{
      .dense_compute_ms = latency,
  };
  if (model_.is_moe()) {
    const std::uint64_t layers_per_stage =
        model_.num_layers /
        std::max<std::uint64_t>(
            1, parallelism_.pipeline_parallel_size);
    const double component =
        latency * static_cast<double>(layers_per_stage);
    execution_time = entities::ExecutionTime{
        .dense_compute_ms = component * 7.0,
        .tp_communication_ms =
            parallelism_.tensor_parallel_size > 1
            ? component
            : 0.0,
        .pp_communication_ms =
            stage_id.index() + 1 <
                    parallelism_.pipeline_parallel_size
            ? latency
            : 0.0,
        .moe_gating_linear_ms = component * 0.5,
        .moe_gating_routing_topk_ms = component * 0.5,
        .moe_grouped_gemm_ms = component,
        .moe_shuffling_ms = component,
        // Phi dummy timing has one MoE norm and one FFN residual add per
        // layer. They share this normalized post-attention bucket.
        .moe_post_attention_norm_ms = component * 2.0,
        .moe_tp_communication_ms =
            parallelism_.moe_tensor_parallel_size > 1
            ? component
            : 0.0,
        .ep_dispatch_ms =
            parallelism_.moe_expert_parallel_size > 1
            ? component
            : 0.0,
        .ep_combine_ms =
            parallelism_.moe_expert_parallel_size > 1
            ? component
            : 0.0,
        .dp_input_communication_ms =
            parallelism_.data_parallel_size > 1
            ? component
            : 0.0,
        .dp_output_communication_ms =
            parallelism_.data_parallel_size > 1
            ? component
            : 0.0,
    };
  }
  std::vector<MoERoutingDiagnostic> routing_diagnostics;
  if (model_.is_moe()) {
    const std::uint64_t layers_per_stage =
        model_.num_layers /
        parallelism_.pipeline_parallel_size;
    routing_diagnostics.reserve(
        static_cast<std::size_t>(layers_per_stage));
    for (std::uint64_t local_layer = 0;
         local_layer < layers_per_stage;
         ++local_layer) {
      const moe::RoutingAllocation allocation = moe::route_tokens(
          batch.total_scheduled_tokens(),
          model_.router_topk,
          model_.runtime_total_experts,
          parallelism_.moe_expert_parallel_size,
          routing_,
          local_layer);
      routing_diagnostics.push_back(MoERoutingDiagnostic{
          .layer_id = LayerId{local_layer},
          .input_tokens = allocation.input_tokens,
          .routed_tokens = allocation.routed_tokens,
          .global_expert_tokens =
              allocation.global_expert_tokens,
          .lane_expert_tokens =
              allocation.lane_expert_tokens,
          .lane_times_ms = std::vector<double>(
              static_cast<std::size_t>(
                  parallelism_.moe_expert_parallel_size),
              latency),
          .critical_lane = 0,
          .critical_lane_time_ms = latency,
      });
    }
  }
  return BatchExecutionPrediction{
      .duration_ms = execution_time.total_ms(),
      .execution_time = execution_time,
      .diagnostics = {
          {"fixed_stage_latency_ms", latency},
          {"stage_duration_ms", execution_time.total_ms()},
      },
      .moe_routing = std::move(routing_diagnostics),
  };
}

AnalyticalBatchExecutionModel::AnalyticalBatchExecutionModel(
    config::AnalyticalExecutionModelConfig config)
    : AnalyticalBatchExecutionModel(
          config,
          config::ParallelismConfig{
              .tensor_parallel_size =
                  config.tensor_parallel_size,
          },
          config::ModelConfig{},
          config::MoeRoutingConfig{}) {}

AnalyticalBatchExecutionModel::AnalyticalBatchExecutionModel(
    config::AnalyticalExecutionModelConfig config,
    config::ParallelismConfig parallelism)
    : AnalyticalBatchExecutionModel(
          std::move(config),
          parallelism,
          config::ModelConfig{},
          config::MoeRoutingConfig{}) {}

AnalyticalBatchExecutionModel::AnalyticalBatchExecutionModel(
    config::AnalyticalExecutionModelConfig config,
    config::ParallelismConfig parallelism,
    config::ModelConfig model,
    config::MoeRoutingConfig routing)
    : AnalyticalBatchExecutionModel(
          config,
          std::move(parallelism),
          std::move(model),
          std::move(routing),
          cc_backend::make_analytical_cc_backend(
              cc_backend::AnalyticalCommunicationConfig{
                  .network_bandwidth_gbps =
                      config.network_bandwidth_gbps,
                  .latency_us = config.network_latency_us,
                  .intra_node_bandwidth_gbps =
                      config.intra_node_bandwidth_gbps,
              })) {}

AnalyticalBatchExecutionModel::AnalyticalBatchExecutionModel(
    config::AnalyticalExecutionModelConfig config,
    config::ParallelismConfig parallelism,
    config::ModelConfig model,
    config::MoeRoutingConfig routing,
    std::shared_ptr<const cc_backend::BaseCCBackend>
        communication_backend)
    : config_(std::move(config)),
      parallelism_(parallelism),
      model_(std::move(model)),
      routing_(routing),
      communication_backend_(std::move(communication_backend)) {
  if (parallelism_.tensor_parallel_size == 0) {
    parallelism_.tensor_parallel_size =
        config_.tensor_parallel_size;
  }
  if (communication_backend_ == nullptr ||
      config_.device != "rubin" ||
      config_.model != model_.name ||
      (config_.precision != "fp16" &&
       config_.precision != "bf16") ||
      config_.num_layers != model_.num_layers ||
      parallelism_.tensor_parallel_size == 0 ||
      parallelism_.pipeline_parallel_size == 0 ||
      config_.num_layers %
              parallelism_.pipeline_parallel_size !=
          0) {
    throw BatchExecutionModelError(
        "unsupported analytical model configuration");
  }
}

BatchExecutionPrediction AnalyticalBatchExecutionModel::predict(
    const entities::Batch& batch,
    const std::vector<entities::Request>& requests,
    StageId stage_id) const {
  if (!stage_id.valid() ||
      stage_id.index() >=
      parallelism_.pipeline_parallel_size) {
    throw BatchExecutionModelError(
        "analytical stage ID exceeds pipeline size");
  }

  DenseBatch dense_batch{
      .total_tokens = batch.total_scheduled_tokens(),
      .prefill_requests = {},
      .decode_requests = {},
  };
  for (const entities::RequestBatchSnapshot& snapshot :
       batch.requests()) {
    const entities::Request& request =
        get_request(requests, snapshot.request_id);
    // Production Python evaluates analytical attention from the request's
    // mutable state when each PP stage starts, rather than from the batch
    // creation snapshot. Earlier overlapping stages may have made additional
    // progress visible by then.
    const AttentionRequestSlice slice{
        .query_tokens = snapshot.scheduled_tokens,
        .past_context = request.num_processed_tokens(),
    };
    if (!request.is_prefill_complete()) {
      dense_batch.prefill_requests.push_back(slice);
    } else {
      dense_batch.decode_requests.push_back(slice);
    }
  }

  const DenseLayerTimes layer = predict_dense_layer(
      DeviceCeilings::rubin(),
      AnalyticalConfig{},
      DenseModel{
          .hidden_size = model_.hidden_size,
          .intermediate_size = model_.intermediate_size,
          .num_query_heads = model_.num_query_heads,
          .num_kv_heads = model_.num_kv_heads,
          .head_dim = model_.head_dim,
          .tensor_parallel_size =
              parallelism_.tensor_parallel_size,
          .gated_mlp = model_.gated_mlp,
          .fused_add_norm = model_.fused_add_norm,
      },
      dense_batch,
      config_.precision == "bf16"
          ? Precision::kBf16
          : Precision::kFp16);

  if (dense_batch.total_tokens >
      std::numeric_limits<std::uint64_t>::max() /
          (model_.hidden_size * 2ULL)) {
    throw BatchExecutionModelError(
        "analytical communication byte count overflows uint64");
  }
  const std::uint64_t activation_bytes =
      dense_batch.total_tokens * model_.hidden_size * 2ULL;
  const double allreduce_ms =
      parallelism_.tensor_parallel_size > 1
      ? communication_backend_->allreduce_ms(
            activation_bytes,
            parallelism_.tensor_parallel_size,
            true)
      : 0.0;
  const double dense_layer_compute_ms = layer.total_ms();
  const double attention_layer_compute_ms =
      layer.attention_pre_projection_ms +
      layer.attention_post_projection_ms + layer.rope_ms +
      layer.kv_cache_save_ms + layer.attention_norm_ms +
      layer.prefill_attention_ms + layer.decode_attention_ms;
  const double tp_layer_ms = 2.0 * allreduce_ms;
  const std::uint64_t layers_per_stage =
      config_.num_layers /
      parallelism_.pipeline_parallel_size;
  double dense_compute_ms =
      static_cast<double>(layers_per_stage) *
      dense_layer_compute_ms;
  double tp_communication_ms =
      static_cast<double>(layers_per_stage) * tp_layer_ms;
  const double pp_communication_ms =
      stage_id.index() + 1 <
              parallelism_.pipeline_parallel_size
      ? communication_backend_->point_to_point_ms(
            activation_bytes, true)
      : 0.0;
  entities::ExecutionTime execution_time{
      .dense_compute_ms = dense_compute_ms,
      .tp_communication_ms = tp_communication_ms,
      .pp_communication_ms = pp_communication_ms,
  };
  std::vector<std::pair<std::string, double>> moe_diagnostics;
  std::vector<MoERoutingDiagnostic> routing_diagnostics;
  if (model_.is_moe()) {
    execution_time.dense_compute_ms =
        static_cast<double>(layers_per_stage) *
        attention_layer_compute_ms;
    execution_time.tp_communication_ms =
        static_cast<double>(layers_per_stage) *
        allreduce_ms;
    execution_time.moe_gating_linear_ms = 0.0;
    execution_time.moe_gating_routing_topk_ms = 0.0;
    execution_time.moe_grouped_gemm_ms = 0.0;
    execution_time.moe_shuffling_ms = 0.0;
    execution_time.moe_post_attention_norm_ms = 0.0;

    const moe::MoEModel moe_model{
        .hidden_size = model_.hidden_size,
        .intermediate_size = model_.intermediate_size,
        .model_num_experts = model_.model_num_experts,
        .moe_tensor_parallel_size =
            parallelism_.moe_tensor_parallel_size,
        .gated_mlp = model_.gated_mlp,
        .fused_add_norm = model_.fused_add_norm,
    };
    const Precision precision = config_.precision == "bf16"
        ? Precision::kBf16
        : Precision::kFp16;
    for (std::uint64_t local_layer = 0;
         local_layer < layers_per_stage;
         ++local_layer) {
      const moe::RoutingAllocation allocation = moe::route_tokens(
          dense_batch.total_tokens,
          model_.router_topk,
          model_.runtime_total_experts,
          parallelism_.moe_expert_parallel_size,
          routing_,
          local_layer);
      const moe::MoELanePrediction lane_prediction =
          moe::predict_moe_lanes(
              DeviceCeilings::rubin(),
              AnalyticalConfig{},
              moe_model,
              allocation,
              model_.router_topk,
              precision);
      std::vector<double> lane_times_ms;
      lane_times_ms.reserve(lane_prediction.lane_times.size());
      for (const moe::MoELayerTime& lane :
           lane_prediction.lane_times) {
        lane_times_ms.push_back(lane.total_ms());
      }
      routing_diagnostics.push_back(MoERoutingDiagnostic{
          .layer_id = LayerId{local_layer},
          .input_tokens = allocation.input_tokens,
          .routed_tokens = allocation.routed_tokens,
          .global_expert_tokens =
              allocation.global_expert_tokens,
          .lane_expert_tokens =
              allocation.lane_expert_tokens,
          .lane_times_ms = std::move(lane_times_ms),
          .critical_lane = lane_prediction.critical_lane,
          .critical_lane_time_ms =
              lane_prediction.critical_lane_time_ms,
      });
      const moe::MoELayerTime& critical =
          lane_prediction.lane_times.at(
              static_cast<std::size_t>(
                  lane_prediction.critical_lane));
      execution_time.moe_gating_linear_ms +=
          critical.gating_linear_ms;
      execution_time.moe_gating_routing_topk_ms +=
          critical.gating_routing_topk_ms;
      execution_time.moe_grouped_gemm_ms +=
          critical.grouped_up_projection_ms +
          critical.grouped_down_projection_ms;
      execution_time.moe_shuffling_ms += critical.shuffling_ms;
      execution_time.moe_post_attention_norm_ms +=
          critical.post_attention_norm_ms +
          2.0 * layer.residual_add_ms;
      moe_diagnostics.emplace_back(
          "layer_" + std::to_string(local_layer) +
              "_critical_lane",
          static_cast<double>(lane_prediction.critical_lane));
      moe_diagnostics.emplace_back(
          "layer_" + std::to_string(local_layer) +
              "_critical_lane_ms",
          lane_prediction.critical_lane_time_ms);
    }
    const moe::MoECommunicationTime communication_time =
        moe::predict_moe_communication(
            *communication_backend_,
            dense_batch.total_tokens,
            model_.hidden_size,
            dense_batch.total_tokens * model_.router_topk,
            parallelism_.tensor_parallel_size,
            parallelism_.moe_tensor_parallel_size,
            parallelism_.moe_expert_parallel_size,
            parallelism_.data_parallel_size,
            false,
            2.0);
    execution_time.moe_tp_communication_ms =
        static_cast<double>(layers_per_stage) *
        communication_time.moe_tp_ms;
    execution_time.ep_dispatch_ms =
        static_cast<double>(layers_per_stage) *
        communication_time.ep_dispatch_ms;
    execution_time.ep_combine_ms =
        static_cast<double>(layers_per_stage) *
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
    throw BatchExecutionModelError(
        "analytical stage duration is invalid");
  }

  BatchExecutionPrediction result{
      .duration_ms = duration_ms,
      .execution_time = execution_time,
      .diagnostics = {
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
      },
      .moe_routing = std::move(routing_diagnostics),
  };
  result.diagnostics.insert(
      result.diagnostics.end(),
      moe_diagnostics.begin(),
      moe_diagnostics.end());
  return result;
}

std::shared_ptr<const BatchExecutionModel>
make_batch_execution_model(
    const config::ExecutionModelConfig& config,
    const std::optional<config::ParallelismConfig>& parallelism,
    const std::optional<config::ModelConfig>& model,
    const std::optional<config::MoeRoutingConfig>& routing,
    std::shared_ptr<const cc_backend::BaseCCBackend>
        communication_backend) {
  if (config.type == config::ExecutionModelType::kFixed) {
    return std::make_shared<FixedBatchExecutionModel>(
        config.fixed,
        parallelism.value_or(config::ParallelismConfig{}),
        model.value_or(config::ModelConfig{}),
        routing.value_or(config::MoeRoutingConfig{}));
  }
  config::ParallelismConfig resolved =
      parallelism.value_or(config::ParallelismConfig{});
  if (!parallelism.has_value()) {
    resolved.tensor_parallel_size =
        config.analytical.tensor_parallel_size;
  }
  return std::make_shared<AnalyticalBatchExecutionModel>(
      config.analytical,
      resolved,
      model.value_or(config::ModelConfig{}),
      routing.value_or(config::MoeRoutingConfig{}),
      communication_backend != nullptr
          ? std::move(communication_backend)
          : cc_backend::make_analytical_cc_backend(
                cc_backend::AnalyticalCommunicationConfig{
                    .network_bandwidth_gbps =
                        config.analytical.network_bandwidth_gbps,
                    .latency_us =
                        config.analytical.network_latency_us,
                    .intra_node_bandwidth_gbps =
                        config.analytical
                            .intra_node_bandwidth_gbps,
                }));
}

}  // namespace frontier::execution_time_predictor
