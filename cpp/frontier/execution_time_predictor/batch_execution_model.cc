#include "frontier/execution_time_predictor/batch_execution_model.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

#include "frontier/cc_backend/analytical_model.h"
#include "frontier/execution_time_predictor/analytical_model.h"

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
    : config_(std::move(config)) {
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
  static_cast<void>(batch);
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
  return BatchExecutionPrediction{
      .duration_ms = latency,
      .execution_time = entities::ExecutionTime{
          .dense_compute_ms = latency,
      },
      .diagnostics = {
          {"fixed_stage_latency_ms", latency},
      },
  };
}

std::unique_ptr<BatchExecutionModel>
FixedBatchExecutionModel::clone() const {
  return std::make_unique<FixedBatchExecutionModel>(config_);
}

AnalyticalBatchExecutionModel::AnalyticalBatchExecutionModel(
    config::AnalyticalExecutionModelConfig config)
    : AnalyticalBatchExecutionModel(
          config,
          config::ParallelismConfig{
              .tensor_parallel_size =
                  config.tensor_parallel_size,
          }) {}

AnalyticalBatchExecutionModel::AnalyticalBatchExecutionModel(
    config::AnalyticalExecutionModelConfig config,
    config::ParallelismConfig parallelism)
    : config_(std::move(config)),
      parallelism_(parallelism) {
  if (parallelism_.tensor_parallel_size == 0) {
    parallelism_.tensor_parallel_size =
        config_.tensor_parallel_size;
  }
  if (config_.device != "rubin" ||
      config_.model != "llama2-7b" ||
      config_.precision != "fp16" ||
      config_.num_layers != 32 ||
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
      DenseModel::llama2_7b(
          parallelism_.tensor_parallel_size),
      dense_batch,
      Precision::kFp16);

  if (dense_batch.total_tokens >
      std::numeric_limits<std::uint64_t>::max() /
          (4'096ULL * 2ULL)) {
    throw BatchExecutionModelError(
        "analytical communication byte count overflows uint64");
  }
  const std::uint64_t activation_bytes =
      dense_batch.total_tokens * 4'096ULL * 2ULL;
  const cc_backend::AnalyticalCommunicationModel communication{
      cc_backend::AnalyticalCommunicationConfig{
          .network_bandwidth_gbps =
              config_.network_bandwidth_gbps,
          .latency_us = config_.network_latency_us,
          .intra_node_bandwidth_gbps =
              config_.intra_node_bandwidth_gbps,
      }};
  const double allreduce_ms =
      parallelism_.tensor_parallel_size > 1
      ? communication.allreduce_ms(
            activation_bytes,
            parallelism_.tensor_parallel_size,
            true)
      : 0.0;
  const double layer_compute_ms = layer.total_ms();
  const double tp_layer_ms = 2.0 * allreduce_ms;
  const std::uint64_t layers_per_stage =
      config_.num_layers /
      parallelism_.pipeline_parallel_size;
  const double dense_compute_ms =
      static_cast<double>(layers_per_stage) *
      layer_compute_ms;
  const double tp_communication_ms =
      static_cast<double>(layers_per_stage) *
      tp_layer_ms;
  const double pp_communication_ms =
      stage_id.index() + 1 <
              parallelism_.pipeline_parallel_size
      ? communication.point_to_point_ms(
            activation_bytes, true)
      : 0.0;
  const entities::ExecutionTime execution_time{
      .dense_compute_ms = dense_compute_ms,
      .tp_communication_ms = tp_communication_ms,
      .pp_communication_ms = pp_communication_ms,
  };
  const double duration_ms = execution_time.total_ms();
  if (!std::isfinite(duration_ms) || duration_ms < 0.0) {
    throw BatchExecutionModelError(
        "analytical stage duration is invalid");
  }

  return BatchExecutionPrediction{
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
          {"dense_layer_compute_ms", layer_compute_ms},
          {"tp_allreduce_ms", allreduce_ms},
          {
              "dense_layer_total_ms",
              layer_compute_ms + tp_layer_ms,
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
  };
}

std::unique_ptr<BatchExecutionModel>
AnalyticalBatchExecutionModel::clone() const {
  return std::make_unique<AnalyticalBatchExecutionModel>(
      config_, parallelism_);
}

std::unique_ptr<BatchExecutionModel>
make_batch_execution_model(
    const config::ExecutionModelConfig& config,
    const std::optional<config::ParallelismConfig>& parallelism) {
  if (config.type == config::ExecutionModelType::kFixed) {
    return std::make_unique<FixedBatchExecutionModel>(
        config.fixed);
  }
  config::ParallelismConfig resolved =
      parallelism.value_or(config::ParallelismConfig{});
  if (!parallelism.has_value()) {
    resolved.tensor_parallel_size =
        config.analytical.tensor_parallel_size;
  }
  return std::make_unique<AnalyticalBatchExecutionModel>(
      config.analytical, resolved);
}

}  // namespace frontier::execution_time_predictor
