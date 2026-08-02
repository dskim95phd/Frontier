#include "frontier/execution_time_predictor/analytical_roofline_execution_time_predictor.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <string>
#include <utility>

#include "frontier/attention/mla.h"
#include "frontier/cc_backend/analytical_model.h"

namespace frontier::execution_time_predictor {
namespace {

const entities::Request &
get_request(const std::vector<entities::Request> &requests,
            RequestId request_id) {
    if (!request_id.valid() || request_id.index() >= requests.size()) {
        throw ExecutionTimePredictorError(
            "batch execution references an unknown request");
    }
    const entities::Request &request = requests.at(request_id.index());
    if (request.id() != request_id) {
        throw ExecutionTimePredictorError(
            "batch execution request arena invariant failed");
    }
    return request;
}

struct StageBatchInfo {
    detail::DenseBatch dense_batch;
    std::uint64_t lm_head_tokens = 0;
};

detail::AttentionRequestSlice
make_attention_request_slice(const entities::RequestBatchSnapshot &snapshot,
                             const entities::Request &request) {
    return detail::AttentionRequestSlice{
        snapshot.scheduled_tokens,
        request.num_processed_tokens(),
    };
}

StageBatchInfo
build_stage_batch_info(const entities::Batch &batch,
                       const std::vector<entities::Request> &requests) {
    StageBatchInfo result{};
    result.dense_batch.total_tokens = batch.total_scheduled_tokens();
    for (const entities::RequestBatchSnapshot &snapshot : batch.requests()) {
        const entities::Request &request =
            get_request(requests, snapshot.request_id);
        // Production Python evaluates analytical attention from the request's
        // mutable state when each PP stage starts, rather than from the batch
        // creation snapshot. Earlier overlapping stages may have made
        // additional progress visible by then.
        const detail::AttentionRequestSlice slice =
            make_attention_request_slice(snapshot, request);
        if (!request.is_prefill_complete()) {
            result.dense_batch.prefill_requests.push_back(slice);
            if (snapshot.processed_tokens + snapshot.scheduled_tokens >=
                request.num_prefill_tokens()) {
                ++result.lm_head_tokens;
            }
        } else {
            result.dense_batch.decode_requests.push_back(slice);
            ++result.lm_head_tokens;
        }
    }
    return result;
}

detail::DenseModel
make_dense_model(const config::ModelConfig &model,
                 const config::ParallelismConfig &parallelism) {
    detail::DenseModel result{};
    result.hidden_size = model.hidden_size;
    result.intermediate_size = model.dense_intermediate_size;
    result.num_query_heads = model.num_query_heads;
    result.num_kv_heads = model.num_kv_heads;
    result.head_dim = model.head_dim;
    result.tensor_parallel_size = parallelism.tensor_parallel_size;
    result.gated_mlp = model.gated_mlp;
    result.fused_add_norm = model.fused_add_norm;
    result.use_mla = model.use_mla;
    result.use_mfa = model.use_mfa;
    result.q_lora_rank = model.q_lora_rank;
    result.kv_lora_rank = model.kv_lora_rank;
    result.qk_nope_head_dim = model.qk_nope_head_dim;
    result.qk_rope_head_dim = model.qk_rope_head_dim;
    result.qk_head_dim = model.qk_head_dim;
    result.v_head_dim = model.v_head_dim;
    result.share_q_dim = model.share_q_dim;
    return result;
}

detail::DenseOperatorPrecisions make_dense_operator_precisions(
    const config::AnalyticalExecutionModelConfig &config) {
    return detail::DenseOperatorPrecisions{
        detail::precision_from_string(config.attention_precision()),
        detail::precision_from_string(config.dense_precision()),
        detail::precision_from_string(config.kv_cache_precision()),
        detail::precision_from_string(config.attention_weight_precision()),
        detail::precision_from_string(config.attention_activation_precision()),
        detail::precision_from_string(config.dense_weight_precision()),
        detail::precision_from_string(config.dense_activation_precision()),
    };
}

detail::MoEModel make_moe_model(const config::ModelConfig &model,
                                const config::ParallelismConfig &parallelism) {
    detail::MoEModel result{};
    result.hidden_size = model.hidden_size;
    result.intermediate_size = model.moe_intermediate_size;
    result.model_num_experts = model.num_experts;
    result.num_shared_experts = model.num_shared_experts;
    result.moe_tensor_parallel_size = parallelism.moe_tensor_parallel_size;
    result.gated_mlp = model.gated_mlp;
    result.fused_add_norm = model.fused_add_norm;
    return result;
}

detail::MoEOperatorPrecisions make_moe_operator_precisions(
    const config::AnalyticalExecutionModelConfig &config) {
    return detail::MoEOperatorPrecisions{
        detail::precision_from_string(config.moe_expert_precision()),
        detail::precision_from_string(config.moe_router_precision()),
        detail::precision_from_string(config.dense_precision()),
        detail::precision_from_string(config.moe_expert_weight_precision()),
        detail::precision_from_string(config.moe_expert_activation_precision()),
        detail::precision_from_string(config.moe_router_weight_precision()),
        detail::precision_from_string(config.moe_router_activation_precision()),
        detail::precision_from_string(config.dense_weight_precision()),
        detail::precision_from_string(config.dense_activation_precision()),
    };
}

double total_attention_layer_compute_ms(const detail::DenseLayerTimes &layer) {
    return layer.attention_pre_projection_ms +
           layer.attention_post_projection_ms + layer.rope_ms +
           layer.kv_cache_save_ms + layer.attention_norm_ms +
           layer.attention_inter_norm_ms + layer.attention_wq_projection_ms +
           layer.prefill_attention_ms + layer.decode_attention_ms;
}

std::uint64_t activation_payload_bytes(std::uint64_t tokens,
                                       std::uint64_t hidden_size,
                                       double element_bytes) {
    const long double bytes = static_cast<long double>(tokens) *
                              static_cast<long double>(hidden_size) *
                              static_cast<long double>(element_bytes);
    if (bytes >
        static_cast<long double>(std::numeric_limits<std::uint64_t>::max())) {
        throw ExecutionTimePredictorError(
            "analytical communication byte count overflows uint64");
    }
    return static_cast<std::uint64_t>(std::ceil(bytes));
}

std::vector<double> lane_times_ms(const detail::MoELanePrediction &prediction) {
    std::vector<double> result;
    result.reserve(prediction.lane_times.size());
    for (const detail::MoELayerTime &lane : prediction.lane_times) {
        result.push_back(lane.total_ms());
    }
    return result;
}

MoERoutingDiagnostic make_moe_routing_diagnostic(
    std::uint64_t moe_layer_index, std::uint64_t model_layer,
    double pre_moe_compute_ms, const detail::RoutingAllocation &allocation,
    const detail::MoELanePrediction &lane_prediction) {
    MoERoutingDiagnostic result{};
    result.layer_id = LayerId{moe_layer_index};
    result.model_layer_id = model_layer;
    result.pre_moe_compute_ms = pre_moe_compute_ms;
    result.input_tokens = allocation.input_tokens;
    result.routed_tokens = allocation.routed_tokens;
    result.global_expert_tokens = allocation.global_expert_tokens;
    result.lane_expert_tokens = allocation.lane_expert_tokens;
    result.lane_times_ms = lane_times_ms(lane_prediction);
    result.critical_lane = lane_prediction.critical_lane;
    result.critical_lane_time_ms = lane_prediction.critical_lane_time_ms;
    return result;
}

void add_critical_moe_layer_time(entities::ExecutionTime &execution_time,
                                 const detail::MoELayerTime &critical_layer,
                                 double residual_add_ms) {
    execution_time.moe_gating_linear_ms += critical_layer.gating_linear_ms;
    execution_time.moe_gating_routing_topk_ms +=
        critical_layer.gating_routing_topk_ms;
    execution_time.moe_grouped_gemm_ms +=
        critical_layer.grouped_up_projection_ms +
        critical_layer.grouped_down_projection_ms;
    execution_time.moe_shuffling_ms += critical_layer.shuffling_ms;
    execution_time.moe_post_attention_norm_ms +=
        critical_layer.post_attention_norm_ms + 2.0 * residual_add_ms;
}

struct MoEStagePrediction {
    entities::ExecutionTime execution_time;
    std::vector<std::pair<std::string, double>> diagnostics;
    std::vector<MoERoutingDiagnostic> routing_diagnostics;
    double suffix_compute_ms = 0.0;
};

struct MoEStageContext {
    ClusterType cluster_type;
    const StageBatchInfo &batch_info;
    const config::AnalyticalExecutionModelConfig &config;
    const config::ParallelismConfig &parallelism;
    const config::ModelConfig &model;
    const config::MoeRoutingConfig &routing;
    const cc_backend::BaseCCBackend &communication_backend;
    config::PipelineStageLayerRange stage_layers;
    const detail::DenseLayerTimes &dense_layer;
    double allreduce_ms;
    double communication_element_bytes;
    entities::ExecutionTime base_execution_time;

    [[nodiscard]] double dense_layer_compute_ms() const noexcept {
        return dense_layer.total_ms();
    }

    [[nodiscard]] double attention_compute_ms() const noexcept {
        return total_attention_layer_compute_ms(dense_layer);
    }

    [[nodiscard]] double tp_layer_ms() const noexcept {
        return 2.0 * allreduce_ms;
    }
};

MoEStagePrediction predict_moe_stage_execution(const MoEStageContext &context) {
    const StageBatchInfo &batch_info = context.batch_info;
    const config::AnalyticalExecutionModelConfig &config = context.config;
    const config::ParallelismConfig &parallelism = context.parallelism;
    const config::ModelConfig &model = context.model;
    const config::MoeRoutingConfig &routing = context.routing;
    const config::PipelineStageLayerRange &stage_layers = context.stage_layers;
    const detail::DenseLayerTimes &dense_layer = context.dense_layer;
    const double dense_layer_compute_ms = context.dense_layer_compute_ms();
    const double attention_compute_ms = context.attention_compute_ms();
    MoEStagePrediction result{};
    result.execution_time = context.base_execution_time;
    result.execution_time.dense_compute_ms = 0.0;
    result.execution_time.tp_communication_ms = 0.0;
    result.execution_time.moe_gating_linear_ms = 0.0;
    result.execution_time.moe_gating_routing_topk_ms = 0.0;
    result.execution_time.moe_grouped_gemm_ms = 0.0;
    result.execution_time.moe_shuffling_ms = 0.0;
    result.execution_time.moe_post_attention_norm_ms = 0.0;
    const detail::MoEModel moe_model = make_moe_model(model, parallelism);
    const detail::MoEOperatorPrecisions moe_precisions =
        make_moe_operator_precisions(config);
    double pending_pre_moe_compute_ms = 0.0;
    std::uint64_t moe_layer_index = 0;
    for (std::uint64_t model_layer = stage_layers.begin;
         model_layer < stage_layers.end; ++model_layer) {
        if (!model.is_moe_layer(model_layer)) {
            result.execution_time.dense_compute_ms += dense_layer_compute_ms;
            result.execution_time.tp_communication_ms += context.tp_layer_ms();
            pending_pre_moe_compute_ms += dense_layer_compute_ms;
            continue;
        }
        result.execution_time.dense_compute_ms += attention_compute_ms;
        result.execution_time.tp_communication_ms += context.allreduce_ms;
        pending_pre_moe_compute_ms += attention_compute_ms;
        const detail::RoutingAllocation allocation = detail::route_tokens(
            batch_info.dense_batch.total_tokens, model.router_topk,
            model.total_expert_num, parallelism.moe_expert_parallel_size,
            routing, model_layer);
        const detail::MoELanePrediction lane_prediction =
            detail::predict_moe_lanes(
                detail::DeviceCeilings::rubin(), detail::AnalyticalConfig{},
                moe_model, allocation, model.router_topk, moe_precisions);
        result.routing_diagnostics.push_back(make_moe_routing_diagnostic(
            moe_layer_index, model_layer, pending_pre_moe_compute_ms,
            allocation, lane_prediction));
        const detail::MoELayerTime &critical = lane_prediction.lane_times.at(
            static_cast<std::size_t>(lane_prediction.critical_lane));
        add_critical_moe_layer_time(result.execution_time, critical,
                                    dense_layer.residual_add_ms);
        result.diagnostics.emplace_back(
            "layer_" + std::to_string(model_layer) + "_critical_lane",
            static_cast<double>(lane_prediction.critical_lane));
        result.diagnostics.emplace_back("layer_" + std::to_string(model_layer) +
                                            "_critical_lane_ms",
                                        lane_prediction.critical_lane_time_ms);
        pending_pre_moe_compute_ms = 0.0;
        ++moe_layer_index;
    }
    result.suffix_compute_ms = pending_pre_moe_compute_ms;
    const detail::MoECommunicationTime communication_time =
        detail::predict_moe_communication(
            context.communication_backend, batch_info.dense_batch.total_tokens,
            model.hidden_size,
            batch_info.dense_batch.total_tokens * model.router_topk,
            parallelism.tensor_parallel_size,
            parallelism.moe_tensor_parallel_size,
            parallelism.moe_expert_parallel_size,
            parallelism.data_parallel_size, false,
            context.communication_element_bytes);
    const double moe_layers =
        static_cast<double>(result.routing_diagnostics.size());
    result.execution_time.moe_tp_communication_ms =
        moe_layers * communication_time.moe_tp_ms;
    result.execution_time.ep_dispatch_ms =
        moe_layers * communication_time.ep_dispatch_ms;
    result.execution_time.ep_combine_ms =
        moe_layers * communication_time.ep_combine_ms;
    if (context.cluster_type == ClusterType::kDecode) {
        result.execution_time.dp_input_communication_ms =
            moe_layers * communication_time.dp_input_ms;
        result.execution_time.dp_output_communication_ms =
            moe_layers * communication_time.dp_output_ms;
    }
    return result;
}

double
kv_cache_bytes_per_token_per_layer(const config::ModelConfig &model,
                                   detail::Precision kv_cache_precision) {
    if (model.use_mla) {
        return attention::mla_kv_cache_bytes_per_token(
            attention::MlaKvCacheLayout{
                model.kv_lora_rank,
                model.qk_rope_head_dim,
                detail::bytes_per_element(kv_cache_precision),
                2.0,
            });
    }
    return static_cast<double>(model.runtime_num_kv_heads()) *
           static_cast<double>(model.runtime_head_size()) *
           static_cast<double>(model.kv_factor()) *
           detail::bytes_per_element(kv_cache_precision);
}

} // namespace

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
    const auto supported_precision = [](std::string_view value) {
        try {
            static_cast<void>(detail::precision_from_string(value));
            return true;
        } catch (const detail::AnalyticalModelError &) {
            return false;
        }
    };
    if (communication_backend_ == nullptr || config_.device != "rubin" ||
        !supported_precision(config_.precision) ||
        !supported_precision(config_.attention_precision()) ||
        !supported_precision(config_.dense_precision()) ||
        !supported_precision(config_.moe_expert_precision()) ||
        !supported_precision(config_.moe_router_precision()) ||
        !supported_precision(config_.kv_cache_precision()) ||
        !supported_precision(config_.communication_precision()) ||
        !supported_precision(config_.attention_weight_precision()) ||
        !supported_precision(config_.attention_activation_precision()) ||
        !supported_precision(config_.dense_weight_precision()) ||
        !supported_precision(config_.dense_activation_precision()) ||
        !supported_precision(config_.moe_expert_weight_precision()) ||
        !supported_precision(config_.moe_expert_activation_precision()) ||
        !supported_precision(config_.moe_router_weight_precision()) ||
        !supported_precision(config_.moe_router_activation_precision()) ||
        !supported_precision(config_.lm_head_weight_precision()) ||
        !supported_precision(config_.lm_head_activation_precision()) ||
        model_.attention.memory_layout ==
            attention::AttentionMemoryLayout::kFrozenDsa ||
        parallelism_.tensor_parallel_size == 0 ||
        parallelism_.pipeline_parallel_size == 0 ||
        parallelism_.pipeline_parallel_size > model_.num_layers) {
        throw ExecutionTimePredictorError(
            "unsupported analytical model configuration");
    }
}

ExecutionTimePrediction
AnalyticalRooflineExecutionTimePredictor::predict_stage_execution_time(
    const entities::Batch &batch,
    const std::vector<entities::Request> &requests, StageId stage_id) const {
    if (!stage_id.valid() ||
        stage_id.index() >= parallelism_.pipeline_parallel_size) {
        throw ExecutionTimePredictorError(
            "analytical stage ID exceeds pipeline size");
    }

    const StageBatchInfo batch_info = build_stage_batch_info(batch, requests);
    const detail::DenseBatch &dense_batch = batch_info.dense_batch;

    const detail::DenseOperatorPrecisions dense_precisions =
        make_dense_operator_precisions(config_);
    const detail::Precision communication_precision =
        detail::precision_from_string(config_.communication_precision());
    const detail::DenseLayerTimes layer = detail::predict_dense_layer(
        detail::DeviceCeilings::rubin(), detail::AnalyticalConfig{},
        make_dense_model(model_, parallelism_), dense_batch, dense_precisions);

    const double communication_element_bytes =
        detail::bytes_per_element(communication_precision);
    const std::uint64_t activation_bytes =
        activation_payload_bytes(dense_batch.total_tokens, model_.hidden_size,
                                 communication_element_bytes);
    const double allreduce_ms =
        parallelism_.tensor_parallel_size > 1
            ? communication_backend_->allreduce_ms(
                  activation_bytes, parallelism_.tensor_parallel_size, true)
            : 0.0;
    const double dense_layer_compute_ms = layer.total_ms();
    const double tp_layer_ms = 2.0 * allreduce_ms;
    const config::PipelineStageLayerRange stage_layers =
        config::pipeline_stage_layer_range(model_.num_layers,
                                           parallelism_.pipeline_parallel_size,
                                           stage_id.index());
    const std::uint64_t layers_per_stage = stage_layers.size();
    double dense_compute_ms =
        static_cast<double>(layers_per_stage) * dense_layer_compute_ms;
    double tp_communication_ms =
        static_cast<double>(layers_per_stage) * tp_layer_ms;
    const double pp_communication_ms =
        stage_id.index() + 1 < parallelism_.pipeline_parallel_size
            ? communication_backend_->point_to_point_ms(activation_bytes, true)
            : 0.0;
    entities::ExecutionTime execution_time{};
    execution_time.dense_compute_ms = dense_compute_ms;
    execution_time.tp_communication_ms = tp_communication_ms;
    execution_time.pp_communication_ms = pp_communication_ms;
    if (stage_id.index() + 1 == parallelism_.pipeline_parallel_size) {
        execution_time.lm_head_ms = detail::predict_output_projection_ms(
            detail::DeviceCeilings::rubin(), detail::AnalyticalConfig{},
            batch_info.lm_head_tokens, model_.hidden_size, model_.vocab_size,
            parallelism_.tensor_parallel_size,
            detail::precision_from_string(config_.lm_head_weight_precision()),
            detail::precision_from_string(
                config_.lm_head_activation_precision()));
    }
    std::vector<std::pair<std::string, double>> moe_diagnostics;
    std::vector<MoERoutingDiagnostic> routing_diagnostics;
    double moe_suffix_compute_ms = 0.0;
    if (model_.is_moe()) {
        const MoEStageContext moe_context{
            batch.cluster_type(),
            batch_info,
            config_,
            parallelism_,
            model_,
            routing_,
            *communication_backend_,
            stage_layers,
            layer,
            allreduce_ms,
            communication_element_bytes,
            execution_time,
        };
        const MoEStagePrediction moe_prediction =
            predict_moe_stage_execution(moe_context);
        execution_time = moe_prediction.execution_time;
        moe_diagnostics = moe_prediction.diagnostics;
        routing_diagnostics = moe_prediction.routing_diagnostics;
        moe_suffix_compute_ms = moe_prediction.suffix_compute_ms;
    }
    dense_compute_ms = execution_time.dense_compute_ms;
    tp_communication_ms = execution_time.tp_communication_ms;
    const double duration_ms = execution_time.total_ms();
    if (!std::isfinite(duration_ms) || duration_ms < 0.0) {
        throw ExecutionTimePredictorError(
            "analytical stage duration is invalid");
    }
    const double kv_cache_bytes_per_token =
        kv_cache_bytes_per_token_per_layer(model_, dense_precisions.kv_cache);

    ExecutionTimePrediction result{};
    result.duration_ms = duration_ms;
    result.execution_time = execution_time;
    result.diagnostics = {
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
        {"attention_weight_element_bytes",
         detail::bytes_per_element(*dense_precisions.attention_weight)},
        {"attention_activation_element_bytes",
         detail::bytes_per_element(*dense_precisions.attention_activation)},
        {"dense_weight_element_bytes",
         detail::bytes_per_element(*dense_precisions.dense_weight)},
        {"dense_activation_element_bytes",
         detail::bytes_per_element(*dense_precisions.dense_activation)},
        {"kv_cache_element_bytes",
         detail::bytes_per_element(dense_precisions.kv_cache)},
        {"kv_cache_bytes_per_token_per_layer", kv_cache_bytes_per_token},
        {"communication_element_bytes", communication_element_bytes},
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
        {"lm_head_ms", execution_time.lm_head_ms},
        {"lm_head_tokens", static_cast<double>(batch_info.lm_head_tokens)},
        {"stage_duration_ms", duration_ms},
        {"batch_duration_ms", duration_ms},
    };
    result.moe_routing = std::move(routing_diagnostics);
    result.moe_suffix_compute_ms = moe_suffix_compute_ms;
    result.diagnostics.insert(result.diagnostics.end(), moe_diagnostics.begin(),
                              moe_diagnostics.end());
    return result;
}

} // namespace frontier::execution_time_predictor

namespace frontier::execution_time_predictor::detail {
namespace {

void require_finite_nonnegative(double value, const char *field) {
    if (!std::isfinite(value) || value < 0.0) {
        throw AnalyticalModelError(std::string{field} +
                                   " must be finite and nonnegative");
    }
}

void validate_efficiency(const Efficiency &efficiency) {
    for (const auto &[value, name] : {
             std::pair{efficiency.compute, "compute efficiency"},
             std::pair{efficiency.memory, "memory efficiency"},
         }) {
        if (!std::isfinite(value) || value <= 0.0 || value > 1.0) {
            throw AnalyticalModelError(std::string{name} +
                                       " must satisfy 0 < value <= 1");
        }
    }
    if (!std::isfinite(efficiency.overlap_penalty) ||
        efficiency.overlap_penalty < 0.0 || efficiency.overlap_penalty > 1.0) {
        throw AnalyticalModelError(
            "overlap penalty must satisfy 0 <= value <= 1");
    }
}

std::uint64_t dense_ceil_div(std::uint64_t numerator,
                             std::uint64_t denominator) {
    if (denominator == 0) {
        throw AnalyticalModelError("division denominator must be positive");
    }
    return numerator / denominator +
           static_cast<std::uint64_t>(numerator % denominator != 0);
}

double predict_dense_kernel_ms(const DeviceCeilings &device,
                               Precision precision, const KernelWork &work,
                               const Efficiency &efficiency,
                               double launch_latency_us) {
    return predict_roofline(device, precision, work, efficiency,
                            launch_latency_us)
        .predicted_time_ms;
}

} // namespace

double DenseLayerTimes::total_ms() const noexcept {
    return attention_pre_projection_ms + attention_post_projection_ms +
           rope_ms + kv_cache_save_ms + attention_norm_ms +
           attention_inter_norm_ms + attention_wq_projection_ms +
           prefill_attention_ms + decode_attention_ms + mlp_up_projection_ms +
           mlp_activation_ms + mlp_down_projection_ms + mlp_norm_ms +
           2.0 * residual_add_ms;
}

Precision precision_from_string(std::string_view precision) {
    if (precision == "fp32") {
        return Precision::kFp32;
    }
    if (precision == "fp16") {
        return Precision::kFp16;
    }
    if (precision == "bf16") {
        return Precision::kBf16;
    }
    if (precision == "fp8") {
        return Precision::kFp8;
    }
    if (precision == "int8") {
        return Precision::kInt8;
    }
    if (precision == "fp4") {
        return Precision::kFp4;
    }
    if (precision == "int4") {
        return Precision::kInt4;
    }
    throw AnalyticalModelError("unsupported analytical precision: " +
                               std::string{precision});
}

double bytes_per_element(Precision precision) noexcept {
    switch (precision) {
    case Precision::kFp32:
        return 4.0;
    case Precision::kFp16:
    case Precision::kBf16:
        return 2.0;
    case Precision::kFp8:
    case Precision::kInt8:
        return 1.0;
    case Precision::kFp4:
    case Precision::kInt4:
        return 0.5;
    }
    return 0.0;
}

double peak_tflops(const DeviceCeilings &device, Precision precision) {
    double peak = 0.0;
    switch (precision) {
    case Precision::kFp32:
        peak = device.fp32_tflops;
        break;
    case Precision::kFp16:
    case Precision::kBf16:
        peak = device.fp16_tflops;
        break;
    case Precision::kFp8:
    case Precision::kInt8:
        peak = device.fp8_tflops;
        break;
    case Precision::kFp4:
    case Precision::kInt4:
        peak = device.fp4_tflops;
        break;
    }
    if (!std::isfinite(peak) || peak <= 0.0) {
        throw AnalyticalModelError(
            "device compute ceiling must be finite and positive");
    }
    return peak;
}

RooflineResult predict_roofline(const DeviceCeilings &device,
                                Precision precision, const KernelWork &work,
                                const Efficiency &efficiency,
                                double kernel_launch_latency_us) {
    require_finite_nonnegative(work.flops, "FLOPs");
    require_finite_nonnegative(work.hbm_bytes, "HBM bytes");
    validate_efficiency(efficiency);
    require_finite_nonnegative(kernel_launch_latency_us,
                               "kernel launch latency");
    if (!std::isfinite(device.hbm_bandwidth_tbps) ||
        device.hbm_bandwidth_tbps <= 0.0) {
        throw AnalyticalModelError("HBM bandwidth must be finite and positive");
    }

    if (work.flops == 0.0 && work.hbm_bytes == 0.0) {
        return [&]() {
            RooflineResult value{};
            value.compute_time_ms = 0.0;
            value.memory_time_ms = 0.0;
            value.launch_time_ms = 0.0;
            value.predicted_time_ms = 0.0;
            value.bottleneck = Bottleneck::kNone;
            return value;
        }();
    }

    const double compute_time_ms =
        work.flops /
        (peak_tflops(device, precision) * 1e12 * efficiency.compute) * 1e3;
    const double memory_time_ms =
        work.hbm_bytes /
        (device.hbm_bandwidth_tbps * 1e12 * efficiency.memory) * 1e3;
    const double launch_time_ms = kernel_launch_latency_us / 1e3;
    const double maximum = std::max(compute_time_ms, memory_time_ms);
    const double minimum = std::min(compute_time_ms, memory_time_ms);
    const double predicted_time_ms =
        launch_time_ms + maximum + efficiency.overlap_penalty * minimum;

    Bottleneck bottleneck = Bottleneck::kHbm;
    if (launch_time_ms >= maximum) {
        bottleneck = Bottleneck::kLaunch;
    } else if (compute_time_ms >= memory_time_ms) {
        bottleneck = Bottleneck::kCompute;
    }

    return [&]() {
        RooflineResult value{};
        value.compute_time_ms = compute_time_ms;
        value.memory_time_ms = memory_time_ms;
        value.launch_time_ms = launch_time_ms;
        value.predicted_time_ms = predicted_time_ms;
        value.bottleneck = bottleneck;
        return value;
    }();
}

KernelWork gemm_work(std::uint64_t m, std::uint64_t k, std::uint64_t n,
                     double element_bytes, std::uint64_t weight_multiplier) {
    return gemm_work(m, k, n, element_bytes, element_bytes, weight_multiplier);
}

KernelWork gemm_work(std::uint64_t m, std::uint64_t k, std::uint64_t n,
                     double weight_element_bytes,
                     double activation_element_bytes,
                     std::uint64_t weight_multiplier) {
    require_finite_nonnegative(weight_element_bytes, "weight element bytes");
    require_finite_nonnegative(activation_element_bytes,
                               "activation element bytes");
    if (weight_element_bytes == 0.0 || activation_element_bytes == 0.0) {
        throw AnalyticalModelError("GEMM element bytes must be positive");
    }
    if (weight_multiplier == 0) {
        throw AnalyticalModelError("weight multiplier must be positive");
    }
    if (m == 0 || k == 0 || n == 0) {
        return [&]() {
            KernelWork value{};
            value.flops = 0.0;
            value.hbm_bytes = 0.0;
            return value;
        }();
    }

    const double resolved_m = static_cast<double>(m);
    const double resolved_k = static_cast<double>(k);
    const double resolved_n = static_cast<double>(n);
    const double multiplier = static_cast<double>(weight_multiplier);
    return [&]() {
        KernelWork value{};
        value.flops = 2.0 * resolved_m * resolved_k * resolved_n * multiplier;
        value.hbm_bytes =
            activation_element_bytes * resolved_m * resolved_k +
            weight_element_bytes * multiplier * resolved_k * resolved_n +
            activation_element_bytes * multiplier * resolved_m * resolved_n;
        return value;
    }();
}

KernelWork streaming_work(double elements_read, double elements_written,
                          double flops, double element_bytes) {
    require_finite_nonnegative(elements_read, "elements read");
    require_finite_nonnegative(elements_written, "elements written");
    require_finite_nonnegative(flops, "streaming FLOPs");
    require_finite_nonnegative(element_bytes, "element bytes");
    if (element_bytes == 0.0) {
        throw AnalyticalModelError("element bytes must be positive");
    }
    return [&]() {
        KernelWork value{};
        value.flops = flops;
        value.hbm_bytes = (elements_read + elements_written) * element_bytes;
        return value;
    }();
}

KernelWork
attention_context_work(const std::vector<AttentionRequestSlice> &requests,
                       std::uint64_t local_query_heads,
                       std::uint64_t local_kv_heads, std::uint64_t head_dim,
                       double activation_element_bytes,
                       double kv_cache_element_bytes) {
    if (local_query_heads == 0 || local_kv_heads == 0 || head_dim == 0) {
        throw AnalyticalModelError("attention dimensions must be positive");
    }
    require_finite_nonnegative(activation_element_bytes,
                               "activation element bytes");
    require_finite_nonnegative(kv_cache_element_bytes,
                               "KV cache element bytes");
    if (activation_element_bytes == 0.0 || kv_cache_element_bytes == 0.0) {
        throw AnalyticalModelError("attention element bytes must be positive");
    }

    double total_flops = 0.0;
    double activation_elements = 0.0;
    double kv_cache_elements = 0.0;
    for (const AttentionRequestSlice &request : requests) {
        const double query_tokens = static_cast<double>(request.query_tokens);
        const double past_context = static_cast<double>(request.past_context);
        const double average_visible_kv =
            past_context + (query_tokens + 1.0) / 2.0;
        total_flops += 4.0 * static_cast<double>(local_query_heads) *
                       static_cast<double>(head_dim) * query_tokens *
                       average_visible_kv;
        const double q_and_output = 2.0 * query_tokens *
                                    static_cast<double>(local_query_heads) *
                                    static_cast<double>(head_dim);
        const double new_kv_input = 2.0 * query_tokens *
                                    static_cast<double>(local_kv_heads) *
                                    static_cast<double>(head_dim);
        const double cached_kv_reads = 2.0 * past_context *
                                       static_cast<double>(local_kv_heads) *
                                       static_cast<double>(head_dim);
        activation_elements += q_and_output + new_kv_input;
        kv_cache_elements += cached_kv_reads;
    }
    return [&]() {
        KernelWork value{};
        value.flops = total_flops;
        value.hbm_bytes = activation_elements * activation_element_bytes +
                          kv_cache_elements * kv_cache_element_bytes;
        return value;
    }();
}

KernelWork
attention_context_work(const std::vector<AttentionRequestSlice> &requests,
                       std::uint64_t local_query_heads,
                       std::uint64_t local_kv_heads, std::uint64_t head_dim,
                       double element_bytes) {
    return attention_context_work(requests, local_query_heads, local_kv_heads,
                                  head_dim, element_bytes, element_bytes);
}

KernelWork mla_unabsorbed_attention_work(
    const std::vector<AttentionRequestSlice> &requests,
    std::uint64_t local_query_heads, std::uint64_t qk_nope_head_dim,
    std::uint64_t qk_rope_head_dim, std::uint64_t v_head_dim,
    double activation_element_bytes, double rope_cache_element_bytes) {
    if (local_query_heads == 0 || qk_nope_head_dim == 0 ||
        qk_rope_head_dim == 0 || v_head_dim == 0) {
        throw AnalyticalModelError("MLA attention dimensions must be positive");
    }
    require_finite_nonnegative(activation_element_bytes,
                               "activation element bytes");
    require_finite_nonnegative(rope_cache_element_bytes,
                               "RoPE cache element bytes");
    if (activation_element_bytes == 0.0 || rope_cache_element_bytes == 0.0) {
        throw AnalyticalModelError("attention element bytes must be positive");
    }

    const double heads = static_cast<double>(local_query_heads);
    const double qk_dim =
        static_cast<double>(qk_nope_head_dim + qk_rope_head_dim);
    const double expanded_kv_dim =
        static_cast<double>(qk_nope_head_dim + v_head_dim);
    const double value_dim = static_cast<double>(v_head_dim);
    const double rope_dim = static_cast<double>(qk_rope_head_dim);
    double total_flops = 0.0;
    double activation_bytes = 0.0;
    double cached_rope_bytes = 0.0;
    for (const AttentionRequestSlice &request : requests) {
        const double query_tokens = static_cast<double>(request.query_tokens);
        const double past_context = static_cast<double>(request.past_context);
        const double visible_kv = past_context + query_tokens;
        const double average_visible_kv =
            past_context + (query_tokens + 1.0) / 2.0;
        total_flops += 2.0 * heads * query_tokens * average_visible_kv *
                       (qk_dim + value_dim);

        // The compute-friendly path materializes expanded NoPE K and V in an
        // HBM workspace. The projection accounts for the workspace write;
        // attention accounts for the read here. RoPE remains unexpanded in
        // the persistent MLA cache and is broadcast across query heads.
        const double query_and_output =
            query_tokens * heads * (qk_dim + value_dim);
        const double expanded_kv = visible_kv * heads * expanded_kv_dim;
        const double new_rope = query_tokens * rope_dim;
        activation_bytes += (query_and_output + expanded_kv + new_rope) *
                            activation_element_bytes;
        cached_rope_bytes += past_context * rope_dim * rope_cache_element_bytes;
    }
    return KernelWork{total_flops, activation_bytes + cached_rope_bytes};
}

KernelWork mla_absorbed_attention_work(
    const std::vector<AttentionRequestSlice> &requests,
    std::uint64_t local_query_heads, std::uint64_t kv_lora_rank,
    std::uint64_t qk_rope_head_dim, double activation_element_bytes,
    double latent_cache_element_bytes, double rope_cache_element_bytes) {
    if (local_query_heads == 0 || kv_lora_rank == 0 || qk_rope_head_dim == 0) {
        throw AnalyticalModelError("MLA attention dimensions must be positive");
    }
    require_finite_nonnegative(activation_element_bytes,
                               "activation element bytes");
    require_finite_nonnegative(latent_cache_element_bytes,
                               "latent cache element bytes");
    require_finite_nonnegative(rope_cache_element_bytes,
                               "RoPE cache element bytes");
    if (activation_element_bytes == 0.0 || latent_cache_element_bytes == 0.0 ||
        rope_cache_element_bytes == 0.0) {
        throw AnalyticalModelError("attention element bytes must be positive");
    }

    const double heads = static_cast<double>(local_query_heads);
    const double latent_dim = static_cast<double>(kv_lora_rank);
    const double rope_dim = static_cast<double>(qk_rope_head_dim);
    double total_flops = 0.0;
    double activation_bytes = 0.0;
    double cache_bytes = 0.0;
    for (const AttentionRequestSlice &request : requests) {
        const double query_tokens = static_cast<double>(request.query_tokens);
        const double past_context = static_cast<double>(request.past_context);
        const double average_visible_kv =
            past_context + (query_tokens + 1.0) / 2.0;
        total_flops += 2.0 * heads * query_tokens * average_visible_kv *
                       (2.0 * latent_dim + rope_dim);

        const double query_and_output =
            query_tokens * heads * (2.0 * latent_dim + rope_dim);
        const double new_latent_and_rope =
            query_tokens * (latent_dim + rope_dim);
        activation_bytes +=
            (query_and_output + new_latent_and_rope) * activation_element_bytes;
        cache_bytes += past_context * (latent_dim * latent_cache_element_bytes +
                                       rope_dim * rope_cache_element_bytes);
    }
    return KernelWork{total_flops, activation_bytes + cache_bytes};
}

namespace {

struct DenseLayerContext {
    const DeviceCeilings &device;
    const AnalyticalConfig &config;
    const DenseModel &model;
    const DenseBatch &batch;
    Precision attention_weight_precision;
    Precision dense_weight_precision;
    double attention_weight_element_bytes;
    double attention_element_bytes;
    double dense_weight_element_bytes;
    double dense_element_bytes;
    double kv_cache_element_bytes;
    std::uint64_t local_query_heads;
    std::uint64_t local_kv_heads;
    std::uint64_t local_intermediate;
    std::uint64_t prefill_tokens;
    std::uint64_t decode_tokens;
    std::uint64_t prefill_past_context;
};

struct AttentionLayerWork {
    double pre_projection_ms = 0.0;
    double post_projection_ms = 0.0;
    double inter_norm_ms = 0.0;
    double wq_projection_ms = 0.0;
    double rope_elements = 0.0;
    KernelWork kv_cache_save{0.0, 0.0};
    KernelWork prefill_attention{0.0, 0.0};
    KernelWork decode_attention{0.0, 0.0};
};

std::uint64_t
sum_query_tokens(const std::vector<AttentionRequestSlice> &requests) {
    std::uint64_t total = 0;
    for (const AttentionRequestSlice &request : requests) {
        total += request.query_tokens;
    }
    return total;
}

std::uint64_t
sum_past_context(const std::vector<AttentionRequestSlice> &requests) {
    std::uint64_t total = 0;
    for (const AttentionRequestSlice &request : requests) {
        total += request.past_context;
    }
    return total;
}

void validate_dense_layer_inputs(const AnalyticalConfig &config,
                                 const DenseModel &model,
                                 const DenseBatch &batch) {
    if (model.hidden_size == 0 || model.intermediate_size == 0 ||
        model.num_query_heads == 0 || model.num_kv_heads == 0 ||
        model.head_dim == 0 || model.tensor_parallel_size == 0) {
        throw AnalyticalModelError("dense model dimensions must be positive");
    }
    if (config.small_gemm_token_threshold == 0) {
        throw AnalyticalModelError(
            "small GEMM token threshold must be positive");
    }
    if (sum_query_tokens(batch.prefill_requests) +
            sum_query_tokens(batch.decode_requests) !=
        batch.total_tokens) {
        throw AnalyticalModelError(
            "dense batch total_tokens does not match request slices");
    }
    if (model.use_mla && model.use_mfa) {
        throw AnalyticalModelError("MLA and MFA are mutually exclusive");
    }
    if (model.use_mla &&
        (model.kv_lora_rank == 0 || model.qk_nope_head_dim == 0 ||
         model.qk_rope_head_dim == 0 || model.qk_head_dim == 0 ||
         model.v_head_dim == 0 ||
         model.qk_head_dim !=
             model.qk_nope_head_dim + model.qk_rope_head_dim)) {
        throw AnalyticalModelError(
            "MLA requires consistent latent attention dimensions");
    }
    if (model.use_mfa && (model.share_q_dim == 0 || model.num_kv_heads != 1)) {
        throw AnalyticalModelError(
            "MFA requires share_q_dim and exactly one KV head");
    }
}

DenseLayerContext
make_dense_layer_context(const DeviceCeilings &device,
                         const AnalyticalConfig &config,
                         const DenseModel &model, const DenseBatch &batch,
                         const DenseOperatorPrecisions &precisions) {
    const Precision attention_weight_precision =
        precisions.attention_weight.value_or(precisions.attention);
    const Precision attention_activation_precision =
        precisions.attention_activation.value_or(precisions.attention);
    const Precision dense_weight_precision =
        precisions.dense_weight.value_or(precisions.dense);
    const Precision dense_activation_precision =
        precisions.dense_activation.value_or(precisions.dense);
    return DenseLayerContext{
        device,
        config,
        model,
        batch,
        attention_weight_precision,
        dense_weight_precision,
        bytes_per_element(attention_weight_precision),
        bytes_per_element(attention_activation_precision),
        bytes_per_element(dense_weight_precision),
        bytes_per_element(dense_activation_precision),
        bytes_per_element(precisions.kv_cache),
        dense_ceil_div(model.num_query_heads, model.tensor_parallel_size),
        dense_ceil_div(model.num_kv_heads, model.tensor_parallel_size),
        dense_ceil_div(model.intermediate_size, model.tensor_parallel_size),
        sum_query_tokens(batch.prefill_requests),
        sum_query_tokens(batch.decode_requests),
        sum_past_context(batch.prefill_requests),
    };
}

const Efficiency &gemm_efficiency_for(const DenseLayerContext &context,
                                      std::uint64_t rows) {
    return rows < context.config.small_gemm_token_threshold
               ? context.config.small_gemm
               : context.config.large_gemm;
}

double predict_attention_work_ms(const DenseLayerContext &context,
                                 const KernelWork &work,
                                 const Efficiency &efficiency) {
    return predict_dense_kernel_ms(
        context.device, context.attention_weight_precision, work, efficiency,
        context.config.kernel_launch_latency_us);
}

double predict_dense_work_ms(const DenseLayerContext &context,
                             const KernelWork &work,
                             const Efficiency &efficiency) {
    return predict_dense_kernel_ms(
        context.device, context.dense_weight_precision, work, efficiency,
        context.config.kernel_launch_latency_us);
}

KernelWork attention_gemm_work(const DenseLayerContext &context,
                               std::uint64_t m, std::uint64_t k,
                               std::uint64_t n, std::uint64_t multiplier = 1) {
    return gemm_work(m, k, n, context.attention_weight_element_bytes,
                     context.attention_element_bytes, multiplier);
}

KernelWork dense_gemm_work(const DenseLayerContext &context, std::uint64_t m,
                           std::uint64_t k, std::uint64_t n,
                           std::uint64_t multiplier = 1) {
    return gemm_work(m, k, n, context.dense_weight_element_bytes,
                     context.dense_element_bytes, multiplier);
}

KernelWork per_head_gemm_work(const DenseLayerContext &context,
                              std::uint64_t rows, std::uint64_t k,
                              std::uint64_t n) {
    if (rows == 0) {
        return KernelWork{0.0, 0.0};
    }
    const double resolved_rows = static_cast<double>(rows);
    const double resolved_heads =
        static_cast<double>(context.local_query_heads);
    const double resolved_k = static_cast<double>(k);
    const double resolved_n = static_cast<double>(n);
    return KernelWork{
        2.0 * resolved_rows * resolved_heads * resolved_k * resolved_n,
        context.attention_element_bytes * resolved_rows * resolved_heads *
                resolved_k +
            context.attention_weight_element_bytes * resolved_heads *
                resolved_k * resolved_n +
            context.attention_element_bytes * resolved_rows * resolved_heads *
                resolved_n,
    };
}

KernelWork mla_unabsorbed_kv_expansion_work(const DenseLayerContext &context,
                                            std::uint64_t expanded_kv_dim) {
    const std::uint64_t visible_tokens =
        context.prefill_tokens + context.prefill_past_context;
    if (visible_tokens == 0) {
        return KernelWork{0.0, 0.0};
    }
    const double visible = static_cast<double>(visible_tokens);
    const double current = static_cast<double>(context.prefill_tokens);
    const double cached = static_cast<double>(context.prefill_past_context);
    const double latent = static_cast<double>(context.model.kv_lora_rank);
    const double output = static_cast<double>(expanded_kv_dim);
    return KernelWork{
        2.0 * visible * latent * output,
        current * latent * context.attention_element_bytes +
            cached * latent * context.kv_cache_element_bytes +
            latent * output * context.attention_weight_element_bytes +
            visible * output * context.attention_element_bytes,
    };
}

AttentionLayerWork
predict_mla_attention_work(const DenseLayerContext &context) {
    constexpr double kMlaRopeCacheElementBytes = 2.0;
    const DenseModel &model = context.model;
    const std::uint64_t latent_and_rope_dim =
        model.kv_lora_rank + model.qk_rope_head_dim;
    const std::uint64_t expanded_kv_dim =
        context.local_query_heads * (model.qk_nope_head_dim + model.v_head_dim);
    const std::uint64_t prefill_visible_tokens =
        context.prefill_tokens + context.prefill_past_context;
    const double tokens = static_cast<double>(context.batch.total_tokens);
    AttentionLayerWork work{};

    if (model.q_lora_rank == 0) {
        work.pre_projection_ms = predict_attention_work_ms(
            context,
            attention_gemm_work(context, context.batch.total_tokens,
                                model.hidden_size,
                                context.local_query_heads * model.qk_head_dim),
            gemm_efficiency_for(context, context.batch.total_tokens));
    } else {
        work.pre_projection_ms =
            predict_attention_work_ms(
                context,
                attention_gemm_work(context, context.batch.total_tokens,
                                    model.hidden_size, model.q_lora_rank),
                gemm_efficiency_for(context, context.batch.total_tokens)) +
            predict_attention_work_ms(
                context,
                attention_gemm_work(
                    context, context.batch.total_tokens, model.q_lora_rank,
                    context.local_query_heads * model.qk_head_dim),
                gemm_efficiency_for(context, context.batch.total_tokens));
        work.inter_norm_ms += predict_attention_work_ms(
            context,
            streaming_work(tokens * static_cast<double>(model.q_lora_rank),
                           tokens * static_cast<double>(model.q_lora_rank),
                           5.0 * tokens *
                               static_cast<double>(model.q_lora_rank),
                           context.attention_element_bytes),
            context.config.streaming);
    }
    work.pre_projection_ms += predict_attention_work_ms(
        context,
        attention_gemm_work(context, context.batch.total_tokens,
                            model.hidden_size, latent_and_rope_dim),
        gemm_efficiency_for(context, context.batch.total_tokens));
    work.inter_norm_ms += predict_attention_work_ms(
        context,
        streaming_work(tokens * static_cast<double>(model.kv_lora_rank),
                       tokens * static_cast<double>(model.kv_lora_rank),
                       5.0 * tokens * static_cast<double>(model.kv_lora_rank),
                       context.attention_element_bytes),
        context.config.streaming);
    work.pre_projection_ms += predict_attention_work_ms(
        context, mla_unabsorbed_kv_expansion_work(context, expanded_kv_dim),
        gemm_efficiency_for(context, prefill_visible_tokens));
    work.pre_projection_ms += predict_attention_work_ms(
        context,
        per_head_gemm_work(context, context.decode_tokens,
                           model.qk_nope_head_dim, model.kv_lora_rank),
        gemm_efficiency_for(context, context.decode_tokens));
    work.post_projection_ms += predict_attention_work_ms(
        context,
        attention_gemm_work(context, context.prefill_tokens,
                            context.local_query_heads * model.v_head_dim,
                            model.hidden_size),
        gemm_efficiency_for(context, context.prefill_tokens));
    work.post_projection_ms += predict_attention_work_ms(
        context,
        per_head_gemm_work(context, context.decode_tokens, model.kv_lora_rank,
                           model.v_head_dim),
        gemm_efficiency_for(context, context.decode_tokens));
    work.post_projection_ms += predict_attention_work_ms(
        context,
        attention_gemm_work(context, context.decode_tokens,
                            context.local_query_heads * model.v_head_dim,
                            model.hidden_size),
        gemm_efficiency_for(context, context.decode_tokens));

    work.rope_elements =
        tokens * (static_cast<double>(context.local_query_heads) + 1.0) *
        static_cast<double>(model.qk_rope_head_dim);
    const double mla_cache_bytes_per_token =
        attention::mla_kv_cache_bytes_per_token(attention::MlaKvCacheLayout{
            model.kv_lora_rank,
            model.qk_rope_head_dim,
            context.kv_cache_element_bytes,
            kMlaRopeCacheElementBytes,
        });
    work.kv_cache_save = KernelWork{
        0.0,
        tokens * (static_cast<double>(latent_and_rope_dim) *
                      context.attention_element_bytes +
                  mla_cache_bytes_per_token),
    };
    work.prefill_attention = mla_unabsorbed_attention_work(
        context.batch.prefill_requests, context.local_query_heads,
        model.qk_nope_head_dim, model.qk_rope_head_dim, model.v_head_dim,
        context.attention_element_bytes, kMlaRopeCacheElementBytes);
    work.decode_attention = mla_absorbed_attention_work(
        context.batch.decode_requests, context.local_query_heads,
        model.kv_lora_rank, model.qk_rope_head_dim,
        context.attention_element_bytes, context.kv_cache_element_bytes,
        kMlaRopeCacheElementBytes);
    return work;
}

AttentionLayerWork
predict_mfa_attention_work(const DenseLayerContext &context) {
    const DenseModel &model = context.model;
    const double tokens = static_cast<double>(context.batch.total_tokens);
    const double query_heads = static_cast<double>(context.local_query_heads);
    const double kv_heads = static_cast<double>(context.local_kv_heads);
    const double head_dim = static_cast<double>(model.head_dim);
    const std::uint64_t replicated_qkv_dim =
        model.share_q_dim + 2 * context.local_kv_heads * model.head_dim;
    AttentionLayerWork work{};
    work.pre_projection_ms = predict_attention_work_ms(
        context,
        attention_gemm_work(context, context.batch.total_tokens,
                            model.hidden_size, replicated_qkv_dim),
        gemm_efficiency_for(context, context.batch.total_tokens));
    work.inter_norm_ms = predict_attention_work_ms(
        context,
        streaming_work(tokens * static_cast<double>(model.share_q_dim),
                       tokens * static_cast<double>(model.share_q_dim),
                       5.0 * tokens * static_cast<double>(model.share_q_dim),
                       context.attention_element_bytes),
        context.config.streaming);
    work.wq_projection_ms = predict_attention_work_ms(
        context,
        attention_gemm_work(context, context.batch.total_tokens,
                            model.share_q_dim,
                            context.local_query_heads * model.head_dim),
        gemm_efficiency_for(context, context.batch.total_tokens));
    work.post_projection_ms = predict_attention_work_ms(
        context,
        attention_gemm_work(context, context.batch.total_tokens,
                            context.local_query_heads * model.head_dim,
                            model.hidden_size),
        gemm_efficiency_for(context, context.batch.total_tokens));
    work.rope_elements = tokens * (query_heads + kv_heads) * head_dim;
    const double kv_elements = tokens * 2.0 * kv_heads * head_dim;
    work.kv_cache_save = KernelWork{
        0.0,
        kv_elements *
            (context.attention_element_bytes + context.kv_cache_element_bytes),
    };
    work.prefill_attention = attention_context_work(
        context.batch.prefill_requests, context.local_query_heads,
        context.local_kv_heads, model.head_dim, context.attention_element_bytes,
        context.kv_cache_element_bytes);
    work.decode_attention = attention_context_work(
        context.batch.decode_requests, context.local_query_heads,
        context.local_kv_heads, model.head_dim, context.attention_element_bytes,
        context.kv_cache_element_bytes);
    return work;
}

AttentionLayerWork
predict_mha_attention_work(const DenseLayerContext &context) {
    const DenseModel &model = context.model;
    const double tokens = static_cast<double>(context.batch.total_tokens);
    const double query_heads = static_cast<double>(context.local_query_heads);
    const double kv_heads = static_cast<double>(context.local_kv_heads);
    const double head_dim = static_cast<double>(model.head_dim);
    const std::uint64_t local_qkv_dim =
        (context.local_query_heads + 2 * context.local_kv_heads) *
        model.head_dim;
    AttentionLayerWork work{};
    work.pre_projection_ms = predict_attention_work_ms(
        context,
        attention_gemm_work(context, context.batch.total_tokens,
                            model.hidden_size, local_qkv_dim),
        gemm_efficiency_for(context, context.batch.total_tokens));
    work.post_projection_ms = predict_attention_work_ms(
        context,
        attention_gemm_work(
            context, context.batch.total_tokens,
            std::max<std::uint64_t>(1, model.hidden_size /
                                           model.tensor_parallel_size),
            model.hidden_size),
        gemm_efficiency_for(context, context.batch.total_tokens));
    work.rope_elements = tokens * (query_heads + kv_heads) * head_dim;
    const double kv_elements = tokens * 2.0 * kv_heads * head_dim;
    work.kv_cache_save = KernelWork{
        0.0,
        kv_elements *
            (context.attention_element_bytes + context.kv_cache_element_bytes),
    };
    work.prefill_attention = attention_context_work(
        context.batch.prefill_requests, context.local_query_heads,
        context.local_kv_heads, model.head_dim, context.attention_element_bytes,
        context.kv_cache_element_bytes);
    work.decode_attention = attention_context_work(
        context.batch.decode_requests, context.local_query_heads,
        context.local_kv_heads, model.head_dim, context.attention_element_bytes,
        context.kv_cache_element_bytes);
    return work;
}

AttentionLayerWork predict_attention_work(const DenseLayerContext &context) {
    if (context.model.use_mla) {
        return predict_mla_attention_work(context);
    }
    if (context.model.use_mfa) {
        return predict_mfa_attention_work(context);
    }
    return predict_mha_attention_work(context);
}

void populate_attention_times(const DenseLayerContext &context,
                              const AttentionLayerWork &work,
                              DenseLayerTimes &times) {
    const double tokens = static_cast<double>(context.batch.total_tokens);
    const double hidden = static_cast<double>(context.model.hidden_size);
    const double norm_factor = context.model.fused_add_norm ? 3.0 : 2.0;
    times.attention_pre_projection_ms = work.pre_projection_ms;
    times.attention_post_projection_ms = work.post_projection_ms;
    times.rope_ms = predict_attention_work_ms(
        context,
        streaming_work(work.rope_elements, work.rope_elements,
                       6.0 * work.rope_elements,
                       context.attention_element_bytes),
        context.config.streaming);
    times.kv_cache_save_ms = predict_attention_work_ms(
        context, work.kv_cache_save, context.config.streaming);
    times.attention_norm_ms = predict_attention_work_ms(
        context,
        streaming_work(tokens * hidden * (norm_factor - 1.0), tokens * hidden,
                       5.0 * tokens * hidden, context.attention_element_bytes),
        context.config.streaming);
    times.attention_inter_norm_ms = work.inter_norm_ms;
    times.attention_wq_projection_ms = work.wq_projection_ms;
    times.prefill_attention_ms = predict_attention_work_ms(
        context, work.prefill_attention, context.config.prefill_attention);
    times.decode_attention_ms = predict_attention_work_ms(
        context, work.decode_attention, context.config.decode_attention);
}

void populate_dense_mlp_and_norm_times(const DenseLayerContext &context,
                                       DenseLayerTimes &times) {
    const double tokens = static_cast<double>(context.batch.total_tokens);
    const double hidden = static_cast<double>(context.model.hidden_size);
    const double intermediate = static_cast<double>(context.local_intermediate);
    const double activation_elements = tokens * intermediate;
    const std::uint64_t gated_multiplier = context.model.gated_mlp ? 2 : 1;
    const double norm_factor = context.model.fused_add_norm ? 3.0 : 2.0;
    const Efficiency &gemm_efficiency =
        gemm_efficiency_for(context, context.batch.total_tokens);
    times.mlp_up_projection_ms = predict_dense_work_ms(
        context,
        dense_gemm_work(context, context.batch.total_tokens,
                        context.model.hidden_size, context.local_intermediate,
                        gated_multiplier),
        gemm_efficiency);
    times.mlp_activation_ms = predict_dense_work_ms(
        context,
        streaming_work(activation_elements *
                           static_cast<double>(gated_multiplier),
                       activation_elements, 8.0 * activation_elements,
                       context.dense_element_bytes),
        context.config.streaming);
    times.mlp_down_projection_ms = predict_dense_work_ms(
        context,
        dense_gemm_work(context, context.batch.total_tokens,
                        context.local_intermediate, context.model.hidden_size),
        gemm_efficiency);
    times.mlp_norm_ms = predict_dense_work_ms(
        context,
        streaming_work(tokens * hidden * (norm_factor - 1.0), tokens * hidden,
                       5.0 * tokens * hidden, context.dense_element_bytes),
        context.config.streaming);
    if (!context.model.fused_add_norm) {
        const double residual_elements = tokens * hidden;
        times.residual_add_ms = predict_dense_work_ms(
            context,
            streaming_work(2.0 * residual_elements, residual_elements,
                           residual_elements, context.dense_element_bytes),
            context.config.streaming);
    }
}

} // namespace

DenseLayerTimes predict_dense_layer(const DeviceCeilings &device,
                                    const AnalyticalConfig &config,
                                    const DenseModel &model,
                                    const DenseBatch &batch,
                                    const DenseOperatorPrecisions &precisions) {
    validate_dense_layer_inputs(config, model, batch);
    const DenseLayerContext context =
        make_dense_layer_context(device, config, model, batch, precisions);
    const AttentionLayerWork attention = predict_attention_work(context);
    DenseLayerTimes times{};
    populate_attention_times(context, attention, times);
    populate_dense_mlp_and_norm_times(context, times);
    return times;
}

DenseLayerTimes predict_dense_layer(const DeviceCeilings &device,
                                    const AnalyticalConfig &config,
                                    const DenseModel &model,
                                    const DenseBatch &batch,
                                    Precision precision) {
    return predict_dense_layer(
        device, config, model, batch,
        DenseOperatorPrecisions{precision, precision, precision});
}

} // namespace frontier::execution_time_predictor::detail

namespace frontier::execution_time_predictor::detail {

ExpertParallelDomain::ExpertParallelDomain(std::uint64_t total_experts,
                                           std::uint64_t expert_parallel_size)
    : total_experts_(total_experts),
      expert_parallel_size_(expert_parallel_size) {
    if (total_experts_ == 0 || expert_parallel_size_ == 0 ||
        total_experts_ % expert_parallel_size_ != 0) {
        throw ParallelDomainError(
            "expert count must be positive and divisible by EP size");
    }
}

ExpertRange ExpertParallelDomain::expert_range(std::uint64_t lane) const {
    if (lane >= expert_parallel_size_) {
        throw ParallelDomainError("EP lane is outside the parallel domain");
    }
    const std::uint64_t width = experts_per_lane();
    return ExpertRange{lane * width, (lane + 1) * width};
}

std::uint64_t ExpertParallelDomain::owner(std::uint64_t expert_id) const {
    if (expert_id >= total_experts_) {
        throw ParallelDomainError("expert ID is outside the parallel domain");
    }
    return expert_id / experts_per_lane();
}

std::vector<std::vector<std::uint64_t>> ExpertParallelDomain::partition(
    const std::vector<std::uint64_t> &global_counts) const {
    if (global_counts.size() != static_cast<std::size_t>(total_experts_)) {
        throw ParallelDomainError(
            "global expert allocation has the wrong size");
    }

    std::vector<std::vector<std::uint64_t>> lanes;
    lanes.reserve(static_cast<std::size_t>(expert_parallel_size_));
    for (std::uint64_t lane = 0; lane < expert_parallel_size_; ++lane) {
        const ExpertRange range = expert_range(lane);
        lanes.emplace_back(
            global_counts.begin() + static_cast<std::ptrdiff_t>(range.begin),
            global_counts.begin() + static_cast<std::ptrdiff_t>(range.end));
    }
    return lanes;
}

} // namespace frontier::execution_time_predictor::detail

namespace frontier::execution_time_predictor::detail {
namespace {

std::uint64_t ceil_div(std::uint64_t numerator, std::uint64_t denominator) {
    if (denominator == 0) {
        throw AnalyticalModelError("MoE TP size must be positive");
    }
    return numerator / denominator +
           static_cast<std::uint64_t>(numerator % denominator != 0);
}

double predict_ms(const DeviceCeilings &device, Precision precision,
                  const KernelWork &work, const Efficiency &efficiency,
                  double launch_latency_us) {
    return predict_roofline(device, precision, work, efficiency,
                            launch_latency_us)
        .predicted_time_ms;
}

std::uint64_t payload_bytes(std::uint64_t tokens, std::uint64_t hidden_size,
                            double element_bytes) {
    const long double bytes = static_cast<long double>(tokens) *
                              static_cast<long double>(hidden_size) *
                              static_cast<long double>(element_bytes);
    if (!std::isfinite(static_cast<double>(bytes)) ||
        bytes > static_cast<long double>(
                    std::numeric_limits<std::uint64_t>::max())) {
        throw AnalyticalModelError(
            "MoE communication payload overflows uint64");
    }
    return static_cast<std::uint64_t>(std::ceil(bytes));
}

struct MoELayerContext {
    const DeviceCeilings &device;
    const AnalyticalConfig &config;
    const MoEModel &model;
    std::uint64_t input_tokens;
    std::uint64_t router_topk;
    Precision expert_weight_precision;
    Precision router_weight_precision;
    Precision dense_weight_precision;
    double expert_weight_element_bytes;
    double expert_element_bytes;
    double router_weight_element_bytes;
    double router_element_bytes;
    double dense_element_bytes;
    std::uint64_t local_intermediate;
};

struct ExpertGemmWork {
    KernelWork up;
    KernelWork down;
    std::uint64_t routed_tokens = 0;
};

void validate_moe_layer_inputs(const MoEModel &model,
                               std::uint64_t router_topk) {
    if (model.hidden_size == 0 || model.intermediate_size == 0 ||
        model.model_num_experts == 0 || model.moe_tensor_parallel_size == 0 ||
        router_topk == 0) {
        throw AnalyticalModelError("MoE model dimensions must be positive");
    }
}

MoELayerContext
make_moe_layer_context(const DeviceCeilings &device,
                       const AnalyticalConfig &config, const MoEModel &model,
                       std::uint64_t input_tokens, std::uint64_t router_topk,
                       const MoEOperatorPrecisions &precisions) {
    const Precision expert_weight_precision =
        precisions.expert_weight.value_or(precisions.expert);
    const Precision expert_activation_precision =
        precisions.expert_activation.value_or(precisions.expert);
    const Precision router_weight_precision =
        precisions.router_weight.value_or(precisions.router);
    const Precision router_activation_precision =
        precisions.router_activation.value_or(precisions.router);
    const Precision dense_weight_precision =
        precisions.dense_weight.value_or(precisions.dense);
    const Precision dense_activation_precision =
        precisions.dense_activation.value_or(precisions.dense);
    return MoELayerContext{
        device,
        config,
        model,
        input_tokens,
        router_topk,
        expert_weight_precision,
        router_weight_precision,
        dense_weight_precision,
        bytes_per_element(expert_weight_precision),
        bytes_per_element(expert_activation_precision),
        bytes_per_element(router_weight_precision),
        bytes_per_element(router_activation_precision),
        bytes_per_element(dense_activation_precision),
        ceil_div(model.intermediate_size, model.moe_tensor_parallel_size),
    };
}

double predict_expert_work_ms(const MoELayerContext &context,
                              const KernelWork &work,
                              const Efficiency &efficiency) {
    return predict_ms(context.device, context.expert_weight_precision, work,
                      efficiency, context.config.kernel_launch_latency_us);
}

double predict_router_work_ms(const MoELayerContext &context,
                              const KernelWork &work,
                              const Efficiency &efficiency) {
    return predict_ms(context.device, context.router_weight_precision, work,
                      efficiency, context.config.kernel_launch_latency_us);
}

double predict_dense_work_ms(const MoELayerContext &context,
                             const KernelWork &work,
                             const Efficiency &efficiency) {
    return predict_ms(context.device, context.dense_weight_precision, work,
                      efficiency, context.config.kernel_launch_latency_us);
}

void add_kernel_work(KernelWork &target, const KernelWork &source) {
    target.flops += source.flops;
    target.hbm_bytes += source.hbm_bytes;
}

void add_expert_gemm_work(ExpertGemmWork &work, const MoELayerContext &context,
                          std::uint64_t tokens, bool count_as_routed) {
    if (tokens == 0) {
        return;
    }
    if (count_as_routed) {
        work.routed_tokens += tokens;
    }
    add_kernel_work(work.up, gemm_work(tokens, context.model.hidden_size,
                                       context.local_intermediate,
                                       context.expert_weight_element_bytes,
                                       context.expert_element_bytes,
                                       context.model.gated_mlp ? 2 : 1));
    add_kernel_work(work.down, gemm_work(tokens, context.local_intermediate,
                                         context.model.hidden_size,
                                         context.expert_weight_element_bytes,
                                         context.expert_element_bytes, 1));
}

ExpertGemmWork
build_expert_gemm_work(const MoELayerContext &context,
                       const std::vector<std::uint64_t> &local_expert_tokens) {
    ExpertGemmWork result{};
    for (const std::uint64_t tokens : local_expert_tokens) {
        add_expert_gemm_work(result, context, tokens, true);
    }
    // Shared experts are replicated on every EP lane. Their weights are only
    // sharded across the MoE TP domain, so each lane processes every input
    // token through every shared expert.
    for (std::uint64_t expert = 0; expert < context.model.num_shared_experts;
         ++expert) {
        add_expert_gemm_work(result, context, context.input_tokens, false);
    }
    return result;
}

const Efficiency &router_gemm_efficiency(const MoELayerContext &context) {
    return context.input_tokens < context.config.small_gemm_token_threshold
               ? context.config.small_gemm
               : context.config.large_gemm;
}

} // namespace

double MoELayerTime::total_ms() const noexcept {
    return gating_linear_ms + gating_routing_topk_ms +
           grouped_up_projection_ms + grouped_down_projection_ms +
           shuffling_ms + post_attention_norm_ms;
}

double MoECommunicationTime::total_ms() const noexcept {
    return attention_tp_ms + moe_tp_ms + ep_dispatch_ms + ep_combine_ms +
           dp_input_ms + dp_output_ms + pipeline_parallel_ms;
}

MoELayerTime
predict_moe_layer(const DeviceCeilings &device, const AnalyticalConfig &config,
                  const MoEModel &model, std::uint64_t input_tokens,
                  std::uint64_t router_topk,
                  const std::vector<std::uint64_t> &local_expert_tokens,
                  const MoEOperatorPrecisions &precisions) {
    validate_moe_layer_inputs(model, router_topk);
    const MoELayerContext context = make_moe_layer_context(
        device, config, model, input_tokens, router_topk, precisions);
    const ExpertGemmWork expert_work =
        build_expert_gemm_work(context, local_expert_tokens);
    const double tokens = static_cast<double>(context.input_tokens);
    const double hidden = static_cast<double>(context.model.hidden_size);
    const double experts = static_cast<double>(context.model.model_num_experts);
    const double routed = static_cast<double>(expert_work.routed_tokens);
    const double norm_factor = context.model.fused_add_norm ? 3.0 : 2.0;

    MoELayerTime result{};
    result.gating_linear_ms = predict_router_work_ms(
        context,
        gemm_work(context.input_tokens, context.model.hidden_size,
                  context.model.model_num_experts,
                  context.router_weight_element_bytes,
                  context.router_element_bytes, 1),
        router_gemm_efficiency(context));
    result.gating_routing_topk_ms = predict_router_work_ms(
        context,
        streaming_work(tokens * experts,
                       tokens * static_cast<double>(context.router_topk),
                       4.0 * tokens * experts, context.router_element_bytes),
        context.config.routing);
    result.grouped_up_projection_ms =
        predict_expert_work_ms(context, expert_work.up, context.config.moe);
    result.grouped_down_projection_ms =
        predict_expert_work_ms(context, expert_work.down, context.config.moe);
    result.shuffling_ms = predict_expert_work_ms(
        context,
        streaming_work(routed * hidden, routed * hidden, 0.0,
                       context.expert_element_bytes),
        context.config.streaming);
    result.post_attention_norm_ms = predict_dense_work_ms(
        context,
        streaming_work(tokens * hidden * (norm_factor - 1.0), tokens * hidden,
                       5.0 * tokens * hidden, context.dense_element_bytes),
        context.config.streaming);
    return result;
}

MoELayerTime
predict_moe_layer(const DeviceCeilings &device, const AnalyticalConfig &config,
                  const MoEModel &model, std::uint64_t input_tokens,
                  std::uint64_t router_topk,
                  const std::vector<std::uint64_t> &local_expert_tokens,
                  Precision precision) {
    return predict_moe_layer(
        device, config, model, input_tokens, router_topk, local_expert_tokens,
        MoEOperatorPrecisions{precision, precision, precision});
}

MoELanePrediction predict_moe_lanes(const DeviceCeilings &device,
                                    const AnalyticalConfig &config,
                                    const MoEModel &model,
                                    const RoutingAllocation &routing,
                                    std::uint64_t router_topk,
                                    const MoEOperatorPrecisions &precisions) {
    if (routing.lane_expert_tokens.empty()) {
        throw AnalyticalModelError(
            "MoE routing must contain at least one lane");
    }
    MoELanePrediction prediction;
    prediction.lane_times.reserve(routing.lane_expert_tokens.size());
    for (const auto &lane : routing.lane_expert_tokens) {
        prediction.lane_times.push_back(
            predict_moe_layer(device, config, model, routing.input_tokens,
                              router_topk, lane, precisions));
    }
    for (std::size_t lane = 0; lane < prediction.lane_times.size(); ++lane) {
        const double time = prediction.lane_times[lane].total_ms();
        if (lane == 0 || time > prediction.critical_lane_time_ms) {
            prediction.critical_lane = static_cast<std::uint64_t>(lane);
            prediction.critical_lane_time_ms = time;
        }
    }
    return prediction;
}

MoELanePrediction
predict_moe_lanes(const DeviceCeilings &device, const AnalyticalConfig &config,
                  const MoEModel &model, const RoutingAllocation &routing,
                  std::uint64_t router_topk, Precision precision) {
    return predict_moe_lanes(
        device, config, model, routing, router_topk,
        MoEOperatorPrecisions{precision, precision, precision});
}

double predict_output_projection_ms(
    const DeviceCeilings &device, const AnalyticalConfig &config,
    std::uint64_t tokens, std::uint64_t hidden_size, std::uint64_t vocab_size,
    std::uint64_t tensor_parallel_size, Precision weight_precision,
    Precision activation_precision) {
    if (hidden_size == 0 || vocab_size == 0 || tensor_parallel_size == 0) {
        throw AnalyticalModelError(
            "output projection dimensions must be positive");
    }
    const std::uint64_t local_vocab =
        ceil_div(vocab_size, tensor_parallel_size);
    const KernelWork work = gemm_work(
        tokens, hidden_size, local_vocab, bytes_per_element(weight_precision),
        bytes_per_element(activation_precision), 1);
    const Efficiency &efficiency = tokens < config.small_gemm_token_threshold
                                       ? config.small_gemm
                                       : config.large_gemm;
    return predict_ms(device, weight_precision, work, efficiency,
                      config.kernel_launch_latency_us);
}

MoECommunicationTime predict_moe_communication(
    const cc_backend::BaseCCBackend &communication, std::uint64_t input_tokens,
    std::uint64_t hidden_size, std::uint64_t routed_tokens,
    std::uint64_t attention_tp_size, std::uint64_t moe_tp_size,
    std::uint64_t expert_parallel_size, std::uint64_t data_parallel_size,
    bool has_pipeline_boundary, double element_bytes) {
    const std::uint64_t activation_bytes =
        payload_bytes(input_tokens, hidden_size, element_bytes);
    const std::uint64_t routed_bytes =
        payload_bytes(routed_tokens, hidden_size, element_bytes);
    return [&]() {
        MoECommunicationTime value{};
        value.attention_tp_ms =
            communication.allreduce_ms(activation_bytes, attention_tp_size);
        value.moe_tp_ms =
            communication.allreduce_ms(activation_bytes, moe_tp_size);
        value.ep_dispatch_ms =
            communication.all_to_all_ms(routed_bytes, expert_parallel_size);
        value.ep_combine_ms =
            communication.all_to_all_ms(routed_bytes, expert_parallel_size);
        value.dp_input_ms =
            communication.allreduce_ms(activation_bytes, data_parallel_size);
        value.dp_output_ms =
            communication.allreduce_ms(activation_bytes, data_parallel_size);
        value.pipeline_parallel_ms =
            has_pipeline_boundary
                ? communication.point_to_point_ms(activation_bytes)
                : 0.0;
        return value;
    }();
}

} // namespace frontier::execution_time_predictor::detail

namespace frontier::execution_time_predictor::detail {
namespace {

constexpr std::uint32_t kInitA = 0x43b0d7e5U;
constexpr std::uint32_t kMultA = 0x931e8875U;
constexpr std::uint32_t kInitB = 0x8b51f9ddU;
constexpr std::uint32_t kMultB = 0x58f38dedU;
constexpr std::uint32_t kMixMultL = 0xca01f9ddU;
constexpr std::uint32_t kMixMultR = 0x4973f715U;

constexpr std::uint64_t rotate_right(std::uint64_t value,
                                     unsigned int shift) noexcept {
    shift &= 63U;
    return shift == 0U ? value : (value >> shift) | (value << (64U - shift));
}

std::uint32_t hashmix(std::uint32_t value, std::uint32_t &hash_constant) {
    value ^= hash_constant;
    hash_constant *= kMultA;
    value *= hash_constant;
    value ^= value >> 16U;
    return value;
}

std::uint32_t mix(std::uint32_t x, std::uint32_t y) {
    std::uint32_t result = kMixMultL * x - kMixMultR * y;
    result ^= result >> 16U;
    return result;
}

std::array<std::uint64_t, 4> seed_sequence_state(std::uint64_t seed) {
    std::vector<std::uint32_t> entropy;
    entropy.push_back(static_cast<std::uint32_t>(seed));
    if (seed > std::numeric_limits<std::uint32_t>::max()) {
        entropy.push_back(static_cast<std::uint32_t>(seed >> 32U));
    }

    std::array<std::uint32_t, 4> pool{};
    std::uint32_t hash_constant = kInitA;
    for (std::size_t i = 0; i < pool.size(); ++i) {
        pool[i] = hashmix(i < entropy.size() ? entropy[i] : 0U, hash_constant);
    }
    for (std::size_t source = 0; source < pool.size(); ++source) {
        for (std::size_t destination = 0; destination < pool.size();
             ++destination) {
            if (source != destination) {
                pool[destination] = mix(pool[destination],
                                        hashmix(pool[source], hash_constant));
            }
        }
    }
    for (std::size_t source = pool.size(); source < entropy.size(); ++source) {
        for (std::size_t destination = 0; destination < pool.size();
             ++destination) {
            pool[destination] =
                mix(pool[destination], hashmix(entropy[source], hash_constant));
        }
    }

    std::array<std::uint32_t, 8> words{};
    hash_constant = kInitB;
    for (std::size_t i = 0; i < words.size(); ++i) {
        std::uint32_t value = pool[i % pool.size()];
        value ^= hash_constant;
        hash_constant *= kMultB;
        value *= hash_constant;
        value ^= value >> 16U;
        words[i] = value;
    }

    std::array<std::uint64_t, 4> result{};
    for (std::size_t i = 0; i < result.size(); ++i) {
        result[i] = static_cast<std::uint64_t>(words[i * 2]) |
                    (static_cast<std::uint64_t>(words[i * 2 + 1]) << 32U);
    }
    return result;
}

struct Uint128 {
    std::uint64_t high = 0;
    std::uint64_t low = 0;
};

Uint128 add128(Uint128 lhs, Uint128 rhs) {
    const std::uint64_t low = lhs.low + rhs.low;
    return Uint128{
        lhs.high + rhs.high + static_cast<std::uint64_t>(low < rhs.low), low};
}

std::pair<std::uint64_t, std::uint64_t> multiply64(std::uint64_t lhs,
                                                   std::uint64_t rhs) {
    const std::uint64_t lhs_low = lhs & 0xffffffffULL;
    const std::uint64_t lhs_high = lhs >> 32U;
    const std::uint64_t rhs_low = rhs & 0xffffffffULL;
    const std::uint64_t rhs_high = rhs >> 32U;
    const std::uint64_t word0 = lhs_low * rhs_low;
    const std::uint64_t temp = lhs_high * rhs_low + (word0 >> 32U);
    std::uint64_t word1 = temp & 0xffffffffULL;
    const std::uint64_t word2 = temp >> 32U;
    word1 += lhs_low * rhs_high;
    const std::uint64_t high = lhs_high * rhs_high + word2 + (word1 >> 32U);
    return {high, lhs * rhs};
}

Uint128 multiply128(Uint128 lhs, Uint128 rhs) {
    auto [high, low] = multiply64(lhs.low, rhs.low);
    high += lhs.high * rhs.low + lhs.low * rhs.high;
    return Uint128{high, low};
}

class NumpyPcg64 {
  public:
    explicit NumpyPcg64(std::uint64_t seed) {
        const auto generated = seed_sequence_state(seed);
        seed_state(Uint128{generated[0], generated[1]},
                   Uint128{generated[2], generated[3]});
    }

    std::uint64_t next64() {
        step();
        return rotate_right(state_.high ^ state_.low,
                            static_cast<unsigned int>(state_.high >> 58U));
    }

    std::uint32_t next32() {
        if (has_uint32_) {
            has_uint32_ = false;
            return buffered_uint32_;
        }
        const std::uint64_t value = next64();
        has_uint32_ = true;
        buffered_uint32_ = static_cast<std::uint32_t>(value >> 32U);
        return static_cast<std::uint32_t>(value);
    }

    std::uint32_t bounded32(std::uint32_t exclusive_upper) {
        if (exclusive_upper == 0) {
            throw RoutingError("bounded PCG64 range must be positive");
        }
        const std::uint32_t inclusive_range = exclusive_upper - 1U;
        if (inclusive_range == 0) {
            return 0;
        }
        const std::uint64_t range = exclusive_upper;
        for (;;) {
            const std::uint64_t product =
                static_cast<std::uint64_t>(next32()) * range;
            const std::uint32_t leftover = static_cast<std::uint32_t>(product);
            if (leftover < range) {
                const std::uint32_t threshold =
                    (std::numeric_limits<std::uint32_t>::max() -
                     inclusive_range) %
                    exclusive_upper;
                if (leftover < threshold) {
                    continue;
                }
            }
            return static_cast<std::uint32_t>(product >> 32U);
        }
    }

    double next_double() {
        return static_cast<double>(next64() >> 11U) *
               (1.0 / 9007199254740992.0);
    }

  private:
    void step() {
        constexpr Uint128 multiplier{2549297995355413924ULL,
                                     4865540595714422341ULL};
        state_ = add128(multiply128(state_, multiplier), increment_);
    }

    void seed_state(Uint128 initial_state, Uint128 initial_sequence) {
        state_ = Uint128{};
        increment_.high =
            (initial_sequence.high << 1U) | (initial_sequence.low >> 63U);
        increment_.low = (initial_sequence.low << 1U) | 1U;
        step();
        state_ = add128(state_, initial_state);
        step();
    }

    Uint128 state_;
    Uint128 increment_;
    bool has_uint32_ = false;
    std::uint32_t buffered_uint32_ = 0;
};

std::vector<double>
distribution_weights(std::uint64_t experts,
                     config::MoeRoutingDistribution distribution,
                     std::uint64_t seed) {
    std::vector<double> weights(static_cast<std::size_t>(experts), 1.0);
    switch (distribution) {
    case config::MoeRoutingDistribution::kBalanced:
        break;
    case config::MoeRoutingDistribution::kRandom: {
        NumpyPcg64 generator(seed);
        for (double &weight : weights) {
            weight = 0.1 + 0.9 * generator.next_double();
        }
        break;
    }
    case config::MoeRoutingDistribution::kSkewed:
        for (std::size_t i = 0; i < weights.size(); ++i) {
            weights[i] = 1.0 / std::pow(static_cast<double>(i + 1), 0.35);
        }
        break;
    case config::MoeRoutingDistribution::kZipf:
        for (std::size_t i = 0; i < weights.size(); ++i) {
            weights[i] = 1.0 / static_cast<double>(i + 1);
        }
        break;
    }
    return weights;
}

} // namespace

std::vector<std::uint64_t>
discretize_expert_weights(std::uint64_t total_tokens,
                          const std::vector<double> &weights) {
    if (weights.empty()) {
        throw RoutingError("expert weights must not be empty");
    }
    double total_weight = 0.0;
    for (const double weight : weights) {
        if (!std::isfinite(weight) || weight < 0.0) {
            throw RoutingError("expert weights must be finite and nonnegative");
        }
        total_weight += weight;
    }
    if (!std::isfinite(total_weight) || total_weight <= 0.0) {
        throw RoutingError("expert weights must have a positive sum");
    }

    std::vector<std::uint64_t> result(weights.size(), 0);
    std::vector<double> normalized(weights.size(), 0.0);
    std::vector<double> fractional(weights.size(), 0.0);
    std::uint64_t allocated = 0;
    for (std::size_t i = 0; i < weights.size(); ++i) {
        normalized[i] = weights[i] / total_weight;
        const double exact = static_cast<double>(total_tokens) * normalized[i];
        result[i] = static_cast<std::uint64_t>(exact);
        fractional[i] = exact - static_cast<double>(result[i]);
        allocated += result[i];
    }

    std::vector<std::size_t> order(weights.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](std::size_t lhs, std::size_t rhs) {
                  if (fractional[lhs] != fractional[rhs]) {
                      return fractional[lhs] > fractional[rhs];
                  }
                  if (normalized[lhs] != normalized[rhs]) {
                      return normalized[lhs] > normalized[rhs];
                  }
                  return lhs < rhs;
              });
    std::uint64_t remainder_index = 0;
    while (allocated < total_tokens) {
        ++result[order[static_cast<std::size_t>(remainder_index %
                                                order.size())]];
        ++allocated;
        ++remainder_index;
    }
    return result;
}

RoutingAllocation
route_tokens(std::uint64_t input_tokens, std::uint64_t router_topk,
             std::uint64_t total_experts, std::uint64_t expert_parallel_size,
             const config::MoeRoutingConfig &config, std::uint64_t layer_id) {
    if (router_topk == 0 || router_topk > total_experts) {
        throw RoutingError("router top-k must be in [1, total experts]");
    }
    if (input_tokens >
        std::numeric_limits<std::uint64_t>::max() / router_topk) {
        throw RoutingError("routed token count overflows uint64");
    }

    ExpertParallelDomain domain(total_experts, expert_parallel_size);
    const std::uint64_t routed_tokens = input_tokens * router_topk;
    std::vector<std::uint64_t> counts(static_cast<std::size_t>(total_experts),
                                      0);

    if (config.mode == config::MoeRoutingMode::kUniformLegacy) {
        const std::uint64_t base = routed_tokens / total_experts;
        const std::uint64_t remainder = routed_tokens % total_experts;
        for (std::uint64_t expert = 0; expert < total_experts; ++expert) {
            counts[static_cast<std::size_t>(expert)] =
                base + static_cast<std::uint64_t>(expert < remainder);
        }
    } else if (config.mode == config::MoeRoutingMode::kUniformRandom) {
        NumpyPcg64 generator(config.seed + layer_id);
        for (std::uint64_t token = 0; token < routed_tokens; ++token) {
            const std::uint32_t expert =
                generator.bounded32(static_cast<std::uint32_t>(total_experts));
            ++counts[expert];
        }
    } else {
        counts = discretize_expert_weights(
            routed_tokens,
            distribution_weights(total_experts, config.distribution,
                                 config.seed + layer_id));
    }

    return [&]() {
        RoutingAllocation value{};
        value.input_tokens = input_tokens;
        value.routed_tokens = routed_tokens;
        value.global_expert_tokens = counts;
        value.lane_expert_tokens = domain.partition(counts);
        return value;
    }();
}

} // namespace frontier::execution_time_predictor::detail
