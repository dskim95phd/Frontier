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
    if (communication_backend_ == nullptr || config_.device != "rubin" ||
        (config_.precision != "fp16" && config_.precision != "bf16") ||
        model_.attention.memory_layout ==
            attention::AttentionMemoryLayout::kFrozenDsa ||
        parallelism_.tensor_parallel_size == 0 ||
        parallelism_.pipeline_parallel_size == 0 ||
        model_.num_layers % parallelism_.pipeline_parallel_size != 0) {
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
            value.use_mla = model_.use_mla;
            value.use_mfa = model_.use_mfa;
            value.q_lora_rank = model_.q_lora_rank;
            value.kv_lora_rank = model_.kv_lora_rank;
            value.qk_nope_head_dim = model_.qk_nope_head_dim;
            value.qk_rope_head_dim = model_.qk_rope_head_dim;
            value.qk_head_dim = model_.qk_head_dim;
            value.v_head_dim = model_.v_head_dim;
            value.share_q_dim = model_.share_q_dim;
            return value;
        }(),
        dense_batch,
        config_.precision == "bf16" ? detail::Precision::kBf16
                                    : detail::Precision::kFp16);

    if (dense_batch.total_tokens > std::numeric_limits<std::uint64_t>::max() /
                                       (model_.hidden_size * 2ULL)) {
        throw ExecutionTimePredictorError(
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
        layer.attention_inter_norm_ms + layer.attention_wq_projection_ms +
        layer.prefill_attention_ms + layer.decode_attention_ms;
    const double tp_layer_ms = 2.0 * allreduce_ms;
    const std::uint64_t layers_per_stage =
        model_.num_layers / parallelism_.pipeline_parallel_size;
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
            value.model_num_experts = model_.num_experts;
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
                model_.total_expert_num,
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
        throw ExecutionTimePredictorError(
            "analytical stage duration is invalid");
    }

    ExecutionTimePrediction result = [&]() {
        ExecutionTimePrediction value{};
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
    require_finite_nonnegative(element_bytes, "element bytes");
    if (element_bytes == 0.0) {
        throw AnalyticalModelError("element bytes must be positive");
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
            element_bytes *
            (resolved_m * resolved_k + multiplier * resolved_k * resolved_n +
             multiplier * resolved_m * resolved_n);
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
                       double element_bytes) {
    if (local_query_heads == 0 || local_kv_heads == 0 || head_dim == 0) {
        throw AnalyticalModelError("attention dimensions must be positive");
    }
    require_finite_nonnegative(element_bytes, "element bytes");
    if (element_bytes == 0.0) {
        throw AnalyticalModelError("element bytes must be positive");
    }

    double total_flops = 0.0;
    double total_elements = 0.0;
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
        total_elements += q_and_output + new_kv_input + cached_kv_reads;
    }
    return [&]() {
        KernelWork value{};
        value.flops = total_flops;
        value.hbm_bytes = total_elements * element_bytes;
        return value;
    }();
}

KernelWork mla_attention_context_work(
    const std::vector<AttentionRequestSlice> &requests,
    std::uint64_t local_query_heads, std::uint64_t qk_head_dim,
    std::uint64_t v_head_dim, std::uint64_t latent_dim,
    double element_bytes) {
    if (local_query_heads == 0 || qk_head_dim == 0 || v_head_dim == 0 ||
        latent_dim == 0) {
        throw AnalyticalModelError("MLA attention dimensions must be positive");
    }
    require_finite_nonnegative(element_bytes, "element bytes");
    if (element_bytes == 0.0) {
        throw AnalyticalModelError("element bytes must be positive");
    }

    double total_flops = 0.0;
    double total_elements = 0.0;
    for (const AttentionRequestSlice &request : requests) {
        const double query_tokens = static_cast<double>(request.query_tokens);
        const double past_context = static_cast<double>(request.past_context);
        const double average_visible_kv =
            past_context + (query_tokens + 1.0) / 2.0;
        const double query_and_value_dim =
            static_cast<double>(qk_head_dim + v_head_dim);
        total_flops += 2.0 * static_cast<double>(local_query_heads) *
                       query_tokens * average_visible_kv *
                       query_and_value_dim;
        total_elements +=
            query_tokens * static_cast<double>(local_query_heads) *
                query_and_value_dim +
            query_tokens * static_cast<double>(latent_dim) +
            past_context * static_cast<double>(latent_dim);
    }
    return KernelWork{total_flops, total_elements * element_bytes};
}

DenseLayerTimes predict_dense_layer(const DeviceCeilings &device,
                                    const AnalyticalConfig &config,
                                    const DenseModel &model,
                                    const DenseBatch &batch,
                                    Precision precision) {
    if (model.hidden_size == 0 || model.intermediate_size == 0 ||
        model.num_query_heads == 0 || model.num_kv_heads == 0 ||
        model.head_dim == 0 || model.tensor_parallel_size == 0) {
        throw AnalyticalModelError("dense model dimensions must be positive");
    }
    if (config.small_gemm_token_threshold == 0) {
        throw AnalyticalModelError(
            "small GEMM token threshold must be positive");
    }

    const std::uint64_t scheduled_tokens = [&batch] {
        std::uint64_t total = 0;
        for (const AttentionRequestSlice &request : batch.prefill_requests) {
            total += request.query_tokens;
        }
        for (const AttentionRequestSlice &request : batch.decode_requests) {
            total += request.query_tokens;
        }
        return total;
    }();
    if (scheduled_tokens != batch.total_tokens) {
        throw AnalyticalModelError(
            "dense batch total_tokens does not match request slices");
    }

    const double element_bytes = bytes_per_element(precision);
    const std::uint64_t local_query_heads =
        dense_ceil_div(model.num_query_heads, model.tensor_parallel_size);
    const std::uint64_t local_kv_heads =
        dense_ceil_div(model.num_kv_heads, model.tensor_parallel_size);
    const std::uint64_t local_intermediate =
        dense_ceil_div(model.intermediate_size, model.tensor_parallel_size);
    const std::uint64_t gated_multiplier = model.gated_mlp ? 2 : 1;
    const Efficiency &gemm_efficiency =
        batch.total_tokens < config.small_gemm_token_threshold
            ? config.small_gemm
            : config.large_gemm;

    const auto kernel_ms = [&](const KernelWork &work,
                               const Efficiency &efficiency) {
        return predict_dense_kernel_ms(device, precision, work, efficiency,
                                       config.kernel_launch_latency_us);
    };

    const double query_heads = static_cast<double>(local_query_heads);
    const double kv_heads = static_cast<double>(local_kv_heads);
    const double head_dim = static_cast<double>(model.head_dim);
    const double tokens = static_cast<double>(batch.total_tokens);
    const double hidden = static_cast<double>(model.hidden_size);
    const double intermediate = static_cast<double>(local_intermediate);

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
    if (model.use_mfa &&
        (model.share_q_dim == 0 || model.num_kv_heads != 1)) {
        throw AnalyticalModelError(
            "MFA requires share_q_dim and exactly one KV head");
    }

    double attention_pre_projection_ms = 0.0;
    double attention_post_projection_ms = 0.0;
    double attention_inter_norm_ms = 0.0;
    double attention_wq_projection_ms = 0.0;
    double rope_elements = 0.0;
    double kv_elements = 0.0;
    KernelWork prefill_attention{0.0, 0.0};
    KernelWork decode_attention{0.0, 0.0};

    if (model.use_mla) {
        const std::uint64_t q_lora_rank =
            model.q_lora_rank == 0 ? model.hidden_size : model.q_lora_rank;
        const std::uint64_t latent_dim =
            model.kv_lora_rank + model.qk_rope_head_dim;
        attention_pre_projection_ms =
            kernel_ms(gemm_work(batch.total_tokens, model.hidden_size,
                                q_lora_rank, element_bytes),
                      gemm_efficiency) +
            kernel_ms(gemm_work(batch.total_tokens, q_lora_rank,
                                local_query_heads * model.qk_head_dim,
                                element_bytes),
                      gemm_efficiency) +
            kernel_ms(gemm_work(batch.total_tokens, model.hidden_size,
                                latent_dim, element_bytes),
                      gemm_efficiency) +
            kernel_ms(gemm_work(
                          batch.total_tokens, model.kv_lora_rank,
                          local_query_heads *
                              (model.qk_nope_head_dim + model.v_head_dim),
                          element_bytes),
                      gemm_efficiency);
        attention_post_projection_ms = kernel_ms(
            gemm_work(batch.total_tokens,
                      local_query_heads * model.v_head_dim, model.hidden_size,
                      element_bytes),
            gemm_efficiency);
        rope_elements = tokens * (query_heads + 1.0) *
                        static_cast<double>(model.qk_rope_head_dim);
        kv_elements = tokens * static_cast<double>(latent_dim);
        prefill_attention = mla_attention_context_work(
            batch.prefill_requests, local_query_heads, model.qk_head_dim,
            model.v_head_dim, latent_dim, element_bytes);
        decode_attention = mla_attention_context_work(
            batch.decode_requests, local_query_heads, model.qk_head_dim,
            model.v_head_dim, latent_dim, element_bytes);
    } else if (model.use_mfa) {
        const std::uint64_t replicated_qkv_dim =
            model.share_q_dim + 2 * local_kv_heads * model.head_dim;
        attention_pre_projection_ms = kernel_ms(
            gemm_work(batch.total_tokens, model.hidden_size,
                      replicated_qkv_dim, element_bytes),
            gemm_efficiency);
        attention_inter_norm_ms = kernel_ms(
            streaming_work(tokens * static_cast<double>(model.share_q_dim),
                           tokens * static_cast<double>(model.share_q_dim),
                           5.0 * tokens *
                               static_cast<double>(model.share_q_dim),
                           element_bytes),
            config.streaming);
        attention_wq_projection_ms = kernel_ms(
            gemm_work(batch.total_tokens, model.share_q_dim,
                      local_query_heads * model.head_dim, element_bytes),
            gemm_efficiency);
        attention_post_projection_ms = kernel_ms(
            gemm_work(batch.total_tokens, local_query_heads * model.head_dim,
                      model.hidden_size, element_bytes),
            gemm_efficiency);
        rope_elements = tokens * (query_heads + kv_heads) * head_dim;
        kv_elements = tokens * 2.0 * kv_heads * head_dim;
        prefill_attention = attention_context_work(
            batch.prefill_requests, local_query_heads, local_kv_heads,
            model.head_dim, element_bytes);
        decode_attention = attention_context_work(
            batch.decode_requests, local_query_heads, local_kv_heads,
            model.head_dim, element_bytes);
    } else {
        const std::uint64_t local_qkv_dim =
            (local_query_heads + 2 * local_kv_heads) * model.head_dim;
        attention_pre_projection_ms = kernel_ms(
            gemm_work(batch.total_tokens, model.hidden_size, local_qkv_dim,
                      element_bytes),
            gemm_efficiency);
        attention_post_projection_ms = kernel_ms(
            gemm_work(batch.total_tokens,
                      std::max<std::uint64_t>(
                          1, model.hidden_size / model.tensor_parallel_size),
                      model.hidden_size, element_bytes),
            gemm_efficiency);
        rope_elements = tokens * (query_heads + kv_heads) * head_dim;
        kv_elements = tokens * 2.0 * kv_heads * head_dim;
        prefill_attention = attention_context_work(
            batch.prefill_requests, local_query_heads, local_kv_heads,
            model.head_dim, element_bytes);
        decode_attention = attention_context_work(
            batch.decode_requests, local_query_heads, local_kv_heads,
            model.head_dim, element_bytes);
    }

    const double attention_norm_factor = model.fused_add_norm ? 3.0 : 2.0;
    const double mlp_norm_factor = model.fused_add_norm ? 3.0 : 2.0;
    const double activation_elements = tokens * intermediate;

    double residual_add_ms = 0.0;
    if (!model.fused_add_norm) {
        const double residual_elements = tokens * hidden;
        residual_add_ms =
            kernel_ms(streaming_work(2.0 * residual_elements, residual_elements,
                                     residual_elements, element_bytes),
                      config.streaming);
    }

    return [&]() {
        DenseLayerTimes value{};
        value.attention_pre_projection_ms = attention_pre_projection_ms;
        value.attention_post_projection_ms = attention_post_projection_ms;
        value.rope_ms =
            kernel_ms(streaming_work(rope_elements, rope_elements,
                                     6.0 * rope_elements, element_bytes),
                      config.streaming);
        value.kv_cache_save_ms = kernel_ms(
            streaming_work(kv_elements, kv_elements, 0.0, element_bytes),
            config.streaming);
        value.attention_norm_ms = kernel_ms(
            streaming_work(tokens * hidden * (attention_norm_factor - 1.0),
                           tokens * hidden, 5.0 * tokens * hidden,
                           element_bytes),
            config.streaming);
        value.attention_inter_norm_ms = attention_inter_norm_ms;
        value.attention_wq_projection_ms = attention_wq_projection_ms;
        value.prefill_attention_ms =
            kernel_ms(prefill_attention, config.prefill_attention);
        value.decode_attention_ms =
            kernel_ms(decode_attention, config.decode_attention);
        value.mlp_up_projection_ms = kernel_ms(
            gemm_work(batch.total_tokens, model.hidden_size, local_intermediate,
                      element_bytes, gated_multiplier),
            gemm_efficiency);
        value.mlp_activation_ms = kernel_ms(
            streaming_work(
                activation_elements * static_cast<double>(gated_multiplier),
                activation_elements, 8.0 * activation_elements, element_bytes),
            config.streaming);
        value.mlp_down_projection_ms =
            kernel_ms(gemm_work(batch.total_tokens, local_intermediate,
                                model.hidden_size, element_bytes),
                      gemm_efficiency);
        value.mlp_norm_ms =
            kernel_ms(streaming_work(tokens * hidden * (mlp_norm_factor - 1.0),
                                     tokens * hidden, 5.0 * tokens * hidden,
                                     element_bytes),
                      config.streaming);
        value.residual_add_ms = residual_add_ms;
        return value;
    }();
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
                  Precision precision) {
    if (model.hidden_size == 0 || model.intermediate_size == 0 ||
        model.model_num_experts == 0 || model.moe_tensor_parallel_size == 0 ||
        router_topk == 0) {
        throw AnalyticalModelError("MoE model dimensions must be positive");
    }

    const double element_bytes = bytes_per_element(precision);
    const std::uint64_t local_intermediate =
        ceil_div(model.intermediate_size, model.moe_tensor_parallel_size);
    const std::uint64_t gated_multiplier = model.gated_mlp ? 2 : 1;
    const auto roofline = [&](const KernelWork &work,
                              const Efficiency &efficiency) {
        return predict_ms(device, precision, work, efficiency,
                          config.kernel_launch_latency_us);
    };

    KernelWork grouped_up{};
    KernelWork grouped_down{};
    std::uint64_t local_routed_tokens = 0;
    for (const std::uint64_t tokens : local_expert_tokens) {
        if (tokens == 0) {
            continue;
        }
        local_routed_tokens += tokens;
        const KernelWork up =
            gemm_work(tokens, model.hidden_size, local_intermediate,
                      element_bytes, gated_multiplier);
        grouped_up.flops += up.flops;
        grouped_up.hbm_bytes += up.hbm_bytes;
        const KernelWork down = gemm_work(tokens, local_intermediate,
                                          model.hidden_size, element_bytes);
        grouped_down.flops += down.flops;
        grouped_down.hbm_bytes += down.hbm_bytes;
    }

    const double tokens = static_cast<double>(input_tokens);
    const double hidden = static_cast<double>(model.hidden_size);
    const double experts = static_cast<double>(model.model_num_experts);
    const double routed = static_cast<double>(local_routed_tokens);
    const double norm_factor = model.fused_add_norm ? 3.0 : 2.0;

    return [&]() {
        MoELayerTime value{};
        value.gating_linear_ms =
            roofline(gemm_work(input_tokens, model.hidden_size,
                               model.model_num_experts, element_bytes),
                     input_tokens < config.small_gemm_token_threshold
                         ? config.small_gemm
                         : config.large_gemm);
        value.gating_routing_topk_ms =
            roofline(streaming_work(tokens * experts,
                                    tokens * static_cast<double>(router_topk),
                                    4.0 * tokens * experts, element_bytes),
                     config.routing);
        value.grouped_up_projection_ms = roofline(grouped_up, config.moe);
        value.grouped_down_projection_ms = roofline(grouped_down, config.moe);
        value.shuffling_ms =
            roofline(streaming_work(routed * hidden, routed * hidden, 0.0,
                                    element_bytes),
                     config.streaming);
        value.post_attention_norm_ms =
            roofline(streaming_work(tokens * hidden * (norm_factor - 1.0),
                                    tokens * hidden, 5.0 * tokens * hidden,
                                    element_bytes),
                     config.streaming);
        return value;
    }();
}

MoELanePrediction
predict_moe_lanes(const DeviceCeilings &device, const AnalyticalConfig &config,
                  const MoEModel &model, const RoutingAllocation &routing,
                  std::uint64_t router_topk, Precision precision) {
    if (routing.lane_expert_tokens.empty()) {
        throw AnalyticalModelError(
            "MoE routing must contain at least one lane");
    }
    MoELanePrediction prediction;
    prediction.lane_times.reserve(routing.lane_expert_tokens.size());
    for (const auto &lane : routing.lane_expert_tokens) {
        prediction.lane_times.push_back(
            predict_moe_layer(device, config, model, routing.input_tokens,
                              router_topk, lane, precision));
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
