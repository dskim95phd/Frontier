#include "frontier/metrics/output_contract.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include "frontier/metrics/output_contract_internal.h"

namespace frontier::metrics {
namespace output_detail {

using OrderedJson = nlohmann::ordered_json;

OrderedJson
serialize_execution_time_components(const entities::ExecutionTime &execution) {
    return OrderedJson::object({
        {"dense_compute_ms", execution.dense_compute_ms},
        {"lm_head_ms", execution.lm_head_ms},
        {"tp_communication_ms", execution.tp_communication_ms},
        {"pp_communication_ms", execution.pp_communication_ms},
        {"moe_gating_linear_ms", execution.moe_gating_linear_ms},
        {"moe_gating_routing_topk_ms", execution.moe_gating_routing_topk_ms},
        {"moe_grouped_gemm_ms", execution.moe_grouped_gemm_ms},
        {"moe_shuffling_ms", execution.moe_shuffling_ms},
        {"moe_post_attention_norm_ms", execution.moe_post_attention_norm_ms},
        {"moe_tp_communication_ms", execution.moe_tp_communication_ms},
        {"ep_dispatch_ms", execution.ep_dispatch_ms},
        {"ep_combine_ms", execution.ep_combine_ms},
        {"dp_input_communication_ms", execution.dp_input_communication_ms},
        {"dp_output_communication_ms", execution.dp_output_communication_ms},
        {"synchronization_wait_ms", execution.synchronization_wait_ms},
        {"moe_pre_barrier_wait_ms", execution.moe_pre_barrier_wait_ms},
        {"moe_ep_aggregation_extra_ms", execution.moe_ep_aggregation_extra_ms},
        {"synchronization_unattributed_wait_ms",
         execution.synchronization_unattributed_wait_ms},
        {"synchronization_attribution_overlap_ms",
         execution.synchronization_attribution_overlap_ms},
        {"prefill_attention_token_pairs",
         execution.prefill_attention_token_pairs},
        {"total_ms", execution.total_ms()},
    });
}

} // namespace output_detail
namespace {

using OrderedJson = nlohmann::ordered_json;

using output_detail::is_pdd;
using output_detail::milliseconds_between;
using output_detail::require_valid_time;
using output_detail::serialize_execution_time_components;
using output_detail::validate_request_metrics;

OrderedJson serialize_request(const RequestMetricsRecord &request,
                              config::SystemArchitecture architecture) {
    OrderedJson json = OrderedJson::object();
    json["request_id"] = request.request_id.value();
    if (request.session_id.valid()) {
        json["session_id"] = request.session_id.value();
    } else {
        json["session_id"] = nullptr;
    }
    json["num_prefill_tokens"] = request.num_prefill_tokens;
    json["num_decode_tokens"] = request.num_decode_tokens;
    json["scheduled_prefill_tokens"] = request.scheduled_prefill_tokens;
    json["preemption_recomputed_prefill_tokens"] =
        request.preemption_recomputed_prefill_tokens;
    json["cached_prefill_tokens"] = request.cached_prefill_tokens;
    json["prefix_cache_query_blocks"] = request.prefix_cache_query_blocks;
    json["prefix_cache_hit_blocks"] = request.prefix_cache_hit_blocks;
    json["gpu_prefix_hit_blocks"] = request.gpu_prefix_hit_blocks;
    json["cpu_prefix_query_blocks"] = request.cpu_prefix_query_blocks;
    json["cpu_prefix_hit_blocks"] = request.cpu_prefix_hit_blocks;
    json["cpu_restore_transferred_blocks"] =
        request.cpu_restore_transferred_blocks;
    json["cpu_restore_consumed_blocks"] = request.cpu_restore_consumed_blocks;
    json["cpu_restore_discarded_blocks"] = request.cpu_restore_discarded_blocks;
    json["cpu_restored_tokens"] = request.cpu_restored_tokens;
    json["cpu_restore_bytes"] = request.cpu_restore_bytes;
    json["cpu_restore_queue_time_ms"] = request.cpu_restore_queue_time_s * 1e3;
    json["cpu_restore_service_time_ms"] =
        request.cpu_restore_service_time_s * 1e3;
    json["cpu_offload_bytes"] = request.cpu_offload_bytes;
    json["cpu_offload_queue_time_ms"] = request.cpu_offload_queue_time_s * 1e3;
    json["cpu_offload_service_time_ms"] =
        request.cpu_offload_service_time_s * 1e3;
    json["prefix_cache_key_mode"] =
        config::to_string(request.prefix_cache_key_mode);
    json["arrived_at_s"] = request.arrived_at.seconds();
    json["first_scheduled_at_s"] = request.first_scheduled_at.seconds();
    json["prefill_completed_at_s"] = request.prefill_completed_at.seconds();
    json["first_token_completed_at_s"] =
        request.first_token_completed_at.seconds();
    json["completed_at_s"] = request.completed_at.seconds();
    json["scheduling_delay_ms"] = milliseconds_between(
        request.first_scheduled_at, request.arrived_at, "first_scheduled_at");
    json["prefill_latency_ms"] =
        milliseconds_between(request.prefill_completed_at, request.arrived_at,
                             "prefill_completed_at");
    json["ttft_ms"] =
        milliseconds_between(request.first_token_completed_at,
                             request.arrived_at, "first_token_completed_at");
    json["e2e_ms"] = milliseconds_between(request.completed_at,
                                          request.arrived_at, "completed_at");
    json["num_processed_tokens"] = request.num_processed_tokens;
    json["preemption_count"] = request.preemption_count;
    json["tokens_at_preemption"] = request.tokens_at_preemption;
    if (!is_pdd(architecture)) {
        json["replica_id"] = request.replica_id.value();
        json["dp_id"] = request.dp_id.value();
    } else {
        if (!request.prefill_replica_id.valid() ||
            !request.prefill_dp_id.valid() ||
            !request.decode_replica_id.valid() ||
            !request.decode_dp_id.valid() || !request.transfer_id.valid() ||
            !request.kv_cache_transfer_start_time.valid() ||
            !request.kv_cache_transfer_end_time.valid() ||
            !request.decode_arrived_at.valid() ||
            request.kv_cache_transfer_size_bytes == 0) {
            throw std::invalid_argument(
                "PDD request metrics require complete ownership");
        }
        json["prefill_replica_id"] = request.prefill_replica_id.value();
        json["prefill_dp_id"] = request.prefill_dp_id.value();
        json["decode_replica_id"] = request.decode_replica_id.value();
        json["decode_dp_id"] = request.decode_dp_id.value();
        json["transfer_id"] = request.transfer_id.value();
        json["kv_cache_transfer_start_time_s"] =
            request.kv_cache_transfer_start_time.seconds();
        json["kv_cache_transfer_end_time_s"] =
            request.kv_cache_transfer_end_time.seconds();
        json["kv_cache_transfer_time_ms"] = milliseconds_between(
            request.kv_cache_transfer_end_time,
            request.kv_cache_transfer_start_time, "kv_cache_transfer_end_time");
        json["kv_cache_transfer_size_bytes"] =
            request.kv_cache_transfer_size_bytes;
        json["decode_arrived_at_s"] = request.decode_arrived_at.seconds();
    }
    return json;
}

OrderedJson serialize_event(const Event &event) {
    require_valid_time(event.time, "event.time");

    OrderedJson json = OrderedJson::object();
    json["time_s"] = event.time.seconds();
    json["sequence"] = event.sequence.value();
    json["type"] = to_string(event.type());
    std::visit(
        [&json](const auto &payload) {
            using Payload =
                std::remove_cv_t<std::remove_reference_t<decltype(payload)>>;
            if constexpr (detail::has_request_id<Payload>::value) {
                json["request_id"] = payload.request_id.value();
            }
            if constexpr (detail::has_batch_id<Payload>::value) {
                json["batch_id"] = payload.batch_id.value();
            }
            if constexpr (detail::has_replica_id<Payload>::value) {
                json["replica_id"] = payload.replica_id.value();
            }
            if constexpr (detail::has_dp_id<Payload>::value) {
                json["dp_id"] = payload.dp_id.value();
            }
            if constexpr (detail::has_stage_id<Payload>::value) {
                json["stage_id"] = payload.stage_id.value();
            }
            if constexpr (detail::has_generation<Payload>::value) {
                json["generation"] = payload.generation.value();
            }
            if constexpr (detail::has_cpu_generation<Payload>::value) {
                json["cpu_generation"] = payload.cpu_generation.value();
            }
            if constexpr (detail::has_sync_generation<Payload>::value) {
                json["sync_generation"] = payload.sync_generation.value();
            }
            if constexpr (detail::has_cluster_type<Payload>::value) {
                json["cluster_type"] = to_string(payload.cluster_type);
            }
            if constexpr (detail::has_transfer_id<Payload>::value) {
                json["transfer_id"] = payload.transfer_id.value();
            }
            if constexpr (detail::has_participant_id<Payload>::value) {
                json["participant_id"] = payload.participant_id.value();
            }
            if constexpr (detail::has_sync_group_id<Payload>::value) {
                json["sync_group_id"] = payload.sync_group_id.value();
            }
            if constexpr (detail::has_layer_id<Payload>::value) {
                json["layer_id"] = payload.layer_id.value();
            }
            if constexpr (detail::has_sync_phase<Payload>::value) {
                json["sync_phase"] = payload.sync_phase == MoESyncPhase::kPreMoe
                                         ? "pre_moe"
                                         : "post_moe";
            }
            if constexpr (detail::has_elapsed_component_ms<Payload>::value) {
                json["elapsed_component_ms"] = payload.elapsed_component_ms;
            }
            if constexpr (detail::has_is_idle<Payload>::value) {
                json["is_idle"] = payload.is_idle;
            }
        },
        event.payload);
    return json;
}

OrderedJson serialize_batch(const BatchMetricsRecord &batch,
                            config::SystemArchitecture architecture) {
    require_valid_time(batch.scheduled_at, "batch.scheduled_at");
    require_valid_time(batch.completed_at, "batch.completed_at");
    if (batch.completed_at.seconds() < batch.scheduled_at.seconds()) {
        throw std::invalid_argument(
            "batch completion must not precede scheduling");
    }
    if (batch.request_ids.empty() ||
        batch.request_ids.size() != batch.scheduled_tokens.size()) {
        throw std::invalid_argument(
            "batch request IDs and token counts must be nonempty and aligned");
    }
    std::uint64_t total = 0;
    for (const std::uint64_t tokens : batch.scheduled_tokens) {
        if (tokens == 0) {
            throw std::invalid_argument(
                "batch scheduled token counts must be positive");
        }
        total += tokens;
    }
    if (total != batch.total_scheduled_tokens ||
        batch.num_prefill_tokens + batch.num_decode_tokens != total) {
        throw std::invalid_argument("batch token totals are inconsistent");
    }
    if (!std::isfinite(batch.predicted_execution_ms) ||
        batch.predicted_execution_ms < 0.0) {
        throw std::invalid_argument(
            "batch predicted execution time must be finite and nonnegative");
    }

    OrderedJson json = OrderedJson::object();
    json["batch_id"] = batch.batch_id.value();
    json["iteration_id"] = batch.iteration_id.value();
    json["scheduled_at_s"] = batch.scheduled_at.seconds();
    json["completed_at_s"] = batch.completed_at.seconds();
    json["request_ids"] = OrderedJson::array();
    for (const RequestId request_id : batch.request_ids) {
        json["request_ids"].push_back(request_id.value());
    }
    json["scheduled_tokens"] = batch.scheduled_tokens;
    json["total_scheduled_tokens"] = batch.total_scheduled_tokens;
    json["num_prefill_tokens"] = batch.num_prefill_tokens;
    json["num_decode_tokens"] = batch.num_decode_tokens;
    json["predicted_execution_ms"] = batch.predicted_execution_ms;
    json["replica_id"] = batch.replica_id.value();
    json["dp_id"] = batch.dp_id.value();
    json["num_pipeline_stages"] = batch.num_pipeline_stages;
    json["model_kind"] = std::string{config::to_string(batch.model_kind)};
    json["runtime_total_experts"] = batch.runtime_total_experts;
    json["router_topk"] = batch.router_topk;
    json["attention_tensor_parallel_size"] =
        batch.parallelism.tensor_parallel_size;
    json["attention_data_parallel_size"] = batch.parallelism.data_parallel_size;
    json["moe_tensor_parallel_size"] =
        batch.parallelism.moe_tensor_parallel_size;
    json["moe_expert_parallel_size"] =
        batch.parallelism.moe_expert_parallel_size;
    json["pipeline_exclusive"] = batch.parallelism.pipeline_exclusive;
    if (batch.moe_sync_group_id.valid()) {
        json["moe_sync_group_id"] = batch.moe_sync_group_id.value();
    }
    if (is_pdd(architecture)) {
        json["cluster_type"] = to_string(batch.cluster_type);
    }
    return json;
}

OrderedJson serialize_batch_stage(const BatchStageMetricsRecord &stage) {
    require_valid_time(stage.arrived_at, "batch_stage.arrived_at");
    require_valid_time(stage.started_at, "batch_stage.started_at");
    require_valid_time(stage.completed_at, "batch_stage.completed_at");
    if (stage.started_at.seconds() < stage.arrived_at.seconds() ||
        stage.completed_at.seconds() < stage.started_at.seconds()) {
        throw std::invalid_argument("batch stage timestamps are out of order");
    }
    const entities::ExecutionTime &execution = stage.execution_time;
    if (!std::isfinite(execution.dense_compute_ms) ||
        !std::isfinite(execution.lm_head_ms) ||
        !std::isfinite(execution.tp_communication_ms) ||
        !std::isfinite(execution.pp_communication_ms) ||
        !std::isfinite(execution.moe_gating_linear_ms) ||
        !std::isfinite(execution.moe_gating_routing_topk_ms) ||
        !std::isfinite(execution.moe_grouped_gemm_ms) ||
        !std::isfinite(execution.moe_shuffling_ms) ||
        !std::isfinite(execution.moe_post_attention_norm_ms) ||
        !std::isfinite(execution.moe_tp_communication_ms) ||
        !std::isfinite(execution.ep_dispatch_ms) ||
        !std::isfinite(execution.ep_combine_ms) ||
        !std::isfinite(execution.dp_input_communication_ms) ||
        !std::isfinite(execution.dp_output_communication_ms) ||
        !std::isfinite(execution.synchronization_wait_ms) ||
        !std::isfinite(execution.moe_pre_barrier_wait_ms) ||
        !std::isfinite(execution.moe_ep_aggregation_extra_ms) ||
        !std::isfinite(execution.synchronization_unattributed_wait_ms) ||
        !std::isfinite(execution.synchronization_attribution_overlap_ms) ||
        execution.dense_compute_ms < 0.0 || execution.lm_head_ms < 0.0 ||
        execution.tp_communication_ms < 0.0 ||
        execution.pp_communication_ms < 0.0 ||
        execution.moe_gating_linear_ms < 0.0 ||
        execution.moe_gating_routing_topk_ms < 0.0 ||
        execution.moe_grouped_gemm_ms < 0.0 ||
        execution.moe_shuffling_ms < 0.0 ||
        execution.moe_post_attention_norm_ms < 0.0 ||
        execution.moe_tp_communication_ms < 0.0 ||
        execution.ep_dispatch_ms < 0.0 || execution.ep_combine_ms < 0.0 ||
        execution.dp_input_communication_ms < 0.0 ||
        execution.dp_output_communication_ms < 0.0 ||
        execution.synchronization_wait_ms < 0.0 ||
        execution.moe_pre_barrier_wait_ms < 0.0 ||
        execution.moe_ep_aggregation_extra_ms < 0.0 ||
        execution.synchronization_unattributed_wait_ms < 0.0 ||
        execution.synchronization_attribution_overlap_ms < 0.0) {
        throw std::invalid_argument("batch stage execution time is invalid");
    }
    OrderedJson json = OrderedJson::object({
        {"batch_id", stage.batch_id.value()},
        {"replica_id", stage.replica_id.value()},
        {"dp_id", stage.dp_id.value()},
        {"stage_id", stage.stage_id.value()},
        {"arrived_at_s", stage.arrived_at.seconds()},
        {"started_at_s", stage.started_at.seconds()},
        {"completed_at_s", stage.completed_at.seconds()},
        {"dense_compute_ms", execution.dense_compute_ms},
        {"lm_head_ms", execution.lm_head_ms},
        {"tp_communication_ms", execution.tp_communication_ms},
        {"pp_communication_ms", execution.pp_communication_ms},
        {"moe_gating_linear_ms", execution.moe_gating_linear_ms},
        {"moe_gating_routing_topk_ms", execution.moe_gating_routing_topk_ms},
        {"moe_grouped_gemm_ms", execution.moe_grouped_gemm_ms},
        {"moe_shuffling_ms", execution.moe_shuffling_ms},
        {"moe_post_attention_norm_ms", execution.moe_post_attention_norm_ms},
        {"moe_tp_communication_ms", execution.moe_tp_communication_ms},
        {"ep_dispatch_ms", execution.ep_dispatch_ms},
        {"ep_combine_ms", execution.ep_combine_ms},
        {"dp_input_communication_ms", execution.dp_input_communication_ms},
        {"dp_output_communication_ms", execution.dp_output_communication_ms},
        {"synchronization_wait_ms", execution.synchronization_wait_ms},
        {"moe_pre_barrier_wait_ms", execution.moe_pre_barrier_wait_ms},
        {"moe_ep_aggregation_extra_ms", execution.moe_ep_aggregation_extra_ms},
        {"synchronization_unattributed_wait_ms",
         execution.synchronization_unattributed_wait_ms},
        {"synchronization_attribution_overlap_ms",
         execution.synchronization_attribution_overlap_ms},
        {"prefill_attention_token_pairs",
         execution.prefill_attention_token_pairs},
        {"duration_ms",
         (stage.completed_at.seconds() - stage.started_at.seconds()) * 1e3},
        {"model_kind", std::string{config::to_string(stage.model_kind)}},
        {"runtime_total_experts", stage.runtime_total_experts},
        {"router_topk", stage.router_topk},
        {"attention_tensor_parallel_size",
         stage.parallelism.tensor_parallel_size},
        {"attention_data_parallel_size", stage.parallelism.data_parallel_size},
        {"moe_tensor_parallel_size",
         stage.parallelism.moe_tensor_parallel_size},
        {"moe_expert_parallel_size",
         stage.parallelism.moe_expert_parallel_size},
        {"pipeline_exclusive", stage.parallelism.pipeline_exclusive},
    });
    if (stage.moe_sync_group_id.valid()) {
        json["moe_sync_group_id"] = stage.moe_sync_group_id.value();
    }
    if (stage.cluster_type != ClusterType::kMonolithic) {
        json["cluster_type"] = to_string(stage.cluster_type);
    }
    return json;
}

OrderedJson serialize_scheduler_trace(const SchedulerTraceRecord &trace,
                                      config::SystemArchitecture architecture) {
    require_valid_time(trace.simulation_time, "scheduler simulation_time");
    if (trace.token_budget_after > trace.token_budget_before) {
        throw std::invalid_argument("invalid scheduler trace counters");
    }
    if (trace.batch_request_ids.size() != trace.request_num_tokens.size()) {
        throw std::invalid_argument(
            "scheduler trace batch IDs and tokens are misaligned");
    }

    OrderedJson json = OrderedJson::object();
    json["iteration_id"] = trace.iteration_id.value();
    json["simulation_time_s"] = trace.simulation_time.seconds();
    json["decisions"] = OrderedJson::array();
    for (const SchedulerDecisionRecord &decision : trace.decisions) {
        json["decisions"].push_back(OrderedJson::object({
            {"decision_result", decision.decision_result},
            {"request_id", decision.request_id.value()},
            {"num_tokens", decision.num_tokens},
            {"token_budget_after", decision.token_budget_after},
            {"available_blocks_after", decision.available_blocks_after},
        }));
    }
    json["token_budget_before"] = trace.token_budget_before;
    json["token_budget_after"] = trace.token_budget_after;
    json["available_blocks_before"] = trace.available_blocks_before;
    json["available_blocks_after"] = trace.available_blocks_after;
    json["waiting_count_before"] = trace.waiting_count_before;
    json["waiting_count_after"] = trace.waiting_count_after;
    json["running_count_before"] = trace.running_count_before;
    json["running_count_after"] = trace.running_count_after;
    json["preempted_count"] = trace.preempted_count;
    json["batch_request_ids"] = OrderedJson::array();
    for (const RequestId request_id : trace.batch_request_ids) {
        json["batch_request_ids"].push_back(request_id.value());
    }
    json["request_num_tokens"] = trace.request_num_tokens;
    json["replica_id"] = trace.replica_id.value();
    json["dp_id"] = trace.dp_id.value();
    if (is_pdd(architecture)) {
        json["cluster_type"] = to_string(trace.cluster_type);
    }
    return json;
}

OrderedJson
serialize_kv_cache_transfer(const KVCacheTransferMetricsRecord &transfer) {
    require_valid_time(transfer.started_at, "transfer.started_at");
    require_valid_time(transfer.completed_at, "transfer.completed_at");
    if (transfer.completed_at.seconds() < transfer.started_at.seconds() ||
        transfer.size_bytes == 0 ||
        !std::isfinite(transfer.predicted_time_ms) ||
        transfer.predicted_time_ms < 0.0) {
        throw std::invalid_argument("KV transfer metrics are invalid");
    }
    return OrderedJson::object({
        {"transfer_id", transfer.transfer_id.value()},
        {"request_id", transfer.request_id.value()},
        {"source_batch_id", transfer.source_batch_id.value()},
        {
            "source_cluster_type",
            to_string(transfer.source_cluster_type),
        },
        {
            "target_cluster_type",
            to_string(transfer.target_cluster_type),
        },
        {"source_replica_id", transfer.source_replica_id.value()},
        {"source_dp_id", transfer.source_dp_id.value()},
        {"size_bytes", transfer.size_bytes},
        {"predicted_time_ms", transfer.predicted_time_ms},
        {"started_at_s", transfer.started_at.seconds()},
        {"completed_at_s", transfer.completed_at.seconds()},
    });
}

OrderedJson serialize_diagnostic(const AnalyticalDiagnostic &diagnostic) {
    if (diagnostic.name.empty()) {
        throw std::invalid_argument(
            "analytical diagnostic name must not be empty");
    }

    OrderedJson values = OrderedJson::object();
    for (const auto &[name, value] : diagnostic.values) {
        if (name.empty()) {
            throw std::invalid_argument(
                "analytical diagnostic field name must not be empty");
        }
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "analytical diagnostic values must be finite");
        }
        if (values.contains(name)) {
            throw std::invalid_argument(
                "analytical diagnostic contains duplicate field '" + name +
                "'");
        }
        values[name] = value;
    }

    OrderedJson json = OrderedJson::object();
    json["name"] = diagnostic.name;
    json["values"] = std::move(values);
    return json;
}

OrderedJson serialize_moe_routing(const MoERoutingMetricsRecord &routing) {
    const auto &geometry = routing.grouped_gemm_geometry;
    const bool invalid_geometry =
        !std::isfinite(geometry.up_wave_utilization) ||
        !std::isfinite(geometry.down_wave_utilization) ||
        !std::isfinite(geometry.up_cluster_task_overhead_ms) ||
        !std::isfinite(geometry.down_cluster_task_overhead_ms) ||
        geometry.up_wave_utilization <= 0.0 ||
        geometry.up_wave_utilization > 1.0 ||
        geometry.down_wave_utilization <= 0.0 ||
        geometry.down_wave_utilization > 1.0 ||
        geometry.up_cluster_task_overhead_ms < 0.0 ||
        geometry.down_cluster_task_overhead_ms < 0.0 ||
        (geometry.enabled &&
         (geometry.block_m == 0 || geometry.routed_m_blocks == 0 ||
          geometry.up_cluster_tasks == 0 || geometry.down_cluster_tasks == 0));
    if (!routing.batch_id.valid() || !routing.stage_id.valid() ||
        !routing.layer_id.valid() || routing.input_tokens == 0 ||
        routing.routed_tokens == 0 || routing.global_expert_tokens.empty() ||
        routing.lane_expert_tokens.empty() ||
        routing.lane_routed_tokens.size() !=
            routing.lane_expert_tokens.size() ||
        routing.lane_active_experts.size() !=
            routing.lane_expert_tokens.size() ||
        routing.lane_unique_tokens.size() !=
            routing.lane_expert_tokens.size() ||
        routing.lane_times_ms.size() != routing.lane_expert_tokens.size() ||
        routing.routed_lane_times_ms.size() !=
            routing.lane_expert_tokens.size() ||
        routing.source_local_lane_times_ms.size() !=
            routing.lane_expert_tokens.size() ||
        routing.critical_lane >= routing.lane_times_ms.size() ||
        !std::isfinite(routing.shared_expert_path_ms) ||
        routing.shared_expert_path_ms < 0.0 ||
        !std::isfinite(routing.critical_lane_time_ms) ||
        routing.critical_lane_time_ms < 0.0 ||
        !std::isfinite(routing.raw_ep_dispatch_ms) ||
        !std::isfinite(routing.raw_ep_combine_ms) ||
        !std::isfinite(routing.exposed_ep_dispatch_ms) ||
        !std::isfinite(routing.exposed_ep_combine_ms) ||
        routing.raw_ep_dispatch_ms < 0.0 || routing.raw_ep_combine_ms < 0.0 ||
        routing.exposed_ep_dispatch_ms < 0.0 ||
        routing.exposed_ep_combine_ms < 0.0 || invalid_geometry ||
        std::any_of(
            routing.lane_times_ms.begin(), routing.lane_times_ms.end(),
            [](double value) {
                return !std::isfinite(value) || value < 0.0;
            }) ||
        std::any_of(
            routing.routed_lane_times_ms.begin(),
            routing.routed_lane_times_ms.end(),
            [](double value) {
                return !std::isfinite(value) || value < 0.0;
            }) ||
        std::any_of(
            routing.source_local_lane_times_ms.begin(),
            routing.source_local_lane_times_ms.end(),
            [](double value) {
                return !std::isfinite(value) || value < 0.0;
            })) {
        throw std::invalid_argument("MoE routing diagnostic is invalid");
    }
    const std::uint64_t global_total =
        std::accumulate(routing.global_expert_tokens.begin(),
                        routing.global_expert_tokens.end(), std::uint64_t{0});
    if (global_total != routing.routed_tokens) {
        throw std::invalid_argument(
            "MoE routing diagnostic does not conserve routed tokens");
    }
    const std::uint64_t lane_total =
        std::accumulate(routing.lane_routed_tokens.begin(),
                        routing.lane_routed_tokens.end(), std::uint64_t{0});
    if (lane_total != routing.routed_tokens ||
        std::any_of(routing.lane_unique_tokens.begin(),
                    routing.lane_unique_tokens.end(),
                    [&routing](std::uint64_t value) {
                        return value > routing.input_tokens;
                    })) {
        throw std::invalid_argument(
            "MoE routing lane traffic diagnostic is inconsistent");
    }
    OrderedJson json = OrderedJson::object({
        {"batch_id", routing.batch_id.value()},
        {"stage_id", routing.stage_id.value()},
        {"cluster_type", to_string(routing.cluster_type)},
        {"layer_id", routing.layer_id.value()},
        {"routing_mode", std::string{config::to_string(routing.mode)}},
        {"routing_distribution",
         std::string{config::to_string(routing.distribution)}},
        {"seed", routing.seed},
        {"input_tokens", routing.input_tokens},
        {"routed_tokens", routing.routed_tokens},
        {"global_expert_tokens", routing.global_expert_tokens},
        {"lane_expert_tokens", routing.lane_expert_tokens},
        {"lane_routed_tokens", routing.lane_routed_tokens},
        {"lane_active_experts", routing.lane_active_experts},
        {"lane_unique_tokens", routing.lane_unique_tokens},
        {"lane_times_ms", routing.lane_times_ms},
        {"routed_lane_times_ms", routing.routed_lane_times_ms},
        {"source_local_lane_times_ms", routing.source_local_lane_times_ms},
        {"shared_expert_path_ms", routing.shared_expert_path_ms},
        {"critical_lane", routing.critical_lane},
        {"critical_lane_time_ms", routing.critical_lane_time_ms},
        {"raw_ep_dispatch_ms", routing.raw_ep_dispatch_ms},
        {"raw_ep_combine_ms", routing.raw_ep_combine_ms},
        {"exposed_ep_dispatch_ms", routing.exposed_ep_dispatch_ms},
        {"exposed_ep_combine_ms", routing.exposed_ep_combine_ms},
        {"grouped_gemm_geometry",
         OrderedJson::object({
             {"enabled", geometry.enabled},
             {"block_m", geometry.block_m},
             {"routed_m_blocks", geometry.routed_m_blocks},
             {"shared_m_blocks", geometry.shared_m_blocks},
             {"routed_padded_tokens", geometry.routed_padded_tokens},
             {"shared_padded_tokens", geometry.shared_padded_tokens},
             {"up_cluster_tasks", geometry.up_cluster_tasks},
             {"down_cluster_tasks", geometry.down_cluster_tasks},
             {"up_wave_utilization", geometry.up_wave_utilization},
             {"down_wave_utilization", geometry.down_wave_utilization},
             {"up_cluster_task_overhead_ms",
              geometry.up_cluster_task_overhead_ms},
             {"down_cluster_task_overhead_ms",
              geometry.down_cluster_task_overhead_ms},
         })},
    });
    if (routing.sync_group_id.valid()) {
        json["sync_group_id"] = routing.sync_group_id.value();
    }
    return json;
}

} // namespace

std::string serialize_simulation_output_json(const SimulationOutput &output) {
    if (output.schema_version != config::kSchemaVersion) {
        throw std::invalid_argument("output schema_version must be 1");
    }
    if (output.run.run_id.empty()) {
        throw std::invalid_argument("output run_id must not be empty");
    }
    validate_request_metrics(output.requests, output.run.system_architecture);

    OrderedJson root = OrderedJson::object();
    root["schema_version"] = output.schema_version;
    if (output.observation_window_seconds.has_value()) {
        root["observation_window_seconds"] =
            output.observation_window_seconds.value();
    }
    root["run"] = OrderedJson::object({
        {"run_id", output.run.run_id},
        {
            "simulation_mode",
            std::string{config::to_string(output.run.simulation_mode)},
        },
        {
            "system_architecture",
            std::string{config::to_string(output.run.system_architecture)},
        },
        {"timestamp_unit", "seconds"},
        {"latency_unit", "milliseconds"},
        {
            "metrics_semantics",
            "canonical",
        },
    });

    root["completed_request_ids"] = OrderedJson::array();
    root["requests"] = OrderedJson::array();
    for (const RequestMetricsRecord &request : output.requests) {
        root["completed_request_ids"].push_back(request.request_id.value());
        root["requests"].push_back(
            serialize_request(request, output.run.system_architecture));
    }

    root["batches"] = OrderedJson::array();
    for (const BatchMetricsRecord &batch : output.batches) {
        root["batches"].push_back(
            serialize_batch(batch, output.run.system_architecture));
    }
    root["gpu_kv_occupancy"] = OrderedJson::array();
    for (const GpuKVCacheOccupancyRecord &record : output.gpu_kv_occupancy) {
        OrderedJson value = OrderedJson::object({
            {"time_s", record.time.seconds()},
            {"cluster_type", to_string(record.cluster_type)},
            {"replica_id", record.replica_id.value()},
            {"dp_id", record.dp_id.value()},
            {"active_blocks", record.active_blocks},
            {"capacity_blocks", record.capacity_blocks},
            {"active_bytes_per_gpu", record.active_bytes_per_gpu},
            {"active_bytes_across_pipeline",
             record.active_bytes_across_pipeline},
            {"active_fraction_of_kv_budget",
             record.active_fraction_of_kv_budget},
        });
        if (record.hbm_fraction.has_value()) {
            value["hbm_fraction"] = record.hbm_fraction.value();
        } else {
            value["hbm_fraction"] = nullptr;
        }
        if (record.active_fraction_of_total_hbm.has_value()) {
            value["active_fraction_of_total_hbm"] =
                record.active_fraction_of_total_hbm.value();
        } else {
            value["active_fraction_of_total_hbm"] = nullptr;
        }
        root["gpu_kv_occupancy"].push_back(std::move(value));
    }
    root["pipeline_memory_diagnostics"] = OrderedJson::array();
    for (const PipelineMemoryDiagnostics &diagnostic :
         output.pipeline_memory_diagnostics) {
        OrderedJson value = OrderedJson::object({
            {"cluster_type", to_string(diagnostic.cluster_type)},
            {"capacity_bytes_per_gpu", diagnostic.capacity_bytes_per_gpu},
            {"model_weight_bytes_per_gpu",
             diagnostic.model_weight_bytes_per_gpu},
            {"kv_cache_budget_bytes_per_gpu",
             diagnostic.kv_cache_budget_bytes_per_gpu},
            {"kv_cache_bytes_per_block", diagnostic.kv_cache_bytes_per_block},
            {"configured_num_blocks", diagnostic.configured_num_blocks},
            {"ordinary_kv_capacity_blocks",
             diagnostic.ordinary_kv_capacity_blocks},
            {"ordinary_kv_limiting_stage",
             diagnostic.ordinary_kv_limiting_stage},
            {"ordinary_kv_limiting_rank", diagnostic.ordinary_kv_limiting_rank},
            {"kda_snapshot_blocks_per_session",
             diagnostic.kda_snapshot_blocks_per_session},
            {"kda_snapshot_limiting_stage",
             diagnostic.kda_snapshot_limiting_stage},
            {"kda_snapshot_limiting_rank",
             diagnostic.kda_snapshot_limiting_rank},
            {"timing_group_count", diagnostic.timing_group_count},
            {"memory_group_count", diagnostic.memory_group_count},
            {"stages", OrderedJson::array()},
        });
        for (const PipelineStageMemoryDiagnostic &stage : diagnostic.stages) {
            OrderedJson stage_value = OrderedJson::object({
                {"stage_id", stage.stage_id.value()},
                {"layer_begin", stage.layer_begin},
                {"layer_end", stage.layer_end},
                {"layer_count", stage.layer_count},
                {"kda_layer_count", stage.kda_layer_count},
                {"mla_layer_count", stage.mla_layer_count},
                {"moe_layer_count", stage.moe_layer_count},
                {"resident_weight_bytes", stage.resident_weight_bytes},
                {"reserved_bytes", stage.reserved_bytes},
                {"free_bytes", stage.free_bytes},
                {"kv_bytes_per_block_by_rank",
                 stage.kv_bytes_per_block_by_rank},
                {"kda_snapshot_bytes_by_rank",
                 stage.kda_snapshot_bytes_by_rank},
                {"timing_group_multiplicity", stage.timing_group_multiplicity},
                {"memory_group_multiplicity", stage.memory_group_multiplicity},
            });
            if (stage.timing_group_id.has_value()) {
                stage_value["timing_group_id"] = stage.timing_group_id.value();
            } else {
                stage_value["timing_group_id"] = nullptr;
            }
            if (stage.memory_group_id.has_value()) {
                stage_value["memory_group_id"] = stage.memory_group_id.value();
            } else {
                stage_value["memory_group_id"] = nullptr;
            }
            value["stages"].push_back(std::move(stage_value));
        }
        root["pipeline_memory_diagnostics"].push_back(std::move(value));
    }
    root["scheduler_trace"] = OrderedJson::array();
    for (const SchedulerTraceRecord &trace : output.scheduler_trace) {
        root["scheduler_trace"].push_back(
            serialize_scheduler_trace(trace, output.run.system_architecture));
    }
    root["batch_stages"] = OrderedJson::array();
    for (const BatchStageMetricsRecord &stage : output.batch_stages) {
        root["batch_stages"].push_back(serialize_batch_stage(stage));
    }

    if (is_pdd(output.run.system_architecture)) {
        root["kv_cache_transfers"] = OrderedJson::array();
        for (const KVCacheTransferMetricsRecord &transfer :
             output.kv_cache_transfers) {
            root["kv_cache_transfers"].push_back(
                serialize_kv_cache_transfer(transfer));
        }
    }

    root["event_trace"] = OrderedJson::array();
    for (const Event &event : output.event_trace) {
        root["event_trace"].push_back(serialize_event(event));
    }

    root["analytical_diagnostics"] = OrderedJson::array();
    for (const AnalyticalDiagnostic &diagnostic :
         output.analytical_diagnostics) {
        root["analytical_diagnostics"].push_back(
            serialize_diagnostic(diagnostic));
    }
    root["moe_routing"] = OrderedJson::array();
    for (const MoERoutingMetricsRecord &routing : output.moe_routing) {
        root["moe_routing"].push_back(serialize_moe_routing(routing));
    }

    const PrefixCacheMetricsAggregate &cache = output.aggregate.prefix_cache;
    const double hit_rate = cache.query_blocks == 0
                                ? 0.0
                                : static_cast<double>(cache.hit_blocks) /
                                      static_cast<double>(cache.query_blocks);
    root["prefix_cache"] = OrderedJson::object({
        {"storage_model", cache.storage_model},
        {"key_mode", config::to_string(cache.key_mode)},
        {"block_size", cache.block_size},
        {"successful_admissions", cache.successful_admissions},
        {"query_blocks", cache.query_blocks},
        {"hit_blocks", cache.hit_blocks},
        {"hit_rate", hit_rate},
        {"evicted_blocks", cache.evicted_blocks},
        {"evicted_sessions", cache.evicted_sessions},
        {"evicted_kda_snapshots", cache.evicted_kda_snapshots},
        {"evicted_kda_snapshot_blocks", cache.evicted_kda_snapshot_blocks},
        {"kda_snapshot_occupied_blocks", cache.kda_snapshot_occupied_blocks},
        {"kda_snapshot_sessions", cache.kda_snapshot_sessions},
    });
    root["prefix_cache_targets"] = OrderedJson::array();
    for (const PrefixCacheTargetMetricsRecord &target :
         output.prefix_cache_targets) {
        root["prefix_cache_targets"].push_back(OrderedJson::object({
            {"cluster_type", to_string(target.cluster_type)},
            {"replica_id", target.replica_id.value()},
            {"dp_id", target.dp_id.value()},
            {"capacity_blocks", target.capacity_blocks},
            {"available_blocks", target.available_blocks},
            {"active_blocks", target.active_blocks},
            {"resident_blocks", target.resident_blocks},
            {"evictable_blocks", target.evictable_blocks},
            {"evictable_sessions", target.evictable_sessions},
            {"sessions_with_nonzero_frontier",
             target.sessions_with_nonzero_frontier},
            {"kda_snapshot_occupied_blocks",
             target.kda_snapshot_occupied_blocks},
            {"kda_snapshot_sessions", target.kda_snapshot_sessions},
            {"kda_snapshot_evictable_sessions",
             target.kda_snapshot_evictable_sessions},
        }));
    }

    const CpuKVCacheMetricsAggregate &cpu = output.aggregate.cpu_kv_cache;
    root["cpu_kv_cache"] = OrderedJson::object({
        {"target_count", cpu.target_count},
        {"capacity_bytes", cpu.capacity_bytes},
        {"capacity_blocks", cpu.capacity_blocks},
        {"bytes_per_block", cpu.bytes_per_block},
        {"kda_snapshot_bytes_per_session", cpu.kda_snapshot_bytes_per_session},
        {"kda_snapshot_blocks_per_session",
         cpu.kda_snapshot_blocks_per_session},
        {"kda_snapshot_occupied_bytes", cpu.kda_snapshot_occupied_bytes},
        {"kda_snapshot_occupied_blocks", cpu.kda_snapshot_occupied_blocks},
        {"kda_snapshot_reserved_bytes", cpu.kda_snapshot_reserved_bytes},
        {"kda_snapshot_reserved_blocks", cpu.kda_snapshot_reserved_blocks},
        {"kda_snapshot_sessions", cpu.kda_snapshot_sessions},
        {"kda_snapshot_evictable_sessions",
         cpu.kda_snapshot_evictable_sessions},
        {"evicted_kda_snapshots", cpu.evicted_kda_snapshots},
        {"evicted_kda_snapshot_blocks", cpu.evicted_kda_snapshot_blocks},
        {"evicted_kda_snapshot_bytes", cpu.evicted_kda_snapshot_bytes},
        {"offload_operations", cpu.offload_operations},
        {"offload_blocks", cpu.offload_blocks},
        {"offload_bytes", cpu.offload_bytes},
        {"kda_snapshot_offload_operations",
         cpu.kda_snapshot_offload_operations},
        {"kda_snapshot_offload_bytes", cpu.kda_snapshot_offload_bytes},
        {"restore_operations", cpu.restore_operations},
        {"restore_blocks", cpu.restore_blocks},
        {"restore_bytes", cpu.restore_bytes},
        {"kda_snapshot_restore_operations",
         cpu.kda_snapshot_restore_operations},
        {"kda_snapshot_restore_bytes", cpu.kda_snapshot_restore_bytes},
        {"d2h_queue_time_ms", cpu.d2h_queue_time_ms},
        {"d2h_service_time_ms", cpu.d2h_service_time_ms},
        {"h2d_queue_time_ms", cpu.h2d_queue_time_ms},
        {"h2d_service_time_ms", cpu.h2d_service_time_ms},
        {"source_gpu_hold_time_ms", cpu.source_gpu_hold_time_ms},
        {"query_blocks", cpu.query_blocks},
        {"hit_blocks", cpu.hit_blocks},
        {"hit_rate", cpu.query_blocks == 0
                         ? 0.0
                         : static_cast<double>(cpu.hit_blocks) /
                               static_cast<double>(cpu.query_blocks)},
        {"resident_bytes", cpu.resident_bytes},
        {"resident_blocks", cpu.resident_blocks},
        {"reserved_bytes", cpu.reserved_bytes},
        {"reserved_blocks", cpu.reserved_blocks},
        {"free_bytes", cpu.free_bytes},
        {"free_blocks", cpu.free_blocks},
        {"peak_resident_bytes", cpu.peak_resident_bytes},
        {"peak_resident_blocks", cpu.peak_resident_blocks},
        {"peak_reserved_bytes", cpu.peak_reserved_bytes},
        {"peak_reserved_blocks", cpu.peak_reserved_blocks},
        {"resident_sessions", cpu.resident_sessions},
        {"evicted_blocks", cpu.evicted_blocks},
        {"evicted_sessions", cpu.evicted_sessions},
        {"evicted_bytes", cpu.evicted_bytes},
        {"skipped_offloads", cpu.skipped_offloads},
        {"truncated_offloads", cpu.truncated_offloads},
        {"stale_generation_completions", cpu.stale_generation_completions},
        {"sessions_with_cpu_hits", cpu.sessions_with_cpu_hits},
        {"pending_restore_operations", cpu.pending_restore_operations},
        {"staged_restore_payloads", cpu.staged_restore_payloads},
        {"active_restore_leases", cpu.active_restore_leases},
        {"active_offload_reservations", cpu.active_offload_reservations},
    });
    root["cpu_kv_cache_targets"] = OrderedJson::array();
    for (const auto &target : output.cpu_kv_cache_targets) {
        root["cpu_kv_cache_targets"].push_back(OrderedJson::object({
            {"cluster_type", to_string(target.cluster_type)},
            {"replica_id", target.replica_id.value()},
            {"dp_id", target.dp_id.value()},
            {"capacity_bytes", target.capacity_bytes},
            {"capacity_blocks", target.capacity_blocks},
            {"bytes_per_block", target.bytes_per_block},
            {"kda_snapshot_bytes_per_session",
             target.kda_snapshot_bytes_per_session},
            {"kda_snapshot_blocks_per_session",
             target.kda_snapshot_blocks_per_session},
            {"kda_snapshot_occupied_bytes", target.kda_snapshot_occupied_bytes},
            {"kda_snapshot_occupied_blocks",
             target.kda_snapshot_occupied_blocks},
            {"kda_snapshot_reserved_bytes", target.kda_snapshot_reserved_bytes},
            {"kda_snapshot_reserved_blocks",
             target.kda_snapshot_reserved_blocks},
            {"kda_snapshot_sessions", target.kda_snapshot_sessions},
            {"kda_snapshot_evictable_sessions",
             target.kda_snapshot_evictable_sessions},
            {"evicted_kda_snapshots", target.evicted_kda_snapshots},
            {"evicted_kda_snapshot_blocks", target.evicted_kda_snapshot_blocks},
            {"evicted_kda_snapshot_bytes", target.evicted_kda_snapshot_bytes},
            {"resident_bytes", target.resident_bytes},
            {"resident_blocks", target.resident_blocks},
            {"reserved_bytes", target.reserved_bytes},
            {"reserved_blocks", target.reserved_blocks},
            {"free_bytes", target.free_bytes},
            {"free_blocks", target.free_blocks},
            {"peak_resident_bytes", target.peak_resident_bytes},
            {"peak_resident_blocks", target.peak_resident_blocks},
            {"peak_reserved_bytes", target.peak_reserved_bytes},
            {"peak_reserved_blocks", target.peak_reserved_blocks},
            {"resident_sessions", target.resident_sessions},
            {"evicted_sessions", target.evicted_sessions},
            {"evicted_blocks", target.evicted_blocks},
            {"evicted_bytes", target.evicted_bytes},
            {"skipped_offloads", target.skipped_offloads},
            {"truncated_offloads", target.truncated_offloads},
            {"stale_generation_completions",
             target.stale_generation_completions},
            {"cpu_query_blocks", target.cpu_query_blocks},
            {"cpu_hit_blocks", target.cpu_hit_blocks},
            {"sessions_with_cpu_hits", target.sessions_with_cpu_hits},
            {"pending_restore_operations", target.pending_restore_operations},
            {"staged_restore_payloads", target.staged_restore_payloads},
            {"active_restore_leases", target.active_restore_leases},
            {"active_offload_reservations", target.active_offload_reservations},
        }));
    }
    root["cpu_kv_cache_transfers"] = OrderedJson::array();
    for (const auto &transfer : output.cpu_kv_cache_transfers) {
        root["cpu_kv_cache_transfers"].push_back(OrderedJson::object({
            {"transfer_id", transfer.transfer_id.value()},
            {"kind", transfer.kind == CpuKVCacheTransferKind::kOffload
                         ? "offload"
                         : "restore"},
            {"request_id", transfer.request_id.value()},
            {"cluster_type", to_string(transfer.cluster_type)},
            {"replica_id", transfer.replica_id.value()},
            {"dp_id", transfer.dp_id.value()},
            {"blocks", transfer.blocks},
            {"size_bytes", transfer.size_bytes},
            {"kda_snapshot_bytes", transfer.kda_snapshot_bytes},
            {"submitted_at_s", transfer.submitted_at.seconds()},
            {"started_at_s", transfer.started_at.seconds()},
            {"completed_at_s", transfer.completed_at.seconds()},
            {"queue_time_ms", transfer.queue_time_ms},
            {"service_time_ms", transfer.service_time_ms},
            {"source_gpu_hold_ms", transfer.source_gpu_hold_ms},
        }));
    }

    std::uint64_t total_scheduled_prefill_tokens = 0;
    std::uint64_t total_recomputed_prefill_tokens = 0;
    for (const RequestMetricsRecord &request : output.requests) {
        total_scheduled_prefill_tokens += request.scheduled_prefill_tokens;
        total_recomputed_prefill_tokens +=
            request.preemption_recomputed_prefill_tokens;
    }
    root["prefill_work"] = OrderedJson::object({
        {"scheduled_prefill_tokens", total_scheduled_prefill_tokens},
        {"preemption_recomputed_prefill_tokens",
         total_recomputed_prefill_tokens},
    });
    root["prefill_attention_token_pairs_by_arrival_time_bucket"] =
        OrderedJson::object();
    for (const auto &[bucket_index, token_pairs] :
         output.aggregate
             .prefill_attention_token_pairs_by_arrival_time_bucket) {
        root["prefill_attention_token_pairs_by_arrival_time_bucket"]
            [std::to_string(bucket_index)] = token_pairs;
    }

    return root.dump(2) + '\n';
}

} // namespace frontier::metrics
