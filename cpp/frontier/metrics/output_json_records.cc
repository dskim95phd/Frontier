#include "frontier/metrics/output_contract_internal.h"

#include <cmath>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

#include <nlohmann/json.hpp>

namespace frontier::metrics::output_detail {

using OrderedJson = nlohmann::ordered_json;

OrderedJson serialize_request(const RequestMetricsRecord &request,
                              config::SystemArchitecture architecture) {
    OrderedJson json = OrderedJson::object();
    json["request_id"] = request.request_id.value();
    if (request.prefill_only) {
        json["prefill_only"] = true;
    }
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
            !request.prefill_dp_id.valid() || !request.transfer_id.valid() ||
            !request.kv_cache_transfer_start_time.valid() ||
            !request.kv_cache_transfer_end_time.valid() ||
            !request.decode_arrived_at.valid() ||
            request.kv_cache_transfer_size_bytes == 0) {
            throw std::invalid_argument(
                "PDD request metrics require complete ownership");
        }
        json["prefill_replica_id"] = request.prefill_replica_id.value();
        json["prefill_dp_id"] = request.prefill_dp_id.value();
        if (request.prefill_only) {
            if (request.decode_replica_id.valid() ||
                request.decode_dp_id.valid()) {
                throw std::invalid_argument(
                    "PREFILL-only request must not report a DECODE owner");
            }
            json["decode_replica_id"] = nullptr;
            json["decode_dp_id"] = nullptr;
        } else {
            if (!request.decode_replica_id.valid() ||
                !request.decode_dp_id.valid()) {
                throw std::invalid_argument(
                    "simulated PDD request requires a DECODE owner");
            }
            json["decode_replica_id"] = request.decode_replica_id.value();
            json["decode_dp_id"] = request.decode_dp_id.value();
        }
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

} // namespace frontier::metrics::output_detail
