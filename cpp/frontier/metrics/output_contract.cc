#include "frontier/metrics/output_contract.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>

#include <nlohmann/json.hpp>

namespace frontier::metrics {
namespace {

using OrderedJson = nlohmann::ordered_json;

bool is_pdd(config::SystemArchitecture architecture) {
    return architecture == config::SystemArchitecture::kPdDisaggregation;
}

void require_valid_time(SimTime time, std::string_view field) {
    if (!std::isfinite(time.seconds()) || time.seconds() < 0.0) {
        throw std::invalid_argument(std::string{field} +
                                    " must be finite and nonnegative");
    }
}

double milliseconds_between(SimTime end, SimTime start,
                            std::string_view field) {
    require_valid_time(start, "arrived_at");
    require_valid_time(end, field);
    if (end.seconds() < start.seconds()) {
        throw std::invalid_argument(std::string{field} +
                                    " must not precede arrived_at");
    }
    return (end.seconds() - start.seconds()) * 1e3;
}

OrderedJson serialize_execution_time_components(
    const entities::ExecutionTime &execution) {
    return OrderedJson::object({
        {"dense_compute_ms", execution.dense_compute_ms},
        {"lm_head_ms", execution.lm_head_ms},
        {"tp_communication_ms", execution.tp_communication_ms},
        {"pp_communication_ms", execution.pp_communication_ms},
        {"moe_gating_linear_ms", execution.moe_gating_linear_ms},
        {"moe_gating_routing_topk_ms",
         execution.moe_gating_routing_topk_ms},
        {"moe_grouped_gemm_ms", execution.moe_grouped_gemm_ms},
        {"moe_shuffling_ms", execution.moe_shuffling_ms},
        {"moe_post_attention_norm_ms",
         execution.moe_post_attention_norm_ms},
        {"moe_tp_communication_ms", execution.moe_tp_communication_ms},
        {"ep_dispatch_ms", execution.ep_dispatch_ms},
        {"ep_combine_ms", execution.ep_combine_ms},
        {"dp_input_communication_ms", execution.dp_input_communication_ms},
        {"dp_output_communication_ms", execution.dp_output_communication_ms},
        {"synchronization_wait_ms", execution.synchronization_wait_ms},
        {"total_ms", execution.total_ms()},
    });
}

void validate_request_metrics(const std::vector<RequestMetricsRecord> &requests,
                              config::SystemArchitecture architecture) {
    std::unordered_set<std::uint64_t> request_ids;
    request_ids.reserve(requests.size());
    for (const RequestMetricsRecord &request : requests) {
        if (!request_ids.insert(request.request_id.value()).second) {
            throw std::invalid_argument(
                "request metrics contain duplicate request_id=" +
                std::to_string(request.request_id.value()));
        }
        if (request.num_prefill_tokens == 0 || request.num_decode_tokens == 0 ||
            request.preemption_recomputed_prefill_tokens >
                request.scheduled_prefill_tokens ||
            request.prefix_cache_hit_blocks >
                request.prefix_cache_query_blocks ||
            request.cpu_prefix_hit_blocks > request.cpu_prefix_query_blocks ||
            request.cpu_restore_consumed_blocks >
                request.cpu_restore_transferred_blocks ||
            request.cpu_restore_discarded_blocks !=
                request.cpu_restore_transferred_blocks -
                    request.cpu_restore_consumed_blocks ||
            request.cpu_prefix_hit_blocks !=
                request.cpu_restore_consumed_blocks ||
            request.gpu_prefix_hit_blocks + request.cpu_prefix_hit_blocks !=
                request.prefix_cache_hit_blocks ||
            request.cached_prefill_tokens > request.num_prefill_tokens) {
            throw std::invalid_argument(
                "request prefix-cache metrics are invalid");
        }
        static_cast<void>(milliseconds_between(request.prefill_completed_at,
                                               request.arrived_at,
                                               "prefill_completed_at"));
        static_cast<void>(milliseconds_between(
            request.completed_at, request.arrived_at, "completed_at"));
        if (request.completed_at.seconds() <
            request.prefill_completed_at.seconds()) {
            throw std::invalid_argument(
                "completed_at must not precede prefill_completed_at");
        }
        if (!request.first_scheduled_at.valid() ||
            !request.first_token_completed_at.valid()) {
            throw std::invalid_argument(
                "request metrics require canonical scheduling and "
                "first-token timestamps");
        }
        static_cast<void>(milliseconds_between(request.first_scheduled_at,
                                               request.arrived_at,
                                               "first_scheduled_at"));
        static_cast<void>(milliseconds_between(request.first_token_completed_at,
                                               request.arrived_at,
                                               "first_token_completed_at"));
        if (request.first_token_completed_at.seconds() <
            request.prefill_completed_at.seconds()) {
            throw std::invalid_argument(
                "first token completion must not precede prefill completion");
        }
        if (is_pdd(architecture)) {
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
            const SimTime transfer_start = request.kv_cache_transfer_start_time;
            const SimTime transfer_end = request.kv_cache_transfer_end_time;
            const SimTime decode_arrival = request.decode_arrived_at;
            require_valid_time(transfer_start, "kv transfer start");
            require_valid_time(transfer_end, "kv transfer end");
            require_valid_time(decode_arrival, "decode arrival");
            if (transfer_start.seconds() <
                    request.prefill_completed_at.seconds() ||
                transfer_end.seconds() < transfer_start.seconds() ||
                decode_arrival != transfer_end) {
                throw std::invalid_argument(
                    "PDD transfer timestamps are out of order");
            }
        }
    }
}

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
    json["cpu_restore_discarded_blocks"] =
        request.cpu_restore_discarded_blocks;
    json["cpu_restored_tokens"] = request.cpu_restored_tokens;
    json["cpu_restore_bytes"] = request.cpu_restore_bytes;
    json["cpu_restore_queue_time_ms"] =
        request.cpu_restore_queue_time_s * 1e3;
    json["cpu_restore_service_time_ms"] =
        request.cpu_restore_service_time_s * 1e3;
    json["cpu_offload_bytes"] = request.cpu_offload_bytes;
    json["cpu_offload_queue_time_ms"] =
        request.cpu_offload_queue_time_s * 1e3;
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
        execution.synchronization_wait_ms < 0.0) {
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
    if (!routing.batch_id.valid() || !routing.stage_id.valid() ||
        !routing.layer_id.valid() || routing.input_tokens == 0 ||
        routing.routed_tokens == 0 || routing.global_expert_tokens.empty() ||
        routing.lane_expert_tokens.empty() ||
        routing.lane_times_ms.size() != routing.lane_expert_tokens.size() ||
        routing.critical_lane >= routing.lane_times_ms.size() ||
        !std::isfinite(routing.critical_lane_time_ms) ||
        routing.critical_lane_time_ms < 0.0 ||
        std::any_of(
            routing.lane_times_ms.begin(), routing.lane_times_ms.end(),
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
        {"lane_times_ms", routing.lane_times_ms},
        {"critical_lane", routing.critical_lane},
        {"critical_lane_time_ms", routing.critical_lane_time_ms},
    });
    if (routing.sync_group_id.valid()) {
        json["sync_group_id"] = routing.sync_group_id.value();
    }
    return json;
}

} // namespace

std::string_view to_string(EventType event_type) noexcept {
    switch (event_type) {
    case EventType::kRequestArrival:
        return "request_arrival";
    case EventType::kGlobalSchedule:
        return "global_schedule";
    case EventType::kClusterSchedule:
        return "cluster_schedule";
    case EventType::kReplicaSchedule:
        return "replica_schedule";
    case EventType::kBatchStageArrival:
        return "batch_stage_arrival";
    case EventType::kReplicaStageSchedule:
        return "replica_stage_schedule";
    case EventType::kBatchStageEnd:
        return "batch_stage_end";
    case EventType::kClusterBatchEnd:
        return "cluster_batch_end";
    case EventType::kGlobalBatchEnd:
        return "global_batch_end";
    case EventType::kKvCacheTransferStart:
        return "kv_cache_transfer_start";
    case EventType::kKvCacheTransferEnd:
        return "kv_cache_transfer_end";
    case EventType::kPrefillSync:
        return "prefill_sync";
    case EventType::kPrefillSyncCollective:
        return "prefill_sync_collective";
    case EventType::kDecodeSync:
        return "decode_sync";
    case EventType::kDecodeSyncCollective:
        return "decode_sync_collective";
    case EventType::kCpuKvCacheOffloadStart:
        return "cpu_kv_cache_offload_start";
    case EventType::kCpuKvCacheOffloadEnd:
        return "cpu_kv_cache_offload_end";
    case EventType::kCpuKvCacheRestoreStart:
        return "cpu_kv_cache_restore_start";
    case EventType::kCpuKvCacheRestoreEnd:
        return "cpu_kv_cache_restore_end";
    }
    return "unknown";
}

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
        }));
    }

    const CpuKVCacheMetricsAggregate &cpu = output.aggregate.cpu_kv_cache;
    root["cpu_kv_cache"] = OrderedJson::object({
        {"target_count", cpu.target_count},
        {"capacity_bytes", cpu.capacity_bytes},
        {"capacity_blocks", cpu.capacity_blocks},
        {"bytes_per_block", cpu.bytes_per_block},
        {"offload_operations", cpu.offload_operations},
        {"offload_blocks", cpu.offload_blocks},
        {"offload_bytes", cpu.offload_bytes},
        {"restore_operations", cpu.restore_operations},
        {"restore_blocks", cpu.restore_blocks},
        {"restore_bytes", cpu.restore_bytes},
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
        {"stale_generation_completions",
         cpu.stale_generation_completions},
        {"sessions_with_cpu_hits", cpu.sessions_with_cpu_hits},
        {"pending_restore_operations", cpu.pending_restore_operations},
        {"staged_restore_payloads", cpu.staged_restore_payloads},
        {"active_restore_leases", cpu.active_restore_leases},
        {"active_offload_reservations",
         cpu.active_offload_reservations},
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
            {"pending_restore_operations",
             target.pending_restore_operations},
            {"staged_restore_payloads", target.staged_restore_payloads},
            {"active_restore_leases", target.active_restore_leases},
            {"active_offload_reservations",
             target.active_offload_reservations},
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

    return root.dump(2) + '\n';
}

std::string serialize_simulation_summary_json(const SimulationOutput &output,
                                              double wall_clock_seconds) {
    if (!std::isfinite(wall_clock_seconds) || wall_clock_seconds < 0.0) {
        throw std::invalid_argument(
            "summary wall_clock_seconds must be finite and nonnegative");
    }
    validate_request_metrics(output.requests, output.run.system_architecture);

    const auto summarize_values = [](std::vector<double> values) {
        OrderedJson summary = OrderedJson::object();
        summary["count"] = values.size();
        if (values.empty()) {
            summary["mean"] = 0.0;
            summary["p50"] = 0.0;
            summary["p90"] = 0.0;
            summary["p99"] = 0.0;
            return summary;
        }
        std::sort(values.begin(), values.end());
        const double total = std::accumulate(values.begin(), values.end(), 0.0);
        const auto percentile = [&values](double quantile) {
            const double position =
                quantile * static_cast<double>(values.size() - 1);
            const std::size_t lower = static_cast<std::size_t>(position);
            const std::size_t upper = std::min(lower + 1, values.size() - 1);
            const double fraction = position - static_cast<double>(lower);
            return values[lower] + (values[upper] - values[lower]) * fraction;
        };
        summary["mean"] = total / static_cast<double>(values.size());
        summary["p50"] = percentile(0.50);
        summary["p90"] = percentile(0.90);
        summary["p99"] = percentile(0.99);
        return summary;
    };

    std::vector<double> scheduling_delay_ms;
    std::vector<double> prefill_latency_ms;
    std::vector<double> ttft_ms;
    std::vector<double> tpot_ms;
    std::vector<double> e2e_ms;
    scheduling_delay_ms.reserve(output.requests.size());
    prefill_latency_ms.reserve(output.requests.size());
    ttft_ms.reserve(output.requests.size());
    tpot_ms.reserve(output.requests.size());
    e2e_ms.reserve(output.requests.size());

    double first_arrival_s = std::numeric_limits<double>::infinity();
    double last_completion_s = 0.0;
    std::uint64_t total_prefill_tokens = 0;
    std::uint64_t total_decode_tokens = 0;
    std::uint64_t total_scheduled_prefill_tokens = 0;
    std::uint64_t total_recomputed_prefill_tokens = 0;
    std::uint64_t total_preemptions = 0;
    for (const RequestMetricsRecord &request : output.requests) {
        first_arrival_s =
            std::min(first_arrival_s, request.arrived_at.seconds());
        last_completion_s =
            std::max(last_completion_s, request.completed_at.seconds());
        total_prefill_tokens += request.num_prefill_tokens;
        total_decode_tokens += request.num_decode_tokens;
        total_scheduled_prefill_tokens += request.scheduled_prefill_tokens;
        total_recomputed_prefill_tokens +=
            request.preemption_recomputed_prefill_tokens;
        total_preemptions += request.preemption_count;
        scheduling_delay_ms.push_back(
            milliseconds_between(request.first_scheduled_at, request.arrived_at,
                                 "first_scheduled_at"));
        prefill_latency_ms.push_back(
            milliseconds_between(request.prefill_completed_at,
                                 request.arrived_at, "prefill_completed_at"));
        ttft_ms.push_back(milliseconds_between(request.first_token_completed_at,
                                               request.arrived_at,
                                               "first_token_completed_at"));
        e2e_ms.push_back(milliseconds_between(
            request.completed_at, request.arrived_at, "completed_at"));
        if (request.num_decode_tokens > 1) {
            tpot_ms.push_back(
                milliseconds_between(request.completed_at,
                                     request.first_token_completed_at,
                                     "completed_at") /
                static_cast<double>(request.num_decode_tokens - 1));
        }
    }

    const double simulation_window_s =
        output.requests.empty() ? 0.0 : last_completion_s - first_arrival_s;
    const auto rate = [simulation_window_s](std::uint64_t count) {
        return simulation_window_s > 0.0
                   ? static_cast<double>(count) / simulation_window_s
                   : 0.0;
    };

    std::uint64_t aggregate_prefill_scheduled_tokens = 0;
    for (const auto &[cluster_type, aggregate] :
         output.aggregate.batches_by_cluster) {
        static_cast<void>(cluster_type);
        aggregate_prefill_scheduled_tokens +=
            aggregate.prefill_scheduled_tokens;
    }
    // A caller may construct a full SimulationOutput directly (as contract
    // tests and a few analysis tools do) without first feeding batches through
    // MetricsStore.  Preserve a useful summary in that case by deriving the
    // same total from the compact per-batch PREFILL counts.
    if (output.aggregate.batches_by_cluster.empty()) {
        for (const BatchMetricsRecord &batch : output.batches) {
            aggregate_prefill_scheduled_tokens += batch.num_prefill_tokens;
        }
    }

    OrderedJson root = OrderedJson::object();
    root["schema_version"] = 1;
    root["run"] = OrderedJson::object({
        {"run_id", output.run.run_id},
        {"simulation_mode",
         std::string{config::to_string(output.run.simulation_mode)}},
        {"system_architecture",
         std::string{config::to_string(output.run.system_architecture)}},
    });
    root["wall_clock_seconds"] = wall_clock_seconds;
    root["simulation_window_seconds"] = simulation_window_s;
    root["counts"] = OrderedJson::object({
        {"requests", output.requests.size()},
        {"batches", output.aggregate.batch_count},
        {"batch_stages", output.aggregate.batch_stage_count},
        {"scheduler_iterations", output.aggregate.scheduler_iteration_count},
        {"events", output.aggregate.event_count},
        {"kv_cache_transfers", output.aggregate.kv_cache_transfer_count},
        {"preemptions", total_preemptions},
        {"prefill_scheduled_tokens", aggregate_prefill_scheduled_tokens},
    });
    root["prefill_work"] = OrderedJson::object({
        {"scheduled_prefill_tokens", total_scheduled_prefill_tokens},
        {"preemption_recomputed_prefill_tokens",
         total_recomputed_prefill_tokens},
    });
    root["throughput"] = OrderedJson::object({
        {"requests_per_second", rate(output.requests.size())},
        {"prompt_tokens_per_second", rate(total_prefill_tokens)},
        {"decode_tokens_per_second", rate(total_decode_tokens)},
        {"total_tokens_per_second",
         rate(total_prefill_tokens + total_decode_tokens)},
    });
    root["latency_ms"] = OrderedJson::object({
        {"scheduling_delay", summarize_values(std::move(scheduling_delay_ms))},
        {"prefill", summarize_values(std::move(prefill_latency_ms))},
        {"ttft", summarize_values(std::move(ttft_ms))},
        {"tpot", summarize_values(std::move(tpot_ms))},
        {"e2e", summarize_values(std::move(e2e_ms))},
    });

    root["batch_summary_by_cluster"] = OrderedJson::object();
    for (const auto &[cluster_type, aggregate] :
         output.aggregate.batches_by_cluster) {
        OrderedJson histogram = OrderedJson::object();
        for (const auto &[batch_size, count] : aggregate.batch_size_histogram) {
            histogram[std::to_string(batch_size)] = count;
        }
        root["batch_summary_by_cluster"][std::string{to_string(cluster_type)}] =
            OrderedJson::object({
                {"batch_count", aggregate.batch_count},
                {"mean_batch_size",
                 aggregate.batch_count == 0
                     ? 0.0
                     : static_cast<double>(aggregate.request_slots) /
                           static_cast<double>(aggregate.batch_count)},
                 {"predicted_execution_ms", aggregate.predicted_execution_ms},
                 {"execution_time_components_ms",
                  serialize_execution_time_components(
                      aggregate.execution_time)},
                 {"prefill_scheduled_tokens",
                 aggregate.prefill_scheduled_tokens},
                {"preemption_recomputed_prefill_tokens",
                 aggregate.preemption_recomputed_prefill_tokens},
                {"batch_size_histogram", std::move(histogram)},
            });
    }

    constexpr std::uint64_t kBatchTimeBucketSeconds = 60;
    root["batch_time_bucket_seconds"] = kBatchTimeBucketSeconds;
    root["batch_summary_by_cluster_time_bucket"] = OrderedJson::object();
    for (const auto &[cluster_type, buckets] :
         output.aggregate.batch_time_buckets_by_cluster) {
        OrderedJson values = OrderedJson::array();
        for (const auto &[bucket_index, aggregate] : buckets) {
            values.push_back(OrderedJson::object({
                {"start_time_s", bucket_index * kBatchTimeBucketSeconds},
                {"end_time_s", (bucket_index + 1) * kBatchTimeBucketSeconds},
                {"batch_count", aggregate.batch_count},
                {"mean_batch_size",
                 aggregate.batch_count == 0
                     ? 0.0
                     : static_cast<double>(aggregate.request_slots) /
                           static_cast<double>(aggregate.batch_count)},
                 {"predicted_execution_ms", aggregate.predicted_execution_ms},
                 {"execution_time_components_ms",
                  serialize_execution_time_components(
                      aggregate.execution_time)},
             }));
        }
        root["batch_summary_by_cluster_time_bucket"]
            [std::string{to_string(cluster_type)}] = std::move(values);
    }

    std::vector<double> transfer_latency_ms;
    std::uint64_t transfer_bytes = 0;
    transfer_latency_ms.reserve(output.kv_cache_transfers.size());
    for (const KVCacheTransferMetricsRecord &transfer :
         output.kv_cache_transfers) {
        transfer_latency_ms.push_back(transfer.predicted_time_ms);
        transfer_bytes += transfer.size_bytes;
    }
    root["kv_cache_transfer"] = OrderedJson::object({
        {"total_bytes", transfer_bytes},
        {"latency_ms", summarize_values(std::move(transfer_latency_ms))},
    });

    const PrefixCacheMetricsAggregate &cache = output.aggregate.prefix_cache;
    root["prefix_cache"] = OrderedJson::object({
        {"query_blocks", cache.query_blocks},
        {"hit_blocks", cache.hit_blocks},
        {"hit_rate", cache.query_blocks == 0
                         ? 0.0
                         : static_cast<double>(cache.hit_blocks) /
                               static_cast<double>(cache.query_blocks)},
        {"evicted_blocks", cache.evicted_blocks},
        {"evicted_sessions", cache.evicted_sessions},
    });
    const CpuKVCacheMetricsAggregate &cpu = output.aggregate.cpu_kv_cache;
    root["cpu_kv_cache"] = OrderedJson::object({
        {"target_count", cpu.target_count},
        {"capacity_bytes", cpu.capacity_bytes},
        {"capacity_blocks", cpu.capacity_blocks},
        {"bytes_per_block", cpu.bytes_per_block},
        {"offload_operations", cpu.offload_operations},
        {"offload_blocks", cpu.offload_blocks},
        {"offload_bytes", cpu.offload_bytes},
        {"restore_operations", cpu.restore_operations},
        {"restore_blocks", cpu.restore_blocks},
        {"restore_bytes", cpu.restore_bytes},
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
        {"stale_generation_completions",
         cpu.stale_generation_completions},
        {"sessions_with_cpu_hits", cpu.sessions_with_cpu_hits},
        {"pending_restore_operations", cpu.pending_restore_operations},
        {"staged_restore_payloads", cpu.staged_restore_payloads},
        {"active_restore_leases", cpu.active_restore_leases},
        {"active_offload_reservations",
         cpu.active_offload_reservations},
    });
    return root.dump(2) + '\n';
}

std::string
serialize_request_metrics_csv(const std::vector<RequestMetricsRecord> &requests,
                              config::SystemArchitecture architecture) {
    validate_request_metrics(requests, architecture);

    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << std::setprecision(std::numeric_limits<double>::max_digits10);
    if (!is_pdd(architecture)) {
        output << "request_id,session_id,num_prefill_tokens,num_decode_tokens,"
                  "cached_prefill_tokens,prefix_cache_query_blocks,"
                  "prefix_cache_hit_blocks,gpu_prefix_hit_blocks,"
                  "cpu_prefix_query_blocks,cpu_prefix_hit_blocks,"
                  "cpu_restore_transferred_blocks,cpu_restore_consumed_blocks,"
                  "cpu_restore_discarded_blocks,cpu_restored_tokens,"
                  "cpu_restore_bytes,cpu_restore_queue_time_ms,"
                  "cpu_restore_service_time_ms,cpu_offload_bytes,"
                  "cpu_offload_queue_time_ms,cpu_offload_service_time_ms,"
                  "prefix_cache_key_mode,arrived_at_s,"
                  "first_scheduled_at_s,"
                  "prefill_completed_at_s,first_token_completed_at_s,"
                  "completed_at_s,scheduling_delay_ms,prefill_latency_ms,"
                  "ttft_ms,e2e_ms,"
                  "num_processed_tokens,preemption_count,replica_id,dp_id,"
                  "scheduled_prefill_tokens,"
                  "preemption_recomputed_prefill_tokens\n";
    } else {
        output << "request_id,session_id,num_prefill_tokens,num_decode_tokens,"
                  "cached_prefill_tokens,prefix_cache_query_blocks,"
                  "prefix_cache_hit_blocks,gpu_prefix_hit_blocks,"
                  "cpu_prefix_query_blocks,cpu_prefix_hit_blocks,"
                  "cpu_restore_transferred_blocks,cpu_restore_consumed_blocks,"
                  "cpu_restore_discarded_blocks,cpu_restored_tokens,"
                  "cpu_restore_bytes,cpu_restore_queue_time_ms,"
                  "cpu_restore_service_time_ms,cpu_offload_bytes,"
                  "cpu_offload_queue_time_ms,cpu_offload_service_time_ms,"
                  "prefix_cache_key_mode,arrived_at_s,"
                  "first_scheduled_at_s,"
                  "prefill_completed_at_s,first_token_completed_at_s,"
                  "completed_at_s,scheduling_delay_ms,prefill_latency_ms,"
                  "ttft_ms,e2e_ms,"
                  "num_processed_tokens,preemption_count,"
                  "prefill_replica_id,prefill_dp_id,decode_replica_id,decode_"
                  "dp_id,"
                  "transfer_id,kv_cache_transfer_start_time_s,"
                  "kv_cache_transfer_end_time_s,kv_cache_transfer_time_ms,"
                  "kv_cache_transfer_size_bytes,decode_arrived_at_s,"
                  "scheduled_prefill_tokens,"
                  "preemption_recomputed_prefill_tokens\n";
    }

    for (const RequestMetricsRecord &request : requests) {
        output << request.request_id.value() << ',';
        if (request.session_id.valid()) {
            output << request.session_id.value();
        }
        output << ',' << request.num_prefill_tokens << ','
               << request.num_decode_tokens << ','
               << request.cached_prefill_tokens << ','
               << request.prefix_cache_query_blocks << ','
               << request.prefix_cache_hit_blocks << ','
               << request.gpu_prefix_hit_blocks << ','
               << request.cpu_prefix_query_blocks << ','
               << request.cpu_prefix_hit_blocks << ','
               << request.cpu_restore_transferred_blocks << ','
               << request.cpu_restore_consumed_blocks << ','
               << request.cpu_restore_discarded_blocks << ','
               << request.cpu_restored_tokens << ','
               << request.cpu_restore_bytes << ','
               << request.cpu_restore_queue_time_s * 1e3 << ','
               << request.cpu_restore_service_time_s * 1e3 << ','
               << request.cpu_offload_bytes << ','
               << request.cpu_offload_queue_time_s * 1e3 << ','
               << request.cpu_offload_service_time_s * 1e3 << ','
               << config::to_string(request.prefix_cache_key_mode) << ','
               << request.arrived_at.seconds() << ',';
        output << request.first_scheduled_at.seconds() << ',';
        output << request.prefill_completed_at.seconds() << ',';
        output << request.first_token_completed_at.seconds() << ',';
        output << request.completed_at.seconds() << ',';
        output << milliseconds_between(request.first_scheduled_at,
                                       request.arrived_at, "first_scheduled_at")
               << ',';
        output << milliseconds_between(request.prefill_completed_at,
                                       request.arrived_at,
                                       "prefill_completed_at")
               << ',';
        output << milliseconds_between(request.first_token_completed_at,
                                       request.arrived_at,
                                       "first_token_completed_at")
               << ','
               << milliseconds_between(request.completed_at, request.arrived_at,
                                       "completed_at");
        output << ',' << request.num_processed_tokens << ','
               << request.preemption_count;
        if (!is_pdd(architecture)) {
            output << ',' << request.replica_id.value() << ','
                   << request.dp_id.value() << ','
                   << request.scheduled_prefill_tokens << ','
                   << request.preemption_recomputed_prefill_tokens;
        } else {
            output << ',' << request.prefill_replica_id.value() << ','
                   << request.prefill_dp_id.value() << ','
                   << request.decode_replica_id.value() << ','
                   << request.decode_dp_id.value() << ','
                   << request.transfer_id.value() << ','
                   << request.kv_cache_transfer_start_time.seconds() << ','
                   << request.kv_cache_transfer_end_time.seconds() << ','
                   << (request.kv_cache_transfer_end_time.seconds() -
                       request.kv_cache_transfer_start_time.seconds()) *
                          1e3
                   << ',' << request.kv_cache_transfer_size_bytes << ','
                   << request.decode_arrived_at.seconds() << ','
                   << request.scheduled_prefill_tokens << ','
                   << request.preemption_recomputed_prefill_tokens;
        }
        output << '\n';
    }
    return output.str();
}

std::string serialize_gpu_kv_occupancy_csv(
    const std::vector<GpuKVCacheOccupancyRecord> &occupancy) {
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << std::setprecision(std::numeric_limits<double>::max_digits10);
    output << "time_s,cluster_type,replica_id,dp_id,active_blocks,"
              "capacity_blocks,active_bytes_per_gpu,hbm_fraction,"
              "active_fraction_of_kv_budget,active_fraction_of_total_hbm\n";

    std::map<std::tuple<ClusterType, ReplicaId, DataParallelId>, SimTime>
        last_time;
    for (const GpuKVCacheOccupancyRecord &record : occupancy) {
        require_valid_time(record.time, "gpu_kv_occupancy.time");
        if (!record.replica_id.valid() || !record.dp_id.valid()) {
            throw std::invalid_argument(
                "gpu KV occupancy target IDs must be valid");
        }
        if (record.active_blocks > record.capacity_blocks ||
            !std::isfinite(record.active_fraction_of_kv_budget) ||
            record.active_fraction_of_kv_budget < 0.0 ||
            record.active_fraction_of_kv_budget > 1.0) {
            throw std::invalid_argument(
                "gpu KV occupancy active fraction is invalid");
        }
        if (record.hbm_fraction.has_value() &&
            (!std::isfinite(record.hbm_fraction.value()) ||
             record.hbm_fraction.value() < 0.0 ||
             record.hbm_fraction.value() > 1.0)) {
            throw std::invalid_argument(
                "gpu KV occupancy total-HBM fraction is invalid");
        }
        if (record.active_fraction_of_total_hbm.has_value() &&
            (!std::isfinite(record.active_fraction_of_total_hbm.value()) ||
             record.active_fraction_of_total_hbm.value() < 0.0 ||
             record.active_fraction_of_total_hbm.value() > 1.0)) {
            throw std::invalid_argument(
                "gpu KV occupancy total-HBM fraction is invalid");
        }
        if (record.hbm_fraction.has_value() &&
            record.active_fraction_of_total_hbm.has_value() &&
            record.hbm_fraction.value() !=
                record.active_fraction_of_total_hbm.value()) {
            throw std::invalid_argument(
                "gpu KV occupancy total-HBM fractions disagree");
        }
        const auto key = std::make_tuple(record.cluster_type,
                                         record.replica_id, record.dp_id);
        const auto previous = last_time.find(key);
        if (previous != last_time.end() && record.time < previous->second) {
            throw std::invalid_argument(
                "gpu KV occupancy samples are not time ordered");
        }
        last_time[key] = record.time;
        output << record.time.seconds() << ',' << to_string(record.cluster_type)
               << ',' << record.replica_id.value() << ','
               << record.dp_id.value() << ',' << record.active_blocks << ','
               << record.capacity_blocks << ','
               << record.active_bytes_per_gpu << ',';
        if (record.hbm_fraction.has_value()) {
            output << record.hbm_fraction.value();
        }
        output << ',' << record.active_fraction_of_kv_budget << ',';
        if (record.active_fraction_of_total_hbm.has_value()) {
            output << record.active_fraction_of_total_hbm.value();
        }
        output << '\n';
    }
    return output.str();
}

} // namespace frontier::metrics
