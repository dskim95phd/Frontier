#include "frontier/metrics/output_contract.h"

#include <stdexcept>
#include <string>
#include <utility>

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
using output_detail::serialize_batch;
using output_detail::serialize_batch_stage;
using output_detail::serialize_diagnostic;
using output_detail::serialize_event;
using output_detail::serialize_execution_time_components;
using output_detail::serialize_kv_cache_transfer;
using output_detail::serialize_moe_routing;
using output_detail::serialize_request;
using output_detail::serialize_scheduler_trace;
using output_detail::validate_request_metrics;

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
    if (output.run.prefill_only) {
        root["run"]["prefill_only"] = true;
        root["run"]["synthetic_decode_tokens_per_second"] =
            output.run.synthetic_decode_tokens_per_second;
    }

    root["completed_request_ids"] = OrderedJson::array();
    root["requests"] = OrderedJson::array();
    for (const RequestMetricsRecord &request : output.requests) {
        root["completed_request_ids"].push_back(request.request_id.value());
        root["requests"].push_back(
            serialize_request(request, output.run.system_architecture));
    }
    if (output.run.prefill_only) {
        root["prefill_completions"] = OrderedJson::array();
        for (const PrefillCompletionMetricsRecord &prefill :
             output.prefill_completions) {
            if (!prefill.request_id.valid() ||
                !prefill.arrived_at.valid() ||
                !prefill.completed_at.valid() ||
                prefill.completed_at < prefill.arrived_at ||
                !prefill.replica_id.valid() || !prefill.dp_id.valid()) {
                throw std::invalid_argument(
                    "invalid PREFILL completion metrics record");
            }
            root["prefill_completions"].push_back(OrderedJson::object({
                {"request_id", prefill.request_id.value()},
                {"num_prefill_tokens", prefill.num_prefill_tokens},
                {"scheduled_prefill_tokens",
                 prefill.scheduled_prefill_tokens},
                {"preemption_recomputed_prefill_tokens",
                 prefill.preemption_recomputed_prefill_tokens},
                {"arrived_at_s", prefill.arrived_at.seconds()},
                {"completed_at_s", prefill.completed_at.seconds()},
                {"prefill_latency_ms",
                 (prefill.completed_at.seconds() -
                  prefill.arrived_at.seconds()) *
                     1e3},
                {"replica_id", prefill.replica_id.value()},
                {"dp_id", prefill.dp_id.value()},
            }));
        }
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
