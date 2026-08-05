#include "frontier/metrics/metrics_store.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "frontier/entities/batch.h"
#include "frontier/entities/batch_stage.h"
#include "frontier/entities/kv_cache_transfer_info.h"
#include "frontier/entities/cpu_kv_cache_transfer_info.h"
#include "frontier/entities/request.h"
#include "frontier/execution_time_predictor/base_execution_time_predictor.h"
#include "frontier/kv_cache/replica_kv_cache_manager.h"
#include "frontier/kv_cache/cpu_kv_cache_manager.h"
#include "frontier/scheduler/replica_scheduler/base_replica_scheduler.h"
#include "frontier/scheduler/scheduler_types.h"
#include "frontier/simulator/entity_arena.h"

namespace frontier::metrics {
namespace {

void accumulate_execution_time(entities::ExecutionTime &total,
                               const entities::ExecutionTime &value) {
    total.dense_compute_ms += value.dense_compute_ms;
    total.lm_head_ms += value.lm_head_ms;
    total.tp_communication_ms += value.tp_communication_ms;
    total.pp_communication_ms += value.pp_communication_ms;
    total.moe_gating_linear_ms += value.moe_gating_linear_ms;
    total.moe_gating_routing_topk_ms += value.moe_gating_routing_topk_ms;
    total.moe_grouped_gemm_ms += value.moe_grouped_gemm_ms;
    total.moe_shuffling_ms += value.moe_shuffling_ms;
    total.moe_post_attention_norm_ms += value.moe_post_attention_norm_ms;
    total.moe_tp_communication_ms += value.moe_tp_communication_ms;
    total.ep_dispatch_ms += value.ep_dispatch_ms;
    total.ep_combine_ms += value.ep_combine_ms;
    total.dp_input_communication_ms += value.dp_input_communication_ms;
    total.dp_output_communication_ms += value.dp_output_communication_ms;
    total.synchronization_wait_ms += value.synchronization_wait_ms;
}

} // namespace

MetricsStore::MetricsStore(const config::SimulationConfig &config,
                           std::size_t expected_request_count)
    : output_([&]() {
          SimulationOutput value{};
          value.schema_version = config.schema_version;
          value.run = [&]() {
              RunMetadata value{};
              value.run_id = config.run_id;
              value.simulation_mode = config.simulation_mode;
              value.system_architecture = config.system_architecture;
              return value;
          }();
          return value;
      }()) {
    output_.requests.reserve(expected_request_count);
    output_.event_trace.reserve(expected_request_count * 12);
}

void MetricsStore::record_event(Event event) {
    ++output_.aggregate.event_count;
    if (detailed_traces_enabled_) {
        output_.event_trace.push_back(std::move(event));
    }
}

void MetricsStore::record_request(RequestMetricsRecord record) {
    output_.requests.push_back(std::move(record));
}

void MetricsStore::record_batch(const entities::Batch &batch,
                                const std::vector<entities::Request> &requests,
                                double predicted_execution_ms,
                                const config::ClusterRuntimeConfig &runtime) {
    if (!batch.completed_at().valid()) {
        throw std::logic_error("cannot emit metrics for incomplete batch");
    }
    const std::size_t batch_size = batch.requests().size();
    BatchMetricsAggregate &aggregate =
        output_.aggregate.batches_by_cluster[batch.cluster_type()];
    ++output_.aggregate.batch_count;
    ++aggregate.batch_count;
    aggregate.request_slots += batch_size;
    aggregate.predicted_execution_ms += predicted_execution_ms;
    aggregate.batch_size_execution_ms +=
        static_cast<double>(batch_size) * predicted_execution_ms;
    ++aggregate.batch_size_histogram[batch_size];
    constexpr double kBatchTimeBucketSeconds = 60.0;
    const auto bucket_index = static_cast<std::uint64_t>(
        std::floor(batch.scheduled_at().seconds() / kBatchTimeBucketSeconds));
    BatchTimeBucketAggregate &time_bucket =
        output_.aggregate.batch_time_buckets_by_cluster[batch.cluster_type()]
                                                        [bucket_index];
    ++time_bucket.batch_count;
    time_bucket.request_slots += batch_size;
    time_bucket.predicted_execution_ms += predicted_execution_ms;
    // Count only work that was part of a PREFILL phase at scheduling time.
    // `processed_tokens` is captured in the batch snapshot before execution;
    // comparing it with the request's current replay boundary also handles
    // monolithic preemption, where committed decode context is replayed as
    // PREFILL.  Cached prefix tokens are already reflected in
    // `processed_tokens`, so they never enter this sum.
    for (const entities::RequestBatchSnapshot &snapshot : batch.requests()) {
        const entities::Request &request =
            requests.at(static_cast<std::size_t>(snapshot.request_id.value()));
        if (snapshot.processed_tokens < request.num_prefill_tokens()) {
            if (aggregate.prefill_scheduled_tokens >
                std::numeric_limits<std::uint64_t>::max() -
                    snapshot.scheduled_tokens) {
                throw std::overflow_error(
                    "aggregate PREFILL token count overflows uint64");
            }
            aggregate.prefill_scheduled_tokens += snapshot.scheduled_tokens;
        }
    }
    if (!detailed_traces_enabled_) {
        return;
    }
    BatchMetricsRecord record = [&]() {
        BatchMetricsRecord value{};
        value.batch_id = batch.id();
        value.iteration_id = batch.iteration_id();
        value.scheduled_at = batch.scheduled_at();
        value.completed_at = batch.completed_at();
        value.request_ids = {};
        value.scheduled_tokens = {};
        value.total_scheduled_tokens = batch.total_scheduled_tokens();
        value.num_prefill_tokens = 0;
        value.num_decode_tokens = 0;
        value.predicted_execution_ms = predicted_execution_ms;
        value.replica_id = batch.replica_id();
        value.dp_id = batch.dp_id();
        value.num_pipeline_stages = batch.num_pipeline_stages();
        value.cluster_type = batch.cluster_type();
        value.model_kind = batch.model_kind();
        value.runtime_total_experts = runtime.model.total_expert_num;
        value.router_topk = runtime.model.router_topk;
        value.moe_sync_group_id = batch.moe_sync_group_id();
        value.parallelism = runtime.parallelism;
        return value;
    }();
    for (const entities::RequestBatchSnapshot &snapshot : batch.requests()) {
        const entities::Request &request =
            requests.at(static_cast<std::size_t>(snapshot.request_id.value()));
        record.request_ids.push_back(snapshot.request_id);
        record.scheduled_tokens.push_back(snapshot.scheduled_tokens);
        if (snapshot.processed_tokens < request.num_prefill_tokens()) {
            record.num_prefill_tokens += snapshot.scheduled_tokens;
        } else {
            record.num_decode_tokens += snapshot.scheduled_tokens;
        }
    }
    output_.batches.push_back(std::move(record));
}

void MetricsStore::record_batch_stage(
    const entities::BatchStage &batch_stage, const entities::Batch &batch,
    const config::ClusterRuntimeConfig &runtime) {
    ++output_.aggregate.batch_stage_count;
    BatchMetricsAggregate &aggregate =
        output_.aggregate.batches_by_cluster[batch.cluster_type()];
    accumulate_execution_time(aggregate.execution_time,
                              batch_stage.execution_time());
    constexpr double kBatchTimeBucketSeconds = 60.0;
    const auto bucket_index = static_cast<std::uint64_t>(
        std::floor(batch.scheduled_at().seconds() / kBatchTimeBucketSeconds));
    BatchTimeBucketAggregate &time_bucket =
        output_.aggregate.batch_time_buckets_by_cluster[batch.cluster_type()]
                                                        [bucket_index];
    accumulate_execution_time(time_bucket.execution_time,
                              batch_stage.execution_time());
    if (!detailed_traces_enabled_) {
        return;
    }
    BatchStageMetricsRecord record = [&]() {
        BatchStageMetricsRecord value{};
        value.batch_id = batch_stage.batch_id();
        value.replica_id = batch.replica_id();
        value.dp_id = batch.dp_id();
        value.stage_id = batch_stage.stage_id();
        value.arrived_at = batch_stage.arrived_at();
        value.started_at = batch_stage.started_at();
        value.completed_at = batch_stage.completed_at();
        value.execution_time = batch_stage.execution_time();
        value.cluster_type = batch.cluster_type();
        value.model_kind = batch.model_kind();
        value.runtime_total_experts = runtime.model.total_expert_num;
        value.router_topk = runtime.model.router_topk;
        value.moe_sync_group_id = batch.moe_sync_group_id();
        value.parallelism = runtime.parallelism;
        return value;
    }();
    output_.batch_stages.push_back(std::move(record));
}

void MetricsStore::record_scheduler_trace(
    const scheduler::ScheduleResult &schedule, scheduler::ReplicaTarget target,
    ClusterType cluster_type) {
    ++output_.aggregate.scheduler_iteration_count;
    if (!detailed_traces_enabled_) {
        return;
    }
    SchedulerTraceRecord trace = [&]() {
        SchedulerTraceRecord value{};
        value.iteration_id = schedule.iteration_id;
        value.simulation_time = schedule.simulation_time;
        value.token_budget_before = schedule.token_budget_before;
        value.token_budget_after = schedule.token_budget_after;
        value.available_blocks_before = schedule.available_blocks_before;
        value.available_blocks_after = schedule.available_blocks_after;
        value.waiting_count_before = schedule.waiting_count_before;
        value.waiting_count_after = schedule.waiting_count_after;
        value.running_count_before = schedule.running_count_before;
        value.running_count_after = schedule.running_count_after;
        value.preempted_count = schedule.preempted_count;
        value.decisions = {};
        value.batch_request_ids = {};
        value.request_num_tokens = {};
        value.replica_id = target.replica_id;
        value.dp_id = target.dp_id;
        value.cluster_type = cluster_type;
        return value;
    }();
    for (const scheduler::SchedulerDecision &decision : schedule.decisions) {
        trace.decisions.push_back([&]() {
            SchedulerDecisionRecord value{};
            value.decision_result =
                std::string{scheduler::to_string(decision.type)};
            value.request_id = decision.request_id;
            value.num_tokens = decision.num_tokens;
            value.token_budget_after = decision.token_budget_after;
            value.available_blocks_after = decision.available_blocks_after;
            return value;
        }());
    }
    for (const scheduler::ScheduledRequest &scheduled :
         schedule.scheduled_requests) {
        trace.batch_request_ids.push_back(scheduled.request_id);
        trace.request_num_tokens.push_back(scheduled.num_tokens);
    }
    output_.scheduler_trace.push_back(std::move(trace));
}

void MetricsStore::record_kv_cache_transfer(
    const entities::KVCacheTransferInfo &transfer) {
    ++output_.aggregate.kv_cache_transfer_count;
    KVCacheTransferMetricsRecord record = [&]() {
        KVCacheTransferMetricsRecord value{};
        value.transfer_id = transfer.id();
        value.request_id = transfer.request_id();
        value.source_batch_id = transfer.source_batch_id();
        value.source_cluster_type = transfer.source_cluster_type();
        value.target_cluster_type = transfer.target_cluster_type();
        value.source_replica_id = transfer.source_replica_id();
        value.source_dp_id = transfer.source_dp_id();
        value.size_bytes = transfer.size_bytes();
        value.predicted_time_ms = transfer.predicted_time_ms();
        value.started_at = transfer.started_at();
        value.completed_at = transfer.completed_at();
        return value;
    }();
    output_.kv_cache_transfers.push_back(std::move(record));
}

void MetricsStore::record_analytical_diagnostic(
    std::string name, std::vector<std::pair<std::string, double>> values) {
    ++output_.aggregate.analytical_diagnostic_count;
    if (!detailed_traces_enabled_) {
        return;
    }
    output_.analytical_diagnostics.push_back([&]() {
        AnalyticalDiagnostic value{};
        value.name = std::move(name);
        value.values = std::move(values);
        return value;
    }());
}

void MetricsStore::record_moe_routing(
    const entities::Batch &batch, StageId stage_id,
    const execution_time_predictor::MoERoutingDiagnostic &diagnostic,
    const config::ClusterRuntimeConfig &runtime) {
    ++output_.aggregate.moe_routing_count;
    if (!detailed_traces_enabled_) {
        return;
    }
    output_.moe_routing.push_back([&]() {
        MoERoutingMetricsRecord value{};
        value.batch_id = batch.id();
        value.stage_id = stage_id;
        value.cluster_type = batch.cluster_type();
        value.sync_group_id = batch.moe_sync_group_id();
        value.layer_id = diagnostic.layer_id;
        value.mode = runtime.moe_routing.mode;
        value.distribution = runtime.moe_routing.distribution;
        value.seed = runtime.moe_routing.seed;
        value.input_tokens = diagnostic.input_tokens;
        value.routed_tokens = diagnostic.routed_tokens;
        value.global_expert_tokens = diagnostic.global_expert_tokens;
        value.lane_expert_tokens = diagnostic.lane_expert_tokens;
        value.lane_times_ms = diagnostic.lane_times_ms;
        value.critical_lane = diagnostic.critical_lane;
        value.critical_lane_time_ms = diagnostic.critical_lane_time_ms;
        return value;
    }());
}

void MetricsStore::collect_completed_requests(
    const config::SimulationConfig &config,
    const simulator::EntityArena &entities) {
    if (!output_.requests.empty()) {
        throw std::logic_error("request metrics were collected more than once");
    }
    const bool is_pdd = config.system_architecture ==
                        config::SystemArchitecture::kPdDisaggregation;
    for (const RequestId request_id : entities.completion_order()) {
        const entities::Request &request = entities.request(request_id);
        if (!request.completed() || !request.first_scheduled_at().valid() ||
            !request.prefill_completed_at().valid() ||
            !request.first_token_completed_at().valid() ||
            !request.completed_at().valid()) {
            throw std::runtime_error(
                "completed request is missing canonical metrics");
        }
        const scheduler::ReplicaTarget target = entities.request_target(
            request_id,
            is_pdd ? ClusterType::kDecode : ClusterType::kMonolithic);
        RequestMetricsRecord record = [&]() {
            RequestMetricsRecord value{};
            value.request_id = request_id;
            value.session_id = request.session_id();
            value.num_prefill_tokens = request.initial_num_prefill_tokens();
            value.num_decode_tokens = request.initial_num_decode_tokens();
            value.scheduled_prefill_tokens =
                request.scheduled_prefill_tokens();
            value.preemption_recomputed_prefill_tokens =
                request.preemption_recomputed_prefill_tokens();
            value.cached_prefill_tokens = request.cached_prefill_tokens();
            value.prefix_cache_query_blocks =
                request.prefix_cache_query_blocks();
            value.prefix_cache_hit_blocks = request.prefix_cache_hit_blocks();
            value.gpu_prefix_hit_blocks = request.gpu_prefix_hit_blocks();
            value.cpu_prefix_query_blocks = request.cpu_prefix_query_blocks();
            value.cpu_prefix_hit_blocks = request.cpu_prefix_hit_blocks();
            value.cpu_restore_transferred_blocks =
                request.cpu_restore_transferred_blocks();
            value.cpu_restore_consumed_blocks =
                request.cpu_restore_consumed_blocks();
            value.cpu_restore_discarded_blocks =
                request.cpu_restore_discarded_blocks();
            value.cpu_restored_tokens = request.cpu_restored_tokens();
            value.cpu_restore_bytes = request.cpu_restore_bytes();
            value.cpu_restore_queue_time_s =
                request.cpu_restore_queue_time_s();
            value.cpu_restore_service_time_s =
                request.cpu_restore_service_time_s();
            value.cpu_offload_bytes = request.cpu_offload_bytes();
            value.cpu_offload_queue_time_s =
                request.cpu_offload_queue_time_s();
            value.cpu_offload_service_time_s =
                request.cpu_offload_service_time_s();
            value.prefix_cache_key_mode = request.prefix_cache_key_mode();
            value.arrived_at = request.arrived_at();
            value.prefill_completed_at = request.prefill_completed_at();
            value.completed_at = request.completed_at();
            value.first_scheduled_at = request.first_scheduled_at();
            value.first_token_completed_at = request.first_token_completed_at();
            value.num_processed_tokens = request.num_processed_tokens();
            value.preemption_count = request.preemption_count();
            value.tokens_at_preemption = {};
            value.replica_id = target.replica_id;
            value.dp_id = target.dp_id;
            value.prefill_replica_id = ReplicaId{};
            value.prefill_dp_id = DataParallelId{};
            value.decode_replica_id = ReplicaId{};
            value.decode_dp_id = DataParallelId{};
            value.transfer_id = TransferId{};
            value.kv_cache_transfer_start_time = SimTime{};
            value.kv_cache_transfer_end_time = SimTime{};
            value.decode_arrived_at = SimTime{};
            value.kv_cache_transfer_size_bytes = 0;
            return value;
        }();
        if (is_pdd) {
            const scheduler::ReplicaTarget prefill =
                entities.request_target(request_id, ClusterType::kPrefill);
            const TransferId transfer_id =
                entities.request_transfer_id(request_id);
            record.prefill_replica_id = prefill.replica_id;
            record.prefill_dp_id = prefill.dp_id;
            record.decode_replica_id = target.replica_id;
            record.decode_dp_id = target.dp_id;
            record.transfer_id = transfer_id;
            record.kv_cache_transfer_start_time =
                request.kv_cache_transfer_start_time();
            record.kv_cache_transfer_end_time =
                request.kv_cache_transfer_end_time();
            record.decode_arrived_at = request.decode_arrived_at();
            record.kv_cache_transfer_size_bytes =
                request.kv_cache_transfer_size_bytes();
        }
        // Replay attribution is request-owned (it spans batches and can
        // include work after a monolithic decode preemption).  Fold it into
        // the logical PREFILL cluster once, when completed requests are
        // collected, so compact output does not depend on detailed batch
        // traces being retained.
        const ClusterType prefill_cluster =
            is_pdd ? ClusterType::kPrefill : ClusterType::kMonolithic;
        BatchMetricsAggregate &prefill_aggregate =
            output_.aggregate.batches_by_cluster[prefill_cluster];
        if (prefill_aggregate.preemption_recomputed_prefill_tokens >
            std::numeric_limits<std::uint64_t>::max() -
                request.preemption_recomputed_prefill_tokens()) {
            throw std::overflow_error(
                "aggregate recomputed PREFILL token count overflows uint64");
        }
        prefill_aggregate.preemption_recomputed_prefill_tokens +=
            request.preemption_recomputed_prefill_tokens();
        record_request(std::move(record));
    }
}

void MetricsStore::record_prefix_cache_target(
    const kv_cache::PrefixCacheStats &stats,
    const kv_cache::PrefixCacheDiagnostics &diagnostics,
    scheduler::ReplicaTarget target, ClusterType cluster_type,
    std::uint64_t block_size, config::PrefixCachingKeyMode key_mode) {
    PrefixCacheMetricsAggregate &aggregate = output_.aggregate.prefix_cache;
    if (aggregate.block_size != 0 && aggregate.block_size != block_size) {
        throw std::logic_error("prefix-cache targets disagree on block size");
    }
    aggregate.key_mode = key_mode;
    aggregate.block_size = block_size;
    aggregate.successful_admissions += stats.successful_admissions;
    aggregate.query_blocks += stats.query_blocks;
    aggregate.hit_blocks += stats.hit_blocks;
    aggregate.evicted_blocks += stats.evicted_blocks;
    aggregate.evicted_sessions += stats.evicted_sessions;
    output_.prefix_cache_targets.push_back(PrefixCacheTargetMetricsRecord{
        cluster_type, target.replica_id, target.dp_id,
        diagnostics.capacity_blocks, diagnostics.available_blocks,
        diagnostics.active_blocks, diagnostics.resident_blocks,
        diagnostics.evictable_blocks, diagnostics.evictable_sessions,
        diagnostics.sessions_with_nonzero_frontier});
}

void MetricsStore::record_cpu_kv_cache_target(
    const config::ResolvedCpuKVCacheTargetConfig &config,
    const kv_cache::CpuKVCacheStats &stats,
    const kv_cache::CpuKVCacheDiagnostics &diagnostics,
    scheduler::ReplicaTarget target, ClusterType cluster_type,
    std::size_t pending_restores, std::size_t staged_restores) {
    const auto bytes = [&](std::uint64_t blocks) {
        if (blocks > std::numeric_limits<std::uint64_t>::max() /
                         config.bytes_per_block) {
            throw std::overflow_error("CPU KV-cache metric bytes overflow");
        }
        return blocks * config.bytes_per_block;
    };
    const std::uint64_t used = diagnostics.resident_blocks +
                               diagnostics.reserved_blocks;
    if (used > diagnostics.capacity_blocks) {
        throw std::logic_error("CPU KV-cache target exceeds capacity");
    }
    CpuKVCacheTargetMetricsRecord record{};
    record.cluster_type = cluster_type;
    record.replica_id = target.replica_id;
    record.dp_id = target.dp_id;
    record.capacity_bytes = config.capacity_bytes;
    record.capacity_blocks = diagnostics.capacity_blocks;
    record.bytes_per_block = config.bytes_per_block;
    record.resident_blocks = diagnostics.resident_blocks;
    record.resident_bytes = bytes(record.resident_blocks);
    record.reserved_blocks = diagnostics.reserved_blocks;
    record.reserved_bytes = bytes(record.reserved_blocks);
    record.free_blocks = diagnostics.capacity_blocks - used;
    record.free_bytes = bytes(record.free_blocks);
    record.peak_resident_blocks = stats.peak_resident_blocks;
    record.peak_resident_bytes = bytes(record.peak_resident_blocks);
    record.peak_reserved_blocks = stats.peak_reserved_blocks;
    record.peak_reserved_bytes = bytes(record.peak_reserved_blocks);
    record.resident_sessions = diagnostics.sessions;
    record.evicted_sessions = stats.evicted_sessions;
    record.evicted_blocks = stats.evicted_blocks;
    record.evicted_bytes = bytes(stats.evicted_blocks);
    record.skipped_offloads = stats.skipped_offloads;
    record.truncated_offloads = stats.truncated_offloads;
    record.stale_generation_completions =
        stats.stale_generation_completions;
    record.cpu_query_blocks = stats.query_blocks;
    record.cpu_hit_blocks = stats.hit_blocks;
    record.sessions_with_cpu_hits = stats.sessions_with_hits;
    record.pending_restore_operations = pending_restores;
    record.staged_restore_payloads = staged_restores;
    record.active_restore_leases = diagnostics.active_restore_leases;
    record.active_offload_reservations = diagnostics.active_reservations;
    output_.cpu_kv_cache_targets.push_back(record);

    CpuKVCacheMetricsAggregate &aggregate = output_.aggregate.cpu_kv_cache;
    ++aggregate.target_count;
    aggregate.capacity_bytes += record.capacity_bytes;
    aggregate.capacity_blocks += record.capacity_blocks;
    if (aggregate.bytes_per_block == 0) {
        aggregate.bytes_per_block = record.bytes_per_block;
    } else if (aggregate.bytes_per_block != record.bytes_per_block) {
        throw std::logic_error(
            "CPU KV-cache targets reported inconsistent block sizes");
    }
    aggregate.query_blocks += record.cpu_query_blocks;
    aggregate.hit_blocks += record.cpu_hit_blocks;
    aggregate.resident_bytes += record.resident_bytes;
    aggregate.resident_blocks += record.resident_blocks;
    aggregate.reserved_bytes += record.reserved_bytes;
    aggregate.reserved_blocks += record.reserved_blocks;
    aggregate.free_bytes += record.free_bytes;
    aggregate.free_blocks += record.free_blocks;
    aggregate.peak_resident_bytes += record.peak_resident_bytes;
    aggregate.peak_resident_blocks += record.peak_resident_blocks;
    aggregate.peak_reserved_bytes += record.peak_reserved_bytes;
    aggregate.peak_reserved_blocks += record.peak_reserved_blocks;
    aggregate.resident_sessions += record.resident_sessions;
    aggregate.evicted_blocks += record.evicted_blocks;
    aggregate.evicted_sessions += record.evicted_sessions;
    aggregate.evicted_bytes += record.evicted_bytes;
    aggregate.skipped_offloads += record.skipped_offloads;
    aggregate.truncated_offloads += record.truncated_offloads;
    aggregate.stale_generation_completions +=
        record.stale_generation_completions;
    aggregate.sessions_with_cpu_hits += record.sessions_with_cpu_hits;
    aggregate.pending_restore_operations +=
        record.pending_restore_operations;
    aggregate.staged_restore_payloads += record.staged_restore_payloads;
    aggregate.active_restore_leases += record.active_restore_leases;
    aggregate.active_offload_reservations +=
        record.active_offload_reservations;
}

void MetricsStore::record_cpu_kv_cache_offload(
    const entities::CpuKVCacheOffloadInfo &operation,
    ClusterType cluster_type, std::uint64_t bytes_per_block) {
    if (operation.state() != entities::CpuKVCacheTransferState::kCompleted) {
        return;
    }
    const auto &timing = operation.timing();
    CpuKVCacheTransferMetricsRecord record{};
    record.transfer_id = operation.transfer_id();
    record.kind = CpuKVCacheTransferKind::kOffload;
    record.request_id = operation.request_id();
    record.cluster_type = cluster_type;
    record.replica_id = operation.replica_id();
    record.dp_id = operation.dp_id();
    record.blocks = timing.size_bytes / bytes_per_block;
    record.size_bytes = timing.size_bytes;
    record.submitted_at = timing.submitted_at;
    record.started_at = timing.started_at;
    record.completed_at = timing.completed_at;
    record.queue_time_ms = timing.queue_time_ms;
    record.service_time_ms = timing.service_time_ms;
    record.source_gpu_hold_ms = operation.attributable_source_hold_ms();
    auto &aggregate = output_.aggregate.cpu_kv_cache;
    ++aggregate.offload_operations;
    aggregate.offload_blocks += record.blocks;
    aggregate.offload_bytes += record.size_bytes;
    aggregate.d2h_queue_time_ms += record.queue_time_ms;
    aggregate.d2h_service_time_ms += record.service_time_ms;
    aggregate.source_gpu_hold_time_ms += record.source_gpu_hold_ms;
    if (detailed_traces_enabled_) {
        output_.cpu_kv_cache_transfers.push_back(record);
    }
}

void MetricsStore::record_cpu_kv_cache_restore(
    const entities::CpuKVCacheRestoreInfo &operation,
    ClusterType cluster_type, std::uint64_t bytes_per_block) {
    if (operation.state() != entities::CpuKVCacheTransferState::kCompleted) {
        return;
    }
    const auto &timing = operation.timing();
    CpuKVCacheTransferMetricsRecord record{};
    record.transfer_id = operation.transfer_id();
    record.kind = CpuKVCacheTransferKind::kRestore;
    record.request_id = operation.request_id();
    record.cluster_type = cluster_type;
    record.replica_id = operation.replica_id();
    record.dp_id = operation.dp_id();
    record.blocks = timing.size_bytes / bytes_per_block;
    record.size_bytes = timing.size_bytes;
    record.submitted_at = timing.submitted_at;
    record.started_at = timing.started_at;
    record.completed_at = timing.completed_at;
    record.queue_time_ms = timing.queue_time_ms;
    record.service_time_ms = timing.service_time_ms;
    auto &aggregate = output_.aggregate.cpu_kv_cache;
    ++aggregate.restore_operations;
    aggregate.restore_blocks += record.blocks;
    aggregate.restore_bytes += record.size_bytes;
    aggregate.h2d_queue_time_ms += record.queue_time_ms;
    aggregate.h2d_service_time_ms += record.service_time_ms;
    if (detailed_traces_enabled_) {
        output_.cpu_kv_cache_transfers.push_back(record);
    }
}

void MetricsStore::record_gpu_kv_cache_occupancy(
    SimTime time, const scheduler::BaseReplicaScheduler &scheduler,
    std::uint64_t bytes_per_block, std::optional<std::uint64_t> total_hbm_bytes,
    bool force) {
    if (!gpu_kv_occupancy_enabled_) {
        return;
    }
    if (!time.valid()) {
        throw std::invalid_argument(
            "GPU KV occupancy time must be finite and nonnegative");
    }
    if (bytes_per_block == 0) {
        throw std::invalid_argument(
            "GPU KV occupancy bytes_per_block must be positive");
    }
    const kv_cache::PrefixCacheDiagnostics diagnostics =
        scheduler.prefix_cache_diagnostics();
    if (diagnostics.active_blocks > diagnostics.capacity_blocks) {
        throw std::logic_error("GPU KV occupancy exceeds target capacity");
    }
    if (diagnostics.active_blocks >
        std::numeric_limits<std::uint64_t>::max() / bytes_per_block) {
        throw std::overflow_error(
            "GPU KV occupancy byte count overflows uint64");
    }
    if (total_hbm_bytes.has_value() && total_hbm_bytes.value() == 0) {
        throw std::invalid_argument(
            "GPU KV occupancy total HBM bytes must be positive");
    }

    GpuKVCacheOccupancyRecord record{};
    record.time = time;
    record.cluster_type = scheduler.cluster_type();
    record.replica_id = scheduler.replica_id();
    record.dp_id = scheduler.dp_id();
    record.active_blocks = diagnostics.active_blocks;
    record.capacity_blocks = diagnostics.capacity_blocks;
    record.active_bytes_per_gpu = diagnostics.active_blocks * bytes_per_block;
    record.active_fraction_of_kv_budget =
        diagnostics.capacity_blocks == 0
            ? 0.0
            : static_cast<double>(diagnostics.active_blocks) /
                  static_cast<double>(diagnostics.capacity_blocks);
    if (total_hbm_bytes.has_value()) {
        const double total_hbm_fraction =
            static_cast<double>(record.active_bytes_per_gpu) /
            static_cast<double>(total_hbm_bytes.value());
        record.hbm_fraction = total_hbm_fraction;
        record.active_fraction_of_total_hbm = total_hbm_fraction;
    }

    const OccupancyTarget key{record.cluster_type, record.replica_id,
                              record.dp_id};
    const auto previous = occupancy_positions_.find(key);
    if (previous != occupancy_positions_.end()) {
        GpuKVCacheOccupancyRecord &last =
            output_.gpu_kv_occupancy.at(previous->second);
        const bool same_state =
            last.active_blocks == record.active_blocks &&
            last.capacity_blocks == record.capacity_blocks &&
            last.active_bytes_per_gpu == record.active_bytes_per_gpu &&
            last.active_fraction_of_kv_budget ==
                record.active_fraction_of_kv_budget &&
            last.hbm_fraction == record.hbm_fraction &&
            last.active_fraction_of_total_hbm ==
                record.active_fraction_of_total_hbm;
        if (last.time == time) {
            // Multiple scheduler mutations can occur at one simulation time;
            // retain only the final target state for that timestamp.
            last = record;
            return;
        }
        if (!force && same_state) {
            return;
        }
    }
    occupancy_positions_[key] = output_.gpu_kv_occupancy.size();
    output_.gpu_kv_occupancy.push_back(std::move(record));
}

SimulationOutput MetricsStore::take_output() noexcept {
    return std::move(output_);
}

} // namespace frontier::metrics
