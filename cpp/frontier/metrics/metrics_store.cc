#include "frontier/metrics/metrics_store.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>

#include "frontier/core/checked_math.h"
#include "frontier/entities/batch.h"
#include "frontier/entities/batch_stage.h"
#include "frontier/entities/cpu_kv_cache_transfer_info.h"
#include "frontier/entities/kv_cache_transfer_info.h"
#include "frontier/entities/request.h"
#include "frontier/execution_time_predictor/base_execution_time_predictor.h"
#include "frontier/kv_cache/cpu_kv_cache_manager.h"
#include "frontier/kv_cache/replica_kv_cache_manager.h"
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
    total.moe_pre_barrier_wait_ms += value.moe_pre_barrier_wait_ms;
    total.moe_ep_aggregation_extra_ms += value.moe_ep_aggregation_extra_ms;
    total.synchronization_unattributed_wait_ms +=
        value.synchronization_unattributed_wait_ms;
    total.synchronization_attribution_overlap_ms +=
        value.synchronization_attribution_overlap_ms;
    total.prefill_attention_token_pairs =
        checked_math::add<std::overflow_error>(
            total.prefill_attention_token_pairs,
            value.prefill_attention_token_pairs,
            "aggregate PREFILL attention token-pair count overflows uint64");
}

std::uint64_t
prefill_attention_token_pairs_for_request(std::uint64_t query_tokens,
                                          std::uint64_t past_context) {
    return checked_math::prefill_attention_token_pairs<std::overflow_error>(
        query_tokens, past_context,
        "PREFILL attention token-pair count overflows uint64");
}

void accumulate_prefill_attention_token_pairs(std::uint64_t &total,
                                              std::uint64_t value) {
    total = checked_math::add<std::overflow_error>(
        total, value,
        "aggregate PREFILL attention token-pair count overflows uint64");
}

PipelineMemoryDiagnostics
make_pipeline_memory_diagnostics(ClusterType cluster_type,
                                 const config::ClusterRuntimeConfig &runtime) {
    const config::GpuMemoryConfig &memory = runtime.gpu_memory;
    PipelineMemoryDiagnostics diagnostics{};
    diagnostics.cluster_type = cluster_type;
    diagnostics.capacity_bytes_per_gpu = memory.capacity_bytes_per_gpu;
    diagnostics.model_weight_bytes_per_gpu = memory.model_weight_bytes_per_gpu;
    diagnostics.kv_cache_budget_bytes_per_gpu =
        memory.kv_cache_budget_bytes_per_gpu;
    diagnostics.kv_cache_bytes_per_block = memory.kv_cache_bytes_per_block;
    diagnostics.configured_num_blocks = runtime.scheduler.num_blocks;
    diagnostics.ordinary_kv_capacity_blocks =
        memory.ordinary_kv_capacity_blocks;
    diagnostics.ordinary_kv_limiting_stage = memory.ordinary_kv_limiting_stage;
    diagnostics.ordinary_kv_limiting_rank = memory.ordinary_kv_limiting_rank;
    diagnostics.kda_snapshot_blocks_per_session =
        runtime.scheduler.kda_snapshot_blocks_per_session;
    diagnostics.kda_snapshot_limiting_stage =
        memory.kda_snapshot_limiting_stage;
    diagnostics.kda_snapshot_limiting_rank = memory.kda_snapshot_limiting_rank;
    diagnostics.timing_group_count =
        memory.pipeline_stage_group_catalogue.timing_groups.size();
    diagnostics.memory_group_count =
        memory.pipeline_stage_group_catalogue.memory_groups.size();

    const std::uint64_t pipeline_size =
        runtime.parallelism.pipeline_parallel_size;
    diagnostics.stages.reserve(static_cast<std::size_t>(pipeline_size));
    for (std::uint64_t stage = 0; stage < pipeline_size; ++stage) {
        PipelineStageMemoryDiagnostic stage_diagnostic{};
        stage_diagnostic.cluster_type = cluster_type;
        stage_diagnostic.stage_id = StageId{stage};
        const config::PipelineStageLayerRange layers =
            config::pipeline_stage_layer_range(runtime.model.num_layers,
                                               runtime.parallelism, stage);
        stage_diagnostic.layer_begin = layers.begin;
        stage_diagnostic.layer_end = layers.end;
        stage_diagnostic.layer_count = layers.size();
        for (std::uint64_t layer = stage_diagnostic.layer_begin;
             layer < stage_diagnostic.layer_end; ++layer) {
            stage_diagnostic.kda_layer_count +=
                static_cast<std::uint64_t>(runtime.model.is_kda_layer(layer));
            stage_diagnostic.mla_layer_count +=
                static_cast<std::uint64_t>(runtime.model.is_mla_layer(layer));
            stage_diagnostic.moe_layer_count +=
                static_cast<std::uint64_t>(runtime.model.is_moe_layer(layer));
        }

        if (stage < memory.pipeline_stage_memory_profiles.size()) {
            const auto &profile =
                memory.pipeline_stage_memory_profiles.at(stage);
            stage_diagnostic.resident_weight_bytes =
                profile.resident_weight_bytes;
            stage_diagnostic.reserved_bytes = profile.reserved_bytes;
            stage_diagnostic.free_bytes = profile.free_bytes;
            stage_diagnostic.kv_bytes_per_block_by_rank =
                profile.kv_bytes_per_block_by_rank;
            stage_diagnostic.kda_snapshot_bytes_by_rank =
                profile.kda_snapshot_bytes_by_rank;
        }

        const auto &catalogue = memory.pipeline_stage_group_catalogue;
        if (stage < catalogue.stage_to_timing_group.size()) {
            const std::uint32_t group =
                catalogue.stage_to_timing_group.at(stage);
            stage_diagnostic.timing_group_id = group;
            if (group < catalogue.timing_group_multiplicity.size()) {
                stage_diagnostic.timing_group_multiplicity =
                    catalogue.timing_group_multiplicity.at(group);
            }
        }
        if (stage < catalogue.stage_to_memory_group.size()) {
            const std::uint32_t group =
                catalogue.stage_to_memory_group.at(stage);
            stage_diagnostic.memory_group_id = group;
            if (group < catalogue.memory_group_multiplicity.size()) {
                stage_diagnostic.memory_group_multiplicity =
                    catalogue.memory_group_multiplicity.at(group);
            }
        }
        diagnostics.stages.push_back(std::move(stage_diagnostic));
    }
    return diagnostics;
}

} // namespace

MetricsStore::MetricsStore(const config::SimulationConfig &config,
                           bool detailed_traces_enabled)
    : output_([&]() {
          SimulationOutput value{};
          value.schema_version = config.schema_version;
          value.run = [&]() {
              RunMetadata value{};
              value.run_id = config.run_id;
              value.simulation_mode = config.simulation_mode;
              value.system_architecture = config.system_architecture;
              value.prefill_only = config.prefill_only.has_value();
              value.synthetic_decode_tokens_per_second =
                  config.prefill_only.has_value()
                      ? config.prefill_only->decode_tokens_per_second
                      : 0.0;
              return value;
          }();
          return value;
      }()),
      detailed_traces_enabled_(detailed_traces_enabled) {
    if (config.system_architecture ==
        config::SystemArchitecture::kPdDisaggregation) {
        output_.pipeline_memory_diagnostics.push_back(
            make_pipeline_memory_diagnostics(ClusterType::kPrefill,
                                             config.pdd().clusters.prefill));
        output_.pipeline_memory_diagnostics.push_back(
            make_pipeline_memory_diagnostics(ClusterType::kDecode,
                                             config.pdd().clusters.decode));
    } else if (std::holds_alternative<config::ClusterRuntimeConfig>(
                   config.runtime)) {
        output_.pipeline_memory_diagnostics.push_back(
            make_pipeline_memory_diagnostics(ClusterType::kMonolithic,
                                             config.cluster()));
    }
}

void MetricsStore::record_event(Event event) {
    ++output_.aggregate.event_count;
    if (detailed_traces_enabled_) {
        output_.event_trace.push_back(std::move(event));
    }
}

void MetricsStore::record_prefill_completion(
    const entities::Request &request, scheduler::ReplicaTarget target) {
    if (!output_.run.prefill_only) {
        return;
    }
    if (!request.is_prefill_complete() ||
        !request.prefill_completed_at().valid() || !target.replica_id.valid() ||
        !target.dp_id.valid()) {
        throw std::logic_error(
            "PREFILL completion metrics require a completed owned PREFILL");
    }
    PrefillCompletionMetricsRecord record{};
    record.request_id = request.id();
    record.num_prefill_tokens = request.initial_num_prefill_tokens();
    record.scheduled_prefill_tokens = request.scheduled_prefill_tokens();
    record.preemption_recomputed_prefill_tokens =
        request.preemption_recomputed_prefill_tokens();
    record.arrived_at = request.arrived_at();
    record.completed_at = request.prefill_completed_at();
    record.replica_id = target.replica_id;
    record.dp_id = target.dp_id;
    output_.prefill_completions.push_back(record);
}

void MetricsStore::record_request(RequestMetricsRecord record) {
    output_.requests.push_back(std::move(record));
}

void MetricsStore::record_batch(const entities::Batch &batch,
                                const entities::RequestCollection &requests,
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
        output_.aggregate
            .batch_time_buckets_by_cluster[batch.cluster_type()][bucket_index];
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
    accumulate_prefill_attention_token_pairs(
        aggregate.prefill_attention_token_pairs,
        batch_stage.execution_time().prefill_attention_token_pairs);
    constexpr double kBatchTimeBucketSeconds = 60.0;
    const auto bucket_index = static_cast<std::uint64_t>(
        std::floor(batch.scheduled_at().seconds() / kBatchTimeBucketSeconds));
    BatchTimeBucketAggregate &time_bucket =
        output_.aggregate
            .batch_time_buckets_by_cluster[batch.cluster_type()][bucket_index];
    accumulate_execution_time(time_bucket.execution_time,
                              batch_stage.execution_time());
    accumulate_prefill_attention_token_pairs(
        time_bucket.prefill_attention_token_pairs,
        batch_stage.execution_time().prefill_attention_token_pairs);
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
        value.lane_routed_tokens = diagnostic.lane_routed_tokens;
        value.lane_active_experts = diagnostic.lane_active_experts;
        value.lane_unique_tokens = diagnostic.lane_unique_tokens;
        value.lane_times_ms = diagnostic.lane_times_ms;
        value.routed_lane_times_ms = diagnostic.routed_lane_times_ms;
        value.source_local_lane_times_ms =
            diagnostic.source_local_lane_times_ms;
        value.shared_expert_path_ms = diagnostic.shared_expert_path_ms;
        value.critical_lane = diagnostic.critical_lane;
        value.critical_lane_time_ms = diagnostic.critical_lane_time_ms;
        value.raw_ep_dispatch_ms = diagnostic.raw_ep_dispatch_ms;
        value.raw_ep_combine_ms = diagnostic.raw_ep_combine_ms;
        value.exposed_ep_dispatch_ms = diagnostic.exposed_ep_dispatch_ms;
        value.exposed_ep_combine_ms = diagnostic.exposed_ep_combine_ms;
        value.grouped_gemm_geometry = diagnostic.grouped_gemm_geometry;
        return value;
    }());
    const MoERoutingMetricsRecord &record = output_.moe_routing.back();
    moe_routing_positions_.insert_or_assign(
        MoERoutingKey{record.batch_id, record.stage_id, record.layer_id},
        output_.moe_routing.size() - 1);
}

void MetricsStore::update_moe_grouped_gemm_geometry(
    BatchId batch_id, StageId stage_id, LayerId layer_id,
    const execution_time_predictor::MoEGroupedGemmGeometryDiagnostic
        &geometry) {
    if (!detailed_traces_enabled_) {
        return;
    }
    const auto position = moe_routing_positions_.find(
        MoERoutingKey{batch_id, stage_id, layer_id});
    if (position != moe_routing_positions_.end()) {
        output_.moe_routing.at(position->second).grouped_gemm_geometry =
            geometry;
    }
}

void MetricsStore::release_batch_diagnostics(BatchId batch_id) {
    if (!detailed_traces_enabled_) {
        return;
    }
    for (auto position = moe_routing_positions_.begin();
         position != moe_routing_positions_.end();) {
        if (position->first.batch_id == batch_id) {
            position = moe_routing_positions_.erase(position);
        } else {
            ++position;
        }
    }
}

void MetricsStore::collect_completed_requests(
    const config::SimulationConfig &config,
    const simulator::EntityArena &entities) {
    if (!output_.requests.empty()) {
        throw std::logic_error("request metrics were collected more than once");
    }
    output_.requests.reserve(entities.completion_order().size());
    // Keep arrival-side demand independent of completion filtering.  A bounded
    // run can intentionally leave requests waiting or in flight, but those
    // requests still contribute demand from the moment their arrival event was
    // observed.
    constexpr double kBatchTimeBucketSeconds = 60.0;
    if (!arrival_demand_collected_) {
        for (const entities::Request &request : entities.requests()) {
            if (request.state() == entities::RequestState::kPending ||
                !request.arrived_at().valid()) {
                continue;
            }
            const std::uint64_t prompt_tokens =
                request.initial_num_prefill_tokens();
            const std::uint64_t cached_tokens = request.cached_prefill_tokens();
            if (cached_tokens > prompt_tokens) {
                throw std::logic_error(
                    "request cached PREFILL tokens exceed initial prompt");
            }
            const std::uint64_t query_tokens = prompt_tokens - cached_tokens;
            const std::uint64_t token_pairs =
                prefill_attention_token_pairs_for_request(query_tokens,
                                                          cached_tokens);
            const auto bucket_index = static_cast<std::uint64_t>(std::floor(
                request.arrived_at().seconds() / kBatchTimeBucketSeconds));
            std::uint64_t &bucket =
                output_.aggregate
                    .prefill_attention_token_pairs_by_arrival_time_bucket
                        [bucket_index];
            accumulate_prefill_attention_token_pairs(bucket, token_pairs);
        }
        arrival_demand_collected_ = true;
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
            request_id, is_pdd && !config.prefill_only.has_value()
                            ? ClusterType::kDecode
                            : (is_pdd ? ClusterType::kPrefill
                                      : ClusterType::kMonolithic));
        RequestMetricsRecord record = [&]() {
            RequestMetricsRecord value{};
            value.request_id = request_id;
            value.session_id = request.session_id();
            value.prefill_only = config.prefill_only.has_value();
            value.num_prefill_tokens = request.initial_num_prefill_tokens();
            value.num_decode_tokens = request.initial_num_decode_tokens();
            value.scheduled_prefill_tokens = request.scheduled_prefill_tokens();
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
            value.cpu_restore_queue_time_s = request.cpu_restore_queue_time_s();
            value.cpu_restore_service_time_s =
                request.cpu_restore_service_time_s();
            value.cpu_offload_bytes = request.cpu_offload_bytes();
            value.cpu_offload_queue_time_s = request.cpu_offload_queue_time_s();
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
            value.tokens_at_preemption = request.tokens_at_preemption();
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
            if (!config.prefill_only.has_value()) {
                record.decode_replica_id = target.replica_id;
                record.decode_dp_id = target.dp_id;
            }
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

SimulationOutput MetricsStore::take_output() noexcept {
    return std::move(output_);
}

} // namespace frontier::metrics
