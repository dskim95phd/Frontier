#include "frontier/metrics/metrics_store.h"

#include <stdexcept>
#include <string>
#include <utility>

#include "frontier/entities/batch.h"
#include "frontier/entities/batch_stage.h"
#include "frontier/entities/kv_cache_transfer_info.h"
#include "frontier/entities/request.h"
#include "frontier/execution_time_predictor/base_execution_time_predictor.h"
#include "frontier/scheduler/replica_scheduler/base_replica_scheduler.h"
#include "frontier/scheduler/scheduler_types.h"
#include "frontier/simulator/entity_arena.h"

namespace frontier::metrics {

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
        record_request(std::move(record));
    }
}

SimulationOutput MetricsStore::take_output() noexcept {
    return std::move(output_);
}

} // namespace frontier::metrics
