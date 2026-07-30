#include "frontier/metrics/metrics_store.h"

#include <stdexcept>
#include <string>
#include <utility>

#include "frontier/entities/batch.h"
#include "frontier/entities/batch_stage.h"
#include "frontier/entities/kv_cache_transfer_info.h"
#include "frontier/entities/request.h"
#include "frontier/execution_time_predictor/batch_execution_model.h"
#include "frontier/scheduler/replica_scheduler/base_replica_scheduler.h"
#include "frontier/scheduler/scheduler_types.h"
#include "frontier/simulator/entity_arena.h"

namespace frontier::metrics {

MetricsStore::MetricsStore(
    const config::SimulationConfig& config,
    std::size_t expected_request_count)
    : output_{
          .schema_version = config.schema_version,
          .run = RunMetadata{
              .run_id = config.run_id,
              .simulation_mode = config.simulation_mode,
              .system_architecture = config.system_architecture,
          },
      } {
  output_.requests.reserve(expected_request_count);
  output_.event_trace.reserve(expected_request_count * 12);
}

void MetricsStore::record_event(Event event) {
  output_.event_trace.push_back(std::move(event));
}

void MetricsStore::record_request(RequestMetricsRecord record) {
  output_.requests.push_back(std::move(record));
}

void MetricsStore::record_batch(
    const entities::Batch& batch,
    const std::vector<entities::Request>& requests,
    double predicted_execution_ms,
    const config::ClusterRuntimeConfig& runtime) {
  if (!batch.completed_at().valid()) {
    throw std::logic_error(
        "cannot emit metrics for incomplete batch");
  }
  BatchMetricsRecord record{
      .batch_id = batch.id(),
      .iteration_id = batch.iteration_id(),
      .scheduled_at = batch.scheduled_at(),
      .completed_at = batch.completed_at(),
      .request_ids = {},
      .scheduled_tokens = {},
      .total_scheduled_tokens = batch.total_scheduled_tokens(),
      .num_prefill_tokens = 0,
      .num_decode_tokens = 0,
      .predicted_execution_ms = predicted_execution_ms,
      .replica_id = batch.replica_id(),
      .dp_id = batch.dp_id(),
      .num_pipeline_stages = batch.num_pipeline_stages(),
      .cluster_type = batch.cluster_type(),
      .model_kind = batch.model_kind(),
      .runtime_total_experts =
          runtime.model.runtime_total_experts,
      .router_topk = runtime.model.router_topk,
      .moe_sync_group_id = batch.moe_sync_group_id(),
      .parallelism = runtime.parallelism,
  };
  for (const entities::RequestBatchSnapshot& snapshot :
       batch.requests()) {
    const entities::Request& request = requests.at(
        static_cast<std::size_t>(snapshot.request_id.value()));
    record.request_ids.push_back(snapshot.request_id);
    record.scheduled_tokens.push_back(snapshot.scheduled_tokens);
    if (snapshot.processed_tokens <
        request.num_prefill_tokens()) {
      record.num_prefill_tokens += snapshot.scheduled_tokens;
    } else {
      record.num_decode_tokens += snapshot.scheduled_tokens;
    }
  }
  output_.batches.push_back(std::move(record));
}

void MetricsStore::record_batch_stage(
    const entities::BatchStage& batch_stage,
    const entities::Batch& batch,
    const config::ClusterRuntimeConfig& runtime) {
  BatchStageMetricsRecord record{
      .batch_id = batch_stage.batch_id(),
      .replica_id = batch.replica_id(),
      .dp_id = batch.dp_id(),
      .stage_id = batch_stage.stage_id(),
      .arrived_at = batch_stage.arrived_at(),
      .started_at = batch_stage.started_at(),
      .completed_at = batch_stage.completed_at(),
      .execution_time = batch_stage.execution_time(),
      .cluster_type = batch.cluster_type(),
      .model_kind = batch.model_kind(),
      .runtime_total_experts =
          runtime.model.runtime_total_experts,
      .router_topk = runtime.model.router_topk,
      .moe_sync_group_id = batch.moe_sync_group_id(),
      .parallelism = runtime.parallelism,
  };
  output_.batch_stages.push_back(std::move(record));
}

void MetricsStore::record_scheduler_trace(
    const scheduler::ScheduleResult& schedule,
    scheduler::ReplicaTarget target,
    ClusterType cluster_type) {
  SchedulerTraceRecord trace{
      .iteration_id = schedule.iteration_id,
      .simulation_time = schedule.simulation_time,
      .token_budget_before = schedule.token_budget_before,
      .token_budget_after = schedule.token_budget_after,
      .available_blocks_before = schedule.available_blocks_before,
      .available_blocks_after = schedule.available_blocks_after,
      .waiting_count_before = schedule.waiting_count_before,
      .waiting_count_after = schedule.waiting_count_after,
      .running_count_before = schedule.running_count_before,
      .running_count_after = schedule.running_count_after,
      .preempted_count = schedule.preempted_count,
      .decisions = {},
      .batch_request_ids = {},
      .request_num_tokens = {},
      .replica_id = target.replica_id,
      .dp_id = target.dp_id,
      .cluster_type = cluster_type,
  };
  for (const scheduler::SchedulerDecision& decision :
       schedule.decisions) {
    trace.decisions.push_back(SchedulerDecisionRecord{
        .decision_result =
            std::string{scheduler::to_string(decision.type)},
        .request_id = decision.request_id,
        .num_tokens = decision.num_tokens,
        .token_budget_after = decision.token_budget_after,
        .available_blocks_after =
            decision.available_blocks_after,
    });
  }
  for (const scheduler::ScheduledRequest& scheduled :
       schedule.scheduled_requests) {
    trace.batch_request_ids.push_back(scheduled.request_id);
    trace.request_num_tokens.push_back(scheduled.num_tokens);
  }
  output_.scheduler_trace.push_back(std::move(trace));
}

void MetricsStore::record_kv_cache_transfer(
    const entities::KVCacheTransferInfo& transfer) {
  KVCacheTransferMetricsRecord record{
      .transfer_id = transfer.id(),
      .request_id = transfer.request_id(),
      .source_batch_id = transfer.source_batch_id(),
      .source_cluster_type = transfer.source_cluster_type(),
      .target_cluster_type = transfer.target_cluster_type(),
      .source_replica_id = transfer.source_replica_id(),
      .source_dp_id = transfer.source_dp_id(),
      .size_bytes = transfer.size_bytes(),
      .predicted_time_ms = transfer.predicted_time_ms(),
      .started_at = transfer.started_at(),
      .completed_at = transfer.completed_at(),
  };
  output_.kv_cache_transfers.push_back(std::move(record));
}

void MetricsStore::record_analytical_diagnostic(
    std::string name,
    std::vector<std::pair<std::string, double>> values) {
  output_.analytical_diagnostics.push_back(
      AnalyticalDiagnostic{
          .name = std::move(name),
          .values = std::move(values),
      });
}

void MetricsStore::record_moe_routing(
    const entities::Batch& batch,
    StageId stage_id,
    const execution_time_predictor::MoERoutingDiagnostic&
        diagnostic,
    const config::ClusterRuntimeConfig& runtime) {
  output_.moe_routing.push_back(MoERoutingMetricsRecord{
      .batch_id = batch.id(),
      .stage_id = stage_id,
      .cluster_type = batch.cluster_type(),
      .sync_group_id = batch.moe_sync_group_id(),
      .layer_id = diagnostic.layer_id,
      .mode = runtime.moe_routing.mode,
      .distribution = runtime.moe_routing.distribution,
      .seed = runtime.moe_routing.seed,
      .input_tokens = diagnostic.input_tokens,
      .routed_tokens = diagnostic.routed_tokens,
      .global_expert_tokens = diagnostic.global_expert_tokens,
      .lane_expert_tokens = diagnostic.lane_expert_tokens,
      .lane_times_ms = diagnostic.lane_times_ms,
      .critical_lane = diagnostic.critical_lane,
      .critical_lane_time_ms =
          diagnostic.critical_lane_time_ms,
  });
}

void MetricsStore::collect_completed_requests(
    const config::SimulationConfig& config,
    const simulator::EntityArena& entities) {
  if (!output_.requests.empty()) {
    throw std::logic_error(
        "request metrics were collected more than once");
  }
  const bool is_pdd =
      config.system_architecture ==
      config::SystemArchitecture::kPdDisaggregation;
  for (const RequestId request_id : entities.completion_order()) {
    const entities::Request& value = entities.request(request_id);
    if (!value.completed() ||
        !value.first_scheduled_at().valid() ||
        !value.prefill_completed_at().valid() ||
        !value.first_token_completed_at().valid() ||
        !value.completed_at().valid()) {
      throw std::runtime_error(
          "completed request is missing canonical metrics");
    }
    const scheduler::ReplicaTarget target =
        entities.request_target(
            request_id,
            is_pdd
                ? ClusterType::kDecode
                : ClusterType::kMonolithic);
    RequestMetricsRecord record{
        .request_id = request_id,
        .arrived_at = value.arrived_at(),
        .prefill_completed_at = value.prefill_completed_at(),
        .completed_at = value.completed_at(),
        .first_scheduled_at = value.first_scheduled_at(),
        .first_token_completed_at =
            value.first_token_completed_at(),
        .num_processed_tokens = value.num_processed_tokens(),
        .preemption_count = value.preemption_count(),
        .tokens_at_preemption = {},
        .replica_id = target.replica_id,
        .dp_id = target.dp_id,
        .prefill_replica_id = ReplicaId{},
        .prefill_dp_id = DataParallelId{},
        .decode_replica_id = ReplicaId{},
        .decode_dp_id = DataParallelId{},
        .transfer_id = TransferId{},
        .kv_cache_transfer_start_time = SimTime{},
        .kv_cache_transfer_end_time = SimTime{},
        .decode_arrived_at = SimTime{},
        .kv_cache_transfer_size_bytes = 0,
    };
    if (is_pdd) {
      const scheduler::ReplicaTarget prefill =
          entities.request_target(
              request_id, ClusterType::kPrefill);
      const TransferId transfer_id =
          entities.request_transfer_id(request_id);
      record.prefill_replica_id = prefill.replica_id;
      record.prefill_dp_id = prefill.dp_id;
      record.decode_replica_id = target.replica_id;
      record.decode_dp_id = target.dp_id;
      record.transfer_id = transfer_id;
      record.kv_cache_transfer_start_time =
          value.kv_cache_transfer_start_time();
      record.kv_cache_transfer_end_time =
          value.kv_cache_transfer_end_time();
      record.decode_arrived_at = value.decode_arrived_at();
      record.kv_cache_transfer_size_bytes =
          value.kv_cache_transfer_size_bytes();
    }
    record_request(std::move(record));
  }
}

SimulationOutput MetricsStore::take_output() noexcept {
  return std::move(output_);
}

}  // namespace frontier::metrics
