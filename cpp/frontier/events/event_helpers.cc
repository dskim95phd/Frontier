#include "frontier/events/event_helpers.h"

#include <stdexcept>
#include <string>

namespace frontier::events {

scheduler::ReplicaTarget make_target(
    ReplicaId replica_id,
    DataParallelId dp_id) noexcept {
  return scheduler::ReplicaTarget{
      .replica_id = replica_id,
      .dp_id = dp_id,
  };
}

metrics::SchedulerTraceRecord make_scheduler_trace(
    const scheduler::ScheduleResult& schedule,
    scheduler::ReplicaTarget target,
    ClusterType cluster_type) {
  metrics::SchedulerTraceRecord trace{
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
    trace.decisions.push_back(metrics::SchedulerDecisionRecord{
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
  return trace;
}

metrics::BatchMetricsRecord make_batch_metrics(
    const entities::Batch& batch,
    const std::vector<entities::Request>& requests,
    double predicted_execution_ms) {
  if (!batch.completed_at().valid()) {
    throw std::logic_error(
        "cannot emit metrics for incomplete batch");
  }
  metrics::BatchMetricsRecord record{
      .batch_id = batch.id(),
      .iteration_id = batch.iteration_id(),
      .scheduled_at = batch.scheduled_at(),
      .completed_at = batch.completed_at(),
      .request_ids = {},
      .scheduled_tokens = {},
      .total_scheduled_tokens =
          batch.total_scheduled_tokens(),
      .num_prefill_tokens = 0,
      .num_decode_tokens = 0,
      .predicted_execution_ms =
          predicted_execution_ms,
      .replica_id = batch.replica_id(),
      .dp_id = batch.dp_id(),
      .num_pipeline_stages =
          batch.num_pipeline_stages(),
      .cluster_type = batch.cluster_type(),
  };
  for (const entities::RequestBatchSnapshot& snapshot :
       batch.requests()) {
    const entities::Request& request = requests.at(
        static_cast<std::size_t>(snapshot.request_id.value()));
    record.request_ids.push_back(snapshot.request_id);
    record.scheduled_tokens.push_back(
        snapshot.scheduled_tokens);
    if (snapshot.processed_tokens <
        request.num_prefill_tokens()) {
      record.num_prefill_tokens += snapshot.scheduled_tokens;
    } else {
      record.num_decode_tokens += snapshot.scheduled_tokens;
    }
  }
  return record;
}

}  // namespace frontier::events
