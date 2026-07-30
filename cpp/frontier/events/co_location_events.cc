#include "frontier/events/co_location_events.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "frontier/simulator/simulation_context.h"

namespace frontier::events {
namespace {

template <typename Id>
Id require_id(
    const std::optional<Id>& value,
    const char* field) {
  if (!value.has_value()) {
    throw std::invalid_argument(
        std::string{"event is missing "} + field);
  }
  return value.value();
}

EventPayload target_payload(
    scheduler::ReplicaTarget target,
    ClusterType cluster_type) {
  EventPayload payload;
  payload.replica_id = target.replica_id;
  payload.dp_id = target.dp_id;
  payload.cluster_type = cluster_type;
  return payload;
}

ClusterType require_cluster(const Event& event) {
  return require_id(
      event.payload.cluster_type, "cluster_type");
}

scheduler::ReplicaTarget require_target(const Event& event) {
  return scheduler::ReplicaTarget{
      .replica_id = require_id(
          event.payload.replica_id, "replica_id"),
      .dp_id = require_id(event.payload.dp_id, "dp_id"),
  };
}

metrics::SchedulerTraceRecord make_trace(
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
  if (!batch.completed_at().has_value()) {
    throw std::logic_error(
        "cannot emit metrics for incomplete batch");
  }
  metrics::BatchMetricsRecord record{
      .batch_id = batch.id(),
      .iteration_id = batch.iteration_id(),
      .scheduled_at = batch.scheduled_at(),
      .completed_at = batch.completed_at().value(),
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
      record.num_prefill_tokens +=
          snapshot.scheduled_tokens;
    } else {
      record.num_decode_tokens +=
          snapshot.scheduled_tokens;
    }
  }
  return record;
}

}  // namespace

void RequestArrivalEvent::handle(
    const Event& event,
    simulator::SimulationContext& context) const {
  const RequestId request_id = require_id(
      event.payload.request_id, "request_id");
  const ClusterType cluster_type = require_cluster(event);
  entities::Request& request = context.request(request_id);
  request.on_arrival(event.time, cluster_type);
  context.global_scheduler().add_request(
      request_id, cluster_type);
  EventPayload payload;
  payload.cluster_type = cluster_type;
  context.event_queue().push(
      event.time, EventType::kGlobalSchedule, std::move(payload));
}

void GlobalScheduleEvent::handle(
    const Event& event,
    simulator::SimulationContext& context) const {
  const auto assignments =
      context.global_scheduler().schedule();
  if (assignments.empty()) {
    return;
  }
  std::vector<ClusterType> affected;
  for (const auto& assignment : assignments) {
    context.global_scheduler()
        .get_cluster_scheduler(assignment.cluster_type)
        .add_request(
            assignment.request_id,
            context.request(assignment.request_id).arrived_at());
    if (std::find(
            affected.begin(),
            affected.end(),
            assignment.cluster_type) == affected.end()) {
      affected.push_back(assignment.cluster_type);
    }
  }
  for (const ClusterType cluster_type : affected) {
    EventPayload payload;
    payload.cluster_type = cluster_type;
    context.event_queue().push(
        event.time,
        EventType::kClusterSchedule,
        std::move(payload));
  }
}

void ClusterScheduleEvent::handle(
    const Event& event,
    simulator::SimulationContext& context) const {
  const ClusterType cluster_type = require_cluster(event);
  const auto assignments =
      context.cluster(cluster_type).schedule();
  std::vector<scheduler::ReplicaTarget> affected;
  for (const auto& assignment : assignments) {
    const scheduler::ReplicaTarget target{
        .replica_id = assignment.replica_id,
        .dp_id = assignment.dp_id,
    };
    context.assign_request_target(
        assignment.request_id, target, cluster_type);
    if (std::find(affected.begin(), affected.end(), target) ==
        affected.end()) {
      affected.push_back(target);
    }
  }
  for (const scheduler::ReplicaTarget target : affected) {
    context.event_queue().push(
        event.time,
        EventType::kReplicaSchedule,
        target_payload(target, cluster_type));
  }
}

void ReplicaScheduleEvent::handle(
    const Event& event,
    simulator::SimulationContext& context) const {
  const scheduler::ReplicaTarget target =
      require_target(event);
  const ClusterType cluster_type = require_cluster(event);
  scheduler::BaseReplicaScheduler& replica =
      context.cluster(cluster_type).get_replica_scheduler(
          target.replica_id, target.dp_id);
  bool returned_empty_schedule = false;
  bool produced_batch = false;
  while (replica.in_flight_batch_count() <
         replica.pipeline_parallel_size()) {
    scheduler::ScheduleResult schedule =
        replica.schedule(event.time);
    context.output().scheduler_trace.push_back(
        make_trace(schedule, target, cluster_type));
    if (schedule.scheduled_requests.empty()) {
      returned_empty_schedule = true;
      break;
    }
    const BatchId batch_id =
        context.create_batch(schedule, target, cluster_type);
    produced_batch = true;
    entities::Batch& batch = context.batch(batch_id);
    replica.mark_batch_started(batch);
    EventPayload payload = target_payload(target, cluster_type);
    payload.batch_id = batch_id;
    payload.stage_id = StageId{0};
    payload.generation = batch.schedule_epoch();
    context.event_queue().push(
        event.time,
        EventType::kBatchStageArrival,
      std::move(payload));
  }
  if (!produced_batch && returned_empty_schedule &&
      replica.consume_terminal_release_followup_poll()) {
    context.event_queue().push(
        event.time,
        EventType::kReplicaSchedule,
        target_payload(target, cluster_type));
  }
}

void BatchStageArrivalEvent::handle(
    const Event& event,
    simulator::SimulationContext& context) const {
  const scheduler::ReplicaTarget target =
      require_target(event);
  const ClusterType cluster_type = require_cluster(event);
  const BatchId batch_id = require_id(
      event.payload.batch_id, "batch_id");
  const StageId stage_id = require_id(
      event.payload.stage_id, "stage_id");
  entities::Batch& batch = context.batch(batch_id);
  if (!event.payload.generation.has_value() ||
      event.payload.generation.value() !=
          batch.schedule_epoch()) {
    return;
  }
  context.record_stage_arrival(
      batch_id, stage_id, event.time);
  scheduler::ReplicaStageScheduler& stage =
      context.cluster(cluster_type)
          .get_replica_scheduler(
              target.replica_id, target.dp_id)
          .get_replica_stage_scheduler(stage_id);
  stage.add_batch(batch);
  if (!stage.is_busy()) {
    EventPayload payload = target_payload(target, cluster_type);
    payload.stage_id = stage_id;
    context.event_queue().push(
        event.time,
        EventType::kReplicaStageSchedule,
        std::move(payload));
  }
}

void ReplicaStageScheduleEvent::handle(
    const Event& event,
    simulator::SimulationContext& context) const {
  const scheduler::ReplicaTarget target =
      require_target(event);
  const ClusterType cluster_type = require_cluster(event);
  const StageId stage_id = require_id(
      event.payload.stage_id, "stage_id");
  scheduler::ReplicaStageScheduler& stage =
      context.cluster(cluster_type)
          .get_replica_scheduler(
              target.replica_id, target.dp_id)
          .get_replica_stage_scheduler(stage_id);
  const auto ticket = stage.pop_batch_if_not_busy();
  if (!ticket.has_value()) {
    return;
  }
  entities::Batch& batch =
      context.batch(ticket->batch_id);
  if (batch.schedule_epoch() != ticket->schedule_epoch) {
    stage.on_stage_end(ticket->batch_id);
    if (!stage.empty()) {
      EventPayload payload = target_payload(target, cluster_type);
      payload.stage_id = stage_id;
      context.event_queue().push(
          event.time,
          EventType::kReplicaStageSchedule,
          std::move(payload));
    }
    return;
  }
  const auto prediction =
      stage.predict(batch, context.requests());
  entities::BatchStage& batch_stage =
      context.create_batch_stage(
          batch.id(), stage_id, event.time, prediction);
  if (context.execution_model(cluster_type).type ==
      config::ExecutionModelType::kAnalytical) {
    context.output().analytical_diagnostics.push_back(
        metrics::AnalyticalDiagnostic{
            .name =
                "batch_" +
                std::to_string(batch.id().value()) +
                "_stage_" +
                std::to_string(stage_id.value()),
            .values = prediction.diagnostics,
        });
  }
  const double completion_seconds =
      event.time.seconds() +
      batch_stage.execution_time().total_ms() * 1e-3;
  if (!std::isfinite(completion_seconds)) {
    throw std::runtime_error(
        "batch stage completion time is nonfinite");
  }
  EventPayload payload = target_payload(target, cluster_type);
  payload.batch_id = batch.id();
  payload.stage_id = stage_id;
  payload.generation = batch.schedule_epoch();
  context.event_queue().push(
      SimTime::from_seconds(completion_seconds),
      EventType::kBatchStageEnd,
      std::move(payload));
}

void BatchStageEndEvent::handle(
    const Event& event,
    simulator::SimulationContext& context) const {
  const scheduler::ReplicaTarget target =
      require_target(event);
  const ClusterType cluster_type = require_cluster(event);
  const BatchId batch_id = require_id(
      event.payload.batch_id, "batch_id");
  const StageId stage_id = require_id(
      event.payload.stage_id, "stage_id");
  entities::Batch& batch = context.batch(batch_id);
  if (!event.payload.generation.has_value() ||
      event.payload.generation.value() !=
          batch.schedule_epoch()) {
    return;
  }
  scheduler::ReplicaStageScheduler& stage =
      context.cluster(cluster_type)
          .get_replica_scheduler(
              target.replica_id, target.dp_id)
          .get_replica_stage_scheduler(stage_id);
  stage.on_stage_end(batch_id);
  entities::BatchStage& batch_stage =
      context.batch_stage(batch_id, stage_id);
  batch_stage.mark_completed(event.time);
  context.output().batch_stages.push_back(
      metrics::BatchStageMetricsRecord{
          .batch_id = batch_id,
          .replica_id = target.replica_id,
          .dp_id = target.dp_id,
          .stage_id = stage_id,
          .arrived_at = batch_stage.arrived_at(),
          .started_at = batch_stage.started_at().value(),
          .completed_at = event.time,
          .execution_time =
              batch_stage.execution_time(),
          .cluster_type = cluster_type,
      });

  if (!stage.empty()) {
    EventPayload same_stage =
        target_payload(target, cluster_type);
    same_stage.stage_id = stage_id;
    context.event_queue().push(
        event.time,
        EventType::kReplicaStageSchedule,
        std::move(same_stage));
  }

  EventPayload next = target_payload(target, cluster_type);
  next.batch_id = batch_id;
  next.generation = batch.schedule_epoch();
  if (stage.is_last_stage()) {
    context.event_queue().push(
        event.time,
        EventType::kClusterBatchEnd,
        std::move(next));
  } else {
    next.stage_id = StageId{stage_id.value() + 1};
    context.event_queue().push(
        event.time,
        EventType::kBatchStageArrival,
        std::move(next));
  }
}

void ClusterBatchEndEvent::handle(
    const Event& event,
    simulator::SimulationContext& context) const {
  const ClusterType cluster_type = require_cluster(event);
  if (cluster_type == ClusterType::kPrefill) {
    const scheduler::ReplicaTarget target =
        require_target(event);
    const BatchId batch_id = require_id(
        event.payload.batch_id, "batch_id");
    entities::Batch& batch = context.batch(batch_id);
    if (!event.payload.generation.has_value() ||
        event.payload.generation.value() !=
            batch.schedule_epoch()) {
      return;
    }
    scheduler::BaseReplicaScheduler& replica =
        context.cluster(cluster_type).get_replica_scheduler(
            target.replica_id, target.dp_id);
    if (!replica.on_batch_completed(batch, event.time)) {
      return;
    }
    context.output().batches.push_back(make_batch_metrics(
        batch,
        context.requests(),
        context.predicted_batch_ms(batch_id)));
    for (const entities::RequestBatchSnapshot& snapshot :
         batch.requests()) {
      const entities::Request& request =
          context.request(snapshot.request_id);
      if (!request.is_prefill_complete() ||
          request.state() !=
              entities::RequestState::kTransferPending) {
        continue;
      }
      const TransferId transfer_id =
          context.create_kv_cache_transfer(
              snapshot.request_id,
              batch_id,
              target);
      EventPayload transfer_payload;
      transfer_payload.transfer_id = transfer_id;
      transfer_payload.request_id = snapshot.request_id;
      transfer_payload.batch_id = batch_id;
      transfer_payload.replica_id = target.replica_id;
      transfer_payload.dp_id = target.dp_id;
      transfer_payload.cluster_type = ClusterType::kDecode;
      transfer_payload.generation =
          batch.schedule_epoch();
      context.event_queue().push(
          event.time,
          EventType::kKvCacheTransferStart,
          std::move(transfer_payload));
    }
    context.event_queue().push(
        event.time,
        EventType::kReplicaSchedule,
        target_payload(target, ClusterType::kPrefill));
    return;
  }
  EventPayload payload = event.payload;
  context.event_queue().push(
      event.time,
      EventType::kGlobalBatchEnd,
      std::move(payload));
}

void GlobalBatchEndEvent::handle(
    const Event& event,
    simulator::SimulationContext& context) const {
  const scheduler::ReplicaTarget target =
      require_target(event);
  const ClusterType cluster_type = require_cluster(event);
  const BatchId batch_id = require_id(
      event.payload.batch_id, "batch_id");
  entities::Batch& batch = context.batch(batch_id);
  if (!event.payload.generation.has_value() ||
      event.payload.generation.value() !=
          batch.schedule_epoch()) {
    return;
  }
  scheduler::BaseReplicaScheduler& replica =
      context.cluster(cluster_type).get_replica_scheduler(
          target.replica_id, target.dp_id);
  static_cast<void>(
      replica.on_batch_completed(batch, event.time));
  context.output().batches.push_back(make_batch_metrics(
      batch,
      context.requests(),
      context.predicted_batch_ms(batch_id)));
  for (const entities::RequestBatchSnapshot& snapshot :
       batch.requests()) {
    if (context.request(snapshot.request_id).completed()) {
      context.record_request_completion(snapshot.request_id);
    }
  }
  context.event_queue().push(
      event.time,
      EventType::kReplicaSchedule,
      target_payload(target, cluster_type));
}

void KVCacheTransferStartEvent::handle(
    const Event& event,
    simulator::SimulationContext& context) const {
  const TransferId transfer_id = require_id(
      event.payload.transfer_id, "transfer_id");
  entities::KVCacheTransferInfo& transfer =
      context.kv_cache_transfer(transfer_id);
  if (!event.payload.generation.has_value() ||
      event.payload.generation.value() !=
          transfer.source_generation()) {
    return;
  }
  transfer.mark_started(event.time);
  context.request(transfer.request_id())
      .on_kv_cache_transfer_start(event.time);
  const double completion_seconds =
      event.time.seconds() +
      transfer.predicted_time_ms() * 1e-3;
  if (!std::isfinite(completion_seconds)) {
    throw std::runtime_error(
        "KV transfer completion time is nonfinite");
  }
  EventPayload payload = event.payload;
  context.event_queue().push(
      SimTime::from_seconds(completion_seconds),
      EventType::kKvCacheTransferEnd,
      std::move(payload));
}

void KVCacheTransferEndEvent::handle(
    const Event& event,
    simulator::SimulationContext& context) const {
  const TransferId transfer_id = require_id(
      event.payload.transfer_id, "transfer_id");
  entities::KVCacheTransferInfo& transfer =
      context.kv_cache_transfer(transfer_id);
  if (!event.payload.generation.has_value() ||
      event.payload.generation.value() !=
          transfer.source_generation()) {
    return;
  }
  transfer.mark_completed(event.time);
  entities::Request& request =
      context.request(transfer.request_id());
  request.on_kv_cache_transfer_complete(
      event.time, transfer.size_bytes());

  // Match Python mutation order: make the target arrival visible first, then
  // release the retained source allocation.
  request.on_arrival(event.time, ClusterType::kDecode);
  context.cluster(ClusterType::kDecode)
      .add_request(request.id(), request.arrived_at());
  const bool schedule_decode =
      context.on_decode_kv_arrival();

  scheduler::BaseReplicaScheduler& source =
      context.cluster(ClusterType::kPrefill)
          .get_replica_scheduler(
              transfer.source_replica_id(),
              transfer.source_dp_id());
  source.complete_kv_transfer(request.id());

  context.output().kv_cache_transfers.push_back(
      metrics::KVCacheTransferMetricsRecord{
          .transfer_id = transfer.id(),
          .request_id = transfer.request_id(),
          .source_batch_id = transfer.source_batch_id(),
          .source_cluster_type =
              transfer.source_cluster_type(),
          .target_cluster_type =
              transfer.target_cluster_type(),
          .source_replica_id =
              transfer.source_replica_id(),
          .source_dp_id = transfer.source_dp_id(),
          .size_bytes = transfer.size_bytes(),
          .predicted_time_ms =
              transfer.predicted_time_ms(),
          .started_at = transfer.started_at().value(),
          .completed_at = event.time,
      });

  if (schedule_decode) {
    EventPayload decode_payload;
    decode_payload.cluster_type = ClusterType::kDecode;
    context.event_queue().push(
        event.time,
        EventType::kClusterSchedule,
        std::move(decode_payload));
  }
  // Python only emits this wake-up when a request is waiting to be
  // scheduled and the source target has no batch in flight.  A request that
  // is already running is not pending work for this purpose.
  if (source.waiting_count() > 0 &&
      source.in_flight_batch_count() == 0) {
    context.event_queue().push(
        event.time,
        EventType::kReplicaSchedule,
        target_payload(
            scheduler::ReplicaTarget{
                .replica_id = transfer.source_replica_id(),
                .dp_id = transfer.source_dp_id(),
            },
            ClusterType::kPrefill));
  }
}

void EventDispatcher::dispatch(
    const Event& event,
    simulator::SimulationContext& context) const {
  const BaseEventHandler* handler = nullptr;
  switch (event.type) {
    case EventType::kRequestArrival:
      handler = &request_arrival_;
      break;
    case EventType::kGlobalSchedule:
      handler = &global_schedule_;
      break;
    case EventType::kClusterSchedule:
      handler = &cluster_schedule_;
      break;
    case EventType::kReplicaSchedule:
      handler = &replica_schedule_;
      break;
    case EventType::kBatchStageArrival:
      handler = &batch_stage_arrival_;
      break;
    case EventType::kReplicaStageSchedule:
      handler = &replica_stage_schedule_;
      break;
    case EventType::kBatchStageEnd:
      handler = &batch_stage_end_;
      break;
    case EventType::kClusterBatchEnd:
      handler = &cluster_batch_end_;
      break;
    case EventType::kGlobalBatchEnd:
      handler = &global_batch_end_;
      break;
    case EventType::kKvCacheTransferStart:
      handler = &kv_cache_transfer_start_;
      break;
    case EventType::kKvCacheTransferEnd:
      handler = &kv_cache_transfer_end_;
      break;
    case EventType::kFoundationCompletion:
    case EventType::kSchedulerPoll:
    case EventType::kBatchCompletion:
      throw std::logic_error(
        "legacy event leaked into typed dispatcher");
  }
  if (handler == nullptr || handler->type() != event.type) {
    throw std::logic_error("event dispatcher registry mismatch");
  }
  handler->handle(event, context);
}

}  // namespace frontier::events
