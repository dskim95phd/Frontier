#include "frontier/events/event_handlers.h"

#include "frontier/events/event_helpers.h"
#include "frontier/simulator/simulation_context.h"

namespace frontier::events {

void handle_event(
    const ReplicaSchedulePayload& payload,
    SimTime time,
    simulator::SimulationContext& context) {
  const scheduler::ReplicaTarget target =
      make_target(payload.replica_id, payload.dp_id);
  scheduler::BaseReplicaScheduler& replica =
      context.cluster(payload.cluster_type).get_replica_scheduler(
          target.replica_id, target.dp_id);
  bool returned_empty_schedule = false;
  bool produced_batch = false;
  while (replica.in_flight_batch_count() <
         replica.pipeline_parallel_size()) {
    scheduler::ScheduleResult schedule = replica.schedule(time);
    context.metrics().record_scheduler_trace(
        schedule, target, payload.cluster_type);
    if (schedule.scheduled_requests.empty()) {
      returned_empty_schedule = true;
      break;
    }
    const BatchId batch_id =
        context.create_batch(
            schedule, target, payload.cluster_type);
    produced_batch = true;
    entities::Batch& batch = context.batch(batch_id);
    replica.mark_batch_started(batch);
    context.event_queue().push(
        time,
        BatchStageArrivalPayload{
            .batch_id = batch_id,
            .replica_id = target.replica_id,
            .dp_id = target.dp_id,
            .stage_id = StageId{0},
            .generation = batch.schedule_epoch(),
            .cluster_type = payload.cluster_type,
        });
  }
  if (!produced_batch && returned_empty_schedule &&
      replica.consume_terminal_release_followup_poll()) {
    context.event_queue().push(time, payload);
  }
}

}  // namespace frontier::events
