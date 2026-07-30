#include "frontier/events/event_handlers.h"

#include "frontier/events/event_helpers.h"
#include "frontier/simulator/simulation_context.h"

namespace frontier::events {

void handle_event(
    const BatchStageArrivalPayload& payload,
    SimTime time,
    simulator::SimulationContext& context) {
  entities::Batch& batch = context.batch(payload.batch_id);
  if (payload.generation != batch.schedule_epoch()) {
    return;
  }
  context.record_stage_arrival(
      payload.batch_id, payload.stage_id, time);
  scheduler::ReplicaStageScheduler& stage =
      context.cluster(payload.cluster_type)
          .get_replica_scheduler(
              payload.replica_id, payload.dp_id)
          .get_replica_stage_scheduler(payload.stage_id);
  stage.add_batch(batch);
  if (!stage.is_busy()) {
    context.event_queue().push(
        time,
        ReplicaStageSchedulePayload{
            .replica_id = payload.replica_id,
            .dp_id = payload.dp_id,
            .stage_id = payload.stage_id,
            .cluster_type = payload.cluster_type,
        });
  }
}

}  // namespace frontier::events
