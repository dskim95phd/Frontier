#include "frontier/events/event_handlers.h"

#include "frontier/simulator/simulation_context.h"

namespace frontier::events {

void handle_event(
    const BatchStageEndPayload& payload,
    SimTime time,
    simulator::SimulationContext& context) {
  entities::Batch& batch = context.batch(payload.batch_id);
  if (payload.generation != batch.schedule_epoch()) {
    return;
  }
  scheduler::ReplicaStageScheduler& stage =
      context.cluster(payload.cluster_type)
          .get_replica_scheduler(
              payload.replica_id, payload.dp_id)
          .get_replica_stage_scheduler(payload.stage_id);
  stage.on_stage_end(payload.batch_id);
  entities::BatchStage& batch_stage =
      context.batch_stage(payload.batch_id, payload.stage_id);
  if (batch.moe_sync_group_id().valid()) {
    batch_stage.reconcile_synchronization_wait(time);
  }
  batch_stage.mark_completed(time);
  context.metrics().record_batch_stage(
      batch_stage,
      batch,
      context.runtime_config(payload.cluster_type));

  if (!stage.empty()) {
    context.event_queue().push(
        time,
        ReplicaStageSchedulePayload{
            .replica_id = payload.replica_id,
            .dp_id = payload.dp_id,
            .stage_id = payload.stage_id,
            .cluster_type = payload.cluster_type,
        });
  }

  if (stage.is_last_stage()) {
    context.event_queue().push(
        time,
        ClusterBatchEndPayload{
            .batch_id = payload.batch_id,
            .replica_id = payload.replica_id,
            .dp_id = payload.dp_id,
            .generation = batch.schedule_epoch(),
            .cluster_type = payload.cluster_type,
        });
  } else {
    context.event_queue().push(
        time,
        BatchStageArrivalPayload{
            .batch_id = payload.batch_id,
            .replica_id = payload.replica_id,
            .dp_id = payload.dp_id,
            .stage_id =
                StageId{payload.stage_id.value() + 1},
            .generation = batch.schedule_epoch(),
            .cluster_type = payload.cluster_type,
        });
  }
}

}  // namespace frontier::events
