#include "frontier/events/event_handlers.h"

#include "frontier/simulator/simulator.h"

namespace frontier::events {

void handle_event(const BatchStageEndPayload &payload, SimTime time,
                  simulator::Simulator &simulator) {
    entities::Batch &batch = simulator.batch(payload.batch_id);
    if (payload.generation != batch.schedule_epoch()) {
        return;
    }
    scheduler::ReplicaStageScheduler &stage =
        simulator.cluster(payload.cluster_type)
            .get_replica_scheduler(payload.replica_id, payload.dp_id)
            .get_replica_stage_scheduler(payload.stage_id);
    stage.on_stage_end(payload.batch_id);
    entities::BatchStage &batch_stage =
        simulator.batch_stage(payload.batch_id, payload.stage_id);
    if (batch.moe_sync_group_id().valid()) {
        batch_stage.reconcile_synchronization_wait(time);
    }
    batch_stage.mark_completed(time);
    simulator.metrics().record_batch_stage(
        batch_stage, batch, simulator.runtime_config(payload.cluster_type));

    if (!stage.empty()) {
        simulator.event_queue().push(time, [&]() {
            ReplicaStageSchedulePayload value{};
            value.replica_id = payload.replica_id;
            value.dp_id = payload.dp_id;
            value.stage_id = payload.stage_id;
            value.cluster_type = payload.cluster_type;
            return value;
        }());
    }

    if (stage.is_last_stage()) {
        simulator.event_queue().push(time, [&]() {
            ClusterBatchEndPayload value{};
            value.batch_id = payload.batch_id;
            value.replica_id = payload.replica_id;
            value.dp_id = payload.dp_id;
            value.generation = batch.schedule_epoch();
            value.cluster_type = payload.cluster_type;
            return value;
        }());
    } else {
        simulator.event_queue().push(time, [&]() {
            BatchStageArrivalPayload value{};
            value.batch_id = payload.batch_id;
            value.replica_id = payload.replica_id;
            value.dp_id = payload.dp_id;
            value.stage_id = StageId{payload.stage_id.value() + 1};
            value.generation = batch.schedule_epoch();
            value.cluster_type = payload.cluster_type;
            return value;
        }());
    }
}

} // namespace frontier::events
