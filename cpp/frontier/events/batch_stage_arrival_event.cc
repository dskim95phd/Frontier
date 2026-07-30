#include "frontier/events/event_handlers.h"

#include "frontier/events/event_helpers.h"
#include "frontier/simulator/simulator.h"

namespace frontier::events {

void handle_event(const BatchStageArrivalPayload &payload, SimTime time,
                  simulator::Simulator &simulator) {
    entities::Batch &batch = simulator.batch(payload.batch_id);
    if (payload.generation != batch.schedule_epoch()) {
        return;
    }
    simulator.record_stage_arrival(payload.batch_id, payload.stage_id, time);
    scheduler::ReplicaStageScheduler &stage =
        simulator.cluster(payload.cluster_type)
            .get_replica_scheduler(payload.replica_id, payload.dp_id)
            .get_replica_stage_scheduler(payload.stage_id);
    stage.add_batch(batch);
    if (!stage.is_busy()) {
        simulator.event_queue().push(time, [&]() {
            ReplicaStageSchedulePayload value{};
            value.replica_id = payload.replica_id;
            value.dp_id = payload.dp_id;
            value.stage_id = payload.stage_id;
            value.cluster_type = payload.cluster_type;
            return value;
        }());
    }
}

} // namespace frontier::events
