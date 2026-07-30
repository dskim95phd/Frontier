#include "frontier/events/event_handlers.h"

#include "frontier/events/event_helpers.h"
#include "frontier/simulator/simulator.h"

namespace frontier::events {

void handle_event(const ReplicaSchedulePayload &payload, SimTime time,
                  simulator::Simulator &simulator) {
    const scheduler::ReplicaTarget target =
        make_target(payload.replica_id, payload.dp_id);
    scheduler::BaseReplicaScheduler &replica =
        simulator.cluster(payload.cluster_type)
            .get_replica_scheduler(target.replica_id, target.dp_id);
    bool returned_empty_schedule = false;
    bool produced_batch = false;
    while (replica.in_flight_batch_count() < replica.pipeline_parallel_size()) {
        scheduler::ScheduleResult schedule = replica.schedule(time);
        simulator.metrics().record_scheduler_trace(schedule, target,
                                                   payload.cluster_type);
        if (schedule.scheduled_requests.empty()) {
            returned_empty_schedule = true;
            break;
        }
        const BatchId batch_id =
            simulator.create_batch(schedule, target, payload.cluster_type);
        produced_batch = true;
        entities::Batch &batch = simulator.batch(batch_id);
        replica.mark_batch_started(batch);
        simulator.event_queue().push(time, [&]() {
            BatchStageArrivalPayload value{};
            value.batch_id = batch_id;
            value.replica_id = target.replica_id;
            value.dp_id = target.dp_id;
            value.stage_id = StageId{0};
            value.generation = batch.schedule_epoch();
            value.cluster_type = payload.cluster_type;
            return value;
        }());
    }
    if (!produced_batch && returned_empty_schedule &&
        replica.consume_terminal_release_followup_poll()) {
        simulator.event_queue().push(time, payload);
    }
}

} // namespace frontier::events
