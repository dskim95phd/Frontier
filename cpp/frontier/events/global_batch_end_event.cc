#include "frontier/events/event_handlers.h"

#include "frontier/events/event_helpers.h"
#include "frontier/simulator/simulator.h"

namespace frontier::events {

void handle_event(const GlobalBatchEndPayload &payload, SimTime time,
                  simulator::Simulator &simulator) {
    const scheduler::ReplicaTarget target =
        make_target(payload.replica_id, payload.dp_id);
    entities::Batch &batch = simulator.batch(payload.batch_id);
    if (payload.generation != batch.schedule_epoch()) {
        return;
    }
    scheduler::BaseReplicaScheduler &replica =
        simulator.cluster(payload.cluster_type)
            .get_replica_scheduler(target.replica_id, target.dp_id);
    static_cast<void>(replica.on_batch_completed(batch, time));
    simulator.metrics().record_batch(
        batch, simulator.requests(),
        simulator.predicted_batch_ms(payload.batch_id),
        simulator.runtime_config(payload.cluster_type));
    for (const entities::RequestBatchSnapshot &snapshot : batch.requests()) {
        if (simulator.request(snapshot.request_id).completed()) {
            simulator.record_request_completion(snapshot.request_id);
        }
    }
    simulator.event_queue().push(time, [&]() {
        ReplicaSchedulePayload value{};
        value.replica_id = target.replica_id;
        value.dp_id = target.dp_id;
        value.cluster_type = payload.cluster_type;
        return value;
    }());
}

} // namespace frontier::events
