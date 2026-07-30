#include "frontier/events/event_handlers.h"

#include "frontier/events/event_helpers.h"
#include "frontier/simulator/simulation_context.h"

namespace frontier::events {

void handle_event(
    const GlobalBatchEndPayload& payload,
    SimTime time,
    simulator::SimulationContext& context) {
  const scheduler::ReplicaTarget target =
      make_target(payload.replica_id, payload.dp_id);
  entities::Batch& batch = context.batch(payload.batch_id);
  if (payload.generation != batch.schedule_epoch()) {
    return;
  }
  scheduler::BaseReplicaScheduler& replica =
      context.cluster(payload.cluster_type)
          .get_replica_scheduler(
              target.replica_id, target.dp_id);
  static_cast<void>(replica.on_batch_completed(batch, time));
  context.output().batches.push_back(make_batch_metrics(
      batch,
      context.requests(),
      context.predicted_batch_ms(payload.batch_id)));
  for (const entities::RequestBatchSnapshot& snapshot :
       batch.requests()) {
    if (context.request(snapshot.request_id).completed()) {
      context.record_request_completion(snapshot.request_id);
    }
  }
  context.event_queue().push(
      time,
      ReplicaSchedulePayload{
          .replica_id = target.replica_id,
          .dp_id = target.dp_id,
          .cluster_type = payload.cluster_type,
      });
}

}  // namespace frontier::events
