#include "frontier/events/event_handlers.h"

#include "frontier/events/event_helpers.h"
#include "frontier/simulator/simulation_context.h"

namespace frontier::events {

void handle_event(
    const ClusterBatchEndPayload& payload,
    SimTime time,
    simulator::SimulationContext& context) {
  if (payload.cluster_type == ClusterType::kPrefill) {
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
    if (!replica.on_batch_completed(batch, time)) {
      return;
    }
    context.output().batches.push_back(make_batch_metrics(
        batch,
        context.requests(),
        context.predicted_batch_ms(payload.batch_id)));
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
              payload.batch_id,
              target);
      context.event_queue().push(
          time,
          KVCacheTransferStartPayload{
              .transfer_id = transfer_id,
              .request_id = snapshot.request_id,
              .batch_id = payload.batch_id,
              .replica_id = target.replica_id,
              .dp_id = target.dp_id,
              .generation = batch.schedule_epoch(),
              .cluster_type = ClusterType::kDecode,
          });
    }
    context.event_queue().push(
        time,
        ReplicaSchedulePayload{
            .replica_id = target.replica_id,
            .dp_id = target.dp_id,
            .cluster_type = ClusterType::kPrefill,
        });
    return;
  }
  context.event_queue().push(
      time,
      GlobalBatchEndPayload{
          .batch_id = payload.batch_id,
          .replica_id = payload.replica_id,
          .dp_id = payload.dp_id,
          .generation = payload.generation,
          .cluster_type = payload.cluster_type,
      });
}

}  // namespace frontier::events
