#include "frontier/events/event_handlers.h"

#include "frontier/simulator/simulation_context.h"

namespace frontier::events {

void handle_event(
    const KVCacheTransferEndPayload& payload,
    SimTime time,
    simulator::SimulationContext& context) {
  entities::KVCacheTransferInfo& transfer =
      context.kv_cache_transfer(payload.transfer_id);
  if (payload.generation != transfer.source_generation()) {
    return;
  }
  transfer.mark_completed(time);
  entities::Request& request =
      context.request(transfer.request_id());
  request.on_kv_cache_transfer_complete(
      time, transfer.size_bytes());

  // Match Python mutation order: expose target arrival before releasing the
  // retained source allocation.
  request.on_arrival(time, ClusterType::kDecode);
  context.cluster(ClusterType::kDecode)
      .add_request(request.id(), request.arrived_at());
  const bool schedule_decode = context.on_decode_kv_arrival();

  scheduler::BaseReplicaScheduler& source =
      context.cluster(ClusterType::kPrefill)
          .get_replica_scheduler(
              transfer.source_replica_id(),
              transfer.source_dp_id());
  source.complete_kv_transfer(request.id());

  context.metrics().record_kv_cache_transfer(transfer);

  if (schedule_decode) {
    context.event_queue().push(
        time,
        ClusterSchedulePayload{
            .cluster_type = ClusterType::kDecode,
        });
  }
  if (source.waiting_count() > 0 &&
      source.in_flight_batch_count() == 0) {
    context.event_queue().push(
        time,
        ReplicaSchedulePayload{
            .replica_id = transfer.source_replica_id(),
            .dp_id = transfer.source_dp_id(),
            .cluster_type = ClusterType::kPrefill,
        });
  }
}

}  // namespace frontier::events
