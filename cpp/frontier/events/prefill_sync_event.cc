#include "frontier/events/event_handlers.h"

#include "frontier/moe/barrier_coordinator.h"
#include "frontier/simulator/simulation_context.h"

namespace frontier::events {

void handle_event(
    const PrefillSyncPayload& payload,
    SimTime time,
    simulator::SimulationContext& context) {
  const entities::Batch& batch = context.batch(payload.batch_id);
  if (batch.schedule_epoch() != payload.generation) {
    return;
  }
  if (batch.is_idle() != payload.is_idle) {
    throw moe::BarrierError(
        "prefill synchronization idle marker disagrees with batch");
  }
  const moe::BarrierKey key{
      .cluster_type = payload.cluster_type,
      .replica_id = payload.replica_id,
      .stage_id = payload.stage_id,
      .sync_group_id = payload.sync_group_id,
      .layer_id = payload.layer_id,
      .phase = payload.sync_phase,
      .generation = payload.sync_generation,
  };
  scheduler::BaseClusterScheduler& cluster_scheduler =
      context.cluster(payload.cluster_type);
  cluster_scheduler.ensure_moe_group_participants(
      key,
      moe::SyncPath::kPrefill,
      payload.participant_id,
      time,
      context);
  const auto ready = cluster_scheduler.moe_barrier().arrive(
      key,
      moe::BarrierParticipant{
          .participant_id = payload.participant_id,
          .batch_id = payload.batch_id,
          .arrival_time = time,
          .elapsed_component_ms = payload.elapsed_component_ms,
          .is_idle = payload.is_idle,
      },
      context.parallelism(payload.cluster_type).data_parallel_size);
  if (ready.has_value()) {
    context.event_queue().push(
        ready->collective_time,
        PrefillSyncCollectivePayload{
            .replica_id = payload.replica_id,
            .stage_id = payload.stage_id,
            .sync_group_id = payload.sync_group_id,
            .layer_id = payload.layer_id,
            .sync_phase = payload.sync_phase,
            .sync_generation = payload.sync_generation,
            .cluster_type = payload.cluster_type,
        });
  }
}

}  // namespace frontier::events
