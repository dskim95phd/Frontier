#include "frontier/events/event_handlers.h"

#include "frontier/moe/barrier_coordinator.h"
#include "frontier/simulator/simulation_context.h"

namespace frontier::events {

void handle_event(
    const PrefillSyncCollectivePayload& payload,
    SimTime time,
    simulator::SimulationContext& context) {
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
  const auto participants =
      cluster_scheduler.moe_barrier().consume(key);
  cluster_scheduler.continue_moe_stage(
      key, moe::SyncPath::kPrefill, participants, time, context);
}

}  // namespace frontier::events
