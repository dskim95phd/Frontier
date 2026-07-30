#include "frontier/events/event_handlers.h"

#include "frontier/moe/barrier_coordinator.h"
#include "frontier/simulator/simulation_context.h"

namespace frontier::events {

void handle_event(
    const DecodeSyncPayload& payload,
    SimTime time,
    simulator::SimulationContext& context) {
  const entities::Batch& batch = context.batch(payload.batch_id);
  if (batch.schedule_epoch() != payload.generation) {
    return;
  }
  if (batch.is_idle() != payload.is_idle) {
    throw moe::BarrierError(
        "decode synchronization idle marker disagrees with batch");
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
      moe::SyncPath::kDecode,
      payload.participant_id,
      time,
      context);
  const config::ParallelismConfig& parallelism =
      context.parallelism(payload.cluster_type);
  const std::uint64_t expected_participants =
      payload.cluster_type == ClusterType::kMonolithic
      ? parallelism.moe_expert_parallel_size
      : parallelism.data_parallel_size;
  auto ready = cluster_scheduler.moe_barrier().arrive(
      key,
      moe::BarrierParticipant{
          .participant_id = payload.participant_id,
          .batch_id = payload.batch_id,
          .arrival_time = time,
          .elapsed_component_ms = payload.elapsed_component_ms,
          .is_idle = payload.is_idle,
      },
      expected_participants);
  if (!ready.has_value() &&
      payload.sync_phase == moe::SyncPhase::kPreMoe) {
    ready = cluster_scheduler.compact_moe_group_participants(
        key, moe::SyncPath::kDecode, time, context);
  }
  if (ready.has_value()) {
    context.event_queue().push(
        ready->collective_time,
        DecodeSyncCollectivePayload{
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
