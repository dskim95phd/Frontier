#include "frontier/scheduler/cluster_scheduler/base_cluster_scheduler.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <utility>

#include "frontier/entities/batch.h"
#include "frontier/entities/cluster.h"
#include "frontier/entities/request.h"
#include "frontier/execution_time_predictor/base_execution_time_predictor.h"
#include "frontier/simulator/simulator.h"

namespace frontier::scheduler {

void BaseClusterScheduler::on_prefill_sync(const PrefillSyncPayload &payload,
                                           SimTime time,
                                           simulator::Simulator &simulator) {
    const entities::Batch &batch = simulator.batch(payload.batch_id);
    if (batch.schedule_epoch() != payload.generation) {
        return;
    }
    if (batch.is_idle() != payload.is_idle) {
        throw MoEBarrierError(
            "prefill synchronization idle marker disagrees with batch");
    }
    const MoEBarrierKey key{payload.cluster_type,   payload.replica_id,
                            payload.stage_id,       payload.sync_group_id,
                            payload.layer_id,       payload.sync_phase,
                            payload.sync_generation};
    ensure_moe_group_participants(key, MoESyncPath::kPrefill,
                                  payload.participant_id, time, simulator);
    const auto ready = moe_barrier_.arrive(
        key,
        MoEBarrierParticipant{payload.participant_id, payload.batch_id, time,
                              payload.elapsed_component_ms, payload.is_idle},
        simulator.parallelism(payload.cluster_type).data_parallel_size);
    if (ready.has_value()) {
        simulator.event_queue().push(
            ready->collective_time,
            PrefillSyncCollectivePayload{
                payload.replica_id, payload.stage_id, payload.sync_group_id,
                payload.layer_id, payload.sync_phase, payload.sync_generation,
                payload.cluster_type});
    }
}

void BaseClusterScheduler::on_prefill_sync_collective(
    const PrefillSyncCollectivePayload &payload, SimTime time,
    simulator::Simulator &simulator) {
    const MoEBarrierKey key{payload.cluster_type,   payload.replica_id,
                            payload.stage_id,       payload.sync_group_id,
                            payload.layer_id,       payload.sync_phase,
                            payload.sync_generation};
    const auto participants = moe_barrier_.consume(key);
    continue_moe_stage(key, MoESyncPath::kPrefill, participants, time,
                       simulator);
}

void BaseClusterScheduler::on_decode_sync(const DecodeSyncPayload &payload,
                                          SimTime time,
                                          simulator::Simulator &simulator) {
    const entities::Batch &batch = simulator.batch(payload.batch_id);
    if (batch.schedule_epoch() != payload.generation) {
        return;
    }
    if (batch.is_idle() != payload.is_idle) {
        throw MoEBarrierError(
            "decode synchronization idle marker disagrees with batch");
    }
    const MoEBarrierKey key{payload.cluster_type,   payload.replica_id,
                            payload.stage_id,       payload.sync_group_id,
                            payload.layer_id,       payload.sync_phase,
                            payload.sync_generation};
    ensure_moe_group_participants(key, MoESyncPath::kDecode,
                                  payload.participant_id, time, simulator);
    const std::uint64_t expected_participants =
        simulator.parallelism(payload.cluster_type).data_parallel_size;
    auto ready = moe_barrier_.arrive(
        key,
        MoEBarrierParticipant{payload.participant_id, payload.batch_id, time,
                              payload.elapsed_component_ms, payload.is_idle},
        expected_participants);
    if (!ready.has_value() && payload.sync_phase == MoESyncPhase::kPreMoe) {
        ready = compact_moe_group_participants(key, MoESyncPath::kDecode, time,
                                               simulator);
    }
    if (ready.has_value()) {
        simulator.event_queue().push(
            ready->collective_time,
            DecodeSyncCollectivePayload{
                payload.replica_id, payload.stage_id, payload.sync_group_id,
                payload.layer_id, payload.sync_phase, payload.sync_generation,
                payload.cluster_type});
    }
}

void BaseClusterScheduler::on_decode_sync_collective(
    const DecodeSyncCollectivePayload &payload, SimTime time,
    simulator::Simulator &simulator) {
    const MoEBarrierKey key{payload.cluster_type,   payload.replica_id,
                            payload.stage_id,       payload.sync_group_id,
                            payload.layer_id,       payload.sync_phase,
                            payload.sync_generation};
    const auto participants = moe_barrier_.consume(key);
    continue_moe_stage(key, MoESyncPath::kDecode, participants, time,
                       simulator);
}

} // namespace frontier::scheduler
