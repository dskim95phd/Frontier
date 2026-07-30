#include "frontier/events/event_handlers.h"

#include "frontier/scheduler/cluster_scheduler/base_cluster_scheduler.h"
#include "frontier/simulator/simulator.h"

namespace frontier::events {

void handle_event(const DecodeSyncPayload &payload, SimTime time,
                  simulator::Simulator &simulator) {
    const entities::Batch &batch = simulator.batch(payload.batch_id);
    if (batch.schedule_epoch() != payload.generation) {
        return;
    }
    if (batch.is_idle() != payload.is_idle) {
        throw scheduler::MoEBarrierError(
            "decode synchronization idle marker disagrees with batch");
    }
    const scheduler::MoEBarrierKey key = [&]() {
        scheduler::MoEBarrierKey value{};
        value.cluster_type = payload.cluster_type;
        value.replica_id = payload.replica_id;
        value.stage_id = payload.stage_id;
        value.sync_group_id = payload.sync_group_id;
        value.layer_id = payload.layer_id;
        value.phase = payload.sync_phase;
        value.generation = payload.sync_generation;
        return value;
    }();
    scheduler::BaseClusterScheduler &cluster_scheduler =
        simulator.cluster(payload.cluster_type);
    cluster_scheduler.ensure_moe_group_participants(
        key, scheduler::MoESyncPath::kDecode, payload.participant_id, time,
        simulator);
    const config::ParallelismConfig &parallelism =
        simulator.parallelism(payload.cluster_type);
    const std::uint64_t expected_participants =
        payload.cluster_type == ClusterType::kMonolithic
            ? parallelism.moe_expert_parallel_size
            : parallelism.data_parallel_size;
    auto ready = cluster_scheduler.moe_barrier().arrive(
        key,
        [&]() {
            scheduler::MoEBarrierParticipant value{};
            value.participant_id = payload.participant_id;
            value.batch_id = payload.batch_id;
            value.arrival_time = time;
            value.elapsed_component_ms = payload.elapsed_component_ms;
            value.is_idle = payload.is_idle;
            return value;
        }(),
        expected_participants);
    if (!ready.has_value() && payload.sync_phase == MoESyncPhase::kPreMoe) {
        ready = cluster_scheduler.compact_moe_group_participants(
            key, scheduler::MoESyncPath::kDecode, time, simulator);
    }
    if (ready.has_value()) {
        simulator.event_queue().push(ready->collective_time, [&]() {
            DecodeSyncCollectivePayload value{};
            value.replica_id = payload.replica_id;
            value.stage_id = payload.stage_id;
            value.sync_group_id = payload.sync_group_id;
            value.layer_id = payload.layer_id;
            value.sync_phase = payload.sync_phase;
            value.sync_generation = payload.sync_generation;
            value.cluster_type = payload.cluster_type;
            return value;
        }());
    }
}

} // namespace frontier::events
