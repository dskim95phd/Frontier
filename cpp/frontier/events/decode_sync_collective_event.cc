#include "frontier/events/event_handlers.h"

#include "frontier/scheduler/cluster_scheduler/base_cluster_scheduler.h"
#include "frontier/simulator/simulator.h"

namespace frontier::events {

void handle_event(const DecodeSyncCollectivePayload &payload, SimTime time,
                  simulator::Simulator &simulator) {
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
    const auto participants = cluster_scheduler.moe_barrier().consume(key);
    cluster_scheduler.continue_moe_stage(key, scheduler::MoESyncPath::kDecode,
                                         participants, time, simulator);
}

} // namespace frontier::events
