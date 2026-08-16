#include "frontier/events/event_handlers.h"

#include <stdexcept>

#include "frontier/simulator/simulator.h"

namespace frontier::events {

void handle_event(const CpuKVCacheRestoreEndPayload &payload, SimTime time,
                  simulator::Simulator &simulator) {
    scheduler::BaseReplicaScheduler &scheduler =
        simulator.cluster(payload.cluster_type)
            .get_replica_scheduler(payload.replica_id, payload.dp_id);
    if (!scheduler.on_cpu_kv_cache_restore_end(
            payload.transfer_id, payload.generation, time)) {
        return;
    }
    auto operation = scheduler.take_completed_cpu_kv_cache_restore(
        payload.transfer_id);
    if (!operation.has_value()) {
        throw std::runtime_error(
            "completed CPU KV-cache restore operation disappeared");
    }
    simulator.metrics().record_cpu_kv_cache_restore(*operation,
                                                     payload.cluster_type);
    simulator.event_queue().push(time, [&]() {
        ReplicaSchedulePayload value{};
        value.replica_id = payload.replica_id;
        value.dp_id = payload.dp_id;
        value.cluster_type = payload.cluster_type;
        return value;
    }());
}

} // namespace frontier::events
