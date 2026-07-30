#include "frontier/events/event_handlers.h"

#include "frontier/simulator/simulator.h"

namespace frontier::events {

void handle_event(const KVCacheTransferEndPayload &payload, SimTime time,
                  simulator::Simulator &simulator) {
    entities::KVCacheTransferInfo &transfer =
        simulator.kv_cache_transfer(payload.transfer_id);
    if (payload.generation != transfer.source_generation()) {
        return;
    }
    transfer.mark_completed(time);
    entities::Request &request = simulator.request(transfer.request_id());
    request.on_kv_cache_transfer_complete(time, transfer.size_bytes());

    // Match Python mutation order: expose target arrival before releasing the
    // retained source allocation.
    request.on_arrival(time, ClusterType::kDecode);
    simulator.cluster(ClusterType::kDecode)
        .add_request(request.id(), request.arrived_at());
    const bool schedule_decode = simulator.on_decode_kv_arrival();

    scheduler::BaseReplicaScheduler &source =
        simulator.cluster(ClusterType::kPrefill)
            .get_replica_scheduler(transfer.source_replica_id(),
                                   transfer.source_dp_id());
    source.complete_kv_transfer(request.id());

    simulator.metrics().record_kv_cache_transfer(transfer);

    if (schedule_decode) {
        simulator.event_queue().push(time, [&]() {
            ClusterSchedulePayload value{};
            value.cluster_type = ClusterType::kDecode;
            return value;
        }());
    }
    if (source.waiting_count() > 0 && source.in_flight_batch_count() == 0) {
        simulator.event_queue().push(time, [&]() {
            ReplicaSchedulePayload value{};
            value.replica_id = transfer.source_replica_id();
            value.dp_id = transfer.source_dp_id();
            value.cluster_type = ClusterType::kPrefill;
            return value;
        }());
    }
}

} // namespace frontier::events
