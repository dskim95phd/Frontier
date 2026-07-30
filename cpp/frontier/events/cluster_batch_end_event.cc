#include "frontier/events/event_handlers.h"

#include "frontier/events/event_helpers.h"
#include "frontier/simulator/simulator.h"

namespace frontier::events {

void handle_event(const ClusterBatchEndPayload &payload, SimTime time,
                  simulator::Simulator &simulator) {
    if (payload.cluster_type == ClusterType::kPrefill) {
        const scheduler::ReplicaTarget target =
            make_target(payload.replica_id, payload.dp_id);
        entities::Batch &batch = simulator.batch(payload.batch_id);
        if (payload.generation != batch.schedule_epoch()) {
            return;
        }
        scheduler::BaseReplicaScheduler &replica =
            simulator.cluster(payload.cluster_type)
                .get_replica_scheduler(target.replica_id, target.dp_id);
        if (!replica.on_batch_completed(batch, time)) {
            return;
        }
        simulator.metrics().record_batch(
            batch, simulator.requests(),
            simulator.predicted_batch_ms(payload.batch_id),
            simulator.runtime_config(payload.cluster_type));
        for (const entities::RequestBatchSnapshot &snapshot :
             batch.requests()) {
            const entities::Request &request =
                simulator.request(snapshot.request_id);
            if (!request.is_prefill_complete() ||
                request.state() != entities::RequestState::kTransferPending) {
                continue;
            }
            const TransferId transfer_id = simulator.create_kv_cache_transfer(
                snapshot.request_id, payload.batch_id, target);
            simulator.event_queue().push(time, [&]() {
                KVCacheTransferStartPayload value{};
                value.transfer_id = transfer_id;
                value.request_id = snapshot.request_id;
                value.batch_id = payload.batch_id;
                value.replica_id = target.replica_id;
                value.dp_id = target.dp_id;
                value.generation = batch.schedule_epoch();
                value.cluster_type = ClusterType::kDecode;
                return value;
            }());
        }
        simulator.event_queue().push(time, [&]() {
            ReplicaSchedulePayload value{};
            value.replica_id = target.replica_id;
            value.dp_id = target.dp_id;
            value.cluster_type = ClusterType::kPrefill;
            return value;
        }());
        return;
    }
    simulator.event_queue().push(time, [&]() {
        GlobalBatchEndPayload value{};
        value.batch_id = payload.batch_id;
        value.replica_id = payload.replica_id;
        value.dp_id = payload.dp_id;
        value.generation = payload.generation;
        value.cluster_type = payload.cluster_type;
        return value;
    }());
}

} // namespace frontier::events
