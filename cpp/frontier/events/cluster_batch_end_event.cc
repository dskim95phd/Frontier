#include "frontier/events/event_handlers.h"

#include <utility>

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
        const bool has_valid_request = replica.on_batch_completed(batch, time);
        if (has_valid_request) {
            simulator.metrics().record_batch(
                batch, simulator.requests(),
                simulator.predicted_batch_ms(payload.batch_id),
                simulator.runtime_config(payload.cluster_type));
            for (const entities::RequestBatchSnapshot &snapshot :
                 batch.requests()) {
                const entities::Request &request =
                    simulator.request(snapshot.request_id);
                if (!request.is_prefill_complete() ||
                    request.state() !=
                        entities::RequestState::kTransferPending) {
                    continue;
                }
                static_cast<void>(replica.prepare_cpu_kv_cache_offload(
                    snapshot.request_id, time));
                for (scheduler::ScheduledAuxiliaryEvent &auxiliary :
                     replica.drain_auxiliary_events()) {
                    simulator.event_queue().push(
                        auxiliary.time, std::move(auxiliary.payload));
                }
                const TransferId transfer_id =
                    simulator.create_kv_cache_transfer(
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
        }
        // An all-stale prefill batch still left the replica's in-flight set in
        // on_batch_completed(). It owns no transfer or metrics record, but its
        // arena entity must be released and the target must be polled just like
        // a batch containing valid work.
        // KV transfer records and queued events own all source metadata needed
        // after this point. The scheduler retains the source KV allocation
        // independently until complete_kv_transfer(), so the completed batch
        // entity can be released immediately.
        simulator.release_batch(payload.batch_id);
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
