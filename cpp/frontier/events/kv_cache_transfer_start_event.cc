#include "frontier/events/event_handlers.h"

#include <cmath>
#include <stdexcept>

#include "frontier/simulator/simulator.h"

namespace frontier::events {

void handle_event(const KVCacheTransferStartPayload &payload, SimTime time,
                  simulator::Simulator &simulator) {
    entities::KVCacheTransferInfo &transfer =
        simulator.kv_cache_transfer(payload.transfer_id);
    if (payload.generation != transfer.source_generation()) {
        return;
    }
    transfer.mark_started(time);
    simulator.request(transfer.request_id()).on_kv_cache_transfer_start(time);
    const double completion_seconds =
        time.seconds() + transfer.predicted_time_ms() * 1e-3;
    if (!std::isfinite(completion_seconds)) {
        throw std::runtime_error("KV transfer completion time is nonfinite");
    }
    simulator.event_queue().push(SimTime::from_seconds(completion_seconds),
                                 [&]() {
                                     KVCacheTransferEndPayload value{};
                                     value.transfer_id = payload.transfer_id;
                                     value.request_id = payload.request_id;
                                     value.batch_id = payload.batch_id;
                                     value.replica_id = payload.replica_id;
                                     value.dp_id = payload.dp_id;
                                     value.generation = payload.generation;
                                     value.cluster_type = payload.cluster_type;
                                     return value;
                                 }());
}

} // namespace frontier::events
