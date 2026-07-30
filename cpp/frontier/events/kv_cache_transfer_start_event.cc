#include "frontier/events/event_handlers.h"

#include <cmath>
#include <stdexcept>

#include "frontier/simulator/simulation_context.h"

namespace frontier::events {

void handle_event(
    const KVCacheTransferStartPayload& payload,
    SimTime time,
    simulator::SimulationContext& context) {
  entities::KVCacheTransferInfo& transfer =
      context.kv_cache_transfer(payload.transfer_id);
  if (payload.generation != transfer.source_generation()) {
    return;
  }
  transfer.mark_started(time);
  context.request(transfer.request_id())
      .on_kv_cache_transfer_start(time);
  const double completion_seconds =
      time.seconds() +
      transfer.predicted_time_ms() * 1e-3;
  if (!std::isfinite(completion_seconds)) {
    throw std::runtime_error(
        "KV transfer completion time is nonfinite");
  }
  context.event_queue().push(
      SimTime::from_seconds(completion_seconds),
      KVCacheTransferEndPayload{
          .transfer_id = payload.transfer_id,
          .request_id = payload.request_id,
          .batch_id = payload.batch_id,
          .replica_id = payload.replica_id,
          .dp_id = payload.dp_id,
          .generation = payload.generation,
          .cluster_type = payload.cluster_type,
      });
}

}  // namespace frontier::events
