#include "frontier/events/event_handlers.h"

#include <cmath>
#include <stdexcept>

#include "frontier/simulator/simulator.h"

namespace frontier::events {

void handle_event(const SyntheticDecodeEndPayload &payload, SimTime time,
                  simulator::Simulator &simulator) {
    const auto &prefill_only = simulator.config().prefill_only;
    if (!prefill_only.has_value()) {
        throw std::logic_error(
            "synthetic decode event requires PREFILL-only mode");
    }
    entities::Request &request = simulator.request(payload.request_id);
    const double first_token_seconds =
        request.decode_arrived_at().seconds() +
        1.0 / prefill_only->decode_tokens_per_second;
    if (!std::isfinite(first_token_seconds)) {
        throw std::overflow_error(
            "synthetic first-token completion time overflowed");
    }
    request.on_synthetic_decode_complete(
        SimTime::from_seconds(first_token_seconds), time);
    simulator.record_request_completion(payload.request_id, time);
}

} // namespace frontier::events
