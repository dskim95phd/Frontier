#include "frontier/events/event_helpers.h"

namespace frontier::events {

scheduler::ReplicaTarget make_target(ReplicaId replica_id,
                                     DataParallelId dp_id) noexcept {
    return [&]() {
        scheduler::ReplicaTarget value{};
        value.replica_id = replica_id;
        value.dp_id = dp_id;
        return value;
    }();
}

} // namespace frontier::events
