#include "frontier/events/event_helpers.h"

namespace frontier::events {

scheduler::ReplicaTarget make_target(
    ReplicaId replica_id,
    DataParallelId dp_id) noexcept {
  return scheduler::ReplicaTarget{
      .replica_id = replica_id,
      .dp_id = dp_id,
  };
}

}  // namespace frontier::events
