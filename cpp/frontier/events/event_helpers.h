#pragma once

#include "frontier/core/ids.h"
#include "frontier/scheduler/scheduler_types.h"

namespace frontier::events {

[[nodiscard]] scheduler::ReplicaTarget
make_target(ReplicaId replica_id, DataParallelId dp_id) noexcept;

} // namespace frontier::events
