#pragma once

#include <cstdint>

#include "frontier/core/cluster_type.h"
#include "frontier/core/ids.h"

namespace frontier::scheduler {

using ::frontier::ClusterType;

struct ReplicaTarget {
  ReplicaId replica_id;
  DataParallelId dp_id;

  friend bool operator==(
      const ReplicaTarget&,
      const ReplicaTarget&) = default;
};

}  // namespace frontier::scheduler
