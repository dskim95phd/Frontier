#pragma once

#include <cstdint>

#include "frontier/core/cluster_type.h"
#include "frontier/core/ids.h"

namespace frontier::scheduler {

using ::frontier::ClusterType;

struct ReplicaTarget {
    ReplicaId replica_id;
    DataParallelId dp_id;

    friend bool operator==(const ReplicaTarget &lhs, const ReplicaTarget &rhs) {
        return lhs.replica_id == rhs.replica_id && lhs.dp_id == rhs.dp_id;
    }
    friend bool operator!=(const ReplicaTarget &lhs, const ReplicaTarget &rhs) {
        return !(lhs == rhs);
    }
};

} // namespace frontier::scheduler
