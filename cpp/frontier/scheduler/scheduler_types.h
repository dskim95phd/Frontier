#pragma once

#include <cstdint>

#include "frontier/core/ids.h"

namespace frontier::scheduler {

enum class ClusterType : std::uint8_t {
  kMonolithic,
};

struct ReplicaTarget {
  ReplicaId replica_id;
  DataParallelId dp_id;

  friend bool operator==(
      const ReplicaTarget&,
      const ReplicaTarget&) = default;
};

}  // namespace frontier::scheduler
