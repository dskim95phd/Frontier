#pragma once

#include <vector>

#include "frontier/config/config.h"
#include "frontier/entities/replica.h"

namespace frontier::entities {

class Cluster {
 public:
  explicit Cluster(const config::ParallelismConfig& parallelism);

  [[nodiscard]] const std::vector<Replica>& replicas() const noexcept {
    return replicas_;
  }
  [[nodiscard]] const Replica& replica(ReplicaId replica_id) const;

 private:
  std::vector<Replica> replicas_;
};

}  // namespace frontier::entities
