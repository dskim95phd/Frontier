#include "frontier/entities/cluster.h"

#include <stdexcept>

namespace frontier::entities {

Cluster::Cluster(const config::ParallelismConfig& parallelism) {
  replicas_.reserve(
      static_cast<std::size_t>(parallelism.num_replicas));
  for (std::uint64_t id = 0; id < parallelism.num_replicas; ++id) {
    replicas_.emplace_back(
        ReplicaId{id}, parallelism);
  }
}

const Replica& Cluster::replica(ReplicaId replica_id) const {
  if (replica_id.value() >= replicas_.size()) {
    throw std::out_of_range("unknown replica ID");
  }
  return replicas_.at(
      static_cast<std::size_t>(replica_id.value()));
}

}  // namespace frontier::entities
