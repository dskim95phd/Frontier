#include "frontier/entities/cluster.h"

#include <stdexcept>
#include <utility>

#include "frontier/cc_backend/analytical_model.h"

namespace frontier::entities {

Cluster::Cluster(ClusterType type, config::ClusterRuntimeConfig runtime_config)
    : type_(type), runtime_config_(std::move(runtime_config)),
      communication_backend_(cc_backend::make_analytical_cc_backend([&]() {
          cc_backend::AnalyticalCommunicationConfig value{};
          value.network_bandwidth_gbps =
              runtime_config_.execution_model.analytical.network_bandwidth_gbps;
          value.latency_us =
              runtime_config_.execution_model.analytical.network_latency_us;
          value.intra_node_bandwidth_gbps =
              runtime_config_.execution_model.analytical
                  .intra_node_bandwidth_gbps;
          return value;
      }())) {
    const config::ParallelismConfig &parallelism = runtime_config_.parallelism;
    replicas_.reserve(static_cast<std::size_t>(parallelism.num_replicas));
    for (std::uint64_t id = 0; id < parallelism.num_replicas; ++id) {
        replicas_.emplace_back(ReplicaId{id}, parallelism,
                               runtime_config_.model);
    }
}

const Replica &Cluster::replica(ReplicaId replica_id) const {
    if (!replica_id.valid() || replica_id.index() >= replicas_.size()) {
        throw std::out_of_range("unknown replica ID");
    }
    return replicas_.at(replica_id.index());
}

} // namespace frontier::entities
