#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "frontier/scheduler/cluster_scheduler/base_cluster_scheduler.h"

namespace frontier::scheduler {

class RoundRobinClusterScheduler final
    : public BaseClusterScheduler {
 public:
  RoundRobinClusterScheduler(
      std::vector<std::unique_ptr<BaseReplicaScheduler>>
          replica_schedulers,
      const entities::Cluster& cluster);

  [[nodiscard]] std::vector<ClusterRequestAssignment>
  schedule() override;

 private:
  std::uint64_t request_counter_ = 0;
};

}  // namespace frontier::scheduler
