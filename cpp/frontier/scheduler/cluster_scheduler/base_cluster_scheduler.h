#pragma once

#include <memory>
#include <stdexcept>
#include <vector>

#include "frontier/core/ids.h"
#include "frontier/scheduler/replica_scheduler/base_replica_scheduler.h"
#include "frontier/scheduler/scheduler_types.h"

namespace frontier::scheduler {

class ClusterSchedulerError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

struct ClusterRequestAssignment {
  ReplicaId replica_id;
  DataParallelId dp_id;
  RequestId request_id;
};

class BaseClusterScheduler {
 public:
  virtual ~BaseClusterScheduler() = default;

  BaseClusterScheduler(const BaseClusterScheduler&) = delete;
  BaseClusterScheduler& operator=(const BaseClusterScheduler&) = delete;
  BaseClusterScheduler(BaseClusterScheduler&&) = delete;
  BaseClusterScheduler& operator=(BaseClusterScheduler&&) = delete;

  [[nodiscard]] virtual ClusterType cluster_type()
      const noexcept = 0;
  virtual void add_request(RequestId request_id) = 0;
  [[nodiscard]] virtual std::vector<ClusterRequestAssignment>
  schedule() = 0;
  [[nodiscard]] virtual bool empty() const noexcept = 0;
  [[nodiscard]] virtual std::vector<
      std::pair<ReplicaId, DataParallelId>>
  targets() const = 0;
  [[nodiscard]] virtual BaseReplicaScheduler& get_replica_scheduler(
      ReplicaId replica_id,
      DataParallelId dp_id) = 0;
  [[nodiscard]] virtual const BaseReplicaScheduler&
  get_replica_scheduler(
      ReplicaId replica_id,
      DataParallelId dp_id) const = 0;

 protected:
  BaseClusterScheduler() = default;
};

}  // namespace frontier::scheduler
