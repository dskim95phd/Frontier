#pragma once

#include <map>
#include <memory>
#include <stdexcept>
#include <vector>

#include "frontier/core/ids.h"
#include "frontier/scheduler/cluster_scheduler/base_cluster_scheduler.h"
#include "frontier/scheduler/scheduler_types.h"

namespace frontier::scheduler {

class GlobalSchedulerError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

struct GlobalRequestAssignment {
  RequestId request_id;
  ClusterType cluster_type;
};

class GlobalScheduler final {
 public:
  explicit GlobalScheduler(
      std::unique_ptr<BaseClusterScheduler> cluster_scheduler);
  explicit GlobalScheduler(
      std::vector<std::unique_ptr<BaseClusterScheduler>>
          cluster_schedulers);

  void add_request(
      RequestId request_id, ClusterType cluster_type);
  [[nodiscard]] std::vector<GlobalRequestAssignment>
  schedule();
  [[nodiscard]] bool empty() const noexcept {
    return request_queue_.empty();
  }
  [[nodiscard]] BaseClusterScheduler& get_cluster_scheduler(
      ClusterType cluster_type);
  [[nodiscard]] const BaseClusterScheduler&
  get_cluster_scheduler(ClusterType cluster_type) const;

 private:
  std::map<
      ClusterType,
      std::unique_ptr<BaseClusterScheduler>>
      cluster_schedulers_;
  std::vector<GlobalRequestAssignment> request_queue_;
};

}  // namespace frontier::scheduler
