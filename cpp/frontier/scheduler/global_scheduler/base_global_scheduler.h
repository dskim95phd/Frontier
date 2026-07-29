#pragma once

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

class BaseGlobalScheduler {
 public:
  virtual ~BaseGlobalScheduler() = default;

  BaseGlobalScheduler(const BaseGlobalScheduler&) = delete;
  BaseGlobalScheduler& operator=(const BaseGlobalScheduler&) = delete;
  BaseGlobalScheduler(BaseGlobalScheduler&&) = delete;
  BaseGlobalScheduler& operator=(BaseGlobalScheduler&&) = delete;

  virtual void add_request(
      RequestId request_id,
      ClusterType cluster_type) = 0;
  [[nodiscard]] virtual std::vector<GlobalRequestAssignment>
  schedule() = 0;
  [[nodiscard]] virtual bool empty() const noexcept = 0;
  [[nodiscard]] virtual BaseClusterScheduler&
  get_cluster_scheduler(ClusterType cluster_type) = 0;
  [[nodiscard]] virtual const BaseClusterScheduler&
  get_cluster_scheduler(ClusterType cluster_type) const = 0;

 protected:
  BaseGlobalScheduler() = default;
};

}  // namespace frontier::scheduler
