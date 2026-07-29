#pragma once

#include <memory>
#include <vector>

#include "frontier/scheduler/global_scheduler/base_global_scheduler.h"

namespace frontier::scheduler {

class CoLocationGlobalScheduler final
    : public BaseGlobalScheduler {
 public:
  explicit CoLocationGlobalScheduler(
      std::unique_ptr<BaseClusterScheduler> cluster_scheduler);

  void add_request(
      RequestId request_id,
      ClusterType cluster_type) override;
  [[nodiscard]] std::vector<GlobalRequestAssignment>
  schedule() override;
  [[nodiscard]] bool empty() const noexcept override {
    return request_queue_.empty();
  }
  [[nodiscard]] BaseClusterScheduler& get_cluster_scheduler(
      ClusterType cluster_type) override;
  [[nodiscard]] const BaseClusterScheduler&
  get_cluster_scheduler(ClusterType cluster_type) const override;

 private:
  std::unique_ptr<BaseClusterScheduler> cluster_scheduler_;
  std::vector<GlobalRequestAssignment> request_queue_;
};

}  // namespace frontier::scheduler
