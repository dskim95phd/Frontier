#pragma once

#include <memory>
#include <map>
#include <vector>

#include "frontier/scheduler/global_scheduler/base_global_scheduler.h"

namespace frontier::scheduler {

class GlobalScheduler final
    : public BaseGlobalScheduler {
 public:
  explicit GlobalScheduler(
      std::unique_ptr<BaseClusterScheduler> cluster_scheduler);
  explicit GlobalScheduler(
      std::vector<std::unique_ptr<BaseClusterScheduler>>
          cluster_schedulers);

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
  std::map<
      ClusterType,
      std::unique_ptr<BaseClusterScheduler>>
      cluster_schedulers_;
  std::vector<GlobalRequestAssignment> request_queue_;
};

}  // namespace frontier::scheduler
