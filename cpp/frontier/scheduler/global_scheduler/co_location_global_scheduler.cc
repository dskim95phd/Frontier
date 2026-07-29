#include "frontier/scheduler/global_scheduler/co_location_global_scheduler.h"

#include <utility>

namespace frontier::scheduler {

CoLocationGlobalScheduler::CoLocationGlobalScheduler(
    std::unique_ptr<BaseClusterScheduler> cluster_scheduler)
    : cluster_scheduler_(std::move(cluster_scheduler)) {
  if (cluster_scheduler_ == nullptr) {
    throw GlobalSchedulerError(
        "co-location global scheduler requires one cluster");
  }
  if (cluster_scheduler_->cluster_type() !=
      ClusterType::kMonolithic) {
    throw GlobalSchedulerError(
        "co-location global scheduler requires a monolithic cluster");
  }
}

void CoLocationGlobalScheduler::add_request(
    RequestId request_id,
    ClusterType cluster_type) {
  static_cast<void>(get_cluster_scheduler(cluster_type));
  request_queue_.push_back(GlobalRequestAssignment{
      .request_id = request_id,
      .cluster_type = cluster_type,
  });
}

std::vector<GlobalRequestAssignment>
CoLocationGlobalScheduler::schedule() {
  std::vector<GlobalRequestAssignment> result =
      std::move(request_queue_);
  request_queue_.clear();
  return result;
}

BaseClusterScheduler&
CoLocationGlobalScheduler::get_cluster_scheduler(
    ClusterType cluster_type) {
  if (cluster_type != cluster_scheduler_->cluster_type()) {
    throw GlobalSchedulerError(
        "co-location run references an unknown cluster");
  }
  return *cluster_scheduler_;
}

const BaseClusterScheduler&
CoLocationGlobalScheduler::get_cluster_scheduler(
    ClusterType cluster_type) const {
  if (cluster_type != cluster_scheduler_->cluster_type()) {
    throw GlobalSchedulerError(
        "co-location run references an unknown cluster");
  }
  return *cluster_scheduler_;
}

}  // namespace frontier::scheduler
