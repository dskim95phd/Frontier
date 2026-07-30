#include "frontier/scheduler/global_scheduler/global_scheduler.h"

#include <utility>

namespace frontier::scheduler {

GlobalScheduler::GlobalScheduler(
    std::unique_ptr<BaseClusterScheduler> cluster_scheduler)
    : GlobalScheduler(
          [&cluster_scheduler] {
            std::vector<std::unique_ptr<BaseClusterScheduler>> result;
            result.push_back(std::move(cluster_scheduler));
            return result;
          }()) {}

GlobalScheduler::GlobalScheduler(
    std::vector<std::unique_ptr<BaseClusterScheduler>>
        cluster_schedulers) {
  if (cluster_schedulers.empty()) {
    throw GlobalSchedulerError(
        "global scheduler requires at least one cluster");
  }
  for (auto& cluster : cluster_schedulers) {
    if (cluster == nullptr) {
      throw GlobalSchedulerError(
          "global scheduler received a null cluster");
    }
    const ClusterType type = cluster->cluster_type();
    if (!cluster_schedulers_.emplace(type, std::move(cluster)).second) {
      throw GlobalSchedulerError(
          "global scheduler received a duplicate cluster type");
    }
  }
}

void GlobalScheduler::add_request(
    RequestId request_id,
    ClusterType cluster_type) {
  static_cast<void>(get_cluster_scheduler(cluster_type));
  request_queue_.push_back(GlobalRequestAssignment{
      .request_id = request_id,
      .cluster_type = cluster_type,
  });
}

std::vector<GlobalRequestAssignment>
GlobalScheduler::schedule() {
  std::vector<GlobalRequestAssignment> result =
      std::move(request_queue_);
  request_queue_.clear();
  return result;
}

BaseClusterScheduler&
GlobalScheduler::get_cluster_scheduler(
    ClusterType cluster_type) {
  const auto position = cluster_schedulers_.find(cluster_type);
  if (position == cluster_schedulers_.end()) {
    throw GlobalSchedulerError(
        "global scheduler references an unknown cluster");
  }
  return *position->second;
}

const BaseClusterScheduler&
GlobalScheduler::get_cluster_scheduler(
    ClusterType cluster_type) const {
  const auto position = cluster_schedulers_.find(cluster_type);
  if (position == cluster_schedulers_.end()) {
    throw GlobalSchedulerError(
        "global scheduler references an unknown cluster");
  }
  return *position->second;
}

}  // namespace frontier::scheduler
