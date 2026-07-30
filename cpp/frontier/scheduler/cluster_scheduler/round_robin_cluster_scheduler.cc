#include "frontier/scheduler/cluster_scheduler/round_robin_cluster_scheduler.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace frontier::scheduler {

RoundRobinClusterScheduler::RoundRobinClusterScheduler(
    std::unique_ptr<BaseReplicaScheduler> replica_scheduler)
    : RoundRobinClusterScheduler(
          [&replica_scheduler] {
            std::vector<std::unique_ptr<BaseReplicaScheduler>> result;
            result.push_back(std::move(replica_scheduler));
            return result;
          }(),
          1,
          1) {}

RoundRobinClusterScheduler::RoundRobinClusterScheduler(
    std::vector<std::unique_ptr<BaseReplicaScheduler>>
        replica_schedulers,
    std::uint64_t num_replicas,
    std::uint64_t data_parallel_size,
    ClusterType cluster_type)
    : replica_schedulers_(std::move(replica_schedulers)),
      num_replicas_(num_replicas),
      data_parallel_size_(data_parallel_size),
      cluster_type_(cluster_type) {
  if (num_replicas_ == 0 || data_parallel_size_ == 0 ||
      num_replicas_ >
          std::numeric_limits<std::uint64_t>::max() /
              data_parallel_size_ ||
      replica_schedulers_.size() !=
          num_replicas_ * data_parallel_size_) {
    throw ClusterSchedulerError(
        "cluster target matrix is invalid");
  }
  for (std::uint64_t replica = 0; replica < num_replicas_;
       ++replica) {
    for (std::uint64_t dp = 0; dp < data_parallel_size_; ++dp) {
      BaseReplicaScheduler* scheduler =
          replica_schedulers_.at(target_index(
              ReplicaId{replica}, DataParallelId{dp})).get();
      if (scheduler == nullptr ||
          scheduler->replica_id() != ReplicaId{replica} ||
          scheduler->dp_id() != DataParallelId{dp}) {
        throw ClusterSchedulerError(
            "cluster scheduler target identity mismatch");
      }
    }
  }
}

void RoundRobinClusterScheduler::add_request(
    RequestId request_id,
    SimTime arrived_at) {
  request_queue_.push_back(QueuedRequest{
      .request_id = request_id,
      .arrived_at = arrived_at,
  });
}

std::vector<ClusterRequestAssignment>
RoundRobinClusterScheduler::schedule() {
  // Python's BaseClusterScheduler sorts the pending queue by the request's
  // original arrival time before every routing pass. std::stable_sort also
  // preserves event insertion order when arrival times are equal.
  std::stable_sort(
      request_queue_.begin(),
      request_queue_.end(),
      [](const QueuedRequest& lhs, const QueuedRequest& rhs) {
        return lhs.arrived_at < rhs.arrived_at;
      });

  std::vector<ClusterRequestAssignment> result;
  result.reserve(request_queue_.size());

  // Production Python uses a flattened lane round robin for unified PDD
  // DECODE.  KV arrivals commonly contain one request per scheduling cycle,
  // so restarting the batch-mode DP split would otherwise pin all online
  // decode traffic to dp_id=0.
  if (cluster_type_ == ClusterType::kDecode) {
    const std::uint64_t total_lanes =
        num_replicas_ * data_parallel_size_;
    for (std::size_t index = 0;
         index < request_queue_.size();
         ++index) {
      const std::uint64_t lane =
          (request_counter_ +
           static_cast<std::uint64_t>(index)) %
          total_lanes;
      const ReplicaId replica_id{lane % num_replicas_};
      const DataParallelId dp_id{lane / num_replicas_};
      BaseReplicaScheduler& target =
          get_replica_scheduler(replica_id, dp_id);
      const RequestId request_id =
          request_queue_.at(index).request_id;
      target.add_request(request_id);
      result.push_back(ClusterRequestAssignment{
          .replica_id = replica_id,
          .dp_id = dp_id,
          .request_id = request_id,
      });
    }
    request_counter_ +=
        static_cast<std::uint64_t>(request_queue_.size());
    request_queue_.clear();
    return result;
  }

  // Match production Python RoundRobinClusterScheduler._schedule_batch_mode:
  // first distribute this scheduling cycle across replicas, then split each
  // replica's slice as evenly as possible across that replica's DP lanes.
  // This is deliberately not a flattened replica/DP round robin.
  std::vector<std::vector<RequestId>> requests_by_replica(
      static_cast<std::size_t>(num_replicas_));
  for (std::size_t index = 0; index < request_queue_.size(); ++index) {
    const std::uint64_t replica =
        (request_counter_ + static_cast<std::uint64_t>(index)) %
        num_replicas_;
    requests_by_replica.at(static_cast<std::size_t>(replica))
        .push_back(request_queue_.at(index).request_id);
  }
  request_counter_ +=
      static_cast<std::uint64_t>(request_queue_.size());

  for (std::uint64_t replica = 0; replica < num_replicas_;
       ++replica) {
    const auto& requests =
        requests_by_replica.at(static_cast<std::size_t>(replica));
    const std::uint64_t base =
        static_cast<std::uint64_t>(requests.size()) /
        data_parallel_size_;
    const std::uint64_t extra =
        static_cast<std::uint64_t>(requests.size()) %
        data_parallel_size_;
    std::size_t request_index = 0;
    for (std::uint64_t dp = 0; dp < data_parallel_size_; ++dp) {
      const std::uint64_t lane_count =
          base + (dp < extra ? 1U : 0U);
      BaseReplicaScheduler& target =
          get_replica_scheduler(
              ReplicaId{replica}, DataParallelId{dp});
      for (std::uint64_t lane_index = 0;
           lane_index < lane_count;
           ++lane_index) {
        const RequestId request_id =
            requests.at(request_index++);
        target.add_request(request_id);
        result.push_back(ClusterRequestAssignment{
            .replica_id = target.replica_id(),
            .dp_id = target.dp_id(),
            .request_id = request_id,
        });
      }
    }
  }
  request_queue_.clear();
  return result;
}

std::vector<std::pair<ReplicaId, DataParallelId>>
RoundRobinClusterScheduler::targets() const {
  std::vector<std::pair<ReplicaId, DataParallelId>> result;
  result.reserve(replica_schedulers_.size());
  for (const auto& scheduler : replica_schedulers_) {
    result.emplace_back(
        scheduler->replica_id(), scheduler->dp_id());
  }
  return result;
}

std::size_t RoundRobinClusterScheduler::target_index(
    ReplicaId replica_id,
    DataParallelId dp_id) const {
  if (!replica_id.valid() || !dp_id.valid() ||
      replica_id.index() >= num_replicas_ ||
      dp_id.index() >= data_parallel_size_) {
    throw ClusterSchedulerError(
        "cluster references an unknown replica target");
  }
  return static_cast<std::size_t>(
      replica_id.index() * data_parallel_size_ +
      dp_id.index());
}

BaseReplicaScheduler&
RoundRobinClusterScheduler::get_replica_scheduler(
    ReplicaId replica_id,
    DataParallelId dp_id) {
  return *replica_schedulers_.at(
      target_index(replica_id, dp_id));
}

const BaseReplicaScheduler&
RoundRobinClusterScheduler::get_replica_scheduler(
    ReplicaId replica_id,
    DataParallelId dp_id) const {
  return *replica_schedulers_.at(
      target_index(replica_id, dp_id));
}

}  // namespace frontier::scheduler
