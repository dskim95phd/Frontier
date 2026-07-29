#pragma once

#include <memory>
#include <vector>

#include "frontier/scheduler/cluster_scheduler/base_cluster_scheduler.h"

namespace frontier::scheduler {

class CoLocationClusterScheduler final
    : public BaseClusterScheduler {
 public:
  explicit CoLocationClusterScheduler(
      std::unique_ptr<BaseReplicaScheduler> replica_scheduler);
  CoLocationClusterScheduler(
      std::vector<std::unique_ptr<BaseReplicaScheduler>>
          replica_schedulers,
      std::uint64_t num_replicas,
      std::uint64_t data_parallel_size);

  [[nodiscard]] ClusterType cluster_type()
      const noexcept override {
    return ClusterType::kMonolithic;
  }
  void add_request(RequestId request_id) override;
  [[nodiscard]] std::vector<ClusterRequestAssignment>
  schedule() override;
  [[nodiscard]] bool empty() const noexcept override {
    return request_queue_.empty();
  }
  [[nodiscard]] std::vector<
      std::pair<ReplicaId, DataParallelId>>
  targets() const override;
  [[nodiscard]] BaseReplicaScheduler& get_replica_scheduler(
      ReplicaId replica_id,
      DataParallelId dp_id) override;
  [[nodiscard]] const BaseReplicaScheduler&
  get_replica_scheduler(
      ReplicaId replica_id,
      DataParallelId dp_id) const override;

 private:
  [[nodiscard]] std::size_t target_index(
      ReplicaId replica_id,
      DataParallelId dp_id) const;

  std::vector<std::unique_ptr<BaseReplicaScheduler>>
      replica_schedulers_;
  std::vector<RequestId> request_queue_;
  std::uint64_t num_replicas_ = 1;
  std::uint64_t data_parallel_size_ = 1;
  std::uint64_t request_counter_ = 0;
};

}  // namespace frontier::scheduler
