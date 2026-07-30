#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "frontier/scheduler/cluster_scheduler/base_cluster_scheduler.h"

namespace frontier::scheduler {

class RoundRobinClusterScheduler final : public BaseClusterScheduler {
  public:
    RoundRobinClusterScheduler(
        const entities::Cluster &cluster,
        std::vector<entities::Request> &requests,
        execution_time_predictor::ExecutionTimePredictorPtr predictor,
        std::shared_ptr<const kv_cache_transfer::BaseKVCacheTransferPredictor>
            kv_cache_transfer_predictor);

    [[nodiscard]] std::vector<ClusterRequestAssignment> schedule() override;

  private:
    std::uint64_t request_counter_ = 0;
};

} // namespace frontier::scheduler
