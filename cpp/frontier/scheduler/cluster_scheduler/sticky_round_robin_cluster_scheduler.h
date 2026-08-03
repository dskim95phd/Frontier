#pragma once

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include "frontier/scheduler/cluster_scheduler/base_cluster_scheduler.h"

namespace frontier::scheduler {

class StickyRoundRobinClusterScheduler final : public BaseClusterScheduler {
  public:
    StickyRoundRobinClusterScheduler(
        const entities::Cluster &cluster,
        std::vector<entities::Request> &requests,
        execution_time_predictor::ExecutionTimePredictorPtr predictor,
        std::shared_ptr<const kv_cache_transfer::BaseKVCacheTransferPredictor>
            kv_cache_transfer_predictor,
        config::PrefixCacheConfig prefix_cache_config = {});

    [[nodiscard]] std::vector<ClusterRequestAssignment> schedule() override;

  private:
    std::unordered_map<SessionId, ReplicaTarget, StrongIdHash<SessionId>>
        session_targets_;
    std::uint64_t next_session_target_ = 0;
};

} // namespace frontier::scheduler
