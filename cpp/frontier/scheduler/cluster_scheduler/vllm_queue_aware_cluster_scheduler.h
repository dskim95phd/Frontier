#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "frontier/scheduler/cluster_scheduler/base_cluster_scheduler.h"

namespace frontier::scheduler {

// vLLM-like cluster routing: choose the target with the smallest observable
// outstanding queue (running + waiting requests), with stable target-ID
// tie-breaking.  No declared/future output length is consulted.
class VllmQueueAwareClusterScheduler final : public BaseClusterScheduler {
  public:
    VllmQueueAwareClusterScheduler(
        const entities::Cluster &cluster,
        std::vector<entities::Request> &requests,
        execution_time_predictor::ExecutionTimePredictorPtr predictor,
        std::shared_ptr<const kv_cache_transfer::BaseKVCacheTransferPredictor>
            kv_cache_transfer_predictor,
        config::PrefixCacheConfig prefix_cache_config = {},
        config::ResolvedCpuKVCacheTargetConfig cpu_kv_cache_config = {});

    [[nodiscard]] std::vector<ClusterRequestAssignment> schedule() override;

  private:
    std::uint64_t next_tie_target_ = 0;
};

} // namespace frontier::scheduler
