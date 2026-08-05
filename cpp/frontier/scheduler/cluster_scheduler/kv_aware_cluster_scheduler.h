#pragma once

#include <memory>
#include <vector>

#include "frontier/scheduler/cluster_scheduler/base_cluster_scheduler.h"

namespace frontier::scheduler {

// KV/sequence-aware cluster routing.  Target load is the currently allocated
// KV blocks plus the KV blocks represented by queued requests' current
// scheduler frontiers.  It intentionally never uses a request's declared
// future decode-token count.
class KvAwareClusterScheduler final : public BaseClusterScheduler {
  public:
    KvAwareClusterScheduler(
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
