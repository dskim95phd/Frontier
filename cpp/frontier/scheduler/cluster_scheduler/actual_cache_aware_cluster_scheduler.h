#pragma once

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "frontier/scheduler/cluster_scheduler/base_cluster_scheduler.h"

namespace frontier::scheduler {

// Actual tiered-cache-aware routing for append-only multi-turn sessions.
//
// Balanced targets keep affinity only when the mapped target still owns at
// least cache_threshold of the request prefix in GPU or CPU KV. Imbalanced
// targets, or a weak cache match, use least-outstanding routing. Migrating a
// session invalidates both tiers on its old target immediately; physical GPU
// and CPU blocks remain only while active requests or transfers drain.
class CacheAwareClusterScheduler final : public BaseClusterScheduler {
  public:
    CacheAwareClusterScheduler(
        const entities::Cluster &cluster,
        entities::RequestCollection &requests,
        execution_time_predictor::ExecutionTimePredictorPtr predictor,
        std::shared_ptr<const kv_cache_transfer::BaseKVCacheTransferPredictor>
            kv_cache_transfer_predictor,
        config::PrefixCacheConfig prefix_cache_config = {},
        config::ResolvedCpuKVCacheTargetConfig cpu_kv_cache_config = {},
        config::ClusterSchedulerConfig routing_config = {});

    [[nodiscard]] std::vector<ClusterRequestAssignment> schedule() override;

  private:
    using Target = std::pair<ReplicaId, DataParallelId>;

    [[nodiscard]] Target pick_least_loaded_target(
        const std::vector<Target> &ordered_targets) const;
    [[nodiscard]] bool targets_are_imbalanced(
        const std::vector<Target> &ordered_targets) const;

    config::ClusterSchedulerConfig routing_config_;
    std::uint64_t next_tie_target_ = 0;
    std::unordered_map<SessionId, Target, StrongIdHash<SessionId>>
        session_to_target_;
};

} // namespace frontier::scheduler
