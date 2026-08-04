#pragma once

#include <map>
#include <memory>
#include <stdexcept>
#include <vector>

#include "frontier/config/config.h"
#include "frontier/core/ids.h"
#include "frontier/entities/cluster.h"
#include "frontier/entities/request.h"
#include "frontier/execution_time_predictor/base_execution_time_predictor.h"
#include "frontier/kv_cache_transfer/base_kv_cache_transfer_predictor.h"
#include "frontier/scheduler/cluster_scheduler/base_cluster_scheduler.h"
#include "frontier/scheduler/scheduler_types.h"

namespace frontier::scheduler {

class GlobalSchedulerError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

struct GlobalRequestAssignment {
    RequestId request_id;
    ClusterType cluster_type;
};

class GlobalScheduler final {
  public:
    using PredictorMap =
        std::map<ClusterType,
                 execution_time_predictor::ExecutionTimePredictorPtr>;

    GlobalScheduler(
        const std::map<ClusterType, entities::Cluster> &clusters,
        std::vector<entities::Request> &requests,
        const PredictorMap &predictors,
        std::shared_ptr<const kv_cache_transfer::BaseKVCacheTransferPredictor>
            kv_cache_transfer_predictor,
        const config::ClusterSchedulerConfig &scheduler_config,
        config::PrefixCacheConfig prefix_cache_config = {},
        config::ResolvedCpuKVCacheTargetConfig cpu_kv_cache_config = {});

    void add_request(RequestId request_id, ClusterType cluster_type);
    [[nodiscard]] std::vector<GlobalRequestAssignment> schedule();
    [[nodiscard]] bool empty() const noexcept { return request_queue_.empty(); }
    [[nodiscard]] BaseClusterScheduler &
    get_cluster_scheduler(ClusterType cluster_type);
    [[nodiscard]] const BaseClusterScheduler &
    get_cluster_scheduler(ClusterType cluster_type) const;

  private:
    const std::map<ClusterType, entities::Cluster> *clusters_;
    std::shared_ptr<const kv_cache_transfer::BaseKVCacheTransferPredictor>
        kv_cache_transfer_predictor_;
    std::map<ClusterType, std::unique_ptr<BaseClusterScheduler>>
        cluster_schedulers_;
    std::vector<GlobalRequestAssignment> request_queue_;
};

} // namespace frontier::scheduler
