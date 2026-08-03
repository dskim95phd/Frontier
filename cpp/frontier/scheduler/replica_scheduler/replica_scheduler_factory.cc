#include "frontier/scheduler/replica_scheduler/replica_scheduler_factory.h"

#include <memory>

#include "frontier/entities/replica.h"
#include "frontier/entities/request.h"
#include "frontier/scheduler/replica_scheduler/base_replica_scheduler.h"
#include "frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.h"

namespace frontier::scheduler {

std::unique_ptr<BaseReplicaScheduler> make_replica_scheduler(
    const config::SchedulerConfig &config,
    std::vector<entities::Request> &requests,
    execution_time_predictor::ExecutionTimePredictorPtr predictor,
    const entities::Replica &replica, DataParallelId dp_id,
    ClusterType cluster_type, config::PrefixCacheConfig prefix_cache_config) {
    switch (config.type) {
    case config::SchedulerType::kVllmV1:
        return std::make_unique<VllmV1Scheduler>(
            config, requests, std::move(predictor), replica, dp_id,
            cluster_type, prefix_cache_config);
    }
    throw SchedulerError("unsupported replica scheduler type");
}

} // namespace frontier::scheduler
