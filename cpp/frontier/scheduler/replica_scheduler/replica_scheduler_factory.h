#pragma once

#include <memory>
#include <vector>

#include "frontier/config/config.h"
#include "frontier/core/ids.h"
#include "frontier/execution_time_predictor/base_execution_time_predictor.h"

namespace frontier::entities {
class Replica;
class Request;
} // namespace frontier::entities

namespace frontier::scheduler {

class BaseReplicaScheduler;

[[nodiscard]] std::unique_ptr<BaseReplicaScheduler> make_replica_scheduler(
    const config::SchedulerConfig &config,
    std::vector<entities::Request> &requests,
    execution_time_predictor::ExecutionTimePredictorPtr predictor,
    const entities::Replica &replica, DataParallelId dp_id,
    ClusterType cluster_type);

} // namespace frontier::scheduler
