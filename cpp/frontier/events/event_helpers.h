#pragma once

#include <vector>

#include "frontier/core/cluster_type.h"
#include "frontier/core/ids.h"
#include "frontier/entities/batch.h"
#include "frontier/entities/request.h"
#include "frontier/metrics/output_contract.h"
#include "frontier/scheduler/replica_scheduler/base_replica_scheduler.h"
#include "frontier/scheduler/scheduler_types.h"

namespace frontier::events {

[[nodiscard]] scheduler::ReplicaTarget make_target(
    ReplicaId replica_id,
    DataParallelId dp_id) noexcept;

[[nodiscard]] metrics::SchedulerTraceRecord make_scheduler_trace(
    const scheduler::ScheduleResult& schedule,
    scheduler::ReplicaTarget target,
    ClusterType cluster_type);

[[nodiscard]] metrics::BatchMetricsRecord make_batch_metrics(
    const entities::Batch& batch,
    const std::vector<entities::Request>& requests,
    double predicted_execution_ms);

}  // namespace frontier::events
