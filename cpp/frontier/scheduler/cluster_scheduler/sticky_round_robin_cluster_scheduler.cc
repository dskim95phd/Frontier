#include "frontier/scheduler/cluster_scheduler/sticky_round_robin_cluster_scheduler.h"

#include <algorithm>
#include <utility>

namespace frontier::scheduler {

StickyRoundRobinClusterScheduler::StickyRoundRobinClusterScheduler(
    const entities::Cluster &cluster, std::vector<entities::Request> &requests,
    execution_time_predictor::ExecutionTimePredictorPtr predictor,
    std::shared_ptr<const kv_cache_transfer::BaseKVCacheTransferPredictor>
        kv_cache_transfer_predictor,
    config::PrefixCacheConfig prefix_cache_config,
    config::ResolvedCpuKVCacheTargetConfig cpu_kv_cache_config)
    : BaseClusterScheduler(cluster, requests, std::move(predictor),
                           std::move(kv_cache_transfer_predictor),
                           prefix_cache_config, cpu_kv_cache_config) {}

std::vector<ClusterRequestAssignment>
StickyRoundRobinClusterScheduler::schedule() {
    std::stable_sort(request_queue_.begin(), request_queue_.end(),
                     [](const QueuedRequest &lhs, const QueuedRequest &rhs) {
                         return lhs.arrived_at < rhs.arrived_at;
                     });
    const std::uint64_t target_count = num_replicas() * data_parallel_size();
    std::vector<ClusterRequestAssignment> result;
    result.reserve(request_queue_.size());
    for (const QueuedRequest &queued : request_queue_) {
        const SessionId session_id = request_session_id(queued.request_id);
        if (!session_id.valid()) {
            throw ClusterSchedulerError(
                "sticky round robin requires a valid session ID");
        }
        auto [position, inserted] =
            session_targets_.try_emplace(session_id, ReplicaTarget{});
        if (inserted) {
            const std::uint64_t lane = next_session_target_++ % target_count;
            position->second.replica_id =
                ReplicaId{lane / data_parallel_size()};
            position->second.dp_id =
                DataParallelId{lane % data_parallel_size()};
        }
        BaseReplicaScheduler &target = get_replica_scheduler(
            position->second.replica_id, position->second.dp_id);
        target.add_request(queued.request_id);
        result.push_back(ClusterRequestAssignment{position->second.replica_id,
                                                  position->second.dp_id,
                                                  queued.request_id});
    }
    request_queue_.clear();
    return result;
}

} // namespace frontier::scheduler
