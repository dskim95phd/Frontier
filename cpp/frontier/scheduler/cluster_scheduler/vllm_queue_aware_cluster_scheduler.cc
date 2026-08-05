#include "frontier/scheduler/cluster_scheduler/vllm_queue_aware_cluster_scheduler.h"

#include <algorithm>
#include <limits>
#include <tuple>
#include <utility>

namespace frontier::scheduler {
namespace {

std::uint64_t saturated_add(std::uint64_t lhs, std::uint64_t rhs) noexcept {
    if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return lhs + rhs;
}

} // namespace

VllmQueueAwareClusterScheduler::VllmQueueAwareClusterScheduler(
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
VllmQueueAwareClusterScheduler::schedule() {
    std::stable_sort(request_queue_.begin(), request_queue_.end(),
                     [](const QueuedRequest &lhs, const QueuedRequest &rhs) {
                         return lhs.arrived_at < rhs.arrived_at;
                     });

    std::vector<ClusterRequestAssignment> result;
    result.reserve(request_queue_.size());
    const auto ordered_targets = targets();
    const std::uint64_t target_count =
        static_cast<std::uint64_t>(ordered_targets.size());
    for (const QueuedRequest &queued : request_queue_) {
        ReplicaId selected_replica;
        DataParallelId selected_dp;
        std::tuple<std::uint64_t, std::uint64_t, std::int64_t, std::int64_t> best{
            std::numeric_limits<std::uint64_t>::max(),
            std::numeric_limits<std::uint64_t>::max(),
            std::numeric_limits<std::int64_t>::max(),
            std::numeric_limits<std::int64_t>::max()};
        for (std::size_t target_index = 0;
             target_index < ordered_targets.size(); ++target_index) {
            const auto [replica_id, dp_id] = ordered_targets.at(target_index);
            const BaseReplicaScheduler &target =
                get_replica_scheduler(replica_id, dp_id);
            const std::uint64_t outstanding = saturated_add(
                static_cast<std::uint64_t>(target.running_count()),
                static_cast<std::uint64_t>(target.waiting_count()));
            const std::uint64_t tie_rank =
                (static_cast<std::uint64_t>(target_index) + target_count -
                 (next_tie_target_ % target_count)) %
                target_count;
            const auto candidate = std::tuple{
                outstanding, tie_rank, replica_id.value(), dp_id.value()};
            if (candidate < best) {
                best = candidate;
                selected_replica = replica_id;
                selected_dp = dp_id;
            }
        }
        BaseReplicaScheduler &target =
            get_replica_scheduler(selected_replica, selected_dp);
        target.add_request(queued.request_id);
        next_tie_target_ = (next_tie_target_ + 1) % target_count;
        result.push_back(ClusterRequestAssignment{
            selected_replica, selected_dp, queued.request_id});
    }
    request_queue_.clear();
    return result;
}

} // namespace frontier::scheduler
