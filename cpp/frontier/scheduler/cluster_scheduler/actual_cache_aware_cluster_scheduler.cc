#include "frontier/scheduler/cluster_scheduler/actual_cache_aware_cluster_scheduler.h"

#include <algorithm>
#include <limits>
#include <tuple>
#include <utility>

namespace frontier::scheduler {

CacheAwareClusterScheduler::CacheAwareClusterScheduler(
    const entities::Cluster &cluster, std::vector<entities::Request> &requests,
    execution_time_predictor::ExecutionTimePredictorPtr predictor,
    std::shared_ptr<const kv_cache_transfer::BaseKVCacheTransferPredictor>
        kv_cache_transfer_predictor,
    config::PrefixCacheConfig prefix_cache_config,
    config::ResolvedCpuKVCacheTargetConfig cpu_kv_cache_config,
    config::ClusterSchedulerConfig routing_config)
    : BaseClusterScheduler(cluster, requests, std::move(predictor),
                           std::move(kv_cache_transfer_predictor),
                           prefix_cache_config, cpu_kv_cache_config),
      routing_config_(std::move(routing_config)) {
    if (!prefix_cache_config.enabled) {
        throw ClusterSchedulerError(
            "cache_aware routing requires prefix_cache.enabled=true");
    }
}

CacheAwareClusterScheduler::Target
CacheAwareClusterScheduler::pick_least_loaded_target(
    const std::vector<Target> &ordered_targets) const {
    Target selected = ordered_targets.front();
    std::tuple<std::uint64_t, std::uint64_t, std::int64_t, std::int64_t> best{
        std::numeric_limits<std::uint64_t>::max(),
        std::numeric_limits<std::uint64_t>::max(),
        std::numeric_limits<std::int64_t>::max(),
        std::numeric_limits<std::int64_t>::max()};
    const std::uint64_t target_count =
        static_cast<std::uint64_t>(ordered_targets.size());
    for (std::size_t index = 0; index < ordered_targets.size(); ++index) {
        const auto [replica_id, dp_id] = ordered_targets.at(index);
        const BaseReplicaScheduler &target =
            get_replica_scheduler(replica_id, dp_id);
        const std::uint64_t load =
            static_cast<std::uint64_t>(target.running_count()) +
            static_cast<std::uint64_t>(target.waiting_count());
        const std::uint64_t tie_rank =
            (static_cast<std::uint64_t>(index) + target_count -
             (next_tie_target_ % target_count)) %
            target_count;
        const auto candidate =
            std::tuple{load, tie_rank, replica_id.value(), dp_id.value()};
        if (candidate < best) {
            best = candidate;
            selected = ordered_targets.at(index);
        }
    }
    return selected;
}

bool CacheAwareClusterScheduler::targets_are_imbalanced(
    const std::vector<Target> &ordered_targets) const {
    std::uint64_t minimum = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t maximum = 0;
    for (const auto &[replica_id, dp_id] : ordered_targets) {
        const BaseReplicaScheduler &target =
            get_replica_scheduler(replica_id, dp_id);
        const std::uint64_t load =
            static_cast<std::uint64_t>(target.running_count()) +
            static_cast<std::uint64_t>(target.waiting_count());
        minimum = std::min(minimum, load);
        maximum = std::max(maximum, load);
    }
    return maximum - minimum > routing_config_.balance_abs_threshold &&
           static_cast<double>(maximum) >
               routing_config_.balance_rel_threshold *
                   static_cast<double>(minimum);
}

std::vector<ClusterRequestAssignment>
CacheAwareClusterScheduler::schedule() {
    std::stable_sort(request_queue_.begin(), request_queue_.end(),
                     [](const QueuedRequest &lhs, const QueuedRequest &rhs) {
                         return lhs.arrived_at < rhs.arrived_at;
                     });

    std::vector<ClusterRequestAssignment> result;
    result.reserve(request_queue_.size());
    const std::vector<Target> ordered_targets = targets();
    const std::uint64_t target_count =
        static_cast<std::uint64_t>(ordered_targets.size());

    for (const QueuedRequest &queued : request_queue_) {
        const entities::Request &incoming = request(queued.request_id);
        const SessionId session_id = incoming.session_id();
        if (!session_id.valid()) {
            throw ClusterSchedulerError(
                "cache_aware routing requires a valid session_id");
        }

        const Target least_loaded = pick_least_loaded_target(ordered_targets);
        Target selected = least_loaded;
        const auto mapped = session_to_target_.find(session_id);
        if (mapped != session_to_target_.end() &&
            !targets_are_imbalanced(ordered_targets)) {
            const BaseReplicaScheduler &cache_target =
                get_replica_scheduler(mapped->second.first,
                                      mapped->second.second);
            const kv_cache::PrefixLookupResult lookup =
                cache_target.gpu_prefix_cache_lookup(incoming);
            const double hit_ratio =
                lookup.query_blocks == 0
                    ? 0.0
                    : static_cast<double>(lookup.hit_blocks) /
                          static_cast<double>(lookup.query_blocks);
            if (hit_ratio >= routing_config_.cache_threshold) {
                selected = mapped->second;
            }
        }

        if (mapped != session_to_target_.end() && mapped->second != selected) {
            BaseReplicaScheduler &old_target = get_replica_scheduler(
                mapped->second.first, mapped->second.second);
            static_cast<void>(
                old_target.discard_gpu_prefix_cache_session(session_id));
        }
        session_to_target_[session_id] = selected;
        BaseReplicaScheduler &target =
            get_replica_scheduler(selected.first, selected.second);
        target.add_request(queued.request_id);
        next_tie_target_ = (next_tie_target_ + 1) % target_count;
        result.push_back(ClusterRequestAssignment{
            selected.first, selected.second, queued.request_id});
    }
    request_queue_.clear();
    return result;
}

} // namespace frontier::scheduler
