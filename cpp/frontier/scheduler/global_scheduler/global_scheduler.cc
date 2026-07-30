#include "frontier/scheduler/global_scheduler/global_scheduler.h"

#include <utility>

#include "frontier/scheduler/cluster_scheduler/round_robin_cluster_scheduler.h"

namespace frontier::scheduler {

GlobalScheduler::GlobalScheduler(
    const std::map<ClusterType, entities::Cluster> &clusters,
    std::vector<entities::Request> &requests, const PredictorMap &predictors,
    std::shared_ptr<const kv_cache_transfer::BaseKVCacheTransferPredictor>
        kv_cache_transfer_predictor,
    const config::ClusterSchedulerConfig &scheduler_config)
    : clusters_(&clusters),
      kv_cache_transfer_predictor_(std::move(kv_cache_transfer_predictor)) {
    if (clusters.empty()) {
        throw GlobalSchedulerError(
            "global scheduler requires at least one cluster");
    }
    for (const auto &[cluster_type, cluster] : clusters) {
        const auto predictor = predictors.find(cluster_type);
        if (predictor == predictors.end() || predictor->second == nullptr) {
            throw GlobalSchedulerError(
                "global scheduler is missing a cluster predictor");
        }
        std::unique_ptr<BaseClusterScheduler> cluster_scheduler;
        switch (scheduler_config.type) {
        case config::ClusterSchedulerType::kRoundRobin:
            cluster_scheduler = std::make_unique<RoundRobinClusterScheduler>(
                cluster, requests, predictor->second,
                kv_cache_transfer_predictor_);
            break;
        }
        if (cluster_scheduler == nullptr ||
            !cluster_schedulers_
                 .emplace(cluster_type, std::move(cluster_scheduler))
                 .second) {
            throw GlobalSchedulerError(
                "global scheduler received a duplicate cluster type");
        }
    }
}

void GlobalScheduler::add_request(RequestId request_id,
                                  ClusterType cluster_type) {
    static_cast<void>(get_cluster_scheduler(cluster_type));
    request_queue_.push_back([&]() {
        GlobalRequestAssignment value{};
        value.request_id = request_id;
        value.cluster_type = cluster_type;
        return value;
    }());
}

std::vector<GlobalRequestAssignment> GlobalScheduler::schedule() {
    std::vector<GlobalRequestAssignment> result = std::move(request_queue_);
    request_queue_.clear();
    return result;
}

BaseClusterScheduler &
GlobalScheduler::get_cluster_scheduler(ClusterType cluster_type) {
    const auto position = cluster_schedulers_.find(cluster_type);
    if (position == cluster_schedulers_.end()) {
        throw GlobalSchedulerError(
            "global scheduler references an unknown cluster");
    }
    return *position->second;
}

const BaseClusterScheduler &
GlobalScheduler::get_cluster_scheduler(ClusterType cluster_type) const {
    const auto position = cluster_schedulers_.find(cluster_type);
    if (position == cluster_schedulers_.end()) {
        throw GlobalSchedulerError(
            "global scheduler references an unknown cluster");
    }
    return *position->second;
}

} // namespace frontier::scheduler
