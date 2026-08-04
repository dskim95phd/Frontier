#include "frontier/scheduler/cluster_scheduler/round_robin_cluster_scheduler.h"

#include <algorithm>
#include <utility>

namespace frontier::scheduler {

RoundRobinClusterScheduler::RoundRobinClusterScheduler(
    const entities::Cluster &cluster, std::vector<entities::Request> &requests,
    execution_time_predictor::ExecutionTimePredictorPtr predictor,
    std::shared_ptr<const kv_cache_transfer::BaseKVCacheTransferPredictor>
        kv_cache_transfer_predictor,
    config::PrefixCacheConfig prefix_cache_config,
    config::ResolvedCpuKVCacheTargetConfig cpu_kv_cache_config)
    : BaseClusterScheduler(cluster, requests, std::move(predictor),
                           std::move(kv_cache_transfer_predictor),
                           prefix_cache_config, cpu_kv_cache_config) {}

std::vector<ClusterRequestAssignment> RoundRobinClusterScheduler::schedule() {
    // Python's BaseClusterScheduler sorts the pending queue by the request's
    // original arrival time before every routing pass. std::stable_sort also
    // preserves event insertion order when arrival times are equal.
    std::stable_sort(request_queue_.begin(), request_queue_.end(),
                     [](const QueuedRequest &lhs, const QueuedRequest &rhs) {
                         return lhs.arrived_at < rhs.arrived_at;
                     });

    std::vector<ClusterRequestAssignment> result;
    result.reserve(request_queue_.size());

    // Production Python uses a flattened lane round robin for unified PDD
    // DECODE.  KV arrivals commonly contain one request per scheduling cycle,
    // so restarting the batch-mode DP split would otherwise pin all online
    // decode traffic to dp_id=0.
    if (cluster_type() == ClusterType::kDecode) {
        const std::uint64_t total_lanes = num_replicas() * data_parallel_size();
        for (std::size_t index = 0; index < request_queue_.size(); ++index) {
            const std::uint64_t lane =
                (request_counter_ + static_cast<std::uint64_t>(index)) %
                total_lanes;
            const ReplicaId replica_id{lane % num_replicas()};
            const DataParallelId dp_id{lane / num_replicas()};
            BaseReplicaScheduler &target =
                get_replica_scheduler(replica_id, dp_id);
            const RequestId request_id = request_queue_.at(index).request_id;
            target.add_request(request_id);
            result.push_back([&]() {
                ClusterRequestAssignment value{};
                value.replica_id = replica_id;
                value.dp_id = dp_id;
                value.request_id = request_id;
                return value;
            }());
        }
        request_counter_ += static_cast<std::uint64_t>(request_queue_.size());
        request_queue_.clear();
        return result;
    }

    // Match production Python RoundRobinClusterScheduler._schedule_batch_mode:
    // first distribute this scheduling cycle across replicas, then split each
    // replica's slice as evenly as possible across that replica's DP lanes.
    // This is deliberately not a flattened replica/DP round robin.
    std::vector<std::vector<RequestId>> requests_by_replica(
        static_cast<std::size_t>(num_replicas()));
    for (std::size_t index = 0; index < request_queue_.size(); ++index) {
        const std::uint64_t replica =
            (request_counter_ + static_cast<std::uint64_t>(index)) %
            num_replicas();
        requests_by_replica.at(static_cast<std::size_t>(replica))
            .push_back(request_queue_.at(index).request_id);
    }
    request_counter_ += static_cast<std::uint64_t>(request_queue_.size());

    for (std::uint64_t replica = 0; replica < num_replicas(); ++replica) {
        const auto &requests =
            requests_by_replica.at(static_cast<std::size_t>(replica));
        const std::uint64_t base =
            static_cast<std::uint64_t>(requests.size()) / data_parallel_size();
        const std::uint64_t extra =
            static_cast<std::uint64_t>(requests.size()) % data_parallel_size();
        std::size_t request_index = 0;
        for (std::uint64_t dp = 0; dp < data_parallel_size(); ++dp) {
            const std::uint64_t lane_count = base + (dp < extra ? 1U : 0U);
            BaseReplicaScheduler &target =
                get_replica_scheduler(ReplicaId{replica}, DataParallelId{dp});
            for (std::uint64_t lane_index = 0; lane_index < lane_count;
                 ++lane_index) {
                const RequestId request_id = requests.at(request_index++);
                target.add_request(request_id);
                result.push_back([&]() {
                    ClusterRequestAssignment value{};
                    value.replica_id = target.replica_id();
                    value.dp_id = target.dp_id();
                    value.request_id = request_id;
                    return value;
                }());
            }
        }
    }
    request_queue_.clear();
    return result;
}

} // namespace frontier::scheduler
