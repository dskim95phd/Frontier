#include "frontier/scheduler/cluster_scheduler/base_cluster_scheduler.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <utility>

#include "frontier/entities/batch.h"
#include "frontier/entities/cluster.h"
#include "frontier/entities/request.h"
#include "frontier/execution_time_predictor/base_execution_time_predictor.h"
#include "frontier/scheduler/replica_scheduler/replica_scheduler_factory.h"
#include "frontier/simulator/simulator.h"

namespace frontier::scheduler {

BaseClusterScheduler::BaseClusterScheduler(
    const entities::Cluster &cluster, entities::RequestCollection &requests,
    execution_time_predictor::ExecutionTimePredictorPtr predictor,
    std::shared_ptr<const kv_cache_transfer::BaseKVCacheTransferPredictor>
        kv_cache_transfer_predictor,
    config::PrefixCacheConfig prefix_cache_config,
    config::ResolvedCpuKVCacheTargetConfig cpu_kv_cache_config)
    : cluster_(&cluster), requests_(&requests),
      kv_cache_transfer_predictor_(std::move(kv_cache_transfer_predictor)) {
    const std::uint64_t num_replicas = cluster.parallelism().num_replicas;
    const std::uint64_t data_parallel_size =
        cluster.parallelism().data_parallel_size;
    if (predictor == nullptr) {
        throw ClusterSchedulerError(
            "cluster scheduler requires an execution predictor");
    }
    if (num_replicas == 0 || data_parallel_size == 0 ||
        num_replicas >
            std::numeric_limits<std::uint64_t>::max() / data_parallel_size) {
        throw ClusterSchedulerError("cluster target matrix is invalid");
    }
    replica_schedulers_.reserve(
        static_cast<std::size_t>(num_replicas * data_parallel_size));
    const config::ClusterRuntimeConfig &runtime = cluster.runtime_config();
    for (std::uint64_t replica = 0; replica < num_replicas; ++replica) {
        for (std::uint64_t dp = 0; dp < data_parallel_size; ++dp) {
            replica_schedulers_.push_back(make_replica_scheduler(
                runtime.scheduler, requests, predictor,
                cluster.replica(ReplicaId{replica}), DataParallelId{dp},
                cluster.type(), prefix_cache_config,
                cluster.type() == ClusterType::kPrefill
                    ? cpu_kv_cache_config
                    : config::ResolvedCpuKVCacheTargetConfig{}));
        }
    }
    for (std::uint64_t replica = 0; replica < num_replicas; ++replica) {
        for (std::uint64_t dp = 0; dp < data_parallel_size; ++dp) {
            BaseReplicaScheduler *scheduler =
                replica_schedulers_
                    .at(target_index(ReplicaId{replica}, DataParallelId{dp}))
                    .get();
            if (scheduler == nullptr ||
                scheduler->replica_id() != ReplicaId{replica} ||
                scheduler->dp_id() != DataParallelId{dp} ||
                &scheduler->replica_entity() !=
                    &cluster.replica(ReplicaId{replica})) {
                throw ClusterSchedulerError(
                    "cluster scheduler target identity mismatch");
            }
        }
    }
}

SessionId BaseClusterScheduler::request_session_id(RequestId request_id) const {
    if (!requests_->contains(request_id)) {
        throw ClusterSchedulerError(
            "cluster scheduler references an unknown request");
    }
    return requests_->at(request_id.index()).session_id();
}

const entities::Request &
BaseClusterScheduler::request(RequestId request_id) const {
    if (!requests_->contains(request_id)) {
        throw ClusterSchedulerError(
            "cluster scheduler references an unknown request");
    }
    const entities::Request &value = requests_->at(request_id.index());
    if (value.id() != request_id) {
        throw ClusterSchedulerError(
            "cluster scheduler request arena ID/index invariant failed");
    }
    return value;
}

kv_cache_transfer::TransferPrediction
BaseClusterScheduler::predict_kv_cache_transfer(
    std::uint64_t num_tokens) const {
    if (kv_cache_transfer_predictor_ == nullptr) {
        throw ClusterSchedulerError(
            "cluster scheduler has no KV-cache transfer predictor");
    }
    return kv_cache_transfer_predictor_->predict(num_tokens, cluster_->model());
}

void BaseClusterScheduler::add_request(RequestId request_id,
                                       SimTime arrived_at) {
    request_queue_.push_back([&]() {
        QueuedRequest value{};
        value.request_id = request_id;
        value.arrived_at = arrived_at;
        return value;
    }());
}

std::vector<std::pair<ReplicaId, DataParallelId>>
BaseClusterScheduler::targets() const {
    std::vector<std::pair<ReplicaId, DataParallelId>> result;
    result.reserve(replica_schedulers_.size());
    for (const auto &scheduler : replica_schedulers_) {
        result.emplace_back(scheduler->replica_id(), scheduler->dp_id());
    }
    return result;
}

std::size_t BaseClusterScheduler::target_index(ReplicaId replica_id,
                                               DataParallelId dp_id) const {
    if (!replica_id.valid() || !dp_id.valid() ||
        replica_id.index() >= num_replicas() ||
        dp_id.index() >= data_parallel_size()) {
        throw ClusterSchedulerError(
            "cluster references an unknown replica target");
    }
    return static_cast<std::size_t>(replica_id.index() * data_parallel_size() +
                                    dp_id.index());
}

BaseReplicaScheduler &
BaseClusterScheduler::get_replica_scheduler(ReplicaId replica_id,
                                            DataParallelId dp_id) {
    return *replica_schedulers_.at(target_index(replica_id, dp_id));
}

const BaseReplicaScheduler &
BaseClusterScheduler::get_replica_scheduler(ReplicaId replica_id,
                                            DataParallelId dp_id) const {
    return *replica_schedulers_.at(target_index(replica_id, dp_id));
}

BatchGlobalId BaseClusterScheduler::next_batch_global_id(ReplicaId replica_id,
                                                         DataParallelId dp_id) {
    const BatchCounterKey key = [&]() {
        BatchCounterKey value{};
        value.replica_id = replica_id;
        value.dp_id = dp_id;
        return value;
    }();
    return batch_counters_[key].next("batch global ID space exhausted");
}

void BaseClusterScheduler::require_quiescent() const {
    moe_barrier_.require_empty();
    if (!moe_stage_states_.empty() || !moe_group_states_.empty() ||
        !moe_domain_reservations_.empty()) {
        throw std::runtime_error(
            "simulation quiesced with live cluster MoE state");
    }
}

} // namespace frontier::scheduler
