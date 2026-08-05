#include "frontier/execution_time_predictor/fixed_execution_time_predictor.h"
#include "frontier/scheduler/cluster_scheduler/actual_cache_aware_cluster_scheduler.h"
#include "frontier/scheduler/cluster_scheduler/kv_aware_cluster_scheduler.h"
#include "frontier/scheduler/cluster_scheduler/sticky_round_robin_cluster_scheduler.h"
#include "frontier/scheduler/cluster_scheduler/vllm_queue_aware_cluster_scheduler.h"
#include "frontier/scheduler/global_scheduler/global_scheduler.h"
#include "tests/test_support.h"

#include <map>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace {

using frontier::BatchGlobalId;
using frontier::BatchId;
using frontier::DataParallelId;
using frontier::Generation;
using frontier::IterationId;
using frontier::ReplicaId;
using frontier::RequestId;
using frontier::SimTime;
using frontier::StageId;
using frontier::config::SchedulerConfig;
using frontier::entities::Batch;
using frontier::entities::Cluster;
using frontier::entities::Request;
using frontier::entities::RequestBatchSnapshot;
using frontier::execution_time_predictor::FixedExecutionTimePredictor;
using frontier::request_generator::WorkloadRequest;
using frontier::scheduler::BaseClusterScheduler;
using frontier::scheduler::BaseReplicaScheduler;
using frontier::scheduler::CacheAwareClusterScheduler;
using frontier::scheduler::ClusterSchedulerError;
using frontier::scheduler::ClusterType;
using frontier::scheduler::GlobalScheduler;
using frontier::scheduler::GlobalSchedulerError;
using frontier::scheduler::ReplicaStageScheduler;
using frontier::scheduler::SchedulerError;
using frontier::config::ClusterSchedulerConfig;
using frontier::config::ClusterSchedulerType;
using frontier::scheduler::KvAwareClusterScheduler;
using frontier::scheduler::StickyRoundRobinClusterScheduler;
using frontier::scheduler::VllmQueueAwareClusterScheduler;
using frontier::test::expect;
using frontier::test::expect_throws;

SchedulerConfig scheduler_config() {
    SchedulerConfig config;
    config.batch_size_cap = 4;
    config.max_tokens_in_batch = 16;
    config.block_size = 4;
    config.num_blocks = 16;
    return config;
}

Request make_request(RequestId id, std::uint64_t prefill_tokens = 2,
                     std::uint64_t decode_tokens = 1,
                     frontier::SessionId session_id = frontier::SessionId{}) {
    WorkloadRequest value{};
    value.request_id = id;
    value.session_start_at = SimTime::from_seconds(0.0);
    value.num_prefill_tokens = prefill_tokens;
    value.num_decode_tokens = decode_tokens;
    value.session_id = session_id;
    value.session_turn_index = std::nullopt;
    return Request{value};
}

void complete_single_monolithic_iteration(BaseReplicaScheduler &scheduler,
                                          Request &request, BatchId batch_id) {
    const auto decision = scheduler.schedule(SimTime::from_seconds(0.0));
    expect(decision.scheduled_requests.size() == 1,
           "cache-aware fixture must schedule exactly one request");
    const auto &scheduled = decision.scheduled_requests.front();
    Batch batch{batch_id,
                decision.iteration_id,
                {RequestBatchSnapshot{
                    scheduled.request_id,
                    scheduled.num_tokens,
                    request.runtime_epoch(),
                    request.execution_epoch(),
                    request.num_processed_tokens(),
                    request.scheduler_num_computed_tokens(),
                }},
                decision.simulation_time,
                Generation{1}};
    batch.set_global_id(BatchGlobalId{batch_id.value()});
    scheduler.mark_batch_started(batch);
    expect(scheduler.on_batch_completed(batch, SimTime::from_seconds(0.001)),
           "cache-aware fixture batch must complete");
}

void test_colocation_scheduler_hierarchy_routes_and_executes() {
    std::vector<Request> requests;
    requests.emplace_back([&]() {
        WorkloadRequest value{};
        value.request_id = RequestId{0};
        value.session_start_at = SimTime::from_seconds(0.0);
        value.num_prefill_tokens = 2;
        value.num_decode_tokens = 1;
        value.session_id = frontier::SessionId{};
        value.session_turn_index = std::nullopt;
        return value;
    }());

    frontier::config::ClusterRuntimeConfig runtime;
    runtime.scheduler = scheduler_config();
    std::map<ClusterType, Cluster> clusters;
    clusters.emplace(ClusterType::kMonolithic,
                     Cluster{ClusterType::kMonolithic, runtime});
    GlobalScheduler::PredictorMap predictors;
    predictors.emplace(ClusterType::kMonolithic,
                       std::make_shared<FixedExecutionTimePredictor>([&]() {
                           frontier::config::FixedExecutionModelConfig value{};
                           value.batch_latency_ms = 2.5;
                           value.stage_latencies_ms = {};
                           return value;
                       }()));
    GlobalScheduler global{clusters, requests, predictors, nullptr,
                           frontier::config::ClusterSchedulerConfig{}};

    requests[0].on_arrival(SimTime::from_seconds(0.0));
    global.add_request(RequestId{0}, ClusterType::kMonolithic);

    BaseClusterScheduler &monolithic =
        global.get_cluster_scheduler(ClusterType::kMonolithic);
    for (const auto &assignment : global.schedule()) {
        monolithic.add_request(assignment.request_id, requests[0].arrived_at());
    }
    static_cast<void>(monolithic.schedule());
    BaseReplicaScheduler &replica_scheduler =
        monolithic.get_replica_scheduler(ReplicaId{0}, DataParallelId{0});
    const auto schedule =
        replica_scheduler.schedule(SimTime::from_seconds(0.0));
    expect(schedule.scheduled_requests.size() == 1 &&
               schedule.scheduled_requests[0].request_id == RequestId{0},
           "global and cluster schedulers must route to the replica scheduler");

    const auto &scheduled = schedule.scheduled_requests[0];
    Batch batch{BatchId{0},
                schedule.iteration_id,
                {
                    [&]() {
                        RequestBatchSnapshot value{};
                        value.request_id = scheduled.request_id;
                        value.scheduled_tokens = scheduled.num_tokens;
                        value.runtime_epoch = requests[0].runtime_epoch();
                        value.execution_epoch = requests[0].execution_epoch();
                        value.processed_tokens =
                            requests[0].num_processed_tokens();
                        value.scheduler_frontier =
                            requests[0].scheduler_num_computed_tokens();
                        return value;
                    }(),
                },
                schedule.simulation_time,
                Generation{1}};
    batch.set_global_id(BatchGlobalId{0});
    replica_scheduler.mark_batch_started(batch);

    ReplicaStageScheduler &stage =
        replica_scheduler.get_replica_stage_scheduler(StageId{0});
    expect(stage.replica_id() == ReplicaId{0} &&
               stage.dp_id() == DataParallelId{0} &&
               stage.stage_id() == StageId{0} && stage.is_last_stage(),
           "replica stage identity must match the co-location target");
    stage.add_batch(batch);
    const auto ticket = stage.pop_batch_if_not_busy();
    expect(ticket.has_value() && ticket->batch_id == batch.id() &&
               ticket->schedule_epoch == batch.schedule_epoch(),
           "stage scheduler must dispatch the replica batch");
    expect(!stage.pop_batch_if_not_busy().has_value(),
           "busy stage must not dispatch another batch");

    const auto prediction = stage.predict(batch, requests);
    expect(prediction.duration_ms == 2.5,
           "stage scheduler must own and invoke the execution model");
    stage.on_stage_end(batch.id());
    expect(replica_scheduler.on_batch_completed(batch,
                                                SimTime::from_seconds(0.0025)),
           "replica scheduler must apply the completed stage batch");
    expect(requests[0].completed() && replica_scheduler.idle() &&
               !stage.is_busy() && stage.empty(),
           "the complete hierarchy must quiesce after request completion");
}

void test_stage_scheduler_prioritizes_global_batch_id() {
    ReplicaStageScheduler stage{
        ReplicaId{0}, DataParallelId{0}, StageId{0}, true,
        std::make_shared<FixedExecutionTimePredictor>([&]() {
            frontier::config::FixedExecutionModelConfig value{};
            value.batch_latency_ms = 1.0;
            value.stage_latencies_ms = {};
            return value;
        }())};
    const auto snapshots = std::vector<RequestBatchSnapshot>{
        [&]() {
            RequestBatchSnapshot value{};
            value.request_id = RequestId{0};
            value.scheduled_tokens = 1;
            value.runtime_epoch = 0;
            value.execution_epoch = 0;
            value.processed_tokens = 0;
            value.scheduler_frontier = 0;
            return value;
        }(),
    };
    Batch inserted_first{BatchId{0}, IterationId{0}, snapshots,
                         SimTime::from_seconds(0.0), Generation{1}};
    inserted_first.set_global_id(BatchGlobalId{10});
    Batch inserted_second{BatchId{1}, IterationId{1}, snapshots,
                          SimTime::from_seconds(0.0), Generation{1}};
    inserted_second.set_global_id(BatchGlobalId{2});

    stage.add_batch(inserted_first);
    stage.add_batch(inserted_second);
    const auto first = stage.pop_batch_if_not_busy();
    expect(first.has_value() && first->batch_id == inserted_second.id(),
           "stage scheduler must prioritize the lower global batch ID");
    stage.on_stage_end(inserted_second.id());
    const auto second = stage.pop_batch_if_not_busy();
    expect(second.has_value() && second->batch_id == inserted_first.id(),
           "stage scheduler must retain the remaining batch");
}

void test_colocation_hierarchy_rejects_unknown_targets() {
    std::vector<Request> requests;
    requests.emplace_back([&]() {
        WorkloadRequest value{};
        value.request_id = RequestId{0};
        value.session_start_at = SimTime::from_seconds(0.0);
        value.num_prefill_tokens = 2;
        value.num_decode_tokens = 1;
        value.session_id = frontier::SessionId{};
        value.session_turn_index = std::nullopt;
        return value;
    }());
    frontier::config::ClusterRuntimeConfig runtime;
    runtime.scheduler = scheduler_config();
    std::map<ClusterType, Cluster> clusters;
    clusters.emplace(ClusterType::kMonolithic,
                     Cluster{ClusterType::kMonolithic, runtime});
    GlobalScheduler::PredictorMap predictors;
    predictors.emplace(ClusterType::kMonolithic,
                       std::make_shared<FixedExecutionTimePredictor>([&]() {
                           frontier::config::FixedExecutionModelConfig value{};
                           value.batch_latency_ms = 0.0;
                           value.stage_latencies_ms = {};
                           return value;
                       }()));
    GlobalScheduler global{clusters, requests, predictors, nullptr,
                           frontier::config::ClusterSchedulerConfig{}};

    expect_throws<GlobalSchedulerError>(
        [&global] {
            static_cast<void>(
                global.get_cluster_scheduler(static_cast<ClusterType>(99)));
        },
        "global scheduler must reject unknown clusters");
    BaseClusterScheduler &monolithic =
        global.get_cluster_scheduler(ClusterType::kMonolithic);
    expect_throws<ClusterSchedulerError>(
        [&monolithic] {
            static_cast<void>(monolithic.get_replica_scheduler(
                ReplicaId{1}, DataParallelId{0}));
        },
        "cluster scheduler must reject unknown replicas");
    BaseReplicaScheduler &replica_scheduler =
        monolithic.get_replica_scheduler(ReplicaId{0}, DataParallelId{0});
    expect_throws<SchedulerError>(
        [&replica_scheduler] {
            static_cast<void>(
                replica_scheduler.get_replica_stage_scheduler(StageId{1}));
        },
        "replica scheduler must reject unknown pipeline stages");
}

void test_stage_specific_factory_and_observable_routing_policies() {
    frontier::config::ClusterRuntimeConfig runtime;
    runtime.parallelism.num_replicas = 2;
    runtime.scheduler = scheduler_config();

    std::vector<Request> queue_requests;
    for (std::uint64_t index = 0; index < 4; ++index) {
        queue_requests.emplace_back(make_request(RequestId{index}));
        queue_requests.back().on_arrival(SimTime::from_seconds(0.0));
    }
    Cluster queue_cluster{ClusterType::kMonolithic, runtime};
    const auto predictor = std::make_shared<FixedExecutionTimePredictor>([&]() {
        frontier::config::FixedExecutionModelConfig value{};
        value.batch_latency_ms = 0.0;
        value.stage_latencies_ms = {};
        return value;
    }());
    VllmQueueAwareClusterScheduler queue_scheduler{
        queue_cluster, queue_requests, predictor, nullptr};
    for (std::uint64_t index = 0; index < queue_requests.size(); ++index) {
        queue_scheduler.add_request(RequestId{index},
                                    SimTime::from_seconds(0.0));
    }
    const auto queue_assignments = queue_scheduler.schedule();
    expect(queue_assignments.size() == 4 &&
               queue_assignments[0].replica_id == ReplicaId{0} &&
               queue_assignments[1].replica_id == ReplicaId{1} &&
               queue_assignments[2].replica_id == ReplicaId{0} &&
               queue_assignments[3].replica_id == ReplicaId{1},
           "vLLM queue-aware routing must balance observable queue counts");

    std::vector<Request> kv_requests;
    kv_requests.emplace_back(make_request(RequestId{0}, 8, 100));
    kv_requests.emplace_back(make_request(RequestId{1}, 8, 1));
    kv_requests[0].on_arrival(SimTime::from_seconds(0.0));
    kv_requests[1].on_arrival(SimTime::from_seconds(0.0));
    KvAwareClusterScheduler kv_scheduler{queue_cluster, kv_requests, predictor,
                                         nullptr};
    kv_scheduler.add_request(RequestId{0}, SimTime::from_seconds(0.0));
    const auto first_kv = kv_scheduler.schedule();
    expect(first_kv.size() == 1 && first_kv[0].replica_id == ReplicaId{0},
           "KV-aware routing must use stable target ordering for an empty set");
    // This is observable current context progress, not the request's future
    // decode length.  The next request should avoid the queued two-block
    // frontier on replica zero.
    kv_requests[0].advance_scheduler_frontier(8);
    kv_scheduler.add_request(RequestId{1}, SimTime::from_seconds(0.0));
    const auto second_kv = kv_scheduler.schedule();
    expect(second_kv.size() == 1 && second_kv[0].replica_id == ReplicaId{1},
           "KV-aware routing must avoid the larger queued KV footprint");

    std::map<ClusterType, Cluster> pdd_clusters;
    pdd_clusters.emplace(ClusterType::kPrefill,
                         Cluster{ClusterType::kPrefill, runtime});
    pdd_clusters.emplace(ClusterType::kDecode,
                         Cluster{ClusterType::kDecode, runtime});
    std::vector<Request> pdd_requests;
    GlobalScheduler::PredictorMap predictors;
    predictors.emplace(ClusterType::kPrefill, predictor);
    predictors.emplace(ClusterType::kDecode, predictor);
    ClusterSchedulerConfig stage_config{};
    stage_config.type = ClusterSchedulerType::kStickyRoundRobin;
    stage_config.prefill_type = ClusterSchedulerType::kStickyRoundRobin;
    stage_config.decode_type = ClusterSchedulerType::kVllmQueueAware;
    GlobalScheduler global{pdd_clusters, pdd_requests, predictors, nullptr,
                           stage_config};
    expect(dynamic_cast<StickyRoundRobinClusterScheduler *>(
               &global.get_cluster_scheduler(ClusterType::kPrefill)) != nullptr &&
               dynamic_cast<VllmQueueAwareClusterScheduler *>(
                   &global.get_cluster_scheduler(ClusterType::kDecode)) !=
                   nullptr,
           "PDD stages must instantiate independent configured cluster policies");
}

void test_actual_cache_aware_routing_migrates_and_discards_old_gpu_kv() {
    frontier::config::ClusterRuntimeConfig runtime;
    runtime.parallelism.num_replicas = 2;
    runtime.scheduler = scheduler_config();
    Cluster cluster{ClusterType::kMonolithic, runtime};
    const auto predictor = std::make_shared<FixedExecutionTimePredictor>([&]() {
        frontier::config::FixedExecutionModelConfig value{};
        value.batch_latency_ms = 0.0;
        value.stage_latencies_ms = {};
        return value;
    }());
    std::vector<Request> requests;
    requests.emplace_back(make_request(RequestId{0}, 8, 1,
                                       frontier::SessionId{7}));
    requests.emplace_back(make_request(RequestId{1}, 2, 1,
                                       frontier::SessionId{99}));
    requests.emplace_back(make_request(RequestId{2}, 12, 1,
                                       frontier::SessionId{7}));
    for (Request &request : requests) {
        request.on_arrival(SimTime::from_seconds(0.0));
    }
    ClusterSchedulerConfig routing{};
    routing.type = ClusterSchedulerType::kCacheAware;
    routing.cache_threshold = 0.5;
    routing.balance_abs_threshold = 0;
    routing.balance_rel_threshold = 1.0;
    CacheAwareClusterScheduler scheduler{
        cluster, requests, predictor, nullptr,
        frontier::config::PrefixCacheConfig{
            true, frontier::config::PrefixCachingKeyMode::kSession},
        {}, routing};

    scheduler.add_request(RequestId{0}, SimTime::from_seconds(0.0));
    const auto producer_assignment = scheduler.schedule();
    expect(producer_assignment.size() == 1 &&
               producer_assignment.front().replica_id == ReplicaId{0},
           "first session turn must use stable least-load placement");
    BaseReplicaScheduler &old_target =
        scheduler.get_replica_scheduler(ReplicaId{0}, DataParallelId{0});
    complete_single_monolithic_iteration(old_target, requests[0], BatchId{0});
    expect(old_target.gpu_cache_valid_prefix_blocks(
               frontier::SessionId{7}) == 2,
           "producer must leave two actual GPU prefix blocks");

    // Create one observable outstanding request on the affinity target. With
    // zero/1.0 test thresholds this forces the imbalance branch.
    old_target.add_request(RequestId{1});
    scheduler.add_request(RequestId{2}, SimTime::from_seconds(0.0));
    const auto migrated = scheduler.schedule();
    expect(migrated.size() == 1 &&
               migrated.front().replica_id == ReplicaId{1},
           "load imbalance must override a nonzero cache hit");
    expect(old_target.gpu_cache_valid_prefix_blocks(
               frontier::SessionId{7}) == 0,
           "migration must discard every old-target GPU KV block");

    std::vector<Request> low_hit_requests;
    low_hit_requests.emplace_back(make_request(
        RequestId{0}, 8, 1, frontier::SessionId{17}));
    low_hit_requests.emplace_back(make_request(
        RequestId{1}, 12, 1, frontier::SessionId{17}));
    for (Request &request : low_hit_requests) {
        request.on_arrival(SimTime::from_seconds(0.0));
    }
    ClusterSchedulerConfig low_hit_routing{};
    low_hit_routing.type = ClusterSchedulerType::kCacheAware;
    low_hit_routing.cache_threshold = 0.8;
    CacheAwareClusterScheduler low_hit_scheduler{
        cluster, low_hit_requests, predictor, nullptr,
        frontier::config::PrefixCacheConfig{
            true, frontier::config::PrefixCachingKeyMode::kSession},
        {}, low_hit_routing};
    low_hit_scheduler.add_request(RequestId{0}, SimTime::from_seconds(0.0));
    const auto low_hit_producer = low_hit_scheduler.schedule();
    BaseReplicaScheduler &low_hit_old = low_hit_scheduler.get_replica_scheduler(
        low_hit_producer.front().replica_id,
        low_hit_producer.front().dp_id);
    complete_single_monolithic_iteration(low_hit_old, low_hit_requests[0],
                                         BatchId{0});
    low_hit_scheduler.add_request(RequestId{1},
                                  SimTime::from_seconds(0.0));
    const auto low_hit_migration = low_hit_scheduler.schedule();
    expect(low_hit_migration.size() == 1 &&
               low_hit_migration.front().replica_id == ReplicaId{1},
           "GPU hit ratio below threshold must use least-load routing");
    expect(low_hit_old.gpu_cache_valid_prefix_blocks(
               frontier::SessionId{17}) == 0,
           "low-hit migration must discard the partial old GPU prefix");
}

} // namespace

int main() {
    int failures = 0;
    failures += frontier::test::run(
        "co-location scheduler hierarchy routes and executes",
        test_colocation_scheduler_hierarchy_routes_and_executes);
    failures +=
        frontier::test::run("stage scheduler prioritizes global batch ID",
                            test_stage_scheduler_prioritizes_global_batch_id);
    failures +=
        frontier::test::run("co-location hierarchy rejects unknown targets",
                            test_colocation_hierarchy_rejects_unknown_targets);
    failures += frontier::test::run(
        "stage-specific factory and observable routing policies",
        test_stage_specific_factory_and_observable_routing_policies);
    failures += frontier::test::run(
        "actual cache-aware routing migrates and discards old GPU KV",
        test_actual_cache_aware_routing_migrates_and_discards_old_gpu_kv);
    return failures == 0 ? 0 : 1;
}
