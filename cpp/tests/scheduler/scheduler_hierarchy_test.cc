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
using frontier::execution_time_predictor::FixedBatchExecutionModel;
using frontier::request_generator::WorkloadRequest;
using frontier::scheduler::BaseClusterScheduler;
using frontier::scheduler::BaseReplicaScheduler;
using frontier::scheduler::ClusterSchedulerError;
using frontier::scheduler::ClusterType;
using frontier::scheduler::GlobalScheduler;
using frontier::scheduler::GlobalSchedulerError;
using frontier::scheduler::ReplicaStageScheduler;
using frontier::scheduler::SchedulerError;
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

void test_colocation_scheduler_hierarchy_routes_and_executes() {
    std::vector<Request> requests;
    requests.emplace_back([&]() {
        WorkloadRequest value{};
        value.request_id = RequestId{0};
        value.arrived_at = SimTime::from_seconds(0.0);
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
                       std::make_shared<FixedBatchExecutionModel>([&]() {
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
        std::make_shared<FixedBatchExecutionModel>([&]() {
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
        value.arrived_at = SimTime::from_seconds(0.0);
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
                       std::make_shared<FixedBatchExecutionModel>([&]() {
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
    return failures == 0 ? 0 : 1;
}
