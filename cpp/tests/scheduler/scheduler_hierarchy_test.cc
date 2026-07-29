#include "frontier/scheduler/cluster_scheduler/co_location_cluster_scheduler.h"
#include "frontier/scheduler/global_scheduler/co_location_global_scheduler.h"
#include "frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.h"
#include "tests/test_support.h"

#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace {

using frontier::BatchId;
using frontier::DataParallelId;
using frontier::Generation;
using frontier::ReplicaId;
using frontier::RequestId;
using frontier::SimTime;
using frontier::StageId;
using frontier::config::SchedulerConfig;
using frontier::entities::Batch;
using frontier::entities::Request;
using frontier::entities::RequestBatchSnapshot;
using frontier::execution_time_predictor::FixedBatchExecutionModel;
using frontier::request_generator::WorkloadRequest;
using frontier::scheduler::BaseClusterScheduler;
using frontier::scheduler::BaseReplicaScheduler;
using frontier::scheduler::ClusterSchedulerError;
using frontier::scheduler::ClusterType;
using frontier::scheduler::CoLocationClusterScheduler;
using frontier::scheduler::CoLocationGlobalScheduler;
using frontier::scheduler::GlobalSchedulerError;
using frontier::scheduler::ReplicaStageScheduler;
using frontier::scheduler::SchedulerError;
using frontier::scheduler::VllmV1Scheduler;
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
  requests.emplace_back(WorkloadRequest{
      .request_id = RequestId{0},
      .arrived_at = SimTime::from_seconds(0.0),
      .num_prefill_tokens = 2,
      .num_decode_tokens = 1,
      .session_id = std::nullopt,
      .session_turn_index = std::nullopt,
  });

  auto replica = std::make_unique<VllmV1Scheduler>(
      scheduler_config(),
      requests,
      std::make_unique<FixedBatchExecutionModel>(
          frontier::config::FixedExecutionModelConfig{
              .batch_latency_ms = 2.5,
              .stage_latencies_ms = {},
          }));
  auto cluster = std::make_unique<CoLocationClusterScheduler>(
      std::move(replica));
  CoLocationGlobalScheduler global{std::move(cluster)};

  requests[0].on_arrival(SimTime::from_seconds(0.0));
  global.add_request(RequestId{0}, ClusterType::kMonolithic);

  BaseClusterScheduler& monolithic =
      global.get_cluster_scheduler(ClusterType::kMonolithic);
  for (const auto& assignment : global.schedule()) {
    monolithic.add_request(assignment.request_id);
  }
  static_cast<void>(monolithic.schedule());
  BaseReplicaScheduler& replica_scheduler =
      monolithic.get_replica_scheduler(
          ReplicaId{0}, DataParallelId{0});
  const auto schedule =
      replica_scheduler.schedule(SimTime::from_seconds(0.0));
  expect(
      schedule.scheduled_requests.size() == 1 &&
          schedule.scheduled_requests[0].request_id == RequestId{0},
      "global and cluster schedulers must route to the replica scheduler");

  const auto& scheduled = schedule.scheduled_requests[0];
  Batch batch{
      BatchId{0},
      schedule.iteration_id,
      {
          RequestBatchSnapshot{
              .request_id = scheduled.request_id,
              .scheduled_tokens = scheduled.num_tokens,
              .runtime_epoch = requests[0].runtime_epoch(),
              .execution_epoch = requests[0].execution_epoch(),
              .processed_tokens = requests[0].num_processed_tokens(),
              .scheduler_frontier =
                  requests[0].scheduler_num_computed_tokens(),
          },
      },
      schedule.simulation_time,
      Generation{1}};
  replica_scheduler.mark_batch_started(batch);

  ReplicaStageScheduler& stage =
      replica_scheduler.get_replica_stage_scheduler(StageId{0});
  expect(
      stage.replica_id() == ReplicaId{0} &&
          stage.dp_id() == DataParallelId{0} &&
          stage.stage_id() == StageId{0} &&
          stage.is_last_stage(),
      "replica stage identity must match the co-location target");
  stage.add_batch(batch);
  const auto ticket = stage.pop_batch_if_not_busy();
  expect(
      ticket.has_value() && ticket->batch_id == batch.id() &&
          ticket->schedule_epoch == batch.schedule_epoch(),
      "stage scheduler must dispatch the replica batch");
  expect(
      !stage.pop_batch_if_not_busy().has_value(),
      "busy stage must not dispatch another batch");

  const auto prediction = stage.predict(batch, requests);
  expect(
      prediction.duration_ms == 2.5,
      "stage scheduler must own and invoke the execution model");
  stage.on_stage_end(batch.id());
  expect(
      replica_scheduler.on_batch_completed(
          batch, SimTime::from_seconds(0.0025)),
      "replica scheduler must apply the completed stage batch");
  expect(
      requests[0].completed() && replica_scheduler.idle() &&
          !stage.is_busy() && stage.empty(),
      "the complete hierarchy must quiesce after request completion");
}

void test_colocation_hierarchy_rejects_unknown_targets() {
  std::vector<Request> requests;
  requests.emplace_back(WorkloadRequest{
      .request_id = RequestId{0},
      .arrived_at = SimTime::from_seconds(0.0),
      .num_prefill_tokens = 2,
      .num_decode_tokens = 1,
      .session_id = std::nullopt,
      .session_turn_index = std::nullopt,
  });
  auto replica = std::make_unique<VllmV1Scheduler>(
      scheduler_config(), requests);
  auto cluster = std::make_unique<CoLocationClusterScheduler>(
      std::move(replica));
  CoLocationGlobalScheduler global{std::move(cluster)};

  expect_throws<GlobalSchedulerError>(
      [&global] {
        static_cast<void>(global.get_cluster_scheduler(
            static_cast<ClusterType>(99)));
      },
      "global scheduler must reject unknown clusters");
  BaseClusterScheduler& monolithic =
      global.get_cluster_scheduler(ClusterType::kMonolithic);
  expect_throws<ClusterSchedulerError>(
      [&monolithic] {
        static_cast<void>(monolithic.get_replica_scheduler(
            ReplicaId{1}, DataParallelId{0}));
      },
      "cluster scheduler must reject unknown replicas");
  BaseReplicaScheduler& replica_scheduler =
      monolithic.get_replica_scheduler(
          ReplicaId{0}, DataParallelId{0});
  expect_throws<SchedulerError>(
      [&replica_scheduler] {
        static_cast<void>(
            replica_scheduler.get_replica_stage_scheduler(StageId{1}));
      },
      "replica scheduler must reject unknown pipeline stages");
}

}  // namespace

int main() {
  int failures = 0;
  failures += frontier::test::run(
      "co-location scheduler hierarchy routes and executes",
      test_colocation_scheduler_hierarchy_routes_and_executes);
  failures += frontier::test::run(
      "co-location hierarchy rejects unknown targets",
      test_colocation_hierarchy_rejects_unknown_targets);
  return failures == 0 ? 0 : 1;
}
