#include "frontier/simulator/co_location_simulator.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "frontier/core/event_queue.h"
#include "frontier/entities/batch.h"
#include "frontier/entities/request.h"
#include "frontier/events/co_location_events.h"
#include "frontier/execution_time_predictor/batch_execution_model.h"
#include "frontier/scheduler/cluster_scheduler/co_location_cluster_scheduler.h"
#include "frontier/scheduler/global_scheduler/co_location_global_scheduler.h"
#include "frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.h"
#include "frontier/simulator/simulation_context.h"

namespace frontier::simulator {
namespace {

RequestId require_request_id(const Event& event) {
  if (!event.payload.request_id.has_value()) {
    throw CoLocationSimulationError(
        "request arrival event is missing request_id");
  }
  return event.payload.request_id.value();
}

BatchId require_batch_id(const Event& event) {
  if (!event.payload.batch_id.has_value()) {
    throw CoLocationSimulationError(
        "batch completion event is missing batch_id");
  }
  return event.payload.batch_id.value();
}

entities::Request& get_request(
    std::vector<entities::Request>& requests,
    RequestId request_id) {
  if (request_id.value() >= requests.size()) {
    throw CoLocationSimulationError(
        "event references an unknown request");
  }
  entities::Request& request =
      requests.at(static_cast<std::size_t>(request_id.value()));
  if (request.id() != request_id) {
    throw CoLocationSimulationError(
        "request arena ID/index invariant failed");
  }
  return request;
}

entities::Batch& get_batch(
    std::vector<entities::Batch>& batches,
    BatchId batch_id) {
  if (batch_id.value() >= batches.size()) {
    throw CoLocationSimulationError(
        "event references an unknown batch");
  }
  entities::Batch& batch =
      batches.at(static_cast<std::size_t>(batch_id.value()));
  if (batch.id() != batch_id) {
    throw CoLocationSimulationError(
        "batch arena ID/index invariant failed");
  }
  return batch;
}

void validate_inputs(
    const config::SimulationConfig& config,
    const std::vector<request_generator::WorkloadRequest>& workload) {
  if (config.schema_version != config::kSchedulerSchemaVersion &&
      config.schema_version != config::kParallelSchemaVersion &&
      config.schema_version != config::kPddSchemaVersion) {
    throw CoLocationSimulationError(
        "typed scheduler requires config schema_version=2, 3, or 4");
  }
  const bool pdd =
      config.schema_version == config::kPddSchemaVersion;
  if ((!pdd &&
       config.system_architecture !=
           config::SystemArchitecture::kCoLocation) ||
      (pdd &&
       config.system_architecture !=
           config::SystemArchitecture::kPdDisaggregation)) {
    throw CoLocationSimulationError(
        "scheduler schema and system architecture disagree");
  }
  if (config.enable_parallel_clusters) {
    throw CoLocationSimulationError(
        "Step 2 scheduler requires sequential execution");
  }
  if (config.prefix_cache.enabled) {
    throw CoLocationSimulationError(
        "Step 2 scheduler requires prefix caching disabled");
  }
  if ((!pdd &&
       (!config.scheduler.has_value() ||
        !config.execution_model.has_value())) ||
      (pdd &&
       (!config.clusters.has_value() ||
        !config.kv_cache_transfer.has_value()))) {
    throw CoLocationSimulationError(
        "schema v2 requires scheduler and execution model configs");
  }
  for (std::size_t index = 0; index < workload.size(); ++index) {
    const request_generator::WorkloadRequest& request = workload[index];
    if (request.request_id.value() != index) {
      throw CoLocationSimulationError(
          "workload request IDs must be contiguous and start at zero");
    }
    if (!std::isfinite(request.arrived_at.seconds()) ||
        request.arrived_at.seconds() < 0.0 ||
        request.num_prefill_tokens == 0 ||
        request.num_decode_tokens == 0) {
      throw CoLocationSimulationError(
          "workload contains an invalid request");
    }
  }
}

metrics::SimulationOutput run_step25_simulation(
    const config::SimulationConfig& config,
    const std::vector<request_generator::WorkloadRequest>& workload) {
  SimulationContext context{config, workload};
  const events::EventDispatcher dispatcher;
  while (!context.event_queue().empty()) {
    Event event = context.event_queue().pop();
    context.output().event_trace.push_back(event);
    dispatcher.dispatch(event, context);
  }
  context.finalize();
  return context.take_output();
}

metrics::SchedulerTraceRecord make_trace(
    const scheduler::ScheduleResult& schedule) {
  metrics::SchedulerTraceRecord trace{
      .iteration_id = schedule.iteration_id,
      .simulation_time = schedule.simulation_time,
      .token_budget_before = schedule.token_budget_before,
      .token_budget_after = schedule.token_budget_after,
      .available_blocks_before = schedule.available_blocks_before,
      .available_blocks_after = schedule.available_blocks_after,
      .waiting_count_before = schedule.waiting_count_before,
      .waiting_count_after = schedule.waiting_count_after,
      .running_count_before = schedule.running_count_before,
      .running_count_after = schedule.running_count_after,
      .preempted_count = schedule.preempted_count,
      .decisions = {},
      .batch_request_ids = {},
      .request_num_tokens = {},
  };
  trace.decisions.reserve(schedule.decisions.size());
  for (const scheduler::SchedulerDecision& decision :
       schedule.decisions) {
    trace.decisions.push_back(metrics::SchedulerDecisionRecord{
        .decision_result =
            std::string{scheduler::to_string(decision.type)},
        .request_id = decision.request_id,
        .num_tokens = decision.num_tokens,
        .token_budget_after = decision.token_budget_after,
        .available_blocks_after =
            decision.available_blocks_after,
    });
  }
  trace.batch_request_ids.reserve(
      schedule.scheduled_requests.size());
  trace.request_num_tokens.reserve(
      schedule.scheduled_requests.size());
  for (const scheduler::ScheduledRequest& request :
       schedule.scheduled_requests) {
    trace.batch_request_ids.push_back(request.request_id);
    trace.request_num_tokens.push_back(request.num_tokens);
  }
  return trace;
}

std::vector<entities::RequestBatchSnapshot> make_batch_snapshots(
    const scheduler::ScheduleResult& schedule,
    std::vector<entities::Request>& requests) {
  std::vector<entities::RequestBatchSnapshot> snapshots;
  snapshots.reserve(schedule.scheduled_requests.size());
  for (const scheduler::ScheduledRequest& scheduled :
       schedule.scheduled_requests) {
    const entities::Request& request =
        get_request(requests, scheduled.request_id);
    snapshots.push_back(entities::RequestBatchSnapshot{
        .request_id = scheduled.request_id,
        .scheduled_tokens = scheduled.num_tokens,
        .runtime_epoch = request.runtime_epoch(),
        .execution_epoch = request.execution_epoch(),
        .processed_tokens = request.num_processed_tokens(),
        .scheduler_frontier =
            request.scheduler_num_computed_tokens(),
    });
  }
  return snapshots;
}

metrics::BatchMetricsRecord make_batch_metrics(
    const entities::Batch& batch,
    const std::vector<entities::Request>& requests,
    double predicted_execution_ms) {
  if (!batch.completed_at().has_value()) {
    throw CoLocationSimulationError(
        "cannot emit metrics for an incomplete batch");
  }
  metrics::BatchMetricsRecord record{
      .batch_id = batch.id(),
      .iteration_id = batch.iteration_id(),
      .scheduled_at = batch.scheduled_at(),
      .completed_at = batch.completed_at().value(),
      .request_ids = {},
      .scheduled_tokens = {},
      .total_scheduled_tokens = batch.total_scheduled_tokens(),
      .num_prefill_tokens = 0,
      .num_decode_tokens = 0,
      .predicted_execution_ms = predicted_execution_ms,
  };
  for (const entities::RequestBatchSnapshot& snapshot :
       batch.requests()) {
    if (snapshot.request_id.value() >= requests.size()) {
      throw CoLocationSimulationError(
          "batch metrics reference an unknown request");
    }
    const entities::Request& request =
        requests.at(
            static_cast<std::size_t>(snapshot.request_id.value()));
    record.request_ids.push_back(snapshot.request_id);
    record.scheduled_tokens.push_back(snapshot.scheduled_tokens);
    if (snapshot.processed_tokens < request.num_prefill_tokens()) {
      record.num_prefill_tokens += snapshot.scheduled_tokens;
    } else {
      record.num_decode_tokens += snapshot.scheduled_tokens;
    }
  }
  return record;
}

metrics::RequestMetricsRecord make_request_metrics(
    const entities::Request& request) {
  if (!request.completed() ||
      !request.first_scheduled_at().has_value() ||
      !request.prefill_completed_at().has_value() ||
      !request.first_token_completed_at().has_value() ||
      !request.completed_at().has_value()) {
    throw CoLocationSimulationError(
        "canonical output requires a fully completed request");
  }
  return metrics::RequestMetricsRecord{
      .request_id = request.id(),
      .arrived_at = request.arrived_at(),
      .prefill_completed_at =
          request.prefill_completed_at().value(),
      .completed_at = request.completed_at().value(),
      .first_scheduled_at = request.first_scheduled_at(),
      .first_token_completed_at =
          request.first_token_completed_at(),
      .num_processed_tokens = request.num_processed_tokens(),
      .preemption_count = request.preemption_count(),
      // Production Python records this list only for disaggregated decode
      // clusters; MONOLITHIC exposes the count but keeps the list empty.
      .tokens_at_preemption = {},
      .replica_id = ReplicaId{0},
      .dp_id = DataParallelId{0},
      .prefill_replica_id = std::nullopt,
      .prefill_dp_id = std::nullopt,
      .decode_replica_id = std::nullopt,
      .decode_dp_id = std::nullopt,
      .transfer_id = std::nullopt,
      .kv_cache_transfer_start_time = std::nullopt,
      .kv_cache_transfer_end_time = std::nullopt,
      .decode_arrived_at = std::nullopt,
      .kv_cache_transfer_size_bytes = 0,
  };
}

}  // namespace

metrics::SimulationOutput run_co_location_simulation(
    const config::SimulationConfig& config,
    const std::vector<request_generator::WorkloadRequest>& workload) {
  validate_inputs(config, workload);
  request_generator::validate_workload_for_config(workload, config);
  if (config.schema_version == config::kParallelSchemaVersion ||
      config.schema_version == config::kPddSchemaVersion) {
    return run_step25_simulation(config, workload);
  }

  std::vector<entities::Request> requests;
  requests.reserve(workload.size());
  for (const request_generator::WorkloadRequest& request : workload) {
    requests.emplace_back(request);
  }

  auto replica_scheduler =
      std::make_unique<scheduler::VllmV1Scheduler>(
          config.scheduler.value(),
          requests,
          execution_time_predictor::make_batch_execution_model(
              config.execution_model.value()));
  auto cluster_scheduler =
      std::make_unique<scheduler::CoLocationClusterScheduler>(
          std::move(replica_scheduler));
  scheduler::CoLocationGlobalScheduler global_scheduler{
      std::move(cluster_scheduler)};
  scheduler::BaseClusterScheduler& monolithic_cluster =
      global_scheduler.get_cluster_scheduler(
          scheduler::ClusterType::kMonolithic);
  scheduler::BaseReplicaScheduler& replica =
      monolithic_cluster.get_replica_scheduler(
          ReplicaId{0},
          DataParallelId{0});
  scheduler::ReplicaStageScheduler& stage =
      replica.get_replica_stage_scheduler(StageId{0});

  metrics::SimulationOutput output{
      .schema_version = config::kSchedulerSchemaVersion,
      .run = metrics::RunMetadata{
          .run_id = config.run_id,
          .simulation_mode = config.simulation_mode,
          .system_architecture = config.system_architecture,
          .metrics_semantics = metrics::MetricsSemantics::kCanonical,
      },
      .requests = {},
      .batches = {},
      .batch_stages = {},
      .scheduler_trace = {},
      .event_trace = {},
      .analytical_diagnostics = {},
      .kv_cache_transfers = {},
  };
  output.event_trace.reserve(workload.size() * 3);

  EventQueue event_queue;
  for (const request_generator::WorkloadRequest& request : workload) {
    EventPayload payload;
    payload.request_id = request.request_id;
    event_queue.push(
        request.arrived_at,
        EventType::kRequestArrival,
        std::move(payload));
  }

  std::vector<entities::Batch> batches;
  std::vector<double> predicted_batch_ms;
  std::vector<RequestId> completion_order;
  std::vector<bool> completion_recorded(requests.size(), false);

  while (!event_queue.empty()) {
    Event event = event_queue.pop();
    output.event_trace.push_back(event);

    switch (event.type) {
      case EventType::kRequestArrival: {
        entities::Request& request =
            get_request(requests, require_request_id(event));
        request.on_arrival(event.time);
        global_scheduler.add_request(
            request.id(),
            scheduler::ClusterType::kMonolithic);
        for (const scheduler::GlobalRequestAssignment& assignment :
             global_scheduler.schedule()) {
          scheduler::BaseClusterScheduler& cluster =
              global_scheduler.get_cluster_scheduler(
                  assignment.cluster_type);
          cluster.add_request(
              assignment.request_id,
              request.arrived_at());
          static_cast<void>(cluster.schedule());
        }
        event_queue.push(event.time, EventType::kSchedulerPoll);
        break;
      }
      case EventType::kSchedulerPoll: {
        if (replica.has_in_flight_batch() ||
            !replica.has_pending_work()) {
          break;
        }
        scheduler::ScheduleResult schedule =
            replica.schedule(event.time);
        output.scheduler_trace.push_back(make_trace(schedule));
        if (schedule.scheduled_requests.empty()) {
          break;
        }

        const BatchId batch_id{
            static_cast<BatchId::ValueType>(batches.size())};
        const Generation generation{
            static_cast<Generation::ValueType>(batches.size() + 1)};
        batches.emplace_back(
            batch_id,
            schedule.iteration_id,
            make_batch_snapshots(schedule, requests),
            event.time,
            generation);
        replica.mark_batch_started(batches.back());
        stage.add_batch(batches.back());
        const std::optional<scheduler::StageBatchTicket> stage_ticket =
            stage.pop_batch_if_not_busy();
        if (!stage_ticket.has_value() ||
            stage_ticket->batch_id != batch_id ||
            stage_ticket->schedule_epoch != generation) {
          throw CoLocationSimulationError(
              "replica stage did not dispatch the scheduled batch");
        }

        const auto prediction = stage.predict(
            batches.back(), requests);
        if (!std::isfinite(prediction.duration_ms) ||
            prediction.duration_ms < 0.0) {
          throw CoLocationSimulationError(
              "execution model returned an invalid duration");
        }
        predicted_batch_ms.push_back(prediction.duration_ms);
        if (config.execution_model->type ==
            config::ExecutionModelType::kAnalytical) {
          output.analytical_diagnostics.push_back(
              metrics::AnalyticalDiagnostic{
                  .name =
                      "batch_" + std::to_string(batch_id.value()),
                  .values = prediction.diagnostics,
              });
        }

        const double completion_seconds =
            event.time.seconds() + prediction.duration_ms / 1e3;
        if (!std::isfinite(completion_seconds)) {
          throw CoLocationSimulationError(
              "batch completion time is nonfinite");
        }
        EventPayload completion_payload;
        completion_payload.batch_id = batch_id;
        completion_payload.generation = generation;
        event_queue.push(
            SimTime::from_seconds(completion_seconds),
            EventType::kBatchCompletion,
            std::move(completion_payload));
        break;
      }
      case EventType::kBatchCompletion: {
        const BatchId batch_id = require_batch_id(event);
        entities::Batch& batch = get_batch(batches, batch_id);
        if (!event.payload.generation.has_value() ||
            event.payload.generation.value() !=
                batch.schedule_epoch()) {
          break;
        }
        stage.on_stage_end(batch_id);
        const bool mutated =
            replica.on_batch_completed(batch, event.time);
        if (!mutated) {
          break;
        }
        output.batches.push_back(make_batch_metrics(
            batch,
            requests,
            predicted_batch_ms.at(
                static_cast<std::size_t>(batch_id.value()))));

        for (const entities::RequestBatchSnapshot& snapshot :
             batch.requests()) {
          const std::size_t index =
              static_cast<std::size_t>(snapshot.request_id.value());
          if (requests.at(index).completed() &&
              !completion_recorded.at(index)) {
            completion_recorded.at(index) = true;
            completion_order.push_back(snapshot.request_id);
          }
        }
        event_queue.push(event.time, EventType::kSchedulerPoll);
        break;
      }
      case EventType::kFoundationCompletion:
        throw CoLocationSimulationError(
            "foundation completion event leaked into schema v2");
      case EventType::kGlobalSchedule:
      case EventType::kClusterSchedule:
      case EventType::kReplicaSchedule:
      case EventType::kBatchStageArrival:
      case EventType::kReplicaStageSchedule:
      case EventType::kBatchStageEnd:
      case EventType::kClusterBatchEnd:
      case EventType::kGlobalBatchEnd:
      case EventType::kKvCacheTransferStart:
      case EventType::kKvCacheTransferEnd:
        throw CoLocationSimulationError(
            "schema v3 event leaked into schema v2");
    }
  }

  if (completion_order.size() != requests.size() ||
      !replica.idle() || stage.is_busy() || !stage.empty()) {
    throw CoLocationSimulationError(
        "scheduler quiesced with incomplete work: completed=" +
        std::to_string(completion_order.size()) +
        "/" + std::to_string(requests.size()) +
        ", waiting=" + std::to_string(replica.waiting_count()) +
        ", running=" + std::to_string(replica.running_count()) +
        ", allocated_blocks=" +
        std::to_string(replica.allocated_kv_blocks()));
  }

  output.requests.reserve(completion_order.size());
  for (const RequestId request_id : completion_order) {
    output.requests.push_back(
        make_request_metrics(get_request(requests, request_id)));
  }
  return output;
}

}  // namespace frontier::simulator
