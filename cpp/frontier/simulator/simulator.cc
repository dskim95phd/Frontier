#include "frontier/simulator/simulator.h"

#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "frontier/core/event_queue.h"
#include "frontier/simulator/co_location_simulator.h"

namespace frontier::simulator {
namespace {

enum class RequestState : std::uint8_t {
  kPending,
  kArrived,
  kCompleted,
};

struct FoundationRequestState {
  RequestState state = RequestState::kPending;
  std::optional<SimTime> completion_time;
};

RequestId require_request_id(const Event& event) {
  if (!event.payload.request_id.has_value()) {
    throw FoundationSimulationError(
        "foundation lifecycle event is missing request_id");
  }
  return event.payload.request_id.value();
}

std::size_t require_request_index(
    RequestId request_id,
    std::size_t request_count) {
  const std::uint64_t value = request_id.value();
  if (value >= request_count) {
    throw FoundationSimulationError(
        "foundation lifecycle event has unknown request_id=" +
        std::to_string(value));
  }
  return static_cast<std::size_t>(value);
}

void validate_foundation_inputs(
    const std::vector<request_generator::WorkloadRequest>& workload,
    const FoundationLifecycleOptions& options) {
  if (!std::isfinite(options.service_time_ms) ||
      options.service_time_ms < 0.0) {
    throw FoundationSimulationError(
        "foundation service_time_ms must be finite and nonnegative");
  }

  for (std::size_t index = 0; index < workload.size(); ++index) {
    const request_generator::WorkloadRequest& request = workload[index];
    if (request.request_id.value() != index) {
      throw FoundationSimulationError(
          "foundation workload request IDs must be contiguous and start "
          "at zero");
    }
    if (!std::isfinite(request.arrived_at.seconds()) ||
        request.arrived_at.seconds() < 0.0) {
      throw FoundationSimulationError(
          "foundation workload arrival times must be finite and "
          "nonnegative");
    }
    if (request.num_prefill_tokens == 0 ||
        request.num_decode_tokens == 0) {
      throw FoundationSimulationError(
          "foundation workload token counts must be positive");
    }
  }
}

}  // namespace

metrics::SimulationOutput run_foundation_lifecycle(
    const config::SimulationConfig& config,
    const std::vector<request_generator::WorkloadRequest>& workload,
    const FoundationLifecycleOptions& options) {
  validate_foundation_inputs(workload, options);
  if (config.system_architecture !=
      config::SystemArchitecture::kCoLocation) {
    throw FoundationSimulationError(
        "foundation lifecycle supports only co-location; "
        "sequential PDD is introduced in Step 3");
  }
  if (config.prefix_cache.enabled) {
    throw FoundationSimulationError(
        "foundation lifecycle does not implement prefix caching; "
        "disable config.prefix_cache.enabled");
  }
  request_generator::validate_workload_for_config(workload, config);

  metrics::SimulationOutput output{
      .schema_version = config::kFoundationSchemaVersion,
      .run = metrics::RunMetadata{
          .run_id = config.run_id,
          .simulation_mode = config.simulation_mode,
          .system_architecture = config.system_architecture,
          .metrics_semantics =
              metrics::MetricsSemantics::kFoundationPlaceholder,
      },
      .requests = {},
      .batches = {},
      .batch_stages = {},
      .scheduler_trace = {},
      .event_trace = {},
      .analytical_diagnostics = {},
      .kv_cache_transfers = {},
  };
  output.requests.reserve(workload.size());
  output.event_trace.reserve(workload.size() * 2);

  EventQueue event_queue;
  for (const request_generator::WorkloadRequest& request : workload) {
    EventPayload payload;
    payload.request_id = request.request_id;
    event_queue.push(
        request.arrived_at,
        EventType::kRequestArrival,
        std::move(payload));
  }

  std::vector<FoundationRequestState> request_states(workload.size());
  const double service_time_s = options.service_time_ms / 1e3;

  while (!event_queue.empty()) {
    Event event = event_queue.pop();
    output.event_trace.push_back(event);

    const RequestId request_id = require_request_id(event);
    const std::size_t request_index =
        require_request_index(request_id, workload.size());
    FoundationRequestState& state = request_states[request_index];
    const request_generator::WorkloadRequest& request =
        workload[request_index];

    switch (event.type) {
      case EventType::kRequestArrival: {
        if (state.state != RequestState::kPending) {
          throw FoundationSimulationError(
              "request arrived more than once; request_id=" +
              std::to_string(request_id.value()));
        }
        if (event.time != request.arrived_at) {
          throw FoundationSimulationError(
              "arrival event time differs from workload; request_id=" +
              std::to_string(request_id.value()));
        }

        const double completion_seconds =
            event.time.seconds() + service_time_s;
        if (!std::isfinite(completion_seconds)) {
          throw FoundationSimulationError(
              "foundation completion time is nonfinite; request_id=" +
              std::to_string(request_id.value()));
        }

        state.state = RequestState::kArrived;
        state.completion_time =
            SimTime::from_seconds(completion_seconds);
        EventPayload completion_payload;
        completion_payload.request_id = request_id;
        event_queue.push(
            state.completion_time.value(),
            EventType::kFoundationCompletion,
            std::move(completion_payload));
        break;
      }
      case EventType::kFoundationCompletion:
        if (state.state != RequestState::kArrived ||
            !state.completion_time.has_value()) {
          throw FoundationSimulationError(
              "request completed without one live arrival; request_id=" +
              std::to_string(request_id.value()));
        }
        if (event.time != state.completion_time.value()) {
          throw FoundationSimulationError(
              "completion event time differs from scheduled time; "
              "request_id=" +
              std::to_string(request_id.value()));
        }

        state.state = RequestState::kCompleted;
        output.requests.push_back(metrics::RequestMetricsRecord{
            .request_id = request_id,
            .arrived_at = request.arrived_at,
            .prefill_completed_at = event.time,
            .completed_at = event.time,
            .first_scheduled_at = std::nullopt,
            .first_token_completed_at = std::nullopt,
            .num_processed_tokens = 0,
            .preemption_count = 0,
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
        });
        break;
      case EventType::kSchedulerPoll:
      case EventType::kBatchCompletion:
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
        throw FoundationSimulationError(
            "scheduler event leaked into foundation lifecycle");
    }
  }

  if (output.requests.size() != workload.size()) {
    throw FoundationSimulationError(
        "foundation lifecycle did not complete every request");
  }
  for (std::size_t index = 0; index < request_states.size(); ++index) {
    if (request_states[index].state != RequestState::kCompleted) {
      throw FoundationSimulationError(
          "foundation lifecycle left an incomplete request_id=" +
          std::to_string(index));
    }
  }
  return output;
}

metrics::SimulationOutput run_simulation(
    const config::SimulationConfig& config,
    const std::vector<request_generator::WorkloadRequest>& workload) {
  if (config.schema_version == config::kFoundationSchemaVersion) {
    return run_foundation_lifecycle(config, workload);
  }
  if (config.schema_version == config::kSchedulerSchemaVersion ||
      config.schema_version == config::kParallelSchemaVersion ||
      config.schema_version == config::kPddSchemaVersion) {
    return run_co_location_simulation(config, workload);
  }
  throw FoundationSimulationError(
      "unsupported simulation config schema_version=" +
      std::to_string(config.schema_version));
}

}  // namespace frontier::simulator
