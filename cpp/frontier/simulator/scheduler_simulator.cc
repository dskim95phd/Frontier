#include "frontier/simulator/scheduler_simulator.h"

#include <cmath>
#include <string>

#include "frontier/events/event_dispatcher.h"
#include "frontier/simulator/simulation_context.h"

namespace frontier::simulator {
namespace {

void validate_inputs(
    const config::SimulationConfig& config,
    const std::vector<request_generator::WorkloadRequest>& workload) {
  if (config.schema_version != config::kSchemaVersion) {
    throw SimulationError(
        "scheduler requires the current config schema_version");
  }
  const bool is_pdd =
      config.system_architecture ==
      config::SystemArchitecture::kPdDisaggregation;
  if (config.enable_parallel_clusters) {
    throw SimulationError(
        "scheduler runtime requires sequential cluster execution");
  }
  if (config.prefix_cache.enabled) {
    throw SimulationError(
        "scheduler runtime requires prefix caching disabled");
  }
  if ((!is_pdd &&
       !std::holds_alternative<config::ClusterRuntimeConfig>(
           config.runtime)) ||
      (is_pdd &&
       !std::holds_alternative<config::PddRuntimeConfig>(
           config.runtime))) {
    throw SimulationError(
        "system architecture has an incompatible runtime config");
  }
  for (std::size_t index = 0; index < workload.size(); ++index) {
    const request_generator::WorkloadRequest& request = workload[index];
    if (!request.request_id.valid() ||
        request.request_id.index() != index) {
      throw SimulationError(
          "workload request IDs must be contiguous and start at zero");
    }
    if (!std::isfinite(request.arrived_at.seconds()) ||
        request.arrived_at.seconds() < 0.0 ||
        request.num_prefill_tokens == 0 ||
        request.num_decode_tokens == 0) {
      throw SimulationError("workload contains an invalid request");
    }
  }
}

}  // namespace

metrics::SimulationOutput run_scheduler_simulation(
    const config::SimulationConfig& config,
    const std::vector<request_generator::WorkloadRequest>& workload) {
  validate_inputs(config, workload);
  request_generator::validate_workload_for_config(workload, config);

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

}  // namespace frontier::simulator
