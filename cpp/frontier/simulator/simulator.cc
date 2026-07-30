#include "frontier/simulator/simulator.h"

#include "frontier/simulator/scheduler_simulator.h"

namespace frontier::simulator {

metrics::SimulationOutput run_simulation(
    const config::SimulationConfig& config,
    const std::vector<request_generator::WorkloadRequest>& workload) {
  return run_scheduler_simulation(config, workload);
}

}  // namespace frontier::simulator
