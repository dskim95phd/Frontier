#pragma once

#include <stdexcept>
#include <vector>

#include "frontier/config/config.h"
#include "frontier/metrics/output_contract.h"
#include "frontier/request_generator/workload.h"

namespace frontier::simulator {

struct FoundationLifecycleOptions {
  // Placeholder latency used only by the scheduler-free Step 1 lifecycle.
  double service_time_ms = 1.0;
};

class FoundationSimulationError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

[[nodiscard]] metrics::SimulationOutput run_foundation_lifecycle(
    const config::SimulationConfig& config,
    const std::vector<request_generator::WorkloadRequest>& workload,
    const FoundationLifecycleOptions& options = {});

[[nodiscard]] metrics::SimulationOutput run_simulation(
    const config::SimulationConfig& config,
    const std::vector<request_generator::WorkloadRequest>& workload);

}  // namespace frontier::simulator
