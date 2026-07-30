#pragma once

#include <vector>

#include "frontier/config/config.h"
#include "frontier/metrics/output_contract.h"
#include "frontier/request_generator/workload.h"
#include "frontier/simulator/simulator.h"

namespace frontier::simulator {

[[nodiscard]] metrics::SimulationOutput run_scheduler_simulation(
    const config::SimulationConfig& config,
    const std::vector<request_generator::WorkloadRequest>& workload);

}  // namespace frontier::simulator
