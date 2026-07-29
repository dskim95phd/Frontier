#pragma once

#include <string>
#include <utility>
#include <vector>

#include "frontier/config/config.h"
#include "frontier/core/event.h"
#include "frontier/core/ids.h"

namespace frontier::metrics {

inline constexpr int kOutputSchemaVersion = 1;

enum class MetricsSemantics {
  kCanonical,
  kFoundationPlaceholder,
};

struct RunMetadata {
  std::string run_id;
  config::SimulationMode simulation_mode;
  config::SystemArchitecture system_architecture;
  MetricsSemantics metrics_semantics = MetricsSemantics::kCanonical;
};

struct RequestMetricsRecord {
  RequestId request_id;
  SimTime arrived_at;
  SimTime prefill_completed_at;
  SimTime completed_at;
};

struct AnalyticalDiagnostic {
  std::string name;
  std::vector<std::pair<std::string, double>> values;
};

struct SimulationOutput {
  RunMetadata run;
  std::vector<RequestMetricsRecord> requests;
  std::vector<Event> event_trace;
  std::vector<AnalyticalDiagnostic> analytical_diagnostics;
};

[[nodiscard]] std::string_view to_string(EventType event_type) noexcept;
[[nodiscard]] std::string_view to_string(
    MetricsSemantics semantics) noexcept;
[[nodiscard]] std::string serialize_simulation_output_json(
    const SimulationOutput& output);
[[nodiscard]] std::string serialize_request_metrics_csv(
    const std::vector<RequestMetricsRecord>& requests);

}  // namespace frontier::metrics
