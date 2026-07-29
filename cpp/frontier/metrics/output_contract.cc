#include "frontier/metrics/output_contract.h"

#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>

#include <nlohmann/json.hpp>

namespace frontier::metrics {
namespace {

using OrderedJson = nlohmann::ordered_json;

void require_valid_time(SimTime time, std::string_view field) {
  if (!std::isfinite(time.seconds()) || time.seconds() < 0.0) {
    throw std::invalid_argument(
        std::string{field} + " must be finite and nonnegative");
  }
}

double milliseconds_between(
    SimTime end,
    SimTime start,
    std::string_view field) {
  require_valid_time(start, "arrived_at");
  require_valid_time(end, field);
  if (end.seconds() < start.seconds()) {
    throw std::invalid_argument(
        std::string{field} + " must not precede arrived_at");
  }
  return (end.seconds() - start.seconds()) * 1e3;
}

void validate_request_metrics(
    const std::vector<RequestMetricsRecord>& requests) {
  std::unordered_set<std::uint64_t> request_ids;
  request_ids.reserve(requests.size());
  for (const RequestMetricsRecord& request : requests) {
    if (!request_ids.insert(request.request_id.value()).second) {
      throw std::invalid_argument(
          "request metrics contain duplicate request_id=" +
          std::to_string(request.request_id.value()));
    }
    static_cast<void>(milliseconds_between(
        request.prefill_completed_at,
        request.arrived_at,
        "prefill_completed_at"));
    static_cast<void>(milliseconds_between(
        request.completed_at,
        request.arrived_at,
        "completed_at"));
    if (request.completed_at.seconds() <
        request.prefill_completed_at.seconds()) {
      throw std::invalid_argument(
          "completed_at must not precede prefill_completed_at");
    }
  }
}

OrderedJson serialize_request(const RequestMetricsRecord& request) {
  OrderedJson json = OrderedJson::object();
  json["request_id"] = request.request_id.value();
  json["arrived_at_s"] = request.arrived_at.seconds();
  json["prefill_completed_at_s"] =
      request.prefill_completed_at.seconds();
  json["completed_at_s"] = request.completed_at.seconds();
  json["ttft_ms"] = milliseconds_between(
      request.prefill_completed_at,
      request.arrived_at,
      "prefill_completed_at");
  json["e2e_ms"] = milliseconds_between(
      request.completed_at,
      request.arrived_at,
      "completed_at");
  return json;
}

OrderedJson serialize_event(const Event& event) {
  require_valid_time(event.time, "event.time");

  OrderedJson json = OrderedJson::object();
  json["time_s"] = event.time.seconds();
  json["sequence"] = event.sequence.value();
  json["type"] = to_string(event.type);
  if (event.payload.request_id.has_value()) {
    json["request_id"] = event.payload.request_id->value();
  }
  if (event.payload.batch_id.has_value()) {
    json["batch_id"] = event.payload.batch_id->value();
  }
  if (event.payload.replica_id.has_value()) {
    json["replica_id"] = event.payload.replica_id->value();
  }
  if (event.payload.dp_id.has_value()) {
    json["dp_id"] = event.payload.dp_id->value();
  }
  if (event.payload.generation.has_value()) {
    json["generation"] = event.payload.generation->value();
  }
  return json;
}

OrderedJson serialize_diagnostic(
    const AnalyticalDiagnostic& diagnostic) {
  if (diagnostic.name.empty()) {
    throw std::invalid_argument(
        "analytical diagnostic name must not be empty");
  }

  OrderedJson values = OrderedJson::object();
  for (const auto& [name, value] : diagnostic.values) {
    if (name.empty()) {
      throw std::invalid_argument(
          "analytical diagnostic field name must not be empty");
    }
    if (!std::isfinite(value)) {
      throw std::invalid_argument(
          "analytical diagnostic values must be finite");
    }
    if (values.contains(name)) {
      throw std::invalid_argument(
          "analytical diagnostic contains duplicate field '" + name + "'");
    }
    values[name] = value;
  }

  OrderedJson json = OrderedJson::object();
  json["name"] = diagnostic.name;
  json["values"] = std::move(values);
  return json;
}

}  // namespace

std::string_view to_string(EventType event_type) noexcept {
  switch (event_type) {
    case EventType::kRequestArrival:
      return "request_arrival";
    case EventType::kFoundationCompletion:
      return "foundation_completion";
  }
  return "unknown";
}

std::string_view to_string(MetricsSemantics semantics) noexcept {
  switch (semantics) {
    case MetricsSemantics::kCanonical:
      return "canonical";
    case MetricsSemantics::kFoundationPlaceholder:
      return "foundation-placeholder";
  }
  return "unknown";
}

std::string serialize_simulation_output_json(
    const SimulationOutput& output) {
  if (output.run.run_id.empty()) {
    throw std::invalid_argument("output run_id must not be empty");
  }
  validate_request_metrics(output.requests);

  OrderedJson root = OrderedJson::object();
  root["schema_version"] = kOutputSchemaVersion;
  root["run"] = OrderedJson::object({
      {"run_id", output.run.run_id},
      {
          "simulation_mode",
          std::string{config::to_string(output.run.simulation_mode)},
      },
      {
          "system_architecture",
          std::string{
              config::to_string(output.run.system_architecture)},
      },
      {"timestamp_unit", "seconds"},
      {"latency_unit", "milliseconds"},
      {
          "metrics_semantics",
          std::string{to_string(output.run.metrics_semantics)},
      },
  });

  root["completed_request_ids"] = OrderedJson::array();
  root["requests"] = OrderedJson::array();
  for (const RequestMetricsRecord& request : output.requests) {
    root["completed_request_ids"].push_back(request.request_id.value());
    root["requests"].push_back(serialize_request(request));
  }

  root["event_trace"] = OrderedJson::array();
  for (const Event& event : output.event_trace) {
    root["event_trace"].push_back(serialize_event(event));
  }

  root["analytical_diagnostics"] = OrderedJson::array();
  for (const AnalyticalDiagnostic& diagnostic :
       output.analytical_diagnostics) {
    root["analytical_diagnostics"].push_back(
        serialize_diagnostic(diagnostic));
  }

  return root.dump(2) + '\n';
}

std::string serialize_request_metrics_csv(
    const std::vector<RequestMetricsRecord>& requests) {
  validate_request_metrics(requests);

  std::ostringstream output;
  output.imbue(std::locale::classic());
  output << std::setprecision(std::numeric_limits<double>::max_digits10);
  output
      << "request_id,arrived_at_s,prefill_completed_at_s,completed_at_s,"
         "ttft_ms,e2e_ms\n";
  for (const RequestMetricsRecord& request : requests) {
    output << request.request_id.value() << ','
           << request.arrived_at.seconds() << ','
           << request.prefill_completed_at.seconds() << ','
           << request.completed_at.seconds() << ','
           << milliseconds_between(
                  request.prefill_completed_at,
                  request.arrived_at,
                  "prefill_completed_at")
           << ','
           << milliseconds_between(
                  request.completed_at,
                  request.arrived_at,
                  "completed_at")
           << '\n';
  }
  return output.str();
}

}  // namespace frontier::metrics
