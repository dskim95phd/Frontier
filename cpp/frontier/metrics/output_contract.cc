#include "frontier/metrics/output_contract.h"

#include <algorithm>
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

bool has_scheduler_contract(int schema_version) {
  return schema_version == config::kSchedulerSchemaVersion ||
      schema_version == config::kParallelSchemaVersion ||
      schema_version == config::kPddSchemaVersion;
}

bool has_parallel_contract(int schema_version) {
  return schema_version == config::kParallelSchemaVersion ||
      schema_version == config::kPddSchemaVersion;
}

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
    const std::vector<RequestMetricsRecord>& requests,
    int schema_version) {
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
    if (has_scheduler_contract(schema_version)) {
      if (!request.first_scheduled_at.has_value() ||
          !request.first_token_completed_at.has_value()) {
        throw std::invalid_argument(
            "schema v2 request metrics require canonical scheduling and "
            "first-token timestamps");
      }
      static_cast<void>(milliseconds_between(
          request.first_scheduled_at.value(),
          request.arrived_at,
          "first_scheduled_at"));
      static_cast<void>(milliseconds_between(
          request.first_token_completed_at.value(),
          request.arrived_at,
          "first_token_completed_at"));
      if (request.first_token_completed_at->seconds() <
          request.prefill_completed_at.seconds()) {
        throw std::invalid_argument(
            "first token completion must not precede prefill completion");
      }
    }
    if (schema_version == config::kPddSchemaVersion) {
      if (!request.prefill_replica_id.has_value() ||
          !request.prefill_dp_id.has_value() ||
          !request.decode_replica_id.has_value() ||
          !request.decode_dp_id.has_value() ||
          !request.transfer_id.has_value() ||
          !request.kv_cache_transfer_start_time.has_value() ||
          !request.kv_cache_transfer_end_time.has_value() ||
          !request.decode_arrived_at.has_value() ||
          request.kv_cache_transfer_size_bytes == 0) {
        throw std::invalid_argument(
            "schema v4 request metrics require complete PDD ownership");
      }
      const SimTime transfer_start =
          request.kv_cache_transfer_start_time.value();
      const SimTime transfer_end =
          request.kv_cache_transfer_end_time.value();
      const SimTime decode_arrival =
          request.decode_arrived_at.value();
      require_valid_time(transfer_start, "kv transfer start");
      require_valid_time(transfer_end, "kv transfer end");
      require_valid_time(decode_arrival, "decode arrival");
      if (transfer_start.seconds() <
              request.prefill_completed_at.seconds() ||
          transfer_end.seconds() < transfer_start.seconds() ||
          decode_arrival != transfer_end) {
        throw std::invalid_argument(
            "schema v4 transfer timestamps are out of order");
      }
    }
  }
}

OrderedJson serialize_request(
    const RequestMetricsRecord& request,
    int schema_version) {
  OrderedJson json = OrderedJson::object();
  json["request_id"] = request.request_id.value();
  json["arrived_at_s"] = request.arrived_at.seconds();
  if (has_scheduler_contract(schema_version)) {
    json["first_scheduled_at_s"] =
        request.first_scheduled_at->seconds();
  }
  json["prefill_completed_at_s"] =
      request.prefill_completed_at.seconds();
  if (has_scheduler_contract(schema_version)) {
    json["first_token_completed_at_s"] =
        request.first_token_completed_at->seconds();
  }
  json["completed_at_s"] = request.completed_at.seconds();
  if (has_scheduler_contract(schema_version)) {
    json["scheduling_delay_ms"] = milliseconds_between(
        request.first_scheduled_at.value(),
        request.arrived_at,
        "first_scheduled_at");
  }
  json["ttft_ms"] = milliseconds_between(
      request.prefill_completed_at,
      request.arrived_at,
      "prefill_completed_at");
  json["e2e_ms"] = milliseconds_between(
      request.completed_at,
      request.arrived_at,
      "completed_at");
  if (has_scheduler_contract(schema_version)) {
    json["num_processed_tokens"] = request.num_processed_tokens;
    json["preemption_count"] = request.preemption_count;
    json["tokens_at_preemption"] = request.tokens_at_preemption;
  }
  if (schema_version == config::kParallelSchemaVersion) {
    json["replica_id"] = request.replica_id.value();
    json["dp_id"] = request.dp_id.value();
  } else if (schema_version == config::kPddSchemaVersion) {
    if (!request.prefill_replica_id.has_value() ||
        !request.prefill_dp_id.has_value() ||
        !request.decode_replica_id.has_value() ||
        !request.decode_dp_id.has_value() ||
        !request.transfer_id.has_value() ||
        !request.kv_cache_transfer_start_time.has_value() ||
        !request.kv_cache_transfer_end_time.has_value() ||
        !request.decode_arrived_at.has_value() ||
        request.kv_cache_transfer_size_bytes == 0) {
      throw std::invalid_argument(
          "schema v4 request metrics require complete PDD ownership");
    }
    json["prefill_replica_id"] =
        request.prefill_replica_id->value();
    json["prefill_dp_id"] = request.prefill_dp_id->value();
    json["decode_replica_id"] =
        request.decode_replica_id->value();
    json["decode_dp_id"] = request.decode_dp_id->value();
    json["transfer_id"] = request.transfer_id->value();
    json["kv_cache_transfer_start_time_s"] =
        request.kv_cache_transfer_start_time->seconds();
    json["kv_cache_transfer_end_time_s"] =
        request.kv_cache_transfer_end_time->seconds();
    json["kv_cache_transfer_time_ms"] =
        milliseconds_between(
            request.kv_cache_transfer_end_time.value(),
            request.kv_cache_transfer_start_time.value(),
            "kv_cache_transfer_end_time");
    json["kv_cache_transfer_size_bytes"] =
        request.kv_cache_transfer_size_bytes;
    json["decode_arrived_at_s"] =
        request.decode_arrived_at->seconds();
  }
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
  if (event.payload.stage_id.has_value()) {
    json["stage_id"] = event.payload.stage_id->value();
  }
  if (event.payload.generation.has_value()) {
    json["generation"] = event.payload.generation->value();
  }
  if (event.payload.cluster_type.has_value()) {
    json["cluster_type"] =
        to_string(event.payload.cluster_type.value());
  }
  if (event.payload.transfer_id.has_value()) {
    json["transfer_id"] =
        event.payload.transfer_id->value();
  }
  return json;
}

OrderedJson serialize_batch(
    const BatchMetricsRecord& batch,
    int schema_version) {
  require_valid_time(batch.scheduled_at, "batch.scheduled_at");
  require_valid_time(batch.completed_at, "batch.completed_at");
  if (batch.completed_at.seconds() < batch.scheduled_at.seconds()) {
    throw std::invalid_argument(
        "batch completion must not precede scheduling");
  }
  if (batch.request_ids.empty() ||
      batch.request_ids.size() != batch.scheduled_tokens.size()) {
    throw std::invalid_argument(
        "batch request IDs and token counts must be nonempty and aligned");
  }
  std::uint64_t total = 0;
  for (const std::uint64_t tokens : batch.scheduled_tokens) {
    if (tokens == 0) {
      throw std::invalid_argument(
          "batch scheduled token counts must be positive");
    }
    total += tokens;
  }
  if (total != batch.total_scheduled_tokens ||
      batch.num_prefill_tokens + batch.num_decode_tokens != total) {
    throw std::invalid_argument(
        "batch token totals are inconsistent");
  }
  if (!std::isfinite(batch.predicted_execution_ms) ||
      batch.predicted_execution_ms < 0.0) {
    throw std::invalid_argument(
        "batch predicted execution time must be finite and nonnegative");
  }

  OrderedJson json = OrderedJson::object();
  json["batch_id"] = batch.batch_id.value();
  json["iteration_id"] = batch.iteration_id.value();
  json["scheduled_at_s"] = batch.scheduled_at.seconds();
  json["completed_at_s"] = batch.completed_at.seconds();
  json["request_ids"] = OrderedJson::array();
  for (const RequestId request_id : batch.request_ids) {
    json["request_ids"].push_back(request_id.value());
  }
  json["scheduled_tokens"] = batch.scheduled_tokens;
  json["total_scheduled_tokens"] = batch.total_scheduled_tokens;
  json["num_prefill_tokens"] = batch.num_prefill_tokens;
  json["num_decode_tokens"] = batch.num_decode_tokens;
  json["predicted_execution_ms"] = batch.predicted_execution_ms;
  if (has_parallel_contract(schema_version)) {
    json["replica_id"] = batch.replica_id.value();
    json["dp_id"] = batch.dp_id.value();
    json["num_pipeline_stages"] =
        batch.num_pipeline_stages;
    if (schema_version == config::kPddSchemaVersion) {
      json["cluster_type"] = to_string(batch.cluster_type);
    }
  }
  return json;
}

OrderedJson serialize_batch_stage(
    const BatchStageMetricsRecord& stage) {
  require_valid_time(stage.arrived_at, "batch_stage.arrived_at");
  require_valid_time(stage.started_at, "batch_stage.started_at");
  require_valid_time(stage.completed_at, "batch_stage.completed_at");
  if (stage.started_at.seconds() < stage.arrived_at.seconds() ||
      stage.completed_at.seconds() < stage.started_at.seconds()) {
    throw std::invalid_argument(
        "batch stage timestamps are out of order");
  }
  const entities::ExecutionTime& execution =
      stage.execution_time;
  if (!std::isfinite(execution.dense_compute_ms) ||
      !std::isfinite(execution.tp_communication_ms) ||
      !std::isfinite(execution.pp_communication_ms) ||
      execution.dense_compute_ms < 0.0 ||
      execution.tp_communication_ms < 0.0 ||
      execution.pp_communication_ms < 0.0) {
    throw std::invalid_argument(
        "batch stage execution time is invalid");
  }
  OrderedJson json = OrderedJson::object({
      {"batch_id", stage.batch_id.value()},
      {"replica_id", stage.replica_id.value()},
      {"dp_id", stage.dp_id.value()},
      {"stage_id", stage.stage_id.value()},
      {"arrived_at_s", stage.arrived_at.seconds()},
      {"started_at_s", stage.started_at.seconds()},
      {"completed_at_s", stage.completed_at.seconds()},
      {"dense_compute_ms", execution.dense_compute_ms},
      {"tp_communication_ms", execution.tp_communication_ms},
      {"pp_communication_ms", execution.pp_communication_ms},
      {"duration_ms", execution.total_ms()},
  });
  if (stage.cluster_type != ClusterType::kMonolithic) {
    json["cluster_type"] = to_string(stage.cluster_type);
  }
  return json;
}

OrderedJson serialize_scheduler_trace(
    const SchedulerTraceRecord& trace,
    int schema_version) {
  require_valid_time(trace.simulation_time, "scheduler simulation_time");
  if (trace.token_budget_after > trace.token_budget_before) {
    throw std::invalid_argument("invalid scheduler trace counters");
  }
  if (trace.batch_request_ids.size() !=
      trace.request_num_tokens.size()) {
    throw std::invalid_argument(
        "scheduler trace batch IDs and tokens are misaligned");
  }

  OrderedJson json = OrderedJson::object();
  json["iteration_id"] = trace.iteration_id.value();
  json["simulation_time_s"] = trace.simulation_time.seconds();
  json["decisions"] = OrderedJson::array();
  for (const SchedulerDecisionRecord& decision : trace.decisions) {
    json["decisions"].push_back(OrderedJson::object({
        {"decision_result", decision.decision_result},
        {"request_id", decision.request_id.value()},
        {"num_tokens", decision.num_tokens},
        {"token_budget_after", decision.token_budget_after},
        {"available_blocks_after", decision.available_blocks_after},
    }));
  }
  json["token_budget_before"] = trace.token_budget_before;
  json["token_budget_after"] = trace.token_budget_after;
  json["available_blocks_before"] = trace.available_blocks_before;
  json["available_blocks_after"] = trace.available_blocks_after;
  json["waiting_count_before"] = trace.waiting_count_before;
  json["waiting_count_after"] = trace.waiting_count_after;
  json["running_count_before"] = trace.running_count_before;
  json["running_count_after"] = trace.running_count_after;
  json["preempted_count"] = trace.preempted_count;
  json["batch_request_ids"] = OrderedJson::array();
  for (const RequestId request_id : trace.batch_request_ids) {
    json["batch_request_ids"].push_back(request_id.value());
  }
  json["request_num_tokens"] = trace.request_num_tokens;
  if (has_parallel_contract(schema_version)) {
    json["replica_id"] = trace.replica_id.value();
    json["dp_id"] = trace.dp_id.value();
    if (schema_version == config::kPddSchemaVersion) {
      json["cluster_type"] = to_string(trace.cluster_type);
    }
  }
  return json;
}

OrderedJson serialize_kv_cache_transfer(
    const KVCacheTransferMetricsRecord& transfer) {
  require_valid_time(transfer.started_at, "transfer.started_at");
  require_valid_time(transfer.completed_at, "transfer.completed_at");
  if (transfer.completed_at.seconds() <
          transfer.started_at.seconds() ||
      transfer.size_bytes == 0 ||
      !std::isfinite(transfer.predicted_time_ms) ||
      transfer.predicted_time_ms < 0.0) {
    throw std::invalid_argument(
        "KV transfer metrics are invalid");
  }
  return OrderedJson::object({
      {"transfer_id", transfer.transfer_id.value()},
      {"request_id", transfer.request_id.value()},
      {"source_batch_id", transfer.source_batch_id.value()},
      {
          "source_cluster_type",
          to_string(transfer.source_cluster_type),
      },
      {
          "target_cluster_type",
          to_string(transfer.target_cluster_type),
      },
      {"source_replica_id", transfer.source_replica_id.value()},
      {"source_dp_id", transfer.source_dp_id.value()},
      {"size_bytes", transfer.size_bytes},
      {"predicted_time_ms", transfer.predicted_time_ms},
      {"started_at_s", transfer.started_at.seconds()},
      {"completed_at_s", transfer.completed_at.seconds()},
  });
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
    case EventType::kSchedulerPoll:
      return "scheduler_poll";
    case EventType::kBatchCompletion:
      return "batch_completion";
    case EventType::kGlobalSchedule:
      return "global_schedule";
    case EventType::kClusterSchedule:
      return "cluster_schedule";
    case EventType::kReplicaSchedule:
      return "replica_schedule";
    case EventType::kBatchStageArrival:
      return "batch_stage_arrival";
    case EventType::kReplicaStageSchedule:
      return "replica_stage_schedule";
    case EventType::kBatchStageEnd:
      return "batch_stage_end";
    case EventType::kClusterBatchEnd:
      return "cluster_batch_end";
    case EventType::kGlobalBatchEnd:
      return "global_batch_end";
    case EventType::kKvCacheTransferStart:
      return "kv_cache_transfer_start";
    case EventType::kKvCacheTransferEnd:
      return "kv_cache_transfer_end";
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
  if (output.schema_version != config::kFoundationSchemaVersion &&
      output.schema_version != config::kSchedulerSchemaVersion &&
      output.schema_version != config::kParallelSchemaVersion &&
      output.schema_version != config::kPddSchemaVersion) {
    throw std::invalid_argument(
        "output schema_version must be 1, 2, 3, or 4");
  }
  if (output.run.run_id.empty()) {
    throw std::invalid_argument("output run_id must not be empty");
  }
  validate_request_metrics(output.requests, output.schema_version);
  if (output.schema_version == config::kFoundationSchemaVersion &&
      (!output.batches.empty() || !output.scheduler_trace.empty())) {
    throw std::invalid_argument(
        "output schema v1 cannot contain scheduler records");
  }

  OrderedJson root = OrderedJson::object();
  root["schema_version"] = output.schema_version;
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
    root["requests"].push_back(
        serialize_request(request, output.schema_version));
  }

  if (has_scheduler_contract(output.schema_version)) {
    root["batches"] = OrderedJson::array();
    for (const BatchMetricsRecord& batch : output.batches) {
      root["batches"].push_back(
          serialize_batch(batch, output.schema_version));
    }
    root["scheduler_trace"] = OrderedJson::array();
    for (const SchedulerTraceRecord& trace : output.scheduler_trace) {
      root["scheduler_trace"].push_back(
          serialize_scheduler_trace(
              trace, output.schema_version));
    }
  }
  if (has_parallel_contract(output.schema_version)) {
    root["batch_stages"] = OrderedJson::array();
    for (const BatchStageMetricsRecord& stage :
         output.batch_stages) {
      root["batch_stages"].push_back(
          serialize_batch_stage(stage));
    }
  }

  if (output.schema_version == config::kPddSchemaVersion) {
    root["kv_cache_transfers"] = OrderedJson::array();
    for (const KVCacheTransferMetricsRecord& transfer :
         output.kv_cache_transfers) {
      root["kv_cache_transfers"].push_back(
          serialize_kv_cache_transfer(transfer));
    }
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
    const std::vector<RequestMetricsRecord>& requests,
    int schema_version) {
  validate_request_metrics(requests, schema_version);

  std::ostringstream output;
  output.imbue(std::locale::classic());
  output << std::setprecision(std::numeric_limits<double>::max_digits10);
  if (schema_version == config::kFoundationSchemaVersion) {
    output
        << "request_id,arrived_at_s,prefill_completed_at_s,completed_at_s,"
           "ttft_ms,e2e_ms\n";
  } else if (schema_version == config::kSchedulerSchemaVersion) {
    output
        << "request_id,arrived_at_s,first_scheduled_at_s,"
           "prefill_completed_at_s,first_token_completed_at_s,"
           "completed_at_s,scheduling_delay_ms,ttft_ms,e2e_ms,"
           "num_processed_tokens,preemption_count\n";
  } else if (schema_version == config::kParallelSchemaVersion) {
    output
        << "request_id,arrived_at_s,first_scheduled_at_s,"
           "prefill_completed_at_s,first_token_completed_at_s,"
           "completed_at_s,scheduling_delay_ms,ttft_ms,e2e_ms,"
           "num_processed_tokens,preemption_count,replica_id,dp_id\n";
  } else if (schema_version == config::kPddSchemaVersion) {
    output
        << "request_id,arrived_at_s,first_scheduled_at_s,"
           "prefill_completed_at_s,first_token_completed_at_s,"
           "completed_at_s,scheduling_delay_ms,ttft_ms,e2e_ms,"
           "num_processed_tokens,preemption_count,"
           "prefill_replica_id,prefill_dp_id,decode_replica_id,decode_dp_id,"
           "transfer_id,kv_cache_transfer_start_time_s,"
           "kv_cache_transfer_end_time_s,kv_cache_transfer_time_ms,"
           "kv_cache_transfer_size_bytes,decode_arrived_at_s\n";
  } else {
    throw std::invalid_argument(
        "request CSV schema_version must be 1, 2, 3, or 4");
  }

  for (const RequestMetricsRecord& request : requests) {
    output << request.request_id.value() << ','
           << request.arrived_at.seconds() << ',';
    if (has_scheduler_contract(schema_version)) {
      output << request.first_scheduled_at->seconds() << ',';
    }
    output << request.prefill_completed_at.seconds() << ',';
    if (has_scheduler_contract(schema_version)) {
      output << request.first_token_completed_at->seconds() << ',';
    }
    output << request.completed_at.seconds() << ',';
    if (has_scheduler_contract(schema_version)) {
      output << milliseconds_between(
                    request.first_scheduled_at.value(),
                    request.arrived_at,
                    "first_scheduled_at")
             << ',';
    }
    output << milliseconds_between(
                  request.prefill_completed_at,
                  request.arrived_at,
                  "prefill_completed_at")
           << ','
           << milliseconds_between(
                  request.completed_at,
                  request.arrived_at,
                  "completed_at");
    if (has_scheduler_contract(schema_version)) {
      output << ',' << request.num_processed_tokens << ','
             << request.preemption_count;
      if (schema_version == config::kParallelSchemaVersion) {
        output << ',' << request.replica_id.value() << ','
               << request.dp_id.value();
      } else if (schema_version == config::kPddSchemaVersion) {
        output << ',' << request.prefill_replica_id->value()
               << ',' << request.prefill_dp_id->value()
               << ',' << request.decode_replica_id->value()
               << ',' << request.decode_dp_id->value()
               << ',' << request.transfer_id->value()
               << ','
               << request.kv_cache_transfer_start_time->seconds()
               << ','
               << request.kv_cache_transfer_end_time->seconds()
               << ','
               << (
                      request.kv_cache_transfer_end_time->seconds() -
                      request.kv_cache_transfer_start_time->seconds()) *
                      1e3
               << ',' << request.kv_cache_transfer_size_bytes
               << ',' << request.decode_arrived_at->seconds();
      }
    }
    output << '\n';
  }
  return output.str();
}

}  // namespace frontier::metrics
