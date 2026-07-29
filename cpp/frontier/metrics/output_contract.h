#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "frontier/config/config.h"
#include "frontier/core/event.h"
#include "frontier/core/ids.h"
#include "frontier/entities/execution_time.h"

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
  std::optional<SimTime> first_scheduled_at;
  std::optional<SimTime> first_token_completed_at;
  std::uint64_t num_processed_tokens = 0;
  std::uint64_t preemption_count = 0;
  std::vector<std::uint64_t> tokens_at_preemption;
  ReplicaId replica_id{0};
  DataParallelId dp_id{0};
};

struct BatchMetricsRecord {
  BatchId batch_id;
  IterationId iteration_id;
  SimTime scheduled_at;
  SimTime completed_at;
  std::vector<RequestId> request_ids;
  std::vector<std::uint64_t> scheduled_tokens;
  std::uint64_t total_scheduled_tokens;
  std::uint64_t num_prefill_tokens;
  std::uint64_t num_decode_tokens;
  double predicted_execution_ms;
  ReplicaId replica_id{0};
  DataParallelId dp_id{0};
  std::uint64_t num_pipeline_stages = 1;
};

struct BatchStageMetricsRecord {
  BatchId batch_id;
  ReplicaId replica_id;
  DataParallelId dp_id;
  StageId stage_id;
  SimTime arrived_at;
  SimTime started_at;
  SimTime completed_at;
  entities::ExecutionTime execution_time;
};

struct SchedulerDecisionRecord {
  std::string decision_result;
  RequestId request_id;
  std::uint64_t num_tokens;
  std::uint64_t token_budget_after;
  std::uint64_t available_blocks_after;
};

struct SchedulerTraceRecord {
  IterationId iteration_id;
  SimTime simulation_time;
  std::uint64_t token_budget_before;
  std::uint64_t token_budget_after;
  std::uint64_t available_blocks_before;
  std::uint64_t available_blocks_after;
  std::uint64_t waiting_count_before;
  std::uint64_t waiting_count_after;
  std::uint64_t running_count_before;
  std::uint64_t running_count_after;
  std::uint64_t preempted_count;
  std::vector<SchedulerDecisionRecord> decisions;
  std::vector<RequestId> batch_request_ids;
  std::vector<std::uint64_t> request_num_tokens;
  ReplicaId replica_id{0};
  DataParallelId dp_id{0};
};

struct AnalyticalDiagnostic {
  std::string name;
  std::vector<std::pair<std::string, double>> values;
};

struct SimulationOutput {
  int schema_version = kOutputSchemaVersion;
  RunMetadata run;
  std::vector<RequestMetricsRecord> requests;
  std::vector<BatchMetricsRecord> batches;
  std::vector<BatchStageMetricsRecord> batch_stages;
  std::vector<SchedulerTraceRecord> scheduler_trace;
  std::vector<Event> event_trace;
  std::vector<AnalyticalDiagnostic> analytical_diagnostics;
};

[[nodiscard]] std::string_view to_string(EventType event_type) noexcept;
[[nodiscard]] std::string_view to_string(
    MetricsSemantics semantics) noexcept;
[[nodiscard]] std::string serialize_simulation_output_json(
    const SimulationOutput& output);
[[nodiscard]] std::string serialize_request_metrics_csv(
    const std::vector<RequestMetricsRecord>& requests,
    int schema_version = kOutputSchemaVersion);

}  // namespace frontier::metrics
