#include "frontier/events/event_handlers.h"

#include <cmath>
#include <stdexcept>
#include <string>

#include "frontier/simulator/simulation_context.h"

namespace frontier::events {

void handle_event(
    const ReplicaStageSchedulePayload& payload,
    SimTime time,
    simulator::SimulationContext& context) {
  scheduler::ReplicaStageScheduler& stage =
      context.cluster(payload.cluster_type)
          .get_replica_scheduler(
              payload.replica_id, payload.dp_id)
          .get_replica_stage_scheduler(payload.stage_id);
  const auto ticket = stage.pop_batch_if_not_busy();
  if (!ticket.has_value()) {
    return;
  }
  entities::Batch& batch = context.batch(ticket->batch_id);
  if (batch.schedule_epoch() != ticket->schedule_epoch) {
    stage.on_stage_end(ticket->batch_id);
    if (!stage.empty()) {
      context.event_queue().push(time, payload);
    }
    return;
  }
  const auto prediction =
      stage.predict(batch, context.requests());
  entities::BatchStage& batch_stage =
      context.create_batch_stage(
          batch.id(), payload.stage_id, time, prediction);
  if (context.execution_model(payload.cluster_type).type ==
      config::ExecutionModelType::kAnalytical) {
    context.output().analytical_diagnostics.push_back(
        metrics::AnalyticalDiagnostic{
            .name =
                "batch_" +
                std::to_string(batch.id().value()) +
                "_stage_" +
                std::to_string(payload.stage_id.value()),
            .values = prediction.diagnostics,
        });
  }
  const double completion_seconds =
      time.seconds() +
      batch_stage.execution_time().total_ms() * 1e-3;
  if (!std::isfinite(completion_seconds)) {
    throw std::runtime_error(
        "batch stage completion time is nonfinite");
  }
  context.event_queue().push(
      SimTime::from_seconds(completion_seconds),
      BatchStageEndPayload{
          .batch_id = batch.id(),
          .replica_id = payload.replica_id,
          .dp_id = payload.dp_id,
          .stage_id = payload.stage_id,
          .generation = batch.schedule_epoch(),
          .cluster_type = payload.cluster_type,
      });
}

}  // namespace frontier::events
