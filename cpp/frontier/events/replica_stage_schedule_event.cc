#include "frontier/events/event_handlers.h"

#include <cmath>
#include <stdexcept>
#include <string>

#include "frontier/simulator/simulator.h"

namespace frontier::events {

void handle_event(const ReplicaStageSchedulePayload &payload, SimTime time,
                  simulator::Simulator &simulator) {
    scheduler::ReplicaStageScheduler &stage =
        simulator.cluster(payload.cluster_type)
            .get_replica_scheduler(payload.replica_id, payload.dp_id)
            .get_replica_stage_scheduler(payload.stage_id);
    const auto ticket = stage.pop_batch_if_not_busy();
    if (!ticket.has_value()) {
        return;
    }
    entities::Batch &batch = simulator.batch(ticket->batch_id);
    if (batch.schedule_epoch() != ticket->schedule_epoch) {
        stage.on_stage_end(ticket->batch_id);
        if (!stage.empty()) {
            simulator.event_queue().push(time, payload);
        }
        return;
    }
    const auto prediction = stage.predict(batch, simulator.requests());
    entities::BatchStage &batch_stage = simulator.create_batch_stage(
        batch.id(), payload.stage_id, time, prediction);
    if (simulator.execution_model(payload.cluster_type).type ==
        config::ExecutionModelType::kAnalytical) {
        simulator.metrics().record_analytical_diagnostic(
            "batch_" + std::to_string(batch.id().value()) + "_stage_" +
                std::to_string(payload.stage_id.value()),
            prediction.diagnostics);
    }
    scheduler::BaseClusterScheduler &cluster_scheduler =
        simulator.cluster(payload.cluster_type);
    const bool requires_sync =
        cluster_scheduler.requires_moe_synchronization(batch, simulator) &&
        !prediction.moe_routing.empty();
    if (requires_sync) {
        cluster_scheduler.begin_moe_stage(batch, payload.stage_id, time,
                                          prediction, simulator);
    }
    const config::ClusterRuntimeConfig &runtime =
        simulator.runtime_config(payload.cluster_type);
    for (const auto &diagnostic : prediction.moe_routing) {
        simulator.metrics().record_moe_routing(batch, payload.stage_id,
                                               diagnostic, runtime);
    }
    if (requires_sync) {
        return;
    }
    const double completion_seconds =
        time.seconds() + batch_stage.execution_time().total_ms() * 1e-3;
    if (!std::isfinite(completion_seconds)) {
        throw std::runtime_error("batch stage completion time is nonfinite");
    }
    simulator.event_queue().push(SimTime::from_seconds(completion_seconds),
                                 [&]() {
                                     BatchStageEndPayload value{};
                                     value.batch_id = batch.id();
                                     value.replica_id = payload.replica_id;
                                     value.dp_id = payload.dp_id;
                                     value.stage_id = payload.stage_id;
                                     value.generation = batch.schedule_epoch();
                                     value.cluster_type = payload.cluster_type;
                                     return value;
                                 }());
}

} // namespace frontier::events
