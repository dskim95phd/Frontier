#include "frontier/events/event_handlers.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "frontier/simulator/simulator.h"

namespace frontier::events {

void handle_event(const ReplicaStageSchedulePayload &payload, SimTime time,
                  simulator::Simulator &simulator) {
    const config::ClusterRuntimeConfig &runtime =
        simulator.runtime_config(payload.cluster_type);
    scheduler::ReplicaStageScheduler &stage =
        simulator.cluster(payload.cluster_type)
            .get_replica_scheduler(payload.replica_id, payload.dp_id)
            .get_replica_stage_scheduler(payload.stage_id);
    scheduler::BaseClusterScheduler &cluster_scheduler =
        simulator.cluster(payload.cluster_type);
    if (runtime.scheduler.pipeline_event_mode == "collapsed" &&
        payload.stage_id.value() == 0) {
        // The stage-zero wake emitted by collapsed mode releases the physical
        // stage reservation lazily.  No wake is needed for later stages: all
        // of their intervals were installed in the same calendar turn.
        stage.release_collapsed_stage_if_due(time);
    }
    cluster_scheduler.release_moe_domain_if_ready(payload.replica_id,
                                                  payload.stage_id, time);
    // An aligned DECODE MoE group reserves the physical replica/stage domain
    // across all DP lanes.  Inspect the queued batch before popping it so a
    // later group remains queued instead of becoming an overlapping active
    // stage.  The open aligned group still admits lanes that have not yet
    // registered, and the reservation emits a wake schedule event when it is
    // released.
    if (const auto pending = stage.peek_batch_if_not_busy();
        pending.has_value() &&
        !cluster_scheduler.can_schedule_moe_stage(
            simulator.batch(pending->batch_id), payload.stage_id, time)) {
        return;
    }
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
    const config::PipelineStageLayerRange stage_layers =
        config::pipeline_stage_layer_range(runtime.model.num_layers,
                                           runtime.parallelism,
                                           payload.stage_id.index());
    bool stage_has_moe = false;
    if (runtime.model.is_moe()) {
        for (std::uint64_t model_layer = stage_layers.begin;
             model_layer < stage_layers.end; ++model_layer) {
            stage_has_moe =
                stage_has_moe || runtime.model.is_moe_layer(model_layer);
        }
    }
    const bool requires_sync =
        stage_has_moe &&
        cluster_scheduler.requires_moe_synchronization(batch, simulator);

    // A stage-zero dense prefix must not hide a synchronized MoE stage later
    // in the same pipeline.  Collapse is therefore selected only after
    // scanning every physical stage for a synchronization-capable MoE range.
    bool pipeline_requires_sync = requires_sync;
    if (runtime.scheduler.pipeline_event_mode == "collapsed" &&
        runtime.parallelism.pipeline_parallel_size > 1 &&
        payload.stage_id.value() == 0 && !pipeline_requires_sync) {
        for (std::uint64_t stage_index = 0;
             stage_index < runtime.parallelism.pipeline_parallel_size;
             ++stage_index) {
            const config::PipelineStageLayerRange candidate_layers =
                config::pipeline_stage_layer_range(
                    runtime.model.num_layers, runtime.parallelism, stage_index);
            bool candidate_has_moe = false;
            if (runtime.model.is_moe()) {
                for (std::uint64_t model_layer = candidate_layers.begin;
                     model_layer < candidate_layers.end; ++model_layer) {
                    candidate_has_moe = candidate_has_moe ||
                                        runtime.model.is_moe_layer(model_layer);
                }
            }
            if (candidate_has_moe &&
                cluster_scheduler.requires_moe_synchronization(batch,
                                                               simulator)) {
                pipeline_requires_sync = true;
                break;
            }
        }
    }
    const bool collapsed_mode =
        runtime.scheduler.pipeline_event_mode == "collapsed" &&
        runtime.parallelism.pipeline_parallel_size > 1 &&
        payload.stage_id.value() == 0 && !pipeline_requires_sync;
    if (collapsed_mode) {
        struct ReservedStage {
            StageId stage_id;
            SimTime started_at;
            SimTime completed_at;
            execution_time_predictor::ExecutionTimePrediction prediction;
        };
        std::vector<ReservedStage> reservations;
        reservations.reserve(static_cast<std::size_t>(
            runtime.parallelism.pipeline_parallel_size));
        SimTime arrival = time;
        for (std::uint64_t stage_index = 0;
             stage_index < runtime.parallelism.pipeline_parallel_size;
             ++stage_index) {
            const StageId stage_id{stage_index};
            if (stage_index != 0) {
                simulator.record_stage_arrival(batch.id(), stage_id, arrival);
            }
            scheduler::ReplicaStageScheduler &calendar_stage =
                simulator.cluster(payload.cluster_type)
                    .get_replica_scheduler(payload.replica_id, payload.dp_id)
                    .get_replica_stage_scheduler(stage_id);
            const auto prediction =
                calendar_stage.predict_collapsed(batch, simulator.requests());
            const double duration_ms = prediction.execution_time.total_ms();
            if (!std::isfinite(duration_ms) || duration_ms < 0.0) {
                throw std::runtime_error(
                    "collapsed batch stage duration is invalid");
            }
            const SimTime started_at =
                calendar_stage.collapsed_start_time(arrival, duration_ms);
            const double completed_seconds =
                started_at.seconds() + duration_ms * 1e-3;
            if (!std::isfinite(completed_seconds)) {
                throw std::runtime_error(
                    "collapsed batch pipeline completion is nonfinite");
            }
            const SimTime completed_at =
                SimTime::from_seconds(completed_seconds);
            calendar_stage.reserve_collapsed_interval(batch.id(), started_at,
                                                      completed_at);
            reservations.push_back(
                ReservedStage{stage_id, started_at, completed_at, prediction});
            arrival = completed_at;
        }

        // Materialize all stage records at reservation time.  Their metrics
        // are emitted by BatchPipelineEnd once every physical interval has
        // elapsed, while the stage timestamps remain exact-mode-equivalent.
        for (const ReservedStage &reserved : reservations) {
            const auto &prediction = reserved.prediction;
            simulator.create_batch_stage(batch.id(), reserved.stage_id,
                                         reserved.started_at, prediction);
            if (simulator.execution_model(payload.cluster_type).type ==
                config::ExecutionModelType::kAnalytical) {
                simulator.metrics().record_analytical_diagnostic(
                    "batch_" + std::to_string(batch.id().value()) + "_stage_" +
                        std::to_string(reserved.stage_id.value()),
                    prediction.diagnostics);
            }
            for (const auto &diagnostic : prediction.moe_routing) {
                simulator.metrics().record_moe_routing(batch, reserved.stage_id,
                                                       diagnostic, runtime);
            }
        }

        // Stage zero must be woken at its own boundary so the reservation
        // does not serialize unrelated batches through the whole pipeline.
        simulator.event_queue().push(
            reservations.front().completed_at,
            ReplicaStageSchedulePayload{payload.replica_id, payload.dp_id,
                                        StageId{0}, payload.cluster_type});
        simulator.event_queue().push(
            reservations.back().completed_at,
            BatchPipelineEndPayload{batch.id(), payload.replica_id,
                                    payload.dp_id, batch.schedule_epoch(),
                                    payload.cluster_type});
        return;
    }

    const bool lazy_moe = requires_sync && stage.supports_lazy_moe_prediction();
    const auto prediction =
        lazy_moe ? stage.prepare_moe_stage(batch, simulator.requests())
                 : stage.predict(batch, simulator.requests());
    entities::BatchStage &batch_stage = simulator.create_batch_stage(
        batch.id(), payload.stage_id, time, prediction);
    if (simulator.execution_model(payload.cluster_type).type ==
        config::ExecutionModelType::kAnalytical) {
        simulator.metrics().record_analytical_diagnostic(
            "batch_" + std::to_string(batch.id().value()) + "_stage_" +
                std::to_string(payload.stage_id.value()),
            prediction.diagnostics);
    }
    if (requires_sync && prediction.moe_routing.empty()) {
        throw std::logic_error(
            "synchronized MoE stage preparation returned no layer");
    }
    if (requires_sync) {
        cluster_scheduler.begin_moe_stage(batch, payload.stage_id, time,
                                          prediction, simulator);
    }
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
