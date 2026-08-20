#include "frontier/scheduler/cluster_scheduler/base_cluster_scheduler.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <utility>

#include "frontier/entities/batch.h"
#include "frontier/entities/cluster.h"
#include "frontier/entities/request.h"
#include "frontier/execution_time_predictor/base_execution_time_predictor.h"
#include "frontier/simulator/simulator.h"

namespace frontier::scheduler {

bool BaseClusterScheduler::requires_moe_synchronization(
    const entities::Batch &batch, const simulator::Simulator &simulator) const {
    if (batch.cluster_type() != cluster_type()) {
        throw ClusterSchedulerError(
            "batch was sent to the wrong cluster scheduler");
    }
    const config::ClusterRuntimeConfig &runtime =
        simulator.runtime_config(cluster_type());
    if (!batch.is_moe() || !runtime.model.is_moe()) {
        return false;
    }
    return runtime.parallelism.data_parallel_size > 1 ||
           runtime.parallelism.moe_expert_parallel_size > 1;
}

BaseClusterScheduler::MoEDomainKey
BaseClusterScheduler::make_moe_domain_key(ReplicaId replica_id,
                                          StageId stage_id) const noexcept {
    return MoEDomainKey{cluster_type(), replica_id, stage_id};
}

bool BaseClusterScheduler::can_schedule_moe_stage(const entities::Batch &batch,
                                                  StageId stage_id,
                                                  SimTime time) {
    if (!time.valid()) {
        throw ClusterSchedulerError(
            "MoE stage scheduling time must be finite and nonnegative");
    }
    const auto position = moe_domain_reservations_.find(
        make_moe_domain_key(batch.replica_id(), stage_id));
    if (position == moe_domain_reservations_.end()) {
        return true;
    }
    const MoEDomainReservation &reservation = position->second;
    if (reservation.release_at.valid() && time >= reservation.release_at) {
        moe_domain_reservations_.erase(position);
        return true;
    }
    // Before the first pre-MoE arrival the group is still collecting real
    // stage registrations.  A lane that has not registered yet may join that
    // group even though the shared domain is already reserved.
    const auto group = moe_group_states_.find(reservation.group_key);
    if (group != moe_group_states_.end() &&
        !group->second.participants_initialized && batch.is_moe()) {
        const MoEParticipantId participant{batch.dp_id().value()};
        if (group->second.participants.find(participant) ==
            group->second.participants.end()) {
            return true;
        }
    }
    return false;
}

void BaseClusterScheduler::release_moe_domain_if_ready(ReplicaId replica_id,
                                                       StageId stage_id,
                                                       SimTime time) {
    if (!time.valid()) {
        throw ClusterSchedulerError(
            "MoE domain release time must be finite and nonnegative");
    }
    const auto position = moe_domain_reservations_.find(
        make_moe_domain_key(replica_id, stage_id));
    if (position != moe_domain_reservations_.end() &&
        position->second.release_at.valid() &&
        time >= position->second.release_at) {
        moe_domain_reservations_.erase(position);
    }
}

BaseClusterScheduler::MoEGroupKey
BaseClusterScheduler::make_moe_group_key(const MoEBarrierKey &key,
                                         MoESyncPath path) const noexcept {
    return [&]() {
        MoEGroupKey value{};
        value.cluster_type = key.cluster_type;
        value.replica_id = key.replica_id;
        value.stage_id = key.stage_id;
        value.sync_group_id = key.sync_group_id;
        value.sync_generation = key.generation;
        value.path = path;
        return value;
    }();
}

BaseClusterScheduler::MoEStageState &
BaseClusterScheduler::moe_stage_state(BatchId batch_id, StageId stage_id) {
    const auto position = moe_stage_states_.find([&]() {
        MoEStageKey value{};
        value.batch_id = batch_id;
        value.stage_id = stage_id;
        return value;
    }());
    if (position == moe_stage_states_.end()) {
        throw std::logic_error(
            "MoE synchronization references unknown stage state");
    }
    return position->second;
}

void BaseClusterScheduler::enqueue_moe_arrival(
    const MoEStageState &state, MoEParticipantId participant_id,
    LayerId layer_id, MoESyncPhase phase, SimTime time,
    double elapsed_component_ms, simulator::Simulator &simulator) {
    if (state.path == MoESyncPath::kPrefill) {
        simulator.event_queue().push(time, [&]() {
            PrefillSyncPayload value{};
            value.batch_id = state.batch_id;
            value.replica_id = state.replica_id;
            value.dp_id = state.dp_id;
            value.participant_id = participant_id;
            value.stage_id = state.stage_id;
            value.sync_group_id = state.sync_group_id;
            value.layer_id = layer_id;
            value.sync_phase = phase;
            value.elapsed_component_ms = elapsed_component_ms;
            value.is_idle = false;
            value.generation = state.batch_generation;
            value.sync_generation = state.sync_generation;
            value.cluster_type = state.cluster_type;
            return value;
        }());
    } else {
        simulator.event_queue().push(time, [&]() {
            DecodeSyncPayload value{};
            value.batch_id = state.batch_id;
            value.replica_id = state.replica_id;
            value.dp_id = state.dp_id;
            value.participant_id = participant_id;
            value.stage_id = state.stage_id;
            value.sync_group_id = state.sync_group_id;
            value.layer_id = layer_id;
            value.sync_phase = phase;
            value.elapsed_component_ms = elapsed_component_ms;
            value.is_idle = false;
            value.generation = state.batch_generation;
            value.sync_generation = state.sync_generation;
            value.cluster_type = state.cluster_type;
            return value;
        }());
    }
}

void BaseClusterScheduler::enqueue_idle_moe_arrival(
    const MoEGroupKey &group_key, BatchId idle_batch_id,
    MoEParticipantId participant_id, DataParallelId dp_id, LayerId layer_id,
    MoESyncPhase phase, SimTime time, double elapsed_component_ms,
    simulator::Simulator &simulator) {
    const entities::Batch &idle = simulator.batch(idle_batch_id);
    if (group_key.path == MoESyncPath::kPrefill) {
        simulator.event_queue().push(time, [&]() {
            PrefillSyncPayload value{};
            value.batch_id = idle_batch_id;
            value.replica_id = group_key.replica_id;
            value.dp_id = dp_id;
            value.participant_id = participant_id;
            value.stage_id = group_key.stage_id;
            value.sync_group_id = group_key.sync_group_id;
            value.layer_id = layer_id;
            value.sync_phase = phase;
            value.elapsed_component_ms = elapsed_component_ms;
            value.is_idle = true;
            value.generation = idle.schedule_epoch();
            value.sync_generation = group_key.sync_generation;
            value.cluster_type = group_key.cluster_type;
            return value;
        }());
    } else {
        simulator.event_queue().push(time, [&]() {
            DecodeSyncPayload value{};
            value.batch_id = idle_batch_id;
            value.replica_id = group_key.replica_id;
            value.dp_id = dp_id;
            value.participant_id = participant_id;
            value.stage_id = group_key.stage_id;
            value.sync_group_id = group_key.sync_group_id;
            value.layer_id = layer_id;
            value.sync_phase = phase;
            value.elapsed_component_ms = elapsed_component_ms;
            value.is_idle = true;
            value.generation = idle.schedule_epoch();
            value.sync_generation = group_key.sync_generation;
            value.cluster_type = group_key.cluster_type;
            return value;
        }());
    }
}

} // namespace frontier::scheduler
