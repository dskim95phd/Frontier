#include "frontier/scheduler/cluster_scheduler/base_cluster_scheduler.h"

#include <algorithm>
#include <limits>
#include <set>
#include <utility>

#include "frontier/entities/batch.h"
#include "frontier/entities/cluster.h"
#include "frontier/entities/request.h"
#include "frontier/execution_time_predictor/batch_execution_model.h"
#include "frontier/simulator/simulation_context.h"

namespace frontier::scheduler {

BaseClusterScheduler::BaseClusterScheduler(
    std::vector<std::unique_ptr<BaseReplicaScheduler>>
        replica_schedulers,
    const entities::Cluster& cluster)
    : replica_schedulers_(std::move(replica_schedulers)),
      cluster_(&cluster) {
  const std::uint64_t num_replicas =
      cluster.parallelism().num_replicas;
  const std::uint64_t data_parallel_size =
      cluster.parallelism().data_parallel_size;
  if (num_replicas == 0 || data_parallel_size == 0 ||
      num_replicas >
          std::numeric_limits<std::uint64_t>::max() /
              data_parallel_size ||
      replica_schedulers_.size() !=
          num_replicas * data_parallel_size) {
    throw ClusterSchedulerError(
        "cluster target matrix is invalid");
  }
  for (std::uint64_t replica = 0; replica < num_replicas;
       ++replica) {
    for (std::uint64_t dp = 0; dp < data_parallel_size; ++dp) {
      BaseReplicaScheduler* scheduler =
          replica_schedulers_.at(target_index(
              ReplicaId{replica}, DataParallelId{dp})).get();
      if (scheduler == nullptr ||
          scheduler->replica_id() != ReplicaId{replica} ||
          scheduler->dp_id() != DataParallelId{dp} ||
          &scheduler->replica_entity() !=
              &cluster.replica(ReplicaId{replica})) {
        throw ClusterSchedulerError(
            "cluster scheduler target identity mismatch");
      }
    }
  }
}

void BaseClusterScheduler::add_request(
    RequestId request_id,
    SimTime arrived_at) {
  request_queue_.push_back(QueuedRequest{
      .request_id = request_id,
      .arrived_at = arrived_at,
  });
}

std::vector<std::pair<ReplicaId, DataParallelId>>
BaseClusterScheduler::targets() const {
  std::vector<std::pair<ReplicaId, DataParallelId>> result;
  result.reserve(replica_schedulers_.size());
  for (const auto& scheduler : replica_schedulers_) {
    result.emplace_back(
        scheduler->replica_id(), scheduler->dp_id());
  }
  return result;
}

std::size_t BaseClusterScheduler::target_index(
    ReplicaId replica_id,
    DataParallelId dp_id) const {
  if (!replica_id.valid() || !dp_id.valid() ||
      replica_id.index() >= num_replicas() ||
      dp_id.index() >= data_parallel_size()) {
    throw ClusterSchedulerError(
        "cluster references an unknown replica target");
  }
  return static_cast<std::size_t>(
      replica_id.index() * data_parallel_size() +
      dp_id.index());
}

BaseReplicaScheduler&
BaseClusterScheduler::get_replica_scheduler(
    ReplicaId replica_id,
    DataParallelId dp_id) {
  return *replica_schedulers_.at(
      target_index(replica_id, dp_id));
}

const BaseReplicaScheduler&
BaseClusterScheduler::get_replica_scheduler(
    ReplicaId replica_id,
    DataParallelId dp_id) const {
  return *replica_schedulers_.at(
      target_index(replica_id, dp_id));
}

std::uint64_t BaseClusterScheduler::next_batch_global_id(
    ReplicaId replica_id,
    DataParallelId dp_id) {
  std::uint64_t& next = batch_counters_[
      BatchCounterKey{
          .replica_id = replica_id,
          .dp_id = dp_id,
      }];
  if (next >
      static_cast<std::uint64_t>(
          std::numeric_limits<std::int64_t>::max())) {
    throw std::overflow_error("batch global ID overflows");
  }
  return next++;
}

bool BaseClusterScheduler::requires_moe_synchronization(
    const entities::Batch& batch,
    const simulator::SimulationContext& context) const {
  if (batch.cluster_type() != cluster_type()) {
    throw ClusterSchedulerError(
        "batch was sent to the wrong cluster scheduler");
  }
  const config::ClusterRuntimeConfig& runtime =
      context.runtime_config(cluster_type());
  if (!batch.is_moe() || !runtime.model.is_moe()) {
    return false;
  }
  return runtime.parallelism.data_parallel_size > 1 ||
      runtime.parallelism.moe_expert_parallel_size > 1;
}

BaseClusterScheduler::MoEGroupKey
BaseClusterScheduler::make_moe_group_key(
    const moe::BarrierKey& key,
    moe::SyncPath path) const noexcept {
  return MoEGroupKey{
      .cluster_type = key.cluster_type,
      .replica_id = key.replica_id,
      .stage_id = key.stage_id,
      .sync_group_id = key.sync_group_id,
      .sync_generation = key.generation,
      .path = path,
  };
}

BaseClusterScheduler::MoEStageState&
BaseClusterScheduler::moe_stage_state(
    BatchId batch_id,
    StageId stage_id) {
  const auto position = moe_stage_states_.find(
      MoEStageKey{.batch_id = batch_id, .stage_id = stage_id});
  if (position == moe_stage_states_.end()) {
    throw std::logic_error(
        "MoE synchronization references unknown stage state");
  }
  return position->second;
}

void BaseClusterScheduler::enqueue_moe_arrival(
    const MoEStageState& state,
    MoEParticipantId participant_id,
    LayerId layer_id,
    moe::SyncPhase phase,
    SimTime time,
    double elapsed_component_ms,
    simulator::SimulationContext& context) {
  if (state.path == moe::SyncPath::kPrefill) {
    context.event_queue().push(
        time,
        PrefillSyncPayload{
            .batch_id = state.batch_id,
            .replica_id = state.replica_id,
            .dp_id = state.dp_id,
            .participant_id = participant_id,
            .stage_id = state.stage_id,
            .sync_group_id = state.sync_group_id,
            .layer_id = layer_id,
            .sync_phase = phase,
            .elapsed_component_ms = elapsed_component_ms,
            .is_idle = false,
            .generation = state.batch_generation,
            .sync_generation = state.sync_generation,
            .cluster_type = state.cluster_type,
        });
  } else {
    context.event_queue().push(
        time,
        DecodeSyncPayload{
            .batch_id = state.batch_id,
            .replica_id = state.replica_id,
            .dp_id = state.dp_id,
            .participant_id = participant_id,
            .stage_id = state.stage_id,
            .sync_group_id = state.sync_group_id,
            .layer_id = layer_id,
            .sync_phase = phase,
            .elapsed_component_ms = elapsed_component_ms,
            .is_idle = false,
            .generation = state.batch_generation,
            .sync_generation = state.sync_generation,
            .cluster_type = state.cluster_type,
        });
  }
}

void BaseClusterScheduler::enqueue_idle_moe_arrival(
    const MoEGroupKey& group_key,
    BatchId idle_batch_id,
    MoEParticipantId participant_id,
    DataParallelId dp_id,
    LayerId layer_id,
    moe::SyncPhase phase,
    SimTime time,
    double elapsed_component_ms,
    simulator::SimulationContext& context) {
  const entities::Batch& idle = context.batch(idle_batch_id);
  if (group_key.path == moe::SyncPath::kPrefill) {
    context.event_queue().push(
        time,
        PrefillSyncPayload{
            .batch_id = idle_batch_id,
            .replica_id = group_key.replica_id,
            .dp_id = dp_id,
            .participant_id = participant_id,
            .stage_id = group_key.stage_id,
            .sync_group_id = group_key.sync_group_id,
            .layer_id = layer_id,
            .sync_phase = phase,
            .elapsed_component_ms = elapsed_component_ms,
            .is_idle = true,
            .generation = idle.schedule_epoch(),
            .sync_generation = group_key.sync_generation,
            .cluster_type = group_key.cluster_type,
        });
  } else {
    context.event_queue().push(
        time,
        DecodeSyncPayload{
            .batch_id = idle_batch_id,
            .replica_id = group_key.replica_id,
            .dp_id = dp_id,
            .participant_id = participant_id,
            .stage_id = group_key.stage_id,
            .sync_group_id = group_key.sync_group_id,
            .layer_id = layer_id,
            .sync_phase = phase,
            .elapsed_component_ms = elapsed_component_ms,
            .is_idle = true,
            .generation = idle.schedule_epoch(),
            .sync_generation = group_key.sync_generation,
            .cluster_type = group_key.cluster_type,
        });
  }
}

void BaseClusterScheduler::begin_moe_stage(
    entities::Batch& batch,
    StageId stage_id,
    SimTime started_at,
    const execution_time_predictor::BatchExecutionPrediction&
        prediction,
    simulator::SimulationContext& context) {
  if (!requires_moe_synchronization(batch, context)) {
    throw std::logic_error(
        "local or dense batch cannot enter MoE synchronization");
  }
  const config::ClusterRuntimeConfig& runtime =
      context.runtime_config(cluster_type());
  bool has_prefill = cluster_type() == ClusterType::kPrefill;
  if (cluster_type() == ClusterType::kMonolithic) {
    has_prefill = std::any_of(
        batch.requests().begin(),
        batch.requests().end(),
        [&context](const entities::RequestBatchSnapshot& snapshot) {
          return snapshot.processed_tokens <
              context.request(snapshot.request_id).num_prefill_tokens();
        });
  }
  const moe::SyncPath path = has_prefill
      ? moe::SyncPath::kPrefill
      : moe::SyncPath::kDecode;

  const std::vector<MoEParticipantId> participants{
      MoEParticipantId{batch.dp_id().value()}};
  const bool monolithic_decode =
      cluster_type() == ClusterType::kMonolithic &&
      path == moe::SyncPath::kDecode;
  const std::uint64_t expected = monolithic_decode
      ? runtime.parallelism.moe_expert_parallel_size
      : runtime.parallelism.data_parallel_size;
  const std::uint64_t layers_per_stage =
      runtime.model.num_layers /
      runtime.parallelism.pipeline_parallel_size;
  const entities::ExecutionTime& execution =
      prediction.execution_time;
  const double attention_ms = execution.dense_compute_ms;
  const SimTime initial_pre_arrival = SimTime::from_seconds(
      started_at.seconds() +
      attention_ms /
          static_cast<double>(layers_per_stage) * 1e-3);
  const MoECounterKey counter_key{
      .replica_id = batch.replica_id(),
      .stage_id = stage_id,
      .path = path,
      .lane_id = batch.dp_id(),
  };
  const std::uint64_t participant_domain =
      runtime.parallelism.moe_expert_parallel_size;
  const std::uint64_t maximum_id =
      static_cast<std::uint64_t>(
          std::numeric_limits<std::int64_t>::max());
  const std::uint64_t lane_value =
      static_cast<std::uint64_t>(batch.dp_id().value());
  std::uint64_t group_value = 0;
  if (monolithic_decode) {
    std::uint64_t& ordinal = moe_group_counters_[counter_key];
    if (ordinal >
        (maximum_id - lane_value) / participant_domain) {
      throw std::overflow_error(
          "MoE synchronization group ID overflows");
    }
    group_value = ordinal * participant_domain + lane_value;
    ++ordinal;
  } else {
    group_value =
        static_cast<std::uint64_t>(batch.global_id().value());
  }
  if (group_value > maximum_id ||
      next_moe_sync_generation_ > maximum_id) {
    throw std::overflow_error(
        "MoE synchronization group ID overflows");
  }
  MoEGroupKey selected_group_key{
      .cluster_type = cluster_type(),
      .replica_id = batch.replica_id(),
      .stage_id = stage_id,
      .sync_group_id = MoESyncGroupId{group_value},
      .sync_generation = Generation{next_moe_sync_generation_},
      .path = path,
  };
  bool joined_open_group = false;
  if (!monolithic_decode) {
    for (auto& [candidate_key, candidate] : moe_group_states_) {
      if (candidate_key.replica_id !=
              selected_group_key.replica_id ||
          candidate_key.stage_id != selected_group_key.stage_id ||
          candidate_key.sync_group_id !=
              selected_group_key.sync_group_id ||
          candidate_key.path != selected_group_key.path ||
          candidate.participants_initialized ||
          candidate.expected_participants != expected ||
          candidate.initial_pre_arrival != initial_pre_arrival ||
          candidate.participants.contains(participants.front())) {
        continue;
      }
      selected_group_key = candidate_key;
      joined_open_group = true;
      break;
    }
  }
  if (!joined_open_group) {
    moe_group_states_.emplace(
        selected_group_key,
        MoEGroupState{
            .expected_participants = expected,
            .participants_initialized = false,
            .initial_pre_arrival = initial_pre_arrival,
            .participants = {},
        });
    ++next_moe_sync_generation_;
  }
  const MoESyncGroupId sync_group_id =
      selected_group_key.sync_group_id;
  const Generation sync_generation =
      selected_group_key.sync_generation;
  if (!batch.moe_sync_group_id().valid()) {
    batch.set_moe_synchronization(
        sync_group_id, participants.front());
  }

  batch.reset_stage_layer();
  const double post_attention_ms =
      execution.tp_communication_ms +
      execution.moe_gating_linear_ms +
      execution.moe_gating_routing_topk_ms +
      execution.moe_grouped_gemm_ms +
      execution.moe_shuffling_ms +
      execution.moe_post_attention_norm_ms +
      execution.moe_tp_communication_ms +
      execution.ep_dispatch_ms +
      execution.ep_combine_ms +
      execution.dp_input_communication_ms +
      execution.dp_output_communication_ms;
  std::vector<std::vector<double>> decode_lane_times_ms;
  decode_lane_times_ms.reserve(prediction.moe_routing.size());
  for (const auto& diagnostic : prediction.moe_routing) {
    if (diagnostic.lane_times_ms.size() !=
        runtime.parallelism.moe_expert_parallel_size) {
      throw std::logic_error(
          "MoE prediction lane count does not match EP domain");
    }
    decode_lane_times_ms.push_back(diagnostic.lane_times_ms);
  }
  if (decode_lane_times_ms.size() != layers_per_stage) {
    throw std::logic_error(
        "MoE prediction layer count does not match pipeline stage");
  }
  double critical_lane_sum_ms = 0.0;
  for (const auto& lane_times : decode_lane_times_ms) {
    critical_lane_sum_ms += *std::max_element(
        lane_times.begin(), lane_times.end());
  }
  const double shared_post_attention_ms =
      post_attention_ms - critical_lane_sum_ms;
  if (shared_post_attention_ms < -1e-12) {
    throw std::logic_error(
        "MoE post-attention prediction is smaller than lane work");
  }
  const double shared_post_attention_ms_per_layer =
      std::max(0.0, shared_post_attention_ms) /
      static_cast<double>(layers_per_stage);
  std::vector<double> prefill_post_attention_ms_by_layer;
  prefill_post_attention_ms_by_layer.reserve(
      static_cast<std::size_t>(layers_per_stage));
  for (const auto& lane_times : decode_lane_times_ms) {
    prefill_post_attention_ms_by_layer.push_back(
        shared_post_attention_ms_per_layer +
        *std::max_element(lane_times.begin(), lane_times.end()));
  }

  MoEStageState state{
      .batch_id = batch.id(),
      .replica_id = batch.replica_id(),
      .dp_id = batch.dp_id(),
      .stage_id = stage_id,
      .cluster_type = cluster_type(),
      .sync_group_id = sync_group_id,
      .batch_generation = batch.schedule_epoch(),
      .sync_generation = sync_generation,
      .path = path,
      .participants = participants,
      .layers_per_stage = layers_per_stage,
      .current_layer = 0,
      .attention_ms_per_layer =
          attention_ms / static_cast<double>(layers_per_stage),
      .prefill_post_attention_ms_by_layer =
          std::move(prefill_post_attention_ms_by_layer),
      .decode_ep_communication_ms_per_layer =
          (execution.ep_dispatch_ms + execution.ep_combine_ms) /
          static_cast<double>(layers_per_stage),
      .decode_dp_communication_ms_per_layer =
          (execution.dp_input_communication_ms +
           execution.dp_output_communication_ms) /
          static_cast<double>(layers_per_stage),
      .decode_lane_times_ms = std::move(decode_lane_times_ms),
      .pp_ms = execution.pp_communication_ms,
  };
  const MoEStageKey stage_key{
      .batch_id = batch.id(), .stage_id = stage_id};
  const auto [state_position, inserted] =
      moe_stage_states_.emplace(stage_key, std::move(state));
  if (!inserted) {
    throw std::logic_error("MoE stage state already exists");
  }
  MoEStageState& stored = state_position->second;
  MoEGroupState& group =
      moe_group_states_.at(selected_group_key);
  for (const MoEParticipantId participant : stored.participants) {
    const auto [unused, participant_inserted] =
        group.participants.emplace(participant, stored.batch_id);
    static_cast<void>(unused);
    if (!participant_inserted) {
      throw std::logic_error(
          "two real MoE batches claim one synchronization participant");
    }
    enqueue_moe_arrival(
        stored,
        participant,
        LayerId{0},
        moe::SyncPhase::kPreMoe,
        SimTime::from_seconds(
            started_at.seconds() +
            stored.attention_ms_per_layer * 1e-3),
        stored.attention_ms_per_layer,
        context);
  }
}

void BaseClusterScheduler::ensure_moe_group_participants(
    const moe::BarrierKey& key,
    moe::SyncPath path,
    MoEParticipantId arriving_participant,
    SimTime time,
    simulator::SimulationContext& context) {
  const MoEGroupKey group_key = make_moe_group_key(key, path);
  MoEGroupState& group = moe_group_states_.at(group_key);
  if (group.participants_initialized) {
    return;
  }
  group.participants_initialized = true;
  const config::ClusterRuntimeConfig& runtime =
      context.runtime_config(cluster_type());
  const bool compact_decode =
      cluster_type() == ClusterType::kMonolithic &&
      path == moe::SyncPath::kDecode;
  for (std::uint64_t value = 0;
       value < group.expected_participants;
       ++value) {
    const MoEParticipantId participant{value};
    if (participant == arriving_participant) {
      continue;
    }
    const DataParallelId dp_id{value};
    const BatchId idle_batch_id =
        context.create_moe_idle_batch(
            group_key.sync_group_id,
            participant,
            ReplicaTarget{
                .replica_id = group_key.replica_id,
                .dp_id = dp_id,
            },
            cluster_type(),
            runtime.parallelism.pipeline_parallel_size,
            time,
            group_key.sync_generation);
    const auto owner = group.participants.find(participant);
    if (owner == group.participants.end()) {
      group.participants.emplace(participant, idle_batch_id);
    }
    if (!compact_decode) {
      enqueue_idle_moe_arrival(
          group_key,
          idle_batch_id,
          participant,
          dp_id,
          key.layer_id,
          key.phase,
          time,
          0.0,
          context);
    }
  }
}

std::optional<moe::BarrierReady>
BaseClusterScheduler::compact_moe_group_participants(
    const moe::BarrierKey& key,
    moe::SyncPath path,
    SimTime time,
    simulator::SimulationContext& context) {
  const MoEGroupKey group_key = make_moe_group_key(key, path);
  const MoEGroupState& group = moe_group_states_.at(group_key);
  if (cluster_type() != ClusterType::kMonolithic ||
      path != moe::SyncPath::kDecode) {
    return std::nullopt;
  }
  std::optional<moe::BarrierReady> ready;
  for (const auto& [participant, batch_id] : group.participants) {
    const entities::Batch& owner = context.batch(batch_id);
    if (!owner.is_idle()) {
      continue;
    }
    const auto result = moe_barrier_.arrive(
        key,
        moe::BarrierParticipant{
            .participant_id = participant,
            .batch_id = batch_id,
            .arrival_time = time,
            .elapsed_component_ms = 0.0,
            .is_idle = true,
        },
        group.expected_participants);
    if (result.has_value()) {
      ready = result;
    }
  }
  return ready;
}

void BaseClusterScheduler::continue_moe_stage(
    const moe::BarrierKey& key,
    moe::SyncPath path,
    const std::vector<moe::BarrierParticipant>& participants,
    SimTime time,
    simulator::SimulationContext& context) {
  if (participants.empty()) {
    return;
  }
  const MoEGroupKey group_key = make_moe_group_key(key, path);
  const auto group_position = moe_group_states_.find(group_key);
  if (group_position == moe_group_states_.end()) {
    return;
  }
  MoEGroupState& group = group_position->second;
  const bool monolithic_decode =
      cluster_type() == ClusterType::kMonolithic &&
      path == moe::SyncPath::kDecode;

  if (key.phase == moe::SyncPhase::kPreMoe) {
    const MoEStageState* sample_state = nullptr;
    for (const auto& [unused, batch_id] : group.participants) {
      static_cast<void>(unused);
      if (!context.batch(batch_id).is_idle()) {
        sample_state =
            &moe_stage_state(batch_id, key.stage_id);
        break;
      }
    }
    if (sample_state == nullptr) {
      throw std::logic_error(
          "MoE collective has no real stage participant");
    }
    const std::size_t layer_index =
        static_cast<std::size_t>(sample_state->current_layer);
    if (layer_index >=
        sample_state->decode_lane_times_ms.size()) {
      throw std::logic_error(
          "MoE collective layer index is out of range");
    }
    const double critical_lane_ms = *std::max_element(
        sample_state->decode_lane_times_ms.at(layer_index).begin(),
        sample_state->decode_lane_times_ms.at(layer_index).end());
    for (const bool idle_pass : {false, true}) {
      for (const auto& [participant, batch_id] : group.participants) {
        const entities::Batch& owner = context.batch(batch_id);
        if (owner.is_idle() != idle_pass) {
          continue;
        }
        double component_ms = 0.0;
        if (path == moe::SyncPath::kPrefill) {
          component_ms = sample_state
              ->prefill_post_attention_ms_by_layer.at(layer_index);
        } else if (monolithic_decode) {
          if (participant.index() >=
              sample_state->decode_lane_times_ms.at(
                  layer_index).size()) {
            throw std::logic_error(
                "decode participant is outside the EP lane domain");
          }
          component_ms =
              sample_state->decode_lane_times_ms.at(layer_index).at(
                  participant.index());
        } else {
          component_ms =
              critical_lane_ms +
              sample_state->decode_dp_communication_ms_per_layer;
        }
        const SimTime arrival = SimTime::from_seconds(
            time.seconds() + component_ms * 1e-3);
        if (owner.is_idle()) {
          enqueue_idle_moe_arrival(
              group_key,
              batch_id,
              participant,
              owner.dp_id(),
              key.layer_id,
              moe::SyncPhase::kPostMoe,
              arrival,
              component_ms,
              context);
        } else {
          const MoEStageState& state =
              moe_stage_state(batch_id, key.stage_id);
          enqueue_moe_arrival(
              state,
              participant,
              key.layer_id,
              moe::SyncPhase::kPostMoe,
              arrival,
              component_ms,
              context);
        }
      }
    }
    return;
  }

  std::set<MoEStageKey> real_states;
  for (const auto& [unused, batch_id] : group.participants) {
    static_cast<void>(unused);
    if (!context.batch(batch_id).is_idle()) {
      real_states.emplace(
          MoEStageKey{
              .batch_id = batch_id,
              .stage_id = key.stage_id,
          });
    }
  }
  bool has_next_layer = false;
  for (const MoEStageKey& stage_key : real_states) {
    MoEStageState& state = moe_stage_states_.at(stage_key);
    ++state.current_layer;
    has_next_layer =
        has_next_layer ||
        state.current_layer < state.layers_per_stage;
    if (state.current_layer < state.layers_per_stage) {
      context.batch(state.batch_id).set_stage_layer(
          LayerId{state.current_layer});
    }
  }

  if (has_next_layer) {
    const LayerId next_layer{key.layer_id.value() + 1};
    for (auto position = group.participants.begin();
         position != group.participants.end();) {
      if (context.batch(position->second).is_idle()) {
        position = group.participants.erase(position);
      } else {
        ++position;
      }
    }
    group.participants_initialized = false;
    for (const auto& [participant, batch_id] : group.participants) {
      const MoEStageState& state =
          moe_stage_state(batch_id, key.stage_id);
      const auto refreshed_prediction =
          get_replica_scheduler(state.replica_id, state.dp_id)
              .get_replica_stage_scheduler(state.stage_id)
              .predict(context.batch(batch_id), context.requests());
      const double refreshed_attention_ms_per_layer =
          refreshed_prediction.execution_time.dense_compute_ms /
          static_cast<double>(state.layers_per_stage);
      const double transition_ms =
          refreshed_attention_ms_per_layer +
          (path == moe::SyncPath::kDecode
              ? state.decode_ep_communication_ms_per_layer
              : 0.0);
      enqueue_moe_arrival(
          state,
          participant,
          next_layer,
          moe::SyncPhase::kPreMoe,
          SimTime::from_seconds(
              time.seconds() + transition_ms * 1e-3),
          transition_ms,
          context);
    }
    return;
  }

  for (const MoEStageKey& stage_key : real_states) {
    const MoEStageState& state = moe_stage_states_.at(stage_key);
    const double final_transition_ms =
        state.pp_ms +
        (path == moe::SyncPath::kDecode
            ? state.decode_ep_communication_ms_per_layer
            : 0.0);
    context.event_queue().push(
        SimTime::from_seconds(
            time.seconds() + final_transition_ms * 1e-3),
        BatchStageEndPayload{
            .batch_id = state.batch_id,
            .replica_id = state.replica_id,
            .dp_id = state.dp_id,
            .stage_id = state.stage_id,
            .generation = state.batch_generation,
            .cluster_type = state.cluster_type,
        });
  }
  for (const MoEStageKey& stage_key : real_states) {
    moe_stage_states_.erase(stage_key);
  }
  moe_group_states_.erase(group_position);
}

void BaseClusterScheduler::require_quiescent() const {
  moe_barrier_.require_empty();
  if (!moe_stage_states_.empty() || !moe_group_states_.empty()) {
    throw std::runtime_error(
        "simulation quiesced with live cluster MoE state");
  }
}

}  // namespace frontier::scheduler
