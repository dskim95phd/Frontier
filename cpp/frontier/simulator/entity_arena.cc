#include "frontier/simulator/entity_arena.h"

#include <stdexcept>

namespace frontier::simulator {
namespace {

std::vector<entities::RequestBatchSnapshot>
make_snapshots(const scheduler::ScheduleResult &schedule,
               const std::vector<entities::Request> &requests) {
    std::vector<entities::RequestBatchSnapshot> result;
    result.reserve(schedule.scheduled_requests.size());
    for (const scheduler::ScheduledRequest &scheduled :
         schedule.scheduled_requests) {
        const entities::Request &request =
            requests.at(scheduled.request_id.index());
        result.push_back([&]() {
            entities::RequestBatchSnapshot value{};
            value.request_id = scheduled.request_id;
            value.scheduled_tokens = scheduled.num_tokens;
            value.runtime_epoch = request.runtime_epoch();
            value.execution_epoch = request.execution_epoch();
            value.processed_tokens = request.num_processed_tokens();
            value.scheduler_frontier = request.scheduler_num_computed_tokens();
            return value;
        }());
    }
    return result;
}

} // namespace

EntityArena::EntityArena(
    const std::vector<request_generator::WorkloadRequest> &workload)
    : request_transfer_ids_(workload.size()),
      completion_recorded_(workload.size(), false) {
    requests_.reserve(workload.size());
    for (const auto &request : workload) {
        requests_.emplace_back(request);
    }
}

entities::Request &EntityArena::request(RequestId request_id) {
    if (!request_id.valid() || request_id.index() >= requests_.size()) {
        throw std::out_of_range("event references unknown request");
    }
    entities::Request &result = requests_.at(request_id.index());
    if (result.id() != request_id) {
        throw std::logic_error("request arena invariant failed");
    }
    return result;
}

const entities::Request &EntityArena::request(RequestId request_id) const {
    if (!request_id.valid() || request_id.index() >= requests_.size()) {
        throw std::out_of_range("event references unknown request");
    }
    const entities::Request &result = requests_.at(request_id.index());
    if (result.id() != request_id) {
        throw std::logic_error("request arena invariant failed");
    }
    return result;
}

entities::Batch &EntityArena::batch(BatchId batch_id) {
    const auto position = batches_.find(batch_id.value());
    if (!batch_id.valid() || position == batches_.end()) {
        throw std::out_of_range("event references unknown batch");
    }
    entities::Batch &result = *position->second.batch;
    if (result.id() != batch_id) {
        throw std::logic_error("batch arena invariant failed");
    }
    return result;
}

const entities::Batch &EntityArena::batch(BatchId batch_id) const {
    const auto position = batches_.find(batch_id.value());
    if (!batch_id.valid() || position == batches_.end()) {
        throw std::out_of_range("event references unknown batch");
    }
    const entities::Batch &result = *position->second.batch;
    if (result.id() != batch_id) {
        throw std::logic_error("batch arena invariant failed");
    }
    return result;
}

BatchId EntityArena::create_batch(const scheduler::ScheduleResult &schedule,
                                  scheduler::ReplicaTarget target,
                                  ClusterType cluster_type,
                                  std::uint64_t pipeline_parallel_size,
                                  config::ModelKind model_kind) {
    const BatchId batch_id{next_batch_id_++};
    const Generation generation{
        static_cast<Generation::ValueType>(batch_id.value() + 1)};
    BatchRuntimeState state;
    state.batch = std::make_unique<entities::Batch>(
        batch_id, schedule.iteration_id, make_snapshots(schedule, requests_),
        schedule.simulation_time, generation, target.replica_id, target.dp_id,
        pipeline_parallel_size, cluster_type, entities::BatchKind::kWork,
        model_kind);
    state.stage_arrival_times.resize(
        static_cast<std::size_t>(pipeline_parallel_size));
    state.batch_stages.resize(static_cast<std::size_t>(pipeline_parallel_size));
    const auto [position, inserted] =
        batches_.emplace(batch_id.value(), std::move(state));
    static_cast<void>(position);
    if (!inserted) {
        throw std::overflow_error("batch ID space exhausted");
    }
    return batch_id;
}

BatchId EntityArena::create_moe_idle_batch(MoESyncGroupId sync_group_id,
                                           MoEParticipantId participant_id,
                                           scheduler::ReplicaTarget target,
                                           ClusterType cluster_type,
                                           std::uint64_t pipeline_parallel_size,
                                           SimTime created_at,
                                           Generation generation) {
    if (!sync_group_id.valid() || !participant_id.valid() ||
        !target.replica_id.valid() || !target.dp_id.valid() ||
        !generation.valid()) {
        throw std::invalid_argument(
            "idle MoE batch requires valid synchronization ownership");
    }
    const BatchId batch_id{next_batch_id_++};
    BatchRuntimeState state;
    state.batch = std::make_unique<entities::Batch>(
        batch_id, IterationId{sync_group_id.value()},
        std::vector<entities::RequestBatchSnapshot>{}, created_at, generation,
        target.replica_id, target.dp_id, pipeline_parallel_size, cluster_type,
        entities::BatchKind::kMoeIdle, config::ModelKind::kMoe, sync_group_id,
        participant_id);
    state.stage_arrival_times.resize(
        static_cast<std::size_t>(pipeline_parallel_size));
    state.batch_stages.resize(static_cast<std::size_t>(pipeline_parallel_size));
    const auto [position, inserted] =
        batches_.emplace(batch_id.value(), std::move(state));
    static_cast<void>(position);
    if (!inserted) {
        throw std::overflow_error("batch ID space exhausted");
    }
    return batch_id;
}

void EntityArena::record_stage_arrival(BatchId batch_id, StageId stage_id,
                                       SimTime time) {
    auto &arrivals = batches_.at(batch_id.value()).stage_arrival_times;
    SimTime &arrival = arrivals.at(stage_id.index());
    if (arrival.valid()) {
        throw std::logic_error("batch stage arrived more than once");
    }
    arrival = time;
}

entities::BatchStage &EntityArena::create_batch_stage(
    BatchId batch_id, StageId stage_id, SimTime started_at,
    const execution_time_predictor::ExecutionTimePrediction &prediction) {
    const std::size_t stage_index = stage_id.index();
    BatchRuntimeState &state = batches_.at(batch_id.value());
    const SimTime arrival = state.stage_arrival_times.at(stage_index);
    if (!arrival.valid()) {
        throw std::logic_error("batch stage schedule has no arrival");
    }
    std::optional<entities::BatchStage> &record =
        state.batch_stages.at(stage_index);
    if (record.has_value()) {
        throw std::logic_error("batch stage record already exists");
    }
    const entities::Batch &owner = batch(batch_id);
    record.emplace(batch_id, owner.replica_id(), owner.dp_id(), stage_id,
                   arrival, prediction.execution_time);
    entities::BatchStage &stage = record.value();
    stage.mark_started(started_at);
    state.predicted_batch_ms += prediction.duration_ms;
    return stage;
}

entities::BatchStage &EntityArena::batch_stage(BatchId batch_id,
                                               StageId stage_id) {
    std::optional<entities::BatchStage> &record =
        batches_.at(batch_id.value()).batch_stages.at(stage_id.index());
    if (!record.has_value()) {
        throw std::logic_error("batch stage record does not exist");
    }
    return record.value();
}

double EntityArena::predicted_batch_ms(BatchId batch_id) const {
    return batches_.at(batch_id.value()).predicted_batch_ms;
}

void EntityArena::release_batch(BatchId batch_id) {
    if (batches_.erase(batch_id.value()) != 1) {
        throw std::out_of_range("cannot release unknown batch");
    }
}

void EntityArena::add_target_domain(ClusterType cluster_type) {
    const auto [position, inserted] =
        request_targets_.try_emplace(cluster_type, requests_.size());
    static_cast<void>(position);
    if (!inserted) {
        throw std::logic_error("request target domain already exists");
    }
}

void EntityArena::assign_request_target(RequestId request_id,
                                        scheduler::ReplicaTarget target,
                                        ClusterType cluster_type) {
    scheduler::ReplicaTarget &current =
        request_targets_.at(cluster_type).at(request_id.index());
    if (current.replica_id.valid() && current != target) {
        throw std::logic_error("request was routed to more than one target");
    }
    current = target;
}

scheduler::ReplicaTarget
EntityArena::request_target(RequestId request_id,
                            ClusterType cluster_type) const {
    const auto &target =
        request_targets_.at(cluster_type).at(request_id.index());
    if (!target.replica_id.valid() || !target.dp_id.valid()) {
        throw std::logic_error("request has no replica target");
    }
    return target;
}

TransferId EntityArena::create_kv_cache_transfer(
    RequestId request_id, BatchId source_batch_id,
    scheduler::ReplicaTarget source_target, std::uint64_t size_bytes,
    double predicted_time_ms) {
    const std::size_t request_index = request_id.index();
    if (request_transfer_ids_.at(request_index).valid()) {
        throw std::logic_error("request already owns a KV transfer");
    }
    const TransferId transfer_id{
        static_cast<TransferId::ValueType>(kv_cache_transfers_.size())};
    kv_cache_transfers_.emplace_back(
        transfer_id, request_id, source_batch_id, source_target.replica_id,
        source_target.dp_id, size_bytes, predicted_time_ms,
        batch(source_batch_id).schedule_epoch());
    request_transfer_ids_.at(request_index) = transfer_id;
    return transfer_id;
}

entities::KVCacheTransferInfo &
EntityArena::kv_cache_transfer(TransferId transfer_id) {
    if (!transfer_id.valid() ||
        transfer_id.index() >= kv_cache_transfers_.size()) {
        throw std::out_of_range("event references unknown KV transfer");
    }
    entities::KVCacheTransferInfo &transfer =
        kv_cache_transfers_.at(transfer_id.index());
    if (transfer.id() != transfer_id) {
        throw std::logic_error("KV transfer arena invariant failed");
    }
    return transfer;
}

const entities::KVCacheTransferInfo &
EntityArena::kv_cache_transfer(TransferId transfer_id) const {
    if (!transfer_id.valid() ||
        transfer_id.index() >= kv_cache_transfers_.size()) {
        throw std::out_of_range("event references unknown KV transfer");
    }
    const entities::KVCacheTransferInfo &transfer =
        kv_cache_transfers_.at(transfer_id.index());
    if (transfer.id() != transfer_id) {
        throw std::logic_error("KV transfer arena invariant failed");
    }
    return transfer;
}

TransferId EntityArena::request_transfer_id(RequestId request_id) const {
    const auto &transfer = request_transfer_ids_.at(request_id.index());
    if (!transfer.valid()) {
        throw std::logic_error("request has no KV transfer");
    }
    return transfer;
}

void EntityArena::record_request_completion(RequestId request_id) {
    const std::size_t index = request_id.index();
    if (!completion_recorded_.at(index)) {
        completion_recorded_[index] = true;
        completion_order_.push_back(request_id);
    }
}

bool EntityArena::request_completion_recorded(RequestId request_id) const {
    return completion_recorded_.at(request_id.index());
}

} // namespace frontier::simulator
