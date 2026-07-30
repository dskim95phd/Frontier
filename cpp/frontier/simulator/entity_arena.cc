#include "frontier/simulator/entity_arena.h"

#include <stdexcept>

namespace frontier::simulator {
namespace {

std::vector<entities::RequestBatchSnapshot> make_snapshots(
    const scheduler::ScheduleResult& schedule,
    const std::vector<entities::Request>& requests) {
  std::vector<entities::RequestBatchSnapshot> result;
  result.reserve(schedule.scheduled_requests.size());
  for (const scheduler::ScheduledRequest& scheduled :
       schedule.scheduled_requests) {
    const entities::Request& request = requests.at(
        scheduled.request_id.index());
    result.push_back(entities::RequestBatchSnapshot{
        .request_id = scheduled.request_id,
        .scheduled_tokens = scheduled.num_tokens,
        .runtime_epoch = request.runtime_epoch(),
        .execution_epoch = request.execution_epoch(),
        .processed_tokens = request.num_processed_tokens(),
        .scheduler_frontier =
            request.scheduler_num_computed_tokens(),
    });
  }
  return result;
}

}  // namespace

EntityArena::EntityArena(
    const std::vector<request_generator::WorkloadRequest>& workload,
    config::SimulationMode simulation_mode)
    : request_transfer_ids_(workload.size()),
      completion_recorded_(workload.size(), false) {
  requests_.reserve(workload.size());
  for (const auto& request : workload) {
    request_generator::WorkloadRequest normalized = request;
    if (simulation_mode == config::SimulationMode::kOffline) {
      normalized.arrived_at = SimTime::from_seconds(0.0);
    }
    requests_.emplace_back(normalized);
  }
}

entities::Request& EntityArena::request(RequestId request_id) {
  if (!request_id.valid() ||
      request_id.index() >= requests_.size()) {
    throw std::out_of_range("event references unknown request");
  }
  entities::Request& result = requests_.at(
      request_id.index());
  if (result.id() != request_id) {
    throw std::logic_error("request arena invariant failed");
  }
  return result;
}

const entities::Request& EntityArena::request(
    RequestId request_id) const {
  if (!request_id.valid() ||
      request_id.index() >= requests_.size()) {
    throw std::out_of_range("event references unknown request");
  }
  const entities::Request& result = requests_.at(
      request_id.index());
  if (result.id() != request_id) {
    throw std::logic_error("request arena invariant failed");
  }
  return result;
}

entities::Batch& EntityArena::batch(BatchId batch_id) {
  if (!batch_id.valid() ||
      batch_id.index() >= batches_.size()) {
    throw std::out_of_range("event references unknown batch");
  }
  entities::Batch& result = batches_.at(
      batch_id.index());
  if (result.id() != batch_id) {
    throw std::logic_error("batch arena invariant failed");
  }
  return result;
}

const entities::Batch& EntityArena::batch(BatchId batch_id) const {
  if (!batch_id.valid() ||
      batch_id.index() >= batches_.size()) {
    throw std::out_of_range("event references unknown batch");
  }
  const entities::Batch& result = batches_.at(
      batch_id.index());
  if (result.id() != batch_id) {
    throw std::logic_error("batch arena invariant failed");
  }
  return result;
}

BatchId EntityArena::create_batch(
    const scheduler::ScheduleResult& schedule,
    scheduler::ReplicaTarget target,
    ClusterType cluster_type,
    std::uint64_t pipeline_parallel_size) {
  const BatchId batch_id{
      static_cast<BatchId::ValueType>(batches_.size())};
  const Generation generation{
      static_cast<Generation::ValueType>(batches_.size() + 1)};
  batches_.emplace_back(
      batch_id,
      schedule.iteration_id,
      make_snapshots(schedule, requests_),
      schedule.simulation_time,
      generation,
      target.replica_id,
      target.dp_id,
      pipeline_parallel_size,
      cluster_type);
  stage_arrival_times_.emplace_back(
      static_cast<std::size_t>(pipeline_parallel_size));
  stage_record_indices_.emplace_back(
      static_cast<std::size_t>(pipeline_parallel_size));
  predicted_batch_ms_.push_back(0.0);
  return batch_id;
}

void EntityArena::record_stage_arrival(
    BatchId batch_id,
    StageId stage_id,
    SimTime time) {
  auto& arrivals = stage_arrival_times_.at(
      batch_id.index());
  SimTime& arrival = arrivals.at(
      stage_id.index());
  if (arrival.valid()) {
    throw std::logic_error("batch stage arrived more than once");
  }
  arrival = time;
}

entities::BatchStage& EntityArena::create_batch_stage(
    BatchId batch_id,
    StageId stage_id,
    SimTime started_at,
    const execution_time_predictor::BatchExecutionPrediction&
        prediction) {
  const std::size_t batch_index =
      batch_id.index();
  const std::size_t stage_index =
      stage_id.index();
  const SimTime arrival =
      stage_arrival_times_.at(batch_index).at(stage_index);
  if (!arrival.valid()) {
    throw std::logic_error("batch stage schedule has no arrival");
  }
  std::optional<std::size_t>& record_index =
      stage_record_indices_.at(batch_index).at(stage_index);
  if (record_index.has_value()) {
    throw std::logic_error("batch stage record already exists");
  }
  const entities::Batch& owner = batch(batch_id);
  batch_stages_.emplace_back(
      batch_id,
      owner.replica_id(),
      owner.dp_id(),
      stage_id,
      arrival,
      prediction.execution_time);
  record_index = batch_stages_.size() - 1;
  entities::BatchStage& stage = batch_stages_.back();
  stage.mark_started(started_at);
  predicted_batch_ms_.at(batch_index) += prediction.duration_ms;
  return stage;
}

entities::BatchStage& EntityArena::batch_stage(
    BatchId batch_id,
    StageId stage_id) {
  const std::optional<std::size_t>& index =
      stage_record_indices_
          .at(batch_id.index())
          .at(stage_id.index());
  if (!index.has_value()) {
    throw std::logic_error("batch stage record does not exist");
  }
  return batch_stages_.at(index.value());
}

double EntityArena::predicted_batch_ms(BatchId batch_id) const {
  return predicted_batch_ms_.at(
      batch_id.index());
}

void EntityArena::add_target_domain(ClusterType cluster_type) {
  const auto [position, inserted] = request_targets_.try_emplace(
      cluster_type, requests_.size());
  static_cast<void>(position);
  if (!inserted) {
    throw std::logic_error("request target domain already exists");
  }
}

void EntityArena::assign_request_target(
    RequestId request_id,
    scheduler::ReplicaTarget target,
    ClusterType cluster_type) {
  scheduler::ReplicaTarget& current =
      request_targets_.at(cluster_type).at(
          request_id.index());
  if (current.replica_id.valid() && current != target) {
    throw std::logic_error(
        "request was routed to more than one target");
  }
  current = target;
}

scheduler::ReplicaTarget EntityArena::request_target(
    RequestId request_id,
    ClusterType cluster_type) const {
  const auto& target = request_targets_.at(cluster_type).at(
      request_id.index());
  if (!target.replica_id.valid() || !target.dp_id.valid()) {
    throw std::logic_error("request has no replica target");
  }
  return target;
}

TransferId EntityArena::create_kv_cache_transfer(
    RequestId request_id,
    BatchId source_batch_id,
    scheduler::ReplicaTarget source_target,
    std::uint64_t size_bytes,
    double predicted_time_ms) {
  const std::size_t request_index =
      request_id.index();
  if (request_transfer_ids_.at(request_index).valid()) {
    throw std::logic_error("request already owns a KV transfer");
  }
  const TransferId transfer_id{
      static_cast<TransferId::ValueType>(
          kv_cache_transfers_.size())};
  kv_cache_transfers_.emplace_back(
      transfer_id,
      request_id,
      source_batch_id,
      source_target.replica_id,
      source_target.dp_id,
      size_bytes,
      predicted_time_ms,
      batch(source_batch_id).schedule_epoch());
  request_transfer_ids_.at(request_index) = transfer_id;
  return transfer_id;
}

entities::KVCacheTransferInfo& EntityArena::kv_cache_transfer(
    TransferId transfer_id) {
  if (!transfer_id.valid() ||
      transfer_id.index() >= kv_cache_transfers_.size()) {
    throw std::out_of_range(
        "event references unknown KV transfer");
  }
  entities::KVCacheTransferInfo& transfer =
      kv_cache_transfers_.at(
          transfer_id.index());
  if (transfer.id() != transfer_id) {
    throw std::logic_error("KV transfer arena invariant failed");
  }
  return transfer;
}

const entities::KVCacheTransferInfo& EntityArena::kv_cache_transfer(
    TransferId transfer_id) const {
  if (!transfer_id.valid() ||
      transfer_id.index() >= kv_cache_transfers_.size()) {
    throw std::out_of_range(
        "event references unknown KV transfer");
  }
  const entities::KVCacheTransferInfo& transfer =
      kv_cache_transfers_.at(
          transfer_id.index());
  if (transfer.id() != transfer_id) {
    throw std::logic_error("KV transfer arena invariant failed");
  }
  return transfer;
}

TransferId EntityArena::request_transfer_id(
    RequestId request_id) const {
  const auto& transfer = request_transfer_ids_.at(
      request_id.index());
  if (!transfer.valid()) {
    throw std::logic_error("request has no KV transfer");
  }
  return transfer;
}

void EntityArena::record_request_completion(RequestId request_id) {
  const std::size_t index =
      request_id.index();
  if (!completion_recorded_.at(index)) {
    completion_recorded_[index] = true;
    completion_order_.push_back(request_id);
  }
}

bool EntityArena::request_completion_recorded(
    RequestId request_id) const {
  return completion_recorded_.at(
      request_id.index());
}

}  // namespace frontier::simulator
