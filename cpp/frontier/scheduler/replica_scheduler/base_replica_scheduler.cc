#include "frontier/scheduler/replica_scheduler/base_replica_scheduler.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>
#include <utility>

namespace frontier::scheduler {

std::string_view to_string(SchedulerDecisionType decision) noexcept {
    switch (decision) {
    case SchedulerDecisionType::kRunningScheduled:
        return "RUNNING_SCHEDULED";
    case SchedulerDecisionType::kAdmission:
        return "ADMISSION";
    case SchedulerDecisionType::kPreempted:
        return "PREEMPTED";
    }
    return "UNKNOWN";
}

BaseReplicaScheduler::BaseReplicaScheduler(
    config::SchedulerConfig config, std::vector<entities::Request> &requests,
    const entities::Replica &replica, DataParallelId dp_id,
    execution_time_predictor::ExecutionTimePredictorPtr predictor,
    ClusterType cluster_type)
    : config_(std::move(config)), requests_(&requests), kv_blocks_(config_),
      pipeline_parallel_size_(replica.pipeline_parallel_size()),
      replica_(&replica), dp_id_(dp_id), cluster_type_(cluster_type) {
    const std::uint64_t pipeline_parallel_size =
        replica.pipeline_parallel_size();
    if (predictor == nullptr || pipeline_parallel_size == 0) {
        throw SchedulerError(
            "replica scheduler requires an execution model and positive PP");
    }
    stage_schedulers_.reserve(static_cast<std::size_t>(pipeline_parallel_size));
    for (std::uint64_t stage = 0; stage < pipeline_parallel_size; ++stage) {
        stage_schedulers_.emplace_back(replica.id(), dp_id, StageId{stage},
                                       stage + 1 == pipeline_parallel_size,
                                       predictor);
    }
}

entities::Request &BaseReplicaScheduler::request(RequestId request_id) {
    if (!request_id.valid() || request_id.index() >= requests_->size()) {
        throw SchedulerError("scheduler references an unknown request ID");
    }
    entities::Request &value = requests_->at(request_id.index());
    if (value.id() != request_id) {
        throw SchedulerError("request arena ID/index invariant failed");
    }
    return value;
}

const entities::Request &
BaseReplicaScheduler::request(RequestId request_id) const {
    if (!request_id.valid() || request_id.index() >= requests_->size()) {
        throw SchedulerError("scheduler references an unknown request ID");
    }
    const entities::Request &value = requests_->at(request_id.index());
    if (value.id() != request_id) {
        throw SchedulerError("request arena ID/index invariant failed");
    }
    return value;
}

bool BaseReplicaScheduler::request_is_active(RequestId request_id) const {
    return active_requests_.find(request_id) != active_requests_.end();
}

void BaseReplicaScheduler::add_request(RequestId request_id) {
    entities::Request &value = request(request_id);
    if (value.state() != entities::RequestState::kWaiting) {
        throw SchedulerError("scheduler accepts only waiting requests");
    }
    if (contains_request(request_id)) {
        throw SchedulerError("request is already present in scheduler state");
    }
    waiting_.push_back(request_id);
    validate_policy_state();
}

ScheduleResult BaseReplicaScheduler::schedule(SimTime time) {
    if (!time.valid() || !std::isfinite(time.seconds()) ||
        time.seconds() < 0.0) {
        throw SchedulerError("schedule time must be finite and nonnegative");
    }
    if (in_flight_batch_count_ >= pipeline_parallel_size_) {
        throw SchedulerError(
            "cannot schedule beyond pipeline in-flight capacity");
    }
    validate_lifecycle_state();
    ScheduleResult result = schedule_requests(time);
    validate_lifecycle_state();
    return result;
}

void BaseReplicaScheduler::mark_batch_started(const entities::Batch &batch) {
    if (in_flight_batch_count_ >= pipeline_parallel_size_) {
        throw SchedulerError("a scheduler batch exceeds pipeline capacity");
    }
    std::vector<RequestId> request_ids;
    request_ids.reserve(batch.requests().size());
    for (const entities::RequestBatchSnapshot &snapshot : batch.requests()) {
        auto [position, inserted] =
            active_requests_.try_emplace(snapshot.request_id, 0);
        static_cast<void>(inserted);
        if (cluster_type() != ClusterType::kPrefill && position->second != 0) {
            throw SchedulerError(
                "request is already active in another pipeline batch");
        }
        if (position->second == std::numeric_limits<std::uint64_t>::max()) {
            throw SchedulerError("active request batch count overflows uint64");
        }
        ++position->second;
        request_ids.push_back(snapshot.request_id);
    }
    if (!in_flight_batches_.emplace(batch.id(), std::move(request_ids))
             .second) {
        throw SchedulerError("batch is already in flight");
    }
    ++in_flight_batch_count_;
    validate_lifecycle_state();
}

bool BaseReplicaScheduler::on_batch_completed(entities::Batch &batch,
                                              SimTime time) {
    const auto in_flight = in_flight_batches_.find(batch.id());
    if (in_flight == in_flight_batches_.end() || batch.completed()) {
        return false;
    }

    const bool has_valid_request = apply_batch_completion(batch, time);
    batch.mark_completed(time);
    for (const RequestId request_id : in_flight->second) {
        const auto active = active_requests_.find(request_id);
        if (active == active_requests_.end() || active->second == 0) {
            throw SchedulerError(
                "completed batch is missing active request state");
        }
        --active->second;
        if (active->second == 0) {
            active_requests_.erase(active);
        }
    }
    in_flight_batches_.erase(in_flight);
    if (in_flight_batch_count_ == 0) {
        throw SchedulerError("in-flight batch count underflow");
    }
    --in_flight_batch_count_;
    validate_lifecycle_state();
    validate_policy_state();
    return has_valid_request;
}

void BaseReplicaScheduler::validate_lifecycle_state() const {
    if (pipeline_parallel_size_ == 0 ||
        in_flight_batch_count_ != in_flight_batches_.size() ||
        in_flight_batch_count_ > pipeline_parallel_size_) {
        throw SchedulerError("replica scheduler in-flight state is invalid");
    }
    std::unordered_map<RequestId, std::uint64_t, StrongIdHash<RequestId>>
        expected_active;
    for (const auto &[batch_id, request_ids] : in_flight_batches_) {
        if (!batch_id.valid()) {
            throw SchedulerError("in-flight scheduler batch ID is invalid");
        }
        for (const RequestId request_id : request_ids) {
            auto [position, inserted] =
                expected_active.try_emplace(request_id, 0);
            static_cast<void>(inserted);
            if (position->second == std::numeric_limits<std::uint64_t>::max()) {
                throw SchedulerError(
                    "in-flight request reference count overflows uint64");
            }
            ++position->second;
        }
    }
    if (expected_active != active_requests_) {
        throw SchedulerError(
            "active request state disagrees with in-flight batches");
    }
}

ReplicaStageScheduler &
BaseReplicaScheduler::get_replica_stage_scheduler(StageId stage_id) {
    if (!stage_id.valid() || stage_id.index() >= stage_schedulers_.size()) {
        throw SchedulerError(
            "single-stage scheduler exposes only pipeline stage zero");
    }
    return stage_schedulers_.at(stage_id.index());
}

const ReplicaStageScheduler &
BaseReplicaScheduler::get_replica_stage_scheduler(StageId stage_id) const {
    if (!stage_id.valid() || stage_id.index() >= stage_schedulers_.size()) {
        throw SchedulerError(
            "single-stage scheduler exposes only pipeline stage zero");
    }
    return stage_schedulers_.at(stage_id.index());
}

} // namespace frontier::scheduler
