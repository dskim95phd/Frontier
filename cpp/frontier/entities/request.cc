#include "frontier/entities/request.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace frontier::entities {

Request::Request(
    const request_generator::WorkloadRequest& workload_request)
    : request_id_(workload_request.request_id),
      arrived_at_(workload_request.arrived_at),
      num_prefill_tokens_(workload_request.num_prefill_tokens),
      num_decode_tokens_(workload_request.num_decode_tokens),
      session_id_(workload_request.session_id),
      session_turn_index_(workload_request.session_turn_index) {
  if (!std::isfinite(arrived_at_.seconds()) ||
      arrived_at_.seconds() < 0.0) {
    throw RequestError("request arrival must be finite and nonnegative");
  }
  if (num_prefill_tokens_ == 0 || num_decode_tokens_ == 0) {
    throw RequestError("request token counts must be positive");
  }
  if (num_prefill_tokens_ >
      std::numeric_limits<std::uint64_t>::max() - num_decode_tokens_) {
    throw RequestError("request total token count overflows uint64");
  }
  if (session_turn_index_.has_value() && !session_id_.has_value()) {
    throw RequestError("session_turn_index requires session_id");
  }
}

std::uint64_t Request::num_processed_prefill_tokens() const noexcept {
  return std::min(num_processed_tokens_, num_prefill_tokens_);
}

std::uint64_t Request::num_processed_decode_tokens() const noexcept {
  return num_processed_tokens_ > num_prefill_tokens_
      ? num_processed_tokens_ - num_prefill_tokens_
      : 0;
}

std::uint64_t Request::remaining_decode_tokens() const noexcept {
  return num_decode_tokens_ - num_processed_decode_tokens();
}

void Request::validate_time(SimTime time, const char* context) const {
  if (!std::isfinite(time.seconds()) || time.seconds() < 0.0) {
    throw RequestError(
        std::string{context} + " time must be finite and nonnegative");
  }
}

void Request::validate_progress() const {
  if (num_processed_tokens_ > total_tokens()) {
    throw RequestError("request-visible progress exceeds total tokens");
  }
  if (scheduler_num_computed_tokens_ > total_tokens()) {
    throw RequestError("scheduler frontier exceeds total tokens");
  }
  if (completed() && num_processed_tokens_ != total_tokens()) {
    throw RequestError("completed request has incomplete token progress");
  }
}

void Request::enter_waiting(SimTime time) {
  validate_time(time, "waiting-entry");
  if (waiting_since_.has_value()) {
    throw RequestError("request entered waiting queue twice");
  }
  waiting_since_ = time;
  state_ = RequestState::kWaiting;
}

void Request::leave_waiting(SimTime time) {
  validate_time(time, "waiting-exit");
  if (!waiting_since_.has_value()) {
    throw RequestError("request left waiting queue without entering it");
  }
  if (time.seconds() < waiting_since_->seconds()) {
    throw RequestError("waiting exit precedes waiting entry");
  }
  cumulative_waiting_time_s_ +=
      time.seconds() - waiting_since_->seconds();
  waiting_since_.reset();
}

void Request::on_arrival(SimTime time) {
  if (state_ != RequestState::kPending) {
    throw RequestError("request arrived more than once");
  }
  if (time != arrived_at_) {
    throw RequestError("arrival event time differs from request arrival");
  }
  enter_waiting(time);
}

void Request::on_admitted(SimTime time) {
  if (state_ != RequestState::kWaiting) {
    throw RequestError("only a waiting request can be admitted");
  }
  leave_waiting(time);
  if (!first_scheduled_at_.has_value()) {
    first_scheduled_at_ = time;
  }
  state_ = RequestState::kRunning;
  preempted_ = false;
}

void Request::advance_scheduler_frontier(
    std::uint64_t scheduled_tokens) {
  if (scheduled_tokens == 0) {
    throw RequestError("scheduled token count must be positive");
  }
  if (scheduled_tokens > total_tokens()) {
    throw RequestError("scheduled token count exceeds request total");
  }
  if (scheduler_num_computed_tokens_ >
      total_tokens() - scheduled_tokens) {
    throw RequestError("scheduled token count exceeds request total");
  }
  scheduler_num_computed_tokens_ += scheduled_tokens;
  validate_progress();
}

void Request::on_batch_completion(
    SimTime time,
    std::uint64_t scheduled_tokens) {
  validate_time(time, "batch-completion");
  if (state_ != RequestState::kRunning || completed()) {
    throw RequestError("batch completed for a non-running request");
  }
  if (scheduled_tokens == 0) {
    throw RequestError("completed batch must contain positive tokens");
  }

  const bool was_prefill_complete = is_prefill_complete_;
  if (!was_prefill_complete) {
    const std::uint64_t remaining_prefill =
        num_prefill_tokens_ - num_processed_tokens_;
    if (scheduled_tokens > remaining_prefill) {
      throw RequestError(
          "prefill batch exceeds remaining prompt tokens");
    }
    num_processed_tokens_ += scheduled_tokens;
    if (num_processed_tokens_ == num_prefill_tokens_) {
      is_prefill_complete_ = true;
      if (!prefill_completed_at_.has_value()) {
        prefill_completed_at_ = time;
      }
      // Frontier's monolithic Python path exposes the first generated token
      // at the same boundary as the final prefill chunk.
      ++num_processed_tokens_;
      if (!first_token_completed_at_.has_value()) {
        first_token_completed_at_ = time;
      }
    }
  } else {
    const std::uint64_t remaining = total_tokens() - num_processed_tokens_;
    if (scheduled_tokens > remaining) {
      throw RequestError(
          "decode batch exceeds remaining output tokens");
    }
    num_processed_tokens_ += scheduled_tokens;
    if (!first_token_completed_at_.has_value() &&
        num_processed_decode_tokens() > 0) {
      first_token_completed_at_ = time;
    }
  }

  if (num_processed_tokens_ == total_tokens()) {
    state_ = RequestState::kCompleted;
    completed_at_ = time;
  }
  validate_progress();
}

void Request::on_preempted(SimTime time) {
  validate_time(time, "preemption");
  if (state_ != RequestState::kRunning || completed()) {
    throw RequestError("only a running request can be preempted");
  }
  tokens_at_preemption_.push_back(scheduler_num_computed_tokens_);
  ++preemption_count_;
  ++runtime_epoch_;
  ++execution_epoch_;
  num_processed_tokens_ = 0;
  scheduler_num_computed_tokens_ = 0;
  is_prefill_complete_ = false;
  preempted_ = true;
  enter_waiting(time);
  validate_progress();
}

}  // namespace frontier::entities
