#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

#include "frontier/core/event.h"
#include "frontier/core/ids.h"
#include "frontier/request_generator/workload.h"

namespace frontier::entities {

enum class RequestState : std::uint8_t {
  kPending,
  kWaiting,
  kRunning,
  kCompleted,
};

class RequestError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class Request {
 public:
  explicit Request(
      const request_generator::WorkloadRequest& workload_request);

  [[nodiscard]] RequestId id() const noexcept { return request_id_; }
  [[nodiscard]] SimTime arrived_at() const noexcept { return arrived_at_; }
  [[nodiscard]] std::uint64_t num_prefill_tokens() const noexcept {
    return num_prefill_tokens_;
  }
  [[nodiscard]] std::uint64_t num_decode_tokens() const noexcept {
    return num_decode_tokens_;
  }
  [[nodiscard]] std::uint64_t total_tokens() const noexcept {
    return num_prefill_tokens_ + num_decode_tokens_;
  }
  [[nodiscard]] const std::optional<SessionId>& session_id() const noexcept {
    return session_id_;
  }
  [[nodiscard]] const std::optional<std::uint64_t>&
  session_turn_index() const noexcept {
    return session_turn_index_;
  }

  [[nodiscard]] RequestState state() const noexcept { return state_; }
  [[nodiscard]] bool is_prefill_complete() const noexcept {
    return is_prefill_complete_;
  }
  [[nodiscard]] bool completed() const noexcept {
    return state_ == RequestState::kCompleted;
  }
  [[nodiscard]] bool preempted() const noexcept { return preempted_; }
  [[nodiscard]] std::uint64_t num_processed_tokens() const noexcept {
    return num_processed_tokens_;
  }
  [[nodiscard]] std::uint64_t scheduler_num_computed_tokens() const noexcept {
    return scheduler_num_computed_tokens_;
  }
  [[nodiscard]] std::uint64_t num_processed_prefill_tokens() const noexcept;
  [[nodiscard]] std::uint64_t num_processed_decode_tokens() const noexcept;
  [[nodiscard]] std::uint64_t remaining_decode_tokens() const noexcept;
  [[nodiscard]] std::uint64_t runtime_epoch() const noexcept {
    return runtime_epoch_;
  }
  [[nodiscard]] std::uint64_t execution_epoch() const noexcept {
    return execution_epoch_;
  }
  [[nodiscard]] std::uint64_t preemption_count() const noexcept {
    return preemption_count_;
  }
  [[nodiscard]] const std::vector<std::uint64_t>&
  tokens_at_preemption() const noexcept {
    return tokens_at_preemption_;
  }

  [[nodiscard]] const std::optional<SimTime>& first_scheduled_at()
      const noexcept {
    return first_scheduled_at_;
  }
  [[nodiscard]] const std::optional<SimTime>& prefill_completed_at()
      const noexcept {
    return prefill_completed_at_;
  }
  [[nodiscard]] const std::optional<SimTime>&
  first_token_completed_at() const noexcept {
    return first_token_completed_at_;
  }
  [[nodiscard]] const std::optional<SimTime>& completed_at() const noexcept {
    return completed_at_;
  }
  [[nodiscard]] double cumulative_waiting_time_s() const noexcept {
    return cumulative_waiting_time_s_;
  }

  void on_arrival(SimTime time);
  void on_admitted(SimTime time);
  void advance_scheduler_frontier(std::uint64_t scheduled_tokens);
  void on_batch_completion(
      SimTime time,
      std::uint64_t scheduled_tokens);
  void on_preempted(SimTime time);

 private:
  void enter_waiting(SimTime time);
  void leave_waiting(SimTime time);
  void validate_time(SimTime time, const char* context) const;
  void validate_progress() const;

  RequestId request_id_;
  SimTime arrived_at_;
  std::uint64_t num_prefill_tokens_;
  std::uint64_t num_decode_tokens_;
  std::optional<SessionId> session_id_;
  std::optional<std::uint64_t> session_turn_index_;

  RequestState state_ = RequestState::kPending;
  std::uint64_t num_processed_tokens_ = 0;
  std::uint64_t scheduler_num_computed_tokens_ = 0;
  bool is_prefill_complete_ = false;
  bool preempted_ = false;
  std::uint64_t runtime_epoch_ = 0;
  std::uint64_t execution_epoch_ = 0;
  std::uint64_t preemption_count_ = 0;
  std::vector<std::uint64_t> tokens_at_preemption_;

  std::optional<SimTime> first_scheduled_at_;
  std::optional<SimTime> prefill_completed_at_;
  std::optional<SimTime> first_token_completed_at_;
  std::optional<SimTime> completed_at_;
  std::optional<SimTime> waiting_since_;
  double cumulative_waiting_time_s_ = 0.0;
};

}  // namespace frontier::entities
