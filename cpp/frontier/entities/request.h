#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

#include "frontier/core/cluster_type.h"
#include "frontier/core/event.h"
#include "frontier/core/ids.h"
#include "frontier/request_generator/workload.h"

namespace frontier::entities {

enum class RequestState : std::uint8_t {
    kPending,
    kWaiting,
    kRunning,
    kTransferPending,
    kTransferInFlight,
    kCompleted,
};

class RequestError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

class Request {
  public:
    explicit Request(
        const request_generator::WorkloadRequest &workload_request);

    [[nodiscard]] RequestId id() const noexcept { return request_id_; }
    [[nodiscard]] SimTime arrived_at() const noexcept { return arrived_at_; }
    [[nodiscard]] SimTime session_start_at() const noexcept {
        return session_start_at_;
    }
    [[nodiscard]] SimTime think_time() const noexcept { return think_time_; }
    void set_pending_arrival(SimTime time);
    [[nodiscard]] std::uint64_t num_prefill_tokens() const noexcept {
        return num_prefill_tokens_;
    }
    [[nodiscard]] std::uint64_t num_decode_tokens() const noexcept {
        return num_decode_tokens_;
    }
    [[nodiscard]] std::uint64_t initial_num_prefill_tokens() const noexcept {
        return initial_num_prefill_tokens_;
    }
    [[nodiscard]] std::uint64_t initial_num_decode_tokens() const noexcept {
        return initial_num_decode_tokens_;
    }
    [[nodiscard]] std::uint64_t total_tokens() const noexcept {
        return num_prefill_tokens_ + num_decode_tokens_;
    }
    [[nodiscard]] SessionId session_id() const noexcept { return session_id_; }
    [[nodiscard]] const std::optional<std::uint64_t> &
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
    [[nodiscard]] const std::vector<std::uint64_t> &
    tokens_at_preemption() const noexcept {
        return tokens_at_preemption_;
    }

    [[nodiscard]] SimTime first_scheduled_at() const noexcept {
        return first_scheduled_at_;
    }
    [[nodiscard]] SimTime prefill_completed_at() const noexcept {
        return prefill_completed_at_;
    }
    [[nodiscard]] SimTime first_token_completed_at() const noexcept {
        return first_token_completed_at_;
    }
    [[nodiscard]] SimTime completed_at() const noexcept {
        return completed_at_;
    }
    [[nodiscard]] SimTime kv_cache_transfer_start_time() const noexcept {
        return kv_cache_transfer_start_time_;
    }
    [[nodiscard]] SimTime kv_cache_transfer_end_time() const noexcept {
        return kv_cache_transfer_end_time_;
    }
    [[nodiscard]] SimTime decode_arrived_at() const noexcept {
        return decode_arrived_at_;
    }
    [[nodiscard]] std::uint64_t kv_cache_transfer_size_bytes() const noexcept {
        return kv_cache_transfer_size_bytes_;
    }
    [[nodiscard]] double kv_cache_transfer_time_s() const noexcept {
        return kv_cache_transfer_time_s_;
    }
    [[nodiscard]] double cumulative_waiting_time_s() const noexcept {
        return cumulative_waiting_time_s_;
    }
    [[nodiscard]] std::uint64_t cached_prefill_tokens() const noexcept {
        return cached_prefill_tokens_;
    }
    [[nodiscard]] std::uint64_t prefix_cache_query_blocks() const noexcept {
        return prefix_cache_query_blocks_;
    }
    [[nodiscard]] std::uint64_t prefix_cache_hit_blocks() const noexcept {
        return prefix_cache_hit_blocks_;
    }
    [[nodiscard]] config::PrefixCachingKeyMode
    prefix_cache_key_mode() const noexcept {
        return prefix_cache_key_mode_;
    }
    [[nodiscard]] bool prefix_cache_lookup_recorded() const noexcept {
        return prefix_cache_lookup_recorded_;
    }

    void on_arrival(SimTime time,
                    ClusterType cluster_type = ClusterType::kMonolithic);
    void on_admitted(SimTime time);
    void
    restore_prefix_cache_lookup(std::uint64_t query_blocks,
                                std::uint64_t hit_blocks,
                                std::uint64_t cached_tokens,
                                config::PrefixCachingKeyMode key_mode =
                                    config::PrefixCachingKeyMode::kSession);
    void advance_scheduler_frontier(std::uint64_t scheduled_tokens);
    void
    on_batch_completion(SimTime time, std::uint64_t scheduled_tokens,
                        ClusterType cluster_type = ClusterType::kMonolithic);
    void on_preempted(SimTime time,
                      ClusterType cluster_type = ClusterType::kMonolithic);
    void mark_prefill_transfer_pending();
    void on_kv_cache_transfer_start(SimTime time);
    void on_kv_cache_transfer_complete(SimTime time, std::uint64_t size_bytes);

  private:
    void enter_waiting(SimTime time);
    void leave_waiting(SimTime time);
    void validate_time(SimTime time, const char *context) const;
    void validate_progress() const;

    RequestId request_id_;
    SimTime session_start_at_;
    SimTime arrived_at_;
    SimTime think_time_;
    std::uint64_t initial_num_prefill_tokens_;
    std::uint64_t initial_num_decode_tokens_;
    std::uint64_t num_prefill_tokens_;
    std::uint64_t num_decode_tokens_;
    SessionId session_id_;
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

    SimTime first_scheduled_at_;
    SimTime prefill_completed_at_;
    SimTime first_token_completed_at_;
    SimTime completed_at_;
    SimTime kv_cache_transfer_start_time_;
    SimTime kv_cache_transfer_end_time_;
    SimTime decode_arrived_at_;
    std::uint64_t kv_cache_transfer_size_bytes_ = 0;
    double kv_cache_transfer_time_s_ = 0.0;
    SimTime waiting_since_;
    double cumulative_waiting_time_s_ = 0.0;
    std::uint64_t cached_prefill_tokens_ = 0;
    std::uint64_t prefix_cache_query_blocks_ = 0;
    std::uint64_t prefix_cache_hit_blocks_ = 0;
    config::PrefixCachingKeyMode prefix_cache_key_mode_ =
        config::PrefixCachingKeyMode::kSession;
    bool prefix_cache_lookup_recorded_ = false;
};

} // namespace frontier::entities
