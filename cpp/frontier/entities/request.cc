#include "frontier/entities/request.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace frontier::entities {

Request::Request(const request_generator::WorkloadRequest &workload_request)
    : request_id_(workload_request.request_id),
      session_start_at_(workload_request.session_start_at),
      arrived_at_(workload_request.session_start_at),
      think_time_(workload_request.think_time),
      initial_num_prefill_tokens_(workload_request.num_prefill_tokens),
      initial_num_decode_tokens_(workload_request.num_decode_tokens),
      num_prefill_tokens_(workload_request.num_prefill_tokens),
      num_decode_tokens_(workload_request.num_decode_tokens),
      session_id_(workload_request.session_id),
      session_turn_index_(workload_request.session_turn_index) {
    if (!think_time_.valid()) {
        throw RequestError("request think time must be finite and nonnegative");
    }
    if (!session_start_at_.valid() &&
        session_start_at_.seconds() != SimTime{}.seconds()) {
        throw RequestError(
            "request session start must be finite and nonnegative");
    }
    if (num_prefill_tokens_ == 0 || num_decode_tokens_ == 0) {
        throw RequestError("request token counts must be positive");
    }
    if (num_prefill_tokens_ >
        std::numeric_limits<std::uint64_t>::max() - num_decode_tokens_) {
        throw RequestError("request total token count overflows uint64");
    }
    if (session_turn_index_.has_value() && !session_id_.valid()) {
        throw RequestError("session_turn_index requires session_id");
    }
}

std::uint64_t Request::num_processed_prefill_tokens() const noexcept {
    return std::min(num_processed_tokens_, num_prefill_tokens_);
}

void Request::set_pending_arrival(SimTime time) {
    validate_time(time, "pending arrival");
    if (state_ != RequestState::kPending) {
        throw RequestError("only a pending request arrival can be rescheduled");
    }
    arrived_at_ = time;
}

std::uint64_t Request::num_processed_decode_tokens() const noexcept {
    return num_processed_tokens_ > num_prefill_tokens_
               ? num_processed_tokens_ - num_prefill_tokens_
               : 0;
}

std::uint64_t Request::remaining_decode_tokens() const noexcept {
    return num_decode_tokens_ - num_processed_decode_tokens();
}

void Request::validate_time(SimTime time, const char *context) const {
    if (!std::isfinite(time.seconds()) || time.seconds() < 0.0) {
        throw RequestError(std::string{context} +
                           " time must be finite and nonnegative");
    }
}

void Request::validate_progress() const {
    if (num_prefill_tokens_ < initial_num_prefill_tokens_ ||
        total_tokens() !=
            initial_num_prefill_tokens_ + initial_num_decode_tokens_) {
        throw RequestError("request replay token split is invalid");
    }
    if (num_processed_tokens_ > total_tokens()) {
        throw RequestError("request-visible progress exceeds total tokens");
    }
    if (scheduler_num_computed_tokens_ > total_tokens()) {
        throw RequestError("scheduler frontier exceeds total tokens");
    }
    if (preemption_recomputed_prefill_tokens_ > scheduled_prefill_tokens_) {
        throw RequestError(
            "preemption-recomputed PREFILL exceeds scheduled PREFILL");
    }
    if (completed() && num_processed_tokens_ != total_tokens()) {
        throw RequestError("completed request has incomplete token progress");
    }
}

void Request::enter_waiting(SimTime time) {
    validate_time(time, "waiting-entry");
    if (waiting_since_.valid()) {
        throw RequestError("request entered waiting queue twice");
    }
    waiting_since_ = time;
    state_ = RequestState::kWaiting;
}

void Request::leave_waiting(SimTime time) {
    validate_time(time, "waiting-exit");
    if (!waiting_since_.valid()) {
        throw RequestError("request left waiting queue without entering it");
    }
    if (time.seconds() < waiting_since_.seconds()) {
        throw RequestError("waiting exit precedes waiting entry");
    }
    cumulative_waiting_time_s_ += time.seconds() - waiting_since_.seconds();
    waiting_since_ = SimTime{};
}

void Request::on_arrival(SimTime time, ClusterType cluster_type) {
    if (cluster_type == ClusterType::kDecode) {
        if (state_ != RequestState::kTransferInFlight ||
            !kv_cache_transfer_end_time_.valid() ||
            time != kv_cache_transfer_end_time_) {
            throw RequestError(
                "decode arrival requires a completed KV transfer");
        }
        decode_arrived_at_ = time;
        enter_waiting(time);
        return;
    }
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
    if (!first_scheduled_at_.valid()) {
        first_scheduled_at_ = time;
    }
    state_ = RequestState::kRunning;
    preempted_ = false;
}

void Request::restore_prefix_cache_lookup(
    std::uint64_t query_blocks, std::uint64_t hit_blocks,
    std::uint64_t cached_tokens, config::PrefixCachingKeyMode key_mode) {
    if (state_ != RequestState::kWaiting || is_prefill_complete_ ||
        num_processed_tokens_ != 0 || scheduler_num_computed_tokens_ != 0) {
        throw RequestError(
            "prefix-cache restoration requires a zero-frontier waiting "
            "prefill request");
    }
    if (hit_blocks > query_blocks || cached_tokens >= num_prefill_tokens_) {
        throw RequestError("prefix-cache restoration exceeds prompt bounds");
    }
    num_processed_tokens_ = cached_tokens;
    scheduler_num_computed_tokens_ = cached_tokens;
    if (!prefix_cache_lookup_recorded_) {
        cached_prefill_tokens_ = cached_tokens;
        prefix_cache_query_blocks_ = query_blocks;
        prefix_cache_hit_blocks_ = hit_blocks;
        gpu_prefix_hit_blocks_ = hit_blocks;
        prefix_cache_key_mode_ = key_mode;
        prefix_cache_lookup_recorded_ = true;
    }
    validate_progress();
}

void Request::record_cpu_restore_transfer(std::uint64_t blocks,
                                          std::uint64_t bytes,
                                          double queue_time_ms,
                                          double service_time_ms) {
    if (!std::isfinite(queue_time_ms) || queue_time_ms < 0.0 ||
        !std::isfinite(service_time_ms) || service_time_ms < 0.0 ||
        cpu_restore_transferred_blocks_ != 0 || cpu_restore_bytes_ != 0) {
        throw RequestError("invalid or duplicate CPU restore transfer metrics");
    }
    cpu_restore_transferred_blocks_ = blocks;
    cpu_restore_bytes_ = bytes;
    cpu_restore_queue_time_s_ = queue_time_ms / 1e3;
    cpu_restore_service_time_s_ = service_time_ms / 1e3;
}

void Request::record_cpu_prefix_admission(
    std::uint64_t gpu_hit_blocks, std::uint64_t cpu_query_blocks,
    std::uint64_t cpu_consumed_blocks, std::uint64_t cpu_restored_tokens) {
    if (cpu_prefix_admission_recorded_) {
        return;
    }
    if (cpu_consumed_blocks > cpu_query_blocks ||
        cpu_consumed_blocks > cpu_restore_transferred_blocks_ ||
        cpu_restored_tokens > num_prefill_tokens_) {
        throw RequestError("invalid or duplicate CPU prefix admission metrics");
    }
    gpu_prefix_hit_blocks_ = gpu_hit_blocks;
    cpu_prefix_query_blocks_ = cpu_query_blocks;
    cpu_prefix_hit_blocks_ = cpu_consumed_blocks;
    cpu_restore_consumed_blocks_ = cpu_consumed_blocks;
    cpu_restored_tokens_ = cpu_restored_tokens;
    cpu_prefix_admission_recorded_ = true;
}

void Request::record_cpu_offload_transfer(std::uint64_t bytes,
                                          double queue_time_ms,
                                          double service_time_ms) {
    if (!std::isfinite(queue_time_ms) || queue_time_ms < 0.0 ||
        !std::isfinite(service_time_ms) || service_time_ms < 0.0 ||
        cpu_offload_bytes_ != 0) {
        throw RequestError("invalid or duplicate CPU offload transfer metrics");
    }
    cpu_offload_bytes_ = bytes;
    cpu_offload_queue_time_s_ = queue_time_ms / 1e3;
    cpu_offload_service_time_s_ = service_time_ms / 1e3;
}

void Request::advance_scheduler_frontier(std::uint64_t scheduled_tokens) {
    if (scheduled_tokens == 0) {
        throw RequestError("scheduled token count must be positive");
    }
    if (scheduled_tokens > total_tokens()) {
        throw RequestError("scheduled token count exceeds request total");
    }
    if (scheduler_num_computed_tokens_ > total_tokens() - scheduled_tokens) {
        throw RequestError("scheduled token count exceeds request total");
    }
    scheduler_num_computed_tokens_ += scheduled_tokens;
    validate_progress();
}

void Request::on_batch_completion(SimTime time, std::uint64_t scheduled_tokens,
                                  ClusterType cluster_type) {
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
            throw RequestError("prefill batch exceeds remaining prompt tokens");
        }
        if (scheduled_prefill_tokens_ >
            std::numeric_limits<std::uint64_t>::max() - scheduled_tokens) {
            throw RequestError("scheduled PREFILL token count overflows uint64");
        }
        if (prefill_recompute_pending_ &&
            preemption_recomputed_prefill_tokens_ >
                std::numeric_limits<std::uint64_t>::max() - scheduled_tokens) {
            throw RequestError(
                "recomputed PREFILL token count overflows uint64");
        }
        scheduled_prefill_tokens_ += scheduled_tokens;
        if (prefill_recompute_pending_) {
            preemption_recomputed_prefill_tokens_ += scheduled_tokens;
        }
        num_processed_tokens_ += scheduled_tokens;
        if (num_processed_tokens_ == num_prefill_tokens_) {
            is_prefill_complete_ = true;
            // All PREFILL work after the preemption has now been accounted for;
            // a later decode preemption does not retroactively classify decode
            // work as recomputed PREFILL.
            prefill_recompute_pending_ = false;
            if (!prefill_completed_at_.valid()) {
                prefill_completed_at_ = time;
            }
            if (cluster_type == ClusterType::kMonolithic) {
                // Frontier's monolithic Python path exposes the first generated
                // token at the same boundary as the final prefill chunk.
                ++num_processed_tokens_;
                if (!first_token_completed_at_.valid()) {
                    first_token_completed_at_ = time;
                }
            }
        }
    } else {
        const std::uint64_t remaining = total_tokens() - num_processed_tokens_;
        if (scheduled_tokens > remaining) {
            throw RequestError("decode batch exceeds remaining output tokens");
        }
        num_processed_tokens_ += scheduled_tokens;
        if (!first_token_completed_at_.valid() &&
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

void Request::on_preempted(SimTime time, ClusterType cluster_type) {
    validate_time(time, "preemption");
    if (state_ != RequestState::kRunning || completed()) {
        throw RequestError("only a running request can be preempted");
    }
    tokens_at_preemption_.push_back(scheduler_num_computed_tokens_);
    ++preemption_count_;
    ++runtime_epoch_;
    ++execution_epoch_;
    if (cluster_type != ClusterType::kDecode) {
        // PREFILL preemption requires replaying the prompt.  In monolithic
        // mode this also covers a decode preemption because committed decode
        // context is folded back into the replay-PREFILL boundary below.
        prefill_recompute_pending_ = true;
    }
    if (cluster_type == ClusterType::kDecode) {
        // Production Python vLLM V1 recompute preemption resets both token
        // frontiers to zero even in the unified DECODE cluster, while retaining
        // the prefill-complete phase marker.  The imported KV is discarded and
        // subsequent decode iterations rebuild the full request frontier.
        num_processed_tokens_ = 0;
        scheduler_num_computed_tokens_ = 0;
        is_prefill_complete_ = true;
    } else {
        if (cluster_type == ClusterType::kMonolithic) {
            const std::uint64_t total = total_tokens();
            // Only committed tokens are replayable. Keeping the larger of the
            // current prefill boundary and committed progress preserves a
            // partial/restarted prefill, while reclassifying generated decode
            // tokens as parallel replay-prefill context.
            num_prefill_tokens_ =
                std::max(num_prefill_tokens_, num_processed_tokens_);
            num_decode_tokens_ = total - num_prefill_tokens_;
        }
        num_processed_tokens_ = 0;
        scheduler_num_computed_tokens_ = 0;
        is_prefill_complete_ = false;
    }
    preempted_ = true;
    enter_waiting(time);
    validate_progress();
}

void Request::mark_prefill_transfer_pending() {
    if (state_ != RequestState::kRunning || !is_prefill_complete_ ||
        num_processed_tokens_ != num_prefill_tokens_) {
        throw RequestError(
            "only a fully prefetched running request can await transfer");
    }
    state_ = RequestState::kTransferPending;
}

void Request::on_kv_cache_transfer_start(SimTime time) {
    validate_time(time, "kv-transfer-start");
    if (state_ != RequestState::kTransferPending ||
        kv_cache_transfer_start_time_.valid()) {
        throw RequestError("KV transfer start requires one pending transfer");
    }
    if (!prefill_completed_at_.valid() ||
        time.seconds() < prefill_completed_at_.seconds()) {
        throw RequestError("KV transfer start precedes prefill completion");
    }
    kv_cache_transfer_start_time_ = time;
    state_ = RequestState::kTransferInFlight;
}

void Request::on_kv_cache_transfer_complete(SimTime time,
                                            std::uint64_t size_bytes) {
    validate_time(time, "kv-transfer-completion");
    if (state_ != RequestState::kTransferInFlight ||
        !kv_cache_transfer_start_time_.valid() ||
        kv_cache_transfer_end_time_.valid()) {
        throw RequestError(
            "KV transfer completion requires one in-flight transfer");
    }
    if (time.seconds() < kv_cache_transfer_start_time_.seconds()) {
        throw RequestError("KV transfer completion precedes transfer start");
    }
    kv_cache_transfer_end_time_ = time;
    kv_cache_transfer_size_bytes_ = size_bytes;
    kv_cache_transfer_time_s_ =
        time.seconds() - kv_cache_transfer_start_time_.seconds();
}

} // namespace frontier::entities
