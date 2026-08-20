#include "frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>
#include <utility>

namespace frontier::scheduler {

void VllmV1Scheduler::free_completed_request(RequestId request_id) {
    entities::Request &value = request(request_id);
    if (!value.completed()) {
        throw SchedulerError(
            "terminal release references an incomplete request");
    }
    release_request_kv(request_id);
    const auto position =
        std::find(running_.begin(), running_.end(), request_id);
    if (position == running_.end()) {
        throw SchedulerError(
            "terminal release request is missing from running order");
    }
    running_.erase(position);
}

void VllmV1Scheduler::release_request_kv(RequestId request_id) {
    if (retired_session_requests_.erase(request_id) != 0) {
        static_cast<void>(
            kv_blocks_.discard_session(request(request_id).session_id()));
    }
    static_cast<void>(kv_blocks_.free(request_id));
}

void VllmV1Scheduler::discard_tiered_prefix_cache_session(
    SessionId session_id) {
    BaseReplicaScheduler::discard_tiered_prefix_cache_session(session_id);
    const auto retire = [&](RequestId request_id) {
        if (request(request_id).session_id() == session_id) {
            retired_session_requests_.insert(request_id);
        }
    };
    for (const RequestId request_id : preempted_) {
        retire(request_id);
    }
    for (const RequestId request_id : restored_ready_) {
        retire(request_id);
    }
    for (const RequestId request_id : waiting_) {
        retire(request_id);
    }
    for (const RequestId request_id : running_) {
        retire(request_id);
    }
    for (const auto &[request_id, state] : pending_exports_) {
        static_cast<void>(state);
        retire(request_id);
    }
    for (const auto &[request_id, transfer_id] : pending_cpu_restores_) {
        static_cast<void>(transfer_id);
        retire(request_id);
    }
    for (const auto &[request_id, staged] : staged_cpu_restores_) {
        static_cast<void>(staged);
        retire(request_id);
    }
    for (const auto &[request_id, remaining] :
         pending_terminal_release_iterations_) {
        static_cast<void>(remaining);
        retire(request_id);
    }
}

void VllmV1Scheduler::materialize_terminal_releases_before_iteration() {
    if (pending_terminal_release_iterations_.empty() ||
        has_visible_waiting_requests()) {
        return;
    }
    const std::uint64_t threshold = iteration_start_release_threshold();
    std::vector<RequestId> ready;
    for (const auto &[request_id, remaining] :
         pending_terminal_release_iterations_) {
        if (remaining <= threshold) {
            ready.push_back(request_id);
        }
    }
    std::sort(ready.begin(), ready.end(), [](RequestId left, RequestId right) {
        return left.value() < right.value();
    });
    for (const RequestId request_id : ready) {
        pending_terminal_release_iterations_.erase(request_id);
        waiting_sensitive_release_extensions_.erase(request_id);
        free_completed_request(request_id);
    }
}

void VllmV1Scheduler::advance_terminal_release_boundary() {
    if (pending_terminal_release_iterations_.empty()) {
        return;
    }
    const std::uint64_t threshold = iteration_start_release_threshold();
    std::vector<RequestId> ready;
    for (auto &[request_id, remaining] : pending_terminal_release_iterations_) {
        if (remaining <= threshold && has_visible_waiting_requests() &&
            waiting_sensitive_release_extensions_.find(request_id) ==
                waiting_sensitive_release_extensions_.end()) {
            waiting_sensitive_release_extensions_.insert(request_id);
            remaining = 1;
            continue;
        }
        if (remaining <= 1) {
            ready.push_back(request_id);
        } else {
            --remaining;
        }
    }
    std::sort(ready.begin(), ready.end(), [](RequestId left, RequestId right) {
        return left.value() < right.value();
    });
    for (const RequestId request_id : ready) {
        pending_terminal_release_iterations_.erase(request_id);
        waiting_sensitive_release_extensions_.erase(request_id);
        free_completed_request(request_id);
    }
    terminal_release_followup_poll_pending_ =
        !pending_terminal_release_iterations_.empty() ||
        has_visible_waiting_requests();
}

void VllmV1Scheduler::preempt_request(RequestId victim, SimTime time,
                                      std::vector<RequestId> &newly_preempted,
                                      ScheduleResult &result) {
    entities::Request &value = request(victim);
    const auto running_position =
        std::find(running_.begin(), running_.end(), victim);
    if (running_position == running_.end()) {
        throw SchedulerError("preemption victim is not running");
    }
    running_.erase(running_position);
    static_cast<void>(kv_blocks_.free(victim));
    value.on_preempted(time, cluster_type());
    waiting_.push_front(victim);
    newly_preempted.push_back(victim);
    ++result.preempted_count;
    result.decisions.push_back([&]() {
        SchedulerDecision value{};
        value.type = SchedulerDecisionType::kPreempted;
        value.request_id = victim;
        value.num_tokens = 0;
        value.token_budget_after = result.token_budget_after;
        value.available_blocks_after = kv_blocks_.available_blocks();
        return value;
    }());
}

void VllmV1Scheduler::rollback_preempted_schedules(
    const std::vector<RequestId> &newly_preempted,
    std::vector<ScheduledRequest> &running_scheduled,
    std::uint64_t &token_budget) {
    if (newly_preempted.empty()) {
        return;
    }
    const std::unordered_set<RequestId, StrongIdHash<RequestId>> ids(
        newly_preempted.begin(), newly_preempted.end());
    std::vector<ScheduledRequest> kept;
    kept.reserve(running_scheduled.size());
    for (const ScheduledRequest &scheduled : running_scheduled) {
        if (ids.find(scheduled.request_id) != ids.end()) {
            if (token_budget > std::numeric_limits<std::uint64_t>::max() -
                                   scheduled.num_tokens) {
                throw SchedulerError("token-budget refund overflows uint64");
            }
            token_budget += scheduled.num_tokens;
        } else {
            kept.push_back(scheduled);
        }
    }
    running_scheduled = std::move(kept);
}

bool VllmV1Scheduler::try_reserve_with_preemption(
    RequestId requester, std::uint64_t scheduled_tokens, SimTime time,
    std::vector<RequestId> &newly_preempted,
    std::vector<ScheduledRequest> &running_scheduled,
    std::uint64_t &token_budget, ScheduleResult &result) {
    while (true) {
        entities::Request &value = request(requester);
        const std::uint64_t accounted = kv_accounted_tokens(value);
        if (kv_blocks_.can_reserve(requester, accounted, scheduled_tokens)) {
            kv_blocks_.reserve(requester, accounted, scheduled_tokens);
            return true;
        }
        if (!config_.enable_preemption) {
            return false;
        }

        const std::size_t preemption_count_before = newly_preempted.size();
        const std::optional<RequestId> victim =
            select_preemption_victim(requester);
        if (!victim.has_value() && cluster_type() == ClusterType::kPrefill) {
            // A PREFILL request owns a full-sequence commitment.  If no other
            // runnable victim can release physical blocks, self-preempting it
            // would discard the very commitment needed to resume the chunk
            // and can livelock a sequential PDD replica.  Stall this running
            // iteration instead; the caller will return false.
            return false;
        }
        preempt_request(victim.value_or(requester), time, newly_preempted,
                        result);
        rollback_preempted_schedules(
            std::vector<RequestId>{
                newly_preempted.begin() +
                    static_cast<std::ptrdiff_t>(preemption_count_before),
                newly_preempted.end()},
            running_scheduled, token_budget);
        result.token_budget_after = token_budget;
        if (!victim.has_value()) {
            return false;
        }
    }
}

std::deque<RequestId> VllmV1Scheduler::take_admission_queue() {
    if (cluster_type() == ClusterType::kDecode) {
        // Python's unified DECODE scheduler has one waiting queue. Newly
        // preempted requests are inserted at its front and keep that order.
        return std::exchange(waiting_, {});
    }

    // A completed CPU restore owns a full-sequence commitment.  Admit it
    // before requests that have not acquired a commitment; otherwise a
    // preempted/fresh head that cannot reserve can strand the restored
    // request behind its own resource ownership.
    std::deque<RequestId> queue = std::exchange(restored_ready_, {});
    // MONOLITHIC and PREFILL otherwise retain Python's two-list priority:
    // preempted requests before newly arrived requests.
    std::deque<RequestId> preempted = std::exchange(preempted_, {});
    queue.insert(queue.end(), preempted.begin(), preempted.end());
    queue.insert(queue.end(), waiting_.begin(), waiting_.end());
    waiting_.clear();
    return queue;
}

void VllmV1Scheduler::restore_admission_queue(std::deque<RequestId> queue) {
    for (const RequestId request_id : queue) {
        if (cluster_type() != ClusterType::kDecode &&
            staged_cpu_restores_.find(request_id) !=
                staged_cpu_restores_.end()) {
            restored_ready_.push_back(request_id);
        } else if (cluster_type() != ClusterType::kDecode &&
                   request(request_id).preempted()) {
            preempted_.push_back(request_id);
        } else {
            waiting_.push_back(request_id);
        }
    }
}

entities::TieredPrefixPlan VllmV1Scheduler::build_tiered_prefix_plan(
    const entities::Request &value) const {
    const std::uint64_t block_size = kv_blocks_.block_size();
    const std::uint64_t prompt_tokens = value.num_prefill_tokens();
    const std::uint64_t query_blocks = prompt_tokens / block_size;
    const kv_cache::PrefixLookupResult gpu = kv_blocks_.lookup(value);
    const kv_cache::CpuPrefixLookupResult cpu =
        cpu_kv_cache_->lookup(value.session_id(), query_blocks);
    entities::TieredPrefixPlan plan = build_contiguous_tiered_prefix_plan(
        query_blocks, gpu.hit_blocks, cpu.hit_blocks, block_size,
        prompt_tokens);
    const bool cpu_has_snapshot =
        cpu_kv_cache_->has_kda_snapshot(value.session_id());
    const bool gpu_has_snapshot =
        kv_blocks_.has_kda_snapshot(value.session_id());
    plan.restore_kda_snapshot =
        cpu_has_snapshot &&
        (!gpu_has_snapshot || kv_blocks_.kda_snapshot_frontier_blocks(
                                  value.session_id()) < plan.cpu_end_block);
    return plan;
}

} // namespace frontier::scheduler
