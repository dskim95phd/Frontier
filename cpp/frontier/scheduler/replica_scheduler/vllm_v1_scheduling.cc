#include "frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>
#include <utility>

namespace frontier::scheduler {
namespace {

std::uint64_t checked_size(std::size_t value, const char *context) {
    if (value > std::numeric_limits<std::uint64_t>::max()) {
        throw SchedulerError(context);
    }
    return static_cast<std::uint64_t>(value);
}

} // namespace

ScheduleResult VllmV1Scheduler::schedule_requests(SimTime time) {
    materialize_terminal_releases_before_iteration();
    validate_policy_state();

    ScheduleResult result = [&]() {
        ScheduleResult value{};
        value.iteration_id =
            iteration_ids_.next("scheduler iteration ID space exhausted");
        value.simulation_time = time;
        value.token_budget_before = config_.max_tokens_in_batch;
        value.token_budget_after = config_.max_tokens_in_batch;
        value.available_blocks_before = kv_blocks_.available_blocks();
        value.available_blocks_after = kv_blocks_.available_blocks();
        value.waiting_count_before = checked_size(
            waiting_count(), "waiting queue size overflows uint64");
        value.waiting_count_after = 0;
        value.running_count_before = checked_size(
            running_.size(), "running queue size overflows uint64");
        value.running_count_after = 0;
        value.preempted_count = 0;
        value.decisions = {};
        value.scheduled_requests = {};
        return value;
    }();

    if (terminal_release_followup_poll_pending_) {
        result.waiting_count_after = result.waiting_count_before;
        result.running_count_after = result.running_count_before;
        return result;
    }

    std::uint64_t token_budget = config_.max_tokens_in_batch;
    std::vector<ScheduledRequest> running_scheduled;
    std::vector<RequestId> preempted_requests;

    std::size_t running_index = 0;
    while (running_index < running_.size() && token_budget > 0) {
        const RequestId request_id = running_[running_index];
        entities::Request &value = request(request_id);
        if (value.completed()) {
            ++running_index;
            continue;
        }
        // A decode step consumes the token sampled by the previous one, so a
        // request in its decode phase may hold only one in-flight batch.  With
        // PP that spaces a request's decodes pipeline_parallel_size steps
        // apart, which is what vLLM V1 enforces through
        // Request.next_decode_eligible_step.
        //
        // A prefill chunk carries no such dependency across the whole model:
        // chunk N+1 reads chunk N's KV only for the layers of the stage it is
        // entering, and chunk N has already written those by the time it left
        // that stage.  Consecutive chunks of one request therefore pipeline one
        // stage apart, and both production engines schedule them that way --
        // vLLM V1 lets a prefill request occupy several in-flight microbatches
        // and stops only when next_num_tokens() reaches zero, and SGLang calls
        // the same thing chunked pipeline parallelism.  Skipping an active
        // prefill here instead would leave PP - 1 stages idle whenever one
        // request's chunks are the only schedulable work.
        //
        // Multiple in-flight chunks are already representable:
        // mark_batch_started exempts PREFILL from the single-batch check,
        // kv_accounted_tokens reserves blocks against the optimistic scheduler
        // frontier, and apply_batch_completion reconstructs a chunk's expected
        // processed tokens from its own snapshot rather than the current
        // frontier.
        if (value.is_prefill_complete() && request_is_active(request_id)) {
            ++running_index;
            continue;
        }
        std::uint64_t num_tokens = next_num_tokens(value);
        if (!value.is_prefill_complete() &&
            config_.long_prefill_token_threshold > 0) {
            num_tokens =
                std::min(num_tokens, config_.long_prefill_token_threshold);
        }
        num_tokens = std::min(num_tokens, token_budget);
        if (num_tokens == 0) {
            ++running_index;
            continue;
        }

        const std::size_t preempted_before = preempted_requests.size();
        if (request_is_active(request_id)) {
            // An extra chunk for a request that already holds a batch exists to
            // occupy a pipeline stage that would otherwise idle, so it takes
            // free blocks only.  Evicting another request to run ahead would
            // trade committed work for a slot that is a bonus rather than a
            // requirement, and it would fire preemption an iteration earlier
            // than the sequential order of the same chunks ever does.  Skip to
            // the next request instead of breaking: a later one may still fit.
            const std::uint64_t accounted = kv_accounted_tokens(value);
            if (!kv_blocks_.can_reserve(request_id, accounted, num_tokens)) {
                ++running_index;
                continue;
            }
            kv_blocks_.reserve(request_id, accounted, num_tokens);
        } else if (!try_reserve_with_preemption(
                       request_id, num_tokens, time, preempted_requests,
                       running_scheduled, token_budget, result)) {
            break;
        }

        value.advance_scheduler_frontier(num_tokens);
        running_scheduled.push_back([&]() {
            ScheduledRequest value{};
            value.request_id = request_id;
            value.num_tokens = num_tokens;
            value.admitted_from_waiting = false;
            return value;
        }());
        token_budget -= num_tokens;
        result.token_budget_after = token_budget;
        result.decisions.push_back([&]() {
            SchedulerDecision value{};
            value.type = SchedulerDecisionType::kRunningScheduled;
            value.request_id = request_id;
            value.num_tokens = num_tokens;
            value.token_budget_after = token_budget;
            value.available_blocks_after = kv_blocks_.available_blocks();
            return value;
        }());
        ++running_index;

        if (preempted_requests.size() != preempted_before) {
            validate_policy_state();
        }
    }

    std::vector<ScheduledRequest> waiting_scheduled;
    if (preempted_requests.empty() &&
        pending_terminal_release_iterations_.empty()) {
        std::deque<RequestId> queue = take_admission_queue();
        std::deque<RequestId> skipped;

        while (!queue.empty() && token_budget > 0) {
            if (running_.size() >= config_.batch_size_cap) {
                break;
            }
            const RequestId request_id = queue.front();
            entities::Request &value = request(request_id);
            const bool prefill_request =
                cluster_type() == ClusterType::kPrefill &&
                !value.is_prefill_complete();
            const std::uint64_t full_sequence_tokens =
                prefill_request ? value.num_prefill_tokens() : 0;
            if (prefill_request &&
                !kv_blocks_.full_sequence_fits_empty(full_sequence_tokens)) {
                queue.insert(queue.end(), skipped.begin(), skipped.end());
                restore_admission_queue(std::move(queue));
                throw SchedulerError("PREFILL prompt plus watermark exceeds "
                                     "empty replica KV capacity");
            }
            kv_cache::PrefixLookupResult prefix_lookup{};
            const auto staged_position = staged_cpu_restores_.find(request_id);
            const bool has_staged_restore =
                staged_position != staged_cpu_restores_.end();
            if (has_staged_restore) {
                const std::uint64_t reusable =
                    revalidate_staged_frontier(value, staged_position->second);
                prefix_lookup.query_blocks =
                    staged_position->second.query_blocks;
                prefix_lookup.hit_blocks = reusable;
                prefix_lookup.cached_tokens =
                    reusable * kv_blocks_.block_size();
            } else if (cpu_kv_cache_ != nullptr &&
                       !value.is_prefill_complete()) {
                const entities::TieredPrefixPlan plan =
                    build_tiered_prefix_plan(value);
                if (plan.cpu_end_block > plan.cpu_begin_block ||
                    plan.restore_kda_snapshot) {
                    bool committed_for_restore = false;
                    try {
                        if (prefill_request &&
                            kv_blocks_.request_committed_blocks(request_id) ==
                                0) {
                            if (!kv_blocks_.can_commit(request_id,
                                                       value.session_id(),
                                                       full_sequence_tokens)) {
                                break;
                            }
                            kv_blocks_.commit_virtual(request_id,
                                                      value.session_id(),
                                                      full_sequence_tokens);
                            committed_for_restore = true;
                        }
                        suspend_for_cpu_restore(request_id, plan, time);
                    } catch (...) {
                        if (committed_for_restore) {
                            kv_blocks_.release_commitment(request_id);
                        }
                        queue.insert(queue.end(), skipped.begin(),
                                     skipped.end());
                        restore_admission_queue(std::move(queue));
                        throw;
                    }
                    queue.pop_front();
                    continue;
                }
                prefix_lookup.query_blocks = plan.query_blocks;
                prefix_lookup.hit_blocks = plan.hit_frontier_blocks;
                prefix_lookup.cached_tokens =
                    plan.hit_frontier_blocks * kv_blocks_.block_size();
            } else if (kv_blocks_.prefix_cache_enabled() &&
                       !value.is_prefill_complete()) {
                prefix_lookup = kv_blocks_.lookup(value);
                if (prefix_lookup.cached_tokens == value.num_prefill_tokens() &&
                    prefix_lookup.hit_blocks > 0) {
                    --prefix_lookup.hit_blocks;
                    prefix_lookup.cached_tokens -= kv_blocks_.block_size();
                }
            }
            std::uint64_t num_tokens =
                value.is_prefill_complete()
                    ? next_num_tokens(value)
                    : value.num_prefill_tokens() - prefix_lookup.cached_tokens;
            if (!value.is_prefill_complete() &&
                config_.long_prefill_token_threshold > 0) {
                num_tokens =
                    std::min(num_tokens, config_.long_prefill_token_threshold);
            }

            if (!config_.enable_chunked_prefill &&
                !value.is_prefill_complete() && num_tokens > token_budget) {
                queue.pop_front();
                skipped.push_back(request_id);
                continue;
            }

            num_tokens = std::min(num_tokens, token_budget);
            if (num_tokens == 0) {
                queue.pop_front();
                continue;
            }
            const std::uint64_t accounted = prefix_lookup.cached_tokens > 0
                                                ? prefix_lookup.cached_tokens
                                                : kv_accounted_tokens(value);
            const bool can_reserve =
                has_staged_restore ? kv_blocks_.can_admit_tiered(
                                         request_id, value.session_id(),
                                         prefix_lookup.hit_blocks, num_tokens,
                                         full_sequence_tokens)
                : kv_blocks_.prefix_cache_enabled() &&
                        !value.is_prefill_complete()
                    ? kv_blocks_.can_admit(request_id, value.session_id(),
                                           prefix_lookup.cached_tokens,
                                           num_tokens, full_sequence_tokens)
                    : kv_blocks_.can_reserve(request_id, accounted, num_tokens,
                                             full_sequence_tokens);
            if (!can_reserve) {
                break;
            }

            queue.pop_front();
            try {
                if (has_staged_restore) {
                    const std::uint64_t current_gpu =
                        kv_blocks_.gpu_cache_valid_prefix_blocks(
                            value.session_id());
                    kv_blocks_.admit_tiered(request_id, value.session_id(),
                                            prefix_lookup.hit_blocks,
                                            num_tokens, full_sequence_tokens);
                    value.restore_prefix_cache_lookup(
                        prefix_lookup.query_blocks, prefix_lookup.hit_blocks,
                        prefix_lookup.cached_tokens);
                    kv_blocks_.record_successful_admission(
                        prefix_lookup.query_blocks, prefix_lookup.hit_blocks);
                    const entities::StagedCpuKVCacheRestore staged =
                        staged_position->second;
                    const std::uint64_t cpu_query =
                        staged.query_blocks > current_gpu
                            ? staged.query_blocks - current_gpu
                            : 0;
                    const std::uint64_t cpu_used =
                        prefix_lookup.hit_blocks > current_gpu
                            ? prefix_lookup.hit_blocks - current_gpu
                            : 0;
                    const std::uint64_t gpu_used =
                        std::min(current_gpu, prefix_lookup.hit_blocks);
                    const bool first_cpu_admission =
                        value.record_cpu_prefix_admission(
                            gpu_used, cpu_query, cpu_used,
                            cpu_used * staged.block_size);
                    if (first_cpu_admission) {
                        cpu_kv_cache_->record_successful_lookup(
                            kv_cache::CpuPrefixLookupResult{cpu_query,
                                                            cpu_used},
                            value.session_id());
                    }
                } else if (kv_blocks_.prefix_cache_enabled() &&
                           !value.is_prefill_complete()) {
                    const std::uint64_t current_gpu =
                        kv_blocks_.lookup(value).hit_blocks;
                    kv_blocks_.admit(request_id, value.session_id(),
                                     prefix_lookup.cached_tokens, num_tokens,
                                     full_sequence_tokens);
                    value.restore_prefix_cache_lookup(
                        prefix_lookup.query_blocks, prefix_lookup.hit_blocks,
                        prefix_lookup.cached_tokens);
                    kv_blocks_.record_successful_admission(
                        prefix_lookup.query_blocks, prefix_lookup.hit_blocks);
                    if (cpu_kv_cache_ != nullptr) {
                        const std::uint64_t cpu_query =
                            prefix_lookup.query_blocks > current_gpu
                                ? prefix_lookup.query_blocks - current_gpu
                                : 0;
                        const std::uint64_t gpu_used =
                            std::min(current_gpu, prefix_lookup.hit_blocks);
                        const bool first_cpu_admission =
                            value.record_cpu_prefix_admission(gpu_used,
                                                              cpu_query, 0, 0);
                        if (first_cpu_admission) {
                            cpu_kv_cache_->record_successful_lookup(
                                kv_cache::CpuPrefixLookupResult{cpu_query, 0},
                                value.session_id());
                        }
                    }
                } else {
                    kv_blocks_.reserve(request_id, accounted, num_tokens,
                                       full_sequence_tokens);
                }
                value.on_admitted(time);
                value.advance_scheduler_frontier(num_tokens);
                running_.push_back(request_id);
            } catch (...) {
                // Admission is a single ownership transaction: if any
                // manager/request validation throws after queue removal, give
                // back physical and future blocks and restore queue ownership.
                kv_blocks_.release_commitment(request_id);
                queue.push_front(request_id);
                queue.insert(queue.end(), skipped.begin(), skipped.end());
                restore_admission_queue(std::move(queue));
                throw;
            }
            if (has_staged_restore) {
                staged_cpu_restores_.erase(request_id);
            }
            waiting_scheduled.push_back([&]() {
                ScheduledRequest value{};
                value.request_id = request_id;
                value.num_tokens = num_tokens;
                value.admitted_from_waiting = true;
                return value;
            }());
            token_budget -= num_tokens;
            result.token_budget_after = token_budget;
            result.decisions.push_back([&]() {
                SchedulerDecision value{};
                value.type = SchedulerDecisionType::kAdmission;
                value.request_id = request_id;
                value.num_tokens = num_tokens;
                value.token_budget_after = token_budget;
                value.available_blocks_after = kv_blocks_.available_blocks();
                return value;
            }());
        }

        queue.insert(queue.end(), skipped.begin(), skipped.end());
        restore_admission_queue(std::move(queue));
    }

    result.scheduled_requests.reserve(waiting_scheduled.size() +
                                      running_scheduled.size());
    result.scheduled_requests.insert(result.scheduled_requests.end(),
                                     waiting_scheduled.begin(),
                                     waiting_scheduled.end());
    result.scheduled_requests.insert(result.scheduled_requests.end(),
                                     running_scheduled.begin(),
                                     running_scheduled.end());
    result.token_budget_after = token_budget;
    result.available_blocks_after = kv_blocks_.available_blocks();
    result.waiting_count_after =
        checked_size(waiting_count(), "waiting queue size overflows uint64");
    result.running_count_after =
        checked_size(running_.size(), "running queue size overflows uint64");
    const bool empty_iteration = result.scheduled_requests.empty();
    if (empty_iteration && cluster_type() == ClusterType::kPrefill) {
        for (const RequestId request_id : running_) {
            const entities::Request &value = request(request_id);
            if (!value.completed() && !value.is_prefill_complete() &&
                !request_is_active(request_id) && next_num_tokens(value) == 0) {
                throw SchedulerError(
                    "PREFILL request is stranded behind an exhausted "
                    "scheduler frontier");
            }
        }
    }
    advance_terminal_release_boundary();
    if (empty_iteration) {
        result.available_blocks_after = kv_blocks_.available_blocks();
        result.waiting_count_after = checked_size(
            waiting_count(), "waiting queue size overflows uint64");
        result.running_count_after = checked_size(
            running_.size(), "running queue size overflows uint64");
    }
    validate_policy_state();
    return result;
}

bool VllmV1Scheduler::apply_batch_completion(entities::Batch &batch,
                                             SimTime time) {
    bool has_valid_request = false;
    for (const entities::RequestBatchSnapshot &snapshot : batch.requests()) {
        entities::Request &value = request(snapshot.request_id);
        if (snapshot.scheduler_frontier < snapshot.scheduled_tokens) {
            throw SchedulerError(
                "batch snapshot scheduler frontier underflows");
        }
        const std::uint64_t expected_processed_tokens =
            std::max(snapshot.processed_tokens,
                     snapshot.scheduler_frontier - snapshot.scheduled_tokens);
        // A prefill whose later chunks were already scheduled has a frontier
        // ahead of this chunk's snapshot, so match on what the snapshot itself
        // implies: every token before this chunk must be processed, and the
        // frontier must be at least where it stood when the chunk was issued.
        // Stages run batches in global creation order, so a request's chunks
        // land in the order they were issued and expected_processed_tokens is
        // reached exactly once.  Decode keeps the strict equality -- its
        // frontier never runs ahead of an in-flight batch.
        const bool frontier_matches =
            !value.is_prefill_complete()
                ? value.num_processed_tokens() == expected_processed_tokens &&
                      value.scheduler_num_computed_tokens() >=
                          snapshot.scheduler_frontier
                : value.num_processed_tokens() == snapshot.processed_tokens &&
                      value.scheduler_num_computed_tokens() ==
                          snapshot.scheduler_frontier;
        if (value.completed() ||
            value.runtime_epoch() != snapshot.runtime_epoch ||
            value.execution_epoch() != snapshot.execution_epoch ||
            !frontier_matches) {
            continue;
        }
        value.on_batch_completion(time, snapshot.scheduled_tokens,
                                  cluster_type());
        kv_blocks_.mark_blocks_computed(value);
        if (kv_blocks_.kda_snapshot_enabled()) {
            const std::uint64_t published_blocks =
                std::min(value.num_processed_tokens() / kv_blocks_.block_size(),
                         kv_blocks_.allocated_blocks(value.id()));
            if (!kv_blocks_.publish_kda_snapshot(value.session_id(),
                                                 published_blocks)) {
                throw SchedulerError(
                    "reserved KDA snapshot publication failed");
            }
        }
        has_valid_request = true;
        if (cluster_type() == ClusterType::kPrefill &&
            value.is_prefill_complete()) {
            const auto position =
                std::find(running_.begin(), running_.end(), value.id());
            if (position == running_.end()) {
                throw SchedulerError(
                    "completed prefill is missing from running order");
            }
            running_.erase(position);
            value.mark_prefill_transfer_pending();
            if (!pending_exports_
                     .emplace(value.id(), PrefillExportState{true, false})
                     .second) {
                throw SchedulerError(
                    "request already has pending PREFILL exports");
            }
        } else if (value.completed()) {
            const std::uint64_t extra = extra_terminal_release_iterations();
            if (extra > 0) {
                auto [position, inserted] =
                    pending_terminal_release_iterations_.try_emplace(value.id(),
                                                                     extra);
                if (!inserted) {
                    position->second = std::max(position->second, extra);
                }
            } else {
                free_completed_request(value.id());
            }
        }
    }
    return has_valid_request;
}

} // namespace frontier::scheduler
