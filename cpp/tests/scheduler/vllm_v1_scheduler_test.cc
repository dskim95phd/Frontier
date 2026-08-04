#include "frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.h"
#include "frontier/execution_time_predictor/fixed_execution_time_predictor.h"
#include "tests/test_support.h"

#include <cstdint>
#include <memory>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace {

using frontier::BatchId;
using frontier::Generation;
using frontier::RequestId;
using frontier::SimTime;
using frontier::config::SchedulerConfig;
using frontier::entities::Batch;
using frontier::entities::Request;
using frontier::entities::RequestBatchSnapshot;
using frontier::request_generator::WorkloadRequest;
using frontier::scheduler::ScheduleResult;
using frontier::scheduler::VllmV1Scheduler;
using frontier::test::expect;
using frontier::test::expect_throws;

SchedulerConfig scheduler_config() {
    SchedulerConfig config;
    config.batch_size_cap = 8;
    config.max_tokens_in_batch = 128;
    config.block_size = 16;
    config.num_blocks = 128;
    return config;
}

std::vector<Request> make_requests(
    const std::vector<std::pair<std::uint64_t, std::uint64_t>> &tokens) {
    std::vector<Request> requests;
    requests.reserve(tokens.size());
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        requests.emplace_back([&]() {
            WorkloadRequest value{};
            value.request_id =
                RequestId{static_cast<RequestId::ValueType>(index)};
            value.session_start_at = SimTime::from_seconds(0.0);
            value.num_prefill_tokens = tokens[index].first;
            value.num_decode_tokens = tokens[index].second;
            value.session_id = frontier::SessionId{};
            value.session_turn_index = std::nullopt;
            return value;
        }());
    }
    return requests;
}

void arrive_all(VllmV1Scheduler &scheduler, std::vector<Request> &requests) {
    for (Request &request : requests) {
        request.on_arrival(request.arrived_at());
        scheduler.add_request(request.id());
    }
}

Batch complete_schedule(VllmV1Scheduler &scheduler,
                        std::vector<Request> &requests,
                        const ScheduleResult &schedule, std::uint64_t batch_id,
                        double completion_time) {
    std::vector<RequestBatchSnapshot> snapshots;
    for (const auto &scheduled : schedule.scheduled_requests) {
        const Request &request =
            requests.at(static_cast<std::size_t>(scheduled.request_id.value()));
        snapshots.push_back([&]() {
            RequestBatchSnapshot value{};
            value.request_id = scheduled.request_id;
            value.scheduled_tokens = scheduled.num_tokens;
            value.runtime_epoch = request.runtime_epoch();
            value.execution_epoch = request.execution_epoch();
            value.processed_tokens = request.num_processed_tokens();
            value.scheduler_frontier = request.scheduler_num_computed_tokens();
            return value;
        }());
    }
    Batch batch{BatchId{batch_id}, schedule.iteration_id, std::move(snapshots),
                schedule.simulation_time, Generation{batch_id + 1}};
    scheduler.mark_batch_started(batch);
    expect(scheduler.on_batch_completed(batch,
                                        SimTime::from_seconds(completion_time)),
           "scheduled batch must mutate at least one request");
    return batch;
}

void test_fcfs_continuous_batch_admission_and_order() {
    auto requests = make_requests({{2, 1}, {2, 1}, {2, 1}});
    SchedulerConfig config = scheduler_config();
    config.batch_size_cap = 2;
    VllmV1Scheduler scheduler{config, requests};
    arrive_all(scheduler, requests);

    const ScheduleResult first = scheduler.schedule(SimTime::from_seconds(0.0));
    expect(first.scheduled_requests.size() == 2,
           "batch size cap must limit initial admissions");
    expect(first.scheduled_requests[0].request_id == RequestId{0} &&
               first.scheduled_requests[1].request_id == RequestId{1},
           "initial admissions must preserve FCFS order");
    static_cast<void>(complete_schedule(scheduler, requests, first, 0, 0.001));

    const ScheduleResult second =
        scheduler.schedule(SimTime::from_seconds(0.001));
    expect(second.scheduled_requests.size() == 1 &&
               second.scheduled_requests[0].request_id == RequestId{2},
           "next waiting request must enter after capacity is released");
}

void test_oversized_unchunked_head_is_skipped_for_follower() {
    auto requests = make_requests({{8, 1}, {2, 1}});
    SchedulerConfig config = scheduler_config();
    config.max_tokens_in_batch = 4;
    config.enable_chunked_prefill = false;
    VllmV1Scheduler scheduler{config, requests};
    arrive_all(scheduler, requests);

    const ScheduleResult schedule =
        scheduler.schedule(SimTime::from_seconds(0.0));
    expect(
        schedule.scheduled_requests.size() == 1 &&
            schedule.scheduled_requests[0].request_id == RequestId{1},
        "oversized head must not block smaller FCFS follower this iteration");
    expect(scheduler.waiting_queue().size() == 1 &&
               scheduler.waiting_queue().front() == RequestId{0},
           "skipped request must remain waiting");
    expect(
        schedule.decisions.size() == 1 &&
            schedule.decisions.front().request_id == RequestId{1},
        "production decision stream must contain only the admitted follower");
}

void test_chunked_prefill_runs_before_new_waiting_work() {
    auto requests = make_requests({{10, 1}, {2, 1}});
    SchedulerConfig config = scheduler_config();
    config.max_tokens_in_batch = 4;
    config.enable_chunked_prefill = true;
    VllmV1Scheduler scheduler{config, requests};
    arrive_all(scheduler, requests);

    const ScheduleResult first = scheduler.schedule(SimTime::from_seconds(0.0));
    expect(first.scheduled_requests.size() == 1 &&
               first.scheduled_requests[0].request_id == RequestId{0} &&
               first.scheduled_requests[0].num_tokens == 4,
           "first long prompt chunk must consume the budget");
    static_cast<void>(complete_schedule(scheduler, requests, first, 0, 0.001));

    const ScheduleResult second =
        scheduler.schedule(SimTime::from_seconds(0.001));
    expect(second.scheduled_requests.size() == 1 &&
               second.scheduled_requests[0].request_id == RequestId{0},
           "running partial prefill must run before waiting request");
    static_cast<void>(complete_schedule(scheduler, requests, second, 1, 0.002));

    const ScheduleResult third =
        scheduler.schedule(SimTime::from_seconds(0.002));
    expect(third.scheduled_requests.size() == 2 &&
               third.scheduled_requests[0].request_id == RequestId{1} &&
               third.scheduled_requests[1].request_id == RequestId{0},
           "new admissions must precede running continuation in emitted batch");
    expect(third.scheduled_requests[0].num_tokens == 2 &&
               third.scheduled_requests[1].num_tokens == 2,
           "remaining budget must mix waiting and running prefill work");
}

void test_decode_frontier_grows_kv_at_boundary() {
    auto requests = make_requests({{4, 2}});
    SchedulerConfig config = scheduler_config();
    config.block_size = 4;
    config.num_blocks = 2;
    VllmV1Scheduler scheduler{config, requests};
    arrive_all(scheduler, requests);

    const ScheduleResult prefill =
        scheduler.schedule(SimTime::from_seconds(0.0));
    static_cast<void>(
        complete_schedule(scheduler, requests, prefill, 0, 0.001));
    expect(scheduler.kv_blocks().allocated_blocks(RequestId{0}) == 1,
           "prefill must own one exact-boundary block");

    const ScheduleResult decode =
        scheduler.schedule(SimTime::from_seconds(0.001));
    expect(decode.scheduled_requests.size() == 1 &&
               decode.scheduled_requests[0].num_tokens == 1,
           "one subsequent decode token is scheduled per iteration");
    expect(scheduler.kv_blocks().allocated_blocks(RequestId{0}) == 2,
           "first scheduled decode step must grow KV beyond boundary");
    static_cast<void>(complete_schedule(scheduler, requests, decode, 1, 0.002));
    expect(requests[0].completed() && scheduler.kv_blocks().empty(),
           "completion must free all KV blocks");
}

void test_preemption_rolls_back_same_iteration_victim() {
    auto requests = make_requests({{3, 3}, {4, 2}});
    SchedulerConfig config = scheduler_config();
    config.block_size = 4;
    config.num_blocks = 2;
    config.max_tokens_in_batch = 8;
    config.enable_preemption = true;
    VllmV1Scheduler scheduler{config, requests};
    arrive_all(scheduler, requests);

    const ScheduleResult prefill =
        scheduler.schedule(SimTime::from_seconds(0.0));
    static_cast<void>(
        complete_schedule(scheduler, requests, prefill, 0, 0.001));

    Batch stale_batch{BatchId{99},
                      frontier::IterationId{99},
                      {
                          [&]() {
                              RequestBatchSnapshot value{};
                              value.request_id = RequestId{0};
                              value.scheduled_tokens = 1;
                              value.runtime_epoch = requests[0].runtime_epoch();
                              value.execution_epoch =
                                  requests[0].execution_epoch();
                              value.processed_tokens =
                                  requests[0].num_processed_tokens();
                              value.scheduler_frontier =
                                  requests[0].scheduler_num_computed_tokens();
                              return value;
                          }(),
                      },
                      SimTime::from_seconds(0.001),
                      Generation{100}};

    const ScheduleResult pressure =
        scheduler.schedule(SimTime::from_seconds(0.001));
    expect(pressure.preempted_count == 1,
           "memory pressure must preempt one FCFS tail victim");
    expect(pressure.scheduled_requests.size() == 1 &&
               pressure.scheduled_requests[0].request_id == RequestId{1},
           "same-iteration victim schedule must be rolled back");
    expect(requests[0].preempted() && requests[0].num_processed_tokens() == 0,
           "victim must reset for recomputation");
    expect(
        pressure.token_budget_after == 7,
        "rolled-back victim token must be refunded before requester schedule");
    static_cast<void>(
        complete_schedule(scheduler, requests, pressure, 1, 0.002));

    const std::uint64_t progress_before_stale =
        requests[0].num_processed_tokens();
    scheduler.mark_batch_started(stale_batch);
    expect(!scheduler.on_batch_completed(stale_batch,
                                         SimTime::from_seconds(0.0025)),
           "epoch-mismatched stale completion must be ignored");
    expect(requests[0].num_processed_tokens() == progress_before_stale,
           "stale batch completion must not mutate request progress");

    std::uint64_t batch_id = 2;
    double time = 0.002;
    while (!scheduler.idle() && batch_id < 10) {
        const ScheduleResult next =
            scheduler.schedule(SimTime::from_seconds(time));
        expect(!next.scheduled_requests.empty(),
               "preempted request must eventually make progress");
        time += 0.001;
        static_cast<void>(
            complete_schedule(scheduler, requests, next, batch_id++, time));
    }
    expect(scheduler.idle() && requests[0].completed() &&
               requests[1].completed(),
           "recompute preemption must recover to successful quiescence");
}

void test_mixed_stale_and_valid_request_snapshots_are_applied_per_request() {
    auto requests = make_requests({{4, 2}, {4, 2}});
    VllmV1Scheduler scheduler{scheduler_config(), requests};
    arrive_all(scheduler, requests);

    const ScheduleResult schedule =
        scheduler.schedule(SimTime::from_seconds(0.0));
    expect(schedule.scheduled_requests.size() == 2,
           "fixture must schedule both requests");

    std::vector<RequestBatchSnapshot> snapshots;
    for (const auto &scheduled : schedule.scheduled_requests) {
        const Request &request =
            requests.at(static_cast<std::size_t>(scheduled.request_id.value()));
        snapshots.push_back([&]() {
            RequestBatchSnapshot value{};
            value.request_id = scheduled.request_id;
            value.scheduled_tokens = scheduled.num_tokens;
            value.runtime_epoch = request.runtime_epoch();
            value.execution_epoch = request.execution_epoch();
            value.processed_tokens = request.num_processed_tokens();
            value.scheduler_frontier = request.scheduler_num_computed_tokens();
            return value;
        }());
    }
    Batch batch{BatchId{99}, schedule.iteration_id, std::move(snapshots),
                schedule.simulation_time, Generation{100}};

    requests[0].on_batch_completion(SimTime::from_seconds(0.0005), 1);
    const std::uint64_t stale_request_progress =
        requests[0].num_processed_tokens();

    scheduler.mark_batch_started(batch);
    expect(scheduler.on_batch_completed(batch, SimTime::from_seconds(0.001)),
           "a valid request snapshot in a mixed batch must still be applied");
    expect(requests[0].num_processed_tokens() == stale_request_progress,
           "a stale request snapshot must not mutate that request");
    expect(
        requests[1].num_processed_tokens() ==
            requests[1].num_prefill_tokens() + 1,
        "a valid request snapshot in the same batch must mutate its request");
    expect(batch.completed(), "a generation-valid mixed batch must complete");
}

void test_requester_self_preemption_and_disabled_pressure() {
    const auto reach_second_decode_boundary =
        [](VllmV1Scheduler &scheduler, std::vector<Request> &requests) {
            const ScheduleResult prefill =
                scheduler.schedule(SimTime::from_seconds(0.0));
            static_cast<void>(
                complete_schedule(scheduler, requests, prefill, 0, 0.001));
            const ScheduleResult first_decode =
                scheduler.schedule(SimTime::from_seconds(0.001));
            static_cast<void>(
                complete_schedule(scheduler, requests, first_decode, 1, 0.002));
        };

    auto preempting_requests = make_requests({{3, 3}});
    SchedulerConfig preempting_config = scheduler_config();
    preempting_config.block_size = 4;
    preempting_config.num_blocks = 1;
    preempting_config.enable_preemption = true;
    VllmV1Scheduler preempting{preempting_config, preempting_requests};
    arrive_all(preempting, preempting_requests);
    reach_second_decode_boundary(preempting, preempting_requests);

    const ScheduleResult self_pressure =
        preempting.schedule(SimTime::from_seconds(0.002));
    expect(self_pressure.preempted_count == 1 &&
               self_pressure.scheduled_requests.empty(),
           "a sole requester must self-preempt when KV cannot grow");
    expect(preempting_requests[0].preempted() &&
               preempting_requests[0].num_processed_tokens() == 0 &&
               preempting.waiting_queue().front() == RequestId{0} &&
               preempting.kv_blocks().empty(),
           "self-preemption must reset recompute state and free KV");

    auto disabled_requests = make_requests({{3, 3}});
    SchedulerConfig disabled_config = preempting_config;
    disabled_config.enable_preemption = false;
    VllmV1Scheduler disabled{disabled_config, disabled_requests};
    arrive_all(disabled, disabled_requests);
    reach_second_decode_boundary(disabled, disabled_requests);

    const ScheduleResult disabled_pressure =
        disabled.schedule(SimTime::from_seconds(0.002));
    expect(disabled_pressure.preempted_count == 0 &&
               disabled_pressure.scheduled_requests.empty(),
           "disabled preemption must leave the pressure iteration empty");
    expect(!disabled_requests[0].preempted() &&
               disabled_requests[0].num_processed_tokens() == 5 &&
               disabled.running_order().front() == RequestId{0} &&
               disabled.kv_blocks().allocated_blocks(RequestId{0}) == 1,
           "disabled pressure must preserve running progress and allocation");
}

void test_session_prefix_hit_and_all_hit_demotion() {
    std::vector<Request> requests;
    for (const auto &[prefill, decode] :
         std::vector<std::pair<std::uint64_t, std::uint64_t>>{
             {4, 1}, {4, 1}, {5, 1}}) {
        const std::size_t index = requests.size();
        WorkloadRequest value{};
        value.request_id = RequestId{index};
        value.session_start_at = SimTime::from_seconds(0.0);
        value.num_prefill_tokens = prefill;
        value.num_decode_tokens = decode;
        value.session_id = frontier::SessionId{7};
        value.session_turn_index = index;
        requests.emplace_back(value);
    }
    SchedulerConfig config = scheduler_config();
    config.block_size = 4;
    config.num_blocks = 8;
    VllmV1Scheduler scheduler{
        config, requests,
        frontier::config::PrefixCacheConfig{
            true, frontier::config::PrefixCachingKeyMode::kSession}};

    requests[0].on_arrival(requests[0].arrived_at());
    scheduler.add_request(RequestId{0});
    const ScheduleResult first = scheduler.schedule(SimTime::from_seconds(0.0));
    static_cast<void>(complete_schedule(scheduler, requests, first, 0, 0.001));
    expect(requests[0].completed() && scheduler.idle(),
           "first cache-producing request must release ownership");

    requests[1].on_arrival(requests[1].arrived_at());
    scheduler.add_request(RequestId{1});
    const ScheduleResult all_hit =
        scheduler.schedule(SimTime::from_seconds(0.001));
    expect(all_hit.scheduled_requests.size() == 1 &&
               all_hit.scheduled_requests[0].num_tokens == 4 &&
               requests[1].cached_prefill_tokens() == 0 &&
               requests[1].prefix_cache_query_blocks() == 1 &&
               requests[1].prefix_cache_hit_blocks() == 0,
           "a fully block-aligned all-hit prompt must recompute its final "
           "block");
    static_cast<void>(
        complete_schedule(scheduler, requests, all_hit, 1, 0.002));

    requests[2].on_arrival(requests[2].arrived_at());
    scheduler.add_request(RequestId{2});
    const ScheduleResult partial =
        scheduler.schedule(SimTime::from_seconds(0.002));
    expect(partial.scheduled_requests.size() == 1 &&
               partial.scheduled_requests[0].num_tokens == 1 &&
               requests[2].num_processed_tokens() == 4 &&
               requests[2].scheduler_num_computed_tokens() == 5 &&
               requests[2].cached_prefill_tokens() == 4,
           "a partial suffix must restore both request frontiers and schedule "
           "only new work");
    static_cast<void>(
        complete_schedule(scheduler, requests, partial, 2, 0.003));
    expect(scheduler.idle() && scheduler.prefix_cache_stats().hit_blocks == 1,
           "zero-ref cache contents must not prevent scheduler quiescence");
}

void test_preemption_reentry_uses_resident_free_cache_once() {
    std::vector<Request> requests;
    for (std::uint64_t index = 0; index < 2; ++index) {
        WorkloadRequest workload{};
        workload.request_id = RequestId{index};
        workload.session_start_at = SimTime::from_seconds(0.0);
        workload.num_prefill_tokens = 5;
        workload.num_decode_tokens = 5;
        workload.session_id = frontier::SessionId{11 + index};
        workload.session_turn_index = 0;
        requests.emplace_back(workload);
    }

    SchedulerConfig config = scheduler_config();
    config.block_size = 4;
    config.num_blocks = 5;
    config.enable_preemption = true;
    VllmV1Scheduler scheduler{
        config, requests,
        frontier::config::PrefixCacheConfig{
            true, frontier::config::PrefixCachingKeyMode::kSession}};
    arrive_all(scheduler, requests);

    double time = 0.0;
    for (std::uint64_t batch_id = 0; batch_id < 4; ++batch_id) {
        const ScheduleResult schedule =
            scheduler.schedule(SimTime::from_seconds(time));
        expect(schedule.scheduled_requests.size() == 2,
               "both requests must progress to nine committed tokens");
        time += 0.001;
        static_cast<void>(
            complete_schedule(scheduler, requests, schedule, batch_id, time));
    }
    expect(requests[0].num_processed_tokens() == 9 &&
               requests[1].num_processed_tokens() == 9,
           "fixture must commit four decode tokens for both requests");
    const ScheduleResult pressure =
        scheduler.schedule(SimTime::from_seconds(time));
    expect(pressure.preempted_count == 1 &&
               pressure.scheduled_requests.size() == 1 &&
               pressure.scheduled_requests[0].request_id == RequestId{1} &&
               requests[0].preempted(),
           "the FCFS tail victim must be preempted after reserving its "
           "uncomputed suffix block");
    expect(requests[0].num_prefill_tokens() == 9 &&
               requests[0].num_decode_tokens() == 1 &&
               scheduler.kv_blocks().available_blocks() == 2 &&
               scheduler.kv_blocks().gpu_cache_valid_prefix_blocks(
                   frontier::SessionId{11}) == 2,
           "preemption must convert committed decode progress to replay "
           "prefill while retaining both complete GPU blocks");
    time += 0.001;
    static_cast<void>(
        complete_schedule(scheduler, requests, pressure, 4, time));

    const ScheduleResult reentry =
        scheduler.schedule(SimTime::from_seconds(time + 0.001));
    expect(reentry.scheduled_requests.size() == 1 &&
               reentry.scheduled_requests[0].num_tokens == 1 &&
               requests[0].num_processed_tokens() == 8 &&
               requests[0].scheduler_num_computed_tokens() == 9 &&
               requests[0].cached_prefill_tokens() == 0,
           "reentry must restore both replay-prefill blocks without "
           "overwriting the request's first-lookup metrics");
    expect(scheduler.prefix_cache_stats().successful_admissions == 3 &&
               scheduler.prefix_cache_stats().query_blocks == 4 &&
               scheduler.prefix_cache_stats().hit_blocks == 2,
           "system metrics must count the exact cached replay admission");
    time += 0.002;
    static_cast<void>(complete_schedule(scheduler, requests, reentry, 5, time));
    expect(requests[0].completed() && scheduler.idle(),
           "one replay token and one remaining decode token must complete the "
           "victim");
}

void test_cache_disabled_reentry_replays_decode_as_prefill() {
    auto requests = make_requests({{5, 5}, {5, 5}});
    SchedulerConfig config = scheduler_config();
    config.block_size = 4;
    config.num_blocks = 5;
    config.enable_preemption = true;
    VllmV1Scheduler scheduler{config, requests};
    arrive_all(scheduler, requests);

    double time = 0.0;
    for (std::uint64_t batch_id = 0; batch_id < 4; ++batch_id) {
        const ScheduleResult schedule =
            scheduler.schedule(SimTime::from_seconds(time));
        time += 0.001;
        static_cast<void>(
            complete_schedule(scheduler, requests, schedule, batch_id, time));
    }
    const ScheduleResult pressure =
        scheduler.schedule(SimTime::from_seconds(time));
    expect(pressure.preempted_count == 1 && requests[0].preempted() &&
               requests[0].num_prefill_tokens() == 9 &&
               requests[0].num_decode_tokens() == 1,
           "cache-disabled victim must still reconstruct the replay split");
    time += 0.001;
    static_cast<void>(
        complete_schedule(scheduler, requests, pressure, 4, time));

    const ScheduleResult replay =
        scheduler.schedule(SimTime::from_seconds(time + 0.001));
    expect(replay.scheduled_requests.size() == 1 &&
               replay.scheduled_requests[0].request_id == RequestId{0} &&
               replay.scheduled_requests[0].num_tokens == 9,
           "without resident cache the committed nine-token context must be "
           "recomputed as one prefill");
    time += 0.002;
    static_cast<void>(complete_schedule(scheduler, requests, replay, 5, time));
    expect(requests[0].completed() && scheduler.idle(),
           "cache-disabled replay must finish without four extra decode "
           "iterations");
}

void test_tiered_prefix_plan_first_gap_and_demotion() {
    const auto gpu_only = frontier::scheduler::build_contiguous_tiered_prefix_plan(
        4, 3, 2, 16, 64);
    expect(gpu_only.cpu_begin_block == gpu_only.cpu_end_block &&
               gpu_only.hit_frontier_blocks == 3,
           "GPU-only hit must not create a CPU restore range");

    const auto cpu_only = frontier::scheduler::build_contiguous_tiered_prefix_plan(
        4, 0, 3, 16, 64);
    expect(cpu_only.cpu_begin_block == 0 && cpu_only.cpu_end_block == 3 &&
               cpu_only.hit_frontier_blocks == 3,
           "CPU-only hit must create one contiguous restore range");

    const auto mixed = frontier::scheduler::build_contiguous_tiered_prefix_plan(
        5, 2, 4, 16, 80);
    expect(mixed.cpu_begin_block == 2 && mixed.cpu_end_block == 4 &&
               mixed.hit_frontier_blocks == 4,
           "GPU prefix and CPU suffix must form one mixed frontier");

    const auto full = frontier::scheduler::build_contiguous_tiered_prefix_plan(
        3, 0, 3, 16, 48);
    expect(full.hit_frontier_blocks == 2 && full.cpu_end_block == 2,
           "fully cached prompt must demote its last full block");

    frontier::entities::StagedCpuKVCacheRestore staged{};
    staged.cpu_begin_block = 2;
    staged.cpu_end_block = 4;
    staged.query_blocks = 5;
    staged.block_size = 16;
    staged.prompt_tokens = 80;
    expect(frontier::scheduler::revalidate_contiguous_tiered_prefix_frontier(
               1, staged) == 1 &&
               frontier::scheduler::revalidate_contiguous_tiered_prefix_frontier(
                   3, staged) == 4 &&
               frontier::scheduler::revalidate_contiguous_tiered_prefix_frontier(
                   5, staged) == 4,
           "admission revalidation must stop at a gap and demote all-hit");
}

std::vector<Request> make_cpu_restore_requests(
    const std::vector<std::pair<std::uint64_t, std::int64_t>> &inputs) {
    std::vector<Request> requests;
    for (std::size_t index = 0; index < inputs.size(); ++index) {
        WorkloadRequest workload{};
        workload.request_id = RequestId{index};
        workload.session_start_at = SimTime::from_seconds(0.0);
        workload.num_prefill_tokens = inputs[index].first;
        workload.num_decode_tokens = 1;
        workload.session_id = frontier::SessionId{inputs[index].second};
        workload.session_turn_index = 0;
        requests.emplace_back(workload);
    }
    return requests;
}

frontier::config::ResolvedCpuKVCacheTargetConfig cpu_target_config() {
    frontier::config::ResolvedCpuKVCacheTargetConfig config{};
    config.enabled = true;
    config.capacity_bytes = 1'600;
    config.capacity_blocks = 16;
    config.bytes_per_block = 100;
    config.d2h_bandwidth_gbps = 1.0;
    config.d2h_latency_ms = 0.1;
    config.h2d_bandwidth_gbps = 1.0;
    config.h2d_latency_ms = 0.1;
    return config;
}

void seed_cpu_prefix(VllmV1Scheduler &scheduler, frontier::SessionId session,
                     std::uint64_t blocks) {
    auto *manager = scheduler.cpu_kv_cache_manager();
    expect(manager != nullptr, "PREFILL scheduler must own a CPU manager");
    const auto reservation = manager->reserve_offload(
        session, frontier::CpuOffloadGeneration{1}, blocks,
        SimTime::from_seconds(0.0));
    expect(reservation.requires_transfer() &&
               manager->commit_offload(reservation.reservation_id,
                                       SimTime::from_seconds(0.001)),
           "test CPU prefix must commit");
}

void test_deferred_restore_lifecycle_and_atomic_admission() {
    auto requests = make_cpu_restore_requests({{48, 10}});
    SchedulerConfig config = scheduler_config();
    config.block_size = 16;
    config.num_blocks = 8;
    frontier::config::PrefixCacheConfig prefix{};
    prefix.enabled = true;
    frontier::config::ParallelismConfig parallelism{};
    frontier::entities::Replica replica{
        frontier::ReplicaId{0}, parallelism, frontier::config::ModelConfig{}};
    auto predictor = std::make_shared<
        frontier::execution_time_predictor::FixedExecutionTimePredictor>(
        frontier::config::FixedExecutionModelConfig{});
    VllmV1Scheduler scheduler{config, requests, predictor, replica,
                              frontier::DataParallelId{0},
                              frontier::ClusterType::kPrefill, prefix,
                              cpu_target_config()};
    seed_cpu_prefix(scheduler, frontier::SessionId{10}, 2);
    arrive_all(scheduler, requests);

    const auto suspended = scheduler.schedule(SimTime::from_seconds(1.0));
    expect(suspended.scheduled_requests.empty() &&
               scheduler.pending_cpu_restore_count() == 1 &&
               scheduler.allocated_kv_blocks() == 0,
           "restore start must suspend waiting work without GPU allocation");
    auto starts = scheduler.drain_auxiliary_events();
    expect(starts.size() == 1,
           "restore suspension must emit exactly one start event");
    const auto start = std::get<frontier::CpuKVCacheRestoreStartPayload>(
        starts.front().payload);
    scheduler.on_cpu_kv_cache_restore_start(start.transfer_id,
                                            start.generation,
                                            starts.front().time);
    auto ends = scheduler.drain_auxiliary_events();
    expect(ends.size() == 1 && scheduler.allocated_kv_blocks() == 0,
           "in-flight restore must still own no GPU pages");
    const auto end = std::get<frontier::CpuKVCacheRestoreEndPayload>(
        ends.front().payload);
    expect(scheduler.on_cpu_kv_cache_restore_end(
               end.transfer_id, end.generation, ends.front().time) &&
               scheduler.pending_cpu_restore_count() == 0 &&
               scheduler.staged_cpu_restore_count() == 1 &&
               scheduler.allocated_kv_blocks() == 0,
           "restore completion must stage payload and release its CPU lease");

    const auto admitted = scheduler.schedule(ends.front().time);
    expect(admitted.scheduled_requests.size() == 1 &&
               admitted.scheduled_requests.front().num_tokens == 16 &&
               requests[0].cached_prefill_tokens() == 32 &&
               scheduler.staged_cpu_restore_count() == 0 &&
               scheduler.allocated_kv_blocks() == 3,
           "staged restore and GPU allocation must publish atomically");
}

void test_restore_pending_does_not_head_of_line_block_gpu_admission() {
    auto requests = make_cpu_restore_requests({{48, 20}, {16, 21}});
    SchedulerConfig config = scheduler_config();
    config.block_size = 16;
    config.num_blocks = 1;
    frontier::config::PrefixCacheConfig prefix{};
    prefix.enabled = true;
    frontier::config::ParallelismConfig parallelism{};
    frontier::entities::Replica replica{
        frontier::ReplicaId{0}, parallelism, frontier::config::ModelConfig{}};
    auto predictor = std::make_shared<
        frontier::execution_time_predictor::FixedExecutionTimePredictor>(
        frontier::config::FixedExecutionModelConfig{});
    VllmV1Scheduler scheduler{config, requests, predictor, replica,
                              frontier::DataParallelId{0},
                              frontier::ClusterType::kPrefill, prefix,
                              cpu_target_config()};
    seed_cpu_prefix(scheduler, frontier::SessionId{20}, 2);
    arrive_all(scheduler, requests);
    const auto schedule = scheduler.schedule(SimTime::from_seconds(1.0));
    expect(schedule.scheduled_requests.size() == 1 &&
               schedule.scheduled_requests.front().request_id == RequestId{1} &&
               scheduler.pending_cpu_restore_count() == 1 &&
               scheduler.kv_blocks().allocated_blocks(RequestId{0}) == 0 &&
               scheduler.kv_blocks().allocated_blocks(RequestId{1}) == 1,
           "restore-pending head must not consume watermark or block follower");
    auto starts = scheduler.drain_auxiliary_events();
    const auto start = std::get<frontier::CpuKVCacheRestoreStartPayload>(
        starts.front().payload);
    scheduler.on_cpu_kv_cache_restore_start(start.transfer_id,
                                            start.generation,
                                            starts.front().time);
    auto ends = scheduler.drain_auxiliary_events();
    const auto end = std::get<frontier::CpuKVCacheRestoreEndPayload>(
        ends.front().payload);
    expect(scheduler.on_cpu_kv_cache_restore_end(
               end.transfer_id, end.generation, ends.front().time),
           "HOL restore must reach staged state");
    const auto blocked = scheduler.schedule(ends.front().time);
    expect(blocked.scheduled_requests.empty() &&
               scheduler.staged_cpu_restore_count() == 1 &&
               scheduler.kv_blocks().allocated_blocks(RequestId{0}) == 0,
           "temporary GPU pressure must retain staged payload atomically");
    expect(scheduler.cancel_cpu_kv_cache_restore(RequestId{0},
                                                 ends.front().time) &&
               !scheduler.cancel_cpu_kv_cache_restore(RequestId{0},
                                                      ends.front().time) &&
               scheduler.staged_cpu_restore_count() == 0 &&
               std::find(scheduler.waiting_queue().begin(),
                         scheduler.waiting_queue().end(), RequestId{0}) ==
                   scheduler.waiting_queue().end(),
           "staged restore cancellation must remove its runnable entry exactly once");
}

void test_pending_restore_cancellation_removes_start_event() {
    auto requests = make_cpu_restore_requests({{48, 30}});
    SchedulerConfig config = scheduler_config();
    config.block_size = 16;
    config.num_blocks = 4;
    frontier::config::PrefixCacheConfig prefix{};
    prefix.enabled = true;
    frontier::config::ParallelismConfig parallelism{};
    frontier::entities::Replica replica{
        frontier::ReplicaId{0}, parallelism, frontier::config::ModelConfig{}};
    auto predictor = std::make_shared<
        frontier::execution_time_predictor::FixedExecutionTimePredictor>(
        frontier::config::FixedExecutionModelConfig{});
    VllmV1Scheduler scheduler{config, requests, predictor, replica,
                              frontier::DataParallelId{0},
                              frontier::ClusterType::kPrefill, prefix,
                              cpu_target_config()};
    seed_cpu_prefix(scheduler, frontier::SessionId{30}, 2);
    arrive_all(scheduler, requests);

    const SimTime now = SimTime::from_seconds(1.0);
    const auto suspended = scheduler.schedule(now);
    expect(suspended.scheduled_requests.empty() &&
               scheduler.pending_cpu_restore_count() == 1 &&
               scheduler.cpu_kv_cache_manager()
                       ->diagnostics()
                       .active_restore_leases == 1,
           "restore must be pending before cancellation");
    expect(scheduler.cancel_cpu_kv_cache_restore(RequestId{0}, now) &&
               !scheduler.cancel_cpu_kv_cache_restore(RequestId{0}, now) &&
               scheduler.pending_cpu_restore_count() == 0 &&
               scheduler.drain_auxiliary_events().empty() &&
               scheduler.cpu_kv_cache_manager()
                       ->diagnostics()
                       .active_restore_leases == 0 &&
               std::count(scheduler.waiting_queue().begin(),
                          scheduler.waiting_queue().end(), RequestId{0}) == 1,
           "pending cancellation must release its lease, suppress the start event, and requeue once");
}

void test_restore_scheduling_failure_restores_queue_ownership() {
    auto requests = make_cpu_restore_requests({{48, 31}});
    SchedulerConfig config = scheduler_config();
    config.block_size = 16;
    config.num_blocks = 4;
    frontier::config::PrefixCacheConfig prefix{};
    prefix.enabled = true;
    frontier::config::ParallelismConfig parallelism{};
    frontier::entities::Replica replica{
        frontier::ReplicaId{0}, parallelism, frontier::config::ModelConfig{}};
    auto predictor = std::make_shared<
        frontier::execution_time_predictor::FixedExecutionTimePredictor>(
        frontier::config::FixedExecutionModelConfig{});
    auto cpu_config = cpu_target_config();
    cpu_config.bytes_per_block = std::numeric_limits<std::uint64_t>::max();
    VllmV1Scheduler scheduler{config, requests, predictor, replica,
                              frontier::DataParallelId{0},
                              frontier::ClusterType::kPrefill, prefix,
                              cpu_config};
    seed_cpu_prefix(scheduler, frontier::SessionId{31}, 2);
    arrive_all(scheduler, requests);

    expect_throws<frontier::scheduler::SchedulerError>(
        [&]() {
            static_cast<void>(
                scheduler.schedule(SimTime::from_seconds(1.0)));
        },
        "overflowing restore transfer must fail deterministically");
    const auto diagnostics = scheduler.cpu_kv_cache_manager()->diagnostics();
    expect(scheduler.pending_cpu_restore_count() == 0 &&
               scheduler.staged_cpu_restore_count() == 0 &&
               scheduler.allocated_kv_blocks() == 0 &&
               diagnostics.active_restore_leases == 0 &&
               scheduler.waiting_queue().size() == 1 &&
               scheduler.waiting_queue().front() == RequestId{0},
           "restore scheduling failure must release the lease and preserve "
           "queue ownership");
}

void test_restore_completion_failure_cancels_matching_operation() {
    auto requests = make_cpu_restore_requests({{48, 32}});
    SchedulerConfig config = scheduler_config();
    config.block_size = 16;
    config.num_blocks = 4;
    frontier::config::PrefixCacheConfig prefix{};
    prefix.enabled = true;
    frontier::config::ParallelismConfig parallelism{};
    frontier::entities::Replica replica{
        frontier::ReplicaId{0}, parallelism, frontier::config::ModelConfig{}};
    auto predictor = std::make_shared<
        frontier::execution_time_predictor::FixedExecutionTimePredictor>(
        frontier::config::FixedExecutionModelConfig{});
    VllmV1Scheduler scheduler{config, requests, predictor, replica,
                              frontier::DataParallelId{0},
                              frontier::ClusterType::kPrefill, prefix,
                              cpu_target_config()};
    seed_cpu_prefix(scheduler, frontier::SessionId{32}, 2);
    arrive_all(scheduler, requests);
    static_cast<void>(scheduler.schedule(SimTime::from_seconds(1.0)));
    const auto operation = scheduler.cpu_kv_cache_restore_operations().front();
    auto starts = scheduler.drain_auxiliary_events();
    const auto start = std::get<frontier::CpuKVCacheRestoreStartPayload>(
        starts.front().payload);
    scheduler.on_cpu_kv_cache_restore_start(start.transfer_id,
                                            start.generation,
                                            starts.front().time);
    auto ends = scheduler.drain_auxiliary_events();
    const auto end = std::get<frontier::CpuKVCacheRestoreEndPayload>(
        ends.front().payload);
    expect(scheduler.cpu_kv_cache_manager()->release_restore(
               operation.lease_id(), false, ends.front().time),
           "fixture must invalidate the in-flight restore lease");
    expect_throws<frontier::scheduler::SchedulerError>(
        [&]() {
            static_cast<void>(scheduler.on_cpu_kv_cache_restore_end(
                end.transfer_id, end.generation, ends.front().time));
        },
        "terminal lease must fail H2D completion deterministically");
    expect(scheduler.pending_cpu_restore_count() == 0 &&
               scheduler.staged_cpu_restore_count() == 0 &&
               scheduler.allocated_kv_blocks() == 0 &&
               scheduler.cpu_kv_cache_restore_operations().front().state() ==
                   frontier::entities::CpuKVCacheTransferState::kCancelled &&
               !scheduler.on_cpu_kv_cache_restore_end(
                   end.transfer_id, end.generation, ends.front().time),
           "H2D completion failure must cancel only its matching operation once");
}

Batch complete_prefill_schedule(VllmV1Scheduler &scheduler,
                                std::vector<Request> &requests,
                                const ScheduleResult &schedule,
                                double completed_at) {
    std::vector<RequestBatchSnapshot> snapshots;
    for (const auto &scheduled : schedule.scheduled_requests) {
        const Request &value = requests.at(scheduled.request_id.index());
        snapshots.push_back(RequestBatchSnapshot{
            scheduled.request_id, scheduled.num_tokens, value.runtime_epoch(),
            value.execution_epoch(), value.num_processed_tokens(),
            value.scheduler_num_computed_tokens()});
    }
    Batch batch{BatchId{100}, schedule.iteration_id, std::move(snapshots),
                schedule.simulation_time, Generation{100},
                frontier::ReplicaId{0}, frontier::DataParallelId{0}, 1,
                frontier::ClusterType::kPrefill};
    scheduler.mark_batch_started(batch);
    expect(scheduler.on_batch_completed(
               batch, SimTime::from_seconds(completed_at)),
           "PREFILL batch must complete before export");
    return batch;
}

void test_cpu_offload_export_barrier_both_orders() {
    const auto run_order = [](bool decode_first) {
        auto requests = make_cpu_restore_requests({{32, 30}});
        SchedulerConfig config = scheduler_config();
        config.block_size = 16;
        config.num_blocks = 8;
        frontier::config::PrefixCacheConfig prefix{};
        prefix.enabled = true;
        frontier::config::ParallelismConfig parallelism{};
        frontier::entities::Replica replica{
            frontier::ReplicaId{0}, parallelism,
            frontier::config::ModelConfig{}};
        auto predictor = std::make_shared<
            frontier::execution_time_predictor::FixedExecutionTimePredictor>(
            frontier::config::FixedExecutionModelConfig{});
        VllmV1Scheduler scheduler{
            config, requests, predictor, replica, frontier::DataParallelId{0},
            frontier::ClusterType::kPrefill, prefix, cpu_target_config()};
        arrive_all(scheduler, requests);
        const auto scheduled = scheduler.schedule(SimTime::from_seconds(1.0));
        static_cast<void>(
            complete_prefill_schedule(scheduler, requests, scheduled, 1.1));
        expect(scheduler.allocated_kv_blocks() == 2 &&
                   scheduler.prepare_cpu_kv_cache_offload(RequestId{0},
                                                          SimTime::from_seconds(1.1)),
               "real offload must add a CPU export branch");
        auto starts = scheduler.drain_auxiliary_events();
        expect(starts.size() == 1,
               "CPU export must emit one D2H start event");
        const auto start = std::get<frontier::CpuKVCacheOffloadStartPayload>(
            starts.front().payload);

        const auto complete_decode = [&]() {
            const double start_time = decode_first ? 1.10001 : 1.2;
            requests[0].on_kv_cache_transfer_start(
                SimTime::from_seconds(start_time));
            requests[0].on_kv_cache_transfer_complete(
                SimTime::from_seconds(start_time + 0.00001), 1);
            scheduler.complete_kv_transfer(RequestId{0});
        };
        const auto complete_cpu = [&]() {
            scheduler.on_cpu_kv_cache_offload_start(
                start.transfer_id, start.cpu_generation, starts.front().time);
            auto ends = scheduler.drain_auxiliary_events();
            const auto end =
                std::get<frontier::CpuKVCacheOffloadEndPayload>(
                    ends.front().payload);
            expect(scheduler.on_cpu_kv_cache_offload_end(
                       end.transfer_id, end.cpu_generation, ends.front().time),
                   "D2H branch must complete once");
            expect(!scheduler.on_cpu_kv_cache_offload_end(
                       end.transfer_id, end.cpu_generation, ends.front().time),
                   "duplicate D2H completion must be stale");
        };

        if (decode_first) {
            complete_decode();
            expect(scheduler.allocated_kv_blocks() == 2,
                   "decode-first completion must retain CPU source pages");
            complete_cpu();
        } else {
            complete_cpu();
            expect(scheduler.allocated_kv_blocks() == 2,
                   "CPU-first completion must retain decode source pages");
            complete_decode();
        }
        expect(scheduler.allocated_kv_blocks() == 0 &&
                   scheduler.cpu_kv_cache_manager()
                           ->committed_frontier_blocks(frontier::SessionId{30}) ==
                       2 &&
                   scheduler.pending_kv_transfer_count() == 0,
               "last export branch must release GPU and preserve CPU snapshot");
    };
    run_order(true);
    run_order(false);
}

void test_noop_cpu_offload_adds_no_export_branch() {
    auto requests = make_cpu_restore_requests({{8, 40}});
    SchedulerConfig config = scheduler_config();
    config.block_size = 16;
    config.num_blocks = 4;
    frontier::config::PrefixCacheConfig prefix{};
    prefix.enabled = true;
    frontier::config::ParallelismConfig parallelism{};
    frontier::entities::Replica replica{
        frontier::ReplicaId{0}, parallelism, frontier::config::ModelConfig{}};
    auto predictor = std::make_shared<
        frontier::execution_time_predictor::FixedExecutionTimePredictor>(
        frontier::config::FixedExecutionModelConfig{});
    VllmV1Scheduler scheduler{config, requests, predictor, replica,
                              frontier::DataParallelId{0},
                              frontier::ClusterType::kPrefill, prefix,
                              cpu_target_config()};
    arrive_all(scheduler, requests);
    const auto scheduled = scheduler.schedule(SimTime::from_seconds(1.0));
    static_cast<void>(
        complete_prefill_schedule(scheduler, requests, scheduled, 1.1));
    expect(!scheduler.prepare_cpu_kv_cache_offload(
               RequestId{0}, SimTime::from_seconds(1.1)) &&
               scheduler.drain_auxiliary_events().empty(),
           "partial-block snapshot must add no CPU branch or event");
    requests[0].on_kv_cache_transfer_start(SimTime::from_seconds(1.11));
    requests[0].on_kv_cache_transfer_complete(SimTime::from_seconds(1.12), 1);
    scheduler.complete_kv_transfer(RequestId{0});
    expect(scheduler.allocated_kv_blocks() == 0,
           "decode-only export must release source pages immediately");
}

void test_cpu_offload_scheduling_failure_rolls_back_cpu_only() {
    auto requests = make_cpu_restore_requests({{32, 50}});
    SchedulerConfig config = scheduler_config();
    config.block_size = 16;
    config.num_blocks = 4;
    frontier::config::PrefixCacheConfig prefix{};
    prefix.enabled = true;
    frontier::config::ParallelismConfig parallelism{};
    frontier::entities::Replica replica{
        frontier::ReplicaId{0}, parallelism, frontier::config::ModelConfig{}};
    auto predictor = std::make_shared<
        frontier::execution_time_predictor::FixedExecutionTimePredictor>(
        frontier::config::FixedExecutionModelConfig{});
    auto cpu = cpu_target_config();
    cpu.bytes_per_block = std::numeric_limits<std::uint64_t>::max();
    VllmV1Scheduler scheduler{config, requests, predictor, replica,
                              frontier::DataParallelId{0},
                              frontier::ClusterType::kPrefill, prefix, cpu};
    arrive_all(scheduler, requests);
    const auto scheduled = scheduler.schedule(SimTime::from_seconds(1.0));
    static_cast<void>(
        complete_prefill_schedule(scheduler, requests, scheduled, 1.1));
    expect_throws<frontier::scheduler::SchedulerError>(
        [&] {
            static_cast<void>(scheduler.prepare_cpu_kv_cache_offload(
                RequestId{0}, SimTime::from_seconds(1.1)));
        },
        "offload size overflow must fail deterministically");
    const auto diagnostics = scheduler.cpu_kv_cache_manager()->diagnostics();
    expect(diagnostics.reserved_blocks == 0 &&
               diagnostics.active_reservations == 0 &&
               scheduler.pending_kv_transfer_count() == 1,
           "D2H scheduling failure must roll back only the CPU branch");
    requests[0].on_kv_cache_transfer_start(SimTime::from_seconds(1.11));
    requests[0].on_kv_cache_transfer_complete(SimTime::from_seconds(1.12), 1);
    scheduler.complete_kv_transfer(RequestId{0});
    expect(scheduler.allocated_kv_blocks() == 0,
           "decode completion must still release PREFILL pages after CPU failure");
}

void test_cpu_offload_completion_failure_closes_cpu_branch() {
    auto requests = make_cpu_restore_requests({{32, 51}});
    SchedulerConfig config = scheduler_config();
    config.block_size = 16;
    config.num_blocks = 4;
    frontier::config::PrefixCacheConfig prefix{};
    prefix.enabled = true;
    frontier::config::ParallelismConfig parallelism{};
    frontier::entities::Replica replica{
        frontier::ReplicaId{0}, parallelism, frontier::config::ModelConfig{}};
    auto predictor = std::make_shared<
        frontier::execution_time_predictor::FixedExecutionTimePredictor>(
        frontier::config::FixedExecutionModelConfig{});
    VllmV1Scheduler scheduler{config, requests, predictor, replica,
                              frontier::DataParallelId{0},
                              frontier::ClusterType::kPrefill, prefix,
                              cpu_target_config()};
    arrive_all(scheduler, requests);
    const auto scheduled = scheduler.schedule(SimTime::from_seconds(1.0));
    static_cast<void>(
        complete_prefill_schedule(scheduler, requests, scheduled, 1.1));
    expect(scheduler.prepare_cpu_kv_cache_offload(
               RequestId{0}, SimTime::from_seconds(1.1)),
           "completion-failure fixture must create a CPU branch");
    const auto operation = scheduler.cpu_kv_cache_offload_operations().front();
    auto starts = scheduler.drain_auxiliary_events();
    const auto start = std::get<frontier::CpuKVCacheOffloadStartPayload>(
        starts.front().payload);
    scheduler.on_cpu_kv_cache_offload_start(
        start.transfer_id, start.cpu_generation, starts.front().time);
    auto ends = scheduler.drain_auxiliary_events();
    const auto end = std::get<frontier::CpuKVCacheOffloadEndPayload>(
        ends.front().payload);
    expect(scheduler.cpu_kv_cache_manager()->abort_offload(
               operation.reservation_id()),
           "fixture must invalidate the in-flight reservation");
    expect_throws<frontier::scheduler::SchedulerError>(
        [&]() {
            static_cast<void>(scheduler.on_cpu_kv_cache_offload_end(
                end.transfer_id, end.cpu_generation, ends.front().time));
        },
        "terminal reservation must fail D2H completion deterministically");
    expect(scheduler.cpu_kv_cache_manager()
                       ->diagnostics()
                       .active_reservations == 0 &&
               scheduler.cpu_kv_cache_offload_operations().front().state() ==
                   frontier::entities::CpuKVCacheTransferState::kCancelled &&
               scheduler.pending_kv_transfer_count() == 1 &&
               scheduler.allocated_kv_blocks() == 2,
           "D2H completion failure must close only the CPU export branch");

    requests[0].on_kv_cache_transfer_start(SimTime::from_seconds(1.2));
    requests[0].on_kv_cache_transfer_complete(SimTime::from_seconds(1.21), 1);
    scheduler.complete_kv_transfer(RequestId{0});
    expect(scheduler.allocated_kv_blocks() == 0,
           "decode branch must release source pages after D2H completion failure");
}

} // namespace

int main() {
    int failures = 0;
    failures +=
        frontier::test::run("FCFS continuous batch admission and order",
                            test_fcfs_continuous_batch_admission_and_order);
    failures += frontier::test::run(
        "oversized unchunked head is skipped",
        test_oversized_unchunked_head_is_skipped_for_follower);
    failures +=
        frontier::test::run("chunked prefill runs before waiting work",
                            test_chunked_prefill_runs_before_new_waiting_work);
    failures += frontier::test::run("decode frontier grows KV at boundary",
                                    test_decode_frontier_grows_kv_at_boundary);
    failures +=
        frontier::test::run("preemption rolls back same-iteration victim",
                            test_preemption_rolls_back_same_iteration_victim);
    failures += frontier::test::run(
        "mixed stale and valid snapshots apply per request",
        test_mixed_stale_and_valid_request_snapshots_are_applied_per_request);
    failures += frontier::test::run(
        "requester self-preemption and disabled pressure",
        test_requester_self_preemption_and_disabled_pressure);
    failures +=
        frontier::test::run("session prefix hit and all-hit demotion",
                            test_session_prefix_hit_and_all_hit_demotion);
    failures += frontier::test::run(
        "preemption reentry uses resident free cache once",
        test_preemption_reentry_uses_resident_free_cache_once);
    failures += frontier::test::run(
        "cache-disabled reentry replays decode as prefill",
        test_cache_disabled_reentry_replays_decode_as_prefill);
    failures += frontier::test::run(
        "tiered prefix plan first gap and demotion",
        test_tiered_prefix_plan_first_gap_and_demotion);
    failures += frontier::test::run(
        "deferred restore lifecycle and atomic admission",
        test_deferred_restore_lifecycle_and_atomic_admission);
    failures += frontier::test::run(
        "restore pending avoids GPU head-of-line blocking",
        test_restore_pending_does_not_head_of_line_block_gpu_admission);
    failures += frontier::test::run(
        "pending CPU restore cancellation cleanup",
        test_pending_restore_cancellation_removes_start_event);
    failures += frontier::test::run(
        "CPU restore scheduling failure cleanup",
        test_restore_scheduling_failure_restores_queue_ownership);
    failures += frontier::test::run(
        "CPU restore completion failure cleanup",
        test_restore_completion_failure_cancels_matching_operation);
    failures += frontier::test::run(
        "CPU offload export barrier both orders",
        test_cpu_offload_export_barrier_both_orders);
    failures += frontier::test::run(
        "no-op CPU offload adds no export branch",
        test_noop_cpu_offload_adds_no_export_branch);
    failures += frontier::test::run(
        "CPU offload scheduling failure rollback",
        test_cpu_offload_scheduling_failure_rolls_back_cpu_only);
    failures += frontier::test::run(
        "CPU offload completion failure rollback",
        test_cpu_offload_completion_failure_closes_cpu_branch);
    return failures == 0 ? 0 : 1;
}
