#include "frontier/kv_cache/replica_kv_cache_manager.h"

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

#include "frontier/entities/request.h"
#include "frontier/request_generator/workload.h"
#include "tests/test_support.h"

namespace {

using frontier::RequestId;
using frontier::SessionId;
using frontier::SimTime;
using frontier::config::PrefixCacheConfig;
using frontier::config::PrefixCachingKeyMode;
using frontier::config::SchedulerConfig;
using frontier::entities::Request;
using frontier::kv_cache::PrefixLookupResult;
using frontier::kv_cache::ReplicaKVCacheManager;
using frontier::kv_cache::ReplicaKVCacheError;
using frontier::request_generator::WorkloadRequest;
using frontier::test::expect;
using frontier::test::expect_throws;

SchedulerConfig scheduler_config(std::uint64_t blocks) {
    SchedulerConfig config{};
    config.block_size = 4;
    config.num_blocks = blocks;
    config.watermark_blocks_fraction = 0.0;
    return config;
}

ReplicaKVCacheManager manager(std::uint64_t blocks, bool prefix_cache = true) {
    return ReplicaKVCacheManager{
        scheduler_config(blocks),
        PrefixCacheConfig{prefix_cache, PrefixCachingKeyMode::kSession}, true};
}

Request make_request(std::uint64_t id, std::uint64_t prefill,
                     std::uint64_t decode, std::uint64_t session) {
    WorkloadRequest workload{};
    workload.request_id = RequestId{id};
    workload.session_start_at = SimTime::from_seconds(0.0);
    workload.num_prefill_tokens = prefill;
    workload.num_decode_tokens = decode;
    workload.session_id = SessionId{session};
    workload.session_turn_index = id;
    return Request{workload};
}

void admit_and_complete(ReplicaKVCacheManager &cache, Request &request,
                        double time) {
    request.on_arrival(request.arrived_at());
    PrefixLookupResult lookup = cache.lookup(request);
    if (lookup.cached_tokens == request.num_prefill_tokens() &&
        lookup.hit_blocks > 0) {
        --lookup.hit_blocks;
        lookup.cached_tokens -= cache.block_size();
    }
    const std::uint64_t scheduled =
        request.num_prefill_tokens() - lookup.cached_tokens;
    expect(cache.can_admit(request.id(), request.session_id(),
                           lookup.cached_tokens, scheduled),
           "fixture request must be analytically admissible");
    cache.admit(request.id(), request.session_id(), lookup.cached_tokens,
                scheduled);
    request.restore_prefix_cache_lookup(lookup.query_blocks, lookup.hit_blocks,
                                        lookup.cached_tokens);
    request.on_admitted(SimTime::from_seconds(time));
    request.advance_scheduler_frontier(scheduled);
    request.on_batch_completion(SimTime::from_seconds(time + 0.001), scheduled);
    cache.mark_blocks_computed(request);
    expect(request.completed(), "one-token decode fixture must complete");
    static_cast<void>(cache.free(request.id()));
}

void test_session_release_and_suffix_lru_eviction() {
    ReplicaKVCacheManager cache = manager(5);
    Request first = make_request(0, 8, 1, 1);
    admit_and_complete(cache, first, 0.0);
    expect(cache.gpu_cache_valid_prefix_blocks(SessionId{1}) == 2 &&
               cache.available_blocks() == 5,
           "release must retain two resident blocks while freeing ownership");

    Request extension = make_request(1, 12, 1, 1);
    const PrefixLookupResult warm = cache.lookup(extension);
    expect(warm.query_blocks == 3 && warm.hit_blocks == 2 &&
               warm.cached_tokens == 8,
           "same-session lookup must return its resident range in O(1)");
    admit_and_complete(cache, extension, 0.01);
    expect(cache.gpu_cache_valid_prefix_blocks(SessionId{1}) == 3,
           "append-only completion must extend the resident range");

    Request second = make_request(2, 12, 1, 2);
    admit_and_complete(cache, second, 0.02);
    expect(cache.gpu_cache_valid_prefix_blocks(SessionId{1}) == 2 &&
               cache.gpu_cache_valid_prefix_blocks(SessionId{2}) == 3 &&
               cache.stats().evicted_blocks == 1,
           "cold allocation must trim only the oldest session suffix needed");

    Request third = make_request(3, 12, 1, 3);
    admit_and_complete(cache, third, 0.03);
    expect(cache.gpu_cache_valid_prefix_blocks(SessionId{1}) == 0 &&
               cache.gpu_cache_valid_prefix_blocks(SessionId{2}) == 2 &&
               cache.gpu_cache_valid_prefix_blocks(SessionId{3}) == 3 &&
               cache.stats().evicted_blocks == 4 &&
               cache.stats().evicted_sessions == 1,
           "range eviction must remove one old session then trim the next");
}

void test_all_hit_recompute_reuses_one_logical_slot() {
    ReplicaKVCacheManager cache = manager(2);
    Request producer = make_request(0, 8, 1, 7);
    admit_and_complete(cache, producer, 0.0);

    Request follower = make_request(1, 8, 1, 7);
    follower.on_arrival(follower.arrived_at());
    PrefixLookupResult lookup = cache.lookup(follower);
    expect(lookup.hit_blocks == 2, "fixture must begin as an all-hit prompt");
    --lookup.hit_blocks;
    lookup.cached_tokens -= cache.block_size();
    expect(cache.can_admit(follower.id(), follower.session_id(),
                           lookup.cached_tokens, 4),
           "all-hit demotion must not require a duplicate physical slot");
    cache.admit(follower.id(), follower.session_id(), lookup.cached_tokens, 4);
    expect(cache.allocated_blocks(follower.id()) == 2 &&
               cache.available_blocks() == 0 &&
               cache.stats().evicted_blocks == 0,
           "analytical recompute must pin the existing logical range");
    static_cast<void>(cache.free(follower.id()));
    expect(cache.gpu_cache_valid_prefix_blocks(SessionId{7}) == 2,
           "aborted recompute must leave the old resident range valid");
}

void test_active_session_is_not_shareable() {
    ReplicaKVCacheManager cache = manager(8);
    Request first = make_request(0, 8, 1, 9);
    first.on_arrival(first.arrived_at());
    const PrefixLookupResult cold = cache.lookup(first);
    cache.admit(first.id(), first.session_id(), cold.cached_tokens, 8);

    Request concurrent = make_request(1, 8, 1, 9);
    concurrent.on_arrival(concurrent.arrived_at());
    const PrefixLookupResult blocked = cache.lookup(concurrent);
    expect(blocked.hit_blocks == 0 &&
               !cache.can_admit(concurrent.id(), concurrent.session_id(), 0, 8),
           "one analytical session cannot have concurrent active turns");
    static_cast<void>(cache.free(first.id()));
}

void test_explicit_session_discard_is_immediate_or_deferred() {
    ReplicaKVCacheManager cache = manager(8);
    Request resident = make_request(0, 8, 1, 21);
    admit_and_complete(cache, resident, 0.0);
    expect(cache.discard_session(SessionId{21}) == 2 &&
               cache.gpu_cache_valid_prefix_blocks(SessionId{21}) == 0 &&
               cache.diagnostics().resident_blocks == 0,
           "inactive session migration must immediately discard all GPU KV");

    Request active = make_request(1, 8, 1, 22);
    active.on_arrival(active.arrived_at());
    cache.admit(active.id(), active.session_id(), 0, 8);
    active.restore_prefix_cache_lookup(2, 0, 0);
    active.on_admitted(SimTime::from_seconds(0.01));
    active.advance_scheduler_frontier(8);
    active.on_batch_completion(SimTime::from_seconds(0.011), 8);
    cache.mark_blocks_computed(active);
    expect(cache.discard_session(SessionId{22}) == 0 &&
               cache.gpu_cache_valid_prefix_blocks(SessionId{22}) == 2,
           "active session migration must defer deletion until release");
    static_cast<void>(cache.free(active.id()));
    expect(cache.gpu_cache_valid_prefix_blocks(SessionId{22}) == 0 &&
               cache.diagnostics().resident_blocks == 0 &&
               cache.stats().evicted_blocks == 4 &&
               cache.stats().evicted_sessions == 2,
           "deferred migration deletion must run atomically at release");
}

void test_partial_prefill_preemption_reuses_resident_range() {
    ReplicaKVCacheManager cache = manager(4);
    Request request = make_request(0, 10, 2, 11);
    request.on_arrival(request.arrived_at());
    cache.admit(request.id(), request.session_id(), 0, 6);
    request.restore_prefix_cache_lookup(2, 0, 0);
    request.on_admitted(SimTime::from_seconds(0.0));
    request.advance_scheduler_frontier(6);
    request.on_batch_completion(SimTime::from_seconds(0.001), 6);
    cache.mark_blocks_computed(request);
    static_cast<void>(cache.free(request.id()));
    request.on_preempted(SimTime::from_seconds(0.002));

    const PrefixLookupResult replay = cache.lookup(request);
    expect(request.num_prefill_tokens() == 10 && replay.query_blocks == 2 &&
               replay.hit_blocks == 1 && replay.cached_tokens == 4,
           "partial-prefill replay must restore its complete resident range");
}

void test_cache_disabled_count_accounting() {
    ReplicaKVCacheManager cache = manager(3, false);
    expect(cache.can_reserve(RequestId{0}, 0, 5),
           "cache-disabled allocation must use analytical capacity counts");
    cache.reserve(RequestId{0}, 0, 5);
    expect(cache.allocated_blocks(RequestId{0}) == 2 &&
               cache.available_blocks() == 1,
           "five tokens require two active blocks");
    static_cast<void>(cache.free(RequestId{0}));
    const auto diagnostics = cache.diagnostics();
    expect(diagnostics.active_blocks == 0 &&
               diagnostics.available_blocks == 3 &&
               diagnostics.resident_blocks == 0,
           "cache-disabled release must return every slot to blank capacity");
}

void test_virtual_full_sequence_commitment_tracks_chunk_materialization() {
    ReplicaKVCacheManager cache = manager(4);
    cache.enable_full_sequence_commitments();
    Request request = make_request(0, 12, 1, 41);

    expect(cache.full_sequence_commitments_enabled() &&
               cache.can_commit(request.id(), request.num_prefill_tokens()),
           "a fitting full prompt must accept a virtual commitment");
    cache.commit_virtual(request.id(), request.session_id(),
                         request.num_prefill_tokens());
    expect(cache.request_committed_blocks(request.id()) == 3 &&
               cache.request_virtual_committed_blocks(request.id()) == 3 &&
               cache.virtual_committed_blocks() == 3 &&
               cache.available_commitment_blocks() == 1,
           "full prompt commitment must reserve three future blocks");

    cache.reserve(request.id(), 0, 4);
    expect(cache.allocated_blocks(request.id()) == 1 &&
               cache.request_committed_blocks(request.id()) == 3 &&
               cache.request_virtual_committed_blocks(request.id()) == 2 &&
               cache.virtual_committed_blocks() == 2 &&
               cache.allocated_blocks(request.id()) +
                       cache.virtual_committed_blocks() ==
                   3,
           "first chunk materialization must consume one virtual block");

    cache.reserve(request.id(), 4, 4);
    expect(cache.allocated_blocks(request.id()) == 2 &&
               cache.request_committed_blocks(request.id()) == 3 &&
               cache.request_virtual_committed_blocks(request.id()) == 1 &&
               cache.virtual_committed_blocks() == 1 &&
               cache.allocated_blocks(request.id()) +
                       cache.virtual_committed_blocks() ==
                   3,
           "second chunk must preserve the physical-plus-future total");

    cache.reserve(request.id(), 8, 4);
    expect(cache.allocated_blocks(request.id()) == 3 &&
               cache.request_committed_blocks(request.id()) == 3 &&
               cache.request_virtual_committed_blocks(request.id()) == 0 &&
               cache.virtual_committed_blocks() == 0 &&
               cache.available_commitment_blocks() == 1,
           "final chunk must materialize the last committed block");
    static_cast<void>(cache.free(request.id()));
    expect(cache.virtual_committed_blocks() == 0 &&
               cache.available_commitment_blocks() == 4,
           "free must release all physical and future commitment state");

    ReplicaKVCacheManager cancelled = manager(4);
    cancelled.enable_full_sequence_commitments();
    Request partial = make_request(1, 12, 1, 42);
    cancelled.commit_virtual(partial.id(), partial.session_id(),
                             partial.num_prefill_tokens());
    cancelled.reserve(partial.id(), 0, 4);
    expect(cancelled.request_virtual_committed_blocks(partial.id()) == 2,
           "partial materialization must leave two future blocks cancellable");
    cancelled.release_commitment(partial.id());
    expect(cancelled.allocated_blocks(partial.id()) == 0 &&
               cancelled.virtual_committed_blocks() == 0 &&
               cancelled.available_commitment_blocks() == 4,
           "cancelling a partial commitment must release future blocks once");
}

void test_virtual_commitment_rejects_another_active_session_owner() {
    ReplicaKVCacheManager cache = manager(8);
    cache.enable_full_sequence_commitments();
    Request owner = make_request(0, 12, 1, 43);
    Request successor = make_request(1, 12, 1, 43);

    cache.commit_virtual(owner.id(), owner.session_id(),
                         owner.num_prefill_tokens());
    expect(cache.session_has_active_request(owner.session_id()),
           "virtual commitment must publish active session ownership");
    expect(cache.can_commit(successor.id(), successor.num_prefill_tokens()) &&
               !cache.can_commit(successor.id(), successor.session_id(),
                                 successor.num_prefill_tokens()),
           "session-aware commitment must reject a successor while the old "
           "request still owns the session");

    cache.release_commitment(owner.id());
    expect(!cache.session_has_active_request(owner.session_id()) &&
               cache.can_commit(successor.id(), successor.session_id(),
                                successor.num_prefill_tokens()),
           "releasing the old commitment must unblock the same-session "
           "successor");
}

void test_empty_fit_is_independent_of_current_commitments() {
    SchedulerConfig config = scheduler_config(10);
    config.watermark_blocks_fraction = 0.1;
    ReplicaKVCacheManager cache{
        config,
        PrefixCacheConfig{true, PrefixCachingKeyMode::kSession}, true};
    cache.enable_full_sequence_commitments();

    Request owner = make_request(0, 24, 1, 44);
    Request waiting = make_request(1, 16, 1, 45);
    cache.commit_virtual(owner.id(), owner.session_id(),
                         owner.num_prefill_tokens());

    expect(cache.full_sequence_fits_empty(waiting.num_prefill_tokens()),
           "a request that fits the fixed empty capacity must remain feasible "
           "while another request owns a commitment");
    expect(!cache.can_commit(waiting.id(), waiting.session_id(),
                             waiting.num_prefill_tokens()),
           "current commitment pressure must block admission without making "
           "the waiting request permanently infeasible");
    expect(!cache.full_sequence_fits_empty(40),
           "a prompt plus watermark larger than fixed capacity must remain "
           "permanently infeasible");
}

void test_kda_admission_reserves_cold_snapshot_capacity() {
    ReplicaKVCacheManager cache = manager(4);
    cache.configure_kda_snapshot(2);
    cache.enable_full_sequence_commitments();

    Request oversized = make_request(0, 12, 1, 46);
    expect(!cache.full_sequence_fits_empty(oversized.num_prefill_tokens()) &&
               !cache.can_admit(oversized.id(), oversized.session_id(), 0, 12,
                                oversized.num_prefill_tokens()),
           "a cold KDA prompt must include its atomic snapshot charge in "
           "empty-fit and admission checks");

    Request fitting = make_request(1, 8, 1, 47);
    expect(cache.full_sequence_fits_empty(fitting.num_prefill_tokens()) &&
               cache.can_admit(fitting.id(), fitting.session_id(), 0, 8,
                               fitting.num_prefill_tokens()),
           "a KDA prompt plus snapshot that exactly fills capacity must fit");
    cache.admit(fitting.id(), fitting.session_id(), 0, 8,
                fitting.num_prefill_tokens());
    expect(cache.has_kda_snapshot(fitting.session_id()) &&
               cache.kda_snapshot_frontier_blocks(fitting.session_id()) == 0 &&
               cache.kda_snapshot_occupied_blocks() == 2 &&
               cache.allocated_blocks(fitting.id()) == 2 &&
               cache.available_blocks() == 0,
           "admission must atomically reserve a cold snapshot before compute");
    expect(cache.publish_kda_snapshot(fitting.session_id(), 2) &&
               cache.kda_snapshot_occupied_blocks() == 2,
           "publishing into a reserved snapshot must not consume more blocks");
    static_cast<void>(cache.free(fitting.id()));

    Request successor = make_request(2, 8, 1, 47);
    expect(cache.can_admit(successor.id(), successor.session_id(), 0, 8,
                           successor.num_prefill_tokens()),
           "an existing session snapshot must not be charged twice");
    cache.admit(successor.id(), successor.session_id(), 0, 8,
                successor.num_prefill_tokens());
    expect(cache.kda_snapshot_occupied_blocks() == 2,
           "same-session admission must reuse the existing snapshot charge");
    static_cast<void>(cache.free(successor.id()));

    ReplicaKVCacheManager pending = manager(4);
    pending.configure_kda_snapshot(2);
    pending.enable_full_sequence_commitments();
    Request restoring = make_request(3, 8, 1, 48);
    expect(pending.can_commit(restoring.id(), restoring.session_id(),
                              restoring.num_prefill_tokens()),
           "a cold asynchronous commitment must reserve KV and snapshot");
    pending.commit_virtual(restoring.id(), restoring.session_id(),
                           restoring.num_prefill_tokens());
    expect(pending.has_kda_snapshot(restoring.session_id()) &&
               pending.kda_snapshot_frontier_blocks(restoring.session_id()) ==
                   0 &&
               pending.kda_snapshot_occupied_blocks() == 2 &&
               pending.virtual_committed_blocks() == 2,
           "virtual admission must atomically hold the cold snapshot charge");
    pending.release_commitment(restoring.id());
    expect(!pending.has_kda_snapshot(restoring.session_id()) &&
               pending.kda_snapshot_occupied_blocks() == 0 &&
               pending.virtual_committed_blocks() == 0 &&
               pending.available_commitment_blocks() == 4,
           "rollback before publication must release the cold snapshot "
           "reservation");
}

void test_failed_virtual_commit_does_not_reserve_kda_snapshot() {
    ReplicaKVCacheManager cache = manager(8);
    cache.enable_full_sequence_commitments();

    // Materialize a three-block legacy prefix before enabling the KDA charge.
    // This represents a pre-materialized/direct-API session whose resident
    // frontier is longer than the next supplied full-sequence commitment.
    Request producer = make_request(0, 12, 1, 49);
    admit_and_complete(cache, producer, 0.0);
    cache.configure_kda_snapshot(2);
    const auto before = cache.diagnostics();
    const auto stats_before = cache.stats();
    const std::uint64_t reusable_frontier_before =
        cache.gpu_cache_valid_prefix_blocks(SessionId{49});

    expect_throws<ReplicaKVCacheError>(
        [&cache] {
            cache.commit_virtual(RequestId{1}, SessionId{49}, 8);
        },
        "a commitment shorter than its resident prefix must fail");

    const auto after = cache.diagnostics();
    const auto stats_after = cache.stats();
    expect(!cache.has_kda_snapshot(SessionId{49}) &&
               after.kda_snapshot_occupied_blocks ==
                   before.kda_snapshot_occupied_blocks &&
               after.kda_snapshot_sessions == before.kda_snapshot_sessions,
           "failed commit validation must not reserve a KDA snapshot");
    expect(cache.allocation_count() == 0 &&
               cache.gpu_cache_valid_prefix_blocks(SessionId{49}) ==
                   reusable_frontier_before &&
               !cache.session_has_active_request(SessionId{49}) &&
               after.resident_blocks == before.resident_blocks &&
               after.resident_blocks == 3,
           "failed commit validation must leave request and session ownership "
           "unchanged");
    expect(after.evictable_blocks == before.evictable_blocks &&
               after.available_commitment_blocks ==
                   before.available_commitment_blocks &&
               stats_after.evicted_blocks == stats_before.evicted_blocks &&
               stats_after.evicted_sessions == stats_before.evicted_sessions &&
               stats_after.evicted_kda_snapshots ==
                   stats_before.evicted_kda_snapshots,
           "failed commit validation must not reclaim or evict cache state");
}

void test_published_zero_frontier_kda_snapshot_survives_free() {
    ReplicaKVCacheManager cache = manager(4);
    cache.configure_kda_snapshot(2);
    Request request = make_request(0, 3, 1, 50);
    request.on_arrival(request.arrived_at());
    expect(cache.can_admit(request.id(), request.session_id(), 0, 3),
           "sub-block KDA prompt must fit with its snapshot charge");
    cache.admit(request.id(), request.session_id(), 0, 3);
    request.on_admitted(SimTime::from_seconds(0.0));
    request.advance_scheduler_frontier(3);
    request.on_batch_completion(SimTime::from_seconds(0.001), 3,
                                frontier::ClusterType::kPrefill);
    cache.mark_blocks_computed(request);
    expect(request.is_prefill_complete() &&
               cache.publish_kda_snapshot(request.session_id(), 0),
           "completed sub-block PREFILL must publish a valid zero frontier");

    static_cast<void>(cache.free(request.id()));
    const auto snapshot = cache.checkpoint_kda_snapshot(request.session_id());
    expect(snapshot.present && snapshot.published &&
               snapshot.frontier_blocks == 0 &&
               cache.kda_snapshot_occupied_blocks() == 2 &&
               cache.diagnostics().kda_snapshot_evictable_sessions == 1 &&
               cache.allocation_count() == 0,
           "normal free must retain a published zero-frontier KDA snapshot");
    expect(cache.discard_kda_snapshot(request.session_id()) == 2 &&
               !cache.has_kda_snapshot(request.session_id()),
           "published snapshot-only state must remain explicitly evictable");
}

void test_seeded_session_range_churn() {
    constexpr std::uint64_t kIterations = 20'000;
    constexpr std::uint64_t kSessions = 31;
    ReplicaKVCacheManager cache = manager(17);
    std::mt19937_64 random{0xA11A7C1CULL};
    std::uniform_int_distribution<std::uint64_t> session_distribution{
        0, kSessions - 1};

    for (std::uint64_t index = 0; index < kIterations; ++index) {
        const std::uint64_t session = session_distribution(random);
        // Keep every session prompt at the same append-only upper bound. A
        // shorter later prompt would be branching, which this model rejects.
        constexpr std::uint64_t prefill = 49;
        Request request = make_request(index, prefill, 1, session);
        admit_and_complete(cache, request,
                           static_cast<double>(index) * 0.00001);
        const auto diagnostics = cache.diagnostics();
        expect(diagnostics.active_blocks == 0 &&
                   diagnostics.available_blocks == 17 &&
                   diagnostics.resident_blocks ==
                       diagnostics.evictable_blocks &&
                   diagnostics.resident_blocks <= diagnostics.capacity_blocks,
               "randomized analytical capacity partition diverged");
    }
    expect(cache.stats().evicted_blocks > kIterations &&
               cache.stats().evicted_sessions > 0,
           "randomized churn must repeatedly reclaim session ranges");
}

void test_hundred_million_block_prefix_has_constant_size_state() {
    constexpr std::uint64_t kCompleteBlocks = 100'000'000;
    constexpr std::uint64_t kTurns = 5'000;
    ReplicaKVCacheManager cache = manager(kCompleteBlocks + 1);

    Request producer = make_request(0, kCompleteBlocks * 4 + 1, 1, 101);
    admit_and_complete(cache, producer, 0.0);
    expect(cache.gpu_cache_valid_prefix_blocks(SessionId{101}) ==
                   kCompleteBlocks &&
               cache.diagnostics().evictable_sessions == 1,
           "a huge logical prefix must occupy one session-range entry");

    for (std::uint64_t turn = 1; turn <= kTurns; ++turn) {
        Request request = make_request(turn, kCompleteBlocks * 4 + 1, 1, 101);
        admit_and_complete(cache, request, static_cast<double>(turn) * 0.00001);
    }
    const auto diagnostics = cache.diagnostics();
    expect(diagnostics.resident_blocks == kCompleteBlocks &&
               diagnostics.evictable_blocks == kCompleteBlocks &&
               diagnostics.evictable_sessions == 1 &&
               diagnostics.sessions_with_nonzero_frontier == 1,
           "repeated huge-prefix turns must retain range-sized metadata");
}

void test_kda_snapshot_hybrid_lookup_and_replacement() {
    ReplicaKVCacheManager cache = manager(8);
    cache.configure_kda_snapshot(2);

    Request request = make_request(0, 8, 1, 301);
    request.on_arrival(request.arrived_at());
    const PrefixLookupResult cold = cache.lookup(request);
    cache.admit(request.id(), request.session_id(), cold.cached_tokens, 8);
    request.restore_prefix_cache_lookup(cold.query_blocks, cold.hit_blocks,
                                        cold.cached_tokens);
    request.on_admitted(SimTime::from_seconds(0.0));
    request.advance_scheduler_frontier(8);
    request.on_batch_completion(SimTime::from_seconds(0.001), 8);
    cache.mark_blocks_computed(request);
    expect(cache.publish_kda_snapshot(request.session_id(), 2),
           "active request must publish a KDA snapshot");
    expect(cache.kda_snapshot_occupied_blocks() == 2 &&
               cache.diagnostics().kda_snapshot_evictable_sessions == 0,
           "active snapshot must be charged once and remain pinned");

    static_cast<void>(cache.free(request.id()));
    Request successor = make_request(1, 8, 1, 301);
    successor.on_arrival(successor.arrived_at());
    PrefixLookupResult warm = cache.lookup(successor);
    expect(warm.hit_blocks == 2 &&
               cache.gpu_cache_valid_prefix_blocks(SessionId{301}) == 2,
           "hybrid lookup must require and honor the KDA frontier");

    expect(cache.publish_kda_snapshot(SessionId{301}, 1),
           "snapshot replacement must succeed");
    expect(cache.kda_snapshot_occupied_blocks() == 2 &&
               cache.kda_snapshot_frontier_blocks(SessionId{301}) == 1 &&
               cache.gpu_cache_valid_prefix_blocks(SessionId{301}) == 1,
           "shorter replacement must not double-charge and must cap KV hits");
    warm = cache.lookup(successor);
    expect(warm.hit_blocks == 1,
           "lookup must reverse to the shorter immutable snapshot frontier");
    expect(cache.can_admit(successor.id(), successor.session_id(),
                           warm.cached_tokens, 4),
           "admission must use the effective KDA frontier");
    cache.admit(successor.id(), successor.session_id(), warm.cached_tokens, 4);
    expect(cache.allocated_blocks(successor.id()) == 2,
           "admission must trim stale KV suffix before allocating");
    static_cast<void>(cache.free(successor.id()));
    expect(cache.gpu_cache_valid_prefix_blocks(SessionId{301}) == 1,
           "free must retain the shorter effective frontier");
}

void test_kda_snapshot_eviction_orders_kv_before_snapshot_lru() {
    ReplicaKVCacheManager cache = manager(8);
    cache.configure_kda_snapshot(2);

    Request first = make_request(0, 8, 1, 401);
    admit_and_complete(cache, first, 0.0);
    expect(cache.publish_kda_snapshot(SessionId{401}, 2),
           "first snapshot must fit");
    Request second = make_request(1, 8, 1, 402);
    admit_and_complete(cache, second, 0.01);
    expect(cache.publish_kda_snapshot(SessionId{402}, 2),
           "second snapshot must fit");

    expect(cache.publish_kda_snapshot(SessionId{403}, 2),
           "new snapshot must reclaim ordinary KV first");
    expect(cache.has_kda_snapshot(SessionId{401}) &&
               cache.gpu_cache_valid_prefix_blocks(SessionId{401}) == 0 &&
               cache.stats().evicted_kda_snapshots == 0,
           "KV suffix eviction must precede snapshot eviction");

    expect(cache.publish_kda_snapshot(SessionId{404}, 2),
           "snapshot allocation must continue after KV is exhausted");
    expect(cache.publish_kda_snapshot(SessionId{405}, 2),
           "snapshot LRU must reclaim one whole group when needed");
    expect(!cache.has_kda_snapshot(SessionId{401}) &&
               cache.has_kda_snapshot(SessionId{402}) &&
               cache.kda_snapshot_occupied_blocks() == 8 &&
               cache.stats().evicted_kda_snapshots == 1 &&
               cache.stats().evicted_kda_snapshot_blocks == 2,
           "snapshot eviction must be LRU and atomic as a whole");
}

void test_kda_unified_lru_prefers_older_snapshot_over_newer_kv() {
    ReplicaKVCacheManager cache = manager(6);
    cache.configure_kda_snapshot(2);
    expect(cache.publish_kda_snapshot(SessionId{701}, 4),
           "old snapshot-only session must fit");

    Request newer = make_request(0, 8, 1, 702);
    admit_and_complete(cache, newer, 0.0);
    expect(cache.publish_kda_snapshot(SessionId{702}, 2),
           "completed newer request must publish into its reserved snapshot");
    expect(cache.diagnostics().resident_blocks == 2,
           "newer session must retain its ordinary resident KV");

    expect(cache.publish_kda_snapshot(SessionId{703}, 4),
           "first filler snapshot must evict the oldest snapshot-only session");
    expect(!cache.has_kda_snapshot(SessionId{701}) &&
               cache.gpu_cache_valid_prefix_blocks(SessionId{702}) == 2 &&
               cache.stats().evicted_kda_snapshots == 1 &&
               cache.stats().evicted_blocks == 0,
           "older snapshot-only victim must precede the newer KDA session");
    expect(cache.publish_kda_snapshot(SessionId{704}, 4),
           "second filler snapshot must continue the newer session chain");
    expect(!cache.has_kda_snapshot(SessionId{701}) &&
               cache.has_kda_snapshot(SessionId{702}) &&
               cache.diagnostics().resident_blocks == 0 &&
               cache.stats().evicted_kda_snapshots == 1 &&
               cache.stats().evicted_blocks == 2,
           "once the older session is gone, newer ordinary KV must be evicted "
           "before its same-session snapshot");
}

void test_kda_atomic_surplus_reclaim_stays_blank() {
    ReplicaKVCacheManager cache = manager(10);
    cache.configure_kda_snapshot(3);

    Request older = make_request(0, 8, 1, 711);
    admit_and_complete(cache, older, 0.0);
    expect(cache.publish_kda_snapshot(SessionId{711}, 2),
           "older resident session must publish a snapshot");

    Request newer = make_request(1, 8, 1, 712);
    admit_and_complete(cache, newer, 0.01);
    expect(cache.publish_kda_snapshot(SessionId{712}, 2),
           "newer resident session must publish a valid snapshot");
    expect(cache.diagnostics().resident_blocks == 4,
           "surplus fixture must retain both ordinary ranges initially");

    expect(cache.can_reserve(RequestId{799}, 0, 16),
           "reservation must use the older session's reclaim chain");
    cache.reserve(RequestId{799}, 0, 16);
    expect(!cache.has_kda_snapshot(SessionId{711}) &&
               cache.diagnostics().resident_blocks == 2 &&
               cache.allocated_blocks(RequestId{799}) == 4 &&
               cache.stats().evicted_blocks == 2 &&
               cache.stats().evicted_kda_snapshots == 1 &&
               cache.stats().evicted_kda_snapshot_blocks == 3,
           "atomic snapshot surplus must not spill into the newer KV victim");
    static_cast<void>(cache.free(RequestId{799}));
    expect(cache.diagnostics().resident_blocks == 2 &&
               cache.diagnostics().available_blocks == 7,
           "surplus snapshot blocks must return to blank capacity while the "
           "newer session snapshot remains charged");
}

void test_kda_snapshot_keeps_zero_resident_session_and_discard_is_explicit() {
    ReplicaKVCacheManager cache = manager(4);
    cache.configure_kda_snapshot(2);
    expect(cache.publish_kda_snapshot(SessionId{501}, 7),
           "snapshot-only session must be publishable");
    expect(cache.has_kda_snapshot(SessionId{501}) &&
               cache.gpu_cache_valid_prefix_blocks(SessionId{501}) == 0 &&
               cache.diagnostics().kda_snapshot_sessions == 1,
           "snapshot ownership must survive with zero resident KV");
    expect(cache.discard_session(SessionId{501}) == 0 &&
               !cache.has_kda_snapshot(SessionId{501}) &&
               cache.kda_snapshot_occupied_blocks() == 0 &&
               cache.diagnostics().available_blocks == 4,
           "explicit discard must retire a snapshot-only session atomically");
}

void test_kda_snapshot_active_pin_rejects_capacity_pressure() {
    ReplicaKVCacheManager cache = manager(4);
    cache.configure_kda_snapshot(2);
    Request request = make_request(0, 8, 1, 601);
    request.on_arrival(request.arrived_at());
    cache.admit(request.id(), request.session_id(), 0, 8);
    request.restore_prefix_cache_lookup(2, 0, 0);
    request.on_admitted(SimTime::from_seconds(0.0));
    request.advance_scheduler_frontier(8);
    request.on_batch_completion(SimTime::from_seconds(0.001), 8);
    cache.mark_blocks_computed(request);
    expect(cache.publish_kda_snapshot(request.session_id(), 2),
           "active pin fixture must publish");
    expect(!cache.publish_kda_snapshot(SessionId{602}, 2),
           "active snapshot must not be evicted under pressure");
    static_cast<void>(cache.free(request.id()));
    expect(cache.diagnostics().kda_snapshot_evictable_sessions == 1,
           "free must release snapshot into its LRU");
}

} // namespace

int main() {
    int failures = 0;
    failures +=
        frontier::test::run("session release and suffix LRU eviction",
                            test_session_release_and_suffix_lru_eviction);
    failures +=
        frontier::test::run("all-hit recompute reuses one logical slot",
                            test_all_hit_recompute_reuses_one_logical_slot);
    failures += frontier::test::run("active session is not shareable",
                                    test_active_session_is_not_shareable);
    failures += frontier::test::run(
        "explicit session discard is immediate or deferred",
        test_explicit_session_discard_is_immediate_or_deferred);
    failures += frontier::test::run(
        "partial prefill reuses analytical resident range",
        test_partial_prefill_preemption_reuses_resident_range);
    failures += frontier::test::run("cache-disabled count accounting",
                                    test_cache_disabled_count_accounting);
    failures += frontier::test::run(
        "virtual full-sequence commitment tracks chunk materialization",
        test_virtual_full_sequence_commitment_tracks_chunk_materialization);
    failures += frontier::test::run(
        "virtual commitment rejects another active session owner",
        test_virtual_commitment_rejects_another_active_session_owner);
    failures += frontier::test::run(
        "empty fit is independent of current commitments",
        test_empty_fit_is_independent_of_current_commitments);
    failures += frontier::test::run(
        "KDA admission reserves cold snapshot capacity",
        test_kda_admission_reserves_cold_snapshot_capacity);
    failures += frontier::test::run(
        "failed virtual commit leaves no KDA snapshot charge",
        test_failed_virtual_commit_does_not_reserve_kda_snapshot);
    failures += frontier::test::run(
        "published zero-frontier KDA snapshot survives free",
        test_published_zero_frontier_kda_snapshot_survives_free);
    failures += frontier::test::run("seeded session range churn",
                                    test_seeded_session_range_churn);
    failures += frontier::test::run(
        "hundred-million-block prefix uses constant-size state",
        test_hundred_million_block_prefix_has_constant_size_state);
    failures +=
        frontier::test::run("KDA snapshot hybrid lookup and replacement",
                            test_kda_snapshot_hybrid_lookup_and_replacement);
    failures += frontier::test::run(
        "KDA snapshot eviction orders KV before snapshot LRU",
        test_kda_snapshot_eviction_orders_kv_before_snapshot_lru);
    failures += frontier::test::run(
        "KDA unified LRU prefers older snapshot over newer KV",
        test_kda_unified_lru_prefers_older_snapshot_over_newer_kv);
    failures += frontier::test::run(
        "KDA snapshot atomic surplus reclaim",
        test_kda_atomic_surplus_reclaim_stays_blank);
    failures += frontier::test::run(
        "KDA snapshot keeps zero-resident session",
        test_kda_snapshot_keeps_zero_resident_session_and_discard_is_explicit);
    failures += frontier::test::run(
        "KDA snapshot active pin rejects pressure",
        test_kda_snapshot_active_pin_rejects_capacity_pressure);
    return failures == 0 ? 0 : 1;
}
