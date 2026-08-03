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
using frontier::request_generator::WorkloadRequest;
using frontier::test::expect;

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
        "partial prefill reuses analytical resident range",
        test_partial_prefill_preemption_reuses_resident_range);
    failures += frontier::test::run("cache-disabled count accounting",
                                    test_cache_disabled_count_accounting);
    failures += frontier::test::run("seeded session range churn",
                                    test_seeded_session_range_churn);
    failures += frontier::test::run(
        "hundred-million-block prefix uses constant-size state",
        test_hundred_million_block_prefix_has_constant_size_state);
    return failures == 0 ? 0 : 1;
}
