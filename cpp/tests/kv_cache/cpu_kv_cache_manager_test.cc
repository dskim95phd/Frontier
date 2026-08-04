#include "frontier/kv_cache/cpu_kv_cache_manager.h"
#include "tests/test_support.h"

#include <cstdint>
#include <random>
#include <vector>

namespace {

using frontier::CpuOffloadGeneration;
using frontier::RequestId;
using frontier::SessionId;
using frontier::SimTime;
using frontier::config::CpuKVCacheCapacityPressurePolicy;
using frontier::kv_cache::CpuKVCacheManager;
using frontier::test::expect;

SimTime at(double seconds) { return SimTime::from_seconds(seconds); }

void commit(CpuKVCacheManager &manager, std::int64_t session,
            std::int64_t generation, std::uint64_t frontier, double time) {
    const auto reserved = manager.reserve_offload(
        SessionId{session}, CpuOffloadGeneration{generation}, frontier,
        at(time));
    expect(reserved.requires_transfer(), "test offload must reserve blocks");
    expect(manager.commit_offload(reserved.reservation_id, at(time + 0.1)),
           "pending offload must commit once");
}

void test_incremental_contiguous_and_metrics() {
    CpuKVCacheManager manager{16,
                              CpuKVCacheCapacityPressurePolicy::kPrefixFit};
    commit(manager, 1, 1, 2, 1.0);
    const auto delta = manager.reserve_offload(
        SessionId{1}, CpuOffloadGeneration{2}, 5, at(2.0));
    expect(delta.reserved_blocks == 3 &&
               delta.admitted_frontier_blocks == 5,
           "incremental offload must reserve only the missing suffix");
    expect(manager.lookup(SessionId{1}, 8).hit_blocks == 2,
           "reserved suffix must remain invisible to lookup");
    expect(manager.commit_offload(delta.reservation_id, at(2.1)) &&
               manager.committed_frontier_blocks(SessionId{1}) == 5,
           "commit must advance one contiguous CPU frontier");

    const auto lookup = manager.lookup(SessionId{1}, 4);
    manager.record_successful_lookup(RequestId{10}, lookup);
    manager.record_successful_lookup(RequestId{10}, lookup);
    expect(manager.stats().successful_lookups == 1 &&
               manager.stats().query_blocks == 4 &&
               manager.stats().hit_blocks == 4,
           "scheduler retry metrics must deduplicate by request");
    manager.validate_invariants();
}

void test_prefix_fit_skip_and_lazy_capacity() {
    CpuKVCacheManager prefix_fit{
        4, CpuKVCacheCapacityPressurePolicy::kPrefixFit};
    const auto truncated = prefix_fit.reserve_offload(
        SessionId{1}, CpuOffloadGeneration{1}, 9, at(1.0));
    expect(truncated.requires_transfer() && truncated.reserved_blocks == 4 &&
               truncated.truncated,
           "prefix_fit must admit the largest contiguous prefix");
    expect(prefix_fit.commit_offload(truncated.reservation_id, at(1.1)),
           "truncated prefix must remain committable");

    CpuKVCacheManager skip{3,
                           CpuKVCacheCapacityPressurePolicy::kSkipOffload};
    commit(skip, 1, 1, 3, 1.0);
    const auto skipped = skip.reserve_offload(
        SessionId{2}, CpuOffloadGeneration{1}, 4, at(2.0));
    expect(skipped.skipped && !skipped.requires_transfer() &&
               skip.committed_frontier_blocks(SessionId{1}) == 3,
           "skip_offload must not evict when the complete delta cannot fit");

    CpuKVCacheManager huge{
        1'000'000'000ULL, CpuKVCacheCapacityPressurePolicy::kPrefixFit};
    const auto idle = huge.diagnostics();
    expect(idle.materialized_blocks == 0 && idle.sessions == 0,
           "huge CPU capacity must use constant-size initial state");
}

void test_lru_suffix_and_restore_pins() {
    CpuKVCacheManager manager{6,
                              CpuKVCacheCapacityPressurePolicy::kPrefixFit};
    commit(manager, 1, 1, 3, 1.0);
    commit(manager, 2, 1, 3, 2.0);
    const auto lease = manager.pin_restore(SessionId{1}, 0, 3, at(3.0));
    const auto replacement = manager.reserve_offload(
        SessionId{3}, CpuOffloadGeneration{1}, 2, at(4.0));
    expect(replacement.reserved_blocks == 2 &&
               manager.committed_frontier_blocks(SessionId{1}) == 3 &&
               manager.committed_frontier_blocks(SessionId{2}) == 1,
           "pinned session must survive LRU suffix eviction");
    expect(manager.release_restore(lease, true, at(4.1)) &&
               !manager.release_restore(lease, true, at(4.2)),
           "restore lease release must be exact once");
    expect(manager.abort_offload(replacement.reservation_id),
           "pending replacement must abort");
    manager.validate_invariants();
}

void test_incremental_occupancy_accounting() {
    CpuKVCacheManager manager{5,
                              CpuKVCacheCapacityPressurePolicy::kPrefixFit};
    const auto reservation = manager.reserve_offload(
        SessionId{11}, CpuOffloadGeneration{1}, 3, at(1.0));
    const auto reserved = manager.diagnostics();
    expect(reserved.materialized_blocks == 3 &&
               reserved.resident_blocks == 0 &&
               reserved.reserved_blocks == 3 && reserved.pinned_blocks == 0 &&
               manager.stats().peak_reserved_blocks == 3,
           "reservation accounting must update occupancy incrementally");
    expect(manager.commit_offload(reservation.reservation_id, at(1.1)),
           "occupancy accounting test reservation must commit");
    const auto committed = manager.diagnostics();
    expect(committed.materialized_blocks == 3 &&
               committed.resident_blocks == 3 &&
               committed.reserved_blocks == 0 &&
               manager.stats().peak_resident_blocks == 3,
           "commit accounting must move blocks from reserved to resident");

    const auto lease = manager.pin_restore(SessionId{11}, 0, 2, at(2.0));
    expect(manager.diagnostics().pinned_blocks == 2,
           "restore pin accounting must count distinct pinned blocks");
    expect(manager.release_restore(lease, false, at(2.1)),
           "occupancy accounting restore lease must release");
    expect(manager.diagnostics().pinned_blocks == 0,
           "restore release accounting must clear distinct pinned blocks");
    manager.validate_invariants();
}

void test_out_of_order_commit_and_dependent_abort() {
    CpuKVCacheManager manager{8,
                              CpuKVCacheCapacityPressurePolicy::kPrefixFit};
    const auto first = manager.reserve_offload(
        SessionId{7}, CpuOffloadGeneration{1}, 2, at(1.0));
    const auto second = manager.reserve_offload(
        SessionId{7}, CpuOffloadGeneration{2}, 4, at(1.1));
    expect(manager.commit_offload(second.reservation_id, at(2.0)) &&
               manager.committed_frontier_blocks(SessionId{7}) == 0,
           "out-of-order suffix commit must remain hidden behind a gap");
    expect(manager.commit_offload(first.reservation_id, at(2.1)) &&
               manager.committed_frontier_blocks(SessionId{7}) == 4 &&
               manager.stats().stale_generation_completions == 1,
           "closing the gap must expose the committed suffix");
    expect(!manager.commit_offload(second.reservation_id, at(2.2)) &&
               manager.stats().stale_generation_completions == 2,
           "duplicate terminal commit must remain idempotent");

    CpuKVCacheManager aborting{
        8, CpuKVCacheCapacityPressurePolicy::kPrefixFit};
    const auto base = aborting.reserve_offload(
        SessionId{8}, CpuOffloadGeneration{1}, 2, at(1.0));
    const auto suffix = aborting.reserve_offload(
        SessionId{8}, CpuOffloadGeneration{2}, 4, at(1.1));
    expect(aborting.commit_offload(suffix.reservation_id, at(2.0)),
           "dependent suffix may finish first");
    expect(aborting.abort_offload(base.reservation_id) &&
               aborting.committed_frontier_blocks(SessionId{8}) == 0 &&
               aborting.diagnostics().materialized_blocks == 0,
           "aborting an earlier reservation must reclaim dependent suffixes");
    expect(!aborting.abort_offload(base.reservation_id),
           "duplicate terminal abort must remain idempotent");
}

void test_noop_lru_empty_metadata_and_zero_fit() {
    CpuKVCacheManager manager{4,
                              CpuKVCacheCapacityPressurePolicy::kPrefixFit};
    commit(manager, 1, 1, 2, 1.0);
    commit(manager, 2, 1, 2, 2.0);
    const auto no_op = manager.reserve_offload(
        SessionId{1}, CpuOffloadGeneration{2}, 2, at(3.0));
    expect(!no_op.requires_transfer() && !no_op.skipped &&
               no_op.admitted_frontier_blocks == 2,
           "already resident snapshot must be a terminal no-op");
    const auto shorter_no_op = manager.reserve_offload(
        SessionId{1}, CpuOffloadGeneration{3}, 1, at(3.1));
    expect(!shorter_no_op.requires_transfer() &&
               shorter_no_op.admitted_frontier_blocks == 1,
           "a no-op snapshot must report the requested admitted frontier");
    const auto replacement = manager.reserve_offload(
        SessionId{3}, CpuOffloadGeneration{1}, 1, at(4.0));
    expect(replacement.requires_transfer() &&
               manager.committed_frontier_blocks(SessionId{1}) == 1 &&
               manager.committed_frontier_blocks(SessionId{2}) == 2,
           "no-op offload must not refresh session LRU");
    expect(manager.abort_offload(replacement.reservation_id),
           "replacement cleanup must succeed");

    CpuKVCacheManager full{2,
                           CpuKVCacheCapacityPressurePolicy::kPrefixFit};
    commit(full, 20, 1, 2, 1.0);
    const auto zero_fit = full.reserve_offload(
        SessionId{20}, CpuOffloadGeneration{2}, 4, at(5.0));
    expect(!zero_fit.skipped && zero_fit.truncated &&
               !zero_fit.requires_transfer() &&
               full.diagnostics().active_restore_leases == 0,
           "zero-block prefix fit must be a truncation and pin nothing");

    CpuKVCacheManager skip{1,
                           CpuKVCacheCapacityPressurePolicy::kSkipOffload};
    commit(skip, 9, 1, 1, 1.0);
    const auto skipped = skip.reserve_offload(
        SessionId{10}, CpuOffloadGeneration{1}, 2, at(2.0));
    expect(skipped.skipped && skip.diagnostics().sessions == 1,
           "skipped new session must not retain empty metadata");
    manager.validate_invariants();
    skip.validate_invariants();
}

void test_randomized_invariants() {
    CpuKVCacheManager manager{32,
                              CpuKVCacheCapacityPressurePolicy::kPrefixFit};
    std::mt19937_64 random{42};
    std::uint64_t generation[6]{};
    double now = 1.0;
    for (std::uint64_t step = 0; step < 2'000; ++step) {
        const std::int64_t session_value =
            static_cast<std::int64_t>(1 + random() % 5);
        const SessionId session{session_value};
        const std::uint64_t current =
            manager.committed_frontier_blocks(session);
        const std::uint64_t desired = current + 1 + random() % 8;
        const auto reservation = manager.reserve_offload(
            session, CpuOffloadGeneration{++generation[session_value]},
            desired, at(now));
        now += 0.001;
        if (reservation.requires_transfer()) {
            if (random() % 4 == 0) {
                expect(manager.abort_offload(reservation.reservation_id),
                       "random pending reservation must abort");
            } else {
                expect(manager.commit_offload(reservation.reservation_id,
                                              at(now)),
                       "random pending reservation must commit");
                now += 0.001;
            }
        }
        if (manager.committed_frontier_blocks(session) > 0 &&
            random() % 5 == 0) {
            const auto lease = manager.pin_restore(
                session, 0, 1, at(now));
            now += 0.001;
            expect(manager.release_restore(lease, random() % 2 == 0, at(now)),
                   "random restore lease must release");
            now += 0.001;
        }
        manager.validate_invariants();
        const auto state = manager.diagnostics();
        expect(state.resident_blocks + state.reserved_blocks <=
                   state.capacity_blocks,
               "random operations must preserve finite CPU capacity");
    }
}

} // namespace

int main() {
    int failures = 0;
    failures += frontier::test::run("CPU incremental contiguous and metrics",
                                    test_incremental_contiguous_and_metrics);
    failures += frontier::test::run("CPU prefix fit, skip, and lazy capacity",
                                    test_prefix_fit_skip_and_lazy_capacity);
    failures += frontier::test::run("CPU LRU suffix and restore pins",
                                    test_lru_suffix_and_restore_pins);
    failures += frontier::test::run(
        "CPU incremental occupancy accounting",
        test_incremental_occupancy_accounting);
    failures += frontier::test::run(
        "CPU out-of-order commit and dependent abort",
        test_out_of_order_commit_and_dependent_abort);
    failures += frontier::test::run(
        "CPU no-op LRU and terminal empty paths",
        test_noop_lru_empty_metadata_and_zero_fit);
    failures += frontier::test::run("CPU randomized invariants",
                                    test_randomized_invariants);
    return failures == 0 ? 0 : 1;
}
