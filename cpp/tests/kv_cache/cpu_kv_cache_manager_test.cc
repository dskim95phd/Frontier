#include "frontier/kv_cache/cpu_kv_cache_manager.h"
#include "tests/test_support.h"

#include <algorithm>
#include <cstdint>
#include <random>
#include <utility>
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
    CpuKVCacheManager manager{16, CpuKVCacheCapacityPressurePolicy::kPrefixFit};
    commit(manager, 1, 1, 2, 1.0);
    const auto delta = manager.reserve_offload(
        SessionId{1}, CpuOffloadGeneration{2}, 5, at(2.0));
    expect(delta.reserved_blocks == 3 && delta.admitted_frontier_blocks == 5,
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
    CpuKVCacheManager prefix_fit{4,
                                 CpuKVCacheCapacityPressurePolicy::kPrefixFit};
    const auto truncated = prefix_fit.reserve_offload(
        SessionId{1}, CpuOffloadGeneration{1}, 9, at(1.0));
    expect(truncated.requires_transfer() && truncated.reserved_blocks == 4 &&
               truncated.truncated,
           "prefix_fit must admit the largest contiguous prefix");
    expect(prefix_fit.commit_offload(truncated.reservation_id, at(1.1)),
           "truncated prefix must remain committable");

    CpuKVCacheManager skip{3, CpuKVCacheCapacityPressurePolicy::kSkipOffload};
    commit(skip, 1, 1, 3, 1.0);
    const auto skipped =
        skip.reserve_offload(SessionId{2}, CpuOffloadGeneration{1}, 4, at(2.0));
    expect(skipped.skipped && !skipped.requires_transfer() &&
               skip.committed_frontier_blocks(SessionId{1}) == 3,
           "skip_offload must not evict when the complete delta cannot fit");

    CpuKVCacheManager huge{1'000'000'000ULL,
                           CpuKVCacheCapacityPressurePolicy::kPrefixFit};
    const auto idle = huge.diagnostics();
    expect(idle.materialized_blocks == 0 && idle.sessions == 0,
           "huge CPU capacity must use constant-size initial state");
}

void test_lru_suffix_and_restore_pins() {
    CpuKVCacheManager manager{6, CpuKVCacheCapacityPressurePolicy::kPrefixFit};
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
    expect(manager.retained_restore_lease_count() == 0,
           "released restore leases must not retain tombstones");
    expect(manager.abort_offload(replacement.reservation_id),
           "pending replacement must abort");
    expect(manager.retained_reservation_count() == 0,
           "aborted offloads must not retain tombstones");
    manager.validate_invariants();
}

void test_incremental_occupancy_accounting() {
    CpuKVCacheManager manager{5, CpuKVCacheCapacityPressurePolicy::kPrefixFit};
    const auto reservation = manager.reserve_offload(
        SessionId{11}, CpuOffloadGeneration{1}, 3, at(1.0));
    const auto reserved = manager.diagnostics();
    expect(reserved.materialized_blocks == 3 && reserved.resident_blocks == 0 &&
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

void test_overlapping_restore_ranges_count_distinct_pins() {
    CpuKVCacheManager manager{16, CpuKVCacheCapacityPressurePolicy::kPrefixFit};
    commit(manager, 12, 1, 8, 1.0);

    const auto first = manager.pin_restore(SessionId{12}, 0, 5, at(2.0));
    const auto second = manager.pin_restore(SessionId{12}, 3, 8, at(2.1));
    expect(manager.diagnostics().pinned_blocks == 8,
           "overlapping restore leases must count their block union");

    expect(manager.release_restore(first, true, at(2.2)) &&
               manager.diagnostics().pinned_blocks == 5,
           "releasing one overlapping lease must preserve the other range");
    expect(manager.release_restore(second, true, at(2.3)) &&
               manager.diagnostics().pinned_blocks == 0,
           "releasing every overlapping lease must clear distinct pins");
    manager.validate_invariants();
}

void test_large_resident_range_has_constant_metadata() {
    constexpr std::uint64_t capacity = 1'000'000'000ULL;
    constexpr std::uint64_t frontier = 500'000'000ULL;
    CpuKVCacheManager manager{capacity,
                              CpuKVCacheCapacityPressurePolicy::kPrefixFit};
    const auto reservation = manager.reserve_offload(
        SessionId{13}, CpuOffloadGeneration{1}, frontier, at(1.0));
    expect(reservation.reserved_blocks == frontier &&
               manager.commit_offload(reservation.reservation_id, at(1.1)),
           "large analytical range must reserve and commit as one record");
    const auto diagnostics = manager.diagnostics();
    expect(diagnostics.materialized_blocks == frontier &&
               diagnostics.resident_blocks == frontier &&
               diagnostics.sessions == 1 &&
               diagnostics.active_reservations == 0,
           "large analytical range must preserve logical block accounting");
    manager.validate_invariants();
}

void test_out_of_order_commit_and_dependent_abort() {
    CpuKVCacheManager manager{8, CpuKVCacheCapacityPressurePolicy::kPrefixFit};
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
    expect(manager.retained_reservation_count() == 0,
           "committed offloads must not retain tombstones");

    CpuKVCacheManager aborting{8, CpuKVCacheCapacityPressurePolicy::kPrefixFit};
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
    expect(aborting.retained_reservation_count() == 0,
           "dependent abort cleanup must erase every terminal reservation");
}

void test_noop_lru_empty_metadata_and_zero_fit() {
    CpuKVCacheManager manager{4, CpuKVCacheCapacityPressurePolicy::kPrefixFit};
    commit(manager, 1, 1, 2, 1.0);
    commit(manager, 2, 1, 2, 2.0);
    const std::uint64_t truncated_before = manager.stats().truncated_offloads;
    const auto no_op = manager.reserve_offload(
        SessionId{1}, CpuOffloadGeneration{2}, 2, at(3.0));
    expect(!no_op.requires_transfer() && !no_op.skipped && !no_op.truncated &&
               no_op.admitted_frontier_blocks == 2 &&
               manager.stats().truncated_offloads == truncated_before,
           "an identical resident frontier must be a metric-neutral no-op");
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

    CpuKVCacheManager full{2, CpuKVCacheCapacityPressurePolicy::kPrefixFit};
    commit(full, 20, 1, 2, 1.0);
    const auto zero_fit = full.reserve_offload(
        SessionId{20}, CpuOffloadGeneration{2}, 4, at(5.0));
    expect(!zero_fit.skipped && zero_fit.truncated &&
               !zero_fit.requires_transfer() &&
               full.diagnostics().active_restore_leases == 0,
           "zero-block prefix fit must be a truncation and pin nothing");

    CpuKVCacheManager skip{1, CpuKVCacheCapacityPressurePolicy::kSkipOffload};
    commit(skip, 9, 1, 1, 1.0);
    const auto skipped = skip.reserve_offload(
        SessionId{10}, CpuOffloadGeneration{1}, 2, at(2.0));
    expect(skipped.skipped && skip.diagnostics().sessions == 1,
           "skipped new session must not retain empty metadata");
    manager.validate_invariants();
    skip.validate_invariants();
}

void test_migration_discard_drains_inflight_transfers() {
    CpuKVCacheManager manager{8, CpuKVCacheCapacityPressurePolicy::kPrefixFit};
    commit(manager, 41, 1, 3, 1.0);
    const auto lease = manager.pin_restore(SessionId{41}, 0, 2, at(2.0));
    const auto suffix = manager.reserve_offload(
        SessionId{41}, CpuOffloadGeneration{2}, 5, at(2.1));
    expect(suffix.requires_transfer() && suffix.reserved_blocks == 2,
           "migration test must own an in-flight offload suffix");

    expect(manager.discard_session(SessionId{41}) &&
               manager.session_discard_pending(SessionId{41}) &&
               manager.lookup(SessionId{41}, 5).hit_blocks == 0 &&
               manager.committed_frontier_blocks(SessionId{41}) == 0,
           "migration discard must hide CPU KV immediately");
    const auto bounced = manager.reserve_offload(
        SessionId{41}, CpuOffloadGeneration{3}, 6, at(2.2));
    expect(bounced.skipped && !bounced.requires_transfer(),
           "a bounced session must not reuse or extend retired CPU KV");

    expect(manager.commit_offload(suffix.reservation_id, at(2.3)) &&
               manager.stats().discarded_offload_completions == 1 &&
               manager.diagnostics().materialized_blocks == 5,
           "retired D2H completion must drain without publishing or freeing "
           "restore-pinned storage");
    expect(manager.release_restore(lease, true, at(2.4)),
           "retired H2D lease must release normally");
    const auto drained = manager.diagnostics();
    expect(drained.sessions == 0 && drained.materialized_blocks == 0 &&
               drained.active_reservations == 0 &&
               drained.active_restore_leases == 0 &&
               manager.retained_reservation_count() == 0 &&
               manager.retained_restore_lease_count() == 0 &&
               !manager.session_discard_pending(SessionId{41}),
           "last retired transfer completion must physically reap the session");
    expect(!manager.commit_offload(suffix.reservation_id, at(2.5)),
           "late duplicate retired completion must remain idempotent");
    manager.validate_invariants();

    commit(manager, 41, 4, 2, 3.0);
    expect(manager.lookup(SessionId{41}, 2).hit_blocks == 2,
           "a fully drained lane may cache a later session incarnation");
    expect(manager.discard_session(SessionId{41}) &&
               manager.diagnostics().materialized_blocks == 0,
           "inactive CPU migration discard must free immediately");
    manager.validate_invariants();
}

void test_randomized_invariants() {
    CpuKVCacheManager manager{32, CpuKVCacheCapacityPressurePolicy::kPrefixFit};
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
            session, CpuOffloadGeneration{++generation[session_value]}, desired,
            at(now));
        now += 0.001;
        if (reservation.requires_transfer()) {
            if (random() % 4 == 0) {
                expect(manager.abort_offload(reservation.reservation_id),
                       "random pending reservation must abort");
            } else {
                expect(
                    manager.commit_offload(reservation.reservation_id, at(now)),
                    "random pending reservation must commit");
                now += 0.001;
            }
        }
        if (manager.committed_frontier_blocks(session) > 0 &&
            random() % 5 == 0) {
            const auto lease = manager.pin_restore(session, 0, 1, at(now));
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

std::uint64_t pinned_range_union(
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges) {
    if (ranges.empty()) {
        return 0;
    }
    std::sort(ranges.begin(), ranges.end());
    std::uint64_t total = 0;
    std::uint64_t begin = ranges.front().first;
    std::uint64_t end = ranges.front().second;
    for (std::size_t index = 1; index < ranges.size(); ++index) {
        if (ranges[index].first > end) {
            total += end - begin;
            begin = ranges[index].first;
            end = ranges[index].second;
        } else {
            end = std::max(end, ranges[index].second);
        }
    }
    return total + end - begin;
}

void test_randomized_out_of_order_ranges_and_overlapping_leases() {
    std::mt19937_64 random{0xC0FFEE};
    for (std::int64_t trial = 0; trial < 200; ++trial) {
        CpuKVCacheManager manager{1'024,
                                  CpuKVCacheCapacityPressurePolicy::kPrefixFit};
        const SessionId session{1 + trial};
        const std::size_t reservation_count =
            2 + static_cast<std::size_t>(random() % 15);
        std::vector<frontier::CpuOffloadReservationId> reservation_ids;
        std::vector<std::uint64_t> range_lengths;
        std::vector<bool> committed(reservation_count, false);
        reservation_ids.reserve(reservation_count);
        range_lengths.reserve(reservation_count);
        std::uint64_t frontier = 0;
        double now = 1.0;
        for (std::size_t index = 0; index < reservation_count; ++index) {
            const std::uint64_t length = 1 + random() % 16;
            frontier += length;
            const auto reservation = manager.reserve_offload(
                session,
                CpuOffloadGeneration{static_cast<std::int64_t>(index + 1)},
                frontier, at(now));
            now += 0.001;
            expect(reservation.reserved_blocks == length,
                   "random range reservation must append its complete suffix");
            reservation_ids.push_back(reservation.reservation_id);
            range_lengths.push_back(length);
        }

        std::vector<std::size_t> completion_order;
        completion_order.reserve(reservation_count);
        for (std::size_t index = 0; index < reservation_count; ++index) {
            completion_order.push_back(index);
        }
        std::shuffle(completion_order.begin(), completion_order.end(), random);
        for (const std::size_t completed : completion_order) {
            expect(manager.commit_offload(reservation_ids[completed], at(now)),
                   "random out-of-order reservation must commit");
            now += 0.001;
            committed[completed] = true;
            std::uint64_t expected_frontier = 0;
            std::uint64_t expected_resident = 0;
            std::uint64_t expected_reserved = 0;
            bool prefix_complete = true;
            for (std::size_t index = 0; index < reservation_count; ++index) {
                if (committed[index]) {
                    expected_resident += range_lengths[index];
                } else {
                    expected_reserved += range_lengths[index];
                }
                if (prefix_complete && committed[index]) {
                    expected_frontier += range_lengths[index];
                } else {
                    prefix_complete = false;
                }
            }
            const auto diagnostics = manager.diagnostics();
            expect(manager.committed_frontier_blocks(session) ==
                           expected_frontier &&
                       diagnostics.resident_blocks == expected_resident &&
                       diagnostics.reserved_blocks == expected_reserved,
                   "random completion order must publish only a contiguous "
                   "prefix while preserving aggregate range accounting");
        }

        struct ActiveLease {
            frontier::CpuRestoreLeaseId id;
            std::pair<std::uint64_t, std::uint64_t> range;
        };
        std::vector<ActiveLease> leases;
        const std::size_t lease_count =
            2 + static_cast<std::size_t>(random() % 25);
        leases.reserve(lease_count);
        for (std::size_t index = 0; index < lease_count; ++index) {
            const std::uint64_t begin = random() % frontier;
            const std::uint64_t end = begin + 1 + random() % (frontier - begin);
            leases.push_back({manager.pin_restore(session, begin, end, at(now)),
                              {begin, end}});
            now += 0.001;
            std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
            ranges.reserve(leases.size());
            for (const auto &lease : leases) {
                ranges.push_back(lease.range);
            }
            expect(manager.diagnostics().pinned_blocks ==
                       pinned_range_union(std::move(ranges)),
                   "overlapping random leases must count their interval union");
        }
        std::shuffle(leases.begin(), leases.end(), random);
        while (!leases.empty()) {
            expect(manager.release_restore(leases.back().id, random() % 2 == 0,
                                           at(now)),
                   "random overlapping lease must release exactly once");
            now += 0.001;
            leases.pop_back();
            std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
            ranges.reserve(leases.size());
            for (const auto &lease : leases) {
                ranges.push_back(lease.range);
            }
            expect(manager.diagnostics().pinned_blocks ==
                       pinned_range_union(std::move(ranges)),
                   "random lease release must preserve the remaining union");
        }
        manager.validate_invariants();
    }
}

void test_kda_snapshot_offload_replacement_and_restore_pin() {
    CpuKVCacheManager manager{16, CpuKVCacheCapacityPressurePolicy::kPrefixFit};
    manager.configure_kda_snapshot(2, 8'192);

    const auto first = manager.reserve_offload(
        SessionId{71}, CpuOffloadGeneration{1}, 3, at(1.0));
    expect(first.requires_transfer() && first.reserved_blocks == 3 &&
               first.kda_snapshot_blocks == 2 &&
               first.kda_snapshot_bytes == 8'192,
           "KDA first offload must carry one fixed snapshot payload");
    expect(manager.commit_offload(first.reservation_id, at(1.1)) &&
               manager.has_kda_snapshot(SessionId{71}) &&
               manager.kda_snapshot_frontier_blocks(SessionId{71}) == 3 &&
               manager.lookup(SessionId{71}, 8).hit_blocks == 3,
           "KDA commit must publish the immutable snapshot frontier");

    const auto lease = manager.pin_restore(SessionId{71}, 0, 2, at(2.0));
    expect(manager.restore_includes_kda_snapshot(lease) &&
               manager.restore_kda_snapshot_blocks(lease) == 2 &&
               manager.restore_kda_snapshot_bytes(lease) == 8'192 &&
               manager.diagnostics().pinned_kda_snapshots == 1,
           "KDA restore lease must pin and expose the snapshot payload");

    const auto replacement = manager.reserve_offload(
        SessionId{71}, CpuOffloadGeneration{2}, 4, at(2.1));
    expect(replacement.requires_transfer() &&
               replacement.reserved_blocks == 1 &&
               replacement.kda_snapshot_blocks == 2 &&
               replacement.kda_snapshot_bytes == 8'192,
           "snapshot replacement must retransmit its full fixed payload");
    expect(manager.commit_offload(replacement.reservation_id, at(2.2)) &&
               manager.kda_snapshot_frontier_blocks(SessionId{71}) == 4 &&
               manager.lookup(SessionId{71}, 8).hit_blocks == 4,
           "newer KDA generation must atomically replace the frontier");
    expect(manager.release_restore(lease, true, at(2.3)),
           "KDA restore lease must release exactly once");

    const std::uint64_t truncated_before = manager.stats().truncated_offloads;
    const auto no_op = manager.reserve_offload(
        SessionId{71}, CpuOffloadGeneration{3}, 4, at(3.0));
    expect(!no_op.requires_transfer() && !no_op.truncated &&
               no_op.kda_snapshot_blocks == 0 &&
               manager.stats().truncated_offloads == truncated_before,
           "equal KDA frontier must avoid transfer and truncation metrics");
    manager.validate_invariants();
}

void test_kda_snapshot_only_offload_and_restore_pin() {
    CpuKVCacheManager manager{4, CpuKVCacheCapacityPressurePolicy::kPrefixFit};
    manager.configure_kda_snapshot(2, 8'192);

    const auto reservation = manager.reserve_offload(
        SessionId{72}, CpuOffloadGeneration{1}, 0, at(1.0));
    expect(reservation.requires_transfer() &&
               reservation.reserved_blocks == 0 &&
               reservation.kda_snapshot_blocks == 2 &&
               reservation.kda_snapshot_bytes == 8'192,
           "a sub-block KDA prefix must reserve one snapshot-only payload");
    expect(manager.commit_offload(reservation.reservation_id, at(1.1)) &&
               manager.committed_frontier_blocks(SessionId{72}) == 0 &&
               manager.has_kda_snapshot(SessionId{72}) &&
               manager.kda_snapshot_frontier_blocks(SessionId{72}) == 0,
           "snapshot-only commit must preserve explicit zero-frontier state");

    const auto lease = manager.pin_restore(SessionId{72}, 0, 0, at(2.0));
    expect(manager.restore_includes_kda_snapshot(lease) &&
               manager.restore_kda_snapshot_bytes(lease) == 8'192 &&
               manager.diagnostics().pinned_blocks == 0 &&
               manager.diagnostics().pinned_kda_snapshots == 1,
           "an empty KV range must be valid only as a snapshot restore lease");
    expect(manager.release_restore(lease, true, at(2.1)),
           "snapshot-only restore lease must release normally");
    manager.validate_invariants();
}

void test_kda_snapshot_eviction_is_atomic_after_normal_kv() {
    CpuKVCacheManager manager{6, CpuKVCacheCapacityPressurePolicy::kPrefixFit};
    manager.configure_kda_snapshot(2, 2'048);
    commit(manager, 81, 1, 2, 1.0);
    commit(manager, 82, 1, 2, 2.0);
    expect(manager.diagnostics().resident_blocks == 2 &&
               manager.kda_snapshot_occupied_blocks() == 4,
           "fixed KDA snapshots must share CPU capacity with ordinary KV");

    // The fourth snapshot forces one whole inactive snapshot group out.  The
    // manager completes the oldest session's eviction chain (ordinary KV,
    // then its snapshot) before considering the newer session's KV, and
    // never tears a snapshot into partial blocks.
    commit(manager, 83, 1, 0, 3.0);
    commit(manager, 84, 1, 0, 4.0);
    expect(manager.stats().evicted_blocks >= 4 &&
               manager.stats().evicted_kda_snapshots == 1 &&
               manager.stats().evicted_kda_snapshot_blocks == 2 &&
               !manager.has_kda_snapshot(SessionId{81}) &&
               manager.has_kda_snapshot(SessionId{82}) &&
               manager.committed_frontier_blocks(SessionId{82}) == 0 &&
               manager.has_kda_snapshot(SessionId{83}) &&
               manager.has_kda_snapshot(SessionId{84}) &&
               manager.kda_snapshot_occupied_blocks() == 6,
           "snapshot pressure must finish one session chain before the next");
    manager.validate_invariants();
}

void test_kda_session_lru_prefers_older_snapshot_before_newer_kv() {
    CpuKVCacheManager manager{8, CpuKVCacheCapacityPressurePolicy::kPrefixFit};
    manager.configure_kda_snapshot(2);
    commit(manager, 111, 1, 2, 1.0);
    commit(manager, 112, 1, 2, 2.0);

    // This allocation needs exactly the two blocks in session 111's ordinary
    // suffix.  Its snapshot must remain as an LRU-only session after the
    // pressure event stops at that exact fit.
    commit(manager, 113, 1, 0, 3.0);
    expect(manager.committed_frontier_blocks(SessionId{111}) == 0 &&
               manager.has_kda_snapshot(SessionId{111}) &&
               manager.committed_frontier_blocks(SessionId{112}) == 2,
           "exact ordinary fit must leave the oldest snapshot-only session");

    // The next pressure event must choose that older snapshot-only session,
    // rather than trimming newer session 112's ordinary KV first.
    commit(manager, 114, 1, 0, 4.0);
    expect(!manager.has_kda_snapshot(SessionId{111}) &&
               manager.has_kda_snapshot(SessionId{112}) &&
               manager.committed_frontier_blocks(SessionId{112}) == 2 &&
               manager.has_kda_snapshot(SessionId{113}) &&
               manager.has_kda_snapshot(SessionId{114}) &&
               manager.stats().evicted_kda_snapshots == 1,
           "session-level LRU must evict the older snapshot before newer KV");
    manager.validate_invariants();
}

void test_kda_snapshot_eviction_reclaims_whole_fixed_charge() {
    CpuKVCacheManager manager{6, CpuKVCacheCapacityPressurePolicy::kPrefixFit};
    manager.configure_kda_snapshot(2);
    commit(manager, 121, 1, 1, 1.0);
    commit(manager, 122, 1, 1, 2.0);

    // Only one free block is needed after trimming session 121's ordinary KV,
    // but its two-block snapshot is the next chain step and must be removed
    // atomically as a whole.
    commit(manager, 123, 1, 0, 3.0);
    expect(!manager.has_kda_snapshot(SessionId{121}) &&
               manager.has_kda_snapshot(SessionId{122}) &&
               manager.has_kda_snapshot(SessionId{123}) &&
               manager.stats().evicted_kda_snapshots == 1 &&
               manager.stats().evicted_kda_snapshot_blocks == 2 &&
               manager.diagnostics().kda_snapshot_occupied_blocks == 4,
           "KDA eviction must reclaim a fixed snapshot charge atomically");
    manager.validate_invariants();
}

void test_kda_skip_offload_uses_session_chain_reclaimability() {
    CpuKVCacheManager manager{8,
                              CpuKVCacheCapacityPressurePolicy::kSkipOffload};
    manager.configure_kda_snapshot(2);
    commit(manager, 131, 1, 2, 1.0);
    commit(manager, 132, 1, 2, 2.0);
    commit(manager, 133, 1, 0, 3.0);

    // The skip estimate must include both ordinary and fixed-charge reclaim
    // steps, while the actual pressure path keeps them in one session chain.
    commit(manager, 134, 1, 0, 4.0);
    expect(!manager.has_kda_snapshot(SessionId{131}) &&
               manager.has_kda_snapshot(SessionId{132}) &&
               manager.committed_frontier_blocks(SessionId{132}) == 2 &&
               manager.has_kda_snapshot(SessionId{133}) &&
               manager.has_kda_snapshot(SessionId{134}) &&
               manager.stats().skipped_offloads == 0,
           "skip_offload must estimate and execute the same session chain");
    manager.validate_invariants();
}

void test_kda_snapshot_discard_waits_for_restore_pin() {
    CpuKVCacheManager manager{8, CpuKVCacheCapacityPressurePolicy::kPrefixFit};
    manager.configure_kda_snapshot(2, 4'096);
    commit(manager, 91, 1, 3, 1.0);
    const auto lease = manager.pin_restore(SessionId{91}, 0, 2, at(2.0));
    expect(manager.discard_session(SessionId{91}) &&
               manager.session_discard_pending(SessionId{91}) &&
               manager.lookup(SessionId{91}, 3).hit_blocks == 0 &&
               manager.has_kda_snapshot(SessionId{91}),
           "discard must hide but retain pinned KDA state until release");
    expect(manager.release_restore(lease, true, at(2.1)) &&
               !manager.has_kda_snapshot(SessionId{91}) &&
               manager.diagnostics().materialized_blocks == 0 &&
               manager.diagnostics().kda_snapshot_occupied_blocks == 0,
           "discarded pinned KDA session must reap atomically on release");
    manager.validate_invariants();
}

void test_kda_snapshot_only_discard_reaps_after_zero_block_update() {
    CpuKVCacheManager manager{2, CpuKVCacheCapacityPressurePolicy::kPrefixFit};
    manager.configure_kda_snapshot(2, 4'096);
    commit(manager, 92, 1, 0, 1.0);
    const auto update = manager.reserve_offload(
        SessionId{92}, CpuOffloadGeneration{2}, 2, at(2.0));
    expect(update.requires_transfer() && update.reserved_blocks == 0 &&
               update.kda_snapshot_blocks == 2 && update.truncated,
           "snapshot-frontier update must remain transferable when ordinary "
           "KV admits zero blocks");
    expect(manager.discard_session(SessionId{92}) &&
               manager.commit_offload(update.reservation_id, at(2.1)),
           "discarded zero-block snapshot update must drain its completion");
    const auto diagnostics = manager.diagnostics();
    expect(diagnostics.sessions == 0 &&
               diagnostics.kda_snapshot_sessions == 0 &&
               diagnostics.kda_snapshot_occupied_blocks == 0 &&
               diagnostics.active_reservations == 0 &&
               manager.stats().discarded_offload_completions == 1,
           "snapshot-only discard must reap already-erased session metadata "
           "without reusing its iterator");
    manager.validate_invariants();
}

void test_kda_first_snapshot_out_of_order_generation_commit() {
    CpuKVCacheManager manager{20, CpuKVCacheCapacityPressurePolicy::kPrefixFit};
    manager.configure_kda_snapshot(2, 4'096);
    const auto older = manager.reserve_offload(
        SessionId{101}, CpuOffloadGeneration{1}, 2, at(1.0));
    const auto newer = manager.reserve_offload(
        SessionId{101}, CpuOffloadGeneration{2}, 4, at(1.1));
    expect(older.requires_transfer() && newer.requires_transfer() &&
               manager.commit_offload(newer.reservation_id, at(1.2)) &&
               manager.kda_snapshot_frontier_blocks(SessionId{101}) == 4,
           "newer first-snapshot generation must consume the shared atomic "
           "slot");
    const auto between = manager.diagnostics();
    expect(between.active_reservations == 1 &&
               between.kda_snapshot_reserved_blocks == 0 &&
               between.kda_snapshot_occupied_blocks == 2,
           "consuming a shared snapshot slot must clear its original owner's "
           "reservation flag before the stale completion arrives");
    expect(manager.commit_offload(older.reservation_id, at(1.3)) &&
               manager.lookup(SessionId{101}, 4).hit_blocks == 4 &&
               manager.diagnostics().reserved_blocks == 0 &&
               manager.diagnostics().kda_snapshot_reserved_blocks == 0 &&
               manager.kda_snapshot_occupied_blocks() == 2,
           "late stale generation must not regress or leak the snapshot slot");
    manager.validate_invariants();
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
    failures += frontier::test::run("CPU incremental occupancy accounting",
                                    test_incremental_occupancy_accounting);
    failures += frontier::test::run(
        "CPU overlapping restore ranges count distinct pins",
        test_overlapping_restore_ranges_count_distinct_pins);
    failures +=
        frontier::test::run("CPU large resident range uses constant metadata",
                            test_large_resident_range_has_constant_metadata);
    failures +=
        frontier::test::run("CPU out-of-order commit and dependent abort",
                            test_out_of_order_commit_and_dependent_abort);
    failures += frontier::test::run("CPU no-op LRU and terminal empty paths",
                                    test_noop_lru_empty_metadata_and_zero_fit);
    failures +=
        frontier::test::run("CPU migration discard drains in-flight transfers",
                            test_migration_discard_drains_inflight_transfers);
    failures += frontier::test::run("CPU randomized invariants",
                                    test_randomized_invariants);
    failures += frontier::test::run(
        "CPU randomized out-of-order ranges and overlapping leases",
        test_randomized_out_of_order_ranges_and_overlapping_leases);
    failures += frontier::test::run(
        "CPU KDA snapshot replacement and restore pin",
        test_kda_snapshot_offload_replacement_and_restore_pin);
    failures +=
        frontier::test::run("CPU KDA snapshot-only offload and restore pin",
                            test_kda_snapshot_only_offload_and_restore_pin);
    failures += frontier::test::run(
        "CPU KDA snapshot atomic eviction after ordinary KV",
        test_kda_snapshot_eviction_is_atomic_after_normal_kv);
    failures += frontier::test::run(
        "CPU KDA session LRU prefers old snapshot before newer KV",
        test_kda_session_lru_prefers_older_snapshot_before_newer_kv);
    failures += frontier::test::run(
        "CPU KDA snapshot eviction uses whole fixed charge",
        test_kda_snapshot_eviction_reclaims_whole_fixed_charge);
    failures += frontier::test::run(
        "CPU KDA skip offload follows session chain",
        test_kda_skip_offload_uses_session_chain_reclaimability);
    failures +=
        frontier::test::run("CPU KDA snapshot discard waits for restore pin",
                            test_kda_snapshot_discard_waits_for_restore_pin);
    failures += frontier::test::run(
        "CPU KDA snapshot-only discard drains zero-block update",
        test_kda_snapshot_only_discard_reaps_after_zero_block_update);
    failures += frontier::test::run(
        "CPU KDA first snapshot out-of-order generation commit",
        test_kda_first_snapshot_out_of_order_generation_commit);
    return failures == 0 ? 0 : 1;
}
