#include <iostream>

#include "frontier/scheduler/cluster_scheduler/moe_barrier_coordinator.h"
#include "tests/test_support.h"

namespace {

using frontier::BatchId;
using frontier::ClusterType;
using frontier::Generation;
using frontier::LayerId;
using frontier::MoEParticipantId;
using frontier::MoESyncGroupId;
using frontier::MoESyncPhase;
using frontier::ReplicaId;
using frontier::SimTime;
using frontier::StageId;
using frontier::scheduler::MoEBarrierCoordinator;
using frontier::scheduler::MoEBarrierError;
using frontier::scheduler::MoEBarrierKey;
using frontier::scheduler::MoEBarrierParticipant;
using frontier::test::expect;
using frontier::test::expect_throws;

MoEBarrierKey key() {
    MoEBarrierKey value{};
    value.cluster_type = ClusterType::kMonolithic;
    value.replica_id = ReplicaId{0};
    value.stage_id = StageId{0};
    value.sync_group_id = MoESyncGroupId{0};
    value.layer_id = LayerId{0};
    value.phase = MoESyncPhase::kPreMoe;
    value.generation = Generation{0};
    return value;
}

MoEBarrierParticipant participant(std::uint64_t id, std::uint64_t batch,
                                  double arrival_s) {
    MoEBarrierParticipant value{};
    value.participant_id = MoEParticipantId{id};
    value.batch_id = BatchId{batch};
    value.arrival_time = SimTime::from_seconds(arrival_s);
    return value;
}

void test_collective_releases_at_latest_arrival() {
    MoEBarrierCoordinator coordinator;
    expect(!coordinator.arrive(key(), participant(1, 11, 1.0), 2).has_value(),
           "first participant must not release a two-way barrier");
    const auto ready = coordinator.arrive(key(), participant(0, 10, 2.0), 2);
    expect(ready.has_value() &&
               ready->collective_time == SimTime::from_seconds(2.0),
           "barrier must release at the latest participant arrival");
    expect_throws<MoEBarrierError>(
        [&coordinator] { coordinator.require_empty(); },
        "ready but unconsumed barrier must not be quiescent");

    const auto participants = coordinator.consume(key());
    expect(participants.size() == 2 &&
               participants[0].participant_id == MoEParticipantId{0} &&
               participants[1].participant_id == MoEParticipantId{1},
           "consumption must return deterministic participant order");
    expect(coordinator.consume(key()).empty(),
           "barrier consumption must be idempotent");
    coordinator.require_empty();
}

void test_real_arrival_replaces_compacted_idle_participant() {
    MoEBarrierCoordinator coordinator;
    expect(!coordinator.arrive(key(), participant(0, 20, 1.0), 2).has_value(),
           "one real lane must leave the barrier pending");
    const auto ready =
        coordinator.compact_missing_idle(key(), 2, SimTime::from_seconds(3.0));
    expect(ready.has_value() &&
               ready->collective_time == SimTime::from_seconds(3.0),
           "idle compaction must close the barrier deterministically");
    expect(!coordinator.arrive(key(), participant(1, 21, 2.0), 2).has_value(),
           "late real replacement must not emit a duplicate collective");

    const auto participants = coordinator.consume(key());
    expect(participants.size() == 2 && !participants[1].is_idle &&
               participants[1].batch_id == BatchId{21},
           "a real participant must replace its compacted idle lane");
}

void test_invalid_domain_is_rejected() {
    MoEBarrierCoordinator coordinator;
    expect_throws<MoEBarrierError>(
        [&coordinator] {
            static_cast<void>(
                coordinator.arrive(key(), participant(2, 30, 1.0), 2));
        },
        "participant IDs outside the barrier domain must fail");
    expect_throws<MoEBarrierError>(
        [&coordinator] {
            static_cast<void>(
                coordinator.arrive(key(), participant(0, 30, 1.0), 0));
        },
        "zero-sized barrier domains must fail");
}

} // namespace

int main() {
    int failures = 0;
    failures += frontier::test::run("latest arrival releases collective",
                                    test_collective_releases_at_latest_arrival);
    failures += frontier::test::run(
        "real arrival replaces compacted idle",
        test_real_arrival_replaces_compacted_idle_participant);
    failures += frontier::test::run("invalid barrier domain is rejected",
                                    test_invalid_domain_is_rejected);
    return failures == 0 ? 0 : 1;
}
