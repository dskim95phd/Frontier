#include "frontier/scheduler/cluster_scheduler/base_cluster_scheduler.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace frontier::scheduler {

std::optional<MoEBarrierReady>
MoEBarrierCoordinator::arrive(const MoEBarrierKey &key,
                              MoEBarrierParticipant participant,
                              std::uint64_t expected_participants) {
    if (!key.replica_id.valid() || !key.stage_id.valid() ||
        !key.sync_group_id.valid() || !key.layer_id.valid() ||
        !key.generation.valid()) {
        throw MoEBarrierError("MoE barrier key contains an invalid ID");
    }
    if (expected_participants == 0 || !participant.participant_id.valid() ||
        static_cast<std::uint64_t>(participant.participant_id.value()) >=
            expected_participants) {
        throw MoEBarrierError("MoE participant is outside the barrier domain");
    }
    if (!participant.arrival_time.valid() ||
        !std::isfinite(participant.elapsed_component_ms) ||
        participant.elapsed_component_ms < 0.0) {
        throw MoEBarrierError("MoE barrier participant timing is invalid");
    }
    if (consumed_.find(key) != consumed_.end()) {
        return std::nullopt;
    }

    Entry &entry = waiting_[key];
    if (entry.expected_participants == 0) {
        entry.expected_participants = expected_participants;
    } else if (entry.expected_participants != expected_participants) {
        throw MoEBarrierError("MoE barrier participant count changed");
    }

    const auto position = entry.participants.find(participant.participant_id);
    if (position == entry.participants.end()) {
        entry.participants.emplace(participant.participant_id,
                                   std::move(participant));
    } else if (position->second.is_idle && !participant.is_idle) {
        position->second = std::move(participant);
    } else if (!position->second.is_idle && participant.is_idle) {
        return maybe_ready(key, entry);
    } else if (position->second == participant) {
        return maybe_ready(key, entry);
    } else {
        throw MoEBarrierError("duplicate MoE barrier participant");
    }
    return maybe_ready(key, entry);
}

std::optional<MoEBarrierReady>
MoEBarrierCoordinator::compact_missing_idle(const MoEBarrierKey &key,
                                            std::uint64_t expected_participants,
                                            SimTime arrival_time) {
    std::optional<MoEBarrierReady> ready;
    for (std::uint64_t participant = 0; participant < expected_participants;
         ++participant) {
        const MoEParticipantId id{participant};
        const auto waiting = waiting_.find(key);
        if (waiting != waiting_.end() &&
            waiting->second.participants.find(id) !=
                waiting->second.participants.end()) {
            continue;
        }
        ready = arrive(
            key,
            [&]() {
                MoEBarrierParticipant value{};
                value.participant_id = id;
                value.batch_id = BatchId{};
                value.arrival_time = arrival_time;
                value.elapsed_component_ms = 0.0;
                value.is_idle = true;
                return value;
            }(),
            expected_participants);
    }
    return ready;
}

std::optional<MoEBarrierReady>
MoEBarrierCoordinator::maybe_ready(const MoEBarrierKey &key, Entry &entry) {
    if (entry.collective_emitted ||
        entry.participants.size() != entry.expected_participants) {
        return std::nullopt;
    }
    SimTime maximum;
    for (const auto &[unused, participant] : entry.participants) {
        static_cast<void>(unused);
        if (!maximum.valid() || participant.arrival_time > maximum) {
            maximum = participant.arrival_time;
        }
    }
    entry.collective_emitted = true;
    return [&]() {
        MoEBarrierReady value{};
        value.key = key;
        value.collective_time = maximum;
        return value;
    }();
}

std::vector<MoEBarrierParticipant>
MoEBarrierCoordinator::consume(const MoEBarrierKey &key) {
    if (consumed_.find(key) != consumed_.end()) {
        return {};
    }
    const auto position = waiting_.find(key);
    if (position == waiting_.end()) {
        return {};
    }
    if (!position->second.collective_emitted ||
        position->second.participants.size() !=
            position->second.expected_participants) {
        throw MoEBarrierError(
            "MoE barrier consumed before collective readiness");
    }
    std::vector<MoEBarrierParticipant> participants;
    participants.reserve(position->second.participants.size());
    for (auto &[unused, participant] : position->second.participants) {
        static_cast<void>(unused);
        participants.push_back(std::move(participant));
    }
    waiting_.erase(position);
    consumed_.emplace(key, true);
    return participants;
}

void MoEBarrierCoordinator::require_empty() const {
    if (!waiting_.empty()) {
        throw MoEBarrierError(
            "MoE barrier state remains at simulator quiescence");
    }
}

} // namespace frontier::scheduler
