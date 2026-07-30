#include "frontier/moe/barrier_coordinator.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace frontier::moe {

std::optional<BarrierReady> BarrierCoordinator::arrive(
    const BarrierKey& key,
    BarrierParticipant participant,
    std::uint64_t expected_participants) {
  if (!key.replica_id.valid() || !key.stage_id.valid() ||
      !key.sync_group_id.valid() || !key.layer_id.valid() ||
      !key.generation.valid()) {
    throw BarrierError("MoE barrier key contains an invalid ID");
  }
  if (expected_participants == 0 ||
      !participant.participant_id.valid() ||
      static_cast<std::uint64_t>(participant.participant_id.value()) >=
          expected_participants) {
    throw BarrierError("MoE participant is outside the barrier domain");
  }
  if (!participant.arrival_time.valid() ||
      !std::isfinite(participant.elapsed_component_ms) ||
      participant.elapsed_component_ms < 0.0) {
    throw BarrierError("MoE barrier participant timing is invalid");
  }
  if (consumed_.contains(key)) {
    return std::nullopt;
  }

  Entry& entry = waiting_[key];
  if (entry.expected_participants == 0) {
    entry.expected_participants = expected_participants;
  } else if (entry.expected_participants != expected_participants) {
    throw BarrierError("MoE barrier participant count changed");
  }

  const auto position =
      entry.participants.find(participant.participant_id);
  if (position == entry.participants.end()) {
    entry.participants.emplace(
        participant.participant_id, std::move(participant));
  } else if (position->second.is_idle && !participant.is_idle) {
    position->second = std::move(participant);
  } else if (!position->second.is_idle && participant.is_idle) {
    return maybe_ready(key, entry);
  } else if (position->second == participant) {
    return maybe_ready(key, entry);
  } else {
    throw BarrierError("duplicate MoE barrier participant");
  }
  return maybe_ready(key, entry);
}

std::optional<BarrierReady> BarrierCoordinator::compact_missing_idle(
    const BarrierKey& key,
    std::uint64_t expected_participants,
    SimTime arrival_time) {
  std::optional<BarrierReady> ready;
  for (std::uint64_t participant = 0;
       participant < expected_participants;
       ++participant) {
    const MoEParticipantId id{participant};
    const auto waiting = waiting_.find(key);
    if (waiting != waiting_.end() &&
        waiting->second.participants.contains(id)) {
      continue;
    }
    ready = arrive(
        key,
        BarrierParticipant{
            .participant_id = id,
            .batch_id = BatchId{},
            .arrival_time = arrival_time,
            .elapsed_component_ms = 0.0,
            .is_idle = true,
        },
        expected_participants);
  }
  return ready;
}

std::optional<BarrierReady> BarrierCoordinator::maybe_ready(
    const BarrierKey& key,
    Entry& entry) {
  if (entry.collective_emitted ||
      entry.participants.size() != entry.expected_participants) {
    return std::nullopt;
  }
  SimTime maximum;
  for (const auto& [unused, participant] : entry.participants) {
    static_cast<void>(unused);
    if (!maximum.valid() || participant.arrival_time > maximum) {
      maximum = participant.arrival_time;
    }
  }
  entry.collective_emitted = true;
  return BarrierReady{.key = key, .collective_time = maximum};
}

std::vector<BarrierParticipant> BarrierCoordinator::consume(
    const BarrierKey& key) {
  if (consumed_.contains(key)) {
    return {};
  }
  const auto position = waiting_.find(key);
  if (position == waiting_.end()) {
    return {};
  }
  if (!position->second.collective_emitted ||
      position->second.participants.size() !=
          position->second.expected_participants) {
    throw BarrierError("MoE barrier consumed before collective readiness");
  }
  std::vector<BarrierParticipant> participants;
  participants.reserve(position->second.participants.size());
  for (auto& [unused, participant] : position->second.participants) {
    static_cast<void>(unused);
    participants.push_back(std::move(participant));
  }
  waiting_.erase(position);
  consumed_.emplace(key, true);
  return participants;
}

void BarrierCoordinator::require_empty() const {
  if (!waiting_.empty()) {
    throw BarrierError("MoE barrier state remains at simulator quiescence");
  }
}

}  // namespace frontier::moe
