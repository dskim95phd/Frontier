#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <vector>

#include "frontier/core/cluster_type.h"
#include "frontier/core/event.h"
#include "frontier/core/ids.h"
#include "frontier/moe/synchronization.h"

namespace frontier::moe {

class BarrierError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

struct BarrierKey {
  ClusterType cluster_type;
  ReplicaId replica_id;
  StageId stage_id;
  MoESyncGroupId sync_group_id;
  LayerId layer_id;
  SyncPhase phase;
  Generation generation;

  friend bool operator==(const BarrierKey&, const BarrierKey&) = default;
  friend auto operator<=>(const BarrierKey&, const BarrierKey&) = default;
};

struct BarrierParticipant {
  MoEParticipantId participant_id;
  BatchId batch_id;
  SimTime arrival_time;
  double elapsed_component_ms = 0.0;
  bool is_idle = false;

  friend bool operator==(
      const BarrierParticipant&,
      const BarrierParticipant&) = default;
};

struct BarrierReady {
  BarrierKey key;
  SimTime collective_time;
};

class BarrierCoordinator {
 public:
  [[nodiscard]] std::optional<BarrierReady> arrive(
      const BarrierKey& key,
      BarrierParticipant participant,
      std::uint64_t expected_participants);
  [[nodiscard]] std::optional<BarrierReady> compact_missing_idle(
      const BarrierKey& key,
      std::uint64_t expected_participants,
      SimTime arrival_time);
  [[nodiscard]] std::vector<BarrierParticipant> consume(
      const BarrierKey& key);

  [[nodiscard]] bool empty() const noexcept {
    return waiting_.empty();
  }
  [[nodiscard]] std::size_t waiting_count() const noexcept {
    return waiting_.size();
  }
  void require_empty() const;

 private:
  struct Entry {
    std::uint64_t expected_participants = 0;
    std::map<MoEParticipantId, BarrierParticipant> participants;
    bool collective_emitted = false;
  };

  [[nodiscard]] std::optional<BarrierReady> maybe_ready(
      const BarrierKey& key,
      Entry& entry);

  std::map<BarrierKey, Entry> waiting_;
  std::map<BarrierKey, bool> consumed_;
};

}  // namespace frontier::moe
