#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "frontier/core/event.h"
#include "frontier/core/ids.h"

namespace frontier::scheduler {

class MoEBarrierError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

struct MoEBarrierKey {
    ClusterType cluster_type;
    ReplicaId replica_id;
    StageId stage_id;
    MoESyncGroupId sync_group_id;
    LayerId layer_id;
    MoESyncPhase phase;
    Generation generation;

    friend bool operator==(const MoEBarrierKey &lhs, const MoEBarrierKey &rhs) {
        return std::tie(lhs.cluster_type, lhs.replica_id, lhs.stage_id,
                        lhs.sync_group_id, lhs.layer_id, lhs.phase,
                        lhs.generation) ==
               std::tie(rhs.cluster_type, rhs.replica_id, rhs.stage_id,
                        rhs.sync_group_id, rhs.layer_id, rhs.phase,
                        rhs.generation);
    }
    friend bool operator<(const MoEBarrierKey &lhs, const MoEBarrierKey &rhs) {
        return std::tie(lhs.cluster_type, lhs.replica_id, lhs.stage_id,
                        lhs.sync_group_id, lhs.layer_id, lhs.phase,
                        lhs.generation) <
               std::tie(rhs.cluster_type, rhs.replica_id, rhs.stage_id,
                        rhs.sync_group_id, rhs.layer_id, rhs.phase,
                        rhs.generation);
    }
};

struct MoEBarrierParticipant {
    MoEParticipantId participant_id;
    BatchId batch_id;
    SimTime arrival_time;
    double elapsed_component_ms = 0.0;
    bool is_idle = false;

    friend bool operator==(const MoEBarrierParticipant &lhs,
                           const MoEBarrierParticipant &rhs) {
        return std::tie(lhs.participant_id, lhs.batch_id, lhs.arrival_time,
                        lhs.elapsed_component_ms, lhs.is_idle) ==
               std::tie(rhs.participant_id, rhs.batch_id, rhs.arrival_time,
                        rhs.elapsed_component_ms, rhs.is_idle);
    }
};

struct MoEBarrierReady {
    MoEBarrierKey key;
    SimTime collective_time;
};

// Coordinates one deterministic collective barrier per MoE layer and phase.
// It owns no simulator or scheduler state, which keeps barrier lifecycle and
// duplicate-arrival rules independently testable.
class MoEBarrierCoordinator {
  public:
    [[nodiscard]] std::optional<MoEBarrierReady>
    arrive(const MoEBarrierKey &key, MoEBarrierParticipant participant,
           std::uint64_t expected_participants);
    [[nodiscard]] std::optional<MoEBarrierReady>
    compact_missing_idle(const MoEBarrierKey &key,
                         std::uint64_t expected_participants,
                         SimTime arrival_time);
    [[nodiscard]] std::vector<MoEBarrierParticipant>
    consume(const MoEBarrierKey &key);
    void require_empty() const;

  private:
    struct Entry {
        std::uint64_t expected_participants = 0;
        std::map<MoEParticipantId, MoEBarrierParticipant> participants;
        bool collective_emitted = false;
    };

    [[nodiscard]] std::optional<MoEBarrierReady>
    maybe_ready(const MoEBarrierKey &key, Entry &entry);

    std::map<MoEBarrierKey, Entry> waiting_;
    std::map<MoEBarrierKey, bool> consumed_;
};

} // namespace frontier::scheduler
