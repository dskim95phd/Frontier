#pragma once

#include <cstdint>
#include <optional>

#include "frontier/core/ids.h"

namespace frontier {

class SimTime {
 public:
  [[nodiscard]] static constexpr SimTime from_seconds(double seconds) noexcept {
    return SimTime{seconds};
  }

  [[nodiscard]] constexpr double seconds() const noexcept { return seconds_; }

  friend constexpr bool operator==(SimTime, SimTime) noexcept = default;
  friend constexpr auto operator<=>(SimTime, SimTime) noexcept = default;

 private:
  explicit constexpr SimTime(double seconds) noexcept : seconds_(seconds) {}

  double seconds_;
};

enum class EventType : std::uint8_t {
  kRequestArrival,
  kFoundationCompletion,
  kSchedulerPoll,
  kBatchCompletion,
  kGlobalSchedule,
  kClusterSchedule,
  kReplicaSchedule,
  kBatchStageArrival,
  kReplicaStageSchedule,
  kBatchStageEnd,
  kClusterBatchEnd,
  kGlobalBatchEnd,
};

struct EventPayload {
  std::optional<RequestId> request_id;
  std::optional<BatchId> batch_id;
  std::optional<ReplicaId> replica_id;
  std::optional<DataParallelId> dp_id;
  std::optional<StageId> stage_id;
  std::optional<Generation> generation;
};

struct Event {
  SimTime time;
  EventSequence sequence;
  EventType type;
  EventPayload payload;

  [[nodiscard]] bool is_stale(Generation current_generation) const noexcept {
    return payload.generation.has_value() &&
           payload.generation.value() != current_generation;
  }
};

}  // namespace frontier
