#pragma once

#include <compare>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>

namespace frontier {

template <typename Tag>
class StrongId {
 public:
  using ValueType = std::int64_t;
  static constexpr ValueType kInvalidValue = -1;

  constexpr StrongId() noexcept = default;
  template <std::integral Integer>
  explicit constexpr StrongId(Integer value) noexcept
      : value_(static_cast<ValueType>(value)) {}

  [[nodiscard]] constexpr ValueType value() const noexcept { return value_; }
  [[nodiscard]] constexpr bool valid() const noexcept {
    return value_ >= 0;
  }
  [[nodiscard]] constexpr std::size_t index() const noexcept {
    return static_cast<std::size_t>(value_);
  }

  friend constexpr bool operator==(StrongId, StrongId) noexcept = default;
  friend constexpr auto operator<=>(StrongId, StrongId) noexcept = default;

 private:
  ValueType value_ = kInvalidValue;
};

template <typename Id>
struct StrongIdHash {
  [[nodiscard]] std::size_t operator()(Id id) const noexcept {
    return std::hash<typename Id::ValueType>{}(id.value());
  }
};

struct EventSequenceTag;
struct RequestIdTag;
struct BatchIdTag;
struct BatchGlobalIdTag;
struct ReplicaIdTag;
struct DataParallelIdTag;
struct StageIdTag;
struct SessionIdTag;
struct GenerationTag;
struct IterationIdTag;
struct TransferIdTag;
struct MoESyncGroupIdTag;
struct MoEParticipantIdTag;
struct LayerIdTag;

using EventSequence = StrongId<EventSequenceTag>;
using RequestId = StrongId<RequestIdTag>;
using BatchId = StrongId<BatchIdTag>;
using BatchGlobalId = StrongId<BatchGlobalIdTag>;
using ReplicaId = StrongId<ReplicaIdTag>;
using DataParallelId = StrongId<DataParallelIdTag>;
using StageId = StrongId<StageIdTag>;
using SessionId = StrongId<SessionIdTag>;
using Generation = StrongId<GenerationTag>;
using IterationId = StrongId<IterationIdTag>;
using TransferId = StrongId<TransferIdTag>;
using MoESyncGroupId = StrongId<MoESyncGroupIdTag>;
using MoEParticipantId = StrongId<MoEParticipantIdTag>;
using LayerId = StrongId<LayerIdTag>;

}  // namespace frontier
