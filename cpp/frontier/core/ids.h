#pragma once

#include <compare>
#include <cstdint>
#include <functional>

namespace frontier {

template <typename Tag>
class StrongId {
 public:
  using ValueType = std::uint64_t;

  explicit constexpr StrongId(ValueType value) noexcept : value_(value) {}

  [[nodiscard]] constexpr ValueType value() const noexcept { return value_; }

  friend constexpr bool operator==(StrongId, StrongId) noexcept = default;
  friend constexpr auto operator<=>(StrongId, StrongId) noexcept = default;

 private:
  ValueType value_;
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
struct ReplicaIdTag;
struct DataParallelIdTag;
struct SessionIdTag;
struct GenerationTag;

using EventSequence = StrongId<EventSequenceTag>;
using RequestId = StrongId<RequestIdTag>;
using BatchId = StrongId<BatchIdTag>;
using ReplicaId = StrongId<ReplicaIdTag>;
using DataParallelId = StrongId<DataParallelIdTag>;
using SessionId = StrongId<SessionIdTag>;
using Generation = StrongId<GenerationTag>;

}  // namespace frontier
