#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace frontier {

template <typename Tag> class StrongId {
  public:
    using ValueType = std::int64_t;
    static constexpr ValueType kInvalidValue = -1;

    constexpr StrongId() noexcept = default;
    template <typename Integer,
              typename = std::enable_if_t<std::is_integral_v<Integer>>>
    explicit constexpr StrongId(Integer value) : value_(checked_value(value)) {}

    [[nodiscard]] constexpr ValueType value() const noexcept { return value_; }
    [[nodiscard]] constexpr bool valid() const noexcept { return value_ >= 0; }
    [[nodiscard]] constexpr std::size_t index() const noexcept {
        return static_cast<std::size_t>(value_);
    }

    friend constexpr bool operator==(StrongId lhs, StrongId rhs) noexcept {
        return lhs.value_ == rhs.value_;
    }
    friend constexpr bool operator!=(StrongId lhs, StrongId rhs) noexcept {
        return !(lhs == rhs);
    }
    friend constexpr bool operator<(StrongId lhs, StrongId rhs) noexcept {
        return lhs.value_ < rhs.value_;
    }
    friend constexpr bool operator<=(StrongId lhs, StrongId rhs) noexcept {
        return !(rhs < lhs);
    }
    friend constexpr bool operator>(StrongId lhs, StrongId rhs) noexcept {
        return rhs < lhs;
    }
    friend constexpr bool operator>=(StrongId lhs, StrongId rhs) noexcept {
        return !(lhs < rhs);
    }

  private:
    template <typename Integer>
    [[nodiscard]] static constexpr ValueType checked_value(Integer value) {
        if constexpr (std::is_signed_v<Integer>) {
            if (value < static_cast<Integer>(kInvalidValue) ||
                (sizeof(Integer) > sizeof(ValueType) &&
                 value > static_cast<Integer>(
                             std::numeric_limits<ValueType>::max()))) {
                throw std::out_of_range("strong ID is outside int64 range");
            }
        } else if (value > static_cast<std::make_unsigned_t<ValueType>>(
                               std::numeric_limits<ValueType>::max())) {
            throw std::out_of_range("strong ID is outside int64 range");
        }
        return static_cast<ValueType>(value);
    }

    ValueType value_ = kInvalidValue;
};

template <typename Id> struct StrongIdHash {
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
struct CpuBlockIdTag;
struct CpuOffloadReservationIdTag;
struct CpuRestoreLeaseIdTag;
struct CpuKvTransferIdTag;
struct CpuOffloadGenerationTag;

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
using CpuBlockId = StrongId<CpuBlockIdTag>;
using CpuOffloadReservationId = StrongId<CpuOffloadReservationIdTag>;
using CpuRestoreLeaseId = StrongId<CpuRestoreLeaseIdTag>;
using CpuKvTransferId = StrongId<CpuKvTransferIdTag>;
using CpuOffloadGeneration = StrongId<CpuOffloadGenerationTag>;

} // namespace frontier
