#pragma once

#include <cmath>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <variant>

#include "frontier/core/cluster_type.h"
#include "frontier/core/ids.h"

namespace frontier {

class SimTime {
  public:
    constexpr SimTime() noexcept = default;

    [[nodiscard]] static constexpr SimTime
    from_seconds(double seconds) noexcept {
        return SimTime{seconds};
    }

    [[nodiscard]] constexpr double seconds() const noexcept { return seconds_; }
    [[nodiscard]] bool valid() const noexcept {
        return std::isfinite(seconds_) && seconds_ >= 0.0;
    }

    friend constexpr bool operator==(SimTime lhs, SimTime rhs) noexcept {
        return lhs.seconds_ == rhs.seconds_;
    }
    friend constexpr bool operator!=(SimTime lhs, SimTime rhs) noexcept {
        return !(lhs == rhs);
    }
    friend constexpr bool operator<(SimTime lhs, SimTime rhs) noexcept {
        return lhs.seconds_ < rhs.seconds_;
    }
    friend constexpr bool operator<=(SimTime lhs, SimTime rhs) noexcept {
        return !(rhs < lhs);
    }
    friend constexpr bool operator>(SimTime lhs, SimTime rhs) noexcept {
        return rhs < lhs;
    }
    friend constexpr bool operator>=(SimTime lhs, SimTime rhs) noexcept {
        return !(lhs < rhs);
    }

  private:
    explicit constexpr SimTime(double seconds) noexcept : seconds_(seconds) {}

    double seconds_ = -1.0;
};

enum class EventType : std::uint8_t {
    kRequestArrival,
    kGlobalSchedule,
    kClusterSchedule,
    kReplicaSchedule,
    kBatchStageArrival,
    kReplicaStageSchedule,
    kBatchStageEnd,
    kClusterBatchEnd,
    kGlobalBatchEnd,
    kKvCacheTransferStart,
    kKvCacheTransferEnd,
    kPrefillSync,
    kPrefillSyncCollective,
    kDecodeSync,
    kDecodeSyncCollective,
};

enum class MoESyncPhase : std::uint8_t {
    kPreMoe,
    kPostMoe,
};

struct RequestArrivalPayload {
    static constexpr EventType kType = EventType::kRequestArrival;
    RequestId request_id;
    ClusterType cluster_type;
};

struct GlobalSchedulePayload {
    static constexpr EventType kType = EventType::kGlobalSchedule;
    ClusterType cluster_type;
};

struct ClusterSchedulePayload {
    static constexpr EventType kType = EventType::kClusterSchedule;
    ClusterType cluster_type;
};

struct ReplicaSchedulePayload {
    static constexpr EventType kType = EventType::kReplicaSchedule;
    ReplicaId replica_id;
    DataParallelId dp_id;
    ClusterType cluster_type;
};

struct BatchStageArrivalPayload {
    static constexpr EventType kType = EventType::kBatchStageArrival;
    BatchId batch_id;
    ReplicaId replica_id;
    DataParallelId dp_id;
    StageId stage_id;
    Generation generation;
    ClusterType cluster_type;
};

struct ReplicaStageSchedulePayload {
    static constexpr EventType kType = EventType::kReplicaStageSchedule;
    ReplicaId replica_id;
    DataParallelId dp_id;
    StageId stage_id;
    ClusterType cluster_type;
};

struct BatchStageEndPayload {
    static constexpr EventType kType = EventType::kBatchStageEnd;
    BatchId batch_id;
    ReplicaId replica_id;
    DataParallelId dp_id;
    StageId stage_id;
    Generation generation;
    ClusterType cluster_type;
};

struct ClusterBatchEndPayload {
    static constexpr EventType kType = EventType::kClusterBatchEnd;
    BatchId batch_id;
    ReplicaId replica_id;
    DataParallelId dp_id;
    Generation generation;
    ClusterType cluster_type;
};

struct GlobalBatchEndPayload {
    static constexpr EventType kType = EventType::kGlobalBatchEnd;
    BatchId batch_id;
    ReplicaId replica_id;
    DataParallelId dp_id;
    Generation generation;
    ClusterType cluster_type;
};

struct KVCacheTransferStartPayload {
    static constexpr EventType kType = EventType::kKvCacheTransferStart;
    TransferId transfer_id;
    RequestId request_id;
    BatchId batch_id;
    ReplicaId replica_id;
    DataParallelId dp_id;
    Generation generation;
    ClusterType cluster_type;
};

struct KVCacheTransferEndPayload {
    static constexpr EventType kType = EventType::kKvCacheTransferEnd;
    TransferId transfer_id;
    RequestId request_id;
    BatchId batch_id;
    ReplicaId replica_id;
    DataParallelId dp_id;
    Generation generation;
    ClusterType cluster_type;
};

struct PrefillSyncPayload {
    static constexpr EventType kType = EventType::kPrefillSync;
    BatchId batch_id;
    ReplicaId replica_id;
    DataParallelId dp_id;
    MoEParticipantId participant_id;
    StageId stage_id;
    MoESyncGroupId sync_group_id;
    LayerId layer_id;
    MoESyncPhase sync_phase;
    double elapsed_component_ms;
    bool is_idle;
    Generation generation;
    Generation sync_generation;
    ClusterType cluster_type;
};

struct PrefillSyncCollectivePayload {
    static constexpr EventType kType = EventType::kPrefillSyncCollective;
    ReplicaId replica_id;
    StageId stage_id;
    MoESyncGroupId sync_group_id;
    LayerId layer_id;
    MoESyncPhase sync_phase;
    Generation sync_generation;
    ClusterType cluster_type;
};

struct DecodeSyncPayload {
    static constexpr EventType kType = EventType::kDecodeSync;
    BatchId batch_id;
    ReplicaId replica_id;
    DataParallelId dp_id;
    MoEParticipantId participant_id;
    StageId stage_id;
    MoESyncGroupId sync_group_id;
    LayerId layer_id;
    MoESyncPhase sync_phase;
    double elapsed_component_ms;
    bool is_idle;
    Generation generation;
    Generation sync_generation;
    ClusterType cluster_type;
};

struct DecodeSyncCollectivePayload {
    static constexpr EventType kType = EventType::kDecodeSyncCollective;
    ReplicaId replica_id;
    StageId stage_id;
    MoESyncGroupId sync_group_id;
    LayerId layer_id;
    MoESyncPhase sync_phase;
    Generation sync_generation;
    ClusterType cluster_type;
};

using EventPayload = std::variant<
    RequestArrivalPayload, GlobalSchedulePayload, ClusterSchedulePayload,
    ReplicaSchedulePayload, BatchStageArrivalPayload,
    ReplicaStageSchedulePayload, BatchStageEndPayload, ClusterBatchEndPayload,
    GlobalBatchEndPayload, KVCacheTransferStartPayload,
    KVCacheTransferEndPayload, PrefillSyncPayload, PrefillSyncCollectivePayload,
    DecodeSyncPayload, DecodeSyncCollectivePayload>;

namespace detail {

#define FRONTIER_DEFINE_FIELD_TRAIT(field)                                     \
    template <typename T, typename = void>                                     \
    struct has_##field : std::false_type {};                                   \
    template <typename T>                                                      \
    struct has_##field<T,                                                      \
                       std::void_t<decltype(std::declval<const T &>().field)>> \
        : std::true_type {}

FRONTIER_DEFINE_FIELD_TRAIT(request_id);
FRONTIER_DEFINE_FIELD_TRAIT(batch_id);
FRONTIER_DEFINE_FIELD_TRAIT(replica_id);
FRONTIER_DEFINE_FIELD_TRAIT(dp_id);
FRONTIER_DEFINE_FIELD_TRAIT(stage_id);
FRONTIER_DEFINE_FIELD_TRAIT(generation);
FRONTIER_DEFINE_FIELD_TRAIT(sync_generation);
FRONTIER_DEFINE_FIELD_TRAIT(cluster_type);
FRONTIER_DEFINE_FIELD_TRAIT(transfer_id);
FRONTIER_DEFINE_FIELD_TRAIT(participant_id);
FRONTIER_DEFINE_FIELD_TRAIT(sync_group_id);
FRONTIER_DEFINE_FIELD_TRAIT(layer_id);
FRONTIER_DEFINE_FIELD_TRAIT(sync_phase);
FRONTIER_DEFINE_FIELD_TRAIT(elapsed_component_ms);
FRONTIER_DEFINE_FIELD_TRAIT(is_idle);

#undef FRONTIER_DEFINE_FIELD_TRAIT

} // namespace detail

class Event {
  public:
    Event(SimTime event_time, EventSequence event_sequence,
          EventPayload event_payload)
        : time(event_time), sequence(event_sequence),
          payload(std::move(event_payload)) {}

    [[nodiscard]] EventType type() const noexcept {
        return std::visit(
            [](const auto &value) {
                using Payload =
                    std::remove_cv_t<std::remove_reference_t<decltype(value)>>;
                return Payload::kType;
            },
            payload);
    }

    template <typename Payload> [[nodiscard]] const Payload &as() const {
        return std::get<Payload>(payload);
    }

    template <typename Payload> [[nodiscard]] Payload &as() {
        return std::get<Payload>(payload);
    }

    [[nodiscard]] bool is_stale(Generation current_generation) const noexcept {
        return std::visit(
            [current_generation](const auto &value) {
                using Payload =
                    std::remove_cv_t<std::remove_reference_t<decltype(value)>>;
                if constexpr (detail::has_generation<Payload>::value) {
                    return value.generation != current_generation;
                }
                return false;
            },
            payload);
    }

    SimTime time;
    EventSequence sequence;
    EventPayload payload;
};

} // namespace frontier
