#pragma once

#include "frontier/events/base_event.h"

namespace frontier::events {

#define FRONTIER_DECLARE_EVENT_HANDLER(class_name, event_type) \
  class class_name final : public BaseEventHandler {           \
   public:                                                      \
    [[nodiscard]] EventType type() const noexcept override {    \
      return EventType::event_type;                             \
    }                                                           \
    void handle(                                                \
        const Event& event,                                     \
        simulator::SimulationContext& context) const override;  \
  }

FRONTIER_DECLARE_EVENT_HANDLER(
    RequestArrivalEvent, kRequestArrival);
FRONTIER_DECLARE_EVENT_HANDLER(
    GlobalScheduleEvent, kGlobalSchedule);
FRONTIER_DECLARE_EVENT_HANDLER(
    ClusterScheduleEvent, kClusterSchedule);
FRONTIER_DECLARE_EVENT_HANDLER(
    ReplicaScheduleEvent, kReplicaSchedule);
FRONTIER_DECLARE_EVENT_HANDLER(
    BatchStageArrivalEvent, kBatchStageArrival);
FRONTIER_DECLARE_EVENT_HANDLER(
    ReplicaStageScheduleEvent, kReplicaStageSchedule);
FRONTIER_DECLARE_EVENT_HANDLER(
    BatchStageEndEvent, kBatchStageEnd);
FRONTIER_DECLARE_EVENT_HANDLER(
    ClusterBatchEndEvent, kClusterBatchEnd);
FRONTIER_DECLARE_EVENT_HANDLER(
    GlobalBatchEndEvent, kGlobalBatchEnd);
FRONTIER_DECLARE_EVENT_HANDLER(
    KVCacheTransferStartEvent, kKvCacheTransferStart);
FRONTIER_DECLARE_EVENT_HANDLER(
    KVCacheTransferEndEvent, kKvCacheTransferEnd);

#undef FRONTIER_DECLARE_EVENT_HANDLER

class EventDispatcher {
 public:
  void dispatch(
      const Event& event,
      simulator::SimulationContext& context) const;

 private:
  RequestArrivalEvent request_arrival_;
  GlobalScheduleEvent global_schedule_;
  ClusterScheduleEvent cluster_schedule_;
  ReplicaScheduleEvent replica_schedule_;
  BatchStageArrivalEvent batch_stage_arrival_;
  ReplicaStageScheduleEvent replica_stage_schedule_;
  BatchStageEndEvent batch_stage_end_;
  ClusterBatchEndEvent cluster_batch_end_;
  GlobalBatchEndEvent global_batch_end_;
  KVCacheTransferStartEvent kv_cache_transfer_start_;
  KVCacheTransferEndEvent kv_cache_transfer_end_;
};

}  // namespace frontier::events
