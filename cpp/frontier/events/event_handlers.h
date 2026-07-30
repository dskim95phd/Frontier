#pragma once

#include "frontier/core/event.h"

namespace frontier::simulator {
class Simulator;
}

namespace frontier::events {

void handle_event(const RequestArrivalPayload &payload, SimTime time,
                  simulator::Simulator &simulator);
void handle_event(const GlobalSchedulePayload &payload, SimTime time,
                  simulator::Simulator &simulator);
void handle_event(const ClusterSchedulePayload &payload, SimTime time,
                  simulator::Simulator &simulator);
void handle_event(const ReplicaSchedulePayload &payload, SimTime time,
                  simulator::Simulator &simulator);
void handle_event(const BatchStageArrivalPayload &payload, SimTime time,
                  simulator::Simulator &simulator);
void handle_event(const ReplicaStageSchedulePayload &payload, SimTime time,
                  simulator::Simulator &simulator);
void handle_event(const BatchStageEndPayload &payload, SimTime time,
                  simulator::Simulator &simulator);
void handle_event(const ClusterBatchEndPayload &payload, SimTime time,
                  simulator::Simulator &simulator);
void handle_event(const GlobalBatchEndPayload &payload, SimTime time,
                  simulator::Simulator &simulator);
void handle_event(const KVCacheTransferStartPayload &payload, SimTime time,
                  simulator::Simulator &simulator);
void handle_event(const KVCacheTransferEndPayload &payload, SimTime time,
                  simulator::Simulator &simulator);
void handle_event(const PrefillSyncPayload &payload, SimTime time,
                  simulator::Simulator &simulator);
void handle_event(const PrefillSyncCollectivePayload &payload, SimTime time,
                  simulator::Simulator &simulator);
void handle_event(const DecodeSyncPayload &payload, SimTime time,
                  simulator::Simulator &simulator);
void handle_event(const DecodeSyncCollectivePayload &payload, SimTime time,
                  simulator::Simulator &simulator);

} // namespace frontier::events
