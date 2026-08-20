#include "frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>
#include <utility>

namespace frontier::scheduler {

void VllmV1Scheduler::suspend_for_cpu_restore(
    RequestId request_id, const entities::TieredPrefixPlan &plan,
    SimTime time) {
    if (cpu_kv_cache_ == nullptr || cpu_transfer_engine_ == nullptr ||
        (plan.cpu_begin_block >= plan.cpu_end_block &&
         !plan.restore_kda_snapshot) ||
        pending_cpu_restores_.find(request_id) != pending_cpu_restores_.end() ||
        staged_cpu_restores_.find(request_id) != staged_cpu_restores_.end()) {
        throw SchedulerError("invalid CPU restore suspension");
    }
    entities::Request &value = request(request_id);
    const CpuRestoreLeaseId lease = cpu_kv_cache_->pin_restore(
        value.session_id(), plan.cpu_begin_block, plan.cpu_end_block, time);
    const std::uint64_t blocks = plan.cpu_end_block - plan.cpu_begin_block;
    if (blocks > std::numeric_limits<std::uint64_t>::max() /
                     cpu_kv_cache_config_.bytes_per_block) {
        static_cast<void>(cpu_kv_cache_->release_restore(lease, false, time));
        throw SchedulerError("CPU restore transfer size overflows uint64");
    }
    std::uint64_t transfer_bytes =
        blocks * cpu_kv_cache_config_.bytes_per_block;
    std::uint64_t snapshot_bytes = 0;
    std::uint64_t snapshot_frontier = 0;
    if (cpu_kv_cache_->restore_includes_kda_snapshot(lease) &&
        plan.restore_kda_snapshot) {
        snapshot_bytes = cpu_kv_cache_->restore_kda_snapshot_bytes(lease);
        snapshot_frontier =
            cpu_kv_cache_->kda_snapshot_frontier_blocks(value.session_id());
        if (snapshot_bytes == 0 || snapshot_frontier < plan.cpu_end_block ||
            transfer_bytes >
                std::numeric_limits<std::uint64_t>::max() - snapshot_bytes) {
            static_cast<void>(
                cpu_kv_cache_->release_restore(lease, false, time));
            throw SchedulerError("CPU KDA snapshot restore payload is invalid");
        }
        transfer_bytes += snapshot_bytes;
    }
    try {
        const auto timing = cpu_transfer_engine_->schedule(
            cpu_kv_cache_transfer::CpuTransferDirection::kH2D, transfer_bytes,
            time);
        const CpuKvTransferId transfer_id =
            cpu_transfer_ids_.next("CPU KV transfer ID space exhausted");
        const Generation generation{value.runtime_epoch()};
        cpu_restore_operations_.emplace(
            transfer_id,
            entities::CpuKVCacheRestoreInfo{
                transfer_id, request_id, replica_id(), dp_id(), lease, plan,
                timing, generation, snapshot_bytes, snapshot_frontier});
        pending_cpu_restores_.emplace(request_id, transfer_id);
        pending_auxiliary_events_.push_back(
            ScheduledAuxiliaryEvent{timing.started_at, [&]() {
                                        CpuKVCacheRestoreStartPayload payload{};
                                        payload.transfer_id = transfer_id;
                                        payload.request_id = request_id;
                                        payload.replica_id = replica_id();
                                        payload.dp_id = dp_id();
                                        payload.generation = generation;
                                        payload.cluster_type = cluster_type();
                                        return EventPayload{payload};
                                    }()});
    } catch (...) {
        static_cast<void>(cpu_kv_cache_->release_restore(lease, false, time));
        throw;
    }
}

std::uint64_t VllmV1Scheduler::revalidate_staged_frontier(
    const entities::Request &value,
    const entities::StagedCpuKVCacheRestore &staged) const {
    return revalidate_contiguous_tiered_prefix_frontier(
        kv_blocks_.gpu_cache_valid_prefix_blocks(value.session_id()), staged);
}

std::vector<ScheduledAuxiliaryEvent> VllmV1Scheduler::drain_auxiliary_events() {
    return std::exchange(pending_auxiliary_events_, {});
}

std::vector<entities::CpuKVCacheOffloadInfo>
VllmV1Scheduler::cpu_kv_cache_offload_operations() const {
    std::vector<entities::CpuKVCacheOffloadInfo> result;
    result.reserve(cpu_offload_operations_.size());
    for (const auto &[id, operation] : cpu_offload_operations_) {
        static_cast<void>(id);
        result.push_back(operation);
    }
    return result;
}

std::vector<entities::CpuKVCacheRestoreInfo>
VllmV1Scheduler::cpu_kv_cache_restore_operations() const {
    std::vector<entities::CpuKVCacheRestoreInfo> result;
    result.reserve(cpu_restore_operations_.size());
    for (const auto &[id, operation] : cpu_restore_operations_) {
        static_cast<void>(id);
        result.push_back(operation);
    }
    return result;
}

std::optional<entities::CpuKVCacheOffloadInfo>
VllmV1Scheduler::take_completed_cpu_kv_cache_offload(
    CpuKvTransferId transfer_id) {
    const auto operation = cpu_offload_operations_.find(transfer_id);
    if (operation == cpu_offload_operations_.end() ||
        operation->second.state() !=
            entities::CpuKVCacheTransferState::kCompleted) {
        return std::nullopt;
    }
    entities::CpuKVCacheOffloadInfo completed = std::move(operation->second);
    cpu_offload_operations_.erase(operation);
    return completed;
}

std::optional<entities::CpuKVCacheRestoreInfo>
VllmV1Scheduler::take_completed_cpu_kv_cache_restore(
    CpuKvTransferId transfer_id) {
    const auto operation = cpu_restore_operations_.find(transfer_id);
    if (operation == cpu_restore_operations_.end() ||
        operation->second.state() !=
            entities::CpuKVCacheTransferState::kCompleted) {
        return std::nullopt;
    }
    entities::CpuKVCacheRestoreInfo completed = std::move(operation->second);
    cpu_restore_operations_.erase(operation);
    return completed;
}

void VllmV1Scheduler::on_cpu_kv_cache_restore_start(CpuKvTransferId transfer_id,
                                                    Generation generation,
                                                    SimTime time) {
    auto operation = cpu_restore_operations_.find(transfer_id);
    if (operation == cpu_restore_operations_.end() ||
        operation->second.request_generation() != generation ||
        operation->second.state() !=
            entities::CpuKVCacheTransferState::kPending) {
        return;
    }
    operation->second.mark_started(time);
    const entities::CpuKVCacheRestoreInfo &restore = operation->second;
    pending_auxiliary_events_.push_back(
        ScheduledAuxiliaryEvent{restore.timing().completed_at, [&]() {
                                    CpuKVCacheRestoreEndPayload payload{};
                                    payload.transfer_id = transfer_id;
                                    payload.request_id = restore.request_id();
                                    payload.replica_id = replica_id();
                                    payload.dp_id = dp_id();
                                    payload.generation = generation;
                                    payload.cluster_type = cluster_type();
                                    return EventPayload{payload};
                                }()});
}

bool VllmV1Scheduler::on_cpu_kv_cache_restore_end(CpuKvTransferId transfer_id,
                                                  Generation generation,
                                                  SimTime time) {
    auto operation = cpu_restore_operations_.find(transfer_id);
    if (operation == cpu_restore_operations_.end() ||
        operation->second.request_generation() != generation ||
        operation->second.state() !=
            entities::CpuKVCacheTransferState::kInFlight) {
        return false;
    }
    entities::CpuKVCacheRestoreInfo &restore = operation->second;
    const auto pending = pending_cpu_restores_.find(restore.request_id());
    if (pending == pending_cpu_restores_.end() ||
        pending->second != transfer_id) {
        return false;
    }
    entities::Request &value = request(restore.request_id());
    const entities::TieredPrefixPlan &plan = restore.plan();
    bool staged_published = false;
    bool runnable_published = false;
    std::optional<kv_cache::KdaSnapshotCheckpoint> kda_checkpoint;
    try {
        if (restore.includes_kda_snapshot()) {
            kda_checkpoint =
                kv_blocks_.checkpoint_kda_snapshot(value.session_id());
            const std::uint64_t frontier = std::max(
                kv_blocks_.kda_snapshot_frontier_blocks(value.session_id()),
                restore.kda_snapshot_frontier_blocks());
            if (!kv_blocks_.publish_kda_snapshot(value.session_id(),
                                                 frontier)) {
                throw SchedulerError(
                    "GPU cache cannot admit restored atomic KDA snapshot");
            }
        }
        if (!cpu_kv_cache_->release_restore(restore.lease_id(), true, time)) {
            throw SchedulerError("CPU restore lease was already terminal");
        }
        entities::StagedCpuKVCacheRestore staged{};
        staged.request_id = value.id();
        staged.session_id = value.session_id();
        staged.replica_id = replica_id();
        staged.dp_id = dp_id();
        staged.cpu_begin_block = plan.cpu_begin_block;
        staged.cpu_end_block = plan.cpu_end_block;
        staged.lookup_gpu_frontier_blocks = plan.gpu_hit_frontier_blocks;
        staged.lookup_cpu_frontier_blocks = plan.cpu_end_block;
        staged.query_blocks = plan.query_blocks;
        staged.block_size = plan.block_size;
        staged.prompt_tokens = plan.prompt_tokens;
        staged.timing = restore.timing();
        if (!staged_cpu_restores_.emplace(value.id(), std::move(staged))
                 .second) {
            throw SchedulerError("CPU restore staged payload already exists");
        }
        staged_published = true;
        // H2D is complete and the request owns its full-sequence
        // commitment.  Keep it in a dedicated FIFO so it cannot be blocked
        // behind a fresh/preempted request that fails admission.
        restored_ready_.push_back(value.id());
        runnable_published = true;
        restore.mark_completed(time);
        value.record_cpu_restore_transfer(
            plan.cpu_end_block - plan.cpu_begin_block,
            restore.timing().size_bytes, restore.timing().queue_time_ms,
            restore.timing().service_time_ms);
        pending_cpu_restores_.erase(pending);
    } catch (...) {
        if (runnable_published) {
            const auto runnable = std::find(restored_ready_.begin(),
                                            restored_ready_.end(), value.id());
            if (runnable != restored_ready_.end()) {
                restored_ready_.erase(runnable);
            }
        }
        if (staged_published) {
            staged_cpu_restores_.erase(value.id());
        }
        // GPU publication precedes CPU lease termination. If any later step
        // fails, restore the exact prior snapshot state (including whether a
        // zero frontier was merely reserved or validly published) before the
        // commitment cleanup decides whether to retain it.
        if (kda_checkpoint.has_value()) {
            kv_blocks_.restore_kda_snapshot(value.session_id(),
                                            *kda_checkpoint);
        }
        if (cpu_kv_cache_->lease_active(restore.lease_id())) {
            static_cast<void>(cpu_kv_cache_->release_restore(restore.lease_id(),
                                                             false, time));
        }
        pending_cpu_restores_.erase(value.id());
        if (restore.state() != entities::CpuKVCacheTransferState::kCompleted) {
            restore.cancel();
        }
        // A failed restore must not strand the PREFILL full-sequence
        // commitment (or a GPU prefix pinned for that restore).
        kv_blocks_.release_commitment(value.id());
        validate_policy_state();
        throw;
    }
    validate_policy_state();
    return true;
}

bool VllmV1Scheduler::cancel_cpu_kv_cache_restore(RequestId request_id,
                                                  SimTime time) {
    if (!time.valid()) {
        throw SchedulerError("CPU restore cancellation time is invalid");
    }
    const auto pending = pending_cpu_restores_.find(request_id);
    if (pending != pending_cpu_restores_.end()) {
        auto operation = cpu_restore_operations_.find(pending->second);
        if (operation == cpu_restore_operations_.end()) {
            throw SchedulerError("pending CPU restore operation disappeared");
        }
        entities::CpuKVCacheRestoreInfo &restore = operation->second;
        const CpuKvTransferId transfer_id = restore.transfer_id();
        restore.cancel();
        static_cast<void>(
            cpu_kv_cache_->release_restore(restore.lease_id(), false, time));
        pending_cpu_restores_.erase(pending);
        pending_auxiliary_events_.erase(
            std::remove_if(pending_auxiliary_events_.begin(),
                           pending_auxiliary_events_.end(),
                           [&](const auto &event) {
                               const auto *start =
                                   std::get_if<CpuKVCacheRestoreStartPayload>(
                                       &event.payload);
                               return start != nullptr &&
                                      start->transfer_id == transfer_id;
                           }),
            pending_auxiliary_events_.end());
        // The pending H2D operation owns the PREFILL full-sequence
        // commitment.  Once that operation is cancelled, the request returns
        // to ordinary admission and must not retain either its virtual suffix
        // or a GPU prefix pinned on behalf of the cancelled restore.
        kv_blocks_.release_commitment(request_id);
        waiting_.push_back(request_id);
        validate_policy_state();
        return true;
    }
    const auto staged = staged_cpu_restores_.find(request_id);
    if (staged != staged_cpu_restores_.end()) {
        const auto runnable = std::find(restored_ready_.begin(),
                                        restored_ready_.end(), request_id);
        if (runnable == restored_ready_.end()) {
            throw SchedulerError(
                "staged CPU restore runnable entry disappeared");
        }
        staged_cpu_restores_.erase(staged);
        restored_ready_.erase(runnable);
        kv_blocks_.release_commitment(request_id);
        validate_policy_state();
        return true;
    }
    return false;
}

void VllmV1Scheduler::on_cpu_kv_cache_offload_start(
    CpuKvTransferId transfer_id, CpuOffloadGeneration generation,
    SimTime time) {
    auto operation = cpu_offload_operations_.find(transfer_id);
    if (operation == cpu_offload_operations_.end() ||
        operation->second.generation() != generation ||
        operation->second.state() !=
            entities::CpuKVCacheTransferState::kPending) {
        return;
    }
    operation->second.mark_started(time);
    const entities::CpuKVCacheOffloadInfo &offload = operation->second;
    pending_auxiliary_events_.push_back(
        ScheduledAuxiliaryEvent{offload.timing().completed_at, [&]() {
                                    CpuKVCacheOffloadEndPayload payload{};
                                    payload.transfer_id = transfer_id;
                                    payload.request_id = offload.request_id();
                                    payload.replica_id = replica_id();
                                    payload.dp_id = dp_id();
                                    payload.cpu_generation = generation;
                                    payload.cluster_type = cluster_type();
                                    return EventPayload{payload};
                                }()});
}

bool VllmV1Scheduler::on_cpu_kv_cache_offload_end(
    CpuKvTransferId transfer_id, CpuOffloadGeneration generation,
    SimTime time) {
    auto operation = cpu_offload_operations_.find(transfer_id);
    if (operation == cpu_offload_operations_.end() ||
        operation->second.generation() != generation ||
        operation->second.state() !=
            entities::CpuKVCacheTransferState::kInFlight) {
        return false;
    }
    entities::CpuKVCacheOffloadInfo &offload = operation->second;
    const auto pending = pending_cpu_offloads_.find(offload.request_id());
    auto export_state = pending_exports_.find(offload.request_id());
    if (pending == pending_cpu_offloads_.end() ||
        pending->second != transfer_id ||
        export_state == pending_exports_.end() ||
        !export_state->second.cpu_offload_pending) {
        return false;
    }
    try {
        if (!cpu_kv_cache_->commit_offload(offload.reservation_id(), time)) {
            throw SchedulerError(
                "CPU offload reservation was already terminal");
        }
        offload.mark_completed(time);
    } catch (...) {
        if (offload.state() != entities::CpuKVCacheTransferState::kCompleted) {
            offload.cancel();
        }
        if (cpu_kv_cache_->reservation_pending(offload.reservation_id())) {
            static_cast<void>(
                cpu_kv_cache_->abort_offload(offload.reservation_id()));
        }
        pending_cpu_offloads_.erase(pending);
        export_state->second.cpu_offload_pending = false;
        if (!export_state->second.decode_pending) {
            release_request_kv(offload.request_id());
            pending_exports_.erase(export_state);
        }
        validate_policy_state();
        throw;
    }
    pending_cpu_offloads_.erase(pending);
    export_state->second.cpu_offload_pending = false;
    if (!export_state->second.decode_pending) {
        release_request_kv(offload.request_id());
        pending_exports_.erase(export_state);
    }
    request(offload.request_id())
        .record_cpu_offload_transfer(offload.timing().size_bytes,
                                     offload.timing().queue_time_ms,
                                     offload.timing().service_time_ms);
    validate_policy_state();
    return true;
}

bool VllmV1Scheduler::prepare_cpu_kv_cache_offload(RequestId request_id,
                                                   SimTime time) {
    if (cpu_kv_cache_ == nullptr) {
        return false;
    }
    auto export_state = pending_exports_.find(request_id);
    entities::Request &value = request(request_id);
    if (export_state == pending_exports_.end() ||
        !export_state->second.decode_pending ||
        export_state->second.cpu_offload_pending ||
        value.state() != entities::RequestState::kTransferPending ||
        !value.session_id().valid()) {
        throw SchedulerError("invalid CPU offload preparation state");
    }
    if (retired_session_requests_.find(request_id) !=
        retired_session_requests_.end()) {
        return false;
    }
    const CpuOffloadGeneration generation =
        cpu_offload_generation_ids_.next("CPU offload generation exhausted");
    const std::uint64_t desired =
        value.num_processed_prefill_tokens() / kv_blocks_.block_size();
    const auto reservation = cpu_kv_cache_->reserve_offload(
        value.session_id(), generation, desired, time);
    if (!reservation.requires_transfer()) {
        return false;
    }
    if (reservation.reserved_blocks >
        std::numeric_limits<std::uint64_t>::max() /
            cpu_kv_cache_config_.bytes_per_block) {
        static_cast<void>(
            cpu_kv_cache_->abort_offload(reservation.reservation_id));
        throw SchedulerError("CPU offload transfer size overflows uint64");
    }
    std::uint64_t transfer_bytes =
        reservation.reserved_blocks * cpu_kv_cache_config_.bytes_per_block;
    if (transfer_bytes > std::numeric_limits<std::uint64_t>::max() -
                             reservation.kda_snapshot_bytes) {
        static_cast<void>(
            cpu_kv_cache_->abort_offload(reservation.reservation_id));
        throw SchedulerError(
            "CPU KDA snapshot offload payload overflows uint64");
    }
    transfer_bytes += reservation.kda_snapshot_bytes;
    try {
        const auto timing = cpu_transfer_engine_->schedule(
            cpu_kv_cache_transfer::CpuTransferDirection::kD2H, transfer_bytes,
            time);
        const CpuKvTransferId transfer_id =
            cpu_transfer_ids_.next("CPU KV transfer ID space exhausted");
        cpu_offload_operations_.emplace(
            transfer_id,
            entities::CpuKVCacheOffloadInfo{
                transfer_id, request_id, replica_id(), dp_id(),
                reservation.reservation_id, timing, desired, generation,
                reservation.reserved_blocks, reservation.kda_snapshot_bytes});
        pending_cpu_offloads_.emplace(request_id, transfer_id);
        export_state->second.cpu_offload_pending = true;
        pending_auxiliary_events_.push_back(
            ScheduledAuxiliaryEvent{timing.started_at, [&]() {
                                        CpuKVCacheOffloadStartPayload payload{};
                                        payload.transfer_id = transfer_id;
                                        payload.request_id = request_id;
                                        payload.replica_id = replica_id();
                                        payload.dp_id = dp_id();
                                        payload.cpu_generation = generation;
                                        payload.cluster_type = cluster_type();
                                        return EventPayload{payload};
                                    }()});
    } catch (...) {
        static_cast<void>(
            cpu_kv_cache_->abort_offload(reservation.reservation_id));
        export_state->second.cpu_offload_pending = false;
        pending_cpu_offloads_.erase(request_id);
        throw;
    }
    validate_policy_state();
    return true;
}

} // namespace frontier::scheduler
