#include "frontier/simulator/simulator.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>

#include "frontier/events/event_dispatcher.h"
#include "frontier/simulator/simulator_memory_layout.h"

namespace frontier::simulator {

using simulator_detail::gpu_kv_physical_block_layout;
using simulator_detail::GpuKvPhysicalBlockLayout;
using simulator_detail::total_hbm_bytes_per_gpu;

void Simulator::finalize() {
    if (!global_scheduler_->empty() ||
        entities_.completion_order().size() != entities_.request_count()) {
        std::ostringstream detail;
        detail << " incomplete_requests=[";
        bool first = true;
        for (const entities::Request &request : entities_.requests()) {
            if (request_completion_recorded(request.id())) {
                continue;
            }
            if (!first) {
                detail << ',';
            }
            first = false;
            detail << request.id().value()
                   << ":state=" << static_cast<int>(request.state())
                   << ":processed=" << request.num_processed_tokens()
                   << ":scheduled=" << request.scheduler_num_computed_tokens();
        }
        detail << "] targets=[";
        first = true;
        for (const auto &[cluster_type, cluster_entity] : clusters_) {
            scheduler::BaseClusterScheduler &cluster_scheduler =
                cluster(cluster_type);
            for (const auto &[replica_id, dp_id] :
                 cluster_scheduler.targets()) {
                if (!first) {
                    detail << ',';
                }
                first = false;
                const scheduler::BaseReplicaScheduler &replica_scheduler =
                    cluster_scheduler.get_replica_scheduler(replica_id, dp_id);
                detail << static_cast<int>(cluster_type) << ':'
                       << replica_id.value() << ':' << dp_id.value()
                       << ":waiting=" << replica_scheduler.waiting_count()
                       << ":running=" << replica_scheduler.running_count()
                       << ":inflight="
                       << replica_scheduler.in_flight_batch_count()
                       << ":transfers="
                       << replica_scheduler.pending_kv_transfer_count();
            }
        }
        detail << ']';
        throw std::runtime_error(
            "simulation quiesced with incomplete global or request state "
            "(global_empty=" +
            std::string{global_scheduler_->empty() ? "true" : "false"} +
            ", completed=" +
            std::to_string(entities_.completion_order().size()) + "/" +
            std::to_string(entities_.request_count()) + ")" + detail.str());
    }
    for (const auto &[cluster_type, cluster_entity] : clusters_) {
        static_cast<void>(cluster_entity);
        scheduler::BaseClusterScheduler &cluster_scheduler =
            cluster(cluster_type);
        cluster_scheduler.require_quiescent();
        if (!cluster_scheduler.empty()) {
            throw std::runtime_error(
                "simulation quiesced with nonempty cluster queue");
        }
        for (const auto &[replica_id, dp_id] : cluster_scheduler.targets()) {
            scheduler::BaseReplicaScheduler &replica_scheduler =
                cluster_scheduler.get_replica_scheduler(replica_id, dp_id);
            if (!replica_scheduler.idle()) {
                throw std::runtime_error(
                    "simulation quiesced with non-idle replica target");
            }
            const config::ClusterRuntimeConfig &runtime =
                cluster_entity.runtime_config();
            const GpuKvPhysicalBlockLayout bytes =
                gpu_kv_physical_block_layout(runtime, config_);
            metrics_.record_gpu_kv_cache_occupancy(
                last_event_time_, replica_scheduler, bytes.max_rank_bytes,
                bytes.pipeline_bytes, total_hbm_bytes_per_gpu(runtime), true);
            if (config_.prefix_cache.enabled &&
                cluster_type != ClusterType::kDecode) {
                metrics_.record_prefix_cache_target(
                    replica_scheduler.prefix_cache_stats(),
                    replica_scheduler.prefix_cache_diagnostics(),
                    scheduler::ReplicaTarget{replica_id, dp_id}, cluster_type,
                    cluster_entity.runtime_config().scheduler.block_size,
                    config_.prefix_cache.key_mode);
            }
            if (const auto *cpu_manager =
                    replica_scheduler.cpu_kv_cache_manager()) {
                const auto *cpu_config =
                    replica_scheduler.cpu_kv_cache_target_config();
                if (cpu_config == nullptr) {
                    throw std::runtime_error(
                        "CPU KV-cache manager has no resolved target config");
                }
                const auto diagnostics = cpu_manager->diagnostics();
                const auto offloads =
                    replica_scheduler.cpu_kv_cache_offload_operations();
                const auto restores =
                    replica_scheduler.cpu_kv_cache_restore_operations();
                const auto transfer_is_pending = [](const auto &operation) {
                    const auto state = operation.state();
                    return state ==
                               entities::CpuKVCacheTransferState::kPending ||
                           state ==
                               entities::CpuKVCacheTransferState::kInFlight;
                };
                if (diagnostics.active_reservations != 0 ||
                    diagnostics.active_restore_leases != 0 ||
                    replica_scheduler.pending_cpu_restore_count() != 0 ||
                    replica_scheduler.staged_cpu_restore_count() != 0 ||
                    std::any_of(offloads.begin(), offloads.end(),
                                transfer_is_pending) ||
                    std::any_of(restores.begin(), restores.end(),
                                transfer_is_pending)) {
                    throw std::runtime_error(
                        "simulation quiesced with nonterminal CPU KV-cache "
                        "ownership or transfer state");
                }
                metrics_.record_cpu_kv_cache_target(
                    *cpu_config, cpu_manager->stats(), diagnostics,
                    scheduler::ReplicaTarget{replica_id, dp_id}, cluster_type,
                    replica_scheduler.pending_cpu_restore_count(),
                    replica_scheduler.staged_cpu_restore_count());
            }
            for (std::uint64_t stage = 0;
                 stage < replica_scheduler.pipeline_parallel_size(); ++stage) {
                const scheduler::ReplicaStageScheduler &stage_scheduler =
                    replica_scheduler.get_replica_stage_scheduler(
                        StageId{stage});
                if (stage_scheduler.is_busy() || !stage_scheduler.empty()) {
                    throw std::runtime_error(
                        "simulation quiesced with nonempty stage scheduler");
                }
            }
        }
    }
    if (config_.system_architecture ==
        config::SystemArchitecture::kPdDisaggregation) {
        const auto &transfers = entities_.kv_cache_transfers();
        if (decode_arrivals_ != expected_decode_arrivals_ ||
            transfers.size() != entities_.request_count() ||
            std::any_of(transfers.begin(), transfers.end(),
                        [](const entities::KVCacheTransferInfo &transfer) {
                            return transfer.state() !=
                                   entities::KVCacheTransferState::kCompleted;
                        })) {
            throw std::runtime_error(
                "simulation quiesced with incomplete PDD transfer state");
        }
    }
    if (entities_.live_batch_count() != 0) {
        throw std::runtime_error(
            "simulation quiesced with unreleased batch entities");
    }
    metrics_.collect_completed_requests(config_, entities_);
}

metrics::SimulationOutput Simulator::take_output() {
    return metrics_.take_output();
}

} // namespace frontier::simulator
