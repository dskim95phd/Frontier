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

void Simulator::set_wall_clock_progress_callback(
    double interval_seconds, WallClockProgressCallback callback) {
    if (!std::isfinite(interval_seconds) || interval_seconds <= 0.0) {
        throw SimulationError(
            "wall-clock progress interval must be finite and positive");
    }
    if (!callback) {
        throw SimulationError("wall-clock progress callback must be set");
    }
    const auto interval =
        std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>{interval_seconds});
    if (interval <= std::chrono::steady_clock::duration::zero()) {
        throw SimulationError(
            "wall-clock progress interval is below clock resolution");
    }
    wall_clock_progress_interval_ = interval;
    wall_clock_progress_callback_ = std::move(callback);
}

void Simulator::start_wall_clock_progress() {
    if (!wall_clock_progress_callback_) {
        return;
    }
    wall_clock_progress_started_at_ = std::chrono::steady_clock::now();
    wall_clock_progress_next_at_ =
        wall_clock_progress_started_at_ + wall_clock_progress_interval_;
}

void Simulator::maybe_report_wall_clock_progress(SimTime simulation_time) {
    if (!wall_clock_progress_callback_) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (now < wall_clock_progress_next_at_) {
        return;
    }
    const double elapsed_seconds =
        std::chrono::duration<double>(now - wall_clock_progress_started_at_)
            .count();
    wall_clock_progress_callback_(simulation_time, elapsed_seconds);
    do {
        wall_clock_progress_next_at_ += wall_clock_progress_interval_;
    } while (wall_clock_progress_next_at_ <= now);
}

metrics::SimulationOutput Simulator::run() {
    if (observation_end_time_.has_value()) {
        throw SimulationError(
            "a bounded simulator must run to its configured observation end "
            "time");
    }
    const events::EventDispatcher dispatcher;
    start_wall_clock_progress();
    while (!event_queue_.empty()) {
        Event event = event_queue_.pop();
        last_event_time_ = event.time;
        metrics_.record_event(event);
        dispatcher.dispatch(event, *this);
        // Scheduler mutations (admission, preemption, completion, and PDD
        // transfer release) are visible only after dispatch.  Sampling here
        // keeps the occupancy stream event-driven and coalesces same-time
        // transitions in MetricsStore.
        record_gpu_kv_occupancy_for_event(event);
        maybe_report_wall_clock_progress(event.time);
        peak_event_queue_size_ =
            std::max(peak_event_queue_size_, event_queue_.size());
    }
    finalize();
    return take_output();
}

metrics::SimulationOutput Simulator::run_until(SimTime end_time) {
    if (!end_time.valid() || end_time.seconds() <= 0.0) {
        throw SimulationError(
            "simulation end time must be finite and positive");
    }
    if (observation_end_time_.has_value() &&
        observation_end_time_.value() != end_time) {
        throw SimulationError(
            "run_until end time differs from the configured observation "
            "end time");
    }
    const events::EventDispatcher dispatcher;
    start_wall_clock_progress();
    while (!event_queue_.empty() && event_queue_.top().time <= end_time) {
        Event event = event_queue_.pop();
        last_event_time_ = event.time;
        metrics_.record_event(event);
        dispatcher.dispatch(event, *this);
        record_gpu_kv_occupancy_for_event(event);
        maybe_report_wall_clock_progress(event.time);
        peak_event_queue_size_ =
            std::max(peak_event_queue_size_, event_queue_.size());
    }
    // A bounded experiment intentionally leaves future session turns and
    // possibly in-flight requests outside the observation horizon.  Export
    // only requests that reached canonical completion, while snapshotting
    // (without quiescence validation) cache state at the boundary. Completed
    // CPU transfers are streamed into MetricsStore by their end events.
    metrics_.set_observation_window_seconds(end_time.seconds());
    record_bounded_run_cache_diagnostics(end_time);
    metrics_.collect_completed_requests(config_, entities_);
    return take_output();
}

void Simulator::record_bounded_run_cache_diagnostics(SimTime observation_time) {
    for (const auto &[cluster_type, cluster_entity] : clusters_) {
        scheduler::BaseClusterScheduler &cluster_scheduler =
            cluster(cluster_type);
        for (const auto &[replica_id, dp_id] : cluster_scheduler.targets()) {
            scheduler::BaseReplicaScheduler &replica_scheduler =
                cluster_scheduler.get_replica_scheduler(replica_id, dp_id);
            const config::ClusterRuntimeConfig &runtime =
                cluster_entity.runtime_config();
            const GpuKvPhysicalBlockLayout bytes =
                gpu_kv_physical_block_layout(runtime, config_);
            metrics_.record_gpu_kv_cache_occupancy(
                observation_time, replica_scheduler, bytes.max_rank_bytes,
                bytes.pipeline_bytes, total_hbm_bytes_per_gpu(runtime), true);
            if (config_.prefix_cache.enabled &&
                cluster_type != ClusterType::kDecode) {
                metrics_.record_prefix_cache_target(
                    replica_scheduler.prefix_cache_stats(),
                    replica_scheduler.prefix_cache_diagnostics(),
                    scheduler::ReplicaTarget{replica_id, dp_id}, cluster_type,
                    runtime.scheduler.block_size,
                    config_.prefix_cache.key_mode);
            }
            const auto *cpu_manager = replica_scheduler.cpu_kv_cache_manager();
            if (cpu_manager == nullptr) {
                continue;
            }
            const auto *cpu_config =
                replica_scheduler.cpu_kv_cache_target_config();
            if (cpu_config == nullptr) {
                throw std::runtime_error(
                    "CPU KV-cache manager has no resolved target config");
            }
            metrics_.record_cpu_kv_cache_target(
                *cpu_config, cpu_manager->stats(), cpu_manager->diagnostics(),
                scheduler::ReplicaTarget{replica_id, dp_id}, cluster_type,
                replica_scheduler.pending_cpu_restore_count(),
                replica_scheduler.staged_cpu_restore_count());
        }
    }
}

} // namespace frontier::simulator
