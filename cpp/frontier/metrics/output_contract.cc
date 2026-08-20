#include "frontier/metrics/output_contract.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include "frontier/metrics/output_contract_internal.h"

namespace frontier::metrics {
namespace {

using OrderedJson = nlohmann::ordered_json;

using output_detail::milliseconds_between;
using output_detail::serialize_execution_time_components;
using output_detail::validate_request_metrics;

} // namespace

std::string_view to_string(EventType event_type) noexcept {
    switch (event_type) {
    case EventType::kRequestArrival:
        return "request_arrival";
    case EventType::kGlobalSchedule:
        return "global_schedule";
    case EventType::kClusterSchedule:
        return "cluster_schedule";
    case EventType::kReplicaSchedule:
        return "replica_schedule";
    case EventType::kBatchStageArrival:
        return "batch_stage_arrival";
    case EventType::kReplicaStageSchedule:
        return "replica_stage_schedule";
    case EventType::kBatchStageEnd:
        return "batch_stage_end";
    case EventType::kBatchPipelineEnd:
        return "batch_pipeline_end";
    case EventType::kClusterBatchEnd:
        return "cluster_batch_end";
    case EventType::kGlobalBatchEnd:
        return "global_batch_end";
    case EventType::kKvCacheTransferStart:
        return "kv_cache_transfer_start";
    case EventType::kKvCacheTransferEnd:
        return "kv_cache_transfer_end";
    case EventType::kPrefillSync:
        return "prefill_sync";
    case EventType::kPrefillSyncCollective:
        return "prefill_sync_collective";
    case EventType::kDecodeSync:
        return "decode_sync";
    case EventType::kDecodeSyncCollective:
        return "decode_sync_collective";
    case EventType::kCpuKvCacheOffloadStart:
        return "cpu_kv_cache_offload_start";
    case EventType::kCpuKvCacheOffloadEnd:
        return "cpu_kv_cache_offload_end";
    case EventType::kCpuKvCacheRestoreStart:
        return "cpu_kv_cache_restore_start";
    case EventType::kCpuKvCacheRestoreEnd:
        return "cpu_kv_cache_restore_end";
    }
    return "unknown";
}

std::string serialize_simulation_summary_json(const SimulationOutput &output,
                                              double wall_clock_seconds) {
    if (!std::isfinite(wall_clock_seconds) || wall_clock_seconds < 0.0) {
        throw std::invalid_argument(
            "summary wall_clock_seconds must be finite and nonnegative");
    }
    validate_request_metrics(output.requests, output.run.system_architecture);

    const auto summarize_values = [](std::vector<double> values) {
        OrderedJson summary = OrderedJson::object();
        summary["count"] = values.size();
        if (values.empty()) {
            summary["mean"] = 0.0;
            summary["p50"] = 0.0;
            summary["p90"] = 0.0;
            summary["p99"] = 0.0;
            return summary;
        }
        std::sort(values.begin(), values.end());
        const double total = std::accumulate(values.begin(), values.end(), 0.0);
        const auto percentile = [&values](double quantile) {
            const double position =
                quantile * static_cast<double>(values.size() - 1);
            const std::size_t lower = static_cast<std::size_t>(position);
            const std::size_t upper = std::min(lower + 1, values.size() - 1);
            const double fraction = position - static_cast<double>(lower);
            return values[lower] + (values[upper] - values[lower]) * fraction;
        };
        summary["mean"] = total / static_cast<double>(values.size());
        summary["p50"] = percentile(0.50);
        summary["p90"] = percentile(0.90);
        summary["p99"] = percentile(0.99);
        return summary;
    };

    std::vector<double> scheduling_delay_ms;
    std::vector<double> prefill_latency_ms;
    std::vector<double> ttft_ms;
    std::vector<double> tpot_ms;
    std::vector<double> e2e_ms;
    scheduling_delay_ms.reserve(output.requests.size());
    prefill_latency_ms.reserve(output.requests.size());
    ttft_ms.reserve(output.requests.size());
    tpot_ms.reserve(output.requests.size());
    e2e_ms.reserve(output.requests.size());

    double first_arrival_s = std::numeric_limits<double>::infinity();
    double last_completion_s = 0.0;
    std::uint64_t total_prefill_tokens = 0;
    std::uint64_t total_decode_tokens = 0;
    std::uint64_t total_scheduled_prefill_tokens = 0;
    std::uint64_t total_recomputed_prefill_tokens = 0;
    std::uint64_t total_preemptions = 0;
    for (const RequestMetricsRecord &request : output.requests) {
        first_arrival_s =
            std::min(first_arrival_s, request.arrived_at.seconds());
        last_completion_s =
            std::max(last_completion_s, request.completed_at.seconds());
        total_prefill_tokens += request.num_prefill_tokens;
        total_decode_tokens += request.num_decode_tokens;
        total_scheduled_prefill_tokens += request.scheduled_prefill_tokens;
        total_recomputed_prefill_tokens +=
            request.preemption_recomputed_prefill_tokens;
        total_preemptions += request.preemption_count;
        scheduling_delay_ms.push_back(
            milliseconds_between(request.first_scheduled_at, request.arrived_at,
                                 "first_scheduled_at"));
        prefill_latency_ms.push_back(
            milliseconds_between(request.prefill_completed_at,
                                 request.arrived_at, "prefill_completed_at"));
        ttft_ms.push_back(milliseconds_between(request.first_token_completed_at,
                                               request.arrived_at,
                                               "first_token_completed_at"));
        e2e_ms.push_back(milliseconds_between(
            request.completed_at, request.arrived_at, "completed_at"));
        if (request.num_decode_tokens > 1) {
            tpot_ms.push_back(
                milliseconds_between(request.completed_at,
                                     request.first_token_completed_at,
                                     "completed_at") /
                static_cast<double>(request.num_decode_tokens - 1));
        }
    }

    const double simulation_window_s =
        output.observation_window_seconds.value_or(
            output.requests.empty() ? 0.0
                                    : last_completion_s - first_arrival_s);
    const auto rate = [simulation_window_s](std::uint64_t count) {
        return simulation_window_s > 0.0
                   ? static_cast<double>(count) / simulation_window_s
                   : 0.0;
    };

    std::uint64_t aggregate_prefill_scheduled_tokens = 0;
    for (const auto &[cluster_type, aggregate] :
         output.aggregate.batches_by_cluster) {
        static_cast<void>(cluster_type);
        aggregate_prefill_scheduled_tokens +=
            aggregate.prefill_scheduled_tokens;
    }
    // A caller may construct a full SimulationOutput directly (as contract
    // tests and a few analysis tools do) without first feeding batches through
    // MetricsStore.  Preserve a useful summary in that case by deriving the
    // same total from the compact per-batch PREFILL counts.
    if (output.aggregate.batches_by_cluster.empty()) {
        for (const BatchMetricsRecord &batch : output.batches) {
            aggregate_prefill_scheduled_tokens += batch.num_prefill_tokens;
        }
    }

    OrderedJson root = OrderedJson::object();
    root["schema_version"] = 1;
    root["run"] = OrderedJson::object({
        {"run_id", output.run.run_id},
        {"simulation_mode",
         std::string{config::to_string(output.run.simulation_mode)}},
        {"system_architecture",
         std::string{config::to_string(output.run.system_architecture)}},
    });
    root["wall_clock_seconds"] = wall_clock_seconds;
    root["simulation_window_seconds"] = simulation_window_s;
    root["counts"] = OrderedJson::object({
        {"requests", output.requests.size()},
        {"batches", output.aggregate.batch_count},
        {"batch_stages", output.aggregate.batch_stage_count},
        {"scheduler_iterations", output.aggregate.scheduler_iteration_count},
        {"events", output.aggregate.event_count},
        {"kv_cache_transfers", output.aggregate.kv_cache_transfer_count},
        {"preemptions", total_preemptions},
        {"prefill_scheduled_tokens", aggregate_prefill_scheduled_tokens},
    });
    root["prefill_work"] = OrderedJson::object({
        {"scheduled_prefill_tokens", total_scheduled_prefill_tokens},
        {"preemption_recomputed_prefill_tokens",
         total_recomputed_prefill_tokens},
    });
    root["throughput"] = OrderedJson::object({
        {"requests_per_second", rate(output.requests.size())},
        {"prompt_tokens_per_second", rate(total_prefill_tokens)},
        {"decode_tokens_per_second", rate(total_decode_tokens)},
        {"total_tokens_per_second",
         rate(total_prefill_tokens + total_decode_tokens)},
    });
    root["latency_ms"] = OrderedJson::object({
        {"scheduling_delay", summarize_values(std::move(scheduling_delay_ms))},
        {"prefill", summarize_values(std::move(prefill_latency_ms))},
        {"ttft", summarize_values(std::move(ttft_ms))},
        {"tpot", summarize_values(std::move(tpot_ms))},
        {"e2e", summarize_values(std::move(e2e_ms))},
    });

    root["batch_summary_by_cluster"] = OrderedJson::object();
    for (const auto &[cluster_type, aggregate] :
         output.aggregate.batches_by_cluster) {
        OrderedJson histogram = OrderedJson::object();
        for (const auto &[batch_size, count] : aggregate.batch_size_histogram) {
            histogram[std::to_string(batch_size)] = count;
        }
        root["batch_summary_by_cluster"][std::string{to_string(cluster_type)}] =
            OrderedJson::object({
                {"batch_count", aggregate.batch_count},
                {"mean_batch_size",
                 aggregate.batch_count == 0
                     ? 0.0
                     : static_cast<double>(aggregate.request_slots) /
                           static_cast<double>(aggregate.batch_count)},
                {"predicted_execution_ms", aggregate.predicted_execution_ms},
                {"execution_time_components_ms",
                 serialize_execution_time_components(aggregate.execution_time)},
                {"prefill_attention_token_pairs",
                 aggregate.prefill_attention_token_pairs},
                {"prefill_scheduled_tokens",
                 aggregate.prefill_scheduled_tokens},
                {"preemption_recomputed_prefill_tokens",
                 aggregate.preemption_recomputed_prefill_tokens},
                {"batch_size_histogram", std::move(histogram)},
            });
    }

    constexpr std::uint64_t kBatchTimeBucketSeconds = 60;
    root["batch_time_bucket_seconds"] = kBatchTimeBucketSeconds;
    root["batch_summary_by_cluster_time_bucket"] = OrderedJson::object();
    for (const auto &[cluster_type, buckets] :
         output.aggregate.batch_time_buckets_by_cluster) {
        OrderedJson values = OrderedJson::array();
        for (const auto &[bucket_index, aggregate] : buckets) {
            values.push_back(OrderedJson::object({
                {"start_time_s", bucket_index * kBatchTimeBucketSeconds},
                {"end_time_s", (bucket_index + 1) * kBatchTimeBucketSeconds},
                {"batch_count", aggregate.batch_count},
                {"mean_batch_size",
                 aggregate.batch_count == 0
                     ? 0.0
                     : static_cast<double>(aggregate.request_slots) /
                           static_cast<double>(aggregate.batch_count)},
                {"predicted_execution_ms", aggregate.predicted_execution_ms},
                {"execution_time_components_ms",
                 serialize_execution_time_components(aggregate.execution_time)},
                {"prefill_attention_token_pairs",
                 aggregate.prefill_attention_token_pairs},
            }));
        }
        root["batch_summary_by_cluster_time_bucket"]
            [std::string{to_string(cluster_type)}] = std::move(values);
    }
    root["prefill_attention_token_pairs_by_arrival_time_bucket"] =
        OrderedJson::object();
    for (const auto &[bucket_index, token_pairs] :
         output.aggregate
             .prefill_attention_token_pairs_by_arrival_time_bucket) {
        root["prefill_attention_token_pairs_by_arrival_time_bucket"]
            [std::to_string(bucket_index)] = token_pairs;
    }

    std::vector<double> transfer_latency_ms;
    std::uint64_t transfer_bytes = 0;
    transfer_latency_ms.reserve(output.kv_cache_transfers.size());
    for (const KVCacheTransferMetricsRecord &transfer :
         output.kv_cache_transfers) {
        transfer_latency_ms.push_back(transfer.predicted_time_ms);
        transfer_bytes += transfer.size_bytes;
    }
    root["kv_cache_transfer"] = OrderedJson::object({
        {"total_bytes", transfer_bytes},
        {"latency_ms", summarize_values(std::move(transfer_latency_ms))},
    });

    const PrefixCacheMetricsAggregate &cache = output.aggregate.prefix_cache;
    root["prefix_cache"] = OrderedJson::object({
        {"query_blocks", cache.query_blocks},
        {"hit_blocks", cache.hit_blocks},
        {"hit_rate", cache.query_blocks == 0
                         ? 0.0
                         : static_cast<double>(cache.hit_blocks) /
                               static_cast<double>(cache.query_blocks)},
        {"evicted_blocks", cache.evicted_blocks},
        {"evicted_sessions", cache.evicted_sessions},
        {"evicted_kda_snapshots", cache.evicted_kda_snapshots},
        {"evicted_kda_snapshot_blocks", cache.evicted_kda_snapshot_blocks},
        {"kda_snapshot_occupied_blocks", cache.kda_snapshot_occupied_blocks},
        {"kda_snapshot_sessions", cache.kda_snapshot_sessions},
    });
    const CpuKVCacheMetricsAggregate &cpu = output.aggregate.cpu_kv_cache;
    root["cpu_kv_cache"] = OrderedJson::object({
        {"target_count", cpu.target_count},
        {"capacity_bytes", cpu.capacity_bytes},
        {"capacity_blocks", cpu.capacity_blocks},
        {"bytes_per_block", cpu.bytes_per_block},
        {"kda_snapshot_bytes_per_session", cpu.kda_snapshot_bytes_per_session},
        {"kda_snapshot_blocks_per_session",
         cpu.kda_snapshot_blocks_per_session},
        {"kda_snapshot_occupied_bytes", cpu.kda_snapshot_occupied_bytes},
        {"kda_snapshot_occupied_blocks", cpu.kda_snapshot_occupied_blocks},
        {"kda_snapshot_reserved_bytes", cpu.kda_snapshot_reserved_bytes},
        {"kda_snapshot_reserved_blocks", cpu.kda_snapshot_reserved_blocks},
        {"kda_snapshot_sessions", cpu.kda_snapshot_sessions},
        {"kda_snapshot_evictable_sessions",
         cpu.kda_snapshot_evictable_sessions},
        {"evicted_kda_snapshots", cpu.evicted_kda_snapshots},
        {"evicted_kda_snapshot_blocks", cpu.evicted_kda_snapshot_blocks},
        {"evicted_kda_snapshot_bytes", cpu.evicted_kda_snapshot_bytes},
        {"offload_operations", cpu.offload_operations},
        {"offload_blocks", cpu.offload_blocks},
        {"offload_bytes", cpu.offload_bytes},
        {"kda_snapshot_offload_operations",
         cpu.kda_snapshot_offload_operations},
        {"kda_snapshot_offload_bytes", cpu.kda_snapshot_offload_bytes},
        {"restore_operations", cpu.restore_operations},
        {"restore_blocks", cpu.restore_blocks},
        {"restore_bytes", cpu.restore_bytes},
        {"kda_snapshot_restore_operations",
         cpu.kda_snapshot_restore_operations},
        {"kda_snapshot_restore_bytes", cpu.kda_snapshot_restore_bytes},
        {"d2h_queue_time_ms", cpu.d2h_queue_time_ms},
        {"d2h_service_time_ms", cpu.d2h_service_time_ms},
        {"h2d_queue_time_ms", cpu.h2d_queue_time_ms},
        {"h2d_service_time_ms", cpu.h2d_service_time_ms},
        {"source_gpu_hold_time_ms", cpu.source_gpu_hold_time_ms},
        {"query_blocks", cpu.query_blocks},
        {"hit_blocks", cpu.hit_blocks},
        {"hit_rate", cpu.query_blocks == 0
                         ? 0.0
                         : static_cast<double>(cpu.hit_blocks) /
                               static_cast<double>(cpu.query_blocks)},
        {"resident_bytes", cpu.resident_bytes},
        {"resident_blocks", cpu.resident_blocks},
        {"reserved_bytes", cpu.reserved_bytes},
        {"reserved_blocks", cpu.reserved_blocks},
        {"free_bytes", cpu.free_bytes},
        {"free_blocks", cpu.free_blocks},
        {"peak_resident_bytes", cpu.peak_resident_bytes},
        {"peak_resident_blocks", cpu.peak_resident_blocks},
        {"peak_reserved_bytes", cpu.peak_reserved_bytes},
        {"peak_reserved_blocks", cpu.peak_reserved_blocks},
        {"resident_sessions", cpu.resident_sessions},
        {"evicted_blocks", cpu.evicted_blocks},
        {"evicted_sessions", cpu.evicted_sessions},
        {"evicted_bytes", cpu.evicted_bytes},
        {"skipped_offloads", cpu.skipped_offloads},
        {"truncated_offloads", cpu.truncated_offloads},
        {"stale_generation_completions", cpu.stale_generation_completions},
        {"sessions_with_cpu_hits", cpu.sessions_with_cpu_hits},
        {"pending_restore_operations", cpu.pending_restore_operations},
        {"staged_restore_payloads", cpu.staged_restore_payloads},
        {"active_restore_leases", cpu.active_restore_leases},
        {"active_offload_reservations", cpu.active_offload_reservations},
    });
    return root.dump(2) + '\n';
}

} // namespace frontier::metrics
