#include "frontier/metrics/output_contract.h"

#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <map>
#include <sstream>
#include <stdexcept>
#include <tuple>

#include "frontier/metrics/output_contract_internal.h"

namespace frontier::metrics {

using output_detail::is_pdd;
using output_detail::milliseconds_between;
using output_detail::require_valid_time;
using output_detail::validate_request_metrics;

std::string
serialize_request_metrics_csv(const std::vector<RequestMetricsRecord> &requests,
                              config::SystemArchitecture architecture) {
    std::ostringstream output;
    serialize_request_metrics_csv(requests, output, architecture);
    return output.str();
}

void serialize_request_metrics_csv(
    const std::vector<RequestMetricsRecord> &requests, std::ostream &output,
    config::SystemArchitecture architecture) {
    validate_request_metrics(requests, architecture);

    output.imbue(std::locale::classic());
    output << std::setprecision(std::numeric_limits<double>::max_digits10);
    if (!is_pdd(architecture)) {
        output << "request_id,session_id,num_prefill_tokens,num_decode_tokens,"
                  "cached_prefill_tokens,prefix_cache_query_blocks,"
                  "prefix_cache_hit_blocks,gpu_prefix_hit_blocks,"
                  "cpu_prefix_query_blocks,cpu_prefix_hit_blocks,"
                  "cpu_restore_transferred_blocks,cpu_restore_consumed_blocks,"
                  "cpu_restore_discarded_blocks,cpu_restored_tokens,"
                  "cpu_restore_bytes,cpu_restore_queue_time_ms,"
                  "cpu_restore_service_time_ms,cpu_offload_bytes,"
                  "cpu_offload_queue_time_ms,cpu_offload_service_time_ms,"
                  "prefix_cache_key_mode,arrived_at_s,"
                  "first_scheduled_at_s,"
                  "prefill_completed_at_s,first_token_completed_at_s,"
                  "completed_at_s,scheduling_delay_ms,prefill_latency_ms,"
                  "ttft_ms,e2e_ms,"
                  "num_processed_tokens,preemption_count,replica_id,dp_id,"
                  "scheduled_prefill_tokens,"
                  "preemption_recomputed_prefill_tokens\n";
    } else {
        output << "request_id,session_id,num_prefill_tokens,num_decode_tokens,"
                  "cached_prefill_tokens,prefix_cache_query_blocks,"
                  "prefix_cache_hit_blocks,gpu_prefix_hit_blocks,"
                  "cpu_prefix_query_blocks,cpu_prefix_hit_blocks,"
                  "cpu_restore_transferred_blocks,cpu_restore_consumed_blocks,"
                  "cpu_restore_discarded_blocks,cpu_restored_tokens,"
                  "cpu_restore_bytes,cpu_restore_queue_time_ms,"
                  "cpu_restore_service_time_ms,cpu_offload_bytes,"
                  "cpu_offload_queue_time_ms,cpu_offload_service_time_ms,"
                  "prefix_cache_key_mode,arrived_at_s,"
                  "first_scheduled_at_s,"
                  "prefill_completed_at_s,first_token_completed_at_s,"
                  "completed_at_s,scheduling_delay_ms,prefill_latency_ms,"
                  "ttft_ms,e2e_ms,"
                  "num_processed_tokens,preemption_count,"
                  "prefill_replica_id,prefill_dp_id,decode_replica_id,decode_"
                  "dp_id,"
                  "transfer_id,kv_cache_transfer_start_time_s,"
                  "kv_cache_transfer_end_time_s,kv_cache_transfer_time_ms,"
                  "kv_cache_transfer_size_bytes,decode_arrived_at_s,"
                  "scheduled_prefill_tokens,"
                  "preemption_recomputed_prefill_tokens\n";
    }

    for (const RequestMetricsRecord &request : requests) {
        output << request.request_id.value() << ',';
        if (request.session_id.valid()) {
            output << request.session_id.value();
        }
        output << ',' << request.num_prefill_tokens << ','
               << request.num_decode_tokens << ','
               << request.cached_prefill_tokens << ','
               << request.prefix_cache_query_blocks << ','
               << request.prefix_cache_hit_blocks << ','
               << request.gpu_prefix_hit_blocks << ','
               << request.cpu_prefix_query_blocks << ','
               << request.cpu_prefix_hit_blocks << ','
               << request.cpu_restore_transferred_blocks << ','
               << request.cpu_restore_consumed_blocks << ','
               << request.cpu_restore_discarded_blocks << ','
               << request.cpu_restored_tokens << ','
               << request.cpu_restore_bytes << ','
               << request.cpu_restore_queue_time_s * 1e3 << ','
               << request.cpu_restore_service_time_s * 1e3 << ','
               << request.cpu_offload_bytes << ','
               << request.cpu_offload_queue_time_s * 1e3 << ','
               << request.cpu_offload_service_time_s * 1e3 << ','
               << config::to_string(request.prefix_cache_key_mode) << ','
               << request.arrived_at.seconds() << ',';
        output << request.first_scheduled_at.seconds() << ',';
        output << request.prefill_completed_at.seconds() << ',';
        output << request.first_token_completed_at.seconds() << ',';
        output << request.completed_at.seconds() << ',';
        output << milliseconds_between(request.first_scheduled_at,
                                       request.arrived_at, "first_scheduled_at")
               << ',';
        output << milliseconds_between(request.prefill_completed_at,
                                       request.arrived_at,
                                       "prefill_completed_at")
               << ',';
        output << milliseconds_between(request.first_token_completed_at,
                                       request.arrived_at,
                                       "first_token_completed_at")
               << ','
               << milliseconds_between(request.completed_at, request.arrived_at,
                                       "completed_at");
        output << ',' << request.num_processed_tokens << ','
               << request.preemption_count;
        if (!is_pdd(architecture)) {
            output << ',' << request.replica_id.value() << ','
                   << request.dp_id.value() << ','
                   << request.scheduled_prefill_tokens << ','
                   << request.preemption_recomputed_prefill_tokens;
        } else {
            output << ',' << request.prefill_replica_id.value() << ','
                   << request.prefill_dp_id.value() << ','
                   << request.decode_replica_id.value() << ','
                   << request.decode_dp_id.value() << ','
                   << request.transfer_id.value() << ','
                   << request.kv_cache_transfer_start_time.seconds() << ','
                   << request.kv_cache_transfer_end_time.seconds() << ','
                   << (request.kv_cache_transfer_end_time.seconds() -
                       request.kv_cache_transfer_start_time.seconds()) *
                          1e3
                   << ',' << request.kv_cache_transfer_size_bytes << ','
                   << request.decode_arrived_at.seconds() << ','
                   << request.scheduled_prefill_tokens << ','
                   << request.preemption_recomputed_prefill_tokens;
        }
        output << '\n';
    }
}

std::string serialize_gpu_kv_occupancy_csv(
    const std::vector<GpuKVCacheOccupancyRecord> &occupancy) {
    std::ostringstream output;
    serialize_gpu_kv_occupancy_csv(occupancy, output);
    return output.str();
}

void serialize_gpu_kv_occupancy_csv(
    const std::vector<GpuKVCacheOccupancyRecord> &occupancy,
    std::ostream &output) {
    output.imbue(std::locale::classic());
    output << std::setprecision(std::numeric_limits<double>::max_digits10);
    output << "time_s,cluster_type,replica_id,dp_id,active_blocks,"
              "capacity_blocks,active_bytes_per_gpu,"
              "active_bytes_across_pipeline,hbm_fraction,"
              "active_fraction_of_kv_budget,active_fraction_of_total_hbm\n";

    std::map<std::tuple<ClusterType, ReplicaId, DataParallelId>, SimTime>
        last_time;
    for (const GpuKVCacheOccupancyRecord &record : occupancy) {
        require_valid_time(record.time, "gpu_kv_occupancy.time");
        if (!record.replica_id.valid() || !record.dp_id.valid()) {
            throw std::invalid_argument(
                "gpu KV occupancy target IDs must be valid");
        }
        if (record.active_blocks > record.capacity_blocks ||
            !std::isfinite(record.active_fraction_of_kv_budget) ||
            record.active_fraction_of_kv_budget < 0.0 ||
            record.active_fraction_of_kv_budget > 1.0) {
            throw std::invalid_argument(
                "gpu KV occupancy active fraction is invalid");
        }
        if (record.hbm_fraction.has_value() &&
            (!std::isfinite(record.hbm_fraction.value()) ||
             record.hbm_fraction.value() < 0.0 ||
             record.hbm_fraction.value() > 1.0)) {
            throw std::invalid_argument(
                "gpu KV occupancy total-HBM fraction is invalid");
        }
        if (record.active_fraction_of_total_hbm.has_value() &&
            (!std::isfinite(record.active_fraction_of_total_hbm.value()) ||
             record.active_fraction_of_total_hbm.value() < 0.0 ||
             record.active_fraction_of_total_hbm.value() > 1.0)) {
            throw std::invalid_argument(
                "gpu KV occupancy total-HBM fraction is invalid");
        }
        if (record.hbm_fraction.has_value() &&
            record.active_fraction_of_total_hbm.has_value() &&
            record.hbm_fraction.value() !=
                record.active_fraction_of_total_hbm.value()) {
            throw std::invalid_argument(
                "gpu KV occupancy total-HBM fractions disagree");
        }
        const auto key = std::make_tuple(record.cluster_type, record.replica_id,
                                         record.dp_id);
        const auto previous = last_time.find(key);
        if (previous != last_time.end() && record.time < previous->second) {
            throw std::invalid_argument(
                "gpu KV occupancy samples are not time ordered");
        }
        last_time[key] = record.time;
        output << record.time.seconds() << ',' << to_string(record.cluster_type)
               << ',' << record.replica_id.value() << ','
               << record.dp_id.value() << ',' << record.active_blocks << ','
               << record.capacity_blocks << ',' << record.active_bytes_per_gpu
               << ',' << record.active_bytes_across_pipeline << ',';
        if (record.hbm_fraction.has_value()) {
            output << record.hbm_fraction.value();
        }
        output << ',' << record.active_fraction_of_kv_budget << ',';
        if (record.active_fraction_of_total_hbm.has_value()) {
            output << record.active_fraction_of_total_hbm.value();
        }
        output << '\n';
    }
}

} // namespace frontier::metrics
