#include "frontier/metrics/output_contract_internal.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace frontier::metrics::output_detail {

bool is_pdd(config::SystemArchitecture architecture) noexcept {
    return architecture == config::SystemArchitecture::kPdDisaggregation;
}

void require_valid_time(SimTime time, std::string_view field) {
    if (!std::isfinite(time.seconds()) || time.seconds() < 0.0) {
        throw std::invalid_argument(std::string{field} +
                                    " must be finite and nonnegative");
    }
}

double milliseconds_between(SimTime end, SimTime start,
                            std::string_view field) {
    require_valid_time(start, "arrived_at");
    require_valid_time(end, field);
    if (end.seconds() < start.seconds()) {
        throw std::invalid_argument(std::string{field} +
                                    " must not precede arrived_at");
    }
    return (end.seconds() - start.seconds()) * 1e3;
}

void validate_request_metrics(const std::vector<RequestMetricsRecord> &requests,
                              config::SystemArchitecture architecture) {
    std::unordered_set<std::uint64_t> request_ids;
    request_ids.reserve(requests.size());
    for (const RequestMetricsRecord &request : requests) {
        if (!request_ids.insert(request.request_id.value()).second) {
            throw std::invalid_argument(
                "request metrics contain duplicate request_id=" +
                std::to_string(request.request_id.value()));
        }
        if (request.num_prefill_tokens == 0 || request.num_decode_tokens == 0 ||
            request.preemption_recomputed_prefill_tokens >
                request.scheduled_prefill_tokens ||
            request.prefix_cache_hit_blocks >
                request.prefix_cache_query_blocks ||
            request.cpu_prefix_hit_blocks > request.cpu_prefix_query_blocks ||
            request.cpu_restore_consumed_blocks >
                request.cpu_restore_transferred_blocks ||
            request.cpu_restore_discarded_blocks !=
                request.cpu_restore_transferred_blocks -
                    request.cpu_restore_consumed_blocks ||
            request.cpu_prefix_hit_blocks !=
                request.cpu_restore_consumed_blocks ||
            request.gpu_prefix_hit_blocks + request.cpu_prefix_hit_blocks !=
                request.prefix_cache_hit_blocks ||
            request.cached_prefill_tokens > request.num_prefill_tokens) {
            throw std::invalid_argument(
                "request prefix-cache metrics are invalid");
        }
        static_cast<void>(milliseconds_between(request.prefill_completed_at,
                                               request.arrived_at,
                                               "prefill_completed_at"));
        static_cast<void>(milliseconds_between(
            request.completed_at, request.arrived_at, "completed_at"));
        if (request.completed_at.seconds() <
            request.prefill_completed_at.seconds()) {
            throw std::invalid_argument(
                "completed_at must not precede prefill_completed_at");
        }
        if (!request.first_scheduled_at.valid() ||
            !request.first_token_completed_at.valid()) {
            throw std::invalid_argument(
                "request metrics require canonical scheduling and "
                "first-token timestamps");
        }
        static_cast<void>(milliseconds_between(request.first_scheduled_at,
                                               request.arrived_at,
                                               "first_scheduled_at"));
        static_cast<void>(milliseconds_between(request.first_token_completed_at,
                                               request.arrived_at,
                                               "first_token_completed_at"));
        if (request.first_token_completed_at.seconds() <
            request.prefill_completed_at.seconds()) {
            throw std::invalid_argument(
                "first token completion must not precede prefill completion");
        }
        if (is_pdd(architecture)) {
            if (!request.prefill_replica_id.valid() ||
                !request.prefill_dp_id.valid() ||
                !request.decode_replica_id.valid() ||
                !request.decode_dp_id.valid() || !request.transfer_id.valid() ||
                !request.kv_cache_transfer_start_time.valid() ||
                !request.kv_cache_transfer_end_time.valid() ||
                !request.decode_arrived_at.valid() ||
                request.kv_cache_transfer_size_bytes == 0) {
                throw std::invalid_argument(
                    "PDD request metrics require complete ownership");
            }
            const SimTime transfer_start = request.kv_cache_transfer_start_time;
            const SimTime transfer_end = request.kv_cache_transfer_end_time;
            const SimTime decode_arrival = request.decode_arrived_at;
            require_valid_time(transfer_start, "kv transfer start");
            require_valid_time(transfer_end, "kv transfer end");
            require_valid_time(decode_arrival, "decode arrival");
            if (transfer_start.seconds() <
                    request.prefill_completed_at.seconds() ||
                transfer_end.seconds() < transfer_start.seconds() ||
                decode_arrival != transfer_end) {
                throw std::invalid_argument(
                    "PDD transfer timestamps are out of order");
            }
        }
    }
}

} // namespace frontier::metrics::output_detail
