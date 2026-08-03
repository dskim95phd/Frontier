#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include "frontier/config/config.h"
#include "frontier/core/event.h"
#include "frontier/core/ids.h"

namespace frontier::request_generator {

struct WorkloadRequest {
    RequestId request_id;
    // Only the first turn of a session has an external start time. Later
    // turns become ready after their predecessor completes and wait for
    // think_time before arriving.
    SimTime session_start_at;
    SimTime think_time = SimTime::from_seconds(0.0);
    std::uint64_t num_prefill_tokens;
    std::uint64_t num_decode_tokens;
    SessionId session_id;
    std::optional<std::uint64_t> session_turn_index;

    friend bool operator==(const WorkloadRequest &lhs,
                           const WorkloadRequest &rhs) {
        return std::tie(lhs.request_id, lhs.session_start_at, lhs.think_time,
                        lhs.num_prefill_tokens, lhs.num_decode_tokens,
                        lhs.session_id, lhs.session_turn_index) ==
               std::tie(rhs.request_id, rhs.session_start_at, rhs.think_time,
                        rhs.num_prefill_tokens, rhs.num_decode_tokens,
                        rhs.session_id, rhs.session_turn_index);
    }
};

class WorkloadError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

[[nodiscard]] std::vector<WorkloadRequest>
parse_workload_csv(std::string_view csv_text);
[[nodiscard]] std::string
serialize_workload_csv(const std::vector<WorkloadRequest> &requests);
void validate_workload_for_config(const std::vector<WorkloadRequest> &requests,
                                  const config::SimulationConfig &config);
[[nodiscard]] std::vector<WorkloadRequest>
materialize_workload_for_config(const std::vector<WorkloadRequest> &raw,
                                const config::SimulationConfig &config);

} // namespace frontier::request_generator
