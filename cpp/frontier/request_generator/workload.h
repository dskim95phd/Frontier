#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "frontier/config/config.h"
#include "frontier/core/event.h"
#include "frontier/core/ids.h"

namespace frontier::request_generator {

struct WorkloadRequest {
  RequestId request_id;
  SimTime arrived_at;
  std::uint64_t num_prefill_tokens;
  std::uint64_t num_decode_tokens;
  SessionId session_id;
  std::optional<std::uint64_t> session_turn_index;

  friend bool operator==(
      const WorkloadRequest&,
      const WorkloadRequest&) = default;
};

class WorkloadError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

[[nodiscard]] std::vector<WorkloadRequest> parse_workload_csv(
    std::string_view csv_text);
[[nodiscard]] std::string serialize_workload_csv(
    const std::vector<WorkloadRequest>& requests);
void validate_workload_for_config(
    const std::vector<WorkloadRequest>& requests,
    const config::SimulationConfig& config);

}  // namespace frontier::request_generator
