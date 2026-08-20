#pragma once

#include <string_view>
#include <vector>

#include <nlohmann/json_fwd.hpp>

#include "frontier/metrics/output_contract.h"

namespace frontier::metrics::output_detail {

[[nodiscard]] nlohmann::ordered_json
serialize_execution_time_components(const entities::ExecutionTime &execution);
[[nodiscard]] bool is_pdd(config::SystemArchitecture architecture) noexcept;
void require_valid_time(SimTime time, std::string_view field);
[[nodiscard]] double milliseconds_between(SimTime end, SimTime start,
                                          std::string_view field);
void validate_request_metrics(const std::vector<RequestMetricsRecord> &requests,
                              config::SystemArchitecture architecture);

} // namespace frontier::metrics::output_detail
