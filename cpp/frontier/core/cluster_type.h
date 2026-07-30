#pragma once

#include <cstdint>
#include <string_view>

namespace frontier {

enum class ClusterType : std::uint8_t {
    kMonolithic,
    kPrefill,
    kDecode,
};

[[nodiscard]] constexpr std::string_view
to_string(ClusterType cluster_type) noexcept {
    switch (cluster_type) {
    case ClusterType::kMonolithic:
        return "MONOLITHIC";
    case ClusterType::kPrefill:
        return "PREFILL";
    case ClusterType::kDecode:
        return "DECODE";
    }
    return "UNKNOWN";
}

} // namespace frontier
