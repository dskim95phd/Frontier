#pragma once

#include <cstdint>

namespace frontier::moe {

enum class SyncPhase : std::uint8_t {
  kPreMoe,
  kPostMoe,
};

enum class SyncPath : std::uint8_t {
  kPrefill,
  kDecode,
};

}  // namespace frontier::moe
