#pragma once

#include <optional>
#include <string>
#include <string_view>

#include "frontier/config/config.h"

namespace frontier::config::model_detail {

// Built-in aliases and fallback model definitions are intentionally isolated
// from filesystem discovery and JSON asset parsing.
[[nodiscard]] std::string canonical_model_alias(std::string_view model_name);
[[nodiscard]] std::optional<ModelConfig>
registered_model(std::string_view model_name);

} // namespace frontier::config::model_detail
