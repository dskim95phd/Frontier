#pragma once

#include "frontier/attention/ops.h"

namespace frontier::config {
struct ModelConfig;
}

namespace frontier::attention {

[[nodiscard]] AttentionFamilyBinding
bind_attention_family(const config::ModelConfig &model);

} // namespace frontier::attention
