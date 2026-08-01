#pragma once

#include <string_view>
#include <tuple>

namespace frontier::attention {

enum class AttentionMemoryLayout {
    kDenseKv,
    kLatentMla,
    kFrozenDsa,
};

enum class AttentionVariant {
    kMha,
    kGqa,
    kMqa,
    kMla,
    kDsa,
};

struct AttentionFamilyBinding {
    AttentionMemoryLayout memory_layout = AttentionMemoryLayout::kDenseKv;
    AttentionVariant variant = AttentionVariant::kMha;
    bool execution_enabled = true;

    friend bool operator==(const AttentionFamilyBinding &lhs,
                           const AttentionFamilyBinding &rhs) {
        return std::tie(lhs.memory_layout, lhs.variant,
                        lhs.execution_enabled) ==
               std::tie(rhs.memory_layout, rhs.variant,
                        rhs.execution_enabled);
    }
};

[[nodiscard]] std::string_view
to_string(AttentionMemoryLayout layout) noexcept;
[[nodiscard]] std::string_view to_string(AttentionVariant variant) noexcept;

} // namespace frontier::attention
