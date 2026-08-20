#pragma once

#include <cstdint>
#include <optional>
#include <string_view>

namespace frontier {

enum class Precision : std::uint8_t {
    kFp32,
    kFp16,
    kBf16,
    kFp8,
    kMxFp8,
    kInt8,
    kFp4,
    kMxFp4,
    kInt4,
};

[[nodiscard]] constexpr std::optional<Precision>
parse_precision(std::string_view value) noexcept {
    if (value == "fp32") {
        return Precision::kFp32;
    }
    if (value == "fp16") {
        return Precision::kFp16;
    }
    if (value == "bf16") {
        return Precision::kBf16;
    }
    if (value == "fp8") {
        return Precision::kFp8;
    }
    if (value == "mxfp8") {
        return Precision::kMxFp8;
    }
    if (value == "int8") {
        return Precision::kInt8;
    }
    if (value == "fp4") {
        return Precision::kFp4;
    }
    if (value == "mxfp4") {
        return Precision::kMxFp4;
    }
    if (value == "int4") {
        return Precision::kInt4;
    }
    return std::nullopt;
}

[[nodiscard]] constexpr double
storage_bytes_per_element(Precision precision) noexcept {
    switch (precision) {
    case Precision::kFp32:
        return 4.0;
    case Precision::kFp16:
    case Precision::kBf16:
        return 2.0;
    case Precision::kFp8:
    case Precision::kInt8:
        return 1.0;
    case Precision::kMxFp8:
        return 1.0 + 1.0 / 32.0;
    case Precision::kFp4:
    case Precision::kInt4:
        return 0.5;
    case Precision::kMxFp4:
        return 0.5 + 1.0 / 32.0;
    }
    return 0.0;
}

} // namespace frontier
