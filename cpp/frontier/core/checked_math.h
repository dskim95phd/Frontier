#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <string_view>

namespace frontier::checked_math {

template <typename Error>
[[nodiscard]] std::uint64_t add(std::uint64_t left, std::uint64_t right,
                                std::string_view message) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        throw Error(std::string{message});
    }
    return left + right;
}

template <typename Error>
[[nodiscard]] std::uint64_t multiply(std::uint64_t left, std::uint64_t right,
                                     std::string_view message) {
    if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
        throw Error(std::string{message});
    }
    return left * right;
}

template <typename Error>
[[nodiscard]] std::uint64_t ceil_div(std::uint64_t numerator,
                                     std::uint64_t denominator,
                                     std::string_view zero_message) {
    if (denominator == 0) {
        throw Error(std::string{zero_message});
    }
    return numerator / denominator +
           static_cast<std::uint64_t>(numerator % denominator != 0);
}

template <typename Error>
[[nodiscard]] std::uint64_t
prefill_attention_token_pairs(std::uint64_t query_tokens,
                              std::uint64_t past_context,
                              std::string_view overflow_message) {
    const std::uint64_t triangular =
        query_tokens % 2 == 0
            ? multiply<Error>(query_tokens / 2, query_tokens + 1,
                              overflow_message)
            : multiply<Error>(query_tokens, query_tokens / 2 + 1,
                              overflow_message);
    return add<Error>(
        multiply<Error>(query_tokens, past_context, overflow_message),
        triangular, overflow_message);
}

} // namespace frontier::checked_math
