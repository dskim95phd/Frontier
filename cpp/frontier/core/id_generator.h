#pragma once

#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace frontier {

// Issues nonnegative strong IDs without narrowing through a wider counter or
// incrementing a signed value past its representable range.
template <typename Id> class CheckedIdGenerator {
  public:
    using ValueType = typename Id::ValueType;

    explicit constexpr CheckedIdGenerator(
        ValueType first = 0,
        ValueType last = std::numeric_limits<ValueType>::max()) noexcept
        : first_(first), next_(first), last_(last),
          exhausted_(first < 0 || first > last) {}

    [[nodiscard]] Id next(std::string_view exhausted_message) {
        if (exhausted_) {
            throw std::overflow_error(std::string{exhausted_message});
        }
        const Id result{next_};
        if (next_ == last_) {
            exhausted_ = true;
        } else {
            ++next_;
        }
        return result;
    }

    [[nodiscard]] constexpr bool was_issued(Id id) const noexcept {
        if (!id.valid()) {
            return false;
        }
        return id.value() >= first_ &&
               (exhausted_ ? id.value() <= last_ : id.value() < next_);
    }

  private:
    ValueType first_;
    ValueType next_;
    ValueType last_;
    bool exhausted_ = false;
};

} // namespace frontier
