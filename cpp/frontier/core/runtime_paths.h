#pragma once

#include <filesystem>
#include <optional>

namespace frontier::core {

// Returns the directory containing the running executable without consulting
// the process working directory. Supported release platforms implement this
// with their native executable-path API.
[[nodiscard]] std::optional<std::filesystem::path> executable_directory();

} // namespace frontier::core
