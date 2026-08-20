#include "frontier/core/runtime_paths.h"

#include <cstdint>
#include <string_view>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

namespace frontier::core {

std::optional<std::filesystem::path> executable_directory() {
#if defined(_WIN32)
    std::vector<wchar_t> buffer(1024);
    while (true) {
        const DWORD length = GetModuleFileNameW(
            nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
        if (length == 0) {
            return std::nullopt;
        }
        // Windows returns the buffer size when the path was truncated.  Any
        // strictly smaller value is therefore a complete path, including the
        // valid size-1 case.
        if (length < buffer.size()) {
            return std::filesystem::path{
                std::wstring_view{buffer.data(), length}}
                .parent_path();
        }
        if (buffer.size() >= 32768) {
            return std::nullopt;
        }
        buffer.resize(buffer.size() * 2);
    }
#elif defined(__APPLE__)
    std::uint32_t size = 1;
    char probe = '\0';
    if (_NSGetExecutablePath(&probe, &size) != -1 || size <= 1) {
        return std::nullopt;
    }
    std::vector<char> buffer(size);
    if (_NSGetExecutablePath(buffer.data(), &size) != 0) {
        return std::nullopt;
    }
    std::error_code error;
    const std::filesystem::path resolved =
        std::filesystem::weakly_canonical(buffer.data(), error);
    return (error ? std::filesystem::path{buffer.data()} : resolved)
        .parent_path();
#elif defined(__linux__)
    std::vector<char> buffer(1024);
    while (true) {
        const auto length =
            readlink("/proc/self/exe", buffer.data(), buffer.size());
        if (length < 0) {
            return std::nullopt;
        }
        if (static_cast<std::size_t>(length) < buffer.size()) {
            return std::filesystem::path{
                std::string_view{buffer.data(),
                                 static_cast<std::size_t>(length)}}
                .parent_path();
        }
        buffer.resize(buffer.size() * 2);
    }
#else
    return std::nullopt;
#endif
}

} // namespace frontier::core
