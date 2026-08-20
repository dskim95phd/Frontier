#include <filesystem>
#include <iostream>

#include "frontier/core/runtime_paths.h"
#include "tests/test_support.h"

int main() {
    try {
        const auto directory = frontier::core::executable_directory();
        frontier::test::expect(directory.has_value(),
                               "executable directory must be discoverable");
        frontier::test::expect(std::filesystem::is_directory(directory.value()),
                               "discovered executable directory must exist");
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "runtime_paths_test: " << error.what() << '\n';
        return 1;
    }
}
