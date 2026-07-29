#pragma once

#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace frontier::test {

inline std::string read_text_file(const std::filesystem::path& path) {
  std::ifstream input{path, std::ios::binary};
  if (!input) {
    throw std::runtime_error(
        "failed to open test fixture: " + path.string());
  }
  std::ostringstream contents;
  contents << input.rdbuf();
  if (!input.good() && !input.eof()) {
    throw std::runtime_error(
        "failed to read test fixture: " + path.string());
  }
  return contents.str();
}

inline void expect(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string{message});
  }
}

template <typename Exception, typename Callable>
void expect_throws(Callable&& callable, std::string_view message) {
  try {
    std::invoke(std::forward<Callable>(callable));
  } catch (const Exception&) {
    return;
  }
  throw std::runtime_error(std::string{message});
}

inline int run(
    std::string_view name,
    const std::function<void()>& test_function) {
  try {
    test_function();
    std::cout << "[PASS] " << name << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[FAIL] " << name << ": " << error.what() << '\n';
    return 1;
  }
}

}  // namespace frontier::test
