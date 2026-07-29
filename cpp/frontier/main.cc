#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

#include "frontier/config/config.h"
#include "frontier/metrics/output_contract.h"
#include "frontier/request_generator/workload.h"
#include "frontier/simulator/simulator.h"

#ifndef FRONTIER_VERSION
#define FRONTIER_VERSION "unknown"
#endif

namespace {

constexpr std::string_view kProgramName = "frontier_sim";

void print_usage(std::ostream& stream) {
  stream
      << "Usage:\n"
      << "  " << kProgramName << " --version\n"
      << "  " << kProgramName << " --normalize-config <config.json>\n"
      << "  " << kProgramName
      << " --normalize-workload <workload.csv>\n"
      << "  " << kProgramName
      << " --config <config.json> --workload <workload.csv>\n";
}

std::string read_text_file(const std::filesystem::path& path) {
  std::ifstream input{path, std::ios::binary};
  if (!input) {
    throw std::runtime_error(
        "failed to open input file: " + path.string());
  }
  std::ostringstream contents;
  contents << input.rdbuf();
  if (!input.good() && !input.eof()) {
    throw std::runtime_error(
        "failed to read input file: " + path.string());
  }
  return contents.str();
}

struct InputPaths {
  std::filesystem::path config;
  std::filesystem::path workload;
};

std::optional<InputPaths> parse_input_paths(int argc, char* argv[]) {
  if (argc != 5) {
    return std::nullopt;
  }

  std::optional<std::filesystem::path> config_path;
  std::optional<std::filesystem::path> workload_path;
  for (int index = 1; index < argc; index += 2) {
    const std::string_view option{argv[index]};
    const std::filesystem::path value{argv[index + 1]};
    if (option == "--config" && !config_path.has_value()) {
      config_path = value;
    } else if (option == "--workload" &&
               !workload_path.has_value()) {
      workload_path = value;
    } else {
      return std::nullopt;
    }
  }

  if (!config_path.has_value() || !workload_path.has_value()) {
    return std::nullopt;
  }
  return InputPaths{
      .config = std::move(config_path.value()),
      .workload = std::move(workload_path.value()),
  };
}

}  // namespace

int main(int argc, char* argv[]) {
  if (argc == 2 && std::string_view{argv[1]} == "--version") {
    std::cout << kProgramName << ' ' << FRONTIER_VERSION << '\n';
    return 0;
  }
  if (argc == 2 && std::string_view{argv[1]} == "--help") {
    print_usage(std::cout);
    return 0;
  }

  try {
    if (argc == 3 &&
        std::string_view{argv[1]} == "--normalize-config") {
      const frontier::config::SimulationConfig config =
          frontier::config::parse_simulation_config_json(
              read_text_file(argv[2]));
      std::cout
          << frontier::config::serialize_simulation_config_json(config);
      return 0;
    }
    if (argc == 3 &&
        std::string_view{argv[1]} == "--normalize-workload") {
      const auto workload =
          frontier::request_generator::parse_workload_csv(
              read_text_file(argv[2]));
      std::cout
          << frontier::request_generator::serialize_workload_csv(
                 workload);
      return 0;
    }

    const std::optional<InputPaths> input_paths =
        parse_input_paths(argc, argv);
    if (!input_paths.has_value()) {
      print_usage(std::cerr);
      return 2;
    }

    const frontier::config::SimulationConfig config =
        frontier::config::parse_simulation_config_json(
            read_text_file(input_paths->config));
    const auto workload =
        frontier::request_generator::parse_workload_csv(
            read_text_file(input_paths->workload));
    const frontier::metrics::SimulationOutput output =
        frontier::simulator::run_foundation_lifecycle(
            config,
            workload);
    std::cout
        << frontier::metrics::serialize_simulation_output_json(output);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << kProgramName << ": error: " << error.what() << '\n';
    return 1;
  }
}
