#include <chrono>
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

enum class OutputMode {
    kSummary,
    kRequests,
    kFull,
};

struct RunOptions {
    std::filesystem::path config;
    std::filesystem::path workload;
    std::optional<std::filesystem::path> output_dir;
    OutputMode output_mode = OutputMode::kSummary;
};

void print_usage(std::ostream &stream) {
    stream
        << "Usage:\n"
        << "  " << kProgramName << " --version\n"
        << "  " << kProgramName << " --normalize-config <config.json>\n"
        << "  " << kProgramName << " --normalize-workload <workload.csv>\n"
        << "  " << kProgramName
        << " --config <config.json> --workload <workload.csv>\n"
        << "  " << kProgramName
        << " --config <config.json> --workload <workload.csv> "
           "--output-dir <directory> "
           "[--output-mode summary|requests|full]\n\n"
        << "Without --output-dir, the complete deterministic JSON trace is "
           "written to stdout.\n"
        << "With --output-dir, normalized inputs and summary.json are always "
           "written.\n"
        << "Mode 'requests' also writes requests.csv; mode 'full' also writes "
           "trace.json.\n";
}

std::string read_text_file(const std::filesystem::path &path) {
    std::ifstream input{path, std::ios::binary};
    if (!input) {
        throw std::runtime_error("failed to open input file: " + path.string());
    }
    std::ostringstream contents;
    contents << input.rdbuf();
    if (!input.good() && !input.eof()) {
        throw std::runtime_error("failed to read input file: " + path.string());
    }
    return contents.str();
}

void write_text_file(const std::filesystem::path &path,
                     std::string_view contents) {
    std::ofstream output{path, std::ios::binary};
    if (!output) {
        throw std::runtime_error("failed to open output file: " +
                                 path.string());
    }
    output.write(contents.data(),
                 static_cast<std::streamsize>(contents.size()));
    if (!output) {
        throw std::runtime_error("failed to write output file: " +
                                 path.string());
    }
}

std::optional<OutputMode> parse_output_mode(std::string_view value) {
    if (value == "summary") {
        return OutputMode::kSummary;
    }
    if (value == "requests") {
        return OutputMode::kRequests;
    }
    if (value == "full") {
        return OutputMode::kFull;
    }
    return std::nullopt;
}

std::optional<RunOptions> parse_run_options(int argc, char *argv[]) {
    if (argc < 5 || argc % 2 == 0) {
        return std::nullopt;
    }

    std::optional<std::filesystem::path> config_path;
    std::optional<std::filesystem::path> workload_path;
    std::optional<std::filesystem::path> output_dir;
    std::optional<OutputMode> output_mode;
    for (int index = 1; index < argc; index += 2) {
        const std::string_view option{argv[index]};
        const std::string_view value{argv[index + 1]};
        if (option == "--config" && !config_path.has_value()) {
            config_path = std::filesystem::path{value};
        } else if (option == "--workload" && !workload_path.has_value()) {
            workload_path = std::filesystem::path{value};
        } else if (option == "--output-dir" && !output_dir.has_value()) {
            output_dir = std::filesystem::path{value};
        } else if (option == "--output-mode" && !output_mode.has_value()) {
            output_mode = parse_output_mode(value);
            if (!output_mode.has_value()) {
                return std::nullopt;
            }
        } else {
            return std::nullopt;
        }
    }

    if (!config_path.has_value() || !workload_path.has_value() ||
        (output_mode.has_value() && !output_dir.has_value())) {
        return std::nullopt;
    }
    RunOptions result{};
    result.config = std::move(config_path.value());
    result.workload = std::move(workload_path.value());
    result.output_dir = std::move(output_dir);
    result.output_mode = output_mode.value_or(OutputMode::kSummary);
    return result;
}

void write_artifacts(
    const RunOptions &options, const frontier::config::SimulationConfig &config,
    const std::vector<frontier::request_generator::WorkloadRequest> &workload,
    const frontier::metrics::SimulationOutput &output,
    double wall_clock_seconds) {
    const std::filesystem::path &directory = options.output_dir.value();
    std::error_code error;
    std::filesystem::create_directories(directory, error);
    if (error) {
        throw std::runtime_error("failed to create output directory: " +
                                 directory.string() + ": " + error.message());
    }
    if (!std::filesystem::is_directory(directory)) {
        throw std::runtime_error("output path is not a directory: " +
                                 directory.string());
    }

    write_text_file(directory / "config.normalized.json",
                    frontier::config::serialize_simulation_config_json(config));
    write_text_file(
        directory / "workload.normalized.csv",
        frontier::request_generator::serialize_workload_csv(workload));
    write_text_file(directory / "summary.json",
                    frontier::metrics::serialize_simulation_summary_json(
                        output, wall_clock_seconds));

    if (options.output_mode == OutputMode::kRequests ||
        options.output_mode == OutputMode::kFull) {
        write_text_file(directory / "requests.csv",
                        frontier::metrics::serialize_request_metrics_csv(
                            output.requests, output.run.system_architecture));
    }
    if (options.output_mode == OutputMode::kFull) {
        write_text_file(
            directory / "trace.json",
            frontier::metrics::serialize_simulation_output_json(output));
    }
}

} // namespace

int main(int argc, char *argv[]) {
    if (argc == 2 && std::string_view{argv[1]} == "--version") {
        std::cout << kProgramName << ' ' << FRONTIER_VERSION << '\n';
        return 0;
    }
    if (argc == 2 && std::string_view{argv[1]} == "--help") {
        print_usage(std::cout);
        return 0;
    }

    try {
        if (argc == 3 && std::string_view{argv[1]} == "--normalize-config") {
            const frontier::config::SimulationConfig config =
                frontier::config::parse_simulation_config_json(
                    read_text_file(argv[2]));
            std::cout << frontier::config::serialize_simulation_config_json(
                config);
            return 0;
        }
        if (argc == 3 && std::string_view{argv[1]} == "--normalize-workload") {
            const auto workload =
                frontier::request_generator::parse_workload_csv(
                    read_text_file(argv[2]));
            std::cout << frontier::request_generator::serialize_workload_csv(
                workload);
            return 0;
        }

        const std::optional<RunOptions> options = parse_run_options(argc, argv);
        if (!options.has_value()) {
            print_usage(std::cerr);
            return 2;
        }

        const frontier::config::SimulationConfig config =
            frontier::config::parse_simulation_config_json(
                read_text_file(options->config));
        const auto workload = frontier::request_generator::parse_workload_csv(
            read_text_file(options->workload));

        const auto started_at = std::chrono::steady_clock::now();
        frontier::simulator::Simulator simulator{config, workload};
        if (options->output_dir.has_value() &&
            options->output_mode != OutputMode::kFull) {
            simulator.metrics().set_detailed_traces_enabled(false);
        }
        const frontier::metrics::SimulationOutput output = simulator.run();
        const double wall_clock_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          started_at)
                .count();

        if (!options->output_dir.has_value()) {
            std::cout << frontier::metrics::serialize_simulation_output_json(
                output);
            return 0;
        }
        write_artifacts(options.value(), config, workload, output,
                        wall_clock_seconds);
        std::cout << "wrote simulation artifacts to "
                  << options->output_dir->string() << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << kProgramName << ": error: " << error.what() << '\n';
        return 1;
    }
}
