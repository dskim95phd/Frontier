#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

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
    bool runtime_validation = true;
    bool gpu_kv_occupancy = true;
    std::optional<double> simulation_end_time_s;
    std::optional<double> wall_progress_interval_s;
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
           "[--output-mode summary|requests|full] "
           "[--runtime-validation true|false] "
           "[--gpu-kv-occupancy true|false] "
           "[--wall-progress-interval-s <seconds>] "
           "[--simulation-end-time-s <seconds>]\n\n"
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
    input.seekg(0, std::ios::end);
    const std::streampos end = input.tellg();
    if (end < 0) {
        throw std::runtime_error("failed to size input file: " + path.string());
    }
    std::string contents(static_cast<std::size_t>(end), '\0');
    input.seekg(0, std::ios::beg);
    input.read(contents.data(), static_cast<std::streamsize>(contents.size()));
    if (input.gcount() != static_cast<std::streamsize>(contents.size())) {
        throw std::runtime_error("failed to read input file: " + path.string());
    }
    return contents;
}

std::vector<frontier::request_generator::WorkloadRequest>
read_workload_file(const std::filesystem::path &path) {
    std::ifstream input{path, std::ios::binary};
    if (!input) {
        throw std::runtime_error("failed to open input file: " + path.string());
    }
    auto workload = frontier::request_generator::parse_workload_csv(input);
    if (!input.good() && !input.eof()) {
        throw std::runtime_error("failed to read input file: " + path.string());
    }
    return workload;
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
    std::optional<bool> runtime_validation;
    std::optional<bool> gpu_kv_occupancy;
    std::optional<double> simulation_end_time_s;
    std::optional<double> wall_progress_interval_s;
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
        } else if (option == "--runtime-validation" &&
                   !runtime_validation.has_value()) {
            if (value == "true") {
                runtime_validation = true;
            } else if (value == "false") {
                runtime_validation = false;
            } else {
                return std::nullopt;
            }
        } else if (option == "--gpu-kv-occupancy" &&
                   !gpu_kv_occupancy.has_value()) {
            if (value == "true") {
                gpu_kv_occupancy = true;
            } else if (value == "false") {
                gpu_kv_occupancy = false;
            } else {
                return std::nullopt;
            }
        } else if (option == "--simulation-end-time-s" &&
                   !simulation_end_time_s.has_value()) {
            try {
                std::size_t consumed = 0;
                const double parsed = std::stod(std::string{value}, &consumed);
                if (consumed != value.size() || !std::isfinite(parsed) ||
                    parsed <= 0.0) {
                    return std::nullopt;
                }
                simulation_end_time_s = parsed;
            } catch (const std::exception &) {
                return std::nullopt;
            }
        } else if (option == "--wall-progress-interval-s" &&
                   !wall_progress_interval_s.has_value()) {
            try {
                std::size_t consumed = 0;
                const double parsed = std::stod(std::string{value}, &consumed);
                if (consumed != value.size() || !std::isfinite(parsed) ||
                    parsed <= 0.0) {
                    return std::nullopt;
                }
                wall_progress_interval_s = parsed;
            } catch (const std::exception &) {
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
    result.runtime_validation = runtime_validation.value_or(true);
    result.gpu_kv_occupancy = gpu_kv_occupancy.value_or(true);
    result.simulation_end_time_s = simulation_end_time_s;
    result.wall_progress_interval_s = wall_progress_interval_s;
    return result;
}

void ensure_output_directory(const std::filesystem::path &directory) {
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
}

template <typename Writer>
void write_stream_file(const std::filesystem::path &path, Writer writer) {
    std::ofstream output{path, std::ios::binary};
    if (!output) {
        throw std::runtime_error("failed to open output file: " +
                                 path.string());
    }
    writer(output);
    if (!output) {
        throw std::runtime_error("failed to write output file: " +
                                 path.string());
    }
}

void write_normalized_inputs(
    const RunOptions &options, const frontier::config::SimulationConfig &config,
    const std::vector<frontier::request_generator::WorkloadRequest> &workload) {
    const std::filesystem::path &directory = options.output_dir.value();
    ensure_output_directory(directory);

    write_text_file(directory / "config.normalized.json",
                    frontier::config::serialize_simulation_config_json(config));
    const std::filesystem::path workload_path =
        directory / "workload.normalized.csv";
    write_stream_file(workload_path, [&](std::ostream &output) {
        frontier::request_generator::serialize_workload_csv(workload, output);
    });
}

void write_artifacts(const RunOptions &options,
                     const frontier::metrics::SimulationOutput &output,
                     double wall_clock_seconds) {
    const std::filesystem::path &directory = options.output_dir.value();
    write_text_file(directory / "summary.json",
                    frontier::metrics::serialize_simulation_summary_json(
                        output, wall_clock_seconds));

    if (options.output_mode == OutputMode::kRequests ||
        options.output_mode == OutputMode::kFull) {
        write_stream_file(
            directory / "requests.csv", [&](std::ostream &stream) {
                frontier::metrics::serialize_request_metrics_csv(
                    output.requests, stream, output.run.system_architecture);
            });
    }
    // Occupancy samples are compact change events and are retained for every
    // output mode, including summary mode where detailed traces are disabled.
    write_stream_file(directory / "gpu_kv_occupancy.csv",
                      [&](std::ostream &stream) {
                          frontier::metrics::serialize_gpu_kv_occupancy_csv(
                              output.gpu_kv_occupancy, stream);
                      });
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
            const auto workload = read_workload_file(argv[2]);
            frontier::request_generator::serialize_workload_csv(workload,
                                                                std::cout);
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
        auto workload = read_workload_file(options->workload);
        if (options->output_dir.has_value()) {
            write_normalized_inputs(options.value(), config, workload);
        }

        const auto started_at = std::chrono::steady_clock::now();
        frontier::simulator::SimulatorOptions simulator_options{};
        simulator_options.detailed_traces_enabled =
            !options->output_dir.has_value() ||
            options->output_mode == OutputMode::kFull;
        if (options->simulation_end_time_s.has_value()) {
            simulator_options.observation_end_time =
                frontier::SimTime::from_seconds(
                    options->simulation_end_time_s.value());
        }
        frontier::simulator::Simulator simulator{config, std::move(workload),
                                                 simulator_options};
        simulator.set_runtime_validation_enabled(options->runtime_validation);
        simulator.metrics().set_gpu_kv_occupancy_enabled(
            options->gpu_kv_occupancy);
        if (options->wall_progress_interval_s.has_value()) {
            simulator.set_wall_clock_progress_callback(
                options->wall_progress_interval_s.value(),
                [](frontier::SimTime simulation_time,
                   double /*wall_clock_elapsed_seconds*/) {
                    std::cerr
                        << "simulation_progress_s=" << std::setprecision(17)
                        << simulation_time.seconds() << '\n'
                        << std::flush;
                });
        }
        const frontier::metrics::SimulationOutput output =
            options->simulation_end_time_s.has_value()
                ? simulator.run_until(frontier::SimTime::from_seconds(
                      options->simulation_end_time_s.value()))
                : simulator.run();
        const double wall_clock_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          started_at)
                .count();

        if (!options->output_dir.has_value()) {
            std::cout << frontier::metrics::serialize_simulation_output_json(
                output);
            return 0;
        }
        write_artifacts(options.value(), output, wall_clock_seconds);
        std::cout << "wrote simulation artifacts to "
                  << options->output_dir->string() << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << kProgramName << ": error: " << error.what() << '\n';
        return 1;
    }
}
