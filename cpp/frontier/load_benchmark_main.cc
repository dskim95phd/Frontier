#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

#include "frontier/config/config.h"
#include "frontier/metrics/output_contract.h"
#include "frontier/request_generator/workload.h"
#include "frontier/simulator/simulator.h"

namespace {

using OrderedJson = nlohmann::ordered_json;

struct InputPaths {
    std::filesystem::path config;
    std::filesystem::path workload;
    std::optional<std::filesystem::path> output;
};

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

std::optional<InputPaths> parse_input_paths(int argc, char *argv[]) {
    if (argc != 5 && argc != 7) {
        return std::nullopt;
    }

    std::optional<std::filesystem::path> config_path;
    std::optional<std::filesystem::path> workload_path;
    std::optional<std::filesystem::path> output_path;
    for (int index = 1; index < argc; index += 2) {
        const std::string_view option{argv[index]};
        const std::filesystem::path value{argv[index + 1]};
        if (option == "--config" && !config_path.has_value()) {
            config_path = value;
        } else if (option == "--workload" && !workload_path.has_value()) {
            workload_path = value;
        } else if (option == "--output" && !output_path.has_value()) {
            output_path = value;
        } else {
            return std::nullopt;
        }
    }
    if (!config_path.has_value() || !workload_path.has_value()) {
        return std::nullopt;
    }
    return InputPaths{std::move(config_path.value()),
                      std::move(workload_path.value()),
                      std::move(output_path)};
}

double milliseconds_between(frontier::SimTime end, frontier::SimTime start) {
    return (end.seconds() - start.seconds()) * 1'000.0;
}

OrderedJson summarize(const frontier::metrics::SimulationOutput &output,
                      double wall_clock_seconds) {
    OrderedJson root;
    root["schema_version"] = 1;
    root["run_id"] = output.run.run_id;
    root["wall_clock_seconds"] = wall_clock_seconds;
    root["counts"] = {
        {"requests", output.requests.size()},
        {"batches", output.aggregate.batch_count},
        {"batch_stages", output.aggregate.batch_stage_count},
        {"scheduler_iterations",
         output.aggregate.scheduler_iteration_count},
        {"events", output.aggregate.event_count},
        {"analytical_diagnostics",
         output.aggregate.analytical_diagnostic_count},
        {"moe_routing_records", output.aggregate.moe_routing_count},
        {"kv_cache_transfers", output.aggregate.kv_cache_transfer_count},
    };

    root["requests"] = OrderedJson::array();
    for (const frontier::metrics::RequestMetricsRecord &request :
         output.requests) {
        root["requests"].push_back({
            {"request_id", request.request_id.value()},
            {"arrived_at_s", request.arrived_at.seconds()},
            {"first_scheduled_at_s", request.first_scheduled_at.seconds()},
            {"prefill_completed_at_s",
             request.prefill_completed_at.seconds()},
            {"first_token_completed_at_s",
             request.first_token_completed_at.seconds()},
            {"completed_at_s", request.completed_at.seconds()},
            {"scheduling_delay_ms",
             milliseconds_between(request.first_scheduled_at,
                                  request.arrived_at)},
            {"ttft_ms", milliseconds_between(request.prefill_completed_at,
                                             request.arrived_at)},
            {"e2e_ms",
             milliseconds_between(request.completed_at, request.arrived_at)},
            {"num_processed_tokens", request.num_processed_tokens},
            {"preemption_count", request.preemption_count},
            {"replica_id", request.replica_id.value()},
            {"dp_id", request.dp_id.value()},
        });
    }

    std::map<std::size_t, std::uint64_t> batch_size_histogram;
    for (const auto &[cluster_type, aggregate] :
         output.aggregate.batches_by_cluster) {
        static_cast<void>(cluster_type);
        for (const auto &[batch_size, count] :
             aggregate.batch_size_histogram) {
            batch_size_histogram[batch_size] += count;
        }
    }
    root["batch_size_histogram"] = OrderedJson::object();
    for (const auto &[batch_size, count] : batch_size_histogram) {
        root["batch_size_histogram"][std::to_string(batch_size)] = count;
    }
    root["batch_size_histogram_by_cluster"] = OrderedJson::object();
    root["batch_summary_by_cluster"] = OrderedJson::object();
    for (const auto &[cluster_type_value, aggregate] :
         output.aggregate.batches_by_cluster) {
        const std::string cluster_type{
            frontier::to_string(cluster_type_value)};
        OrderedJson cluster_histogram = OrderedJson::object();
        for (const auto &[batch_size, count] :
             aggregate.batch_size_histogram) {
            cluster_histogram[std::to_string(batch_size)] = count;
        }
        root["batch_size_histogram_by_cluster"][cluster_type] =
            std::move(cluster_histogram);
        const double execution_ms = aggregate.predicted_execution_ms;
        root["batch_summary_by_cluster"][cluster_type] = {
            {"batches", aggregate.batch_count},
            {"mean_batch_size",
             aggregate.batch_count == 0
                 ? 0.0
                 : static_cast<double>(aggregate.request_slots) /
                       static_cast<double>(aggregate.batch_count)},
            {"execution_time_weighted_mean_batch_size",
             execution_ms == 0.0
                 ? 0.0
                 : aggregate.batch_size_execution_ms / execution_ms},
            {"predicted_execution_ms", execution_ms},
        };
    }

    root["kv_cache_transfers"] = OrderedJson::array();
    for (const frontier::metrics::KVCacheTransferMetricsRecord &transfer :
         output.kv_cache_transfers) {
        root["kv_cache_transfers"].push_back({
            {"transfer_id", transfer.transfer_id.value()},
            {"request_id", transfer.request_id.value()},
            {"size_bytes", transfer.size_bytes},
            {"predicted_time_ms", transfer.predicted_time_ms},
            {"started_at_s", transfer.started_at.seconds()},
            {"completed_at_s", transfer.completed_at.seconds()},
        });
    }
    return root;
}

void print_usage() {
    std::cerr
        << "Usage: frontier_load_benchmark --config <config.json> "
           "--workload <workload.csv> [--output <summary.json>]\n";
}

} // namespace

int main(int argc, char *argv[]) {
    try {
        const std::optional<InputPaths> input_paths =
            parse_input_paths(argc, argv);
        if (!input_paths.has_value()) {
            print_usage();
            return 2;
        }

        const frontier::config::SimulationConfig config =
            frontier::config::parse_simulation_config_json(
                read_text_file(input_paths->config));
        const auto workload = frontier::request_generator::parse_workload_csv(
            read_text_file(input_paths->workload));

        const auto started_at = std::chrono::steady_clock::now();
        frontier::simulator::Simulator simulator{config, workload};
        simulator.metrics().set_detailed_traces_enabled(false);
        simulator.set_runtime_validation_enabled(false);
        const frontier::metrics::SimulationOutput output = simulator.run();
        const double wall_clock_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          started_at)
                .count();
        const std::string summary =
            summarize(output, wall_clock_seconds).dump(2) + '\n';
        if (input_paths->output.has_value()) {
            std::ofstream summary_output{input_paths->output.value(),
                                         std::ios::binary};
            if (!summary_output) {
                throw std::runtime_error(
                    "failed to open summary output file: " +
                    input_paths->output->string());
            }
            summary_output << summary;
            if (!summary_output) {
                throw std::runtime_error(
                    "failed to write summary output file: " +
                    input_paths->output->string());
            }
        } else {
            std::cout << summary;
        }
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "frontier_load_benchmark: error: " << error.what()
                  << '\n';
        return 1;
    }
}
