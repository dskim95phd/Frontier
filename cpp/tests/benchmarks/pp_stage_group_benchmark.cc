#include "frontier/config/config.h"
#include "frontier/request_generator/workload.h"
#include "frontier/simulator/simulator.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <nlohmann/json.hpp>

#ifndef FRONTIER_TEST_FIXTURE_DIR
#error "FRONTIER_TEST_FIXTURE_DIR must be defined for PP benchmarks"
#endif

namespace {

using Json = nlohmann::json;
using frontier::config::parse_simulation_config_json;
using frontier::request_generator::parse_workload_csv;

std::string read_file(const std::string &path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open benchmark fixture: " + path);
    }
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

struct BenchmarkResult {
    std::string pipeline_event_mode;
    std::uint64_t pp = 0;
    double wall_clock_ms = 0.0;
    std::uint64_t events = 0;
    std::uint64_t batch_stages = 0;
    std::uint64_t predictor_calls = 0;
    std::uint64_t unique_timing_groups = 0;
    std::uint64_t timing_cache_hits = 0;
    std::uint64_t timing_cache_misses = 0;
    std::uint64_t unique_timing_templates = 0;
    std::size_t peak_event_queue = 0;
};

frontier::config::SimulationConfig make_config(std::uint64_t pp,
                                               std::string event_mode,
                                               const std::string &fixture) {
    Json root = Json::parse(fixture);
    Json &cluster = root.at("clusters").at("monolithic");
    cluster.at("parallelism").at("pipeline_parallel_size") = pp;
    // Keep a loaded DP pipeline while varying only PP.
    cluster.at("parallelism").at("data_parallel_size") = 2;
    cluster.at("scheduler").erase("num_blocks");
    cluster.at("scheduler")["pipeline_event_mode"] = event_mode;
    cluster.at("execution_model")["moe_layer_event_mode"] =
        "stage_group_scaled";
    // The benchmark deliberately supplies HBM capacity and lets the resolver
    // derive stage-local F/B/S and the PP-wide logical block count.
    cluster["gpu_memory"] = {
        {"auto_calculate_num_blocks", true},
        {"capacity_bytes_per_gpu", 80'000'000'000ULL},
        {"runtime_reserve_fraction", 0.10},
    };
    return parse_simulation_config_json(root.dump());
}

BenchmarkResult run_once(
    std::uint64_t pp, std::string event_mode, const std::string &fixture,
    const std::vector<frontier::request_generator::WorkloadRequest> &workload) {
    auto config = make_config(pp, event_mode, fixture);
    frontier::simulator::Simulator simulator(config, workload);
    const auto start = std::chrono::steady_clock::now();
    const frontier::metrics::SimulationOutput output = simulator.run();
    const auto end = std::chrono::steady_clock::now();

    std::set<std::uint64_t> timing_groups;
    std::uint64_t timing_cache_hits = 0;
    std::uint64_t timing_cache_misses = 0;
    std::uint64_t unique_timing_templates = 0;
    for (const auto &diagnostic : output.analytical_diagnostics) {
        for (const auto &[name, value] : diagnostic.values) {
            if (name == "timing_group_id" && std::isfinite(value) &&
                value >= 0.0) {
                timing_groups.insert(static_cast<std::uint64_t>(value));
            } else if (name == "timing_cache_hits_total" &&
                       std::isfinite(value) && value >= 0.0) {
                timing_cache_hits = std::max(timing_cache_hits,
                                             static_cast<std::uint64_t>(value));
            } else if (name == "timing_cache_misses_total" &&
                       std::isfinite(value) && value >= 0.0) {
                timing_cache_misses = std::max(
                    timing_cache_misses, static_cast<std::uint64_t>(value));
            } else if (name == "timing_cache_unique_templates_total" &&
                       std::isfinite(value) && value >= 0.0) {
                unique_timing_templates = std::max(
                    unique_timing_templates, static_cast<std::uint64_t>(value));
            }
        }
    }
    const double wall_clock_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    return BenchmarkResult{
        std::move(event_mode),
        pp,
        wall_clock_ms,
        output.aggregate.event_count,
        output.aggregate.batch_stage_count,
        output.aggregate.analytical_diagnostic_count,
        static_cast<std::uint64_t>(timing_groups.size()),
        timing_cache_hits,
        timing_cache_misses,
        unique_timing_templates,
        simulator.peak_event_queue_size(),
    };
}

void print_result(const BenchmarkResult &result) {
    std::cout << "mode=" << result.pipeline_event_mode << " pp=" << result.pp
              << " wall_clock_ms=" << std::fixed << std::setprecision(3)
              << result.wall_clock_ms << " events=" << result.events
              << " batch_stages=" << result.batch_stages
              << " predictor_calls=" << result.predictor_calls
              << " unique_timing_groups=" << result.unique_timing_groups
              << " timing_cache_hits=" << result.timing_cache_hits
              << " timing_cache_misses=" << result.timing_cache_misses
              << " unique_timing_templates=" << result.unique_timing_templates
              << " peak_event_queue=" << result.peak_event_queue << '\n';
}

} // namespace

int main(int argc, char **argv) {
    try {
        std::uint64_t iterations = 1;
        if (argc > 1) {
            const auto parsed = std::stoull(argv[1]);
            if (parsed == 0 || parsed > 100) {
                throw std::invalid_argument(
                    "iterations must be in the range [1, 100]");
            }
            iterations = parsed;
        }
        const std::string fixture =
            read_file(std::string{FRONTIER_TEST_FIXTURE_DIR} +
                      "/config/analytical_parallel_colocation.json");
        const auto workload = parse_workload_csv(
            "session_start_at,think_time,num_prefill_tokens,num_decode_tokens\n"
            "0,0,32,2\n"
            "0,0,48,2\n"
            "0,0,64,2\n"
            "0.0001,0,40,2\n"
            "0.0001,0,56,2\n"
            "0.0001,0,72,2\n"
            "0.0002,0,36,2\n"
            "0.0002,0,52,2\n"
            "0.0002,0,68,2\n");

        std::cout << "# PP stage-group benchmark (iterations=" << iterations
                  << ")\n";
        for (const std::string event_mode : {"exact", "collapsed"}) {
            for (const std::uint64_t pp : {1ULL, 4ULL, 24ULL}) {
                BenchmarkResult selected{};
                for (std::uint64_t iteration = 0; iteration < iterations;
                     ++iteration) {
                    const BenchmarkResult result =
                        run_once(pp, event_mode, fixture, workload);
                    // Report the last sample for deterministic machine
                    // parsing; callers can request multiple iterations to
                    // inspect noise.
                    selected = result;
                }
                print_result(selected);
            }
        }
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "pp_stage_group_benchmark: " << error.what() << '\n';
        return 1;
    }
}
