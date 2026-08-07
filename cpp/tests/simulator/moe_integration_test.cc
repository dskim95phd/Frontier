#include "frontier/config/config.h"
#include "frontier/core/event.h"
#include "frontier/execution_time_predictor/analytical_roofline_execution_time_predictor.h"
#include "frontier/request_generator/workload.h"
#include "frontier/simulator/simulator.h"
#include "tests/test_support.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <utility>
#include <vector>

#ifndef FRONTIER_TEST_FIXTURE_DIR
#error "FRONTIER_TEST_FIXTURE_DIR must be defined for simulator tests"
#endif

namespace {

using frontier::EventType;
using frontier::config::parse_simulation_config_json;
using frontier::request_generator::parse_workload_csv;
using frontier::simulator::run_simulation;
using frontier::test::expect;
using frontier::test::read_text_file;

const std::filesystem::path kFixtureRoot{FRONTIER_TEST_FIXTURE_DIR};

frontier::config::SimulationConfig load_config(const char *name) {
    return parse_simulation_config_json(
        read_text_file(kFixtureRoot / "config" / name));
}

auto load_small_workload() {
    return parse_workload_csv(
        read_text_file(kFixtureRoot / "workloads/step3_pdd_small.csv"));
}

std::size_t count_events(const frontier::metrics::SimulationOutput &output,
                         EventType type) {
    return static_cast<std::size_t>(std::count_if(
        output.event_trace.begin(), output.event_trace.end(),
        [type](const frontier::Event &event) { return event.type() == type; }));
}

void test_local_moe_uses_fast_path() {
    const auto output = run_simulation(
        load_config("fixed_moe_local_colocation.json"), load_small_workload());

    expect(output.requests.size() == 2 &&
               count_events(output, EventType::kPrefillSync) == 0 &&
               count_events(output, EventType::kPrefillSyncCollective) == 0 &&
               count_events(output, EventType::kDecodeSync) == 0 &&
               count_events(output, EventType::kDecodeSyncCollective) == 0,
           "EP1/DP1 MoE must retain the direct stage path");
    expect(std::all_of(output.batch_stages.begin(), output.batch_stages.end(),
                       [](const auto &stage) {
                           return stage.execution_time.moe_grouped_gemm_ms >
                                      0.0 &&
                                  stage.execution_time.ep_dispatch_ms == 0.0 &&
                                  stage.execution_time.ep_combine_ms == 0.0;
                       }),
           "local MoE stages must expose compute without EP collectives");
}

void test_colocation_runs_all_sync_event_families() {
    const auto output = run_simulation(
        load_config("analytical_moe_ep4_colocation.json"),
        parse_workload_csv(
            "session_start_at,think_time,num_prefill_tokens,num_decode_tokens\n"
            "0,0,4,3\n"));
    const std::array types{
        EventType::kPrefillSync,
        EventType::kPrefillSyncCollective,
        EventType::kDecodeSync,
        EventType::kDecodeSyncCollective,
    };
    expect(
        std::all_of(types.begin(), types.end(),
                    [&output](EventType type) {
                        return count_events(output, type) > 0;
                    }),
        "synchronized co-location MoE must exercise all four event families");

    const bool has_prefill_idle = std::any_of(
        output.event_trace.begin(), output.event_trace.end(),
        [](const frontier::Event &event) {
            if (event.type() == EventType::kPrefillSync) {
                return event.as<frontier::PrefillSyncPayload>().is_idle;
            }
            return false;
        });
    const bool has_decode_pre_idle_event = std::any_of(
        output.event_trace.begin(), output.event_trace.end(),
        [](const frontier::Event &event) {
            if (event.type() != EventType::kDecodeSync) {
                return false;
            }
            const auto &payload = event.as<frontier::DecodeSyncPayload>();
            return payload.is_idle &&
                   payload.sync_phase == frontier::MoESyncPhase::kPreMoe;
        });
    const bool has_decode_post_idle_event = std::any_of(
        output.event_trace.begin(), output.event_trace.end(),
        [](const frontier::Event &event) {
            if (event.type() != EventType::kDecodeSync) {
                return false;
            }
            const auto &payload = event.as<frontier::DecodeSyncPayload>();
            return payload.is_idle &&
                   payload.sync_phase == frontier::MoESyncPhase::kPostMoe;
        });
    expect(has_prefill_idle && !has_decode_pre_idle_event &&
               has_decode_post_idle_event,
           "prefill emits idle events while monolithic decode compacts pre-MoE "
           "idles and emits lane-timed post-MoE arrivals");
    expect(output.requests.size() == 1 &&
               std::all_of(output.batch_stages.begin(),
                           output.batch_stages.end(),
                           [](const auto &stage) {
                               return stage.execution_time.total_ms() > 0.0;
                           }),
           "synchronized MoE must finish each real request and stage");
}

void test_pdd_moe_preserves_phi_kv_contract() {
    const auto workload = load_small_workload();
    const auto output =
        run_simulation(load_config("fixed_moe_sequential_pdd.json"), workload);
    expect(output.requests.size() == workload.size() &&
               output.kv_cache_transfers.size() == workload.size() &&
               count_events(output, EventType::kPrefillSync) > 0 &&
               count_events(output, EventType::kDecodeSync) > 0,
           "sequential PDD MoE must synchronize both clusters and complete "
           "transfers");

    constexpr std::uint64_t kBytesPerPrefillToken =
        32ULL * 4ULL * 128ULL * 2ULL * 2ULL;
    for (std::size_t index = 0; index < workload.size(); ++index) {
        expect(output.kv_cache_transfers.at(index).size_bytes ==
                   workload.at(index).num_prefill_tokens *
                       kBytesPerPrefillToken,
               "Phi KV transfer size must use model layers, KV heads, and head "
               "dim");
    }
}

void test_colocation_dp2_ep2_decode_lockstep() {
    auto config = load_config("analytical_moe_ep4_colocation.json");
    auto &runtime = config.cluster();
    // TP1*DP2 and MoE-TP1*EP2 describe one shared two-GPU domain.  Three
    // equal-size requests are routed as two requests on DP0 and one on DP1,
    // which makes the first decode attention work intentionally unequal while
    // the synchronized prefill keeps both decode lanes in the same wave.
    runtime.parallelism.tensor_parallel_size = 1;
    runtime.parallelism.pipeline_parallel_size = 2;
    runtime.parallelism.data_parallel_size = 2;
    runtime.parallelism.moe_tensor_parallel_size = 1;
    runtime.parallelism.moe_expert_parallel_size = 2;
    runtime.scheduler.batch_size_cap = 8;
    runtime.scheduler.max_tokens_in_batch = 32;
    const auto output = run_simulation(
        config,
        parse_workload_csv(
            "session_start_at,think_time,num_prefill_tokens,num_decode_tokens\n"
            "0,0,8,2\n"
            "0,0,8,2\n"
            "0,0,8,2\n"));

    struct DecodeBoundary {
        std::vector<double> real_pre_arrivals;
        bool has_idle_pre = false;
        double pre_collective = -1.0;
        double post_collective = -1.0;
    };
    using GroupKey = std::pair<std::int64_t, std::int64_t>;
    std::map<GroupKey, DecodeBoundary> boundaries;
    for (const frontier::Event &event : output.event_trace) {
        if (event.type() == EventType::kDecodeSync) {
            const auto &payload = event.as<frontier::DecodeSyncPayload>();
            if (payload.stage_id != frontier::StageId{0} ||
                payload.layer_id != frontier::LayerId{0}) {
                continue;
            }
            DecodeBoundary &boundary =
                boundaries[{payload.sync_group_id.value(),
                            payload.sync_generation.value()}];
            if (payload.sync_phase == frontier::MoESyncPhase::kPreMoe) {
                if (payload.is_idle) {
                    boundary.has_idle_pre = true;
                } else {
                    boundary.real_pre_arrivals.push_back(event.time.seconds());
                }
            }
            continue;
        }
        if (event.type() != EventType::kDecodeSyncCollective) {
            continue;
        }
        const auto &payload = event.as<frontier::DecodeSyncCollectivePayload>();
        if (payload.stage_id != frontier::StageId{0} ||
            payload.layer_id != frontier::LayerId{0}) {
            continue;
        }
        DecodeBoundary &boundary = boundaries[{
            payload.sync_group_id.value(), payload.sync_generation.value()}];
        if (payload.sync_phase == frontier::MoESyncPhase::kPreMoe) {
            boundary.pre_collective = event.time.seconds();
        } else {
            boundary.post_collective = event.time.seconds();
        }
    }

    const auto aligned = std::find_if(
        boundaries.begin(), boundaries.end(), [](const auto &entry) {
            const DecodeBoundary &boundary = entry.second;
            return boundary.real_pre_arrivals.size() == 2 &&
                   !boundary.has_idle_pre && boundary.pre_collective >= 0.0 &&
                   boundary.post_collective >= 0.0;
        });
    expect(aligned != boundaries.end(),
           "DP2/EP2 decode must form an aligned two-real-lane group");
    if (aligned == boundaries.end()) {
        return;
    }
    const DecodeBoundary &boundary = aligned->second;
    expect(std::abs(boundary.real_pre_arrivals.at(0) -
                    boundary.real_pre_arrivals.at(1)) > 1e-12,
           "aligned regression must exercise unequal DP attention arrivals");
    const double latest_pre = *std::max_element(
        boundary.real_pre_arrivals.begin(), boundary.real_pre_arrivals.end());
    expect(std::abs(boundary.pre_collective - latest_pre) < 1e-12,
           "decode collective must release at the latest real DP arrival");

    // No DBO is modeled: a later aligned group's expert interval may not
    // overlap this replica/stage's current dispatch/FFN/combine interval.
    for (const auto &[key, other] : boundaries) {
        if (key == aligned->first || other.pre_collective < 0.0 ||
            other.post_collective < 0.0) {
            continue;
        }
        const bool overlaps = other.pre_collective < boundary.post_collective &&
                              boundary.pre_collective < other.post_collective;
        expect(!overlaps,
               "PDD/co-location shared expert domain must not overlap groups");
    }
}

void test_colocation_dp2_ep2_repredicts_aggregated_expert_tokens() {
    auto config = load_config("analytical_moe_ep4_colocation.json");
    auto &runtime = config.cluster();
    runtime.parallelism.tensor_parallel_size = 1;
    runtime.parallelism.pipeline_parallel_size = 2;
    runtime.parallelism.data_parallel_size = 2;
    runtime.parallelism.moe_tensor_parallel_size = 1;
    runtime.parallelism.moe_expert_parallel_size = 2;
    runtime.scheduler.batch_size_cap = 8;
    runtime.scheduler.max_tokens_in_batch = 32;
    runtime.moe_routing.mode = frontier::config::MoeRoutingMode::kUniformLegacy;
    runtime.moe_routing.distribution =
        frontier::config::MoeRoutingDistribution::kBalanced;

    const auto output = run_simulation(
        config,
        parse_workload_csv(
            "session_start_at,think_time,num_prefill_tokens,num_decode_tokens\n"
            "0,0,8,2\n"
            "0,0,8,2\n"
            "0,0,8,2\n"));

    using RoutingRecord = frontier::metrics::MoERoutingMetricsRecord;
    std::map<frontier::MoESyncGroupId, std::vector<const RoutingRecord *>>
        routing_by_group;
    for (const RoutingRecord &record : output.moe_routing) {
        const auto batch = std::find_if(
            output.batches.begin(), output.batches.end(),
            [&record](const auto &candidate) {
                return candidate.batch_id == record.batch_id;
            });
        if (record.cluster_type == frontier::ClusterType::kMonolithic &&
            record.stage_id == frontier::StageId{0} &&
            record.layer_id == frontier::LayerId{0} &&
            batch != output.batches.end() && batch->num_prefill_tokens == 0 &&
            batch->num_decode_tokens > 0) {
            routing_by_group[record.sync_group_id].push_back(&record);
        }
    }
    const auto group = std::find_if(
        routing_by_group.begin(), routing_by_group.end(),
        [](const auto &entry) { return entry.second.size() == 2; });
    expect(group != routing_by_group.end(),
           "aggregate-token regression requires two real DP routing records");
    if (group == routing_by_group.end()) {
        return;
    }

    const std::size_t expert_count =
        group->second.front()->global_expert_tokens.size();
    const std::size_t lane_count = group->second.front()->lane_times_ms.size();
    frontier::execution_time_predictor::detail::RoutingAllocation allocation;
    allocation.global_expert_tokens.assign(expert_count, 0);
    allocation.lane_expert_tokens.assign(lane_count, {});
    std::vector<double> time_sum_by_lane(lane_count, 0.0);
    for (const RoutingRecord *record : group->second) {
        expect(record->global_expert_tokens.size() == expert_count &&
                   record->lane_times_ms.size() == lane_count &&
                   record->lane_expert_tokens.size() == lane_count,
               "DP routing records must share one expert/EP domain");
        allocation.input_tokens += record->input_tokens;
        allocation.routed_tokens += record->routed_tokens;
        for (std::size_t expert = 0; expert < expert_count; ++expert) {
            allocation.global_expert_tokens.at(expert) +=
                record->global_expert_tokens.at(expert);
        }
        for (std::size_t lane = 0; lane < lane_count; ++lane) {
            if (allocation.lane_expert_tokens.at(lane).empty()) {
                allocation.lane_expert_tokens.at(lane).assign(
                    record->lane_expert_tokens.at(lane).size(), 0);
            }
            expect(allocation.lane_expert_tokens.at(lane).size() ==
                       record->lane_expert_tokens.at(lane).size(),
                   "DP routing records must share expert ownership");
            for (std::size_t expert = 0;
                 expert < record->lane_expert_tokens.at(lane).size();
                 ++expert) {
                allocation.lane_expert_tokens.at(lane).at(expert) +=
                    record->lane_expert_tokens.at(lane).at(expert);
            }
            time_sum_by_lane.at(lane) += record->lane_times_ms.at(lane);
        }
    }

    namespace analytical = frontier::execution_time_predictor::detail;
    analytical::MoEModel moe_model{};
    moe_model.hidden_size = runtime.model.hidden_size;
    moe_model.intermediate_size = runtime.model.moe_intermediate_size;
    moe_model.model_num_experts = runtime.model.num_experts;
    moe_model.num_shared_experts = runtime.model.num_shared_experts;
    moe_model.moe_tensor_parallel_size =
        runtime.parallelism.moe_tensor_parallel_size;
    moe_model.gated_mlp = runtime.model.gated_mlp;
    moe_model.fused_add_norm = runtime.model.fused_add_norm;
    const auto &analytical_config = runtime.execution_model.analytical;
    const analytical::MoEOperatorPrecisions precisions{
        analytical::precision_from_string(
            analytical_config.moe_expert_precision()),
        analytical::precision_from_string(
            analytical_config.moe_router_precision()),
        analytical::precision_from_string(analytical_config.dense_precision()),
        analytical::precision_from_string(
            analytical_config.moe_expert_weight_precision()),
        analytical::precision_from_string(
            analytical_config.moe_expert_activation_precision()),
        analytical::precision_from_string(
            analytical_config.moe_router_weight_precision()),
        analytical::precision_from_string(
            analytical_config.moe_router_activation_precision()),
        analytical::precision_from_string(
            analytical_config.dense_weight_precision()),
        analytical::precision_from_string(
            analytical_config.dense_activation_precision()),
    };
    const analytical::MoELanePrediction aggregate_prediction =
        analytical::predict_moe_lanes(
            analytical::DeviceCeilings::from_config(analytical_config),
            analytical::AnalyticalConfig{}, moe_model, allocation,
            runtime.model.router_topk, precisions);
    const double time_sum_critical_ms =
        *std::max_element(time_sum_by_lane.begin(), time_sum_by_lane.end());
    const double aggregate_critical_ms =
        aggregate_prediction.critical_lane_time_ms;

    double pre_collective_ms = -1.0;
    double post_collective_ms = -1.0;
    for (const frontier::Event &event : output.event_trace) {
        if (event.type() != EventType::kDecodeSyncCollective) {
            continue;
        }
        const auto &payload =
            event.as<frontier::DecodeSyncCollectivePayload>();
        if (payload.cluster_type != frontier::ClusterType::kMonolithic ||
            payload.stage_id != frontier::StageId{0} ||
            payload.layer_id != frontier::LayerId{0} ||
            payload.sync_group_id != group->first) {
            continue;
        }
        if (payload.sync_phase == frontier::MoESyncPhase::kPreMoe) {
            pre_collective_ms = event.time.seconds() * 1e3;
        } else {
            post_collective_ms = event.time.seconds() * 1e3;
        }
    }
    expect(pre_collective_ms >= 0.0 && post_collective_ms >= 0.0,
           "aligned group must expose pre/post expert collectives");
    const double observed_critical_ms =
        post_collective_ms - pre_collective_ms;
    const std::string diagnostic =
        " time_sum_critical_ms=" + std::to_string(time_sum_critical_ms) +
        " aggregate_critical_ms=" + std::to_string(aggregate_critical_ms) +
        " observed_critical_ms=" + std::to_string(observed_critical_ms);
    expect(std::abs(time_sum_critical_ms - aggregate_critical_ms) > 1e-6,
           "regression must distinguish time-sum from token aggregation:" +
               diagnostic);
    expect(std::abs(observed_critical_ms - time_sum_critical_ms) > 1e-6,
           "collective must not use the DP lane-time sum:" + diagnostic);
    expect(std::abs(observed_critical_ms - aggregate_critical_ms) < 1e-9,
           "collective must re-predict one aggregated expert allocation:" +
               diagnostic);
}

void test_pdd_dp2_ep2_waits_for_slowest_real_attention_lane() {
    auto config = load_config("fixed_moe_sequential_pdd.json");
    const auto analytical_source =
        load_config("analytical_moe_ep4_colocation.json");

    auto &prefill = config.pdd().clusters.prefill;
    prefill.scheduler.max_tokens_in_batch = 1024;
    prefill.scheduler.num_blocks = 1024;

    auto &decode = config.pdd().clusters.decode;
    decode.parallelism.tensor_parallel_size = 1;
    decode.parallelism.pipeline_parallel_size = 1;
    decode.parallelism.data_parallel_size = 2;
    decode.parallelism.moe_tensor_parallel_size = 1;
    decode.parallelism.moe_expert_parallel_size = 2;
    decode.scheduler.batch_size_cap = 1;
    decode.scheduler.max_tokens_in_batch = 1;
    decode.scheduler.num_blocks = 1024;
    decode.execution_model = analytical_source.cluster().execution_model;
    decode.execution_model.analytical.tensor_parallel_size = 1;
    decode.moe_routing.distribution =
        frontier::config::MoeRoutingDistribution::kBalanced;

    // Keep the two PDD handoffs effectively simultaneous so the measured
    // pre-MoE difference comes from context-dependent attention work.
    config.pdd().kv_cache_transfer.network_bandwidth_gbps = 1.0e15;
    config.pdd().kv_cache_transfer.network_latency_ms = 0.0;

    const auto output = run_simulation(
        config,
        parse_workload_csv(
            "session_start_at,think_time,num_prefill_tokens,num_decode_tokens\n"
            "0,0,8,1\n"
            "0,0,256,1\n"));

    struct SyncArrival {
        frontier::SimTime time;
        frontier::DecodeSyncPayload payload;
    };
    std::vector<SyncArrival> real_pre;
    std::vector<SyncArrival> idle_pre;
    std::vector<
        std::pair<frontier::SimTime, frontier::DecodeSyncCollectivePayload>>
        pre_collectives;
    for (const frontier::Event &event : output.event_trace) {
        if (event.type() == EventType::kDecodeSync) {
            const auto &payload = event.as<frontier::DecodeSyncPayload>();
            if (payload.cluster_type != frontier::ClusterType::kDecode ||
                payload.layer_id.value() != 0 ||
                payload.sync_phase != frontier::MoESyncPhase::kPreMoe) {
                continue;
            }
            (payload.is_idle ? idle_pre : real_pre)
                .push_back(SyncArrival{event.time, payload});
        } else if (event.type() == EventType::kDecodeSyncCollective) {
            const auto &payload =
                event.as<frontier::DecodeSyncCollectivePayload>();
            if (payload.cluster_type == frontier::ClusterType::kDecode &&
                payload.layer_id.value() == 0 &&
                payload.sync_phase == frontier::MoESyncPhase::kPreMoe) {
                pre_collectives.emplace_back(event.time, payload);
            }
        }
    }

    expect(real_pre.size() == 2,
           "DP2 decode must emit two real layer-0 pre-MoE arrivals");
    expect(idle_pre.empty(),
           "an active DP2 decode lane must not be replaced by a dummy");
    expect(real_pre.at(0).payload.dp_id != real_pre.at(1).payload.dp_id &&
               real_pre.at(0).payload.sync_group_id ==
                   real_pre.at(1).payload.sync_group_id &&
               real_pre.at(0).payload.sync_generation ==
                   real_pre.at(1).payload.sync_generation,
           "unequal real DP lanes must share one MoE group and generation");
    expect(real_pre.at(0).time != real_pre.at(1).time,
           "regression workload must produce unequal attention completion");
    expect(pre_collectives.size() == 1,
           "one aligned DP2 forward must emit one pre-MoE collective");

    const frontier::SimTime slowest =
        std::max(real_pre.at(0).time, real_pre.at(1).time);
    expect(pre_collectives.front().first == slowest &&
               pre_collectives.front().second.sync_group_id ==
                   real_pre.front().payload.sync_group_id &&
               pre_collectives.front().second.sync_generation ==
                   real_pre.front().payload.sync_generation,
           "expert FFN must start at the slowest real DP attention arrival");
}

void test_pdd_dp2_ep2_reserves_dummy_lane_until_stage_end() {
    auto config = load_config("fixed_moe_sequential_pdd.json");
    auto &prefill = config.pdd().clusters.prefill;
    prefill.execution_model.fixed.stage_latencies_ms = {1.0e-6, 1.0e-6};
    prefill.scheduler.max_tokens_in_batch = 64;

    auto &decode = config.pdd().clusters.decode;
    decode.parallelism.tensor_parallel_size = 1;
    decode.parallelism.pipeline_parallel_size = 1;
    decode.parallelism.data_parallel_size = 2;
    decode.parallelism.moe_tensor_parallel_size = 1;
    decode.parallelism.moe_expert_parallel_size = 2;
    decode.scheduler.batch_size_cap = 1;
    decode.scheduler.max_tokens_in_batch = 1;
    decode.execution_model.fixed.stage_latencies_ms = {1.0};

    config.pdd().kv_cache_transfer.network_bandwidth_gbps = 1.0e15;
    config.pdd().kv_cache_transfer.network_latency_ms = 0.0;

    const auto output = run_simulation(
        config,
        parse_workload_csv(
            "session_start_at,think_time,num_prefill_tokens,num_decode_tokens\n"
            "0,0,4,1\n"
            "0.01,0,4,1\n"));

    struct RealPreArrival {
        frontier::SimTime time;
        frontier::DecodeSyncPayload payload;
    };
    std::vector<RealPreArrival> layer_zero_real;
    for (const frontier::Event &event : output.event_trace) {
        if (event.type() != EventType::kDecodeSync) {
            continue;
        }
        const auto &payload = event.as<frontier::DecodeSyncPayload>();
        if (payload.cluster_type == frontier::ClusterType::kDecode &&
            payload.layer_id.value() == 0 &&
            payload.sync_phase == frontier::MoESyncPhase::kPreMoe &&
            !payload.is_idle) {
            layer_zero_real.push_back(RealPreArrival{event.time, payload});
        }
    }
    expect(layer_zero_real.size() == 2,
           "staggered DP2 workload must create two real decode forwards");
    expect(layer_zero_real.at(0).payload.sync_generation !=
               layer_zero_real.at(1).payload.sync_generation,
           "late real work must use the next aligned generation");

    const auto &first = layer_zero_real.front().payload;
    frontier::BatchId persistent_idle;
    std::size_t first_group_idle_pre_count = 0;
    for (const frontier::Event &event : output.event_trace) {
        if (event.type() != EventType::kDecodeSync) {
            continue;
        }
        const auto &payload = event.as<frontier::DecodeSyncPayload>();
        if (payload.cluster_type != frontier::ClusterType::kDecode ||
            payload.sync_group_id != first.sync_group_id ||
            payload.sync_generation != first.sync_generation ||
            payload.sync_phase != frontier::MoESyncPhase::kPreMoe ||
            !payload.is_idle) {
            continue;
        }
        if (!persistent_idle.valid()) {
            persistent_idle = payload.batch_id;
        }
        expect(payload.batch_id == persistent_idle,
               "one dummy batch must persist across all MoE layers");
        ++first_group_idle_pre_count;
    }
    expect(persistent_idle.valid() && first_group_idle_pre_count > 1,
           "the first aligned forward must retain its dummy across layers");

    frontier::SimTime first_stage_end;
    for (const frontier::Event &event : output.event_trace) {
        if (event.type() != EventType::kBatchStageEnd) {
            continue;
        }
        const auto &payload = event.as<frontier::BatchStageEndPayload>();
        if (payload.cluster_type == frontier::ClusterType::kDecode &&
            payload.stage_id.value() == 0 &&
            payload.batch_id == first.batch_id) {
            first_stage_end = event.time;
            break;
        }
    }
    expect(first_stage_end.valid(),
           "first aligned decode forward must emit a stage end");
    const double second_attention_start_s =
        layer_zero_real.at(1).time.seconds() -
        layer_zero_real.at(1).payload.elapsed_component_ms * 1e-3;
    expect(second_attention_start_s >= first_stage_end.seconds(),
           "a late real batch must wait for the reserved aligned stage");
}

void test_kimi_k2_dense_prefix_runs_with_uneven_pp() {
    auto config = load_config("analytical_moe_ep4_colocation.json");
    auto &runtime = config.cluster();
    runtime.model =
        frontier::config::load_model_config("moonshotai/Kimi-K2-Instruct");
    runtime.parallelism.tensor_parallel_size = 4;
    runtime.parallelism.pipeline_parallel_size = 4;
    runtime.parallelism.data_parallel_size = 2;
    runtime.parallelism.moe_tensor_parallel_size = 1;
    runtime.parallelism.moe_expert_parallel_size = 8;
    runtime.execution_model.analytical.tensor_parallel_size = 4;
    runtime.execution_model.analytical.precision = "fp8";
    runtime.execution_model.analytical.operator_precisions.moe_expert_weight =
        "fp4";
    runtime.execution_model.analytical.operator_precisions
        .moe_expert_activation = "fp8";

    const auto output = run_simulation(
        config,
        parse_workload_csv(
            "session_start_at,think_time,num_prefill_tokens,num_decode_tokens\n"
            "0,0,1,1\n"));
    expect(output.requests.size() == 1 && output.batch_stages.size() >= 4 &&
               output.batch_stages.size() % 4 == 0,
           "Kimi K2 PP4 must complete every four-stage pipeline pass");
    expect(std::any_of(output.batch_stages.begin(), output.batch_stages.end(),
                       [](const auto &stage) {
                           return stage.stage_id.value() == 3 &&
                                  stage.execution_time.lm_head_ms > 0.0;
                       }),
           "Kimi K2 final stage must report LM-head projection time");
}

void test_first_layer_scaled_repeats_dp_attention_wait() {
    auto detailed_config = load_config("fixed_moe_sequential_pdd.json");
    const auto analytical_source =
        load_config("analytical_moe_ep4_colocation.json");
    const auto model =
        frontier::config::load_model_config("moonshotai/Kimi-K2-Instruct");

    auto &prefill = detailed_config.pdd().clusters.prefill;
    prefill.model = model;
    prefill.parallelism.tensor_parallel_size = 1;
    prefill.parallelism.pipeline_parallel_size = 1;
    prefill.parallelism.data_parallel_size = 1;
    prefill.parallelism.moe_tensor_parallel_size = 1;
    prefill.parallelism.moe_expert_parallel_size = 1;
    prefill.scheduler.batch_size_cap = 2;
    prefill.scheduler.max_tokens_in_batch = 1024;
    prefill.scheduler.num_blocks = 1024;
    prefill.execution_model.type = frontier::config::ExecutionModelType::kFixed;
    prefill.execution_model.fixed.stage_latencies_ms = {1.0e-6};

    auto &decode = detailed_config.pdd().clusters.decode;
    decode.model = model;
    decode.parallelism.tensor_parallel_size = 1;
    decode.parallelism.pipeline_parallel_size = 1;
    decode.parallelism.data_parallel_size = 2;
    decode.parallelism.moe_tensor_parallel_size = 1;
    decode.parallelism.moe_expert_parallel_size = 2;
    decode.scheduler.batch_size_cap = 1;
    decode.scheduler.max_tokens_in_batch = 1;
    decode.scheduler.num_blocks = 1024;
    decode.execution_model = analytical_source.cluster().execution_model;
    decode.execution_model.analytical.tensor_parallel_size = 1;
    decode.execution_model.analytical.moe_layer_event_mode = "detailed";
    decode.moe_routing.mode = frontier::config::MoeRoutingMode::kUniformLegacy;
    decode.moe_routing.distribution =
        frontier::config::MoeRoutingDistribution::kBalanced;

    detailed_config.pdd().kv_cache_transfer.network_bandwidth_gbps = 1.0e15;
    detailed_config.pdd().kv_cache_transfer.network_latency_ms = 0.0;
    auto scaled_config = detailed_config;
    scaled_config.pdd()
        .clusters.decode.execution_model.analytical.moe_layer_event_mode =
        "first_layer_scaled";

    const auto workload = parse_workload_csv(
        "session_start_at,think_time,num_prefill_tokens,num_decode_tokens\n"
        "0,0,8,1\n"
        "0,0,256,1\n");
    const auto detailed = run_simulation(detailed_config, workload);
    const auto scaled = run_simulation(scaled_config, workload);

    struct LaneObservation {
        frontier::DataParallelId dp_id;
        double arrival_ms = 0.0;
        double attention_ms = 0.0;
    };
    const auto layer_arrivals = [](const auto &output,
                                   frontier::LayerId layer_id) {
        std::vector<LaneObservation> result;
        for (const frontier::Event &event : output.event_trace) {
            if (event.type() != EventType::kDecodeSync) {
                continue;
            }
            const auto &payload = event.as<frontier::DecodeSyncPayload>();
            if (payload.cluster_type == frontier::ClusterType::kDecode &&
                payload.stage_id == frontier::StageId{0} &&
                payload.layer_id == layer_id &&
                payload.sync_phase == frontier::MoESyncPhase::kPreMoe &&
                !payload.is_idle) {
                result.push_back(LaneObservation{payload.dp_id,
                                                 event.time.seconds() * 1e3,
                                                 payload.elapsed_component_ms});
            }
        }
        return result;
    };
    const auto detailed_arrivals =
        layer_arrivals(detailed, frontier::LayerId{0});
    const auto detailed_repeated_arrivals =
        layer_arrivals(detailed, frontier::LayerId{1});
    const auto scaled_arrivals = layer_arrivals(scaled, frontier::LayerId{0});
    expect(detailed_arrivals.size() == 2 && scaled_arrivals.size() == 2,
           "scaled wait regression requires two real DP attention arrivals");
    expect(detailed_repeated_arrivals.size() == 2,
           "detailed mode must expose a repeated MoE attention layer");

    const double first_attention_delta_ms =
        std::abs(detailed_arrivals.at(0).attention_ms -
                 detailed_arrivals.at(1).attention_ms);
    const double repeated_attention_delta_ms =
        std::abs(detailed_repeated_arrivals.at(0).attention_ms -
                 detailed_repeated_arrivals.at(1).attention_ms);
    const double arrival_delta_ms =
        std::abs(detailed_arrivals.at(0).arrival_ms -
                 detailed_arrivals.at(1).arrival_ms);
    expect(first_attention_delta_ms > 1e-9 &&
               repeated_attention_delta_ms > 1e-9 &&
               std::abs(first_attention_delta_ms - arrival_delta_ms) < 1e-9,
           "regression workload must isolate unequal DP attention work");

    const auto synchronization_wait_by_dp = [](const auto &output) {
        std::array<double, 2> waits{-1.0, -1.0};
        for (const auto &stage : output.batch_stages) {
            if (stage.cluster_type == frontier::ClusterType::kDecode &&
                stage.stage_id == frontier::StageId{0}) {
                waits.at(stage.dp_id.index()) =
                    stage.execution_time.synchronization_wait_ms;
            }
        }
        return waits;
    };
    const auto detailed_waits = synchronization_wait_by_dp(detailed);
    const auto scaled_waits = synchronization_wait_by_dp(scaled);
    const auto synchronization_breakdown_by_dp = [](const auto &output) {
        std::array<frontier::entities::ExecutionTime, 2> breakdown{};
        for (const auto &stage : output.batch_stages) {
            if (stage.cluster_type == frontier::ClusterType::kDecode &&
                stage.stage_id == frontier::StageId{0}) {
                breakdown.at(stage.dp_id.index()) = stage.execution_time;
            }
        }
        return breakdown;
    };
    const auto detailed_breakdown = synchronization_breakdown_by_dp(detailed);
    const auto scaled_breakdown = synchronization_breakdown_by_dp(scaled);
    expect(detailed_waits.at(0) >= 0.0 && detailed_waits.at(1) >= 0.0 &&
               scaled_waits.at(0) >= 0.0 && scaled_waits.at(1) >= 0.0,
           "both DP decode stages must report synchronization wait");

    const frontier::DataParallelId fast_dp =
        detailed_arrivals.at(0).attention_ms <
                detailed_arrivals.at(1).attention_ms
            ? detailed_arrivals.at(0).dp_id
            : detailed_arrivals.at(1).dp_id;
    const frontier::DataParallelId slow_dp =
        fast_dp == detailed_arrivals.at(0).dp_id
            ? detailed_arrivals.at(1).dp_id
            : detailed_arrivals.at(0).dp_id;
    const double detailed_wait_gap_ms =
        detailed_waits.at(fast_dp.index()) - detailed_waits.at(slow_dp.index());
    const double scaled_wait_gap_ms =
        scaled_waits.at(fast_dp.index()) - scaled_waits.at(slow_dp.index());
    std::uint64_t logical_moe_layers = 0;
    for (std::uint64_t layer = 0; layer < model.num_layers; ++layer) {
        logical_moe_layers += model.is_moe_layer(layer) ? 1 : 0;
    }
    const double expected_wait_gap_ms =
        first_attention_delta_ms +
        repeated_attention_delta_ms *
            static_cast<double>(logical_moe_layers - 1);
    const std::string diagnostic =
        " first_attention_delta_ms=" +
        std::to_string(first_attention_delta_ms) +
        " repeated_attention_delta_ms=" +
        std::to_string(repeated_attention_delta_ms) +
        " logical_moe_layers=" + std::to_string(logical_moe_layers) +
        " expected_wait_gap_ms=" + std::to_string(expected_wait_gap_ms) +
        " detailed_wait_gap_ms=" + std::to_string(detailed_wait_gap_ms) +
        " scaled_wait_gap_ms=" + std::to_string(scaled_wait_gap_ms);
    expect(std::abs(detailed_wait_gap_ms - expected_wait_gap_ms) < 1e-6,
           "detailed mode must repeat the DP attention wait per layer:" +
               diagnostic);
    expect(std::abs(scaled_wait_gap_ms - expected_wait_gap_ms) < 1e-6,
           "first_layer_scaled must repeat the DP attention wait per layer:" +
               diagnostic);
    const double detailed_pre_barrier_gap_ms =
        detailed_breakdown.at(fast_dp.index()).moe_pre_barrier_wait_ms -
        detailed_breakdown.at(slow_dp.index()).moe_pre_barrier_wait_ms;
    const double scaled_pre_barrier_gap_ms =
        scaled_breakdown.at(fast_dp.index()).moe_pre_barrier_wait_ms -
        scaled_breakdown.at(slow_dp.index()).moe_pre_barrier_wait_ms;
    expect(
        std::abs(detailed_pre_barrier_gap_ms - expected_wait_gap_ms) < 1e-6 &&
            std::abs(scaled_pre_barrier_gap_ms - expected_wait_gap_ms) < 1e-6,
        "explicit pre-MoE barrier breakdown must isolate the repeated DP "
        "wait:" +
            diagnostic);
    for (const auto &execution : scaled_breakdown) {
        const double attributed_ms =
            execution.moe_pre_barrier_wait_ms +
            execution.moe_ep_aggregation_extra_ms +
            execution.synchronization_unattributed_wait_ms -
            execution.synchronization_attribution_overlap_ms;
        expect(std::abs(execution.synchronization_wait_ms - attributed_ms) <
                   1e-6,
               "synchronization breakdown must sum to total wait");
    }
}

void test_kimi_k2_first_layer_scaled_preserves_completion_time() {
    auto detailed_config = load_config("analytical_moe_ep4_colocation.json");
    auto &runtime = detailed_config.cluster();
    runtime.model =
        frontier::config::load_model_config("moonshotai/Kimi-K2-Instruct");
    runtime.parallelism.tensor_parallel_size = 4;
    runtime.parallelism.pipeline_parallel_size = 4;
    runtime.parallelism.data_parallel_size = 2;
    runtime.parallelism.moe_tensor_parallel_size = 1;
    runtime.parallelism.moe_expert_parallel_size = 8;
    runtime.execution_model.analytical.tensor_parallel_size = 4;
    runtime.execution_model.analytical.precision = "fp8";
    runtime.execution_model.analytical.operator_precisions.moe_expert_weight =
        "fp4";
    runtime.execution_model.analytical.operator_precisions
        .moe_expert_activation = "fp8";
    auto scaled_config = detailed_config;
    scaled_config.cluster().execution_model.analytical.moe_layer_event_mode =
        "first_layer_scaled";
    const auto workload = parse_workload_csv(
        "session_start_at,think_time,num_prefill_tokens,num_decode_tokens\n"
        "0,0,8,2\n");

    const auto detailed = run_simulation(detailed_config, workload);
    const auto scaled = run_simulation(scaled_config, workload);
    expect(
        detailed.requests.size() == 1 && scaled.requests.size() == 1 &&
            std::abs(detailed.requests.front().completed_at.seconds() -
                     scaled.requests.front().completed_at.seconds()) < 1e-12,
        "scaled MoE layers must preserve request completion time: " +
            std::to_string(detailed.requests.front().completed_at.seconds()) +
            " vs " +
            std::to_string(scaled.requests.front().completed_at.seconds()));
    expect(scaled.aggregate.event_count < detailed.aggregate.event_count &&
               scaled.aggregate.moe_routing_count * 10 <
                   detailed.aggregate.moe_routing_count,
           "scaled MoE layers must materially reduce events and routing "
           "records");
}

void test_kimi_k2_first_layer_scaled_preserves_pdd_completion_time() {
    auto detailed_config = load_config("fixed_moe_sequential_pdd.json");
    const auto model =
        frontier::config::load_model_config("moonshotai/Kimi-K2-Instruct");
    const auto configure_cluster =
        [&model](frontier::config::ClusterRuntimeConfig &cluster,
                 std::uint64_t data_parallel, std::uint64_t expert_parallel) {
            cluster.model = model;
            cluster.parallelism.tensor_parallel_size = 4;
            cluster.parallelism.pipeline_parallel_size = 4;
            cluster.parallelism.data_parallel_size = data_parallel;
            cluster.parallelism.moe_tensor_parallel_size = 1;
            cluster.parallelism.moe_expert_parallel_size = expert_parallel;
            cluster.execution_model.type =
                frontier::config::ExecutionModelType::kAnalytical;
            cluster.execution_model.analytical.tensor_parallel_size = 4;
            cluster.execution_model.analytical.precision = "fp8";
            cluster.execution_model.analytical.operator_precisions
                .moe_expert_weight = "fp4";
            cluster.execution_model.analytical.operator_precisions
                .moe_expert_activation = "fp8";
        };
    configure_cluster(detailed_config.pdd().clusters.prefill, 1, 4);
    configure_cluster(detailed_config.pdd().clusters.decode, 2, 8);
    detailed_config.pdd().clusters.prefill.moe_routing.distribution =
        frontier::config::MoeRoutingDistribution::kBalanced;
    detailed_config.pdd().clusters.decode.moe_routing.distribution =
        frontier::config::MoeRoutingDistribution::kBalanced;
    detailed_config.pdd().clusters.prefill.moe_routing.mode =
        frontier::config::MoeRoutingMode::kUniformLegacy;
    detailed_config.pdd().clusters.decode.moe_routing.mode =
        frontier::config::MoeRoutingMode::kUniformLegacy;
    detailed_config.pdd().kv_cache_transfer.kv_cache_dtype_size_bytes = 1.0;
    auto scaled_config = detailed_config;
    scaled_config.pdd()
        .clusters.prefill.execution_model.analytical.moe_layer_event_mode =
        "first_layer_scaled";
    scaled_config.pdd()
        .clusters.decode.execution_model.analytical.moe_layer_event_mode =
        "first_layer_scaled";
    const auto workload = parse_workload_csv(
        "session_start_at,think_time,num_prefill_tokens,num_decode_tokens\n"
        "0,0,8,2\n");

    const auto detailed = run_simulation(detailed_config, workload);
    const auto scaled = run_simulation(scaled_config, workload);
    expect(detailed.batch_stages.size() == scaled.batch_stages.size(),
           "scaled PDD run must preserve the batch-stage count");
    for (std::size_t index = 0; index < detailed.batch_stages.size(); ++index) {
        const double detailed_ms =
            (detailed.batch_stages.at(index).completed_at.seconds() -
             detailed.batch_stages.at(index).started_at.seconds()) *
            1e3;
        const double scaled_ms =
            (scaled.batch_stages.at(index).completed_at.seconds() -
             scaled.batch_stages.at(index).started_at.seconds()) *
            1e3;
        expect(std::abs(detailed_ms - scaled_ms) < 1e-9,
               "scaled PDD stage duration mismatch at index " +
                   std::to_string(index) + ": " + std::to_string(detailed_ms) +
                   " vs " + std::to_string(scaled_ms));
    }
    expect(
        std::abs(detailed.requests.front().completed_at.seconds() -
                 scaled.requests.front().completed_at.seconds()) < 1e-12,
        "scaled PDD MoE layers must preserve request completion time: " +
            std::to_string(detailed.requests.front().completed_at.seconds()) +
            " vs " +
            std::to_string(scaled.requests.front().completed_at.seconds()));
    expect(scaled.aggregate.event_count < detailed.aggregate.event_count &&
               scaled.aggregate.moe_routing_count * 10 <
                   detailed.aggregate.moe_routing_count,
           "scaled PDD MoE layers must materially reduce detailed records");
}

} // namespace

int main() {
    int failures = 0;
    failures += frontier::test::run("local MoE fast path",
                                    test_local_moe_uses_fast_path);
    failures +=
        frontier::test::run("co-location MoE synchronization",
                            test_colocation_runs_all_sync_event_families);
    failures += frontier::test::run("PDD MoE Phi KV contract",
                                    test_pdd_moe_preserves_phi_kv_contract);
    failures += frontier::test::run("co-location DP2/EP2 decode lockstep",
                                    test_colocation_dp2_ep2_decode_lockstep);
    failures += frontier::test::run(
        "co-location DP2/EP2 aggregate expert re-prediction",
        test_colocation_dp2_ep2_repredicts_aggregated_expert_tokens);
    failures += frontier::test::run(
        "PDD DP2 EP2 waits for slowest real attention lane",
        test_pdd_dp2_ep2_waits_for_slowest_real_attention_lane);
    failures += frontier::test::run(
        "PDD DP2 EP2 reserves dummy lane until stage end",
        test_pdd_dp2_ep2_reserves_dummy_lane_until_stage_end);
    failures +=
        frontier::test::run("Kimi K2 dense prefix and uneven PP",
                            test_kimi_k2_dense_prefix_runs_with_uneven_pp);
    failures +=
        frontier::test::run("first-layer-scaled repeats DP attention wait",
                            test_first_layer_scaled_repeats_dp_attention_wait);
    failures += frontier::test::run(
        "Kimi K2 first-layer-scaled completion",
        test_kimi_k2_first_layer_scaled_preserves_completion_time);
    failures += frontier::test::run(
        "Kimi K2 first-layer-scaled PDD completion",
        test_kimi_k2_first_layer_scaled_preserves_pdd_completion_time);
    return failures == 0 ? 0 : 1;
}
