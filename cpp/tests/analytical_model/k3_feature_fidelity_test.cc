// Behavioral characterization of the Kimi K2/K3 analytical features.
//
// These tests do not re-derive the roofline arithmetic; they pin the
// qualitative behavior a reader would expect from a real serving system, so a
// future change that silently inverts one of these relationships fails here.
// Two checks deliberately record a KNOWN DEVIATION from real hardware; they are
// marked as such and must be flipped, not deleted, when the model is corrected.

#include "frontier/config/config.h"
#include "frontier/entities/batch.h"
#include "frontier/entities/request.h"
#include "frontier/execution_time_predictor/analytical_roofline_execution_time_predictor.h"
#include "frontier/request_generator/workload.h"
#include "tests/test_support.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace analytical = frontier::execution_time_predictor::detail;
namespace predictor = frontier::execution_time_predictor;

using frontier::BatchId;
using frontier::ClusterType;
using frontier::DataParallelId;
using frontier::Generation;
using frontier::IterationId;
using frontier::ReplicaId;
using frontier::RequestId;
using frontier::SimTime;
using frontier::StageId;
using frontier::entities::Batch;
using frontier::entities::BatchKind;
using frontier::entities::Request;
using frontier::entities::RequestCollection;
using frontier::entities::RequestBatchSnapshot;
using frontier::request_generator::WorkloadRequest;
using frontier::test::expect;

analytical::DenseModel
k3_dense_model(const frontier::config::ModelConfig &model,
               std::uint64_t tensor_parallel_size) {
    analytical::DenseModel value{};
    value.hidden_size = model.hidden_size;
    value.intermediate_size = model.dense_intermediate_size;
    value.num_query_heads = model.num_query_heads;
    value.num_kv_heads = model.num_kv_heads;
    value.head_dim = model.head_dim;
    value.tensor_parallel_size = tensor_parallel_size;
    value.gated_mlp = model.gated_mlp;
    value.fused_add_norm = model.fused_add_norm;
    value.use_mla = model.use_mla;
    value.mla_use_output_gate = model.mla_use_output_gate;
    value.mla_use_nope = model.mla_use_nope;
    value.q_lora_rank = model.q_lora_rank;
    value.kv_lora_rank = model.kv_lora_rank;
    value.qk_nope_head_dim = model.qk_nope_head_dim;
    value.qk_rope_head_dim = model.qk_rope_head_dim;
    value.qk_head_dim = model.qk_head_dim;
    value.v_head_dim = model.v_head_dim;
    value.kda_num_heads = model.kda_num_heads;
    value.kda_num_k_heads = model.kda_num_k_heads;
    value.kda_num_v_heads = model.kda_num_v_heads;
    value.kda_key_head_dim = model.kda_key_head_dim;
    value.kda_value_head_dim = model.kda_value_head_dim;
    value.kda_head_dim = model.kda_head_dim;
    value.kda_short_conv_kernel_size = model.kda_short_conv_kernel_size;
    value.kda_conv_state_dim = model.kda_conv_state_dim;
    return value;
}

analytical::DenseLayerTimes predict_layer(const analytical::DenseModel &model,
                                          const analytical::DenseBatch &batch) {
    return analytical::predict_dense_layer(
        analytical::DeviceCeilings::rubin(), analytical::AnalyticalConfig{},
        model, batch, analytical::Precision::kFp8);
}

analytical::DenseBatch decode_batch(std::uint64_t requests,
                                    std::uint64_t context) {
    analytical::DenseBatch value{};
    value.total_tokens = requests;
    value.decode_requests.assign(static_cast<std::size_t>(requests),
                                 analytical::AttentionRequestSlice{1, context});
    return value;
}

analytical::DenseBatch prefill_batch(std::uint64_t chunk,
                                     std::uint64_t past_context) {
    analytical::DenseBatch value{};
    value.total_tokens = chunk;
    value.prefill_requests = {{chunk, past_context}};
    return value;
}

// KDA keeps a fixed-size recurrent state, so a decode step must cost the same
// at 1 K and at 512 K of context while MLA grows with the cached latent.
void test_kda_decode_is_context_free_and_mla_is_not() {
    const auto config =
        frontier::config::load_model_config("moonshotai/Kimi-K3");
    auto mla = k3_dense_model(config, 4);
    mla.use_mla = true;
    auto kda = k3_dense_model(config, 4);
    kda.use_mla = false;
    kda.use_kda = true;

    const auto kda_short = predict_layer(kda, decode_batch(64, 1'024));
    const auto kda_long = predict_layer(kda, decode_batch(64, 524'288));
    expect(kda_short.decode_attention_ms == kda_long.decode_attention_ms,
           "KDA decode attention must not depend on context length");
    expect(kda_short.kv_cache_save_ms == 0.0 && kda_short.rope_ms == 0.0,
           "KDA layers must not write a growing KV cache or apply RoPE");

    const auto mla_short = predict_layer(mla, decode_batch(64, 1'024));
    const auto mla_long = predict_layer(mla, decode_batch(64, 524'288));
    expect(mla_long.decode_attention_ms > 100.0 * mla_short.decode_attention_ms,
           "MLA decode attention must grow with the cached latent");

    // The K3 recurrent state is large enough that a short-context decode step
    // costs more on a KDA layer than on an MLA layer.  That crossover is real,
    // so it is pinned here to catch an accidental sign flip.
    expect(kda_short.decode_attention_ms > mla_short.decode_attention_ms,
           "at 1K context the K3 recurrent state outweighs the latent cache");
    expect(kda_long.decode_attention_ms < mla_long.decode_attention_ms,
           "at 512K context KDA must be far cheaper than MLA");
}

// KNOWN DEVIATION, tracked in docs/design/kimi-k3-support.md section 12.1.
// The delta-rule update is charged one full recurrent-state read and write per
// token, which models a strictly sequential scan.  FlashKDA-style kernels
// process a chunk of tokens against a state held in registers and combine only
// the chunk results sequentially, so prefill state traffic here is roughly two
// orders of magnitude too high.  Flip this test when the model gains chunking.
void test_kda_prefill_recurrent_traffic_is_per_token() {
    const auto config =
        frontier::config::load_model_config("moonshotai/Kimi-K3");
    auto kda = k3_dense_model(config, 4);
    kda.use_mla = false;
    kda.use_kda = true;

    const auto small = predict_layer(kda, prefill_batch(512, 0));
    const auto large = predict_layer(kda, prefill_batch(4'096, 0));
    const double growth =
        large.prefill_attention_ms / small.prefill_attention_ms;
    expect(growth > 7.0 && growth < 8.1,
           "KDA prefill recurrent cost currently scales linearly with tokens");

    auto mla = k3_dense_model(config, 4);
    mla.use_mla = true;
    const auto mla_prefill = predict_layer(mla, prefill_batch(4'096, 0));
    // A K3 linear-attention layer should be the cheap one during prefill.  It
    // is not, because of the per-token state traffic above.
    expect(large.total_ms() > 3.0 * mla_prefill.total_ms(),
           "KNOWN DEVIATION: KDA prefill is modeled far above MLA prefill");
}

// DCP is deliberately decode-only: a context-parallel prefill would pay an
// all-gather/reduce-scatter pair among the DCP peers that outweighs the sharded
// read, and sequential PDD runs its prefill cluster at DCP=1 so the two never
// meet.  The residual gap, tracked in docs/design/kimi-k3-support.md section
// 12.2, is co-location with DCP > 1 over a prefix-cache hit: prefill then reads
// latent entries the rank does not own while the write stays sharded.
void test_mla_dcp_shards_decode_but_not_cached_prefill() {
    const auto config =
        frontier::config::load_model_config("moonshotai/Kimi-K3");
    auto model = k3_dense_model(config, 8);
    model.use_mla = true;

    model.decode_context_parallel_size = 1;
    const auto replicated_decode =
        predict_layer(model, decode_batch(64, 131'072));
    model.decode_context_parallel_size = 8;
    const auto sharded_decode = predict_layer(model, decode_batch(64, 131'072));
    expect(sharded_decode.decode_attention_ms <
               0.30 * replicated_decode.decode_attention_ms,
           "DCP must shard the decode latent-cache read");

    model.decode_context_parallel_size = 1;
    const auto replicated_prefill =
        predict_layer(model, prefill_batch(2'048, 131'072));
    model.decode_context_parallel_size = 8;
    const auto sharded_prefill =
        predict_layer(model, prefill_batch(2'048, 131'072));
    expect(sharded_prefill.kv_cache_save_ms <
               replicated_prefill.kv_cache_save_ms,
           "DCP must shard the persistent latent write produced by prefill");
    expect(sharded_prefill.prefill_attention_ms ==
               replicated_prefill.prefill_attention_ms,
           "DCP is decode-only, so cached-context prefill reads stay whole");
}

struct Scenario {
    RequestCollection requests;
    std::optional<Batch> batch;
};

Scenario make_decode_scenario(std::uint64_t count, std::uint64_t context,
                              std::uint64_t pipeline_stages) {
    Scenario scenario{};
    const SimTime now = SimTime::from_seconds(0.0);
    std::vector<RequestBatchSnapshot> snapshots;
    for (std::uint64_t index = 0; index < count; ++index) {
        WorkloadRequest value{};
        value.request_id = RequestId{index};
        value.session_start_at = now;
        value.num_prefill_tokens = context;
        value.num_decode_tokens = 64;
        scenario.requests.emplace_back(value);
        Request &request = scenario.requests.back();
        request.on_arrival(now);
        request.on_admitted(now);
        request.advance_scheduler_frontier(context);
        request.on_batch_completion(now, context, ClusterType::kPrefill);
        request.advance_scheduler_frontier(1);
        RequestBatchSnapshot snapshot{};
        snapshot.request_id = RequestId{index};
        snapshot.scheduled_tokens = 1;
        snapshot.processed_tokens = context;
        // The scheduler advances the optimistic frontier before it snapshots a
        // request, so a decode step's frontier already counts its own token.
        // The analytical model derives past context from this field.
        snapshot.scheduler_frontier = request.scheduler_num_computed_tokens();
        snapshots.push_back(snapshot);
    }
    scenario.batch.emplace(BatchId{0}, IterationId{0}, snapshots, now,
                           Generation{0}, ReplicaId{0}, DataParallelId{0},
                           pipeline_stages, ClusterType::kDecode,
                           BatchKind::kWork, frontier::config::ModelKind::kMoe);
    return scenario;
}

frontier::config::AnalyticalExecutionModelConfig
moe_mode_config(const std::string &mode) {
    frontier::config::AnalyticalExecutionModelConfig value{};
    value.device = "rubin";
    value.precision = "fp8";
    value.moe_layer_event_mode = mode;
    value.network_bandwidth_gbps = 28'800.0;
    value.network_latency_us = 1.0;
    value.intra_node_bandwidth_gbps = 28'800.0;
    return value;
}

frontier::config::ParallelismConfig moe_parallelism(std::uint64_t pipeline,
                                                    std::uint64_t experts) {
    frontier::config::ParallelismConfig value{};
    value.tensor_parallel_size = 1;
    value.pipeline_parallel_size = pipeline;
    value.decode_context_parallel_size = 1;
    value.moe_tensor_parallel_size = 1;
    value.moe_expert_parallel_size = experts;
    value.data_parallel_size = 1;
    return value;
}

double stage_sum(const predictor::AnalyticalRooflineExecutionTimePredictor &p,
                 const Scenario &scenario, std::uint64_t pipeline) {
    double total = 0.0;
    for (std::uint64_t stage = 0; stage < pipeline; ++stage) {
        total += p.predict_stage_execution_time(
                      *scenario.batch, scenario.requests, StageId{stage})
                     .duration_ms;
    }
    return total;
}

// moe_routing.layer_scope decides whether every MoE layer routes the batch to
// the same experts or re-draws per layer.  It is orthogonal to `distribution`,
// which only shapes one draw, so a seed-dependent distribution must honour both
// settings.
void test_routing_layer_scope_controls_expert_reuse() {
    const auto config =
        frontier::config::load_model_config("moonshotai/Kimi-K3");
    constexpr std::uint64_t kPipeline = 24;
    const auto parallelism = moe_parallelism(kPipeline, 32);
    const Scenario scenario = make_decode_scenario(8, 4'096, kPipeline);

    const auto routing_for =
        [](frontier::config::MoeRoutingLayerScope layer_scope) {
            frontier::config::MoeRoutingConfig value{};
            value.mode = frontier::config::MoeRoutingMode::kSimulation;
            // A seed-dependent distribution is the only one that can tell the
            // two scopes apart: balanced ignores the layer seed entirely.
            value.distribution =
                frontier::config::MoeRoutingDistribution::kRandom;
            value.seed = 42;
            value.layer_scope = layer_scope;
            return value;
        };

    // `detailed` still reports one routing record per MoE layer under either
    // scope; only the contents may differ.
    const auto records_for =
        [&](frontier::config::MoeRoutingLayerScope layer_scope) {
            const predictor::AnalyticalRooflineExecutionTimePredictor predictor{
                moe_mode_config("detailed"), parallelism, config,
                routing_for(layer_scope)};
            return predictor
                .predict_stage_execution_time(*scenario.batch,
                                              scenario.requests, StageId{1})
                .moe_routing;
        };

    const auto shared =
        records_for(frontier::config::MoeRoutingLayerScope::kShared);
    const auto per_layer =
        records_for(frontier::config::MoeRoutingLayerScope::kPerLayer);
    expect(shared.size() > 1 && shared.size() == per_layer.size(),
           "both scopes must report one routing record per MoE layer");

    const bool shared_is_uniform =
        std::all_of(shared.begin(), shared.end(), [&](const auto &record) {
            return record.global_expert_tokens ==
                       shared.front().global_expert_tokens &&
                   record.lane_expert_tokens ==
                       shared.front().lane_expert_tokens;
        });
    expect(shared_is_uniform,
           "shared scope must give every layer the same expert assignment");

    const bool per_layer_varies = std::any_of(
        per_layer.begin(), per_layer.end(), [&](const auto &record) {
            return record.global_expert_tokens !=
                   per_layer.front().global_expert_tokens;
        });
    expect(per_layer_varies,
           "per_layer scope must re-draw the assignment for each layer");

    // Sharing is what makes the compressed modes exact, so a distribution that
    // could not be compressed before now can be.
    const predictor::AnalyticalRooflineExecutionTimePredictor shared_detailed{
        moe_mode_config("detailed"), parallelism, config,
        routing_for(frontier::config::MoeRoutingLayerScope::kShared)};
    const predictor::AnalyticalRooflineExecutionTimePredictor shared_group{
        moe_mode_config("stage_group_scaled"), parallelism, config,
        routing_for(frontier::config::MoeRoutingLayerScope::kShared)};
    expect(stage_sum(shared_group, scenario, kPipeline) ==
               stage_sum(shared_detailed, scenario, kPipeline),
           "shared scope must let stage_group_scaled compress a seed-dependent "
           "distribution exactly");
}

// The two compression modes must agree with `detailed` to within the routing
// noise they are allowed to drop, and `stage_group_scaled` must fall back to
// exact per-layer routing whenever the route depends on the layer index.
void test_scaled_moe_modes_track_detailed_prediction() {
    const auto config =
        frontier::config::load_model_config("moonshotai/Kimi-K3");
    constexpr std::uint64_t kPipeline = 24;
    const auto parallelism = moe_parallelism(kPipeline, 32);
    const Scenario scenario = make_decode_scenario(64, 8'192, kPipeline);

    const auto make_routing =
        [](frontier::config::MoeRoutingDistribution distribution,
           frontier::config::MoeRoutingLayerScope layer_scope) {
            frontier::config::MoeRoutingConfig value{};
            value.mode = frontier::config::MoeRoutingMode::kSimulation;
            value.distribution = distribution;
            value.seed = 42;
            value.layer_scope = layer_scope;
            return value;
        };

    // Shared routing is layer invariant, so both compressions must be exact.
    const auto balanced =
        make_routing(frontier::config::MoeRoutingDistribution::kBalanced,
                     frontier::config::MoeRoutingLayerScope::kShared);
    const predictor::AnalyticalRooflineExecutionTimePredictor detailed{
        moe_mode_config("detailed"), parallelism, config, balanced};
    const predictor::AnalyticalRooflineExecutionTimePredictor first_layer{
        moe_mode_config("first_layer_scaled"), parallelism, config, balanced};
    const predictor::AnalyticalRooflineExecutionTimePredictor stage_group{
        moe_mode_config("stage_group_scaled"), parallelism, config, balanced};
    const double reference = stage_sum(detailed, scenario, kPipeline);
    expect(stage_sum(first_layer, scenario, kPipeline) == reference &&
               stage_sum(stage_group, scenario, kPipeline) == reference,
           "layer-invariant routing must compress without changing the result");

    // Per-layer random routing is not layer invariant. stage_group_scaled has
    // to fall back to detailed exactly. first_layer_scaled deliberately
    // overrides the raw scope with one batch-shared representative route.
    const auto random =
        make_routing(frontier::config::MoeRoutingDistribution::kRandom,
                     frontier::config::MoeRoutingLayerScope::kPerLayer);
    const predictor::AnalyticalRooflineExecutionTimePredictor random_detailed{
        moe_mode_config("detailed"), parallelism, config, random};
    const predictor::AnalyticalRooflineExecutionTimePredictor random_first{
        moe_mode_config("first_layer_scaled"), parallelism, config, random};
    const predictor::AnalyticalRooflineExecutionTimePredictor
        random_first_explicit_shared{
            moe_mode_config("first_layer_scaled"), parallelism, config,
            make_routing(frontier::config::MoeRoutingDistribution::kRandom,
                         frontier::config::MoeRoutingLayerScope::kShared)};
    const predictor::AnalyticalRooflineExecutionTimePredictor
        random_shared_detailed{
            moe_mode_config("detailed"), parallelism, config,
            make_routing(frontier::config::MoeRoutingDistribution::kRandom,
                         frontier::config::MoeRoutingLayerScope::kShared)};
    const predictor::AnalyticalRooflineExecutionTimePredictor random_group{
        moe_mode_config("stage_group_scaled"), parallelism, config, random};
    const double random_reference =
        stage_sum(random_detailed, scenario, kPipeline);
    expect(stage_sum(random_group, scenario, kPipeline) == random_reference,
           "stage_group_scaled must fall back to detailed layer-dependent "
           "routing");
    const double random_first_total =
        stage_sum(random_first, scenario, kPipeline);
    expect(random_first_total ==
               stage_sum(random_first_explicit_shared, scenario, kPipeline),
           "first_layer_scaled must always use batch-shared routing");
    expect(random_first_total ==
               stage_sum(random_shared_detailed, scenario, kPipeline),
           "first_layer_scaled must exactly match detailed shared routing");
}

// K3's 3:1 KDA/MLA interleave collapses 24 physical stages onto a handful of
// distinct static signatures.  That dedup is the point of the timing cache.
void test_pipeline_stage_signatures_deduplicate() {
    const auto config =
        frontier::config::load_model_config("moonshotai/Kimi-K3");
    constexpr std::uint64_t kPipeline = 24;
    const Scenario scenario = make_decode_scenario(8, 4'096, kPipeline);
    const predictor::AnalyticalRooflineExecutionTimePredictor stage_group{
        moe_mode_config("stage_group_scaled"), moe_parallelism(kPipeline, 32),
        config, frontier::config::MoeRoutingConfig{}};

    double unique_templates = 0.0;
    double active = 0.0;
    for (std::uint64_t stage = 0; stage < kPipeline; ++stage) {
        const auto prediction = stage_group.predict_stage_execution_time(
            *scenario.batch, scenario.requests, StageId{stage});
        for (const auto &[key, value] : prediction.diagnostics) {
            if (key == "timing_cache_unique_templates_total") {
                unique_templates = value;
            }
            if (key == "stage_group_active") {
                active = value;
            }
        }
    }
    expect(active == 1.0, "stage_group_scaled must engage for K3");
    expect(unique_templates >= 1.0 && unique_templates < kPipeline,
           "K3 must cache only reusable groups and fewer templates than "
           "physical stages");
}

double diagnostic_value(const predictor::ExecutionTimePrediction &prediction,
                        const std::string &key) {
    const auto position = std::find_if(
        prediction.diagnostics.begin(), prediction.diagnostics.end(),
        [&](const auto &entry) { return entry.first == key; });
    expect(position != prediction.diagnostics.end(),
           "missing analytical diagnostic: " + key);
    return position->second;
}

void test_compact_predictions_skip_diagnostics_and_reuse_complete_result() {
    const auto model =
        frontier::config::load_model_config("moonshotai/Kimi-K3");
    constexpr std::uint64_t kPipeline = 24;
    const auto parallelism = moe_parallelism(kPipeline, 32);
    const Scenario scenario = make_decode_scenario(8, 4'096, kPipeline);

    std::optional<std::pair<std::uint64_t, std::uint64_t>> equivalent_stages;
    for (std::uint64_t lhs = 0; lhs < kPipeline && !equivalent_stages; ++lhs) {
        const auto lhs_signature =
            frontier::config::build_pipeline_stage_timing_signature(
                model, parallelism, lhs);
        for (std::uint64_t rhs = lhs + 1; rhs < kPipeline; ++rhs) {
            if (lhs_signature ==
                frontier::config::build_pipeline_stage_timing_signature(
                    model, parallelism, rhs)) {
                equivalent_stages = std::pair{lhs, rhs};
                break;
            }
        }
    }
    expect(equivalent_stages.has_value(),
           "K3 PP24 must contain a reusable timing group");

    predictor::AnalyticalRooflineExecutionTimePredictor compact{
        moe_mode_config("stage_group_scaled"), parallelism, model,
        frontier::config::MoeRoutingConfig{}};
    compact.set_detailed_diagnostics_enabled(false);
    const auto first = compact.predict_stage_execution_time(
        *scenario.batch, scenario.requests, StageId{equivalent_stages->first});
    const auto cached = compact.predict_stage_execution_time(
        *scenario.batch, scenario.requests, StageId{equivalent_stages->second});

    expect(first.diagnostics.empty() && cached.diagnostics.empty(),
           "compact predictions must not construct detailed diagnostics");
    expect(first.execution_time == cached.execution_time &&
               first.duration_ms == cached.duration_ms,
           "a complete prediction cache hit must preserve execution time");
    expect(!cached.moe_routing.empty(),
           "disabling diagnostics must preserve scheduler routing data");
    const auto cached_layers = frontier::config::pipeline_stage_layer_range(
        model.num_layers, kPipeline, equivalent_stages->second);
    for (const auto &routing : cached.moe_routing) {
        expect(routing.model_layer_id >= cached_layers.begin &&
                   routing.model_layer_id < cached_layers.end &&
                   model.is_moe_layer(routing.model_layer_id),
               "cached routing must remap to the consuming stage's model "
               "layers");
    }
}

void test_singleton_timing_group_bypasses_cache() {
    const auto model =
        frontier::config::load_model_config("moonshotai/Kimi-K3");
    constexpr std::uint64_t kPipeline = 1;
    const Scenario scenario = make_decode_scenario(8, 4'096, kPipeline);
    const predictor::AnalyticalRooflineExecutionTimePredictor predictor{
        moe_mode_config("stage_group_scaled"), moe_parallelism(kPipeline, 32),
        model, frontier::config::MoeRoutingConfig{}};
    const auto prediction = predictor.predict_stage_execution_time(
        *scenario.batch, scenario.requests, StageId{0});

    expect(diagnostic_value(prediction, "timing_group_multiplicity") == 1.0,
           "PP1 must produce a singleton timing group");
    expect(diagnostic_value(prediction, "timing_cache_enabled") == 0.0 &&
               diagnostic_value(prediction, "timing_cache_entries") == 0.0,
           "singleton timing groups must bypass the batch timing cache");
}

} // namespace

int main() {
    int failures = 0;
    failures +=
        frontier::test::run("kda_decode_is_context_free",
                            test_kda_decode_is_context_free_and_mla_is_not);
    failures +=
        frontier::test::run("kda_prefill_recurrent_traffic_per_token",
                            test_kda_prefill_recurrent_traffic_is_per_token);
    failures +=
        frontier::test::run("mla_dcp_prefill_versus_decode",
                            test_mla_dcp_shards_decode_but_not_cached_prefill);
    failures +=
        frontier::test::run("routing_layer_scope_controls_expert_reuse",
                            test_routing_layer_scope_controls_expert_reuse);
    failures +=
        frontier::test::run("scaled_moe_modes_track_detailed",
                            test_scaled_moe_modes_track_detailed_prediction);
    failures += frontier::test::run("pipeline_stage_signature_dedup",
                                    test_pipeline_stage_signatures_deduplicate);
    failures += frontier::test::run(
        "compact_prediction_cache",
        test_compact_predictions_skip_diagnostics_and_reuse_complete_result);
    failures += frontier::test::run("singleton_timing_group_bypass",
                                    test_singleton_timing_group_bypasses_cache);
    return failures == 0 ? 0 : 1;
}
