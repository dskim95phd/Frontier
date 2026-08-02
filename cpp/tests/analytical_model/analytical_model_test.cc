#include "frontier/attention/mla.h"
#include "frontier/cc_backend/analytical_model.h"
#include "frontier/entities/batch.h"
#include "frontier/entities/request.h"
#include "frontier/execution_time_predictor/analytical_roofline_execution_time_predictor.h"
#include "frontier/kv_cache_transfer/analytical_transfer.h"
#include "frontier/request_generator/workload.h"
#include "tests/test_support.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#ifndef FRONTIER_TEST_FIXTURE_DIR
#error "FRONTIER_TEST_FIXTURE_DIR must be defined for analytical tests"
#endif

namespace {

namespace analytical = frontier::execution_time_predictor::detail;
namespace attention = frontier::attention;
namespace predictor = frontier::execution_time_predictor;
namespace communication = frontier::cc_backend;
namespace kv_transfer = frontier::kv_cache_transfer;

using frontier::BatchId;
using frontier::Generation;
using frontier::IterationId;
using frontier::RequestId;
using frontier::SimTime;
using frontier::entities::Batch;
using frontier::entities::Request;
using frontier::entities::RequestBatchSnapshot;
using frontier::request_generator::WorkloadRequest;
using frontier::test::expect;
using frontier::test::expect_throws;
using frontier::test::read_text_file;
using Json = nlohmann::json;

bool approximately_equal(double actual, double expected,
                         double relative_tolerance = 1e-12,
                         double absolute_tolerance = 1e-12) {
    const double difference = std::abs(actual - expected);
    const double scale = std::max(std::abs(actual), std::abs(expected));
    return difference <=
           std::max(absolute_tolerance, relative_tolerance * scale);
}

void expect_approximately_equal(double actual, double expected,
                                std::string_view field) {
    if (!approximately_equal(actual, expected)) {
        throw std::runtime_error(std::string{field} +
                                 " mismatch: actual=" + std::to_string(actual) +
                                 ", expected=" + std::to_string(expected));
    }
}

Json load_golden() {
    const std::filesystem::path path =
        std::filesystem::path{FRONTIER_TEST_FIXTURE_DIR} /
        "analytical/analytical_v1.json";
    return Json::parse(read_text_file(path));
}

Json load_batch_golden() {
    const std::filesystem::path path =
        std::filesystem::path{FRONTIER_TEST_FIXTURE_DIR} /
        "analytical/analytical_batch_v1.json";
    return Json::parse(read_text_file(path));
}

Json load_attention_family_golden() {
    const std::filesystem::path path =
        std::filesystem::path{FRONTIER_TEST_FIXTURE_DIR} /
        "analytical/analytical_attention_families_v1.json";
    return Json::parse(read_text_file(path));
}

void test_roofline_matches_python_golden() {
    const Json golden = load_golden();
    expect(golden.at("schema_version").get<int>() == 1,
           "analytical golden schema must be version 1");
    const Json &input = golden.at("roofline").at("input");
    const Json &expected = golden.at("roofline").at("expected");

    const analytical::RooflineResult result = analytical::predict_roofline(
        analytical::DeviceCeilings::rubin(), analytical::Precision::kFp16,
        [&]() {
            analytical::KernelWork value{};
            value.flops = input.at("flops").get<double>();
            value.hbm_bytes = input.at("hbm_bytes").get<double>();
            return value;
        }(),
        [&]() {
            analytical::Efficiency value{};
            value.compute = input.at("compute_efficiency").get<double>();
            value.memory = input.at("memory_efficiency").get<double>();
            value.overlap_penalty = input.at("overlap_penalty").get<double>();
            return value;
        }(),
        input.at("kernel_launch_latency_us").get<double>());

    expect_approximately_equal(result.compute_time_ms,
                               expected.at("compute_time_ms").get<double>(),
                               "roofline.compute_time_ms");
    expect_approximately_equal(result.memory_time_ms,
                               expected.at("memory_time_ms").get<double>(),
                               "roofline.memory_time_ms");
    expect_approximately_equal(result.launch_time_ms,
                               expected.at("launch_time_ms").get<double>(),
                               "roofline.launch_time_ms");
    expect_approximately_equal(result.predicted_time_ms,
                               expected.at("predicted_time_ms").get<double>(),
                               "roofline.predicted_time_ms");
    expect(result.bottleneck == analytical::Bottleneck::kHbm,
           "roofline bottleneck must match Python");
}

void test_device_presets_and_overrides() {
    frontier::config::AnalyticalExecutionModelConfig config{};
    config.device = "gb300";
    const analytical::DeviceCeilings gb300 =
        analytical::DeviceCeilings::from_config(config);
    expect_approximately_equal(gb300.hbm_bandwidth_tbps, 8.0,
                               "GB300 HBM bandwidth");
    expect_approximately_equal(gb300.fp32_tflops, 83.33333333333333,
                               "GB300 FP32 ceiling");
    expect_approximately_equal(gb300.fp16_tflops, 2'500.0,
                               "GB300 FP16 ceiling");
    expect_approximately_equal(gb300.fp8_tflops, 5'000.0, "GB300 FP8 ceiling");
    expect_approximately_equal(gb300.fp4_tflops, 15'000.0, "GB300 FP4 ceiling");

    config.device_overrides.hbm_bandwidth_tbps = 7.25;
    config.device_overrides.fp8_tflops = 4'500.0;
    const analytical::DeviceCeilings overridden =
        analytical::DeviceCeilings::from_config(config);
    expect_approximately_equal(overridden.hbm_bandwidth_tbps, 7.25,
                               "overridden HBM bandwidth");
    expect_approximately_equal(overridden.fp8_tflops, 4'500.0,
                               "overridden FP8 ceiling");
    expect_approximately_equal(overridden.fp16_tflops, gb300.fp16_tflops,
                               "non-overridden GB300 ceiling");

    config.device = "custom";
    expect_throws<analytical::AnalyticalModelError>(
        [&config] {
            static_cast<void>(analytical::DeviceCeilings::from_config(config));
        },
        "incomplete custom device ceilings must fail fast");

    config.device_overrides.fp32_tflops = 75.0;
    config.device_overrides.fp16_tflops = 2'100.0;
    config.device_overrides.fp4_tflops = 11'000.0;
    const analytical::DeviceCeilings custom =
        analytical::DeviceCeilings::from_config(config);
    expect_approximately_equal(custom.hbm_bandwidth_tbps, 7.25,
                               "custom HBM bandwidth");
    expect_approximately_equal(custom.fp4_tflops, 11'000.0,
                               "custom FP4 ceiling");
}

void check_dense_fields(const analytical::DenseLayerTimes &actual,
                        const Json &expected) {
    for (const auto &[name, value] : {
             std::pair{"attention_pre_projection_ms",
                       actual.attention_pre_projection_ms},
             std::pair{"attention_post_projection_ms",
                       actual.attention_post_projection_ms},
             std::pair{"rope_ms", actual.rope_ms},
             std::pair{"kv_cache_save_ms", actual.kv_cache_save_ms},
             std::pair{"attention_norm_ms", actual.attention_norm_ms},
             std::pair{"prefill_attention_ms", actual.prefill_attention_ms},
             std::pair{"decode_attention_ms", actual.decode_attention_ms},
             std::pair{"mlp_up_projection_ms", actual.mlp_up_projection_ms},
             std::pair{"mlp_activation_ms", actual.mlp_activation_ms},
             std::pair{"mlp_down_projection_ms", actual.mlp_down_projection_ms},
             std::pair{"mlp_norm_ms", actual.mlp_norm_ms},
             std::pair{"residual_add_ms", actual.residual_add_ms},
         }) {
        expect_approximately_equal(value, expected.at(name).get<double>(),
                                   name);
    }
}

void check_attention_fields(const analytical::DenseLayerTimes &actual,
                            const Json &expected) {
    for (const auto &[name, value] : {
             std::pair{"attention_pre_projection_ms",
                       actual.attention_pre_projection_ms},
             std::pair{"attention_post_projection_ms",
                       actual.attention_post_projection_ms},
             std::pair{"rope_ms", actual.rope_ms},
             std::pair{"kv_cache_save_ms", actual.kv_cache_save_ms},
             std::pair{"attention_norm_ms", actual.attention_norm_ms},
             std::pair{"attention_inter_norm_ms",
                       actual.attention_inter_norm_ms},
             std::pair{"attention_wq_projection_ms",
                       actual.attention_wq_projection_ms},
             std::pair{"prefill_attention_ms", actual.prefill_attention_ms},
             std::pair{"decode_attention_ms", actual.decode_attention_ms},
         }) {
        expect_approximately_equal(value, expected.at(name).get<double>(),
                                   name);
    }
}

void test_dense_layer_matches_python_golden() {
    const Json golden = load_golden();
    const analytical::DeviceCeilings device =
        analytical::DeviceCeilings::rubin();
    const analytical::AnalyticalConfig config;
    const analytical::DenseModel model =
        analytical::DenseModel::llama2_7b_tp8();

    for (const Json &test_case : golden.at("dense_cases")) {
        const Json &input = test_case.at("input");
        const std::uint64_t tokens =
            input.at("scheduled_tokens").get<std::uint64_t>();
        const analytical::AttentionRequestSlice request = [&]() {
            analytical::AttentionRequestSlice value{};
            value.query_tokens = tokens;
            value.past_context = input.at("past_context").get<std::uint64_t>();
            return value;
        }();
        analytical::DenseBatch batch = [&]() {
            analytical::DenseBatch value{};
            value.total_tokens = tokens;
            value.prefill_requests = {};
            value.decode_requests = {};
            return value;
        }();
        if (input.at("prefill").get<bool>()) {
            batch.prefill_requests.push_back(request);
        } else {
            batch.decode_requests.push_back(request);
        }

        const analytical::DenseLayerTimes result =
            analytical::predict_dense_layer(device, config, model, batch,
                                            analytical::Precision::kFp16);
        check_dense_fields(result, test_case.at("expected"));
        expect(result.total_ms() > 0.0, "dense layer total must be positive");
    }
}

void test_long_context_decode_cost_increases() {
    const analytical::DeviceCeilings device =
        analytical::DeviceCeilings::rubin();
    const analytical::AnalyticalConfig config;
    const analytical::DenseModel model =
        analytical::DenseModel::llama2_7b_tp8();
    const auto predict = [&](std::uint64_t past_context) {
        return analytical::predict_dense_layer(
            device, config, model,
            [&]() {
                analytical::DenseBatch value{};
                value.total_tokens = 1;
                value.prefill_requests = {};
                value.decode_requests = {
                    [&]() {
                        analytical::AttentionRequestSlice value{};
                        value.query_tokens = 1;
                        value.past_context = past_context;
                        return value;
                    }(),
                };
                return value;
            }(),
            analytical::Precision::kFp16);
    };
    expect(predict(4'096).decode_attention_ms >
               predict(128).decode_attention_ms,
           "long-context decode attention must cost more");
}

void test_operator_precisions_split_dense_and_kv_costs() {
    expect(
        analytical::precision_from_string("fp8") ==
                analytical::Precision::kFp8 &&
            analytical::precision_from_string("fp4") ==
                analytical::Precision::kFp4 &&
            analytical::bytes_per_element(analytical::Precision::kFp8) == 1.0 &&
            analytical::bytes_per_element(analytical::Precision::kFp4) == 0.5,
        "FP8 and FP4 precision names must map to packed byte sizes");

    analytical::DenseBatch batch{};
    batch.total_tokens = 1;
    batch.decode_requests = {{1, 8'192}};
    const auto predict = [&](analytical::DenseOperatorPrecisions precisions) {
        return analytical::predict_dense_layer(
            analytical::DeviceCeilings::rubin(), analytical::AnalyticalConfig{},
            analytical::DenseModel::llama2_7b_tp8(), batch, precisions);
    };
    const auto fp16 =
        predict({analytical::Precision::kFp16, analytical::Precision::kFp16,
                 analytical::Precision::kFp16});
    const auto fp8_kv =
        predict({analytical::Precision::kFp8, analytical::Precision::kFp16,
                 analytical::Precision::kFp8});
    expect(fp8_kv.decode_attention_ms < fp16.decode_attention_ms &&
               fp8_kv.kv_cache_save_ms < fp16.kv_cache_save_ms,
           "FP8 attention/KV must reduce long-context reads and KV writes");
    expect_approximately_equal(fp8_kv.mlp_up_projection_ms,
                               fp16.mlp_up_projection_ms,
                               "dense precision independence");

    const auto fp4_dense =
        predict({analytical::Precision::kFp16, analytical::Precision::kFp4,
                 analytical::Precision::kFp16});
    expect(fp4_dense.mlp_up_projection_ms < fp16.mlp_up_projection_ms,
           "FP4 dense override must reduce MLP projection time");
    expect_approximately_equal(fp4_dense.attention_pre_projection_ms,
                               fp16.attention_pre_projection_ms,
                               "attention precision independence");

    const auto fp4_weight_fp16_activation =
        predict({analytical::Precision::kFp16, analytical::Precision::kFp16,
                 analytical::Precision::kFp16, analytical::Precision::kFp4,
                 analytical::Precision::kFp16, analytical::Precision::kFp4,
                 analytical::Precision::kFp16});
    expect(fp4_weight_fp16_activation.attention_pre_projection_ms <
                   fp16.attention_pre_projection_ms &&
               fp4_weight_fp16_activation.mlp_up_projection_ms <
                   fp16.mlp_up_projection_ms,
           "W4A16 must reduce weight traffic and use the weight compute "
           "ceiling without changing activation bytes");
}

void test_operator_precisions_split_moe_expert_and_router_costs() {
    analytical::MoEModel model{};
    model.hidden_size = 4'096;
    model.intermediate_size = 14'336;
    model.model_num_experts = 8;
    model.moe_tensor_parallel_size = 1;
    const std::vector<std::uint64_t> expert_tokens = {128, 128, 128, 128,
                                                      128, 128, 128, 128};
    const auto predict = [&](analytical::MoEOperatorPrecisions precisions) {
        return analytical::predict_moe_layer(
            analytical::DeviceCeilings::rubin(), analytical::AnalyticalConfig{},
            model, 512, 2, expert_tokens, precisions);
    };
    const auto fp16 =
        predict({analytical::Precision::kFp16, analytical::Precision::kFp16,
                 analytical::Precision::kFp16});
    const auto fp4_expert =
        predict({analytical::Precision::kFp4, analytical::Precision::kFp16,
                 analytical::Precision::kFp16});
    expect(fp4_expert.grouped_up_projection_ms <
                   fp16.grouped_up_projection_ms &&
               fp4_expert.grouped_down_projection_ms <
                   fp16.grouped_down_projection_ms,
           "FP4 expert override must reduce grouped expert GEMM time");
    expect_approximately_equal(fp4_expert.gating_linear_ms,
                               fp16.gating_linear_ms,
                               "router precision independence");

    const auto fp8_router =
        predict({analytical::Precision::kFp16, analytical::Precision::kFp8,
                 analytical::Precision::kFp16});
    expect(fp8_router.gating_linear_ms < fp16.gating_linear_ms &&
               fp8_router.gating_routing_topk_ms < fp16.gating_routing_topk_ms,
           "FP8 router override must reduce router operator time");
    expect_approximately_equal(fp8_router.grouped_up_projection_ms,
                               fp16.grouped_up_projection_ms,
                               "expert precision independence");

    const auto fp4_weight_fp8_activation =
        predict({analytical::Precision::kFp16, analytical::Precision::kFp16,
                 analytical::Precision::kFp16, analytical::Precision::kFp4,
                 analytical::Precision::kFp8, analytical::Precision::kFp16,
                 analytical::Precision::kFp16, analytical::Precision::kFp16,
                 analytical::Precision::kFp16});
    expect(fp4_weight_fp8_activation.grouped_up_projection_ms <
               fp16.grouped_up_projection_ms,
           "MoE W4A8 must model expert weight and activation dtypes "
           "independently");
}

void test_mla_uses_latent_cache_context_costs() {
    analytical::DenseModel model{};
    model.hidden_size = 7'168;
    model.intermediate_size = 18'432;
    model.num_query_heads = 64;
    model.num_kv_heads = 64;
    model.head_dim = 128;
    model.tensor_parallel_size = 4;
    model.gated_mlp = true;
    model.fused_add_norm = true;
    model.use_mla = true;
    model.q_lora_rank = 1'536;
    model.kv_lora_rank = 512;
    model.qk_nope_head_dim = 128;
    model.qk_rope_head_dim = 64;
    model.qk_head_dim = 192;
    model.v_head_dim = 128;

    const auto predict = [&](std::uint64_t past_context) {
        analytical::DenseBatch batch{};
        batch.total_tokens = 1;
        batch.decode_requests = {{1, past_context}};
        return analytical::predict_dense_layer(
            analytical::DeviceCeilings::rubin(), analytical::AnalyticalConfig{},
            model, batch, analytical::Precision::kFp16);
    };

    const auto short_context = predict(128);
    const auto long_context = predict(8'192);
    expect(long_context.decode_attention_ms > short_context.decode_attention_ms,
           "MLA decode must scale with latent-cache context length");
    expect(short_context.attention_pre_projection_ms > 0.0 &&
               short_context.attention_post_projection_ms > 0.0,
           "MLA projections must be included in analytical timing");
    expect(short_context.attention_inter_norm_ms > 0.0 &&
               short_context.attention_wq_projection_ms == 0.0,
           "MLA must include Q/KV LoRA norms without using MFA shared-Q "
           "operators");

    const analytical::KernelWork unabsorbed =
        analytical::mla_unabsorbed_attention_work({{1, 8'192}}, 16, 128, 64,
                                                  128, 2.0, 2.0);
    const analytical::KernelWork absorbed =
        analytical::mla_absorbed_attention_work({{1, 8'192}}, 16, 512, 64, 2.0,
                                                1.0, 2.0);
    expect(absorbed.flops > unabsorbed.flops &&
               absorbed.hbm_bytes < unabsorbed.hbm_bytes,
           "absorbed MLA decode must trade more attention FLOPs for less HBM "
           "traffic");

    const auto predict_prefill = [&](std::uint64_t past_context) {
        analytical::DenseBatch batch{};
        batch.total_tokens = 1;
        batch.prefill_requests = {{1, past_context}};
        return analytical::predict_dense_layer(
            analytical::DeviceCeilings::rubin(), analytical::AnalyticalConfig{},
            model, batch,
            analytical::DenseOperatorPrecisions{
                analytical::Precision::kFp8,
                analytical::Precision::kBf16,
                analytical::Precision::kFp8,
                analytical::Precision::kFp8,
                analytical::Precision::kBf16,
                analytical::Precision::kFp8,
                analytical::Precision::kBf16,
            });
    };
    const auto short_prefill = predict_prefill(0);
    const auto long_prefill = predict_prefill(8'192);
    expect(long_prefill.attention_pre_projection_ms >
               short_prefill.attention_pre_projection_ms,
           "unabsorbed MLA prefill must expand cached latent KV");
}

void test_mla_kv_layout_splits_latent_and_rope() {
    const attention::MlaKvCacheLayout layout{
        512,
        64,
        1.0,
        2.0,
    };
    expect_approximately_equal(attention::mla_kv_cache_bytes_per_token(layout),
                               640.0, "MLA KV bytes per token");
    expect(attention::mla_kv_cache_size_bytes(10, 61, layout) == 390'400,
           "MLA KV size must use FP8 latent and BF16 RoPE components");

    const auto kimi =
        frontier::config::load_model_config("moonshotai/Kimi-K2-Instruct");
    expect(kv_transfer::model_kv_cache_size_bytes(10, kimi, 1.0) == 390'400,
           "Kimi PDD transfer must use the shared MLA KV layout");
}

void test_mfa_models_shared_q_projection_path() {
    const Json golden = load_attention_family_golden();
    analytical::DenseModel model{};
    model.hidden_size = 7'168;
    model.intermediate_size = 18'432;
    model.num_query_heads = 64;
    model.num_kv_heads = 1;
    model.head_dim = 256;
    model.tensor_parallel_size = 4;
    model.gated_mlp = true;
    model.fused_add_norm = true;
    model.use_mfa = true;
    model.share_q_dim = 2'048;

    analytical::DenseBatch batch{};
    batch.total_tokens = 32;
    batch.prefill_requests = {{32, 256}};
    const auto result = analytical::predict_dense_layer(
        analytical::DeviceCeilings::rubin(), analytical::AnalyticalConfig{},
        model, batch, analytical::Precision::kFp16);

    const auto expected_inter_norm = analytical::predict_roofline(
        analytical::DeviceCeilings::rubin(), analytical::Precision::kFp16,
        analytical::streaming_work(65'536.0, 65'536.0, 327'680.0, 2.0),
        analytical::AnalyticalConfig{}.streaming,
        analytical::AnalyticalConfig{}.kernel_launch_latency_us);
    expect_approximately_equal(result.attention_inter_norm_ms,
                               expected_inter_norm.predicted_time_ms,
                               "MFA inter_norm");
    expect(result.attention_wq_projection_ms > 0.0,
           "MFA WQ projection must be timed separately");
    expect(result.prefill_attention_ms > 0.0,
           "MFA must retain dense-KV attention context timing");
    check_attention_fields(result, golden.at("mfa_prefill_32_past_256"));
}

double diagnostic_value(const predictor::ExecutionTimePrediction &prediction,
                        std::string_view name) {
    const auto iterator = std::find_if(
        prediction.diagnostics.begin(), prediction.diagnostics.end(),
        [name](const auto &item) { return item.first == name; });
    if (iterator == prediction.diagnostics.end()) {
        throw std::runtime_error("missing analytical batch diagnostic: " +
                                 std::string{name});
    }
    return iterator->second;
}

void test_batch_model_matches_python_golden() {
    const Json golden = load_batch_golden();
    expect(golden.at("schema_version").get<int>() == 1,
           "analytical batch golden schema must be version 1");
    const frontier::execution_time_predictor::
        AnalyticalRooflineExecutionTimePredictor model{
            frontier::config::AnalyticalExecutionModelConfig{}};
    frontier::config::AnalyticalExecutionModelConfig gb300_config{};
    gb300_config.device = "gb300";
    const frontier::execution_time_predictor::
        AnalyticalRooflineExecutionTimePredictor gb300_model{gb300_config};
    bool observed_device_effect = false;

    for (const Json &test_case : golden.at("cases")) {
        std::vector<Request> requests;
        std::vector<RequestBatchSnapshot> snapshots;
        std::uint64_t request_index = 0;
        for (const Json &slice : test_case.at("input").at("slices")) {
            const std::string phase = slice.at("phase").get<std::string>();
            const std::uint64_t past_context =
                slice.at("past_context").get<std::uint64_t>();
            const std::uint64_t scheduled_tokens =
                slice.at("scheduled_tokens").get<std::uint64_t>();
            const std::uint64_t prefill_tokens =
                phase == "decode" ? past_context
                                  : past_context + scheduled_tokens + 1;
            requests.emplace_back([&]() {
                WorkloadRequest value{};
                value.request_id = RequestId{request_index};
                value.arrived_at = SimTime::from_seconds(0.0);
                value.num_prefill_tokens = prefill_tokens;
                value.num_decode_tokens = 2;
                value.session_id = frontier::SessionId{};
                value.session_turn_index = std::nullopt;
                return value;
            }());
            Request &request = requests.back();
            const SimTime now = SimTime::from_seconds(0.0);
            request.on_arrival(now);
            request.on_admitted(now);
            if (past_context > 0) {
                request.advance_scheduler_frontier(past_context);
                request.on_batch_completion(
                    now, past_context,
                    phase == "decode" ? frontier::ClusterType::kPrefill
                                      : frontier::ClusterType::kMonolithic);
            }
            request.advance_scheduler_frontier(scheduled_tokens);
            snapshots.push_back([&]() {
                RequestBatchSnapshot value{};
                value.request_id = RequestId{request_index};
                value.scheduled_tokens = scheduled_tokens;
                value.runtime_epoch = 0;
                value.execution_epoch = 0;
                value.processed_tokens = past_context;
                value.scheduler_frontier = past_context + scheduled_tokens;
                return value;
            }());
            ++request_index;
        }
        const Batch batch{
            BatchId{0},           IterationId{0},
            std::move(snapshots), SimTime::from_seconds(0.0),
            Generation{0},
        };
        const predictor::ExecutionTimePrediction prediction =
            model.predict_stage_execution_time(batch, requests,
                                               frontier::StageId{0});
        const predictor::ExecutionTimePrediction gb300_prediction =
            gb300_model.predict_stage_execution_time(batch, requests,
                                                     frontier::StageId{0});
        observed_device_effect =
            observed_device_effect ||
            gb300_prediction.duration_ms > prediction.duration_ms;
        const Json &expected = test_case.at("expected");
        expect_approximately_equal(
            prediction.duration_ms - prediction.execution_time.lm_head_ms,
            expected.at("batch_duration_ms").get<double>(),
            "batch_duration_ms_without_lm_head");
        if (diagnostic_value(prediction, "lm_head_tokens") > 0.0) {
            expect(prediction.execution_time.lm_head_ms > 0.0,
                   "final pipeline stage must model the LM head");
        }
        for (const std::string_view field : {
                 "total_tokens",
                 "prefill_request_count",
                 "decode_request_count",
                 "dense_layer_compute_ms",
                 "tp_allreduce_ms",
                 "dense_layer_total_ms",
                 "num_layers",
             }) {
            expect_approximately_equal(diagnostic_value(prediction, field),
                                       expected.at(field).get<double>(), field);
        }
        expect_approximately_equal(
            diagnostic_value(prediction, "batch_duration_ms") -
                prediction.execution_time.lm_head_ms,
            expected.at("batch_duration_ms").get<double>(),
            "diagnostic_batch_duration_ms_without_lm_head");
    }
    expect(observed_device_effect,
           "selected device preset must affect predictor stage duration");
}

void test_communication_matches_python_golden() {
    const Json section = load_golden().at("communication");
    const Json &input = section.at("input");
    const Json &expected = section.at("expected");
    const communication::AnalyticalCommunicationModel model{[&]() {
        communication::AnalyticalCommunicationConfig value{};
        value.network_bandwidth_gbps =
            input.at("network_bandwidth_gbps").get<double>();
        value.latency_us = input.at("latency_us").get<double>();
        value.intra_node_bandwidth_gbps =
            input.at("intra_node_bandwidth_gbps").get<double>();
        return value;
    }()};
    const std::uint64_t bytes =
        input.at("data_size_bytes").get<std::uint64_t>();
    const std::uint64_t devices = input.at("num_devices").get<std::uint64_t>();

    for (const auto &[field, actual] : {
             std::pair{"point_to_point_ms", model.point_to_point_ms(bytes)},
             std::pair{"allreduce_ms", model.allreduce_ms(bytes, devices)},
             std::pair{"allgather_ms", model.allgather_ms(bytes, devices)},
             std::pair{"broadcast_ms", model.broadcast_ms(bytes, devices)},
             std::pair{"reduce_scatter_ms",
                       model.reduce_scatter_ms(bytes, devices)},
             std::pair{"all_to_all_ms", model.all_to_all_ms(bytes, devices)},
         }) {
        expect_approximately_equal(actual, expected.at(field).get<double>(),
                                   field);
    }
    expect(model.allreduce_ms(bytes, 1) == 0.0,
           "single-device collective must be free");
    expect_approximately_equal(model.point_to_point_ms(0),
                               input.at("latency_us").get<double>() / 1e3,
                               "zero-byte point-to-point latency");
    expect_approximately_equal(model.allreduce_ms(0, devices),
                               input.at("latency_us").get<double>() / 1e3,
                               "zero-byte multi-device collective latency");
    expect(devices == 72,
           "golden collective must exercise the NVL72 participant boundary");
}

void test_kv_transfer_matches_python_golden() {
    const Json section = load_golden().at("kv_transfer");
    const Json &input = section.at("input");
    const Json &expected = section.at("expected");
    const kv_transfer::DenseKvLayout layout = [&]() {
        kv_transfer::DenseKvLayout value{};
        value.num_layers = input.at("num_layers").get<std::uint64_t>();
        value.num_kv_heads_per_worker =
            input.at("num_kv_heads_per_worker").get<std::uint64_t>();
        value.head_dim = input.at("head_dim").get<std::uint64_t>();
        value.kv_factor = input.at("kv_factor").get<std::uint64_t>();
        value.dtype_size_bytes = input.at("dtype_size_bytes").get<double>();
        return value;
    }();
    const std::uint64_t size = kv_transfer::dense_kv_cache_size_bytes(
        input.at("num_tokens").get<std::uint64_t>(), layout);
    expect(size == expected.at("size_bytes").get<std::uint64_t>(),
           "dense KV size must match Python");

    const kv_transfer::TransferPrediction prediction =
        kv_transfer::predict_transfer(size, [&]() {
            kv_transfer::TransferConfig value{};
            value.network_bandwidth_gbps =
                input.at("network_bandwidth_gbps").get<double>();
            value.network_latency_ms =
                input.at("network_latency_ms").get<double>();
            return value;
        }());
    expect_approximately_equal(prediction.transfer_time_ms,
                               expected.at("transfer_time_ms").get<double>(),
                               "kv_transfer.transfer_time_ms");

    expect(kv_transfer::dense_kv_cache_size_bytes(0, layout) == 0,
           "zero-token KV size must be zero");
    const kv_transfer::TransferPrediction empty_prediction =
        kv_transfer::predict_transfer(0, [&]() {
            kv_transfer::TransferConfig value{};
            value.network_bandwidth_gbps =
                input.at("network_bandwidth_gbps").get<double>();
            value.network_latency_ms =
                input.at("network_latency_ms").get<double>();
            return value;
        }());
    expect_approximately_equal(empty_prediction.transfer_time_ms,
                               input.at("network_latency_ms").get<double>(),
                               "zero-byte KV transfer latency");
}

void test_kimi_k2_uneven_pipeline_and_lm_head() {
    frontier::config::ParallelismConfig parallelism{};
    parallelism.tensor_parallel_size = 4;
    parallelism.pipeline_parallel_size = 4;
    parallelism.data_parallel_size = 2;
    parallelism.moe_tensor_parallel_size = 1;
    parallelism.moe_expert_parallel_size = 8;

    frontier::config::AnalyticalExecutionModelConfig execution{};
    execution.precision = "fp8";
    execution.operator_precisions.moe_expert_weight = "fp4";
    execution.operator_precisions.moe_expert_activation = "fp8";
    const frontier::config::ModelConfig model_config =
        frontier::config::load_model_config("moonshotai/Kimi-K2-Instruct");
    const predictor::AnalyticalRooflineExecutionTimePredictor model{
        execution, parallelism, model_config,
        frontier::config::MoeRoutingConfig{}};

    std::vector<Request> requests;
    requests.emplace_back([&]() {
        WorkloadRequest value{};
        value.request_id = RequestId{0};
        value.arrived_at = SimTime::from_seconds(0.0);
        value.num_prefill_tokens = 1;
        value.num_decode_tokens = 1;
        return value;
    }());
    requests.front().on_arrival(SimTime::from_seconds(0.0));
    requests.front().on_admitted(SimTime::from_seconds(0.0));
    requests.front().advance_scheduler_frontier(1);
    RequestBatchSnapshot snapshot{};
    snapshot.request_id = RequestId{0};
    snapshot.scheduled_tokens = 1;
    snapshot.processed_tokens = 0;
    snapshot.scheduler_frontier = 1;
    const Batch batch{BatchId{0},
                      IterationId{0},
                      {snapshot},
                      SimTime::from_seconds(0.0),
                      Generation{0}};

    const auto stage0 = model.predict_stage_execution_time(
        batch, requests, frontier::StageId{0});
    const auto stage1 = model.predict_stage_execution_time(
        batch, requests, frontier::StageId{1});
    const auto stage3 = model.predict_stage_execution_time(
        batch, requests, frontier::StageId{3});
    expect(stage0.moe_routing.size() == 15 &&
               stage0.moe_routing.front().model_layer_id == 1 &&
               stage0.moe_routing.back().model_layer_id == 15,
           "Kimi stage 0 must contain dense layer 0 followed by MoE layers "
           "1..15");
    expect(stage1.moe_routing.size() == 15 &&
               stage1.moe_routing.front().model_layer_id == 16 &&
               stage1.moe_routing.back().model_layer_id == 30,
           "Kimi stage 1 must continue at global layer 16");
    expect(stage0.execution_time.lm_head_ms == 0.0 &&
               stage1.execution_time.lm_head_ms == 0.0 &&
               stage3.execution_time.lm_head_ms > 0.0,
           "only the final PP stage must execute the LM-head projection");
}

void test_kimi_k2_first_layer_scaled_prediction() {
    frontier::config::ParallelismConfig parallelism{};
    parallelism.tensor_parallel_size = 4;
    parallelism.pipeline_parallel_size = 4;
    parallelism.data_parallel_size = 4;
    parallelism.moe_tensor_parallel_size = 1;
    parallelism.moe_expert_parallel_size = 16;

    frontier::config::AnalyticalExecutionModelConfig detailed_config{};
    detailed_config.precision = "fp8";
    detailed_config.operator_precisions.moe_expert_weight = "fp4";
    detailed_config.operator_precisions.moe_expert_activation = "fp8";
    auto scaled_config = detailed_config;
    scaled_config.moe_layer_event_mode = "first_layer_scaled";
    const auto model_config =
        frontier::config::load_model_config("moonshotai/Kimi-K2-Instruct");
    const predictor::AnalyticalRooflineExecutionTimePredictor detailed{
        detailed_config, parallelism, model_config,
        frontier::config::MoeRoutingConfig{}};
    const predictor::AnalyticalRooflineExecutionTimePredictor scaled{
        scaled_config, parallelism, model_config,
        frontier::config::MoeRoutingConfig{}};

    std::vector<Request> requests;
    requests.emplace_back([&]() {
        WorkloadRequest value{};
        value.request_id = RequestId{0};
        value.arrived_at = SimTime::from_seconds(0.0);
        value.num_prefill_tokens = 4'096;
        value.num_decode_tokens = 1;
        return value;
    }());
    requests.front().on_arrival(SimTime::from_seconds(0.0));
    requests.front().on_admitted(SimTime::from_seconds(0.0));
    requests.front().advance_scheduler_frontier(4'096);
    RequestBatchSnapshot snapshot{};
    snapshot.request_id = RequestId{0};
    snapshot.scheduled_tokens = 4'096;
    snapshot.processed_tokens = 0;
    snapshot.scheduler_frontier = 4'096;
    const Batch batch{BatchId{0}, IterationId{0}, {snapshot},
                      SimTime::from_seconds(0.0), Generation{0}};

    const auto detailed_stage = detailed.predict_stage_execution_time(
        batch, requests, frontier::StageId{0});
    const auto scaled_stage = scaled.predict_stage_execution_time(
        batch, requests, frontier::StageId{0});
    const auto first_lazy_layer = detailed.prepare_moe_stage_execution(
        batch, requests, frontier::StageId{0});
    expect(detailed_stage.moe_routing.size() == 15 &&
               scaled_stage.moe_routing.size() == 1 &&
               scaled_stage.logical_moe_layer_count == 15 &&
               scaled_stage.repeated_moe_layer_pre_compute_ms > 0.0,
           "scaled Kimi stage must retain one detailed layer for 15 logical "
           "MoE layers");
    expect(detailed_stage.execution_time == scaled_stage.execution_time,
           "scaled prediction must preserve every execution-time component");
    expect(detailed_stage.moe_routing.front().model_layer_id ==
                   scaled_stage.moe_routing.front().model_layer_id &&
               detailed_stage.moe_routing.front().lane_times_ms ==
                   scaled_stage.moe_routing.front().lane_times_ms &&
               detailed_stage.moe_routing.front().pre_moe_compute_ms ==
                   scaled_stage.moe_routing.front().pre_moe_compute_ms,
            "scaled prediction must preserve the first detailed MoE layer");

    expect(first_lazy_layer.lazy_moe_layer_prediction &&
               first_lazy_layer.moe_routing.size() == 1 &&
               first_lazy_layer.logical_moe_layer_count == 15 &&
               first_lazy_layer.moe_routing.front().layer_id.index() == 0 &&
               first_lazy_layer.moe_routing.front().model_layer_id == 1,
           "detailed Kimi execution must prepare only its first MoE layer");
    double lazy_total_ms = first_lazy_layer.execution_time.total_ms();
    for (std::uint64_t layer = 1; layer < 15; ++layer) {
        const auto lazy_layer = detailed.predict_moe_layer_execution(
            batch, requests, frontier::StageId{0}, layer);
        expect(lazy_layer.lazy_moe_layer_prediction &&
                   lazy_layer.moe_routing.size() == 1 &&
                   lazy_layer.logical_moe_layer_count == 15 &&
                   lazy_layer.moe_routing.front().layer_id.index() == layer &&
                   lazy_layer.moe_routing.front().model_layer_id == layer + 1,
               "lazy Kimi prediction must retain the requested layer identity");
        lazy_total_ms += lazy_layer.execution_time.total_ms();
    }
    expect_approximately_equal(lazy_total_ms,
                               detailed_stage.execution_time.total_ms(),
                               "sum of lazy Kimi layer predictions");
}

void test_invalid_analytical_inputs_are_rejected() {
    expect_throws<analytical::AnalyticalModelError>(
        [] {
            static_cast<void>(analytical::predict_roofline(
                analytical::DeviceCeilings::rubin(),
                analytical::Precision::kFp16,
                [&]() {
                    analytical::KernelWork value{};
                    value.flops = -1.0;
                    value.hbm_bytes = 0.0;
                    return value;
                }(),
                analytical::Efficiency{0.5, 0.5, 0.0}, 1.0));
        },
        "negative roofline work must be rejected");
    expect_throws<communication::CommunicationModelError>(
        [] {
            const communication::AnalyticalCommunicationModel model{
                communication::AnalyticalCommunicationConfig{
                    100.0,
                    1.0,
                    1'000.0,
                }};
            static_cast<void>(model.allreduce_ms(1'000, 0));
        },
        "zero-device collective must be rejected");
    expect_throws<kv_transfer::TransferModelError>(
        [] {
            static_cast<void>(kv_transfer::dense_kv_cache_size_bytes(1, [&]() {
                kv_transfer::DenseKvLayout value{};
                value.num_layers = 0;
                value.num_kv_heads_per_worker = 1;
                value.head_dim = 1;
                return value;
            }()));
        },
        "invalid KV layout must be rejected");
}

} // namespace

int main() {
    int failures = 0;
    failures += frontier::test::run("roofline matches Python golden",
                                    test_roofline_matches_python_golden);
    failures += frontier::test::run("device presets and overrides",
                                    test_device_presets_and_overrides);
    failures += frontier::test::run("dense layer matches Python golden",
                                    test_dense_layer_matches_python_golden);
    failures += frontier::test::run("long-context decode cost increases",
                                    test_long_context_decode_cost_increases);
    failures +=
        frontier::test::run("operator precisions split dense and KV costs",
                            test_operator_precisions_split_dense_and_kv_costs);
    failures += frontier::test::run(
        "operator precisions split MoE expert and router costs",
        test_operator_precisions_split_moe_expert_and_router_costs);
    failures += frontier::test::run("MLA uses latent-cache context costs",
                                    test_mla_uses_latent_cache_context_costs);
    failures += frontier::test::run("MLA KV layout splits latent and RoPE",
                                    test_mla_kv_layout_splits_latent_and_rope);
    failures += frontier::test::run("MFA models shared-Q projection path",
                                    test_mfa_models_shared_q_projection_path);
    failures += frontier::test::run("batch model matches Python golden",
                                    test_batch_model_matches_python_golden);
    failures += frontier::test::run("communication matches Python golden",
                                    test_communication_matches_python_golden);
    failures += frontier::test::run("KV transfer matches Python golden",
                                    test_kv_transfer_matches_python_golden);
    failures += frontier::test::run("Kimi K2 uneven pipeline and LM head",
                                    test_kimi_k2_uneven_pipeline_and_lm_head);
    failures += frontier::test::run(
        "Kimi K2 first-layer-scaled prediction",
        test_kimi_k2_first_layer_scaled_prediction);
    failures +=
        frontier::test::run("invalid analytical inputs are rejected",
                            test_invalid_analytical_inputs_are_rejected);
    return failures == 0 ? 0 : 1;
}
