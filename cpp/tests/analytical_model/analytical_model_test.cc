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
#include <sstream>
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
using frontier::entities::RequestCollection;
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
        std::ostringstream message;
        message.precision(17);
        message << field << " mismatch: actual=" << actual
                << ", expected=" << expected;
        throw std::runtime_error(message.str());
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

void test_dense_layer_matches_cpp_golden() {
    // The C++ shape-aware GEMM curve intentionally moved ahead of the legacy
    // Python two-bucket predictor. This fixture pins the active C++ contract.
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

void test_prefill_attention_token_pairs_are_exact() {
    const std::vector<analytical::AttentionRequestSlice> slices{
        {3, 10}, // 3 * (10 + (3 + 1) / 2) = 36
        {4, 7},  // 4 * (7 + (4 + 1) / 2) = 38
        {2, 5},  // A chunk with cached context contributes 13.
    };
    expect(analytical::prefill_attention_token_pairs(slices) == 87,
           "PREFILL attention token-pair work must retain exact arithmetic");
    expect(analytical::prefill_attention_token_pairs({{1, 0}, {1, 1}}) == 3,
           "PREFILL attention token-pair work must include cached context");
    expect(analytical::prefill_attention_token_pairs({}) == 0,
           "empty PREFILL attention slices must have zero token-pair work");
}

void test_operator_precisions_split_dense_and_kv_costs() {
    expect(
        analytical::precision_from_string("fp8") ==
                analytical::Precision::kFp8 &&
            analytical::precision_from_string("fp4") ==
                analytical::Precision::kFp4 &&
            analytical::precision_from_string("mxfp8") ==
                analytical::Precision::kMxFp8 &&
            analytical::precision_from_string("mxfp4") ==
                analytical::Precision::kMxFp4 &&
            analytical::bytes_per_element(analytical::Precision::kFp8) == 1.0 &&
            analytical::bytes_per_element(analytical::Precision::kFp4) == 0.5 &&
            analytical::bytes_per_element(analytical::Precision::kMxFp8) ==
                1.0 + 1.0 / 32.0 &&
            analytical::bytes_per_element(analytical::Precision::kMxFp4) ==
                0.5 + 1.0 / 32.0,
        "FP and MXFP precision names must include packed values and MX block "
        "scales");

    analytical::DenseBatch batch{};
    batch.total_tokens = 1;
    batch.decode_requests = {{1, 8'192}};
    const auto predict = [&](analytical::DenseOperatorPrecisions precisions) {
        return analytical::predict_dense_layer(
            analytical::DeviceCeilings::rubin(), analytical::AnalyticalConfig{},
            analytical::DenseModel::llama2_7b_tp8(), batch, precisions);
    };
    const auto precisions = [](analytical::Precision attention,
                               analytical::Precision dense,
                               analytical::Precision kv_cache) {
        analytical::DenseOperatorPrecisions value{};
        value.attention = attention;
        value.dense = dense;
        value.kv_cache = kv_cache;
        return value;
    };
    const auto fp16 = predict(precisions(analytical::Precision::kFp16,
                                         analytical::Precision::kFp16,
                                         analytical::Precision::kFp16));
    const auto fp8_kv = predict(precisions(analytical::Precision::kFp8,
                                           analytical::Precision::kFp16,
                                           analytical::Precision::kFp8));
    expect(fp8_kv.decode_attention_ms < fp16.decode_attention_ms &&
               fp8_kv.kv_cache_save_ms < fp16.kv_cache_save_ms,
           "FP8 attention/KV must reduce long-context reads and KV writes");
    expect_approximately_equal(fp8_kv.mlp_up_projection_ms,
                               fp16.mlp_up_projection_ms,
                               "dense precision independence");

    auto fp8_attention_core =
        precisions(analytical::Precision::kFp16,
                   analytical::Precision::kFp16,
                   analytical::Precision::kFp8);
    fp8_attention_core.attention_weight = analytical::Precision::kBf16;
    fp8_attention_core.attention_activation = analytical::Precision::kBf16;
    fp8_attention_core.attention_core = analytical::Precision::kFp8;
    const auto split_attention = predict(fp8_attention_core);
    auto bf16_attention_core = fp8_attention_core;
    bf16_attention_core.attention_core = analytical::Precision::kBf16;
    const auto bf16_core = predict(bf16_attention_core);
    expect(split_attention.decode_attention_ms < bf16_core.decode_attention_ms,
           "FP8 attention core must use the FP8 QK/PV ceiling");
    expect_approximately_equal(split_attention.attention_pre_projection_ms,
                               bf16_core.attention_pre_projection_ms,
                               "attention core must not change QKV projection");
    expect_approximately_equal(split_attention.attention_post_projection_ms,
                               bf16_core.attention_post_projection_ms,
                               "attention core must not change output projection");
    expect_approximately_equal(split_attention.attention_norm_ms,
                               bf16_core.attention_norm_ms,
                               "attention core must not change normalization");

    const auto fp4_dense = predict(precisions(analytical::Precision::kFp16,
                                              analytical::Precision::kFp4,
                                              analytical::Precision::kFp16));
    expect(fp4_dense.mlp_up_projection_ms < fp16.mlp_up_projection_ms,
           "FP4 dense override must reduce MLP projection time");
    expect_approximately_equal(fp4_dense.attention_pre_projection_ms,
                               fp16.attention_pre_projection_ms,
                               "attention precision independence");

    auto fp4_weight_precisions =
        precisions(analytical::Precision::kFp16, analytical::Precision::kFp16,
                   analytical::Precision::kFp16);
    fp4_weight_precisions.attention_weight = analytical::Precision::kFp4;
    fp4_weight_precisions.attention_activation = analytical::Precision::kFp16;
    fp4_weight_precisions.dense_weight = analytical::Precision::kFp4;
    fp4_weight_precisions.dense_activation = analytical::Precision::kFp16;
    const auto fp4_weight_fp16_activation = predict(fp4_weight_precisions);
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
    const auto precisions = [](analytical::Precision expert,
                               analytical::Precision router,
                               analytical::Precision dense) {
        analytical::MoEOperatorPrecisions value{};
        value.expert = expert;
        value.router = router;
        value.dense = dense;
        return value;
    };
    const auto fp16 = predict(precisions(analytical::Precision::kFp16,
                                         analytical::Precision::kFp16,
                                         analytical::Precision::kFp16));
    const auto fp4_expert = predict(precisions(analytical::Precision::kFp4,
                                               analytical::Precision::kFp16,
                                               analytical::Precision::kFp16));
    expect(fp4_expert.grouped_up_projection_ms <
                   fp16.grouped_up_projection_ms &&
               fp4_expert.grouped_down_projection_ms <
                   fp16.grouped_down_projection_ms,
           "FP4 expert override must reduce grouped expert GEMM time");
    expect_approximately_equal(fp4_expert.gating_linear_ms,
                               fp16.gating_linear_ms,
                               "router precision independence");

    const auto fp8_router = predict(precisions(analytical::Precision::kFp16,
                                               analytical::Precision::kFp8,
                                               analytical::Precision::kFp16));
    expect(fp8_router.gating_linear_ms < fp16.gating_linear_ms &&
               fp8_router.gating_routing_topk_ms < fp16.gating_routing_topk_ms,
           "FP8 router override must reduce router operator time");
    expect_approximately_equal(fp8_router.grouped_up_projection_ms,
                               fp16.grouped_up_projection_ms,
                               "expert precision independence");

    analytical::MoEOperatorPrecisions mixed{};
    mixed.expert = analytical::Precision::kFp16;
    mixed.router = analytical::Precision::kFp16;
    mixed.dense = analytical::Precision::kFp16;
    mixed.shared_expert = analytical::Precision::kFp16;
    mixed.expert_weight = analytical::Precision::kFp4;
    mixed.expert_activation = analytical::Precision::kFp8;
    const auto fp4_weight_fp8_activation = predict(mixed);
    expect(fp4_weight_fp8_activation.grouped_up_projection_ms <
               fp16.grouped_up_projection_ms,
           "MoE W4A8 must model expert weight and activation dtypes "
           "independently");

    model.num_shared_experts = 1;
    analytical::MoEOperatorPrecisions native_mixed{};
    native_mixed.expert = analytical::Precision::kMxFp4;
    native_mixed.shared_expert = analytical::Precision::kBf16;
    native_mixed.router = analytical::Precision::kFp32;
    native_mixed.dense = analytical::Precision::kBf16;
    native_mixed.expert_weight = analytical::Precision::kMxFp4;
    native_mixed.expert_activation = analytical::Precision::kMxFp8;
    native_mixed.shared_expert_weight = analytical::Precision::kBf16;
    native_mixed.shared_expert_activation = analytical::Precision::kBf16;
    const auto native_shared = predict(native_mixed);
    native_mixed.shared_expert = analytical::Precision::kMxFp4;
    native_mixed.shared_expert_weight = analytical::Precision::kMxFp4;
    native_mixed.shared_expert_activation = analytical::Precision::kMxFp8;
    const auto quantized_shared = predict(native_mixed);
    expect(native_shared.grouped_up_projection_ms >
                   quantized_shared.grouped_up_projection_ms &&
               native_shared.grouped_down_projection_ms >
                   quantized_shared.grouped_down_projection_ms,
           "shared-expert BF16 work must remain separate from routed MXFP4/8 "
           "work");
}

void test_router_storage_and_compute_precisions_are_independent() {
    analytical::MoEModel model{};
    model.hidden_size = 4'096;
    model.intermediate_size = 14'336;
    model.model_num_experts = 256;
    model.moe_tensor_parallel_size = 1;
    const std::vector<std::uint64_t> expert_tokens(model.model_num_experts, 1);

    analytical::MoEOperatorPrecisions bf16_storage{};
    bf16_storage.expert = analytical::Precision::kFp16;
    bf16_storage.router = analytical::Precision::kFp16;
    bf16_storage.dense = analytical::Precision::kFp16;
    bf16_storage.shared_expert = analytical::Precision::kFp16;
    bf16_storage.router_weight = analytical::Precision::kBf16;
    bf16_storage.router_activation = analytical::Precision::kBf16;
    bf16_storage.router_compute = analytical::Precision::kFp32;
    const auto bf16 = analytical::predict_moe_layer(
        analytical::DeviceCeilings::rubin(), analytical::AnalyticalConfig{},
        model, 512, 2, expert_tokens, bf16_storage);

    auto bf16_logits = bf16_storage;
    bf16_logits.router_compute = analytical::Precision::kBf16;
    const auto bf16_output = analytical::predict_moe_layer(
        analytical::DeviceCeilings::rubin(), analytical::AnalyticalConfig{},
        model, 512, 2, expert_tokens, bf16_logits);
    expect(bf16.gating_linear_ms >= bf16_output.gating_linear_ms,
           "FP32 router logits may add output traffic but must retain the "
           "BF16 operand GEMM ceiling");

    auto fp32_storage = bf16_storage;
    fp32_storage.router_weight = analytical::Precision::kFp32;
    const auto fp32 = analytical::predict_moe_layer(
        analytical::DeviceCeilings::rubin(), analytical::AnalyticalConfig{},
        model, 512, 2, expert_tokens, fp32_storage);
    expect(fp32.gating_linear_ms > bf16.gating_linear_ms,
           "router resident FP32 weights must increase HBM traffic even when "
           "router compute remains FP32");

    const auto output_fp32 = analytical::gemm_work(
        512, model.hidden_size, model.model_num_experts,
        analytical::bytes_per_element(analytical::Precision::kBf16),
        analytical::bytes_per_element(analytical::Precision::kBf16),
        analytical::bytes_per_element(analytical::Precision::kFp32), 1);
    expect(output_fp32.hbm_bytes > 0.0,
           "router GEMM must account for FP32 logits writes");
}

void test_gemm_efficiency_is_shape_continuous() {
    const auto config = analytical::analytical_config_from_profile(
        "k3_sglang_mxfp4", "gb300");
    const auto efficiency = [&](std::uint64_t m, std::uint64_t k,
                                std::uint64_t n) {
        return analytical::gemm_efficiency_for_shape(config.gemm, m, k, n);
    };
    const auto m1 = efficiency(1, 7'168, 7'168);
    const auto m32 = efficiency(32, 7'168, 7'168);
    const auto m64 = efficiency(64, 7'168, 7'168);
    const auto m127 = efficiency(127, 7'168, 7'168);
    const auto m128 = efficiency(128, 7'168, 7'168);
    const auto m256 = efficiency(256, 7'168, 7'168);
    expect(m1.compute == config.gemm.floor.compute &&
               m1.memory == config.gemm.floor.memory &&
               m1.overlap_penalty == config.gemm.floor.overlap_penalty,
           "M=1 must retain the measured GEMV-like efficiency floor");
    expect(m1.compute < m32.compute && m32.compute < m64.compute &&
               m64.compute < m127.compute && m127.compute < m128.compute &&
               m128.compute < m256.compute,
           "Tensor Core GEMM compute efficiency must increase smoothly with M");
    expect(m1.memory < m32.memory && m32.memory < m64.memory &&
               m64.memory < m127.memory && m127.memory < m128.memory &&
               m128.memory < m256.memory,
           "Tensor Core GEMM memory efficiency must increase smoothly with M");
    expect(m1.overlap_penalty > m32.overlap_penalty &&
               m32.overlap_penalty > m64.overlap_penalty &&
               m64.overlap_penalty > m127.overlap_penalty &&
               m127.overlap_penalty > m128.overlap_penalty &&
               m128.overlap_penalty > m256.overlap_penalty,
           "roofline overlap penalty must decay smoothly with GEMM size");
    expect(std::abs(m128.compute - m127.compute) < 0.002,
           "M=128 must not introduce the former efficiency discontinuity");

    const auto narrow_n = efficiency(64, 7'168, 512);
    const auto wide_n = efficiency(64, 7'168, 28'672);
    const auto shallow_k = efficiency(64, 128, 7'168);
    expect(narrow_n.compute < wide_n.compute,
           "a narrow N grid must expose less of the GPU than a full CTA wave");
    expect(shallow_k.compute < m64.compute,
           "a shallow K loop must expose Tensor Core pipeline fill");
}

void test_k3_latent_moe_front_is_one_gemm() {
    analytical::MoEModel model{};
    model.hidden_size = 7'168;
    model.intermediate_size = 18'432;
    model.model_num_experts = 896;
    model.moe_tensor_parallel_size = 1;
    model.routed_expert_hidden_size = 3'584;
    const std::vector<std::uint64_t> expert_tokens(model.model_num_experts, 1);

    analytical::MoEOperatorPrecisions precisions{};
    precisions.expert = analytical::Precision::kMxFp4;
    precisions.expert_weight = analytical::Precision::kMxFp4;
    precisions.expert_activation = analytical::Precision::kMxFp8;
    precisions.latent_moe_projection_weight = analytical::Precision::kBf16;
    precisions.latent_moe_projection_activation = analytical::Precision::kBf16;
    precisions.router_weight = analytical::Precision::kBf16;
    precisions.router_activation = analytical::Precision::kBf16;
    precisions.router_compute = analytical::Precision::kFp32;

    analytical::AnalyticalConfig fused =
        analytical::analytical_config_from_profile("k3_sglang_mxfp4",
                                                   "gb300");
    analytical::AnalyticalConfig separate = fused;
    separate.fuse_latent_moe_front = false;
    const auto fused_front = analytical::predict_moe_layer(
        analytical::DeviceCeilings::gb300(), fused, model, 64, 16,
        expert_tokens, precisions);
    const auto separate_front = analytical::predict_moe_layer(
        analytical::DeviceCeilings::gb300(), separate, model, 64, 16,
        expert_tokens, precisions);

    expect(fused_front.gating_linear_ms > separate_front.gating_linear_ms,
           "fused-front bucket must include router and latent down work");
    expect(fused_front.latent_projection_ms <
               separate_front.latent_projection_ms,
           "fused latent bucket must retain only the up projection");
    expect(fused_front.gating_linear_ms + fused_front.latent_projection_ms <
               separate_front.gating_linear_ms +
                   separate_front.latent_projection_ms,
           "one fused-front GEMM must save the duplicate input read");
    expect_approximately_equal(
        fused_front.gating_routing_topk_ms,
        separate_front.gating_routing_topk_ms,
        "front fusion must not change router top-k work");
    expect_approximately_equal(
        fused_front.grouped_up_projection_ms,
        separate_front.grouped_up_projection_ms,
        "front fusion must not change grouped expert up work");
    expect_approximately_equal(
        fused_front.grouped_down_projection_ms,
        separate_front.grouped_down_projection_ms,
        "front fusion must not change grouped expert down work");

    analytical::AnalyticalConfig slow_tgv = fused;
    slow_tgv.latent_moe_front_floor =
        analytical::Efficiency{0.10, 0.10, 0.375};
    const auto slow_small_front = analytical::predict_moe_layer(
        analytical::DeviceCeilings::gb300(), slow_tgv, model, 64, 16,
        expert_tokens, precisions);
    expect(slow_small_front.gating_linear_ms > fused_front.gating_linear_ms,
           "small-M fused front must use its TGV-specific efficiency");
    const auto fused_large = analytical::predict_moe_layer(
        analytical::DeviceCeilings::gb300(), fused, model, 512, 16,
        expert_tokens, precisions);
    const auto slow_tgv_large = analytical::predict_moe_layer(
        analytical::DeviceCeilings::gb300(), slow_tgv, model, 512, 16,
        expert_tokens, precisions);
    expect(slow_tgv_large.gating_linear_ms > fused_large.gating_linear_ms &&
               slow_tgv_large.gating_linear_ms - fused_large.gating_linear_ms <
                   slow_small_front.gating_linear_ms -
                       fused_front.gating_linear_ms,
           "the fused-front floor influence must decay smoothly with M");

    // The released plain-TP decode graph applies several additional
    // schedules around this two-way front: shared gate/up is the third GEMM
    // output, route+quant is one launch, shared down overlaps routed experts,
    // and TP8 uses the column-sharded latent-up GEMM+all-gather for M <= 12.
    analytical::MoEModel tp8 = model;
    tp8.intermediate_size = 3'072;
    tp8.num_shared_experts = 2;
    tp8.moe_tensor_parallel_size = 8;
    tp8.expert_parallel_size = 1;
    tp8.latent_moe_use_norm = true;
    tp8.decode_only = true;
    analytical::MoEOperatorPrecisions k3_precisions = precisions;
    k3_precisions.shared_expert = analytical::Precision::kBf16;
    k3_precisions.shared_expert_weight = analytical::Precision::kBf16;
    k3_precisions.shared_expert_activation = analytical::Precision::kBf16;
    k3_precisions.dense = analytical::Precision::kBf16;
    k3_precisions.dense_weight = analytical::Precision::kBf16;
    k3_precisions.dense_activation = analytical::Precision::kBf16;
    std::vector<std::uint64_t> one_token_routing(tp8.model_num_experts, 0);
    std::fill_n(one_token_routing.begin(), 16, 1);

    analytical::AnalyticalConfig k3_decode = fused;
    analytical::AnalyticalConfig serial_decode = k3_decode;
    serial_decode.fuse_shared_expert_moe_front = false;
    serial_decode.fuse_route_quant = false;
    serial_decode.overlap_shared_routed_moe = false;
    serial_decode.use_tp8_latent_up_gemm_allgather = false;
    const auto scheduled = analytical::predict_moe_layer(
        analytical::DeviceCeilings::gb300(), k3_decode, tp8, 1, 16,
        one_token_routing, k3_precisions);
    const auto serial = analytical::predict_moe_layer(
        analytical::DeviceCeilings::gb300(), serial_decode, tp8, 1, 16,
        one_token_routing, k3_precisions);
    expect(scheduled.total_ms() < serial.total_ms(),
           "K3 plain-TP decode schedules must shorten the MoE critical path");
    expect(scheduled.shuffling_ms == 0.0 && serial.shuffling_ms > 0.0,
           "fused route+quant must remove the standalone shuffle launch");
    expect(scheduled.latent_projection_ms < serial.latent_projection_ms,
           "TP8 small-M latent up must use the sharded GEMM+all-gather path");

    analytical::MoEModel prefill_tp8 = tp8;
    prefill_tp8.decode_only = false;
    const auto prefill_scheduled = analytical::predict_moe_layer(
        analytical::DeviceCeilings::gb300(), k3_decode, prefill_tp8, 1, 16,
        one_token_routing, k3_precisions);
    const auto prefill_serial = analytical::predict_moe_layer(
        analytical::DeviceCeilings::gb300(), serial_decode, prefill_tp8, 1,
        16, one_token_routing, k3_precisions);
    expect_approximately_equal(
        prefill_scheduled.total_ms(), prefill_serial.total_ms(),
        "decode-only K3 MoE schedules must not alter prefill");

    std::vector<std::uint64_t> thirteen_token_routing(tp8.model_num_experts,
                                                       0);
    std::fill_n(thirteen_token_routing.begin(), 16, 13);
    analytical::AnalyticalConfig no_small_gather = k3_decode;
    no_small_gather.use_tp8_latent_up_gemm_allgather = false;
    const auto m13 = analytical::predict_moe_layer(
        analytical::DeviceCeilings::gb300(), k3_decode, tp8, 13, 16,
        thirteen_token_routing, k3_precisions);
    const auto m13_without_gather = analytical::predict_moe_layer(
        analytical::DeviceCeilings::gb300(), no_small_gather, tp8, 13, 16,
        thirteen_token_routing, k3_precisions);
    expect_approximately_equal(
        m13.latent_projection_ms, m13_without_gather.latent_projection_ms,
        "TP8 latent GEMM+all-gather must stop above the published M=12 limit");
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
        analytical::DenseOperatorPrecisions precisions{};
        precisions.attention = analytical::Precision::kFp8;
        precisions.dense = analytical::Precision::kBf16;
        precisions.kv_cache = analytical::Precision::kFp8;
        precisions.attention_weight = analytical::Precision::kFp8;
        precisions.attention_activation = analytical::Precision::kBf16;
        precisions.dense_weight = analytical::Precision::kFp8;
        precisions.dense_activation = analytical::Precision::kBf16;
        return analytical::predict_dense_layer(
            analytical::DeviceCeilings::rubin(), analytical::AnalyticalConfig{},
            model, batch, precisions);
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
    expect(attention::mla_dcp_local_token_count(10, 4, 0) == 3 &&
               attention::mla_dcp_local_token_count(10, 4, 1) == 3 &&
               attention::mla_dcp_local_token_count(10, 4, 2) == 2 &&
               attention::mla_dcp_local_token_count(10, 4, 3) == 2,
           "MLA DCP must interleave tokens across ranks");
    expect(attention::mla_dcp_rank_kv_cache_size_bytes(10, 61, layout, 4, 0) ==
               117'120,
           "MLA DCP rank-local KV size must use the local token count");

    const auto kimi =
        frontier::config::load_model_config("moonshotai/Kimi-K2-Instruct");
    expect(kv_transfer::model_kv_cache_size_bytes(10, kimi, 1.0) == 390'400,
           "Kimi one-copy KV size must use the shared MLA KV layout");
    expect(kv_transfer::model_kv_cache_size_bytes_target_physical(
               10, kimi, 1.0, 4) == 1'561'600,
           "Kimi TP4 target KV size must include MLA replication");
    expect(kv_transfer::model_kv_cache_size_bytes_target_physical(
               10, kimi, 1.0, 4, 4) == 390'400,
           "Kimi TP4/DCP4 target KV size must contain one physical copy");
    expect(kv_transfer::model_kv_cache_size_bytes_target_physical(
               16, kimi, 1.0, 4, 2) == 1'249'280,
           "Kimi TP4/DCP2 target KV size must contain two physical copies");
    expect(kv_transfer::model_kv_cache_size_bytes_rank_local(10, kimi, 1.0, 4,
                                                             0) == 117'120,
           "Kimi TP4/DCP4 rank-local KV must contain only assigned tokens");
    expect(kv_transfer::model_kv_cache_size_bytes_one_copy(16, kimi, 1.0) ==
               624'640,
           "Kimi one-copy block size must remain rank-local");
    expect(kv_transfer::model_kv_cache_size_bytes_target_physical(
               16, kimi, 1.0, 4) == 2'498'560,
           "Kimi TP4 target block size must include MLA replication");
    expect(kv_transfer::model_kv_cache_size_bytes_target_physical(
               65'536, kimi, 1.0, 4) == 10'234'101'760ULL,
           "Kimi TP4 target 64Ki footprint must match the physical contract");
    expect(kv_transfer::model_kv_cache_size_bytes_target_physical(
               65'536, kimi, 1.0, 4, 2) == 5'117'050'880ULL,
           "Kimi TP4/DCP2 target 64Ki footprint must keep two copies");

    frontier::config::KvCacheTransferConfig transfer_config{};
    transfer_config.network_bandwidth_gbps = 200.0;
    transfer_config.network_latency_ms = 0.5;
    transfer_config.kv_cache_dtype_size_bytes = 1.0;
    const auto predictor =
        kv_transfer::make_kv_cache_transfer_predictor(transfer_config, 4, 4);
    expect(predictor->predict(16, kimi).size_bytes == 624'640,
           "PDD predictor must transfer one TP4/DCP4 MLA copy");
    const auto replicated_predictor =
        kv_transfer::make_kv_cache_transfer_predictor(transfer_config, 4);
    expect(replicated_predictor->predict(16, kimi).size_bytes == 2'498'560,
           "PDD predictor must report target-physical MLA bytes");

    const auto k3 = frontier::config::load_model_config("moonshotai/Kimi-K3");
    expect(kv_transfer::model_kv_cache_size_bytes(10, k3, 1.0) == 153'600,
           "K3 token-proportional KV must count only 24 MLA layers");
    expect(
        kv_transfer::model_kda_state_snapshot_size_bytes(k3, 4.0) ==
                464'633'856ULL &&
            kv_transfer::model_kda_state_snapshot_size_bytes_rank_local(
                k3, 4.0, 8) == 58'079'232ULL,
        "K3 must expose an explicitly requested FP32 snapshot sharded over TP");
    expect(predictor->predict(16, k3).size_bytes == 232'562'688ULL,
           "PDD K3 transfer must include MLA KV plus one atomic KDA snapshot");
    const auto fp32_snapshot_predictor =
        kv_transfer::make_kv_cache_transfer_predictor(transfer_config, 4, 4,
                                                      4.0);
    expect(fp32_snapshot_predictor->predict(16, k3).size_bytes ==
               464'879'616ULL,
           "PDD K3 transfer must honor an explicit FP32 KDA snapshot dtype");
}

void test_mla_dcp_shards_decode_context_work() {
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

    analytical::DenseBatch batch{};
    batch.total_tokens = 1;
    batch.decode_requests = {{1, 8'192}};
    const auto replicated = analytical::predict_dense_layer(
        analytical::DeviceCeilings::rubin(), analytical::AnalyticalConfig{},
        model, batch, analytical::Precision::kFp16);
    model.decode_context_parallel_size = 4;
    const auto sharded = analytical::predict_dense_layer(
        analytical::DeviceCeilings::rubin(), analytical::AnalyticalConfig{},
        model, batch, analytical::Precision::kFp16);
    expect_approximately_equal(
        sharded.kv_cache_save_ms, replicated.kv_cache_save_ms,
        "a single new token still has one critical-rank KV write");
    expect(sharded.decode_attention_ms < replicated.decode_attention_ms,
           "MLA DCP must reduce rank-local latent-cache read traffic");

    analytical::DenseBatch decode_write_batch{};
    decode_write_batch.total_tokens = 65'536;
    decode_write_batch.decode_requests = {{65'536, 8'192}};
    model.decode_context_parallel_size = 1;
    const auto replicated_decode_write = analytical::predict_dense_layer(
        analytical::DeviceCeilings::rubin(), analytical::AnalyticalConfig{},
        model, decode_write_batch, analytical::Precision::kFp16);
    model.decode_context_parallel_size = 4;
    const auto sharded_decode_write = analytical::predict_dense_layer(
        analytical::DeviceCeilings::rubin(), analytical::AnalyticalConfig{},
        model, decode_write_batch, analytical::Precision::kFp16);
    expect(sharded_decode_write.kv_cache_save_ms <
               replicated_decode_write.kv_cache_save_ms,
           "MLA DCP must shard persistent KV writes produced by decode");

    analytical::DenseBatch prefill_batch{};
    prefill_batch.total_tokens = 65'536;
    prefill_batch.prefill_requests = {{65'536, 0}};
    model.decode_context_parallel_size = 1;
    const auto replicated_prefill = analytical::predict_dense_layer(
        analytical::DeviceCeilings::rubin(), analytical::AnalyticalConfig{},
        model, prefill_batch, analytical::Precision::kFp16);
    model.decode_context_parallel_size = 4;
    const auto sharded_prefill = analytical::predict_dense_layer(
        analytical::DeviceCeilings::rubin(), analytical::AnalyticalConfig{},
        model, prefill_batch, analytical::Precision::kFp16);
    expect_approximately_equal(sharded_prefill.prefill_attention_ms,
                               replicated_prefill.prefill_attention_ms,
                               "MLA DCP prefill attention compute");
    expect(sharded_prefill.kv_cache_save_ms <
               replicated_prefill.kv_cache_save_ms,
           "MLA DCP must shard persistent KV writes produced by prefill");
}

void test_mla_output_gate_and_nope_costs() {
    analytical::DenseModel model{};
    model.hidden_size = 7'168;
    model.intermediate_size = 18'432;
    model.num_query_heads = 96;
    model.num_kv_heads = 96;
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

    analytical::DenseBatch batch{};
    batch.total_tokens = 4;
    batch.decode_requests = {{4, 2'048}};
    const auto predict = [&](const analytical::DenseModel &value) {
        return analytical::predict_dense_layer(
            analytical::DeviceCeilings::rubin(), analytical::AnalyticalConfig{},
            value, batch, analytical::Precision::kBf16);
    };

    // K2/default MLA remains ungated and rotary.  Explicitly spelling the
    // false defaults must preserve the exact same analytical result.
    const auto k2_default = predict(model);
    model.mla_use_output_gate = false;
    model.mla_use_nope = false;
    const auto k2_explicit_defaults = predict(model);
    expect_approximately_equal(k2_default.total_ms(),
                               k2_explicit_defaults.total_ms(),
                               "K2 MLA default flags");
    expect(k2_default.rope_ms > 0.0,
           "legacy MLA must retain rotary-compute cost");

    model.mla_use_output_gate = true;
    const auto gated = predict(model);
    expect(gated.attention_post_projection_ms >
               k2_explicit_defaults.attention_post_projection_ms,
           "Gated MLA must add full-rank gate projection and fusion work");
    expect(gated.total_ms() > k2_explicit_defaults.total_ms(),
           "Gated MLA total time must include gate work");

    analytical::AnalyticalConfig overlapped =
        analytical::analytical_config_from_profile("k3_sglang_mxfp4",
                                                   "gb300");
    analytical::AnalyticalConfig serial_gate = overlapped;
    serial_gate.overlap_mla_output_gate = false;
    const auto overlapped_gate = analytical::predict_dense_layer(
        analytical::DeviceCeilings::gb300(), overlapped, model, batch,
        analytical::Precision::kBf16);
    const auto serial_gate_times = analytical::predict_dense_layer(
        analytical::DeviceCeilings::gb300(), serial_gate, model, batch,
        analytical::Precision::kBf16);
    expect(overlapped_gate.total_ms() < serial_gate_times.total_ms(),
           "K3 decode must overlap the MLA output-gate GEMM with attention");

    analytical::DenseBatch large_decode = batch;
    large_decode.total_tokens = 129;
    large_decode.decode_requests = {{129, 2'048}};
    const auto large_overlapped = analytical::predict_dense_layer(
        analytical::DeviceCeilings::gb300(), overlapped, model, large_decode,
        analytical::Precision::kBf16);
    const auto large_serial = analytical::predict_dense_layer(
        analytical::DeviceCeilings::gb300(), serial_gate, model, large_decode,
        analytical::Precision::kBf16);
    expect_approximately_equal(
        large_overlapped.total_ms(), large_serial.total_ms(),
        "MLA gate overlap must stop above the published batch-128 limit");

    model.mla_use_output_gate = false;
    model.mla_use_nope = true;
    const auto nope = predict(model);
    expect(nope.rope_ms == 0.0, "NoPE MLA must omit rotary transform compute");
    // NoPE only removes the rotary transform.  qk_rope channels remain in
    // attention and in the persistent decoupled RoPE cache.
    expect_approximately_equal(nope.decode_attention_ms,
                               k2_explicit_defaults.decode_attention_ms,
                               "NoPE attention dimensions");
    expect_approximately_equal(nope.kv_cache_save_ms,
                               k2_explicit_defaults.kv_cache_save_ms,
                               "NoPE persistent RoPE cache traffic");
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

void test_batch_model_matches_cpp_golden() {
    // Stage totals include the C++-only shape-aware GEMM curve.
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
        RequestCollection requests;
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
                value.session_start_at = SimTime::from_seconds(0.0);
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
        std::uint64_t expected_prefill_attention_token_pairs = 0;
        for (const Json &slice : test_case.at("input").at("slices")) {
            if (slice.at("phase").get<std::string>() == "decode") {
                continue;
            }
            expected_prefill_attention_token_pairs +=
                analytical::prefill_attention_token_pairs(
                    {{slice.at("scheduled_tokens").get<std::uint64_t>(),
                      slice.at("past_context").get<std::uint64_t>()}});
        }
        expect(prediction.execution_time.prefill_attention_token_pairs ==
                   expected_prefill_attention_token_pairs,
               "analytical stage must expose exact PREFILL attention "
               "token-pair work");
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

    RequestCollection requests;
    requests.emplace_back([&]() {
        WorkloadRequest value{};
        value.request_id = RequestId{0};
        value.session_start_at = SimTime::from_seconds(0.0);
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
    expect(stage1.moe_routing.size() == 16 &&
               stage1.moe_routing.front().model_layer_id == 16 &&
               stage1.moe_routing.back().model_layer_id == 31,
           "Kimi stage 1 must contain the next 16 front-filled layers");
    expect(stage0.execution_time.lm_head_ms == 0.0 &&
               stage1.execution_time.lm_head_ms == 0.0 &&
               stage3.execution_time.lm_head_ms > 0.0,
           "only the final PP stage must execute the LM-head projection");
}

void test_kimi_k2_dcp_adds_decode_collectives() {
    frontier::config::ParallelismConfig replicated_parallelism{};
    replicated_parallelism.tensor_parallel_size = 4;
    replicated_parallelism.data_parallel_size = 4;
    replicated_parallelism.moe_tensor_parallel_size = 1;
    replicated_parallelism.moe_expert_parallel_size = 16;
    auto dcp_parallelism = replicated_parallelism;
    dcp_parallelism.decode_context_parallel_size = 4;

    frontier::config::AnalyticalExecutionModelConfig execution{};
    execution.precision = "fp8";
    const auto model_config =
        frontier::config::load_model_config("moonshotai/Kimi-K2-Instruct");
    const predictor::AnalyticalRooflineExecutionTimePredictor replicated{
        execution, replicated_parallelism, model_config,
        frontier::config::MoeRoutingConfig{}};
    const predictor::AnalyticalRooflineExecutionTimePredictor dcp{
        execution, dcp_parallelism, model_config,
        frontier::config::MoeRoutingConfig{}};

    RequestCollection requests;
    requests.emplace_back([&]() {
        WorkloadRequest value{};
        value.request_id = RequestId{0};
        value.session_start_at = SimTime::from_seconds(0.0);
        value.num_prefill_tokens = 8'192;
        value.num_decode_tokens = 2;
        return value;
    }());
    const SimTime now = SimTime::from_seconds(0.0);
    requests.front().on_arrival(now);
    requests.front().on_admitted(now);
    requests.front().advance_scheduler_frontier(8'192);
    requests.front().on_batch_completion(now, 8'192,
                                         frontier::ClusterType::kPrefill);
    requests.front().advance_scheduler_frontier(1);
    RequestBatchSnapshot snapshot{};
    snapshot.request_id = RequestId{0};
    snapshot.scheduled_tokens = 1;
    snapshot.processed_tokens = 8'192;
    snapshot.scheduler_frontier = 8'193;
    const Batch batch{
        BatchId{0}, IterationId{0}, {snapshot}, now, Generation{0}};

    const auto replicated_stage = replicated.predict_stage_execution_time(
        batch, requests, frontier::StageId{0});
    const auto dcp_stage =
        dcp.predict_stage_execution_time(batch, requests, frontier::StageId{0});
    const auto routing_tp_total = [](const auto &prediction) {
        double total = 0.0;
        for (const auto &diagnostic : prediction.moe_routing) {
            total += diagnostic.pre_moe_tp_communication_ms;
        }
        return total;
    };
    expect(diagnostic_value(replicated_stage,
                            "dcp_attention_communication_ms") == 0.0 &&
               diagnostic_value(dcp_stage, "dcp_attention_communication_ms") >
                   0.0 &&
               dcp_stage.execution_time.tp_communication_ms >
                   replicated_stage.execution_time.tp_communication_ms,
           "MLA DCP decode must add all-gather/reduce-scatter communication");
    expect(routing_tp_total(dcp_stage) > routing_tp_total(replicated_stage) &&
               std::abs(routing_tp_total(dcp_stage) -
                        dcp_stage.execution_time.tp_communication_ms) < 1e-12,
           "MLA DCP communication must be attached to the per-MoE-layer "
           "decode arrivals consumed by the scheduler");
    expect_approximately_equal(
        diagnostic_value(dcp_stage,
                         "kv_cache_rank_local_bytes_per_token_per_layer"),
        diagnostic_value(dcp_stage, "kv_cache_bytes_per_token_per_layer") / 4.0,
        "MLA DCP diagnostics must expose per-rank KV-cache bytes");
}

void test_kimi_k3_tp8_dcp8_hybrid_decode() {
    frontier::config::ParallelismConfig replicated_parallelism{};
    replicated_parallelism.tensor_parallel_size = 8;
    replicated_parallelism.data_parallel_size = 1;
    replicated_parallelism.moe_tensor_parallel_size = 1;
    replicated_parallelism.moe_expert_parallel_size = 8;
    auto dcp_parallelism = replicated_parallelism;
    dcp_parallelism.decode_context_parallel_size = 8;

    frontier::config::AnalyticalExecutionModelConfig detailed_config{};
    detailed_config.precision = "bf16";
    auto scaled_config = detailed_config;
    scaled_config.moe_layer_event_mode = "first_layer_scaled";
    const auto model_config =
        frontier::config::load_model_config("moonshotai/Kimi-K3");
    const predictor::AnalyticalRooflineExecutionTimePredictor replicated{
        detailed_config, replicated_parallelism, model_config,
        frontier::config::MoeRoutingConfig{}};
    const predictor::AnalyticalRooflineExecutionTimePredictor detailed_dcp{
        detailed_config, dcp_parallelism, model_config,
        frontier::config::MoeRoutingConfig{}};
    const predictor::AnalyticalRooflineExecutionTimePredictor scaled_dcp{
        scaled_config, dcp_parallelism, model_config,
        frontier::config::MoeRoutingConfig{}};

    RequestCollection requests;
    requests.emplace_back([&]() {
        WorkloadRequest value{};
        value.request_id = RequestId{0};
        value.session_start_at = SimTime::from_seconds(0.0);
        value.num_prefill_tokens = 8'192;
        value.num_decode_tokens = 2;
        return value;
    }());
    const SimTime now = SimTime::from_seconds(0.0);
    requests.front().on_arrival(now);
    requests.front().on_admitted(now);
    requests.front().advance_scheduler_frontier(8'192);
    requests.front().on_batch_completion(now, 8'192,
                                         frontier::ClusterType::kPrefill);
    requests.front().advance_scheduler_frontier(1);
    RequestBatchSnapshot snapshot{};
    snapshot.request_id = RequestId{0};
    snapshot.scheduled_tokens = 1;
    snapshot.processed_tokens = 8'192;
    snapshot.scheduler_frontier = 8'193;
    const Batch batch{
        BatchId{0}, IterationId{0}, {snapshot}, now, Generation{0}};

    const auto replicated_stage = replicated.predict_stage_execution_time(
        batch, requests, frontier::StageId{0});
    const auto detailed_stage = detailed_dcp.predict_stage_execution_time(
        batch, requests, frontier::StageId{0});
    const auto scaled_stage = scaled_dcp.predict_stage_execution_time(
        batch, requests, frontier::StageId{0});
    const auto routing_for_layer = [](const auto &prediction,
                                      std::uint64_t model_layer) {
        return std::find_if(prediction.moe_routing.begin(),
                            prediction.moe_routing.end(),
                            [model_layer](const auto &record) {
                                return record.model_layer_id == model_layer;
                            });
    };
    const auto replicated_kda = routing_for_layer(replicated_stage, 1);
    const auto dcp_kda = routing_for_layer(detailed_stage, 1);
    const auto replicated_mla = routing_for_layer(replicated_stage, 3);
    const auto dcp_mla = routing_for_layer(detailed_stage, 3);
    expect(replicated_kda != replicated_stage.moe_routing.end() &&
               dcp_kda != detailed_stage.moe_routing.end() &&
               replicated_mla != replicated_stage.moe_routing.end() &&
               dcp_mla != detailed_stage.moe_routing.end(),
           "K3 TP8/DCP8 decode must retain both KDA and MLA routing records");
    expect_approximately_equal(dcp_kda->pre_moe_compute_ms,
                               replicated_kda->pre_moe_compute_ms,
                               "K3 KDA compute must ignore replica DCP size");
    expect_approximately_equal(
        dcp_kda->pre_moe_tp_communication_ms,
        replicated_kda->pre_moe_tp_communication_ms,
        "K3 KDA communication must remain TP-only under DCP");
    expect(dcp_mla->pre_moe_compute_ms < replicated_mla->pre_moe_compute_ms &&
               dcp_mla->pre_moe_tp_communication_ms >
                   replicated_mla->pre_moe_tp_communication_ms,
           "K3 MLA decode must shard context work and add DCP collectives");
    expect(diagnostic_value(detailed_stage, "dcp_attention_communication_ms") >
               0.0,
           "K3 TP8/DCP8 decode must expose MLA DCP communication");
    expect_approximately_equal(
        diagnostic_value(detailed_stage,
                         "kv_cache_rank_local_bytes_per_token_per_layer"),
        diagnostic_value(detailed_stage, "kv_cache_bytes_per_token_per_layer") /
            8.0,
        "K3 TP8/DCP8 MLA KV cache must be sequence-sharded eight ways");
    expect(detailed_stage.execution_time == scaled_stage.execution_time,
           "K3 TP8/DCP8 detailed and family-scaled totals must match");
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

    RequestCollection requests;
    requests.emplace_back([&]() {
        WorkloadRequest value{};
        value.request_id = RequestId{0};
        value.session_start_at = SimTime::from_seconds(0.0);
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
    const Batch batch{BatchId{0},
                      IterationId{0},
                      {snapshot},
                      SimTime::from_seconds(0.0),
                      Generation{0}};

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

void test_kimi_k3_attention_family_scaled_prediction() {
    const auto model_config =
        frontier::config::load_model_config("moonshotai/Kimi-K3");
    std::uint64_t kda_layers = 0;
    std::uint64_t mla_layers = 0;
    std::uint64_t moe_kda_layers = 0;
    std::uint64_t moe_mla_layers = 0;
    for (std::uint64_t layer = 0; layer < model_config.num_layers; ++layer) {
        if (model_config.is_kda_layer(layer)) {
            ++kda_layers;
            if (model_config.is_moe_layer(layer)) {
                ++moe_kda_layers;
            }
        }
        if (model_config.is_mla_layer(layer)) {
            ++mla_layers;
            if (model_config.is_moe_layer(layer)) {
                ++moe_mla_layers;
            }
        }
    }
    expect(model_config.num_layers == 93 && model_config.num_kda_layers == 69 &&
               model_config.num_mla_layers == 24 && kda_layers == 69 &&
               mla_layers == 24 && moe_kda_layers == 68 && moe_mla_layers == 24,
           "Kimi K3 must expose 69 KDA/24 MLA layers and 68/24 MoE layers "
           "after the first dense layer");

    frontier::config::ParallelismConfig parallelism{};
    parallelism.tensor_parallel_size = 4;
    parallelism.pipeline_parallel_size = 1;
    parallelism.data_parallel_size = 1;
    parallelism.moe_tensor_parallel_size = 1;
    parallelism.moe_expert_parallel_size = 16;

    frontier::config::AnalyticalExecutionModelConfig detailed_config{};
    detailed_config.precision = "fp8";
    auto scaled_config = detailed_config;
    scaled_config.moe_layer_event_mode = "first_layer_scaled";
    const predictor::AnalyticalRooflineExecutionTimePredictor detailed{
        detailed_config, parallelism, model_config,
        frontier::config::MoeRoutingConfig{}};
    const predictor::AnalyticalRooflineExecutionTimePredictor scaled{
        scaled_config, parallelism, model_config,
        frontier::config::MoeRoutingConfig{}};

    RequestCollection requests;
    requests.emplace_back([&]() {
        WorkloadRequest value{};
        value.request_id = RequestId{0};
        value.session_start_at = SimTime::from_seconds(0.0);
        value.num_prefill_tokens = 64;
        value.num_decode_tokens = 1;
        return value;
    }());
    requests.front().on_arrival(SimTime::from_seconds(0.0));
    requests.front().on_admitted(SimTime::from_seconds(0.0));
    requests.front().advance_scheduler_frontier(64);
    RequestBatchSnapshot snapshot{};
    snapshot.request_id = RequestId{0};
    snapshot.scheduled_tokens = 64;
    snapshot.processed_tokens = 0;
    snapshot.scheduler_frontier = 64;
    const Batch batch{BatchId{0},
                      IterationId{0},
                      {snapshot},
                      SimTime::from_seconds(0.0),
                      Generation{0}};

    const auto detailed_stage = detailed.predict_stage_execution_time(
        batch, requests, frontier::StageId{0});
    const auto scaled_stage = scaled.predict_stage_execution_time(
        batch, requests, frontier::StageId{0});
    expect(detailed_stage.logical_moe_layer_count == 92 &&
               scaled_stage.logical_moe_layer_count == 92 &&
               detailed_stage.moe_routing.size() == 92 &&
               scaled_stage.moe_routing.size() == 1 &&
               scaled_stage.scaled_moe_layer_prediction,
           "Kimi K3 scaling must retain one routing record for all 92 logical "
           "MoE layers");
    expect(detailed_stage.execution_time == scaled_stage.execution_time,
           "Kimi K3 family-scaled prediction must preserve every execution-"
           "time component");
    expect_approximately_equal(
        diagnostic_value(scaled_stage, "attention_weight_element_bytes"), 2.0,
        "Kimi K3 native BF16 attention weight");
    expect_approximately_equal(
        diagnostic_value(scaled_stage, "dense_weight_element_bytes"), 2.0,
        "Kimi K3 native BF16 dense MLP weight");
    expect_approximately_equal(
        diagnostic_value(scaled_stage, "routed_expert_weight_element_bytes"),
        0.5 + 1.0 / 32.0, "Kimi K3 native MXFP4 routed-expert weight");
    expect_approximately_equal(
        diagnostic_value(scaled_stage,
                         "routed_expert_activation_element_bytes"),
        1.0 + 1.0 / 32.0, "Kimi K3 native MXFP8 routed-expert activation");
    expect_approximately_equal(
        diagnostic_value(scaled_stage,
                         "latent_moe_projection_weight_element_bytes"),
        2.0, "Kimi K3 native BF16 Stable LatentMoE projection weight");
    expect_approximately_equal(
        diagnostic_value(scaled_stage,
                         "latent_moe_projection_activation_element_bytes"),
        2.0, "Kimi K3 native BF16 Stable LatentMoE projection activation");
    expect_approximately_equal(
        diagnostic_value(scaled_stage, "shared_expert_weight_element_bytes"),
        2.0, "Kimi K3 native BF16 shared-expert weight");
    expect_approximately_equal(
        diagnostic_value(scaled_stage, "router_compute_element_bytes"), 4.0,
        "Kimi K3 native FP32 router compute");
    expect_approximately_equal(
        diagnostic_value(scaled_stage, "kv_cache_element_bytes"), 1.0,
        "Kimi K3 native FP8 KV cache");
    expect_approximately_equal(
        diagnostic_value(scaled_stage, "kda_snapshot_element_bytes"), 2.0,
        "Kimi K3 native BF16 KDA snapshot");

    const auto &attention_groups = scaled_stage.scaled_moe_attention_groups;
    const auto kda_group = std::find_if(
        attention_groups.begin(), attention_groups.end(),
        [](const auto &group) {
            return group.family == predictor::ScaledMoEAttentionFamily::kKda;
        });
    const auto mla_group = std::find_if(
        attention_groups.begin(), attention_groups.end(),
        [](const auto &group) {
            return group.family == predictor::ScaledMoEAttentionFamily::kMla;
        });
    expect(attention_groups.size() == 2 &&
               kda_group != attention_groups.end() &&
               mla_group != attention_groups.end() &&
               kda_group->layer_count == 67 && mla_group->layer_count == 24 &&
               kda_group->pre_moe_compute_ms_per_layer > 0.0 &&
               mla_group->pre_moe_compute_ms_per_layer > 0.0,
           "Kimi K3 scaling must expose 67 repeated KDA and 24 repeated MLA "
           "attention layers");

    const auto detailed_for_model_layer =
        [&detailed_stage](std::uint64_t model_layer) {
            return std::find_if(detailed_stage.moe_routing.begin(),
                                detailed_stage.moe_routing.end(),
                                [model_layer](const auto &record) {
                                    return record.model_layer_id == model_layer;
                                });
        };
    const auto scaled_first = scaled_stage.moe_routing.begin();
    const auto detailed_kda = detailed_for_model_layer(1);
    const auto detailed_kda_repeat = detailed_for_model_layer(2);
    const auto detailed_mla = detailed_for_model_layer(3);
    expect(scaled_first != scaled_stage.moe_routing.end() &&
               detailed_kda != detailed_stage.moe_routing.end() &&
               detailed_kda_repeat != detailed_stage.moe_routing.end() &&
               detailed_mla != detailed_stage.moe_routing.end() &&
               scaled_first->model_layer_id == detailed_kda->model_layer_id &&
               scaled_first->lane_times_ms == detailed_kda->lane_times_ms &&
               scaled_first->pre_moe_compute_ms ==
                   detailed_kda->pre_moe_compute_ms &&
               detailed_kda->model_layer_id != detailed_mla->model_layer_id,
           "Kimi K3 scaled routing must retain the first detailed MoE layer "
           "while attention-family groups carry the remaining timing");
    expect_approximately_equal(kda_group->pre_moe_compute_ms_per_layer,
                               detailed_kda_repeat->pre_moe_compute_ms,
                               "Kimi K3 repeated KDA attention pre-compute");
    expect_approximately_equal(mla_group->pre_moe_compute_ms_per_layer,
                               detailed_mla->pre_moe_compute_ms,
                               "Kimi K3 repeated MLA attention pre-compute");

    // Exercise the ModelConfig -> DenseModel plumbing used by the stage
    // predictor, rather than only the detail-model helper above.  Disabling
    // K3's full-rank gate removes work from every one of the 24 MLA layers.
    auto no_gate_model_config = model_config;
    no_gate_model_config.mla_use_output_gate = false;
    const predictor::AnalyticalRooflineExecutionTimePredictor no_gate{
        scaled_config, parallelism, no_gate_model_config,
        frontier::config::MoeRoutingConfig{}};
    const auto no_gate_stage = no_gate.predict_stage_execution_time(
        batch, requests, frontier::StageId{0});
    const auto no_gate_mla = std::find_if(
        no_gate_stage.scaled_moe_attention_groups.begin(),
        no_gate_stage.scaled_moe_attention_groups.end(), [](const auto &group) {
            return group.family == predictor::ScaledMoEAttentionFamily::kMla;
        });
    expect(no_gate_stage.duration_ms < scaled_stage.duration_ms &&
               no_gate_mla != no_gate_stage.scaled_moe_attention_groups.end() &&
               no_gate_mla->pre_moe_compute_ms_per_layer <
                   mla_group->pre_moe_compute_ms_per_layer,
           "K3 gated MLA must add gate work to each MLA representative");

    // K3's NoPE flag suppresses only rotary transform compute.  Turning it
    // off on the loaded model must therefore increase the MLA representative
    // time while leaving the latent/decoupled cache layout untouched.
    auto rotary_model_config = model_config;
    rotary_model_config.mla_use_nope = false;
    const predictor::AnalyticalRooflineExecutionTimePredictor rotary{
        scaled_config, parallelism, rotary_model_config,
        frontier::config::MoeRoutingConfig{}};
    const auto rotary_stage = rotary.predict_stage_execution_time(
        batch, requests, frontier::StageId{0});
    const auto rotary_mla = std::find_if(
        rotary_stage.scaled_moe_attention_groups.begin(),
        rotary_stage.scaled_moe_attention_groups.end(), [](const auto &group) {
            return group.family == predictor::ScaledMoEAttentionFamily::kMla;
        });
    expect(rotary_stage.duration_ms > scaled_stage.duration_ms &&
               rotary_mla != rotary_stage.scaled_moe_attention_groups.end() &&
               rotary_mla->pre_moe_compute_ms_per_layer >
                   mla_group->pre_moe_compute_ms_per_layer,
           "K3 NoPE must remove only rotary compute from MLA layers");
}

void test_stage_group_scaled_pp_signatures_and_routing_guard() {
    const auto model_config =
        frontier::config::load_model_config("moonshotai/Kimi-K2-Instruct");

    const auto make_batch = [](std::uint64_t scheduled_tokens = 128,
                               BatchId batch_id = BatchId{0}) {
        RequestCollection requests;
        requests.emplace_back([&]() {
            WorkloadRequest value{};
            value.request_id = RequestId{0};
            value.session_start_at = SimTime::from_seconds(0.0);
            value.num_prefill_tokens = scheduled_tokens;
            value.num_decode_tokens = 1;
            return value;
        }());
        requests.front().on_arrival(SimTime::from_seconds(0.0));
        requests.front().on_admitted(SimTime::from_seconds(0.0));
        requests.front().advance_scheduler_frontier(scheduled_tokens);
        RequestBatchSnapshot snapshot{};
        snapshot.request_id = RequestId{0};
        snapshot.scheduled_tokens = scheduled_tokens;
        snapshot.processed_tokens = 0;
        snapshot.scheduler_frontier = scheduled_tokens;
        const Batch batch{batch_id,
                          IterationId{0},
                          {snapshot},
                          SimTime::from_seconds(0.0),
                          Generation{0}};
        return std::make_pair(std::move(requests), batch);
    };

    for (const std::uint64_t pipeline_parallel_size : {1ULL, 4ULL, 24ULL}) {
        frontier::config::ParallelismConfig parallelism{};
        parallelism.tensor_parallel_size = 4;
        parallelism.pipeline_parallel_size = pipeline_parallel_size;
        parallelism.data_parallel_size = 1;
        parallelism.moe_tensor_parallel_size = 1;
        parallelism.moe_expert_parallel_size = 16;

        frontier::config::AnalyticalExecutionModelConfig detailed_config{};
        detailed_config.precision = "fp8";
        auto grouped_config = detailed_config;
        grouped_config.moe_layer_event_mode = "stage_group_scaled";
        const predictor::AnalyticalRooflineExecutionTimePredictor detailed{
            detailed_config, parallelism, model_config,
            frontier::config::MoeRoutingConfig{}};
        const predictor::AnalyticalRooflineExecutionTimePredictor grouped{
            grouped_config, parallelism, model_config,
            frontier::config::MoeRoutingConfig{}};

        auto detailed_input = make_batch();
        auto grouped_input = make_batch();
        bool saw_timing_cache_hit = false;
        double prior_cache_misses = 0.0;
        for (std::uint64_t stage = 0; stage < pipeline_parallel_size; ++stage) {
            const auto detailed_prediction =
                detailed.predict_stage_execution_time(detailed_input.second,
                                                      detailed_input.first,
                                                      frontier::StageId{stage});
            const auto grouped_prediction =
                grouped.predict_stage_execution_time(grouped_input.second,
                                                     grouped_input.first,
                                                     frontier::StageId{stage});
            expect(detailed_prediction.execution_time ==
                       grouped_prediction.execution_time,
                   "stage_group_scaled must match detailed K2 timing");
            expect(grouped_prediction.scaled_moe_layer_prediction,
                   "balanced stage_group_scaled K2 must use compressed MoE "
                   "events");
            expect(diagnostic_value(grouped_prediction, "timing_group_id") >=
                           0.0 &&
                       diagnostic_value(grouped_prediction,
                                        "timing_group_multiplicity") >= 1.0,
                   "PP stage timing diagnostics must expose deterministic "
                   "group IDs");
            const bool final_stage = stage + 1 == pipeline_parallel_size;
            expect((grouped_prediction.execution_time.pp_communication_ms >
                    0.0) == !final_stage,
                   "stage_group_scaled must preserve PP boundary sends");
            saw_timing_cache_hit =
                saw_timing_cache_hit ||
                diagnostic_value(grouped_prediction, "timing_cache_hit") == 1.0;
            const double cache_misses = diagnostic_value(
                grouped_prediction, "timing_cache_misses_total");
            expect(cache_misses >= prior_cache_misses,
                   "timing cache miss counter must be monotonic");
            prior_cache_misses = cache_misses;
        }
        if (pipeline_parallel_size >= 4) {
            expect(saw_timing_cache_hit,
                   "equivalent PP stages must reuse a timing template");
        }

        // Reuse is batch-scoped even when another batch has exactly the same
        // attention shape. Releasing the cache must also make a repeated
        // direct predictor call miss again.
        if (pipeline_parallel_size > 1) {
            auto independent_input = make_batch(128, BatchId{1});
            const auto independent_prediction =
                grouped.predict_stage_execution_time(independent_input.second,
                                                     independent_input.first,
                                                     frontier::StageId{1});
            expect(diagnostic_value(independent_prediction,
                                    "timing_cache_hit") == 0.0,
                   "identical shapes from different batches must not share "
                   "a timing template");

            grouped.release_batch_timing_cache(BatchId{1});
            const auto released_prediction =
                grouped.predict_stage_execution_time(independent_input.second,
                                                     independent_input.first,
                                                     frontier::StageId{1});
            expect(diagnostic_value(released_prediction, "timing_cache_hit") ==
                       0.0,
                   "released batch timing templates must not remain cached");
            grouped.release_batch_timing_cache(BatchId{1});
        }
        grouped.release_batch_timing_cache(BatchId{0});
    }

    // Randomized and weighted routing include the logical layer in their
    // realized allocation. Grouped mode must fall back to detailed timing
    // instead of reusing the first layer's lane allocation.
    frontier::config::ParallelismConfig parallelism{};
    parallelism.tensor_parallel_size = 4;
    parallelism.pipeline_parallel_size = 4;
    parallelism.moe_tensor_parallel_size = 1;
    parallelism.moe_expert_parallel_size = 16;
    frontier::config::AnalyticalExecutionModelConfig grouped_config{};
    grouped_config.precision = "fp8";
    grouped_config.moe_layer_event_mode = "stage_group_scaled";
    frontier::config::MoeRoutingConfig random_routing{};
    random_routing.distribution =
        frontier::config::MoeRoutingDistribution::kRandom;
    random_routing.layer_scope =
        frontier::config::MoeRoutingLayerScope::kPerLayer;
    const predictor::AnalyticalRooflineExecutionTimePredictor guarded{
        grouped_config, parallelism, model_config, random_routing};
    auto guarded_input = make_batch();
    const auto guarded_prediction = guarded.predict_stage_execution_time(
        guarded_input.second, guarded_input.first, frontier::StageId{1});
    expect(!guarded_prediction.scaled_moe_layer_prediction &&
               guarded_prediction.moe_routing.size() ==
                   guarded_prediction.logical_moe_layer_count,
           "layer-dependent routing must fall back to detailed stage timing");
    expect(diagnostic_value(guarded_prediction,
                            "stage_group_routing_layer_invariant") == 0.0,
           "routing guard diagnostic must identify layer-dependent routing");

    // A hybrid K3 stage keeps one representative expert lane (the scheduler
    // compressed-event contract) while grouping attention work independently
    // for KDA and MLA layers.
    const auto k3_model_config =
        frontier::config::load_model_config("moonshotai/Kimi-K3");
    parallelism.pipeline_parallel_size = 4;
    frontier::config::AnalyticalExecutionModelConfig k3_grouped_config{};
    k3_grouped_config.precision = "fp8";
    k3_grouped_config.moe_layer_event_mode = "stage_group_scaled";
    const predictor::AnalyticalRooflineExecutionTimePredictor k3_grouped{
        k3_grouped_config, parallelism, k3_model_config,
        frontier::config::MoeRoutingConfig{}};
    auto k3_input = make_batch();
    const auto k3_prediction = k3_grouped.predict_stage_execution_time(
        k3_input.second, k3_input.first, frontier::StageId{0});
    const auto k3_kda_group = std::find_if(
        k3_prediction.scaled_moe_attention_groups.begin(),
        k3_prediction.scaled_moe_attention_groups.end(), [](const auto &group) {
            return group.family == predictor::ScaledMoEAttentionFamily::kKda;
        });
    const auto k3_mla_group = std::find_if(
        k3_prediction.scaled_moe_attention_groups.begin(),
        k3_prediction.scaled_moe_attention_groups.end(), [](const auto &group) {
            return group.family == predictor::ScaledMoEAttentionFamily::kMla;
        });
    expect(k3_prediction.scaled_moe_layer_prediction &&
               k3_prediction.moe_routing.size() == 1 &&
               k3_kda_group !=
                   k3_prediction.scaled_moe_attention_groups.end() &&
               k3_mla_group != k3_prediction.scaled_moe_attention_groups.end(),
           "stage_group_scaled K3 must keep one lane representative and "
           "separate KDA/MLA attention groups");
}

void test_kimi_k3_stage_group_scaled_all_pp_stages() {
    const auto model =
        frontier::config::load_model_config("moonshotai/Kimi-K3");
    const auto make_batch = [] {
        RequestCollection requests;
        requests.emplace_back([&]() {
            WorkloadRequest value{};
            value.request_id = RequestId{0};
            value.session_start_at = SimTime::from_seconds(0.0);
            value.num_prefill_tokens = 128;
            value.num_decode_tokens = 1;
            return value;
        }());
        requests.front().on_arrival(SimTime::from_seconds(0.0));
        requests.front().on_admitted(SimTime::from_seconds(0.0));
        requests.front().advance_scheduler_frontier(128);
        RequestBatchSnapshot snapshot{};
        snapshot.request_id = RequestId{0};
        snapshot.scheduled_tokens = 128;
        snapshot.scheduler_frontier = 128;
        const Batch batch{BatchId{0},
                          IterationId{0},
                          {snapshot},
                          SimTime::from_seconds(0.0),
                          Generation{0}};
        return std::make_pair(std::move(requests), batch);
    };

    for (const std::uint64_t pp : {1ULL, 4ULL, 24ULL}) {
        frontier::config::ParallelismConfig parallelism{};
        parallelism.tensor_parallel_size = 4;
        parallelism.pipeline_parallel_size = pp;
        parallelism.moe_tensor_parallel_size = 1;
        parallelism.moe_expert_parallel_size = 16;

        frontier::config::AnalyticalExecutionModelConfig detailed_config{};
        detailed_config.precision = "fp8";
        auto grouped_config = detailed_config;
        grouped_config.moe_layer_event_mode = "stage_group_scaled";
        const predictor::AnalyticalRooflineExecutionTimePredictor detailed{
            detailed_config, parallelism, model,
            frontier::config::MoeRoutingConfig{}};
        const predictor::AnalyticalRooflineExecutionTimePredictor grouped{
            grouped_config, parallelism, model,
            frontier::config::MoeRoutingConfig{}};
        auto detailed_input = make_batch();
        auto grouped_input = make_batch();

        std::uint64_t total_kda = 0;
        std::uint64_t total_mla = 0;
        std::uint64_t total_moe = 0;
        std::vector<std::uint64_t> seen_timing_groups;
        for (std::uint64_t stage = 0; stage < pp; ++stage) {
            const auto layers = frontier::config::pipeline_stage_layer_range(
                model.num_layers, pp, stage);
            std::uint64_t expected_kda = 0;
            std::uint64_t expected_mla = 0;
            std::uint64_t expected_moe = 0;
            std::uint64_t repeated_kda = 0;
            std::uint64_t repeated_mla = 0;
            bool saw_first_moe = false;
            for (std::uint64_t layer = layers.begin; layer < layers.end;
                 ++layer) {
                expected_kda +=
                    static_cast<std::uint64_t>(model.is_kda_layer(layer));
                expected_mla +=
                    static_cast<std::uint64_t>(model.is_mla_layer(layer));
                if (!model.is_moe_layer(layer)) {
                    continue;
                }
                ++expected_moe;
                if (!saw_first_moe) {
                    saw_first_moe = true;
                } else if (model.is_kda_layer(layer)) {
                    ++repeated_kda;
                } else {
                    ++repeated_mla;
                }
            }
            total_kda += expected_kda;
            total_mla += expected_mla;
            total_moe += expected_moe;

            const auto detailed_prediction =
                detailed.predict_stage_execution_time(detailed_input.second,
                                                      detailed_input.first,
                                                      frontier::StageId{stage});
            const auto grouped_prediction =
                grouped.predict_stage_execution_time(grouped_input.second,
                                                     grouped_input.first,
                                                     frontier::StageId{stage});
            expect(detailed_prediction.execution_time ==
                       grouped_prediction.execution_time,
                   "K3 detailed and grouped PP stage times must match");
            expect(saw_first_moe &&
                       detailed_prediction.logical_moe_layer_count ==
                           expected_moe &&
                       grouped_prediction.logical_moe_layer_count ==
                           expected_moe &&
                       grouped_prediction.moe_routing.size() == 1 &&
                       grouped_prediction.scaled_moe_layer_prediction,
                   "K3 grouped PP stage must preserve its exact MoE count");
            expect(
                diagnostic_value(grouped_prediction, "kda_layer_count") ==
                        static_cast<double>(expected_kda) &&
                    diagnostic_value(grouped_prediction, "mla_layer_count") ==
                        static_cast<double>(expected_mla),
                "K3 grouped PP stage must preserve KDA/MLA counts");

            std::uint64_t actual_repeated_kda = 0;
            std::uint64_t actual_repeated_mla = 0;
            for (const auto &group :
                 grouped_prediction.scaled_moe_attention_groups) {
                if (group.family == predictor::ScaledMoEAttentionFamily::kKda) {
                    actual_repeated_kda += group.layer_count;
                } else if (group.family ==
                           predictor::ScaledMoEAttentionFamily::kMla) {
                    actual_repeated_mla += group.layer_count;
                }
            }
            expect(actual_repeated_kda == repeated_kda &&
                       actual_repeated_mla == repeated_mla,
                   "K3 grouped PP stage must preserve repeated family counts");

            const bool final_stage = stage + 1 == pp;
            expect((grouped_prediction.execution_time.pp_communication_ms >
                    0.0) == !final_stage,
                   "K3 PP sends must exist only on non-final stages");
            const auto timing_group = static_cast<std::uint64_t>(
                diagnostic_value(grouped_prediction, "timing_group_id"));
            const bool repeated_group =
                std::find(seen_timing_groups.begin(), seen_timing_groups.end(),
                          timing_group) != seen_timing_groups.end();
            const bool reusable_group =
                diagnostic_value(grouped_prediction,
                                 "timing_group_multiplicity") > 1.0;
            if (reusable_group) {
                expect(diagnostic_value(grouped_prediction,
                                        repeated_group
                                            ? "timing_cache_hit"
                                            : "timing_cache_miss") == 1.0,
                       "reusable K3 timing groups must miss once and then hit");
            } else {
                expect(diagnostic_value(grouped_prediction,
                                        "timing_cache_enabled") == 0.0 &&
                           diagnostic_value(grouped_prediction,
                                            "timing_cache_hit") == 0.0 &&
                           diagnostic_value(grouped_prediction,
                                            "timing_cache_miss") == 0.0,
                       "singleton K3 timing groups must bypass the cache");
            }
            if (!repeated_group) {
                seen_timing_groups.push_back(timing_group);
            }
        }
        expect(total_kda == 69 && total_mla == 24 && total_moe == 92,
               "K3 PP stages must reconstruct the complete hybrid model");
    }
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

void test_kda_roofline_components_and_fixed_context_cost() {
    analytical::DenseModel kda{};
    kda.hidden_size = 7'168;
    kda.intermediate_size = 33'792;
    kda.num_query_heads = 96;
    kda.num_kv_heads = 96;
    kda.head_dim = 128;
    kda.tensor_parallel_size = 1;
    kda.gated_mlp = true;
    kda.fused_add_norm = true;
    kda.use_kda = true;
    kda.kda_num_heads = 96;
    kda.kda_num_k_heads = 96;
    kda.kda_num_v_heads = 96;
    kda.kda_head_dim = 128;
    kda.kda_key_head_dim = 128;
    kda.kda_value_head_dim = 128;
    kda.kda_short_conv_kernel_size = 4;
    kda.kda_conv_state_dim = 36'864;
    kda.attn_res_block_size = 12;

    analytical::DenseBatch prefill{};
    prefill.total_tokens = 32;
    prefill.prefill_requests.push_back({32, 0});
    const analytical::DenseLayerTimes prefill_times =
        analytical::predict_dense_layer(analytical::DeviceCeilings::rubin(),
                                        analytical::AnalyticalConfig{}, kda,
                                        prefill, analytical::Precision::kFp16);
    expect(prefill_times.kda_projection_ms > 0.0 &&
               prefill_times.kda_short_conv_ms > 0.0 &&
               prefill_times.kda_recurrent_ms > 0.0 &&
               prefill_times.attn_res_ms == 0.0 &&
               prefill_times.kv_cache_save_ms == 0.0,
           "KDA roofline must expose projection, short-conv, and recurrent "
           "work while deliberately omitting AttnRes and sequence KV writes");

    analytical::DenseOperatorPrecisions split_precisions{};
    split_precisions.attention = analytical::Precision::kBf16;
    split_precisions.dense = analytical::Precision::kBf16;
    split_precisions.kv_cache = analytical::Precision::kFp8;
    split_precisions.kda_state = analytical::Precision::kFp32;
    split_precisions.attention_core = analytical::Precision::kFp8;
    const auto split_prefill = analytical::predict_dense_layer(
        analytical::DeviceCeilings::rubin(), analytical::AnalyticalConfig{},
        kda, prefill, split_precisions);
    split_precisions.attention_core = analytical::Precision::kBf16;
    const auto bf16_core_prefill = analytical::predict_dense_layer(
        analytical::DeviceCeilings::rubin(), analytical::AnalyticalConfig{},
        kda, prefill, split_precisions);
    expect_approximately_equal(
        split_prefill.total_ms(), bf16_core_prefill.total_ms(),
        "KDA must ignore sequence attention core precision");

    analytical::DenseBatch short_decode{};
    short_decode.total_tokens = 1;
    short_decode.decode_requests.push_back({1, 32});
    analytical::DenseBatch long_decode = short_decode;
    long_decode.decode_requests.front().past_context = 16'384;
    const auto short_times = analytical::predict_dense_layer(
        analytical::DeviceCeilings::rubin(), analytical::AnalyticalConfig{},
        kda, short_decode, analytical::Precision::kFp16);
    const auto long_times = analytical::predict_dense_layer(
        analytical::DeviceCeilings::rubin(), analytical::AnalyticalConfig{},
        kda, long_decode, analytical::Precision::kFp16);
    expect_approximately_equal(
        short_times.decode_attention_ms, long_times.decode_attention_ms,
        "KDA decode recurrent cost must not grow with past context");

    analytical::AnalyticalConfig fused_decode =
        analytical::analytical_config_from_profile("k3_sglang_mxfp4",
                                                   "gb300");
    analytical::AnalyticalConfig serial_decode = fused_decode;
    serial_decode.fuse_kda_decode_chain = false;
    serial_decode.overlap_kda_aux_projections = false;
    const auto fused_kda = analytical::predict_dense_layer(
        analytical::DeviceCeilings::gb300(), fused_decode, kda, short_decode,
        analytical::Precision::kBf16);
    const auto serial_kda = analytical::predict_dense_layer(
        analytical::DeviceCeilings::gb300(), serial_decode, kda, short_decode,
        analytical::Precision::kBf16);
    expect(fused_kda.total_ms() < serial_kda.total_ms(),
           "K3 must overlap auxiliary KDA projections and fuse the decode "
           "conv/recurrent/gated-norm chain");
    expect(fused_kda.kda_short_conv_ms == 0.0 &&
               fused_kda.kda_gate_norm_ms == 0.0 &&
               serial_kda.kda_short_conv_ms > 0.0 &&
               serial_kda.kda_gate_norm_ms > 0.0,
           "fused KDA decode must expose one recurrent critical-path bucket");

    const auto fused_prefill = analytical::predict_dense_layer(
        analytical::DeviceCeilings::gb300(), fused_decode, kda, prefill,
        analytical::Precision::kBf16);
    const auto serial_prefill = analytical::predict_dense_layer(
        analytical::DeviceCeilings::gb300(), serial_decode, kda, prefill,
        analytical::Precision::kBf16);
    expect_approximately_equal(
        fused_prefill.total_ms(), serial_prefill.total_ms(),
        "decode-only KDA schedules must not alter prefill");

    auto tp8_kda = kda;
    tp8_kda.tensor_parallel_size = 8;
    const auto tp8_times = analytical::predict_dense_layer(
        analytical::DeviceCeilings::rubin(), analytical::AnalyticalConfig{},
        tp8_kda, long_decode, analytical::Precision::kFp16);
    tp8_kda.decode_context_parallel_size = 8;
    const auto tp8_dcp8_times = analytical::predict_dense_layer(
        analytical::DeviceCeilings::rubin(), analytical::AnalyticalConfig{},
        tp8_kda, long_decode, analytical::Precision::kFp16);
    expect_approximately_equal(tp8_dcp8_times.total_ms(), tp8_times.total_ms(),
                               "KDA TP8 total time must ignore DCP8");
    expect_approximately_equal(tp8_dcp8_times.kda_projection_ms,
                               tp8_times.kda_projection_ms,
                               "KDA TP8 projection time must ignore DCP8");
    expect_approximately_equal(tp8_dcp8_times.kda_recurrent_ms,
                               tp8_times.kda_recurrent_ms,
                               "KDA TP8 recurrent time must ignore DCP8");

    auto asymmetric_heads = kda;
    asymmetric_heads.kda_num_k_heads = 48;
    expect_throws<analytical::AnalyticalModelError>(
        [&asymmetric_heads, &prefill] {
            static_cast<void>(analytical::predict_dense_layer(
                analytical::DeviceCeilings::rubin(),
                analytical::AnalyticalConfig{}, asymmetric_heads, prefill,
                analytical::Precision::kFp16));
        },
        "KDA roofline must reject asymmetric Q/K/V head counts");

    auto asymmetric_dimensions = kda;
    asymmetric_dimensions.kda_value_head_dim = 64;
    expect_throws<analytical::AnalyticalModelError>(
        [&asymmetric_dimensions, &prefill] {
            static_cast<void>(analytical::predict_dense_layer(
                analytical::DeviceCeilings::rubin(),
                analytical::AnalyticalConfig{}, asymmetric_dimensions, prefill,
                analytical::Precision::kFp16));
        },
        "KDA roofline must reject asymmetric Q/K/V head dimensions");
}

} // namespace

int main() {
    int failures = 0;
    failures += frontier::test::run("roofline matches Python golden",
                                    test_roofline_matches_python_golden);
    failures += frontier::test::run("device presets and overrides",
                                    test_device_presets_and_overrides);
    failures += frontier::test::run("dense layer matches C++ golden",
                                    test_dense_layer_matches_cpp_golden);
    failures += frontier::test::run("long-context decode cost increases",
                                    test_long_context_decode_cost_increases);
    failures +=
        frontier::test::run("PREFILL attention token pairs are exact",
                            test_prefill_attention_token_pairs_are_exact);
    failures +=
        frontier::test::run("operator precisions split dense and KV costs",
                            test_operator_precisions_split_dense_and_kv_costs);
    failures += frontier::test::run(
        "operator precisions split MoE expert and router costs",
        test_operator_precisions_split_moe_expert_and_router_costs);
    failures += frontier::test::run(
        "router storage and compute precisions are independent",
        test_router_storage_and_compute_precisions_are_independent);
    failures += frontier::test::run(
        "GEMM efficiency is continuous and shape-aware",
        test_gemm_efficiency_is_shape_continuous);
    failures += frontier::test::run(
        "K3 LatentMoE front is one GEMM",
        test_k3_latent_moe_front_is_one_gemm);
    failures += frontier::test::run("MLA uses latent-cache context costs",
                                    test_mla_uses_latent_cache_context_costs);
    failures += frontier::test::run("MLA KV layout splits latent and RoPE",
                                    test_mla_kv_layout_splits_latent_and_rope);
    failures += frontier::test::run("MLA DCP shards decode context work",
                                    test_mla_dcp_shards_decode_context_work);
    failures += frontier::test::run("MLA output gate and NoPE costs",
                                    test_mla_output_gate_and_nope_costs);
    failures += frontier::test::run("MFA models shared-Q projection path",
                                    test_mfa_models_shared_q_projection_path);
    failures += frontier::test::run("batch model matches C++ golden",
                                    test_batch_model_matches_cpp_golden);
    failures += frontier::test::run("communication matches Python golden",
                                    test_communication_matches_python_golden);
    failures += frontier::test::run("KV transfer matches Python golden",
                                    test_kv_transfer_matches_python_golden);
    failures += frontier::test::run("Kimi K2 uneven pipeline and LM head",
                                    test_kimi_k2_uneven_pipeline_and_lm_head);
    failures += frontier::test::run("Kimi K2 DCP decode collectives",
                                    test_kimi_k2_dcp_adds_decode_collectives);
    failures += frontier::test::run("Kimi K3 TP8 DCP8 hybrid decode",
                                    test_kimi_k3_tp8_dcp8_hybrid_decode);
    failures += frontier::test::run("Kimi K2 first-layer-scaled prediction",
                                    test_kimi_k2_first_layer_scaled_prediction);
    failures +=
        frontier::test::run("Kimi K3 attention-family-scaled prediction",
                            test_kimi_k3_attention_family_scaled_prediction);
    failures += frontier::test::run(
        "stage-group scaled PP signatures and routing guard",
        test_stage_group_scaled_pp_signatures_and_routing_guard);
    failures +=
        frontier::test::run("Kimi K3 stage-group scaled all PP stages",
                            test_kimi_k3_stage_group_scaled_all_pp_stages);
    failures +=
        frontier::test::run("invalid analytical inputs are rejected",
                            test_invalid_analytical_inputs_are_rejected);
    failures += frontier::test::run(
        "KDA roofline components and fixed context cost",
        test_kda_roofline_components_and_fixed_context_cost);
    return failures == 0 ? 0 : 1;
}
