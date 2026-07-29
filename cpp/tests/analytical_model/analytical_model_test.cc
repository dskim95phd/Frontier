#include "frontier/cc_backend/analytical_model.h"
#include "frontier/execution_time_predictor/analytical_model.h"
#include "frontier/kv_cache_transfer/analytical_transfer.h"
#include "tests/test_support.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <string>

#include <nlohmann/json.hpp>

#ifndef FRONTIER_TEST_FIXTURE_DIR
#error "FRONTIER_TEST_FIXTURE_DIR must be defined for analytical tests"
#endif

namespace {

namespace analytical = frontier::execution_time_predictor;
namespace communication = frontier::cc_backend;
namespace kv_transfer = frontier::kv_cache_transfer;

using frontier::test::expect;
using frontier::test::expect_throws;
using frontier::test::read_text_file;
using Json = nlohmann::json;

bool approximately_equal(
    double actual,
    double expected,
    double relative_tolerance = 1e-12,
    double absolute_tolerance = 1e-12) {
  const double difference = std::abs(actual - expected);
  const double scale = std::max(std::abs(actual), std::abs(expected));
  return difference <=
         std::max(absolute_tolerance, relative_tolerance * scale);
}

void expect_approximately_equal(
    double actual,
    double expected,
    std::string_view field) {
  if (!approximately_equal(actual, expected)) {
    throw std::runtime_error(
        std::string{field} + " mismatch: actual=" +
        std::to_string(actual) + ", expected=" +
        std::to_string(expected));
  }
}

Json load_golden() {
  const std::filesystem::path path =
      std::filesystem::path{FRONTIER_TEST_FIXTURE_DIR} /
      "analytical/analytical_v1.json";
  return Json::parse(read_text_file(path));
}

void test_roofline_matches_python_golden() {
  const Json golden = load_golden();
  expect(
      golden.at("schema_version").get<int>() == 1,
      "analytical golden schema must be version 1");
  const Json& input = golden.at("roofline").at("input");
  const Json& expected = golden.at("roofline").at("expected");

  const analytical::RooflineResult result =
      analytical::predict_roofline(
          analytical::DeviceCeilings::rubin(),
          analytical::Precision::kFp16,
          analytical::KernelWork{
              .flops = input.at("flops").get<double>(),
              .hbm_bytes = input.at("hbm_bytes").get<double>(),
          },
          analytical::Efficiency{
              .compute =
                  input.at("compute_efficiency").get<double>(),
              .memory =
                  input.at("memory_efficiency").get<double>(),
              .overlap_penalty =
                  input.at("overlap_penalty").get<double>(),
          },
          input.at("kernel_launch_latency_us").get<double>());

  expect_approximately_equal(
      result.compute_time_ms,
      expected.at("compute_time_ms").get<double>(),
      "roofline.compute_time_ms");
  expect_approximately_equal(
      result.memory_time_ms,
      expected.at("memory_time_ms").get<double>(),
      "roofline.memory_time_ms");
  expect_approximately_equal(
      result.launch_time_ms,
      expected.at("launch_time_ms").get<double>(),
      "roofline.launch_time_ms");
  expect_approximately_equal(
      result.predicted_time_ms,
      expected.at("predicted_time_ms").get<double>(),
      "roofline.predicted_time_ms");
  expect(
      result.bottleneck == analytical::Bottleneck::kHbm,
      "roofline bottleneck must match Python");
}

void check_dense_fields(
    const analytical::DenseLayerTimes& actual,
    const Json& expected) {
  for (const auto& [name, value] : {
           std::pair{
               "attention_pre_projection_ms",
               actual.attention_pre_projection_ms},
           std::pair{
               "attention_post_projection_ms",
               actual.attention_post_projection_ms},
           std::pair{"rope_ms", actual.rope_ms},
           std::pair{"kv_cache_save_ms", actual.kv_cache_save_ms},
           std::pair{"attention_norm_ms", actual.attention_norm_ms},
           std::pair{
               "prefill_attention_ms",
               actual.prefill_attention_ms},
           std::pair{
               "decode_attention_ms",
               actual.decode_attention_ms},
           std::pair{
               "mlp_up_projection_ms",
               actual.mlp_up_projection_ms},
           std::pair{
               "mlp_activation_ms",
               actual.mlp_activation_ms},
           std::pair{
               "mlp_down_projection_ms",
               actual.mlp_down_projection_ms},
           std::pair{"mlp_norm_ms", actual.mlp_norm_ms},
           std::pair{"residual_add_ms", actual.residual_add_ms},
       }) {
    expect_approximately_equal(
        value,
        expected.at(name).get<double>(),
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

  for (const Json& test_case : golden.at("dense_cases")) {
    const Json& input = test_case.at("input");
    const std::uint64_t tokens =
        input.at("scheduled_tokens").get<std::uint64_t>();
    const analytical::AttentionRequestSlice request{
        .query_tokens = tokens,
        .past_context =
            input.at("past_context").get<std::uint64_t>(),
    };
    analytical::DenseBatch batch{
        .total_tokens = tokens,
        .prefill_requests = {},
        .decode_requests = {},
    };
    if (input.at("prefill").get<bool>()) {
      batch.prefill_requests.push_back(request);
    } else {
      batch.decode_requests.push_back(request);
    }

    const analytical::DenseLayerTimes result =
        analytical::predict_dense_layer(
            device,
            config,
            model,
            batch,
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
        device,
        config,
        model,
        analytical::DenseBatch{
            .total_tokens = 1,
            .prefill_requests = {},
            .decode_requests = {
                analytical::AttentionRequestSlice{
                    .query_tokens = 1,
                    .past_context = past_context,
                },
            },
        },
        analytical::Precision::kFp16);
  };
  expect(
      predict(4'096).decode_attention_ms >
          predict(128).decode_attention_ms,
      "long-context decode attention must cost more");
}

void test_communication_matches_python_golden() {
  const Json section = load_golden().at("communication");
  const Json& input = section.at("input");
  const Json& expected = section.at("expected");
  const communication::AnalyticalCommunicationModel model{
      communication::AnalyticalCommunicationConfig{
          .network_bandwidth_gbps =
              input.at("network_bandwidth_gbps").get<double>(),
          .latency_us = input.at("latency_us").get<double>(),
          .intra_node_bandwidth_gbps =
              input.at("intra_node_bandwidth_gbps").get<double>(),
      }};
  const std::uint64_t bytes =
      input.at("data_size_bytes").get<std::uint64_t>();
  const std::uint64_t devices =
      input.at("num_devices").get<std::uint64_t>();

  for (const auto& [field, actual] : {
           std::pair{
               "point_to_point_ms",
               model.point_to_point_ms(bytes)},
           std::pair{
               "allreduce_ms",
               model.allreduce_ms(bytes, devices)},
           std::pair{
               "allgather_ms",
               model.allgather_ms(bytes, devices)},
           std::pair{
               "broadcast_ms",
               model.broadcast_ms(bytes, devices)},
           std::pair{
               "reduce_scatter_ms",
               model.reduce_scatter_ms(bytes, devices)},
           std::pair{
               "all_to_all_ms",
               model.all_to_all_ms(bytes, devices)},
       }) {
    expect_approximately_equal(
        actual,
        expected.at(field).get<double>(),
        field);
  }
  expect(
      model.allreduce_ms(bytes, 1) == 0.0,
      "single-device collective must be free");
  expect_approximately_equal(
      model.point_to_point_ms(0),
      input.at("latency_us").get<double>() / 1e3,
      "zero-byte point-to-point latency");
  expect_approximately_equal(
      model.allreduce_ms(0, devices),
      input.at("latency_us").get<double>() / 1e3,
      "zero-byte multi-device collective latency");
  expect(
      devices == 72,
      "golden collective must exercise the NVL72 participant boundary");
}

void test_kv_transfer_matches_python_golden() {
  const Json section = load_golden().at("kv_transfer");
  const Json& input = section.at("input");
  const Json& expected = section.at("expected");
  const kv_transfer::DenseKvLayout layout{
      .num_layers = input.at("num_layers").get<std::uint64_t>(),
      .num_kv_heads_per_worker =
          input.at("num_kv_heads_per_worker").get<std::uint64_t>(),
      .head_dim = input.at("head_dim").get<std::uint64_t>(),
      .kv_factor = input.at("kv_factor").get<std::uint64_t>(),
      .dtype_size_bytes =
          input.at("dtype_size_bytes").get<double>(),
  };
  const std::uint64_t size = kv_transfer::dense_kv_cache_size_bytes(
      input.at("num_tokens").get<std::uint64_t>(),
      layout);
  expect(
      size == expected.at("size_bytes").get<std::uint64_t>(),
      "dense KV size must match Python");

  const kv_transfer::TransferPrediction prediction =
      kv_transfer::predict_transfer(
          size,
          kv_transfer::TransferConfig{
              .network_bandwidth_gbps =
                  input.at("network_bandwidth_gbps").get<double>(),
              .network_latency_ms =
                  input.at("network_latency_ms").get<double>(),
          });
  expect_approximately_equal(
      prediction.transfer_time_ms,
      expected.at("transfer_time_ms").get<double>(),
      "kv_transfer.transfer_time_ms");

  expect(
      kv_transfer::dense_kv_cache_size_bytes(0, layout) == 0,
      "zero-token KV size must be zero");
  const kv_transfer::TransferPrediction empty_prediction =
      kv_transfer::predict_transfer(
          0,
          kv_transfer::TransferConfig{
              .network_bandwidth_gbps =
                  input.at("network_bandwidth_gbps").get<double>(),
              .network_latency_ms =
                  input.at("network_latency_ms").get<double>(),
          });
  expect_approximately_equal(
      empty_prediction.transfer_time_ms,
      input.at("network_latency_ms").get<double>(),
      "zero-byte KV transfer latency");
}

void test_invalid_analytical_inputs_are_rejected() {
  expect_throws<analytical::AnalyticalModelError>(
      [] {
        static_cast<void>(analytical::predict_roofline(
            analytical::DeviceCeilings::rubin(),
            analytical::Precision::kFp16,
            analytical::KernelWork{.flops = -1.0, .hbm_bytes = 0.0},
            analytical::Efficiency{0.5, 0.5, 0.0},
            1.0));
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
        static_cast<void>(kv_transfer::dense_kv_cache_size_bytes(
            1,
            kv_transfer::DenseKvLayout{
                .num_layers = 0,
                .num_kv_heads_per_worker = 1,
                .head_dim = 1,
            }));
      },
      "invalid KV layout must be rejected");
}

}  // namespace

int main() {
  int failures = 0;
  failures += frontier::test::run(
      "roofline matches Python golden",
      test_roofline_matches_python_golden);
  failures += frontier::test::run(
      "dense layer matches Python golden",
      test_dense_layer_matches_python_golden);
  failures += frontier::test::run(
      "long-context decode cost increases",
      test_long_context_decode_cost_increases);
  failures += frontier::test::run(
      "communication matches Python golden",
      test_communication_matches_python_golden);
  failures += frontier::test::run(
      "KV transfer matches Python golden",
      test_kv_transfer_matches_python_golden);
  failures += frontier::test::run(
      "invalid analytical inputs are rejected",
      test_invalid_analytical_inputs_are_rejected);
  return failures == 0 ? 0 : 1;
}
