#pragma once

#include <cstdint>
#include <stdexcept>

namespace frontier::kv_cache_transfer {

struct DenseKvLayout {
  std::uint64_t num_layers;
  std::uint64_t num_kv_heads_per_worker;
  std::uint64_t head_dim;
  std::uint64_t kv_factor = 2;
  double dtype_size_bytes = 2.0;
};

struct TransferConfig {
  double network_bandwidth_gbps;
  double network_latency_ms;
  bool enable_compression = false;
  double compression_ratio = 1.0;
};

struct TransferPrediction {
  std::uint64_t size_bytes;
  double effective_size_bytes;
  double transfer_time_ms;
};

class TransferModelError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

[[nodiscard]] std::uint64_t dense_kv_cache_size_bytes(
    std::uint64_t num_tokens,
    const DenseKvLayout& layout);
[[nodiscard]] TransferPrediction predict_transfer(
    std::uint64_t size_bytes,
    const TransferConfig& config);

}  // namespace frontier::kv_cache_transfer
