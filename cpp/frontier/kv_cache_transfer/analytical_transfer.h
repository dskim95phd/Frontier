#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>

#include "frontier/config/config.h"
#include "frontier/kv_cache_transfer/base_kv_cache_transfer_predictor.h"

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

class TransferModelError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

[[nodiscard]] std::uint64_t
dense_kv_cache_size_bytes(std::uint64_t num_tokens,
                          const DenseKvLayout &layout);
[[nodiscard]] std::uint64_t
model_kv_cache_size_bytes(std::uint64_t num_tokens,
                          const config::ModelConfig &model,
                          double kv_cache_dtype_size_bytes);
// Return model KV bytes for a target with the requested attention TP size.
// Dense/GQA/MQA layouts retain their existing sharding contract.  MLA's
// latent KV is replicated on every attention TP rank, so target-physical
// bytes are one-copy bytes multiplied by attention_tensor_parallel_size.
[[nodiscard]] std::uint64_t
model_kv_cache_size_bytes(std::uint64_t num_tokens,
                          const config::ModelConfig &model,
                          double kv_cache_dtype_size_bytes,
                          std::uint64_t attention_tensor_parallel_size);
[[nodiscard]] std::uint64_t model_kv_cache_size_bytes_one_copy(
    std::uint64_t num_tokens, const config::ModelConfig &model,
    double kv_cache_dtype_size_bytes);
[[nodiscard]] std::uint64_t model_kv_cache_size_bytes_target_physical(
    std::uint64_t num_tokens, const config::ModelConfig &model,
    double kv_cache_dtype_size_bytes,
    std::uint64_t attention_tensor_parallel_size);
[[nodiscard]] TransferPrediction predict_transfer(std::uint64_t size_bytes,
                                                  const TransferConfig &config);

class AnalyticalKVCacheTransferPredictor final
    : public BaseKVCacheTransferPredictor {
  public:
    explicit AnalyticalKVCacheTransferPredictor(
        config::KvCacheTransferConfig config,
        std::uint64_t attention_tensor_parallel_size = 1);

    [[nodiscard]] TransferPrediction
    predict(std::uint64_t num_tokens,
            const config::ModelConfig &model) const override;

  private:
    config::KvCacheTransferConfig config_;
    std::uint64_t attention_tensor_parallel_size_ = 1;
};

[[nodiscard]] std::shared_ptr<const BaseKVCacheTransferPredictor>
make_kv_cache_transfer_predictor(
    const config::KvCacheTransferConfig &config,
    std::uint64_t attention_tensor_parallel_size = 1);

} // namespace frontier::kv_cache_transfer
