#include "frontier/kv_cache_transfer/analytical_transfer.h"

#include <cmath>
#include <limits>
#include <utility>

namespace frontier::kv_cache_transfer {

std::uint64_t dense_kv_cache_size_bytes(std::uint64_t num_tokens,
                                        const DenseKvLayout &layout) {
    if (layout.num_layers == 0 || layout.num_kv_heads_per_worker == 0 ||
        layout.head_dim == 0 || layout.kv_factor == 0) {
        throw TransferModelError("dense KV layout dimensions must be positive");
    }
    if (!std::isfinite(layout.dtype_size_bytes) ||
        layout.dtype_size_bytes <= 0.0) {
        throw TransferModelError("KV dtype size must be finite and positive");
    }

    const long double size =
        static_cast<long double>(num_tokens) *
        static_cast<long double>(layout.num_layers) *
        static_cast<long double>(layout.num_kv_heads_per_worker) *
        static_cast<long double>(layout.head_dim) *
        static_cast<long double>(layout.kv_factor) *
        static_cast<long double>(layout.dtype_size_bytes);
    if (!std::isfinite(size) ||
        size > static_cast<long double>(
                   std::numeric_limits<std::uint64_t>::max())) {
        throw TransferModelError("dense KV cache size overflows uint64");
    }
    return static_cast<std::uint64_t>(size);
}

TransferPrediction predict_transfer(std::uint64_t size_bytes,
                                    const TransferConfig &config) {
    if (!std::isfinite(config.network_bandwidth_gbps) ||
        config.network_bandwidth_gbps <= 0.0) {
        throw TransferModelError(
            "transfer bandwidth must be finite and positive");
    }
    if (!std::isfinite(config.network_latency_ms) ||
        config.network_latency_ms < 0.0) {
        throw TransferModelError(
            "transfer latency must be finite and nonnegative");
    }
    if (!std::isfinite(config.compression_ratio) ||
        config.compression_ratio <= 0.0) {
        throw TransferModelError(
            "compression ratio must be finite and positive");
    }

    const double effective_size =
        config.enable_compression
            ? static_cast<double>(size_bytes) / config.compression_ratio
            : static_cast<double>(size_bytes);
    const double bandwidth_bytes_per_ms =
        config.network_bandwidth_gbps * 1e9 / (8.0 * 1e3);
    return [&]() {
        TransferPrediction value{};
        value.size_bytes = size_bytes;
        value.effective_size_bytes = effective_size;
        value.transfer_time_ms =
            config.network_latency_ms + effective_size / bandwidth_bytes_per_ms;
        return value;
    }();
}

AnalyticalKVCacheTransferPredictor::AnalyticalKVCacheTransferPredictor(
    config::KvCacheTransferConfig config)
    : config_(std::move(config)) {}

TransferPrediction AnalyticalKVCacheTransferPredictor::predict(
    std::uint64_t num_tokens, const config::ModelConfig &model) const {
    const std::uint64_t size_bytes =
        dense_kv_cache_size_bytes(num_tokens, [&]() {
            DenseKvLayout value{};
            value.num_layers = model.num_layers;
            value.num_kv_heads_per_worker = model.num_kv_heads;
            value.head_dim = model.head_dim;
            value.kv_factor = 2;
            value.dtype_size_bytes = config_.kv_cache_dtype_size_bytes;
            return value;
        }());
    return predict_transfer(size_bytes, [&]() {
        TransferConfig value{};
        value.network_bandwidth_gbps = config_.network_bandwidth_gbps;
        value.network_latency_ms = config_.network_latency_ms;
        value.enable_compression = config_.enable_compression;
        value.compression_ratio = 1.0;
        return value;
    }());
}

std::shared_ptr<const BaseKVCacheTransferPredictor>
make_kv_cache_transfer_predictor(const config::KvCacheTransferConfig &config) {
    return std::make_shared<AnalyticalKVCacheTransferPredictor>(config);
}

} // namespace frontier::kv_cache_transfer
