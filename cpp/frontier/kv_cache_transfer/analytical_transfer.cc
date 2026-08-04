#include "frontier/kv_cache_transfer/analytical_transfer.h"

#include <cmath>
#include <limits>
#include <utility>

#include "frontier/attention/mla.h"

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

std::uint64_t model_kv_cache_size_bytes(std::uint64_t num_tokens,
                                        const config::ModelConfig &model,
                                        double kv_cache_dtype_size_bytes) {
    return model_kv_cache_size_bytes(num_tokens, model,
                                     kv_cache_dtype_size_bytes, 1);
}

std::uint64_t model_kv_cache_size_bytes(
    std::uint64_t num_tokens, const config::ModelConfig &model,
    double kv_cache_dtype_size_bytes,
    std::uint64_t attention_tensor_parallel_size) {
    if (!std::isfinite(kv_cache_dtype_size_bytes) ||
        kv_cache_dtype_size_bytes <= 0.0) {
        throw TransferModelError("KV dtype size must be finite and positive");
    }
    if (attention_tensor_parallel_size == 0) {
        throw TransferModelError(
            "attention tensor parallel size must be positive");
    }

    const std::uint64_t one_copy_bytes = [&]() {
        if (model.use_mla) {
            try {
                return attention::mla_kv_cache_size_bytes(
                    num_tokens, model.num_layers,
                    attention::MlaKvCacheLayout{
                        model.kv_lora_rank,
                        model.qk_rope_head_dim,
                        kv_cache_dtype_size_bytes,
                        2.0,
                    });
            } catch (const attention::MlaLayoutError &error) {
                throw TransferModelError(error.what());
            }
        }
        return dense_kv_cache_size_bytes(num_tokens, [&]() {
            DenseKvLayout value{};
            value.num_layers = model.num_layers;
            value.num_kv_heads_per_worker = model.runtime_num_kv_heads();
            value.head_dim = model.runtime_head_size();
            value.kv_factor = model.kv_factor();
            value.dtype_size_bytes = kv_cache_dtype_size_bytes;
            return value;
        }());
    }();

    // MLA stores complete latent KV on every attention TP rank.  Keep the
    // existing dense/GQA/MQA accounting unchanged.
    if (!model.use_mla || attention_tensor_parallel_size == 1) {
        return one_copy_bytes;
    }
    if (one_copy_bytes >
        std::numeric_limits<std::uint64_t>::max() /
            attention_tensor_parallel_size) {
        throw TransferModelError(
            "target-physical KV cache size overflows uint64");
    }
    return one_copy_bytes * attention_tensor_parallel_size;
}

std::uint64_t model_kv_cache_size_bytes_one_copy(
    std::uint64_t num_tokens, const config::ModelConfig &model,
    double kv_cache_dtype_size_bytes) {
    return model_kv_cache_size_bytes(num_tokens, model,
                                     kv_cache_dtype_size_bytes, 1);
}

std::uint64_t model_kv_cache_size_bytes_target_physical(
    std::uint64_t num_tokens, const config::ModelConfig &model,
    double kv_cache_dtype_size_bytes,
    std::uint64_t attention_tensor_parallel_size) {
    return model_kv_cache_size_bytes(num_tokens, model,
                                     kv_cache_dtype_size_bytes,
                                     attention_tensor_parallel_size);
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
    config::KvCacheTransferConfig config,
    std::uint64_t attention_tensor_parallel_size)
    : config_(std::move(config)),
      attention_tensor_parallel_size_(attention_tensor_parallel_size) {
    if (attention_tensor_parallel_size_ == 0) {
        throw TransferModelError(
            "attention tensor parallel size must be positive");
    }
}

TransferPrediction AnalyticalKVCacheTransferPredictor::predict(
    std::uint64_t num_tokens, const config::ModelConfig &model) const {
    const std::uint64_t size_bytes =
        model_kv_cache_size_bytes_target_physical(
            num_tokens, model, config_.kv_cache_dtype_size_bytes,
            attention_tensor_parallel_size_);
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
make_kv_cache_transfer_predictor(
    const config::KvCacheTransferConfig &config,
    std::uint64_t attention_tensor_parallel_size) {
    return std::make_shared<AnalyticalKVCacheTransferPredictor>(
        config, attention_tensor_parallel_size);
}

} // namespace frontier::kv_cache_transfer
