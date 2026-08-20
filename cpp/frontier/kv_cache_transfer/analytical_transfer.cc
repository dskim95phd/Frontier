#include "frontier/kv_cache_transfer/analytical_transfer.h"

#include <cmath>
#include <limits>
#include <utility>

#include "frontier/attention/mla.h"
#include "frontier/core/checked_math.h"

namespace frontier::kv_cache_transfer {

namespace {

std::uint64_t mla_cache_layer_count(const config::ModelConfig &model) {
    const std::uint64_t layers =
        model.has_kda() ? model.num_mla_layers : model.num_layers;
    if (layers == 0) {
        throw TransferModelError(
            "MLA model must expose at least one full-attention layer");
    }
    return layers;
}

std::uint64_t checked_state_size(long double elements,
                                 double dtype_size_bytes) {
    if (!std::isfinite(dtype_size_bytes) || dtype_size_bytes <= 0.0) {
        throw TransferModelError(
            "KDA state dtype size must be finite and positive");
    }
    const long double bytes =
        elements * static_cast<long double>(dtype_size_bytes);
    if (!std::isfinite(bytes) || bytes < 0.0L ||
        bytes > static_cast<long double>(
                    std::numeric_limits<std::uint64_t>::max())) {
        throw TransferModelError("KDA state snapshot size overflows uint64");
    }
    return static_cast<std::uint64_t>(std::ceil(bytes));
}

void validate_stage_range(const config::ModelConfig &model,
                          const config::PipelineStageLayerRange &layers) {
    if (layers.begin > layers.end || layers.end > model.num_layers) {
        throw TransferModelError("pipeline stage layer range is out of bounds");
    }
}

std::uint64_t
count_stage_kv_layers(const config::ModelConfig &model,
                      const config::PipelineStageLayerRange &layers) {
    validate_stage_range(model, layers);
    std::uint64_t count = 0;
    for (std::uint64_t layer = layers.begin; layer < layers.end; ++layer) {
        // KDA's recurrent state is not token-proportional KV.  For MLA
        // checkpoints only explicitly marked MLA layers own latent KV; for a
        // dense/KDA checkpoint every non-KDA layer retains the dense KV path.
        const bool kv_bearing = model.use_mla ? model.is_mla_layer(layer)
                                              : !model.is_kda_layer(layer);
        count += static_cast<std::uint64_t>(kv_bearing);
    }
    return count;
}

std::uint64_t
count_stage_kda_layers(const config::ModelConfig &model,
                       const config::PipelineStageLayerRange &layers) {
    validate_stage_range(model, layers);
    if (!model.has_kda()) {
        return 0;
    }
    if (model.kda_layer_indices.empty()) {
        // A KDA-only synthetic model may omit an explicit index list.  A
        // partially hybrid model cannot be partitioned safely without exact
        // indices, so fail rather than manufacture a positional distribution.
        if (model.num_kda_layers == model.num_layers) {
            return layers.size();
        }
        throw TransferModelError(
            "KDA stage accounting requires explicit kda_layer_indices for a "
            "partially hybrid model");
    }
    std::uint64_t count = 0;
    for (std::uint64_t layer = layers.begin; layer < layers.end; ++layer) {
        count += static_cast<std::uint64_t>(model.is_kda_layer(layer));
    }
    return count;
}

std::uint64_t checked_ceil_div(std::uint64_t numerator,
                               std::uint64_t denominator) {
    return checked_math::ceil_div<TransferModelError>(
        numerator, denominator, "division by zero in stage memory layout");
}

} // namespace

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

std::uint64_t
model_kv_cache_size_bytes(std::uint64_t num_tokens,
                          const config::ModelConfig &model,
                          double kv_cache_dtype_size_bytes,
                          std::uint64_t attention_tensor_parallel_size) {
    return model_kv_cache_size_bytes(num_tokens, model,
                                     kv_cache_dtype_size_bytes,
                                     attention_tensor_parallel_size, 1);
}

std::uint64_t
model_kv_cache_size_bytes(std::uint64_t num_tokens,
                          const config::ModelConfig &model,
                          double kv_cache_dtype_size_bytes,
                          std::uint64_t attention_tensor_parallel_size,
                          std::uint64_t decode_context_parallel_size) {
    if (!std::isfinite(kv_cache_dtype_size_bytes) ||
        kv_cache_dtype_size_bytes <= 0.0) {
        throw TransferModelError("KV dtype size must be finite and positive");
    }
    if (attention_tensor_parallel_size == 0) {
        throw TransferModelError(
            "attention tensor parallel size must be positive");
    }
    if (decode_context_parallel_size == 0 ||
        attention_tensor_parallel_size % decode_context_parallel_size != 0) {
        throw TransferModelError(
            "attention tensor parallel size must be divisible by decode "
            "context parallel size");
    }
    if (decode_context_parallel_size > 1 && !model.use_mla) {
        throw TransferModelError(
            "decode context parallel KV layout is currently supported only "
            "for MLA models");
    }

    const std::uint64_t one_copy_bytes = [&]() {
        if (model.use_mla) {
            try {
                return attention::mla_kv_cache_size_bytes(
                    num_tokens, mla_cache_layer_count(model),
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

    // Token-interleaved DCP forms TP/DCP independent MLA KV copies. Summing
    // all ranks is exact even when an individual rank owns one extra token.
    if (!model.use_mla || attention_tensor_parallel_size == 1) {
        return one_copy_bytes;
    }
    const std::uint64_t copies =
        attention_tensor_parallel_size / decode_context_parallel_size;
    if (one_copy_bytes > std::numeric_limits<std::uint64_t>::max() / copies) {
        throw TransferModelError(
            "target-physical KV cache size overflows uint64");
    }
    return one_copy_bytes * copies;
}

std::uint64_t
model_kv_cache_size_bytes_one_copy(std::uint64_t num_tokens,
                                   const config::ModelConfig &model,
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

std::uint64_t model_kv_cache_size_bytes_target_physical(
    std::uint64_t num_tokens, const config::ModelConfig &model,
    double kv_cache_dtype_size_bytes,
    std::uint64_t attention_tensor_parallel_size,
    std::uint64_t decode_context_parallel_size) {
    return model_kv_cache_size_bytes(
        num_tokens, model, kv_cache_dtype_size_bytes,
        attention_tensor_parallel_size, decode_context_parallel_size);
}

std::uint64_t model_kv_cache_size_bytes_rank_local(
    std::uint64_t num_tokens, const config::ModelConfig &model,
    double kv_cache_dtype_size_bytes,
    std::uint64_t decode_context_parallel_size,
    std::uint64_t decode_context_parallel_rank) {
    if (!model.use_mla) {
        if (decode_context_parallel_size != 1 ||
            decode_context_parallel_rank != 0) {
            throw TransferModelError(
                "decode context parallel KV layout is currently supported "
                "only for MLA models");
        }
        return model_kv_cache_size_bytes_one_copy(num_tokens, model,
                                                  kv_cache_dtype_size_bytes);
    }
    try {
        return attention::mla_dcp_rank_kv_cache_size_bytes(
            num_tokens, mla_cache_layer_count(model),
            attention::MlaKvCacheLayout{
                model.kv_lora_rank,
                model.qk_rope_head_dim,
                kv_cache_dtype_size_bytes,
                2.0,
            },
            decode_context_parallel_size, decode_context_parallel_rank);
    } catch (const attention::MlaLayoutError &error) {
        throw TransferModelError(error.what());
    }
}

std::uint64_t model_kv_cache_size_bytes_stage_rank_local(
    std::uint64_t num_tokens, const config::ModelConfig &model,
    double kv_cache_dtype_size_bytes,
    const config::PipelineStageLayerRange &layers,
    std::uint64_t decode_context_parallel_size,
    std::uint64_t decode_context_parallel_rank) {
    if (!std::isfinite(kv_cache_dtype_size_bytes) ||
        kv_cache_dtype_size_bytes <= 0.0) {
        throw TransferModelError("KV dtype size must be finite and positive");
    }
    if (decode_context_parallel_size == 0 ||
        decode_context_parallel_rank >= decode_context_parallel_size) {
        throw TransferModelError("invalid decode context parallel rank");
    }
    if (decode_context_parallel_size > 1 && !model.use_mla) {
        throw TransferModelError(
            "decode context parallel KV layout is currently supported only "
            "for MLA models");
    }
    const std::uint64_t layer_count = count_stage_kv_layers(model, layers);
    if (layer_count == 0) {
        return 0;
    }
    if (model.use_mla) {
        try {
            return attention::mla_dcp_rank_kv_cache_size_bytes(
                num_tokens, layer_count,
                attention::MlaKvCacheLayout{
                    model.kv_lora_rank,
                    model.qk_rope_head_dim,
                    kv_cache_dtype_size_bytes,
                    2.0,
                },
                decode_context_parallel_size, decode_context_parallel_rank);
        } catch (const attention::MlaLayoutError &error) {
            throw TransferModelError(error.what());
        }
    }
    return dense_kv_cache_size_bytes(num_tokens,
                                     DenseKvLayout{
                                         layer_count,
                                         model.runtime_num_kv_heads(),
                                         model.runtime_head_size(),
                                         model.kv_factor(),
                                         kv_cache_dtype_size_bytes,
                                     });
}

std::uint64_t model_kv_cache_size_bytes_stage_physical(
    std::uint64_t num_tokens, const config::ModelConfig &model,
    double kv_cache_dtype_size_bytes,
    const config::PipelineStageLayerRange &layers,
    std::uint64_t attention_tensor_parallel_size,
    std::uint64_t decode_context_parallel_size) {
    if (attention_tensor_parallel_size == 0 ||
        decode_context_parallel_size == 0 ||
        attention_tensor_parallel_size % decode_context_parallel_size != 0) {
        throw TransferModelError(
            "attention tensor parallel size must be positive and divisible "
            "by decode context parallel size");
    }
    std::uint64_t total = 0;
    for (std::uint64_t rank = 0; rank < decode_context_parallel_size; ++rank) {
        const std::uint64_t bytes = model_kv_cache_size_bytes_stage_rank_local(
            num_tokens, model, kv_cache_dtype_size_bytes, layers,
            decode_context_parallel_size, rank);
        if (bytes > std::numeric_limits<std::uint64_t>::max() - total) {
            throw TransferModelError(
                "stage physical KV cache size overflows uint64");
        }
        total += bytes;
    }
    // DCP ranks cover one TP/DCP-independent MLA copy.  Replication across
    // the remaining attention TP groups is part of the full target-physical
    // footprint (the same copies factor used by the legacy full-model helper).
    if (model.use_mla) {
        const std::uint64_t copies =
            attention_tensor_parallel_size / decode_context_parallel_size;
        if (copies != 0 &&
            total > std::numeric_limits<std::uint64_t>::max() / copies) {
            throw TransferModelError(
                "stage physical KV cache size overflows uint64");
        }
        total *= copies;
    }
    return total;
}

std::uint64_t
model_kda_state_snapshot_size_bytes(const config::ModelConfig &model,
                                    double state_dtype_size_bytes) {
    if (!model.has_kda()) {
        return 0;
    }
    const long double elements =
        static_cast<long double>(model.num_kda_layers) *
        static_cast<long double>(model.kda_state_elements_per_layer());
    if (elements <= 0.0L) {
        throw TransferModelError(
            "KDA model must expose positive recurrent and convolution state "
            "dimensions");
    }
    return checked_state_size(elements, state_dtype_size_bytes);
}

std::uint64_t model_kda_state_snapshot_size_bytes_rank_local(
    const config::ModelConfig &model, double state_dtype_size_bytes,
    std::uint64_t attention_tensor_parallel_size) {
    if (attention_tensor_parallel_size == 0) {
        throw TransferModelError(
            "attention tensor parallel size must be positive");
    }
    const std::uint64_t total =
        model_kda_state_snapshot_size_bytes(model, state_dtype_size_bytes);
    if (total == 0) {
        return 0;
    }
    return total / attention_tensor_parallel_size +
           static_cast<std::uint64_t>(total % attention_tensor_parallel_size !=
                                      0);
}

std::uint64_t model_kda_state_snapshot_size_bytes_stage_rank_local(
    const config::ModelConfig &model, double state_dtype_size_bytes,
    std::uint64_t attention_tensor_parallel_size,
    const config::PipelineStageLayerRange &layers) {
    if (attention_tensor_parallel_size == 0) {
        throw TransferModelError(
            "attention tensor parallel size must be positive");
    }
    const std::uint64_t layer_count = count_stage_kda_layers(model, layers);
    if (layer_count == 0) {
        return 0;
    }
    const std::uint64_t elements_per_layer =
        model.kda_state_elements_per_layer();
    if (elements_per_layer == 0) {
        throw TransferModelError(
            "KDA model must expose positive recurrent and convolution state "
            "dimensions");
    }
    const std::uint64_t stage_bytes =
        checked_state_size(static_cast<long double>(layer_count) *
                               static_cast<long double>(elements_per_layer),
                           state_dtype_size_bytes);
    return checked_ceil_div(stage_bytes, attention_tensor_parallel_size);
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
    std::uint64_t attention_tensor_parallel_size,
    std::uint64_t decode_context_parallel_size,
    double kda_snapshot_dtype_size_bytes)
    : config_(std::move(config)),
      attention_tensor_parallel_size_(attention_tensor_parallel_size),
      decode_context_parallel_size_(decode_context_parallel_size),
      kda_snapshot_dtype_size_bytes_(kda_snapshot_dtype_size_bytes) {
    if (attention_tensor_parallel_size_ == 0 ||
        decode_context_parallel_size_ == 0 ||
        attention_tensor_parallel_size_ % decode_context_parallel_size_ != 0) {
        throw TransferModelError(
            "attention tensor parallel size must be positive and divisible "
            "by decode context parallel size");
    }
    if (!std::isfinite(kda_snapshot_dtype_size_bytes_) ||
        kda_snapshot_dtype_size_bytes_ <= 0.0) {
        throw TransferModelError(
            "KDA snapshot dtype size must be finite and positive");
    }
}

TransferPrediction AnalyticalKVCacheTransferPredictor::predict(
    std::uint64_t num_tokens, const config::ModelConfig &model) const {
    const std::uint64_t kv_size_bytes =
        model_kv_cache_size_bytes_target_physical(
            num_tokens, model, config_.kv_cache_dtype_size_bytes,
            attention_tensor_parallel_size_, decode_context_parallel_size_);
    // The simplified K3 contract transfers one latest KDA snapshot as an
    // indivisible object.  Rewinding that snapshot is modeled as free, but
    // moving it between PREFILL and DECODE still consumes bandwidth.
    const std::uint64_t kda_size_bytes = model_kda_state_snapshot_size_bytes(
        model, kda_snapshot_dtype_size_bytes_);
    if (kv_size_bytes >
        std::numeric_limits<std::uint64_t>::max() - kda_size_bytes) {
        throw TransferModelError(
            "hybrid KV/KDA transfer size overflows uint64");
    }
    const std::uint64_t size_bytes = kv_size_bytes + kda_size_bytes;
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
make_kv_cache_transfer_predictor(const config::KvCacheTransferConfig &config,
                                 std::uint64_t attention_tensor_parallel_size,
                                 std::uint64_t decode_context_parallel_size,
                                 double kda_snapshot_dtype_size_bytes) {
    return std::make_shared<AnalyticalKVCacheTransferPredictor>(
        config, attention_tensor_parallel_size, decode_context_parallel_size,
        kda_snapshot_dtype_size_bytes);
}

} // namespace frontier::kv_cache_transfer
