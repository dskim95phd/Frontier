#pragma once

#include <cstdint>

#include "frontier/config/config.h"

namespace frontier::kv_cache_transfer {

struct TransferPrediction {
  std::uint64_t size_bytes;
  double effective_size_bytes;
  double transfer_time_ms;
};

class BaseKVCacheTransferPredictor {
 public:
  virtual ~BaseKVCacheTransferPredictor() = default;

  [[nodiscard]] virtual TransferPrediction predict(
      std::uint64_t num_tokens,
      const config::ModelConfig& model) const = 0;
};

}  // namespace frontier::kv_cache_transfer
