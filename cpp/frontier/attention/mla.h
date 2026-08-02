#pragma once

#include <cstdint>
#include <stdexcept>

namespace frontier::attention {

enum class MlaExecutionMode {
    kUnabsorbedMha,
    kAbsorbedMqa,
};

struct MlaKvCacheLayout {
    std::uint64_t latent_dim = 0;
    std::uint64_t rope_dim = 0;
    double latent_element_bytes = 0.0;
    double rope_element_bytes = 2.0;
};

class MlaLayoutError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

[[nodiscard]] double
mla_kv_cache_bytes_per_token(const MlaKvCacheLayout &layout);

[[nodiscard]] std::uint64_t
mla_kv_cache_size_bytes(std::uint64_t num_tokens, std::uint64_t num_layers,
                        const MlaKvCacheLayout &layout);

} // namespace frontier::attention
