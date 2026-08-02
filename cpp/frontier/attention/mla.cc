#include "frontier/attention/mla.h"

#include <cmath>
#include <limits>

namespace frontier::attention {

double mla_kv_cache_bytes_per_token(const MlaKvCacheLayout &layout) {
    if (layout.latent_dim == 0 || layout.rope_dim == 0) {
        throw MlaLayoutError("MLA KV-cache dimensions must be positive");
    }
    if (!std::isfinite(layout.latent_element_bytes) ||
        layout.latent_element_bytes <= 0.0 ||
        !std::isfinite(layout.rope_element_bytes) ||
        layout.rope_element_bytes <= 0.0) {
        throw MlaLayoutError(
            "MLA KV-cache element sizes must be finite and positive");
    }
    return static_cast<double>(layout.latent_dim) *
               layout.latent_element_bytes +
           static_cast<double>(layout.rope_dim) * layout.rope_element_bytes;
}

std::uint64_t mla_kv_cache_size_bytes(std::uint64_t num_tokens,
                                      std::uint64_t num_layers,
                                      const MlaKvCacheLayout &layout) {
    if (num_layers == 0) {
        throw MlaLayoutError("MLA KV-cache layer count must be positive");
    }
    const long double size =
        static_cast<long double>(num_tokens) *
        static_cast<long double>(num_layers) *
        static_cast<long double>(mla_kv_cache_bytes_per_token(layout));
    if (!std::isfinite(size) ||
        size > static_cast<long double>(
                   std::numeric_limits<std::uint64_t>::max())) {
        throw MlaLayoutError("MLA KV-cache size overflows uint64");
    }
    return static_cast<std::uint64_t>(size);
}

} // namespace frontier::attention
