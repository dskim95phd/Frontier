#pragma once

#include <cstdint>

namespace frontier::cc_backend {

class BaseCCBackend {
  public:
    virtual ~BaseCCBackend() = default;

    [[nodiscard]] virtual double
    point_to_point_ms(std::uint64_t data_size_bytes,
                      bool intra_node = true) const = 0;
    [[nodiscard]] virtual double allreduce_ms(std::uint64_t data_size_bytes,
                                              std::uint64_t num_devices,
                                              bool intra_node = true) const = 0;
    [[nodiscard]] virtual double
    allgather_ms(std::uint64_t data_size_bytes_per_device,
                 std::uint64_t num_devices, bool intra_node = true) const = 0;
    [[nodiscard]] virtual double broadcast_ms(std::uint64_t data_size_bytes,
                                              std::uint64_t num_devices,
                                              bool intra_node = true) const = 0;
    [[nodiscard]] virtual double
    reduce_scatter_ms(std::uint64_t data_size_bytes, std::uint64_t num_devices,
                      bool intra_node = true) const = 0;
    [[nodiscard]] virtual double
    all_to_all_ms(std::uint64_t data_size_bytes, std::uint64_t num_devices,
                  bool intra_node = true) const = 0;
};

} // namespace frontier::cc_backend
