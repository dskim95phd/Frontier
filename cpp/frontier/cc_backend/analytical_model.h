#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>

#include "frontier/cc_backend/base_cc_backend.h"

namespace frontier::cc_backend {

struct AnalyticalCommunicationConfig {
    double network_bandwidth_gbps;
    double latency_us;
    double intra_node_bandwidth_gbps;
};

class CommunicationModelError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

class AnalyticalCommunicationModel final : public BaseCCBackend {
  public:
    explicit AnalyticalCommunicationModel(AnalyticalCommunicationConfig config);

    [[nodiscard]] double
    point_to_point_ms(std::uint64_t data_size_bytes,
                      bool intra_node = true) const override;
    [[nodiscard]] double allreduce_ms(std::uint64_t data_size_bytes,
                                      std::uint64_t num_devices,
                                      bool intra_node = true) const override;
    [[nodiscard]] double allgather_ms(std::uint64_t data_size_bytes_per_device,
                                      std::uint64_t num_devices,
                                      bool intra_node = true) const override;
    [[nodiscard]] double broadcast_ms(std::uint64_t data_size_bytes,
                                      std::uint64_t num_devices,
                                      bool intra_node = true) const override;
    [[nodiscard]] double
    reduce_scatter_ms(std::uint64_t data_size_bytes, std::uint64_t num_devices,
                      bool intra_node = true) const override;
    [[nodiscard]] double all_to_all_ms(std::uint64_t data_size_bytes,
                                       std::uint64_t num_devices,
                                       bool intra_node = true) const override;

  private:
    [[nodiscard]] double transfer_ms(double data_size_bytes,
                                     bool intra_node) const;
    static void require_devices(std::uint64_t num_devices);

    AnalyticalCommunicationConfig config_;
};

[[nodiscard]] std::shared_ptr<const BaseCCBackend>
make_analytical_cc_backend(AnalyticalCommunicationConfig config);

} // namespace frontier::cc_backend
