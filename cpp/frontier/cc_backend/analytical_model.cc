#include "frontier/cc_backend/analytical_model.h"

#include <cmath>

namespace frontier::cc_backend {

AnalyticalCommunicationModel::AnalyticalCommunicationModel(
    AnalyticalCommunicationConfig config)
    : config_(config) {
  if (!std::isfinite(config_.network_bandwidth_gbps) ||
      config_.network_bandwidth_gbps <= 0.0) {
    throw CommunicationModelError(
        "network bandwidth must be finite and positive");
  }
  if (!std::isfinite(config_.intra_node_bandwidth_gbps) ||
      config_.intra_node_bandwidth_gbps <= 0.0) {
    throw CommunicationModelError(
        "intra-node bandwidth must be finite and positive");
  }
  if (!std::isfinite(config_.latency_us) ||
      config_.latency_us < 0.0) {
    throw CommunicationModelError(
        "communication latency must be finite and nonnegative");
  }
}

double AnalyticalCommunicationModel::point_to_point_ms(
    std::uint64_t data_size_bytes,
    bool intra_node) const {
  return transfer_ms(static_cast<double>(data_size_bytes), intra_node);
}

double AnalyticalCommunicationModel::allreduce_ms(
    std::uint64_t data_size_bytes,
    std::uint64_t num_devices,
    bool intra_node) const {
  require_devices(num_devices);
  if (num_devices == 1) {
    return 0.0;
  }
  const double devices = static_cast<double>(num_devices);
  const double effective_size =
      2.0 * (devices - 1.0) / devices *
      static_cast<double>(data_size_bytes);
  return transfer_ms(effective_size, intra_node);
}

double AnalyticalCommunicationModel::allgather_ms(
    std::uint64_t data_size_bytes_per_device,
    std::uint64_t num_devices,
    bool intra_node) const {
  require_devices(num_devices);
  if (num_devices == 1) {
    return 0.0;
  }
  const double effective_size =
      static_cast<double>(num_devices - 1) *
      static_cast<double>(data_size_bytes_per_device);
  return transfer_ms(effective_size, intra_node);
}

double AnalyticalCommunicationModel::broadcast_ms(
    std::uint64_t data_size_bytes,
    std::uint64_t num_devices,
    bool intra_node) const {
  require_devices(num_devices);
  if (num_devices == 1) {
    return 0.0;
  }
  const double steps =
      std::ceil(std::log2(static_cast<double>(num_devices)));
  return steps *
         transfer_ms(static_cast<double>(data_size_bytes), intra_node);
}

double AnalyticalCommunicationModel::reduce_scatter_ms(
    std::uint64_t data_size_bytes,
    std::uint64_t num_devices,
    bool intra_node) const {
  require_devices(num_devices);
  if (num_devices == 1) {
    return 0.0;
  }
  const double devices = static_cast<double>(num_devices);
  const double effective_size =
      (devices - 1.0) / devices *
      static_cast<double>(data_size_bytes);
  return transfer_ms(effective_size, intra_node);
}

double AnalyticalCommunicationModel::all_to_all_ms(
    std::uint64_t data_size_bytes,
    std::uint64_t num_devices,
    bool intra_node) const {
  require_devices(num_devices);
  if (num_devices == 1) {
    return 0.0;
  }
  const std::uint64_t data_per_device =
      data_size_bytes / num_devices;
  const double effective_size =
      static_cast<double>(num_devices - 1) *
      static_cast<double>(data_per_device);
  return transfer_ms(effective_size, intra_node);
}

double AnalyticalCommunicationModel::transfer_ms(
    double data_size_bytes,
    bool intra_node) const {
  const double bandwidth_gbps =
      intra_node ? config_.intra_node_bandwidth_gbps
                 : config_.network_bandwidth_gbps;
  const double bandwidth_bytes_per_ms =
      bandwidth_gbps * 1e9 / (8.0 * 1e3);
  const double latency_ms = config_.latency_us / 1e3;
  return latency_ms + data_size_bytes / bandwidth_bytes_per_ms;
}

void AnalyticalCommunicationModel::require_devices(
    std::uint64_t num_devices) {
  if (num_devices == 0) {
    throw CommunicationModelError(
        "collective device count must be positive");
  }
}

std::shared_ptr<const BaseCCBackend>
make_analytical_cc_backend(AnalyticalCommunicationConfig config) {
  return std::make_shared<AnalyticalCommunicationModel>(config);
}

}  // namespace frontier::cc_backend
