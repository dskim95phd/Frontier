#pragma once

#include <cstdint>
#include <stdexcept>

#include "frontier/config/config.h"
#include "frontier/core/ids.h"

namespace frontier::entities {

class ReplicaError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class Replica {
 public:
  Replica(
      ReplicaId replica_id,
      const config::ParallelismConfig& parallelism,
      std::uint64_t num_layers = 32);

  [[nodiscard]] ReplicaId id() const noexcept { return replica_id_; }
  [[nodiscard]] std::uint64_t tensor_parallel_size() const noexcept {
    return tensor_parallel_size_;
  }
  [[nodiscard]] std::uint64_t pipeline_parallel_size() const noexcept {
    return pipeline_parallel_size_;
  }
  [[nodiscard]] std::uint64_t data_parallel_size() const noexcept {
    return data_parallel_size_;
  }
  [[nodiscard]] std::uint64_t num_layers_per_pipeline_stage()
      const noexcept {
    return num_layers_ / pipeline_parallel_size_;
  }
  [[nodiscard]] std::uint64_t accelerator_count() const noexcept {
    return tensor_parallel_size_ * pipeline_parallel_size_ *
        data_parallel_size_;
  }

 private:
  ReplicaId replica_id_;
  std::uint64_t tensor_parallel_size_;
  std::uint64_t pipeline_parallel_size_;
  std::uint64_t data_parallel_size_;
  std::uint64_t num_layers_;
};

}  // namespace frontier::entities
