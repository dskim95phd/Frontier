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
    Replica(ReplicaId replica_id, config::ParallelismConfig parallelism,
            config::ModelConfig model);

    [[nodiscard]] ReplicaId id() const noexcept { return replica_id_; }
    [[nodiscard]] const config::ParallelismConfig &
    parallelism() const noexcept {
        return parallelism_;
    }
    [[nodiscard]] const config::ModelConfig &model() const noexcept {
        return model_;
    }
    [[nodiscard]] std::uint64_t tensor_parallel_size() const noexcept {
        return parallelism_.tensor_parallel_size;
    }
    [[nodiscard]] std::uint64_t pipeline_parallel_size() const noexcept {
        return parallelism_.pipeline_parallel_size;
    }
    [[nodiscard]] std::uint64_t data_parallel_size() const noexcept {
        return parallelism_.data_parallel_size;
    }
    [[nodiscard]] std::uint64_t moe_tensor_parallel_size() const noexcept {
        return parallelism_.moe_tensor_parallel_size;
    }
    [[nodiscard]] std::uint64_t moe_expert_parallel_size() const noexcept {
        return parallelism_.moe_expert_parallel_size;
    }
    [[nodiscard]] std::uint64_t num_layers_per_pipeline_stage() const noexcept {
        return model_.num_layers / parallelism_.pipeline_parallel_size;
    }
    [[nodiscard]] std::uint64_t accelerator_count() const noexcept {
        return parallelism_.attention_parallel_size() *
               parallelism_.pipeline_parallel_size;
    }

  private:
    ReplicaId replica_id_;
    config::ParallelismConfig parallelism_;
    config::ModelConfig model_;
};

} // namespace frontier::entities
