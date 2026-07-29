#include "frontier/entities/replica.h"

namespace frontier::entities {

Replica::Replica(
    ReplicaId replica_id,
    const config::ParallelismConfig& parallelism,
    std::uint64_t num_layers)
    : replica_id_(replica_id),
      tensor_parallel_size_(parallelism.tensor_parallel_size),
      pipeline_parallel_size_(parallelism.pipeline_parallel_size),
      data_parallel_size_(parallelism.data_parallel_size),
      num_layers_(num_layers) {
  if (tensor_parallel_size_ == 0 ||
      pipeline_parallel_size_ == 0 ||
      data_parallel_size_ == 0 ||
      num_layers_ == 0) {
    throw ReplicaError("replica topology dimensions must be positive");
  }
  if (num_layers_ % pipeline_parallel_size_ != 0 ||
      4'096 % tensor_parallel_size_ != 0 ||
      32 % tensor_parallel_size_ != 0) {
    throw ReplicaError(
        "replica topology does not divide Llama-2-7B model dimensions");
  }
}

}  // namespace frontier::entities
