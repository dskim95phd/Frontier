#include "frontier/entities/replica.h"

#include <utility>

namespace frontier::entities {

Replica::Replica(ReplicaId replica_id, config::ParallelismConfig parallelism,
                 config::ModelConfig model)
    : replica_id_(replica_id), parallelism_(std::move(parallelism)),
      model_(std::move(model)) {
    if (!replica_id_.valid() || parallelism_.tensor_parallel_size == 0 ||
        parallelism_.pipeline_parallel_size == 0 ||
        parallelism_.data_parallel_size == 0 ||
        parallelism_.moe_tensor_parallel_size == 0 ||
        parallelism_.moe_expert_parallel_size == 0 || model_.num_layers == 0) {
        throw ReplicaError("replica topology dimensions must be positive");
    }
    if (model_.num_layers % parallelism_.pipeline_parallel_size != 0 ||
        model_.hidden_size % parallelism_.tensor_parallel_size != 0 ||
        model_.num_query_heads % parallelism_.tensor_parallel_size != 0 ||
        model_.num_kv_heads % parallelism_.tensor_parallel_size != 0) {
        throw ReplicaError("replica topology does not divide model dimensions");
    }
    if (model_.is_moe() && parallelism_.attention_parallel_size() !=
                               parallelism_.moe_parallel_size()) {
        throw ReplicaError(
            "MoE replica requires a shared attention/expert domain");
    }
}

} // namespace frontier::entities
