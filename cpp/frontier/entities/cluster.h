#pragma once

#include <memory>
#include <vector>

#include "frontier/config/config.h"
#include "frontier/core/cluster_type.h"
#include "frontier/entities/replica.h"

namespace frontier::execution_time_predictor {
class BatchExecutionModel;
}

namespace frontier::cc_backend {
class BaseCCBackend;
}

namespace frontier::entities {

class Cluster {
 public:
  Cluster(
      ClusterType type,
      config::ClusterRuntimeConfig runtime_config);

  [[nodiscard]] ClusterType type() const noexcept { return type_; }
  [[nodiscard]] const config::ClusterRuntimeConfig& runtime_config()
      const noexcept {
    return runtime_config_;
  }
  [[nodiscard]] const config::ParallelismConfig& parallelism()
      const noexcept {
    return runtime_config_.parallelism;
  }
  [[nodiscard]] const config::ModelConfig& model() const noexcept {
    return runtime_config_.model;
  }
  [[nodiscard]] const std::shared_ptr<
      const execution_time_predictor::BatchExecutionModel>&
  execution_model() const noexcept {
    return execution_model_;
  }
  [[nodiscard]] const std::shared_ptr<
      const cc_backend::BaseCCBackend>&
  communication_backend() const noexcept {
    return communication_backend_;
  }
  [[nodiscard]] const std::vector<Replica>& replicas() const noexcept {
    return replicas_;
  }
  [[nodiscard]] const Replica& replica(ReplicaId replica_id) const;

 private:
  ClusterType type_;
  config::ClusterRuntimeConfig runtime_config_;
  std::shared_ptr<const cc_backend::BaseCCBackend>
      communication_backend_;
  std::shared_ptr<
      const execution_time_predictor::BatchExecutionModel>
      execution_model_;
  std::vector<Replica> replicas_;
};

}  // namespace frontier::entities
