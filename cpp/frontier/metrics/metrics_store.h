#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "frontier/config/config.h"
#include "frontier/metrics/output_contract.h"

namespace frontier::simulator {
class EntityArena;
}

namespace frontier::entities {
class Batch;
class BatchStage;
class KVCacheTransferInfo;
class Request;
} // namespace frontier::entities

namespace frontier::execution_time_predictor {
struct MoERoutingDiagnostic;
}

namespace frontier::scheduler {
struct ReplicaTarget;
struct ScheduleResult;
} // namespace frontier::scheduler

namespace frontier::metrics {

// Runtime metrics ownership corresponding to Python's MetricsStore.
// Event handlers report observations here; output serialization remains in
// output_contract.
class MetricsStore {
  public:
    explicit MetricsStore(const config::SimulationConfig &config,
                          std::size_t expected_request_count = 0);

    void set_detailed_traces_enabled(bool enabled) noexcept {
        detailed_traces_enabled_ = enabled;
    }

    void record_event(Event event);
    void record_batch(const entities::Batch &batch,
                      const std::vector<entities::Request> &requests,
                      double predicted_execution_ms,
                      const config::ClusterRuntimeConfig &runtime);
    void record_batch_stage(const entities::BatchStage &batch_stage,
                            const entities::Batch &batch,
                            const config::ClusterRuntimeConfig &runtime);
    void record_scheduler_trace(const scheduler::ScheduleResult &schedule,
                                scheduler::ReplicaTarget target,
                                ClusterType cluster_type);
    void
    record_kv_cache_transfer(const entities::KVCacheTransferInfo &transfer);
    void record_analytical_diagnostic(
        std::string name, std::vector<std::pair<std::string, double>> values);
    void record_moe_routing(
        const entities::Batch &batch, StageId stage_id,
        const execution_time_predictor::MoERoutingDiagnostic &diagnostic,
        const config::ClusterRuntimeConfig &runtime);

    void collect_completed_requests(const config::SimulationConfig &config,
                                    const simulator::EntityArena &entities);

    [[nodiscard]] SimulationOutput take_output() noexcept;

  private:
    void record_request(RequestMetricsRecord record);
    SimulationOutput output_;
    bool detailed_traces_enabled_ = true;
};

} // namespace frontier::metrics
