#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

#include "frontier/cc_backend/base_cc_backend.h"
#include "frontier/config/config.h"
#include "frontier/core/ids.h"
#include "frontier/entities/batch.h"
#include "frontier/entities/request.h"
#include "frontier/execution_time_predictor/analytical_attention_model.h"
#include "frontier/execution_time_predictor/analytical_moe_model.h"
#include "frontier/execution_time_predictor/base_execution_time_predictor.h"

namespace frontier::execution_time_predictor {

// Scheduler-facing facade for the analytical execution model. Attention,
// expert, routing, cache, and stage-assembly details live in focused internal
// modules; callers depend only on BaseExecutionTimePredictor.
class AnalyticalRooflineExecutionTimePredictor final
    : public BaseExecutionTimePredictor {
  public:
    explicit AnalyticalRooflineExecutionTimePredictor(
        config::AnalyticalExecutionModelConfig config);
    AnalyticalRooflineExecutionTimePredictor(
        config::AnalyticalExecutionModelConfig config,
        config::ParallelismConfig parallelism);
    AnalyticalRooflineExecutionTimePredictor(
        config::AnalyticalExecutionModelConfig config,
        config::ParallelismConfig parallelism, config::ModelConfig model,
        config::MoeRoutingConfig routing);
    AnalyticalRooflineExecutionTimePredictor(
        config::AnalyticalExecutionModelConfig config,
        config::ParallelismConfig parallelism, config::ModelConfig model,
        config::MoeRoutingConfig routing,
        std::shared_ptr<const cc_backend::BaseCCBackend> communication_backend);

    [[nodiscard]] ExecutionTimePrediction
    predict_stage_execution_time(const entities::Batch &batch,
                                 const std::vector<entities::Request> &requests,
                                 StageId stage_id) const override;
    [[nodiscard]] MoEGroupLayerPrediction
    predict_moe_group_layer(const MoEGroupLayerInput &input) const override;
    void release_batch_timing_cache(BatchId batch_id) const override;
    void
    set_detailed_diagnostics_enabled(bool enabled) const noexcept override {
        detailed_diagnostics_enabled_ = enabled;
    }
    [[nodiscard]] bool supports_lazy_moe_prediction() const noexcept override {
        return true;
    }
    [[nodiscard]] ExecutionTimePrediction
    prepare_moe_stage_execution(const entities::Batch &batch,
                                const std::vector<entities::Request> &requests,
                                StageId stage_id) const override;
    [[nodiscard]] ExecutionTimePrediction
    predict_moe_layer_execution(const entities::Batch &batch,
                                const std::vector<entities::Request> &requests,
                                StageId stage_id,
                                std::uint64_t local_moe_layer) const override;

  private:
    struct StageTimingCacheValue {
        std::vector<detail::DenseLayerTimes> layer_times;
        double allreduce_ms = 0.0;
        double dcp_attention_communication_ms = 0.0;
        detail::MoECommunicationTime moe_communication;
        bool has_moe_communication = false;
        mutable std::shared_ptr<const ExecutionTimePrediction>
            complete_prediction;
    };

    using BatchStageTimingCache =
        std::unordered_map<std::uint32_t,
                           std::shared_ptr<const StageTimingCacheValue>>;

    struct SharedRoutingValue {
        detail::RoutingAllocation allocation;
        detail::MoELanePrediction lane_prediction;
    };

    [[nodiscard]] std::shared_ptr<const SharedRoutingValue>
    shared_routing_for_batch(BatchId batch_id,
                             const detail::DenseBatch &dense_batch) const;

    struct StageTimingCacheLookup {
        std::shared_ptr<const StageTimingCacheValue> value;
        bool retained = false;
        bool hit = false;
        std::uint64_t hits_total = 0;
        std::uint64_t misses_total = 0;
        std::uint64_t unique_templates_total = 0;
        std::uint64_t entries = 0;
    };

    [[nodiscard]] StageTimingCacheLookup lookup_stage_timing_template(
        BatchId batch_id, std::uint32_t timing_group_id,
        const detail::DenseBatch &dense_batch,
        const config::PipelineStageLayerRange &stage_layers) const;
    [[nodiscard]] std::shared_ptr<const ExecutionTimePrediction>
    lookup_complete_prediction(BatchId batch_id,
                               std::uint32_t timing_group_id) const;
    void
    store_complete_prediction(BatchId batch_id, std::uint32_t timing_group_id,
                              const ExecutionTimePrediction &prediction) const;

    [[nodiscard]] config::StageTimingSignature
    make_stage_timing_signature(std::uint64_t stage) const;
    void build_stage_timing_groups();

    [[nodiscard]] std::vector<detail::DenseLayerTimes> build_stage_layer_times(
        const config::PipelineStageLayerRange &stage_layers,
        const detail::DenseBatch &dense_batch,
        const detail::DenseOperatorPrecisions &precisions) const;
    [[nodiscard]] double
    compute_allreduce_ms(std::uint64_t activation_bytes) const;
    [[nodiscard]] double compute_dcp_attention_communication_ms(
        const detail::DenseBatch &dense_batch,
        const config::PipelineStageLayerRange &stage_layers,
        double communication_element_bytes) const;
    [[nodiscard]] detail::MoECommunicationTime
    compute_moe_communication(const detail::DenseBatch &dense_batch,
                              double communication_element_bytes) const;

    [[nodiscard]] ExecutionTimePrediction
    predict_execution(const entities::Batch &batch,
                      const std::vector<entities::Request> &requests,
                      StageId stage_id,
                      std::optional<std::uint64_t> selected_moe_layer) const;

    config::AnalyticalExecutionModelConfig config_;
    detail::DeviceCeilings device_;
    detail::AnalyticalConfig analytical_;
    config::ParallelismConfig parallelism_;
    config::ModelConfig model_;
    config::MoeRoutingConfig routing_;
    std::shared_ptr<const cc_backend::BaseCCBackend> communication_backend_;
    config::PipelineStageGroupCatalogue timing_catalogue_;
    mutable std::mutex timing_cache_mutex_;
    mutable std::unordered_map<BatchId, BatchStageTimingCache,
                               StrongIdHash<BatchId>>
        timing_cache_by_batch_;
    mutable std::unordered_map<BatchId,
                               std::shared_ptr<const SharedRoutingValue>,
                               StrongIdHash<BatchId>>
        shared_routing_by_batch_;
    mutable std::uint64_t timing_cache_entries_ = 0;
    mutable std::uint64_t timing_cache_hits_total_ = 0;
    mutable std::uint64_t timing_cache_misses_total_ = 0;
    mutable std::uint64_t timing_cache_unique_templates_total_ = 0;
    // Incremented on release so work computed outside the mutex cannot
    // resurrect a cache entry after its owning batch has retired.
    mutable std::uint64_t timing_cache_generation_ = 0;
    mutable bool detailed_diagnostics_enabled_ = true;
};

} // namespace frontier::execution_time_predictor
