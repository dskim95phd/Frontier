#include "frontier/execution_time_predictor/analytical_roofline_execution_time_predictor.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "frontier/execution_time_predictor/analytical_moe_stage.h"
#include "frontier/execution_time_predictor/analytical_roofline_internal.h"
#include "frontier/execution_time_predictor/analytical_stage_diagnostics.h"

namespace frontier::execution_time_predictor {

using roofline_detail::build_stage_batch_info;
using roofline_detail::make_dense_operator_precisions;
using roofline_detail::materialize_cached_prediction;
using roofline_detail::StageBatchInfo;

ExecutionTimePrediction
AnalyticalRooflineExecutionTimePredictor::prepare_moe_stage_execution(
    const entities::Batch &batch, const entities::RequestCollection &requests,
    StageId stage_id) const {
    if (!model_.is_moe()) {
        throw ExecutionTimePredictorError(
            "dense model cannot prepare lazy MoE execution");
    }
    if (config_.moe_layer_event_mode == "first_layer_scaled" ||
        config_.moe_layer_event_mode == "stage_group_scaled") {
        return predict_execution(batch, requests, stage_id, std::nullopt);
    }
    return predict_execution(batch, requests, stage_id, std::uint64_t{0});
}

ExecutionTimePrediction
AnalyticalRooflineExecutionTimePredictor::predict_moe_layer_execution(
    const entities::Batch &batch, const entities::RequestCollection &requests,
    StageId stage_id, std::uint64_t local_moe_layer) const {
    if (!model_.is_moe() ||
        config_.moe_layer_event_mode == "first_layer_scaled" ||
        config_.moe_layer_event_mode == "stage_group_scaled") {
        throw ExecutionTimePredictorError(
            "lazy MoE layer prediction requires detailed analytical mode");
    }
    return predict_execution(batch, requests, stage_id, local_moe_layer);
}

ExecutionTimePrediction
AnalyticalRooflineExecutionTimePredictor::predict_execution(
    const entities::Batch &batch, const entities::RequestCollection &requests,
    StageId stage_id, std::optional<std::uint64_t> selected_moe_layer) const {
    if (!stage_id.valid() ||
        stage_id.index() >= parallelism_.pipeline_parallel_size) {
        throw ExecutionTimePredictorError(
            "analytical stage ID exceeds pipeline size");
    }

    const config::PipelineStageLayerRange stage_layers =
        config::pipeline_stage_layer_range(model_.num_layers, parallelism_,
                                           stage_id.index());
    const bool stage_group_requested =
        config_.moe_layer_event_mode == "stage_group_scaled";
    const bool stage_group_active =
        stage_group_requested && model_.is_moe() &&
        internal::routing_is_layer_invariant(routing_) &&
        internal::has_contiguous_moe_suffix(model_, stage_layers);
    const std::uint32_t timing_group_id =
        timing_catalogue_.stage_to_timing_group.at(stage_id.index());
    const std::uint64_t timing_group_multiplicity =
        timing_catalogue_.timing_group_multiplicity.at(timing_group_id);
    // A singleton timing group has no second stage that can consume a cached
    // value for this batch. Avoid allocating, inserting, copying, and later
    // erasing a template that is structurally guaranteed never to hit.
    const bool reusable_timing_group = stage_group_active &&
                                       timing_group_multiplicity > 1 &&
                                       !selected_moe_layer.has_value();
    if (!detailed_diagnostics_enabled_ && reusable_timing_group) {
        const std::shared_ptr<const ExecutionTimePrediction> cached =
            lookup_complete_prediction(batch.id(), timing_group_id);
        if (cached != nullptr) {
            return materialize_cached_prediction(*cached, model_, stage_layers);
        }
    }

    const StageBatchInfo batch_info = build_stage_batch_info(batch, requests);
    const detail::DenseBatch &dense_batch = batch_info.dense_batch;
    const detail::DenseOperatorPrecisions dense_precisions =
        make_dense_operator_precisions(config_);
    const detail::Precision communication_precision =
        detail::precision_from_string(config_.communication_precision());
    std::vector<detail::DenseLayerTimes> layer_times;
    layer_times.reserve(static_cast<std::size_t>(stage_layers.size()));
    StageTimingCacheLookup timing_cache_lookup{};
    std::shared_ptr<const StageTimingCacheValue> timing_template;
    if (reusable_timing_group) {
        timing_cache_lookup = lookup_stage_timing_template(
            batch.id(), timing_group_id, dense_batch, stage_layers);
        timing_template = timing_cache_lookup.value;
        layer_times = timing_template->layer_times;
    } else {
        {
            std::lock_guard<std::mutex> lock(timing_cache_mutex_);
            timing_cache_lookup.hits_total = timing_cache_hits_total_;
            timing_cache_lookup.misses_total = timing_cache_misses_total_;
            timing_cache_lookup.unique_templates_total =
                timing_cache_unique_templates_total_;
            timing_cache_lookup.entries = timing_cache_entries_;
        }
        layer_times = build_stage_layer_times(stage_layers, dense_batch,
                                              dense_precisions);
    }
    if (layer_times.empty()) {
        throw ExecutionTimePredictorError(
            "analytical stage has no logical layers");
    }

    const double communication_element_bytes =
        detail::bytes_per_element(communication_precision);
    const std::uint64_t activation_bytes = internal::activation_payload_bytes(
        dense_batch.total_tokens, model_.hidden_size,
        communication_element_bytes);
    const double allreduce_ms = timing_template != nullptr
                                    ? timing_template->allreduce_ms
                                    : compute_allreduce_ms(activation_bytes);
    const double dcp_attention_communication_ms =
        timing_template != nullptr
            ? timing_template->dcp_attention_communication_ms
            : compute_dcp_attention_communication_ms(
                  dense_batch, stage_layers, communication_element_bytes);
    double dense_compute_ms = 0.0;
    double tp_communication_ms = 0.0;
    for (std::size_t index = 0; index < layer_times.size(); ++index) {
        dense_compute_ms += layer_times[index].total_ms();
        const std::uint64_t model_layer =
            stage_layers.begin + static_cast<std::uint64_t>(index);
        const bool kda_layer =
            model_.has_kda() && model_.is_kda_layer(model_layer);
        tp_communication_ms +=
            2.0 * allreduce_ms +
            (kda_layer ? 0.0 : dcp_attention_communication_ms);
    }
    const double first_layer_compute_ms = layer_times.front().total_ms();
    const double first_tp_layer_ms =
        2.0 * allreduce_ms +
        ((model_.has_kda() && model_.is_kda_layer(stage_layers.begin))
             ? 0.0
             : dcp_attention_communication_ms);
    const double pp_communication_ms =
        stage_id.index() + 1 < parallelism_.pipeline_parallel_size
            ? communication_backend_->point_to_point_ms(activation_bytes, true)
            : 0.0;
    entities::ExecutionTime execution_time{};
    execution_time.dense_compute_ms = dense_compute_ms;
    execution_time.tp_communication_ms = tp_communication_ms;
    execution_time.pp_communication_ms = pp_communication_ms;
    if (stage_id.index() + 1 == parallelism_.pipeline_parallel_size) {
        execution_time.lm_head_ms = detail::predict_output_projection_ms(
            device_, analytical_, batch_info.lm_head_tokens, model_.hidden_size,
            model_.vocab_size, parallelism_.tensor_parallel_size,
            detail::precision_from_string(config_.lm_head_weight_precision()),
            detail::precision_from_string(
                config_.lm_head_activation_precision()));
    }
    if (selected_moe_layer.has_value() && selected_moe_layer.value() > 0) {
        execution_time.lm_head_ms = 0.0;
        execution_time.pp_communication_ms = 0.0;
    }
    std::vector<std::pair<std::string, double>> moe_diagnostics;
    std::vector<MoERoutingDiagnostic> routing_diagnostics;
    std::uint64_t logical_moe_layer_count = 0;
    std::vector<ScaledMoEAttentionGroup> scaled_moe_attention_groups;
    double repeated_moe_layer_pre_compute_ms = 0.0;
    double moe_suffix_compute_ms = 0.0;
    double moe_suffix_tp_communication_ms = 0.0;
    if (model_.is_moe()) {
        // Shared layer scope and first_layer_scaled both reuse one assignment
        // for every MoE layer of the batch. Resolving it here covers whole
        // stages, stage-group templates, and single lazy-layer predictions.
        const std::shared_ptr<const SharedRoutingValue> shared_routing =
            internal::routing_is_batch_shared(config_, routing_)
                ? shared_routing_for_batch(batch.id(), dense_batch)
                : nullptr;
        const internal::MoEStageContext moe_context{
            batch.cluster_type(),
            dense_batch,
            config_,
            device_,
            analytical_,
            parallelism_,
            model_,
            routing_,
            *communication_backend_,
            stage_layers,
            layer_times,
            allreduce_ms,
            dcp_attention_communication_ms,
            communication_element_bytes,
            execution_time,
            shared_routing != nullptr ? &shared_routing->lane_prediction
                                      : nullptr,
            timing_template != nullptr && timing_template->has_moe_communication
                ? &timing_template->moe_communication
                : nullptr,
            shared_routing != nullptr ? &shared_routing->allocation : nullptr,
            detailed_diagnostics_enabled_,
            static_cast<std::uint64_t>(batch.id().value()),
        };
        internal::MoEStagePrediction moe_prediction =
            selected_moe_layer.has_value()
                ? internal::predict_selected_moe_layer_execution(
                      moe_context, selected_moe_layer.value())
                : internal::predict_moe_stage_execution(moe_context);
        execution_time = moe_prediction.execution_time;
        moe_diagnostics = std::move(moe_prediction.diagnostics);
        routing_diagnostics = std::move(moe_prediction.routing_diagnostics);
        logical_moe_layer_count = moe_prediction.logical_moe_layer_count;
        scaled_moe_attention_groups =
            std::move(moe_prediction.scaled_attention_groups);
        repeated_moe_layer_pre_compute_ms =
            moe_prediction.repeated_moe_layer_pre_compute_ms;
        moe_suffix_compute_ms = moe_prediction.suffix_compute_ms;
        moe_suffix_tp_communication_ms =
            moe_prediction.suffix_tp_communication_ms;
    }
    // A synchronized lazy-MoE stage is assembled from one initial prediction
    // followed by one prediction per additional MoE layer.  Attribute the
    // stage-level PREFILL attention demand once, on the initial prediction,
    // rather than multiplying it by those layer callbacks.  This field is
    // instrumentation only and does not participate in duration_ms.
    if (!selected_moe_layer.has_value() || selected_moe_layer.value() == 0) {
        execution_time.prefill_attention_token_pairs =
            detail::prefill_attention_token_pairs(dense_batch.prefill_requests);
    }
    dense_compute_ms = execution_time.dense_compute_ms;
    tp_communication_ms = execution_time.tp_communication_ms;
    const double duration_ms = execution_time.total_ms();
    if (!std::isfinite(duration_ms) || duration_ms < 0.0) {
        throw ExecutionTimePredictorError(
            "analytical stage duration is invalid");
    }
    ExecutionTimePrediction result{};
    result.duration_ms = duration_ms;
    result.execution_time = execution_time;
    if (detailed_diagnostics_enabled_) {
        result.diagnostics =
            detail::build_stage_diagnostics(detail::StageDiagnosticsInput{
                config_,
                parallelism_,
                model_,
                timing_catalogue_,
                dense_batch,
                dense_precisions,
                layer_times,
                stage_layers,
                execution_time,
                stage_id.index(),
                timing_group_id,
                timing_group_multiplicity,
                detail::StageTimingCacheDiagnostics{
                    timing_cache_lookup.retained,
                    timing_cache_lookup.hit,
                    timing_cache_lookup.hits_total,
                    timing_cache_lookup.misses_total,
                    timing_cache_lookup.unique_templates_total,
                    timing_cache_lookup.entries,
                },
                stage_group_requested,
                stage_group_active,
                internal::routing_is_layer_invariant(routing_),
                batch_info.lm_head_tokens,
                communication_element_bytes,
                allreduce_ms,
                dcp_attention_communication_ms,
                first_layer_compute_ms,
                first_tp_layer_ms,
                dense_compute_ms,
                tp_communication_ms,
                pp_communication_ms,
                duration_ms,
            });
    }
    result.moe_routing = std::move(routing_diagnostics);
    result.logical_moe_layer_count = logical_moe_layer_count;
    result.scaled_moe_attention_groups = std::move(scaled_moe_attention_groups);
    result.repeated_moe_layer_pre_compute_ms =
        repeated_moe_layer_pre_compute_ms;
    result.moe_suffix_compute_ms = moe_suffix_compute_ms;
    result.moe_suffix_tp_communication_ms = moe_suffix_tp_communication_ms;
    result.lazy_moe_layer_prediction = selected_moe_layer.has_value();
    result.scaled_moe_layer_prediction =
        !selected_moe_layer.has_value() &&
        (config_.moe_layer_event_mode == "first_layer_scaled" ||
         stage_group_active);
    if (detailed_diagnostics_enabled_) {
        result.diagnostics.insert(result.diagnostics.end(),
                                  moe_diagnostics.begin(),
                                  moe_diagnostics.end());
    }
    if (!detailed_diagnostics_enabled_ && reusable_timing_group &&
        timing_cache_lookup.retained) {
        store_complete_prediction(batch.id(), timing_group_id, result);
    }
    return result;
}

} // namespace frontier::execution_time_predictor
