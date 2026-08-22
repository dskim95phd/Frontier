#include "frontier/execution_time_predictor/analytical_roofline_execution_time_predictor.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <utility>

#include "frontier/execution_time_predictor/analytical_moe_stage.h"
#include "frontier/execution_time_predictor/analytical_roofline_internal.h"

namespace frontier::execution_time_predictor {

using roofline_detail::make_dense_model;
using roofline_detail::make_dense_operator_precisions;

config::StageTimingSignature
AnalyticalRooflineExecutionTimePredictor::make_stage_timing_signature(
    std::uint64_t stage) const {
    return config::build_pipeline_stage_timing_signature(model_, parallelism_,
                                                         stage);
}

void AnalyticalRooflineExecutionTimePredictor::build_stage_timing_groups() {
    timing_catalogue_.timing_groups.clear();
    timing_catalogue_.stage_to_timing_group.clear();
    timing_catalogue_.timing_group_multiplicity.clear();
    timing_catalogue_.stage_to_timing_group.reserve(
        static_cast<std::size_t>(parallelism_.pipeline_parallel_size));
    for (std::uint64_t stage = 0; stage < parallelism_.pipeline_parallel_size;
         ++stage) {
        config::StageTimingSignature signature =
            make_stage_timing_signature(stage);
        const auto existing =
            std::find(timing_catalogue_.timing_groups.begin(),
                      timing_catalogue_.timing_groups.end(), signature);
        std::uint32_t group_id = 0;
        if (existing == timing_catalogue_.timing_groups.end()) {
            group_id = static_cast<std::uint32_t>(
                timing_catalogue_.timing_groups.size());
            timing_catalogue_.timing_groups.push_back(std::move(signature));
            timing_catalogue_.timing_group_multiplicity.push_back(0);
        } else {
            group_id = static_cast<std::uint32_t>(std::distance(
                timing_catalogue_.timing_groups.begin(), existing));
        }
        timing_catalogue_.stage_to_timing_group.push_back(group_id);
        ++timing_catalogue_.timing_group_multiplicity.at(group_id);
    }
}

std::vector<detail::DenseLayerTimes>
AnalyticalRooflineExecutionTimePredictor::build_stage_layer_times(
    const config::PipelineStageLayerRange &stage_layers,
    const detail::DenseBatch &dense_batch,
    const detail::DenseOperatorPrecisions &precisions) const {
    std::vector<detail::DenseLayerTimes> result;
    result.reserve(static_cast<std::size_t>(stage_layers.size()));
    detail::DenseModel dense_model = make_dense_model(model_, parallelism_);
    // One representative prediction per attention family is enough: layers of
    // the same family in one stage see the same batch and dimensions.
    std::optional<detail::DenseLayerTimes> standard_layer_times;
    std::optional<detail::DenseLayerTimes> mla_layer_times;
    std::optional<detail::DenseLayerTimes> kda_layer_times;
    for (std::uint64_t model_layer = stage_layers.begin;
         model_layer < stage_layers.end; ++model_layer) {
        dense_model.use_kda =
            model_.has_kda() && model_.is_kda_layer(model_layer);
        // KDA and MLA are mutually exclusive on a logical layer.  Non-hybrid
        // MLA checkpoints retain the old path.
        dense_model.use_mla = !dense_model.use_kda && model_.use_mla;
        std::optional<detail::DenseLayerTimes> *representative =
            dense_model.use_kda ? &kda_layer_times
                                : (dense_model.use_mla ? &mla_layer_times
                                                       : &standard_layer_times);
        if (!representative->has_value()) {
            *representative = detail::predict_dense_layer(
                device_, analytical_, dense_model, dense_batch, precisions);
        }
        result.push_back(representative->value());
    }
    return result;
}

double AnalyticalRooflineExecutionTimePredictor::compute_allreduce_ms(
    std::uint64_t activation_bytes) const {
    if (parallelism_.tensor_parallel_size <= 1) {
        return 0.0;
    }
    return communication_backend_->allreduce_ms(
        activation_bytes, parallelism_.tensor_parallel_size, true);
}

double AnalyticalRooflineExecutionTimePredictor::
    compute_dcp_attention_communication_ms(
        const detail::DenseBatch &dense_batch,
        const config::PipelineStageLayerRange &stage_layers,
        double communication_element_bytes) const {
    if (!model_.use_mla || parallelism_.decode_context_parallel_size <= 1 ||
        dense_batch.decode_requests.empty()) {
        return 0.0;
    }
    // DCP collectives are charged per MLA layer, so a stage made entirely of
    // KDA layers pays nothing.  Decide that from model metadata rather than
    // from predicted KDA component times.
    bool has_mla_layer = false;
    for (std::uint64_t model_layer = stage_layers.begin;
         model_layer < stage_layers.end; ++model_layer) {
        if (!(model_.has_kda() && model_.is_kda_layer(model_layer))) {
            has_mla_layer = true;
            break;
        }
    }
    if (!has_mla_layer) {
        return 0.0;
    }

    const std::uint64_t local_query_heads =
        model_.num_query_heads / parallelism_.tensor_parallel_size;
    std::uint64_t decode_tokens = 0;
    for (const detail::AttentionRequestSlice &request :
         dense_batch.decode_requests) {
        decode_tokens += request.query_tokens;
    }
    const std::uint64_t query_bytes = internal::activation_payload_bytes(
        decode_tokens,
        local_query_heads * (model_.kv_lora_rank + model_.qk_rope_head_dim),
        communication_element_bytes);
    const std::uint64_t gathered_output_bytes =
        internal::activation_payload_bytes(
            decode_tokens,
            local_query_heads * parallelism_.decode_context_parallel_size *
                model_.v_head_dim,
            communication_element_bytes);
    return communication_backend_->allgather_ms(
               query_bytes, parallelism_.decode_context_parallel_size, true) +
           communication_backend_->reduce_scatter_ms(
               gathered_output_bytes, parallelism_.decode_context_parallel_size,
               true);
}

detail::MoECommunicationTime
AnalyticalRooflineExecutionTimePredictor::compute_moe_communication(
    const detail::DenseBatch &dense_batch,
    double communication_element_bytes) const {
    return detail::predict_moe_communication(
        *communication_backend_, dense_batch.total_tokens, model_.hidden_size,
        dense_batch.total_tokens * model_.router_topk,
        parallelism_.tensor_parallel_size,
        parallelism_.moe_tensor_parallel_size,
        parallelism_.moe_expert_parallel_size, parallelism_.data_parallel_size,
        false, communication_element_bytes, model_.routed_expert_hidden_size);
}

AnalyticalRooflineExecutionTimePredictor::StageTimingCacheLookup
AnalyticalRooflineExecutionTimePredictor::lookup_stage_timing_template(
    BatchId batch_id, std::uint32_t timing_group_id,
    const detail::DenseBatch &dense_batch,
    const config::PipelineStageLayerRange &stage_layers) const {
    if (!batch_id.valid()) {
        throw ExecutionTimePredictorError(
            "stage timing cache requires a valid batch ID");
    }

    std::uint64_t cache_generation = 0;
    {
        std::lock_guard<std::mutex> lock(timing_cache_mutex_);
        cache_generation = timing_cache_generation_;
        const auto batch_position = timing_cache_by_batch_.find(batch_id);
        if (batch_position != timing_cache_by_batch_.end()) {
            const auto existing = batch_position->second.find(timing_group_id);
            if (existing != batch_position->second.end()) {
                ++timing_cache_hits_total_;
                return StageTimingCacheLookup{
                    existing->second,
                    true,
                    true,
                    timing_cache_hits_total_,
                    timing_cache_misses_total_,
                    timing_cache_unique_templates_total_,
                    timing_cache_entries_};
            }
        }
    }

    // Template construction includes roofline, communication and possibly MoE
    // calculations. Keep that work outside the cache mutex so predictions for
    // unrelated batches can proceed concurrently.
    auto value = std::make_shared<StageTimingCacheValue>();
    const detail::DenseOperatorPrecisions dense_precisions =
        make_dense_operator_precisions(config_);
    value->layer_times =
        build_stage_layer_times(stage_layers, dense_batch, dense_precisions);

    const detail::Precision communication_precision =
        detail::precision_from_string(config_.communication_precision());
    const double communication_element_bytes =
        detail::bytes_per_element(communication_precision);
    const std::uint64_t activation_bytes = internal::activation_payload_bytes(
        dense_batch.total_tokens, model_.hidden_size,
        communication_element_bytes);
    value->allreduce_ms = compute_allreduce_ms(activation_bytes);
    value->dcp_attention_communication_ms =
        compute_dcp_attention_communication_ms(dense_batch, stage_layers,
                                               communication_element_bytes);

    // The representative expert lane lives in the batch-level shared-routing
    // memo. Keeping it out of each timing group avoids recalculating the same
    // routing allocation and expert roofline on every template miss.
    if (model_.is_moe() &&
        config_.moe_layer_event_mode == "stage_group_scaled" &&
        internal::routing_is_layer_invariant(routing_) &&
        internal::has_contiguous_moe_suffix(model_, stage_layers)) {
        value->moe_communication =
            compute_moe_communication(dense_batch, communication_element_bytes);
        value->has_moe_communication = true;
    }

    std::lock_guard<std::mutex> lock(timing_cache_mutex_);
    if (cache_generation != timing_cache_generation_) {
        return StageTimingCacheLookup{std::move(value),
                                      false,
                                      false,
                                      timing_cache_hits_total_,
                                      timing_cache_misses_total_,
                                      timing_cache_unique_templates_total_,
                                      timing_cache_entries_};
    }
    auto &batch_cache = timing_cache_by_batch_[batch_id];
    const auto [position, inserted] =
        batch_cache.emplace(timing_group_id, value);
    if (!inserted) {
        ++timing_cache_hits_total_;
        return StageTimingCacheLookup{position->second,
                                      true,
                                      true,
                                      timing_cache_hits_total_,
                                      timing_cache_misses_total_,
                                      timing_cache_unique_templates_total_,
                                      timing_cache_entries_};
    }
    ++timing_cache_entries_;
    ++timing_cache_misses_total_;
    ++timing_cache_unique_templates_total_;
    return StageTimingCacheLookup{std::move(value),
                                  true,
                                  false,
                                  timing_cache_hits_total_,
                                  timing_cache_misses_total_,
                                  timing_cache_unique_templates_total_,
                                  timing_cache_entries_};
}

std::shared_ptr<const ExecutionTimePrediction>
AnalyticalRooflineExecutionTimePredictor::lookup_complete_prediction(
    BatchId batch_id, std::uint32_t timing_group_id) const {
    std::lock_guard<std::mutex> lock(timing_cache_mutex_);
    const auto batch_position = timing_cache_by_batch_.find(batch_id);
    if (batch_position == timing_cache_by_batch_.end()) {
        return nullptr;
    }
    const auto group_position = batch_position->second.find(timing_group_id);
    if (group_position == batch_position->second.end() ||
        group_position->second->complete_prediction == nullptr) {
        return nullptr;
    }
    ++timing_cache_hits_total_;
    return group_position->second->complete_prediction;
}

void AnalyticalRooflineExecutionTimePredictor::store_complete_prediction(
    BatchId batch_id, std::uint32_t timing_group_id,
    const ExecutionTimePrediction &prediction) const {
    auto stored = std::make_shared<const ExecutionTimePrediction>(prediction);
    std::lock_guard<std::mutex> lock(timing_cache_mutex_);
    const auto batch_position = timing_cache_by_batch_.find(batch_id);
    if (batch_position == timing_cache_by_batch_.end()) {
        throw ExecutionTimePredictorError(
            "complete prediction requires an existing timing template");
    }
    const auto group_position = batch_position->second.find(timing_group_id);
    if (group_position == batch_position->second.end()) {
        throw ExecutionTimePredictorError(
            "complete prediction timing group is not cached");
    }
    group_position->second->complete_prediction = std::move(stored);
}

std::shared_ptr<
    const AnalyticalRooflineExecutionTimePredictor::SharedRoutingValue>
AnalyticalRooflineExecutionTimePredictor::shared_routing_for_batch(
    BatchId batch_id, const detail::DenseBatch &dense_batch) const {
    if (!batch_id.valid()) {
        throw ExecutionTimePredictorError(
            "shared routing memo requires a valid batch ID");
    }
    std::uint64_t cache_generation = 0;
    {
        std::lock_guard<std::mutex> lock(timing_cache_mutex_);
        cache_generation = timing_cache_generation_;
        const auto existing = shared_routing_by_batch_.find(batch_id);
        if (existing != shared_routing_by_batch_.end()) {
            return existing->second;
        }
    }

    auto value = std::make_shared<SharedRoutingValue>();
    // Layer zero is the canonical seed: a shared assignment must not depend on
    // which layer or stage happened to ask for it first.
    value->allocation = detail::route_tokens(
        dense_batch.total_tokens, model_.router_topk, model_.total_expert_num,
        parallelism_.moe_expert_parallel_size, routing_, 0,
        static_cast<std::uint64_t>(batch_id.value()));
    detail::MoEModel moe_model =
        internal::make_moe_model(model_, parallelism_);
    moe_model.decode_only = dense_batch.prefill_requests.empty() &&
                            !dense_batch.decode_requests.empty();
    value->lane_prediction = detail::predict_moe_lanes(
        device_, analytical_, moe_model, value->allocation,
        model_.router_topk, internal::make_moe_operator_precisions(config_));
    std::lock_guard<std::mutex> lock(timing_cache_mutex_);
    if (cache_generation != timing_cache_generation_) {
        return value;
    }
    const auto [position, inserted] =
        shared_routing_by_batch_.emplace(batch_id, value);
    return inserted ? std::move(value) : position->second;
}

void AnalyticalRooflineExecutionTimePredictor::release_batch_timing_cache(
    BatchId batch_id) const {
    if (!batch_id.valid()) {
        return;
    }
    std::lock_guard<std::mutex> lock(timing_cache_mutex_);
    ++timing_cache_generation_;
    shared_routing_by_batch_.erase(batch_id);
    const auto position = timing_cache_by_batch_.find(batch_id);
    if (position == timing_cache_by_batch_.end()) {
        return;
    }
    timing_cache_entries_ -=
        static_cast<std::uint64_t>(position->second.size());
    timing_cache_by_batch_.erase(position);
}

} // namespace frontier::execution_time_predictor
