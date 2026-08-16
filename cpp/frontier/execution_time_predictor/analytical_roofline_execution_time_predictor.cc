#include "frontier/execution_time_predictor/analytical_roofline_execution_time_predictor.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

#include "frontier/cc_backend/analytical_model.h"
#include "frontier/execution_time_predictor/analytical_moe_stage.h"
#include "frontier/execution_time_predictor/analytical_stage_diagnostics.h"

namespace frontier::execution_time_predictor {
namespace {

const entities::Request &
get_request(const std::vector<entities::Request> &requests,
            RequestId request_id) {
    if (!request_id.valid() || request_id.index() >= requests.size()) {
        throw ExecutionTimePredictorError(
            "batch execution references an unknown request");
    }
    const entities::Request &request = requests.at(request_id.index());
    if (request.id() != request_id) {
        throw ExecutionTimePredictorError(
            "batch execution request arena invariant failed");
    }
    return request;
}

struct StageBatchInfo {
    detail::DenseBatch dense_batch;
    std::uint64_t lm_head_tokens = 0;
};

// A chunk covering [start, start + scheduled_tokens) attends to exactly the
// tokens ahead of its own queries, and that offset is fixed for the chunk's
// entire life: no sibling chunk of the same request can move it.  The
// scheduler advances the optimistic frontier before it builds the snapshot,
// so the offset is recoverable from the snapshot alone -- this is the same
// identity VllmV1Scheduler::apply_batch_completion uses to decide which
// completion belongs to which chunk.
std::uint64_t
snapshot_past_context(const entities::RequestBatchSnapshot &snapshot) {
    if (snapshot.scheduler_frontier < snapshot.scheduled_tokens) {
        throw ExecutionTimePredictorError(
            "batch snapshot scheduler frontier underflows");
    }
    return snapshot.scheduler_frontier - snapshot.scheduled_tokens;
}

detail::AttentionRequestSlice
make_attention_request_slice(const entities::RequestBatchSnapshot &snapshot) {
    return detail::AttentionRequestSlice{
        snapshot.scheduled_tokens,
        snapshot_past_context(snapshot),
    };
}

StageBatchInfo
build_stage_batch_info(const entities::Batch &batch,
                       const std::vector<entities::Request> &requests) {
    StageBatchInfo result{};
    result.dense_batch.total_tokens = batch.total_scheduled_tokens();
    for (const entities::RequestBatchSnapshot &snapshot : batch.requests()) {
        const entities::Request &request =
            get_request(requests, snapshot.request_id);
        // Every attention input comes from the batch's own snapshot rather
        // than from live request state.  A request may hold several prefill
        // chunks at once, and its live frontier then advances while an earlier
        // chunk is still traversing the pipeline, so reading it here would make
        // a stage's predicted work depend on when that stage happened to run.
        // That is what made "collapsed" and "exact" pipeline event modes
        // disagree, and it understated attention under both: the live counter
        // only reaches this chunk's true offset once every earlier chunk has
        // retired, which may be after the chunk has already left the pipeline.
        const detail::AttentionRequestSlice slice =
            make_attention_request_slice(snapshot);
        if (slice.past_context < request.num_prefill_tokens()) {
            result.dense_batch.prefill_requests.push_back(slice);
            // The chunk that reaches the end of the prompt is the one that
            // samples the request's first token.
            if (snapshot.scheduler_frontier >= request.num_prefill_tokens()) {
                ++result.lm_head_tokens;
            }
        } else {
            result.dense_batch.decode_requests.push_back(slice);
            ++result.lm_head_tokens;
        }
    }
    return result;
}

detail::DenseModel
make_dense_model(const config::ModelConfig &model,
                 const config::ParallelismConfig &parallelism) {
    detail::DenseModel result{};
    result.hidden_size = model.hidden_size;
    result.intermediate_size = model.dense_intermediate_size;
    result.num_query_heads = model.num_query_heads;
    result.num_kv_heads = model.num_kv_heads;
    result.head_dim = model.head_dim;
    result.tensor_parallel_size = parallelism.tensor_parallel_size;
    result.gated_mlp = model.gated_mlp;
    result.fused_add_norm = model.fused_add_norm;
    result.use_mla = model.use_mla;
    result.mla_use_output_gate = model.mla_use_output_gate;
    result.mla_use_nope = model.mla_use_nope;
    result.use_mfa = model.use_mfa;
    result.q_lora_rank = model.q_lora_rank;
    result.kv_lora_rank = model.kv_lora_rank;
    result.qk_nope_head_dim = model.qk_nope_head_dim;
    result.qk_rope_head_dim = model.qk_rope_head_dim;
    result.qk_head_dim = model.qk_head_dim;
    result.v_head_dim = model.v_head_dim;
    result.share_q_dim = model.share_q_dim;
    result.decode_context_parallel_size =
        parallelism.decode_context_parallel_size;
    // KDA is selected per logical layer in predict_execution.  Keep the
    // dimensions in the compact detail model even when this particular
    // invocation represents an MLA layer so a stage can cheaply switch
    // variants without rebuilding model metadata.
    result.kda_num_heads = model.kda_num_heads;
    result.kda_num_k_heads = model.kda_num_k_heads;
    result.kda_num_v_heads = model.kda_num_v_heads;
    result.kda_key_head_dim = model.kda_key_head_dim;
    result.kda_value_head_dim = model.kda_value_head_dim;
    result.kda_head_dim = model.kda_head_dim;
    result.kda_short_conv_kernel_size = model.kda_short_conv_kernel_size;
    result.kda_conv_state_dim = model.kda_conv_state_dim;
    result.attn_res_block_size = model.attn_res_block_size;
    return result;
}

detail::DenseOperatorPrecisions make_dense_operator_precisions(
    const config::AnalyticalExecutionModelConfig &config) {
    return detail::DenseOperatorPrecisions{
        detail::precision_from_string(config.attention_precision()),
        detail::precision_from_string(config.dense_precision()),
        detail::precision_from_string(config.kv_cache_precision()),
        detail::precision_from_string(config.attention_weight_precision()),
        detail::precision_from_string(config.attention_activation_precision()),
        detail::precision_from_string(config.dense_mlp_weight_precision()),
        detail::precision_from_string(config.dense_mlp_activation_precision()),
        detail::precision_from_string(config.kda_snapshot_precision()),
    };
}

std::uint64_t
model_layer_for_local_moe(const config::ModelConfig &model,
                          const config::PipelineStageLayerRange &stage_layers,
                          std::uint64_t local_moe_layer) {
    std::uint64_t current = 0;
    for (std::uint64_t model_layer = stage_layers.begin;
         model_layer < stage_layers.end; ++model_layer) {
        if (!model.is_moe_layer(model_layer)) {
            continue;
        }
        if (current == local_moe_layer) {
            return model_layer;
        }
        ++current;
    }
    throw ExecutionTimePredictorError(
        "cached MoE layer index is outside the pipeline stage");
}

ExecutionTimePrediction materialize_cached_prediction(
    const ExecutionTimePrediction &cached, const config::ModelConfig &model,
    const config::PipelineStageLayerRange &stage_layers) {
    ExecutionTimePrediction result = cached;
    for (MoERoutingDiagnostic &routing : result.moe_routing) {
        if (!routing.layer_id.valid()) {
            throw ExecutionTimePredictorError(
                "cached MoE routing has an invalid local layer ID");
        }
        routing.model_layer_id = model_layer_for_local_moe(
            model, stage_layers, routing.layer_id.index());
    }
    return result;
}

[[noreturn]] void invalid_predictor_config(std::string_view reason) {
    throw ExecutionTimePredictorError(
        "unsupported analytical model configuration: " + std::string(reason));
}

void validate_predictor_configuration(
    const config::AnalyticalExecutionModelConfig &config,
    const config::ParallelismConfig &parallelism,
    const config::ModelConfig &model,
    const std::shared_ptr<const cc_backend::BaseCCBackend> &backend) {
    if (backend == nullptr) {
        invalid_predictor_config("communication backend is null");
    }

    const std::pair<std::string_view, std::string_view> precisions[] = {
        {"default", config.precision},
        {"attention", config.attention_precision()},
        {"dense", config.dense_precision()},
        {"moe_expert", config.moe_expert_precision()},
        {"moe_router", config.moe_router_precision()},
        {"kv_cache", config.kv_cache_precision()},
        {"communication", config.communication_precision()},
        {"attention_weight", config.attention_weight_precision()},
        {"attention_activation", config.attention_activation_precision()},
        {"dense_weight", config.dense_weight_precision()},
        {"dense_activation", config.dense_activation_precision()},
        {"routed_expert_weight", config.routed_expert_weight_precision()},
        {"routed_expert_activation",
         config.routed_expert_activation_precision()},
        {"latent_moe_projection_weight",
         config.latent_moe_projection_weight_precision()},
        {"latent_moe_projection_activation",
         config.latent_moe_projection_activation_precision()},
        {"shared_expert_weight", config.shared_expert_weight_precision()},
        {"shared_expert_activation",
         config.shared_expert_activation_precision()},
        {"dense_mlp_weight", config.dense_mlp_weight_precision()},
        {"dense_mlp_activation", config.dense_mlp_activation_precision()},
        {"router_weight_storage", config.router_weight_storage_precision()},
        {"moe_router_activation", config.moe_router_activation_precision()},
        {"router_compute", config.router_compute_precision()},
        {"kda_snapshot", config.kda_snapshot_precision()},
        {"lm_head_weight", config.lm_head_weight_precision()},
        {"lm_head_activation", config.lm_head_activation_precision()},
    };
    for (const auto &[name, value] : precisions) {
        try {
            static_cast<void>(detail::precision_from_string(value));
        } catch (const detail::AnalyticalModelError &) {
            invalid_predictor_config(std::string(name) +
                                     " precision is unsupported");
        }
    }

    if (config.moe_layer_event_mode != "detailed" &&
        config.moe_layer_event_mode != "first_layer_scaled" &&
        config.moe_layer_event_mode != "stage_group_scaled") {
        invalid_predictor_config("unknown MoE layer event mode");
    }
    if (model.attention.memory_layout ==
        attention::AttentionMemoryLayout::kFrozenDsa) {
        invalid_predictor_config("frozen DSA attention is unsupported");
    }
    if (parallelism.tensor_parallel_size == 0 ||
        parallelism.decode_context_parallel_size == 0 ||
        parallelism.pipeline_parallel_size == 0) {
        invalid_predictor_config("parallelism sizes must be positive");
    }
    if (parallelism.tensor_parallel_size %
            parallelism.decode_context_parallel_size !=
        0) {
        invalid_predictor_config("DCP size must divide TP size");
    }
    if (parallelism.decode_context_parallel_size > 1 && !model.use_mla) {
        invalid_predictor_config("DCP requires MLA");
    }
    if (parallelism.pipeline_parallel_size > model.num_layers) {
        invalid_predictor_config("PP size exceeds the model layer count");
    }
}

} // namespace

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
                device_, detail::AnalyticalConfig{}, dense_model, dense_batch,
                precisions);
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
        parallelism_.moe_expert_parallel_size, routing_, 0);
    value->lane_prediction = detail::predict_moe_lanes(
        device_, detail::AnalyticalConfig{},
        internal::make_moe_model(model_, parallelism_), value->allocation,
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

AnalyticalRooflineExecutionTimePredictor::
    AnalyticalRooflineExecutionTimePredictor(
        config::AnalyticalExecutionModelConfig config)
    : AnalyticalRooflineExecutionTimePredictor(
          config,
          [&]() {
              config::ParallelismConfig value{};
              value.tensor_parallel_size = config.tensor_parallel_size;
              return value;
          }(),
          config::ModelConfig{}, config::MoeRoutingConfig{}) {}

AnalyticalRooflineExecutionTimePredictor::
    AnalyticalRooflineExecutionTimePredictor(
        config::AnalyticalExecutionModelConfig config,
        config::ParallelismConfig parallelism)
    : AnalyticalRooflineExecutionTimePredictor(std::move(config), parallelism,
                                               config::ModelConfig{},
                                               config::MoeRoutingConfig{}) {}

AnalyticalRooflineExecutionTimePredictor::
    AnalyticalRooflineExecutionTimePredictor(
        config::AnalyticalExecutionModelConfig config,
        config::ParallelismConfig parallelism, config::ModelConfig model,
        config::MoeRoutingConfig routing)
    : AnalyticalRooflineExecutionTimePredictor(
          config, std::move(parallelism), std::move(model), std::move(routing),
          cc_backend::make_analytical_cc_backend([&]() {
              cc_backend::AnalyticalCommunicationConfig value{};
              value.network_bandwidth_gbps = config.network_bandwidth_gbps;
              value.latency_us = config.network_latency_us;
              value.intra_node_bandwidth_gbps =
                  config.intra_node_bandwidth_gbps;
              return value;
          }())) {}

AnalyticalRooflineExecutionTimePredictor::
    AnalyticalRooflineExecutionTimePredictor(
        config::AnalyticalExecutionModelConfig config,
        config::ParallelismConfig parallelism, config::ModelConfig model,
        config::MoeRoutingConfig routing,
        std::shared_ptr<const cc_backend::BaseCCBackend> communication_backend)
    : config_(std::move(config)),
      device_(detail::DeviceCeilings::from_config(config_)),
      parallelism_(parallelism), model_(std::move(model)), routing_(routing),
      communication_backend_(std::move(communication_backend)) {
    config::apply_model_native_precision_defaults(config_, model_);
    if (parallelism_.tensor_parallel_size == 0) {
        parallelism_.tensor_parallel_size = config_.tensor_parallel_size;
    }
    validate_predictor_configuration(config_, parallelism_, model_,
                                     communication_backend_);
    build_stage_timing_groups();
}

ExecutionTimePrediction
AnalyticalRooflineExecutionTimePredictor::predict_stage_execution_time(
    const entities::Batch &batch,
    const std::vector<entities::Request> &requests, StageId stage_id) const {
    return predict_execution(batch, requests, stage_id, std::nullopt);
}

MoEGroupLayerPrediction
AnalyticalRooflineExecutionTimePredictor::predict_moe_group_layer(
    const MoEGroupLayerInput &input) const {
    if (!input.layer_id.valid() || input.input_tokens == 0 ||
        input.global_expert_tokens.size() != model_.total_expert_num ||
        input.fallback_lane_times_ms.size() !=
            parallelism_.moe_expert_parallel_size) {
        throw ExecutionTimePredictorError(
            "analytical group MoE input has an invalid domain");
    }
    std::uint64_t routed_tokens = 0;
    for (const std::uint64_t count : input.global_expert_tokens) {
        if (count > std::numeric_limits<std::uint64_t>::max() - routed_tokens) {
            throw ExecutionTimePredictorError(
                "analytical group MoE routed token count overflows");
        }
        routed_tokens += count;
    }
    if (routed_tokens != input.routed_tokens ||
        input.input_tokens >
            std::numeric_limits<std::uint64_t>::max() / model_.router_topk ||
        input.input_tokens * model_.router_topk != input.routed_tokens) {
        throw ExecutionTimePredictorError(
            "analytical group MoE token counts are inconsistent");
    }

    const detail::ExpertParallelDomain domain(
        model_.total_expert_num, parallelism_.moe_expert_parallel_size);
    detail::RoutingAllocation allocation{};
    allocation.input_tokens = input.input_tokens;
    allocation.routed_tokens = input.routed_tokens;
    allocation.global_expert_tokens = input.global_expert_tokens;
    allocation.lane_expert_tokens =
        domain.partition(allocation.global_expert_tokens);
    const detail::MoELanePrediction prediction = detail::predict_moe_lanes(
        device_, detail::AnalyticalConfig{},
        internal::make_moe_model(model_, parallelism_), allocation,
        model_.router_topk, internal::make_moe_operator_precisions(config_));

    MoEGroupLayerPrediction result{};
    result.lane_times_ms = internal::lane_times_ms(prediction);
    result.critical_lane = prediction.critical_lane;
    result.critical_lane_time_ms = prediction.critical_lane_time_ms;
    return result;
}

ExecutionTimePrediction
AnalyticalRooflineExecutionTimePredictor::prepare_moe_stage_execution(
    const entities::Batch &batch,
    const std::vector<entities::Request> &requests, StageId stage_id) const {
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
    const entities::Batch &batch,
    const std::vector<entities::Request> &requests, StageId stage_id,
    std::uint64_t local_moe_layer) const {
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
    const entities::Batch &batch,
    const std::vector<entities::Request> &requests, StageId stage_id,
    std::optional<std::uint64_t> selected_moe_layer) const {
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
            device_, detail::AnalyticalConfig{}, batch_info.lm_head_tokens,
            model_.hidden_size, model_.vocab_size,
            parallelism_.tensor_parallel_size,
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
