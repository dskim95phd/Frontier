#include "frontier/execution_time_predictor/analytical_stage_diagnostics.h"

#include "frontier/attention/mla.h"
#include "frontier/execution_time_predictor/analytical_roofline_primitives.h"

namespace frontier::execution_time_predictor::detail {
namespace {

double kv_cache_bytes_per_token_per_layer(const config::ModelConfig &model,
                                          Precision precision) {
    if (model.use_mla) {
        const double bytes =
            attention::mla_kv_cache_bytes_per_token(attention::MlaKvCacheLayout{
                model.kv_lora_rank,
                model.qk_rope_head_dim,
                bytes_per_element(precision),
                2.0,
            });
        if (model.has_kda() && model.num_layers != 0) {
            return bytes * static_cast<double>(model.num_mla_layers) /
                   static_cast<double>(model.num_layers);
        }
        return bytes;
    }
    return static_cast<double>(model.runtime_num_kv_heads()) *
           static_cast<double>(model.runtime_head_size()) *
           static_cast<double>(model.kv_factor()) *
           bytes_per_element(precision);
}

} // namespace

std::vector<std::pair<std::string, double>>
build_stage_diagnostics(const StageDiagnosticsInput &input) {
    std::uint64_t kda_layer_count = 0;
    for (std::uint64_t layer = input.stage_layers.begin;
         layer < input.stage_layers.end; ++layer) {
        kda_layer_count += static_cast<std::uint64_t>(
            input.model.has_kda() && input.model.is_kda_layer(layer));
    }
    const std::uint64_t layers_per_stage = input.stage_layers.size();
    const std::uint64_t mla_layer_count =
        input.model.use_mla ? layers_per_stage - kda_layer_count : 0;

    double kda_projection_ms = 0.0;
    double kda_short_conv_ms = 0.0;
    double kda_recurrent_ms = 0.0;
    double kda_gate_norm_ms = 0.0;
    double attn_res_ms = 0.0;
    for (const DenseLayerTimes &times : input.layer_times) {
        kda_projection_ms += times.kda_projection_ms;
        kda_short_conv_ms += times.kda_short_conv_ms;
        kda_recurrent_ms += times.kda_recurrent_ms;
        kda_gate_norm_ms += times.kda_gate_norm_ms;
        attn_res_ms += times.attn_res_ms;
    }

    const double kv_cache_bytes = kv_cache_bytes_per_token_per_layer(
        input.model, input.precisions.kv_cache);
    const double rank_local_kv_cache_bytes =
        input.model.use_mla
            ? kv_cache_bytes /
                  static_cast<double>(
                      input.parallelism.decode_context_parallel_size)
            : kv_cache_bytes;
    const auto &timing_group =
        input.timing_catalogue.timing_groups.at(input.timing_group_id);

    return {
        {"total_tokens", static_cast<double>(input.batch.total_tokens)},
        {"stage_id", static_cast<double>(input.stage)},
        {"timing_group_id", static_cast<double>(input.timing_group_id)},
        {"timing_group_multiplicity",
         static_cast<double>(input.timing_group_multiplicity)},
        {"timing_cache_enabled", input.timing_cache.enabled ? 1.0 : 0.0},
        {"timing_cache_hit", input.timing_cache.hit ? 1.0 : 0.0},
        {"timing_cache_miss",
         input.timing_cache.enabled && !input.timing_cache.hit ? 1.0 : 0.0},
        {"timing_cache_hits_total",
         static_cast<double>(input.timing_cache.hits_total)},
        {"timing_cache_misses_total",
         static_cast<double>(input.timing_cache.misses_total)},
        {"timing_cache_unique_templates_total",
         static_cast<double>(input.timing_cache.unique_templates_total)},
        {"timing_cache_entries",
         static_cast<double>(input.timing_cache.entries)},
        {"timing_group_layer_count",
         static_cast<double>(timing_group.ordered_layers.size())},
        {"timing_owns_input_embedding",
         timing_group.owns_input_embedding ? 1.0 : 0.0},
        {"timing_owns_final_norm", timing_group.owns_final_norm ? 1.0 : 0.0},
        {"timing_owns_lm_head", timing_group.owns_lm_head ? 1.0 : 0.0},
        {"timing_emits_pp_send", timing_group.emits_pp_send ? 1.0 : 0.0},
        {"stage_group_requested", input.stage_group_requested ? 1.0 : 0.0},
        {"stage_group_active", input.stage_group_active ? 1.0 : 0.0},
        {"stage_group_routing_layer_invariant",
         input.routing_is_layer_invariant ? 1.0 : 0.0},
        {"prefill_request_count",
         static_cast<double>(input.batch.prefill_requests.size())},
        {"decode_request_count",
         static_cast<double>(input.batch.decode_requests.size())},
        {"dense_layer_compute_ms", input.first_layer_compute_ms},
        {"attention_weight_element_bytes",
         bytes_per_element(*input.precisions.attention_weight)},
        {"attention_activation_element_bytes",
         bytes_per_element(*input.precisions.attention_activation)},
        {"dense_weight_element_bytes",
         bytes_per_element(*input.precisions.dense_weight)},
        {"dense_activation_element_bytes",
         bytes_per_element(*input.precisions.dense_activation)},
        {"routed_expert_weight_element_bytes",
         bytes_per_element(precision_from_string(
             input.config.routed_expert_weight_precision()))},
        {"routed_expert_activation_element_bytes",
         bytes_per_element(precision_from_string(
             input.config.routed_expert_activation_precision()))},
        {"latent_moe_projection_weight_element_bytes",
         bytes_per_element(precision_from_string(
             input.config.latent_moe_projection_weight_precision()))},
        {"latent_moe_projection_activation_element_bytes",
         bytes_per_element(precision_from_string(
             input.config.latent_moe_projection_activation_precision()))},
        {"shared_expert_weight_element_bytes",
         bytes_per_element(precision_from_string(
             input.config.shared_expert_weight_precision()))},
        {"shared_expert_activation_element_bytes",
         bytes_per_element(precision_from_string(
             input.config.shared_expert_activation_precision()))},
        {"router_weight_storage_element_bytes",
         bytes_per_element(precision_from_string(
             input.config.router_weight_storage_precision()))},
        {"router_activation_storage_element_bytes",
         bytes_per_element(precision_from_string(
             input.config.moe_router_activation_precision()))},
        {"router_compute_element_bytes",
         bytes_per_element(
             precision_from_string(input.config.router_compute_precision()))},
        {"kda_snapshot_element_bytes",
         bytes_per_element(
             precision_from_string(input.config.kda_snapshot_precision()))},
        {"kv_cache_element_bytes",
         bytes_per_element(input.precisions.kv_cache)},
        {"kv_cache_bytes_per_token_per_layer", kv_cache_bytes},
        {"kv_cache_rank_local_bytes_per_token_per_layer",
         rank_local_kv_cache_bytes},
        {"communication_element_bytes", input.communication_element_bytes},
        {"tp_allreduce_ms", input.allreduce_ms},
        {"dcp_attention_communication_ms",
         input.dcp_attention_communication_ms},
        {"decode_context_parallel_size",
         static_cast<double>(input.parallelism.decode_context_parallel_size)},
        {"dense_layer_total_ms",
         input.first_layer_compute_ms + input.first_tp_layer_ms},
        {"num_layers", static_cast<double>(layers_per_stage)},
        {"dense_compute_ms", input.dense_compute_ms},
        {"tp_communication_ms", input.tp_communication_ms},
        {"kda_layer_count", static_cast<double>(kda_layer_count)},
        {"mla_layer_count", static_cast<double>(mla_layer_count)},
        {"kda_projection_ms", kda_projection_ms},
        {"kda_short_conv_ms", kda_short_conv_ms},
        {"kda_recurrent_ms", kda_recurrent_ms},
        {"kda_gate_norm_ms", kda_gate_norm_ms},
        {"attn_res_ms", attn_res_ms},
        {"routed_expert_hidden_size",
         static_cast<double>(input.model.routed_expert_hidden_size)},
        {"latent_moe_use_norm", input.model.latent_moe_use_norm ? 1.0 : 0.0},
        {"attn_res_block_size",
         static_cast<double>(input.model.attn_res_block_size)},
        {"pp_communication_ms", input.pp_communication_ms},
        {"lm_head_ms", input.execution_time.lm_head_ms},
        {"lm_head_tokens", static_cast<double>(input.lm_head_tokens)},
        {"stage_duration_ms", input.duration_ms},
        {"batch_duration_ms", input.duration_ms},
    };
}

} // namespace frontier::execution_time_predictor::detail
