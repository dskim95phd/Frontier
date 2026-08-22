#include "frontier/execution_time_predictor/analytical_roofline_execution_time_predictor.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>

#include "frontier/execution_time_predictor/analytical_moe_stage.h"

namespace frontier::execution_time_predictor {

MoEGroupLayerPrediction
AnalyticalRooflineExecutionTimePredictor::predict_moe_group_layer(
    const MoEGroupLayerInput &input) const {
    if (!std::isfinite(analytical_.ep_shared_routed_overlap_fraction) ||
        analytical_.ep_shared_routed_overlap_fraction < 0.0 ||
        analytical_.ep_shared_routed_overlap_fraction > 1.0) {
        throw ExecutionTimePredictorError(
            "EP shared/routed overlap fraction must be in [0, 1]");
    }
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
    const std::size_t ep_size =
        static_cast<std::size_t>(parallelism_.moe_expert_parallel_size);
    if (input.source_lane_routed_tokens.empty() ||
        input.source_lane_routed_tokens.size() !=
            input.source_lane_unique_tokens.size() ||
        input.source_lane_routed_tokens.size() !=
            input.source_input_tokens.size()) {
        throw ExecutionTimePredictorError(
            "analytical group MoE source traffic matrix is missing");
    }
    std::uint64_t source_input_total = 0;
    for (const std::uint64_t tokens : input.source_input_tokens) {
        if (tokens >
            std::numeric_limits<std::uint64_t>::max() - source_input_total) {
            throw ExecutionTimePredictorError(
                "analytical group MoE source input token count overflows");
        }
        source_input_total += tokens;
    }
    if (source_input_total != input.input_tokens) {
        throw ExecutionTimePredictorError(
            "analytical group MoE source input tokens do not conserve the "
            "group total");
    }
    allocation.lane_routed_tokens.assign(ep_size, 0);
    allocation.lane_unique_tokens.assign(ep_size, 0);
    for (std::size_t source = 0;
         source < input.source_lane_routed_tokens.size(); ++source) {
        const auto &routed_row = input.source_lane_routed_tokens.at(source);
        const auto &unique_row = input.source_lane_unique_tokens.at(source);
        if (routed_row.size() != ep_size || unique_row.size() != ep_size) {
            throw ExecutionTimePredictorError(
                "analytical group MoE source traffic row has invalid EP "
                "width");
        }
        for (std::size_t lane = 0; lane < ep_size; ++lane) {
            if (routed_row.at(lane) >
                    std::numeric_limits<std::uint64_t>::max() -
                        allocation.lane_routed_tokens.at(lane) ||
                unique_row.at(lane) >
                    std::numeric_limits<std::uint64_t>::max() -
                        allocation.lane_unique_tokens.at(lane)) {
                throw ExecutionTimePredictorError(
                    "analytical group MoE source traffic overflows");
            }
            allocation.lane_routed_tokens.at(lane) += routed_row.at(lane);
            allocation.lane_unique_tokens.at(lane) += unique_row.at(lane);
        }
    }
    const std::uint64_t lane_routed_total =
        std::accumulate(allocation.lane_routed_tokens.begin(),
                        allocation.lane_routed_tokens.end(), std::uint64_t{0});
    if (lane_routed_total != allocation.routed_tokens) {
        throw ExecutionTimePredictorError(
            "analytical group MoE source traffic does not conserve routed "
            "tokens");
    }
    for (std::size_t lane = 0; lane < ep_size; ++lane) {
        const std::uint64_t expert_lane_total = std::accumulate(
            allocation.lane_expert_tokens.at(lane).begin(),
            allocation.lane_expert_tokens.at(lane).end(), std::uint64_t{0});
        if (expert_lane_total != allocation.lane_routed_tokens.at(lane) ||
            allocation.lane_unique_tokens.at(lane) > allocation.input_tokens) {
            throw ExecutionTimePredictorError(
                "analytical group MoE destination traffic is inconsistent "
                "with expert routing");
        }
    }
    allocation.lane_active_experts.reserve(ep_size);
    for (const auto &lane : allocation.lane_expert_tokens) {
        allocation.lane_active_experts.push_back(static_cast<std::uint64_t>(
            std::count_if(lane.begin(), lane.end(),
                          [](std::uint64_t tokens) { return tokens > 0; })));
    }
    const detail::MoEModel moe_model =
        internal::make_moe_model(model_, parallelism_);
    const detail::MoEOperatorPrecisions moe_precisions =
        internal::make_moe_operator_precisions(config_);
    const detail::MoERoutedLanePrediction prediction =
        detail::predict_routed_moe_lanes(device_, analytical_, moe_model,
                                         allocation, model_.router_topk,
                                         moe_precisions, true);

    MoEGroupLayerPrediction result{};
    result.lane_times_ms = prediction.lane_times_ms;
    result.lane_times_are_routed_only = true;
    result.source_shared_expert_overlap_fraction =
        analytical_.overlap_shared_routed_moe
            ? analytical_.ep_shared_routed_overlap_fraction
            : 0.0;
    result.critical_lane = prediction.critical_lane;
    result.critical_lane_time_ms = prediction.critical_lane_time_ms;
    const detail::MoEGroupedGemmGeometry &geometry =
        prediction.critical_lane_geometry;
    result.grouped_gemm_geometry = {
        geometry.enabled,
        geometry.block_m,
        geometry.routed_m_blocks,
        geometry.shared_m_blocks,
        geometry.routed_padded_tokens,
        geometry.shared_padded_tokens,
        geometry.up_cluster_tasks,
        geometry.down_cluster_tasks,
        geometry.up_wave_utilization,
        geometry.down_wave_utilization,
        geometry.up_cluster_task_overhead_ms,
        geometry.down_cluster_task_overhead_ms,
    };
    result.destination_lane_routed_tokens = allocation.lane_routed_tokens;
    result.destination_lane_unique_tokens = allocation.lane_unique_tokens;
    if (config_.moe_communication_backend == "sm100_megamoe_public" &&
        parallelism_.moe_expert_parallel_size > 1) {
        if (input.source_shared_expert_path_ms.size() !=
            input.source_input_tokens.size()) {
            throw ExecutionTimePredictorError(
                "analytical group MoE requires one shared-expert path per "
                "DP source");
        }
        double maximum_shared_expert_ms = 0.0;
        for (const double source_shared_expert_ms :
             input.source_shared_expert_path_ms) {
            if (!std::isfinite(source_shared_expert_ms) ||
                source_shared_expert_ms < 0.0) {
                throw ExecutionTimePredictorError(
                    "analytical group MoE shared-expert path is invalid");
            }
            maximum_shared_expert_ms =
                std::max(maximum_shared_expert_ms, source_shared_expert_ms);
        }
        // Preserve the historical overlap window's shared-expert component,
        // but evaluate it per source instead of on the DP-summed token count.
        const double overlap_credit_ms =
            result.source_shared_expert_overlap_fraction *
            std::min(prediction.critical_lane_time_ms,
                     maximum_shared_expert_ms);
        const double fused_expert_compute_ms =
            prediction.critical_lane_time_ms + maximum_shared_expert_ms -
            overlap_credit_ms;
        std::uint64_t maximum_source_unique_token_copies = 0;
        for (const auto &source_row : input.source_lane_unique_tokens) {
            std::uint64_t source_unique_token_copies = 0;
            for (const std::uint64_t tokens : source_row) {
                if (tokens > std::numeric_limits<std::uint64_t>::max() -
                                 source_unique_token_copies) {
                    throw ExecutionTimePredictorError(
                        "analytical group MoE source logical traffic "
                        "overflows");
                }
                source_unique_token_copies += tokens;
            }
            maximum_source_unique_token_copies =
                std::max(maximum_source_unique_token_copies,
                         source_unique_token_copies);
        }
        const detail::Precision communication_precision =
            detail::precision_from_string(config_.communication_precision());
        const detail::MoECommunicationTime communication =
            detail::predict_moe_communication(
                *communication_backend_, input.input_tokens, model_.hidden_size,
                input.routed_tokens, parallelism_.tensor_parallel_size,
                parallelism_.moe_tensor_parallel_size,
                parallelism_.moe_expert_parallel_size,
                parallelism_.data_parallel_size, false,
                detail::bytes_per_element(communication_precision),
                model_.routed_expert_hidden_size,
                config_.moe_communication_backend, &allocation,
                fused_expert_compute_ms, analytical_.moe_a2a_overlap_residual,
                analytical_.mega_moe_a2a_bandwidth_scale,
                analytical_.mega_moe_a2a_startup_scale,
                detail::bytes_per_element(detail::precision_from_string(
                    config_.routed_expert_activation_precision())),
                maximum_source_unique_token_copies);
        result.has_source_aware_ep_communication = true;
        result.raw_ep_dispatch_ms = communication.raw_ep_dispatch_ms;
        result.raw_ep_combine_ms = communication.raw_ep_combine_ms;
        result.ep_dispatch_ms = communication.ep_dispatch_ms;
        result.ep_combine_ms = communication.ep_combine_ms;
    }
    return result;
}

} // namespace frontier::execution_time_predictor
