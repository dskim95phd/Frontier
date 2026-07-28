"""Profile-free analytical execution-time predictor.

The model is deliberately transparent: every kernel is reduced to FLOPs,
HBM bytes, a precision-specific device ceiling, efficiency priors, launch
latency, and a configurable amount of non-overlapped compute/memory service.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import math
from typing import Dict, Optional

import numpy as np

from frontier.config import (
    AnalyticalRooflineExecutionTimePredictorConfig,
    BaseReplicaSchedulerConfig,
    MetricsConfig,
    PrecisionType,
    ReplicaConfig,
    get_quantization_manager,
)
from frontier.entities import Batch, ExecutionTime
from frontier.entities.time_components import AttentionTime, MLPTime, MoETime
from frontier.execution_time_predictor.base_execution_time_predictor import (
    BaseExecutionTimePredictor,
)
from frontier.logger import init_logger
from frontier.types import ClusterType

logger = init_logger(__name__)


@dataclass(frozen=True)
class AnalyticalOperatorDiagnostic:
    operator_name: str
    phase: str
    precision: str
    local_shape: str
    flops: float
    hbm_bytes: float
    compute_efficiency: float
    memory_efficiency: float
    compute_time_ms: float
    memory_time_ms: float
    launch_time_ms: float
    overlap_penalty: float
    predicted_time_ms: float
    bottleneck: str


class AnalyticalRooflineExecutionTimePredictor(BaseExecutionTimePredictor):
    """FLOP/byte roofline predictor that does not load profiling CSV files."""

    def __init__(
        self,
        predictor_config: AnalyticalRooflineExecutionTimePredictorConfig,
        replica_config: ReplicaConfig,
        replica_scheduler_config: BaseReplicaSchedulerConfig,
        metrics_config: MetricsConfig,
        cluster_config=None,
        model_manager=None,
        cluster_type: ClusterType = None,
        training_file_paths=None,
        actual_replica_ids=None,
        cc_backend=None,
    ) -> None:
        del cluster_config, model_manager, training_file_paths, actual_replica_ids
        self._cc_backend = cc_backend
        self._cluster_type = cluster_type or ClusterType.MONOLITHIC
        self._diagnostics = deque(
            maxlen=int(predictor_config.max_diagnostic_records)
        )
        super().__init__(
            predictor_config,
            replica_config,
            replica_scheduler_config,
            metrics_config,
        )
        self._moe_routing_mode = str(
            getattr(replica_config, "moe_routing_mode", "simulation")
        ).strip()
        self._moe_routing_seed = int(
            getattr(replica_config, "moe_routing_seed", 42)
        )
        self._moe_routing_distribution_type = str(
            getattr(
                replica_config,
                "moe_routing_distribution_type",
                "balanced",
            )
        ).strip().lower()
        if self._moe_routing_mode not in {
            "simulation",
            "uniform_legacy",
            "uniform_random",
        }:
            raise ValueError(
                "moe_routing_mode must be one of "
                "['simulation', 'uniform_legacy', 'uniform_random'], "
                f"got {self._moe_routing_mode!r}"
            )
        if self._moe_routing_seed < 0:
            raise ValueError(
                "moe_routing_seed must be non-negative, "
                f"got {self._moe_routing_seed}"
            )
        if self._moe_routing_distribution_type not in {
            "balanced",
            "random",
            "skewed",
            "zipf",
        }:
            raise ValueError(
                "moe_routing_distribution_type must be one of "
                "['balanced', 'random', 'skewed', 'zipf'], "
                f"got {self._moe_routing_distribution_type!r}"
            )
        if self._model_config.uses_mla():
            raise NotImplementedError(
                "analytical_roofline MVP supports dense-KV attention; MLA needs "
                "a separate latent-cache operator model"
            )
        self._quantization_manager = get_quantization_manager()
        self._quantization_manager.configure_from_model_config(self._model_config)
        device = self._replica_config.device_config
        if float(getattr(device, "hbm_bandwidth_tbps", 0.0)) <= 0.0:
            raise ValueError(
                "analytical_roofline requires device_config.hbm_bandwidth_tbps > 0; "
                f"device={self._replica_config.device!r}"
            )

    def _initialize_normal_mode(self) -> None:
        # Unlike sklearn predictors, there is no training or profile loading.
        return None

    def get_diagnostics(self) -> list[dict]:
        """Return auditable operator predictions accumulated so far."""

        return [asdict(item) for item in self._diagnostics]

    def clear_diagnostics(self) -> None:
        self._diagnostics.clear()

    def _precision(self, op_name: str, cluster_type: ClusterType) -> PrecisionType:
        return self._quantization_manager.get_precision(op_name, cluster_type)

    def _peak_tflops(self, precision: PrecisionType) -> float:
        device = self._replica_config.device_config
        if precision in (PrecisionType.FP16, PrecisionType.BF16):
            peak = float(device.fp16_tflops)
        elif precision is PrecisionType.FP32:
            peak = float(getattr(device, "fp32_tflops", 0.0))
        elif precision in (PrecisionType.FP8, PrecisionType.INT8):
            peak = float(getattr(device, "fp8_tflops", 0.0))
        elif precision in (PrecisionType.FP4, PrecisionType.INT4):
            peak = float(getattr(device, "fp4_tflops", 0.0))
        else:  # pragma: no cover - future enum member
            peak = 0.0
        if peak <= 0.0:
            raise ValueError(
                f"No analytical compute ceiling for precision={precision.value} "
                f"on device={self._replica_config.device!r}"
            )
        return peak

    def predict_roofline_kernel(
        self,
        *,
        operator_name: str,
        phase: str,
        local_shape: str,
        flops: float,
        hbm_bytes: float,
        precision: PrecisionType,
        compute_efficiency: float,
        memory_efficiency: float,
        overlap_penalty: float,
    ) -> float:
        """Apply launch + max(Tc,Tm) + rho*min(Tc,Tm), returning milliseconds."""

        resolved_flops = float(flops)
        resolved_bytes = float(hbm_bytes)
        if resolved_flops < 0.0 or resolved_bytes < 0.0:
            raise ValueError("Roofline FLOPs and HBM bytes must be non-negative")
        if resolved_flops == 0.0 and resolved_bytes == 0.0:
            return 0.0

        compute_time_ms = (
            resolved_flops
            / (self._peak_tflops(precision) * 1e12 * compute_efficiency)
            * 1e3
        )
        memory_time_ms = (
            resolved_bytes
            / (
                float(self._replica_config.device_config.hbm_bandwidth_tbps)
                * 1e12
                * memory_efficiency
            )
            * 1e3
        )
        launch_time_ms = float(self._config.kernel_launch_latency_us) / 1e3
        predicted_time_ms = (
            launch_time_ms
            + max(compute_time_ms, memory_time_ms)
            + overlap_penalty * min(compute_time_ms, memory_time_ms)
        )
        if launch_time_ms >= max(compute_time_ms, memory_time_ms):
            bottleneck = "LAUNCH"
        elif compute_time_ms >= memory_time_ms:
            bottleneck = "COMPUTE"
        else:
            bottleneck = "HBM"
        if self._config.keep_diagnostics:
            diagnostic = AnalyticalOperatorDiagnostic(
                operator_name=operator_name,
                phase=phase,
                precision=precision.value,
                local_shape=local_shape,
                flops=resolved_flops,
                hbm_bytes=resolved_bytes,
                compute_efficiency=float(compute_efficiency),
                memory_efficiency=float(memory_efficiency),
                compute_time_ms=compute_time_ms,
                memory_time_ms=memory_time_ms,
                launch_time_ms=launch_time_ms,
                overlap_penalty=float(overlap_penalty),
                predicted_time_ms=predicted_time_ms,
                bottleneck=bottleneck,
            )
            self._diagnostics.append(diagnostic)
            logger.debug("[ANALYTICAL_ROOFLINE] %s", asdict(diagnostic))
        return predicted_time_ms

    def _phase(self, batch: Batch) -> str:
        if batch.num_prefill_tokens and batch.num_decode_tokens:
            return "mixed"
        if batch.num_prefill_tokens:
            return "prefill"
        return "decode"

    def _tokens(self, batch: Batch, cluster_type: Optional[ClusterType] = None) -> int:
        return int(
            batch.get_effective_total_tokens_for_compute(
                cluster_type or self._cluster_type
            )
        )

    def _gemm(
        self,
        op_name: str,
        batch: Batch,
        m: int,
        k: int,
        n: int,
        cluster_type: ClusterType,
        *,
        operator_class: str = "gemm",
        weight_multiplier: int = 1,
    ) -> float:
        if m <= 0 or k <= 0 or n <= 0:
            return 0.0
        precision = self._precision(op_name, cluster_type)
        element_bytes = precision.bytes_per_element
        flops = 2.0 * m * k * n * weight_multiplier
        hbm_bytes = element_bytes * (
            m * k + weight_multiplier * k * n + weight_multiplier * m * n
        )
        if operator_class == "moe":
            compute_efficiency = self._config.moe_compute_efficiency
            memory_efficiency = self._config.moe_memory_efficiency
            overlap_penalty = self._config.moe_overlap_penalty
        elif m < self._config.small_gemm_token_threshold:
            compute_efficiency = self._config.small_gemm_compute_efficiency
            memory_efficiency = self._config.small_gemm_memory_efficiency
            overlap_penalty = self._config.small_gemm_overlap_penalty
        else:
            compute_efficiency = self._config.large_gemm_compute_efficiency
            memory_efficiency = self._config.large_gemm_memory_efficiency
            overlap_penalty = self._config.large_gemm_overlap_penalty
        return self.predict_roofline_kernel(
            operator_name=op_name,
            phase=self._phase(batch),
            local_shape=f"M={m},K={k},N={n},multiplicity={weight_multiplier}",
            flops=flops,
            hbm_bytes=hbm_bytes,
            precision=precision,
            compute_efficiency=compute_efficiency,
            memory_efficiency=memory_efficiency,
            overlap_penalty=overlap_penalty,
        )

    def _streaming(
        self,
        op_name: str,
        batch: Batch,
        *,
        elements_read: float,
        elements_written: float,
        flops: float,
        cluster_type: ClusterType,
        operator_class: str = "streaming",
    ) -> float:
        if elements_read <= 0 and elements_written <= 0 and flops <= 0:
            return 0.0
        precision = self._precision(op_name, cluster_type)
        if operator_class == "routing":
            compute_efficiency = self._config.routing_compute_efficiency
            memory_efficiency = self._config.routing_memory_efficiency
            overlap_penalty = self._config.routing_overlap_penalty
        else:
            compute_efficiency = self._config.streaming_compute_efficiency
            memory_efficiency = self._config.streaming_memory_efficiency
            overlap_penalty = self._config.streaming_overlap_penalty
        return self.predict_roofline_kernel(
            operator_name=op_name,
            phase=self._phase(batch),
            local_shape=(
                f"read_elements={elements_read},write_elements={elements_written}"
            ),
            flops=flops,
            hbm_bytes=(elements_read + elements_written)
            * precision.bytes_per_element,
            precision=precision,
            compute_efficiency=compute_efficiency,
            memory_efficiency=memory_efficiency,
            overlap_penalty=overlap_penalty,
        )

    def _attention_context_costs(
        self, batch: Batch, *, prefill: bool
    ) -> tuple[float, float, str]:
        tp = int(self._replica_config.attn_tensor_parallel_size)
        q_heads = math.ceil(int(self._model_config.num_q_heads) / tp)
        kv_heads = math.ceil(int(self._model_config.get_runtime_num_kv_heads()) / tp)
        head_dim = int(self._model_config.get_runtime_head_size())
        total_flops = 0.0
        total_elements = 0.0
        shapes: list[str] = []
        for request, scheduled_tokens in zip(batch.requests, batch.num_tokens):
            is_prefill_request = not bool(request.is_prefill_complete)
            if is_prefill_request != prefill:
                continue
            query_tokens = int(scheduled_tokens)
            past_context = int(request.num_processed_tokens)
            average_visible_kv = past_context + (query_tokens + 1) / 2.0
            total_flops += (
                4.0
                * q_heads
                * head_dim
                * query_tokens
                * average_visible_kv
            )
            # IO-aware attention: do not materialize an SxS score matrix.
            q_and_output = 2.0 * query_tokens * q_heads * head_dim
            new_kv_input = 2.0 * query_tokens * kv_heads * head_dim
            cached_kv_reads = 2.0 * past_context * kv_heads * head_dim
            total_elements += q_and_output + new_kv_input + cached_kv_reads
            shapes.append(f"q={query_tokens}:past={past_context}")
        return total_flops, total_elements, ",".join(shapes)

    def predict_attention_layer_time(
        self,
        batch: Batch,
        layer_id: int,
        cluster_type: ClusterType,
    ) -> AttentionTime:
        del layer_id
        if cluster_type == ClusterType.DECODE_FFN:
            return AttentionTime()
        tokens = self._tokens(batch, cluster_type)
        hidden = int(self._model_config.embedding_dim)
        tp = int(self._replica_config.attn_tensor_parallel_size)
        q_heads = math.ceil(int(self._model_config.num_q_heads) / tp)
        kv_heads = math.ceil(int(self._model_config.get_runtime_num_kv_heads()) / tp)
        head_dim = int(self._model_config.get_runtime_head_size())
        local_qkv_dim = (q_heads + 2 * kv_heads) * head_dim

        pre_proj = self._gemm(
            "attn_pre_proj", batch, tokens, hidden, local_qkv_dim, cluster_type
        )
        post_proj = self._gemm(
            "attn_post_proj",
            batch,
            tokens,
            max(1, hidden // tp),
            hidden,
            cluster_type,
        )
        rope_elements = tokens * (q_heads + kv_heads) * head_dim
        rope = self._streaming(
            "attn_rope",
            batch,
            elements_read=rope_elements,
            elements_written=rope_elements,
            flops=6.0 * rope_elements,
            cluster_type=cluster_type,
        )
        kv_elements = tokens * 2 * kv_heads * head_dim
        kv_save = self._streaming(
            "attn_kv_cache_save",
            batch,
            elements_read=kv_elements,
            elements_written=kv_elements,
            flops=0.0,
            cluster_type=cluster_type,
        )
        norm_io_factor = 3 if self._model_config.uses_fused_add_norm else 2
        norm = self._streaming(
            "input_layernorm",
            batch,
            elements_read=tokens * hidden * (norm_io_factor - 1),
            elements_written=tokens * hidden,
            flops=5.0 * tokens * hidden,
            cluster_type=cluster_type,
        )

        prefill_flops, prefill_elements, prefill_shape = (
            self._attention_context_costs(batch, prefill=True)
        )
        decode_flops, decode_elements, decode_shape = (
            self._attention_context_costs(batch, prefill=False)
        )
        prefill_precision = self._precision("attn_prefill", cluster_type)
        decode_precision = self._precision("attn_decode", cluster_type)
        prefill_time = self.predict_roofline_kernel(
            operator_name="attn_prefill",
            phase="prefill",
            local_shape=prefill_shape,
            flops=prefill_flops,
            hbm_bytes=prefill_elements * prefill_precision.bytes_per_element,
            precision=prefill_precision,
            compute_efficiency=self._config.prefill_attention_compute_efficiency,
            memory_efficiency=self._config.prefill_attention_memory_efficiency,
            overlap_penalty=self._config.prefill_attention_overlap_penalty,
        )
        decode_time = self.predict_roofline_kernel(
            operator_name="attn_decode",
            phase="decode",
            local_shape=decode_shape,
            flops=decode_flops,
            hbm_bytes=decode_elements * decode_precision.bytes_per_element,
            precision=decode_precision,
            compute_efficiency=self._config.decode_attention_compute_efficiency,
            memory_efficiency=self._config.decode_attention_memory_efficiency,
            overlap_penalty=self._config.decode_attention_overlap_penalty,
        )
        return AttentionTime(
            attention_prefill_execution_time=prefill_time,
            attention_decode_execution_time=decode_time,
            attention_layer_pre_proj_execution_time=pre_proj,
            attention_layer_post_proj_execution_time=post_proj,
            attention_rope_execution_time=rope,
            attention_kv_cache_save_execution_time=kv_save,
            attn_norm_time=norm,
        )

    def predict_mlp_layer_time(
        self,
        batch: Batch,
        layer_id: int,
        cluster_type: ClusterType,
    ) -> MLPTime:
        del layer_id
        if self._model_config.is_moe:
            raise ValueError("Dense MLP prediction requested for an MoE model")
        if cluster_type == ClusterType.DECODE_ATTN:
            return MLPTime()
        tokens = self._tokens(batch, cluster_type)
        hidden = int(self._model_config.embedding_dim)
        intermediate = int(self._model_config.mlp_hidden_dim)
        tp = max(1, int(self._replica_config.attn_tensor_parallel_size))
        local_intermediate = math.ceil(intermediate / tp)
        gated = 2 if self._model_config.use_gated_mlp else 1
        up = self._gemm(
            "mlp_up_proj",
            batch,
            tokens,
            hidden,
            local_intermediate,
            cluster_type,
            weight_multiplier=gated,
        )
        act_elements = tokens * local_intermediate
        act = self._streaming(
            "mlp_act",
            batch,
            elements_read=act_elements * gated,
            elements_written=act_elements,
            flops=8.0 * act_elements,
            cluster_type=cluster_type,
        )
        down = self._gemm(
            "mlp_down_proj",
            batch,
            tokens,
            local_intermediate,
            hidden,
            cluster_type,
        )
        norm_factor = 3 if self._model_config.uses_fused_add_norm else 2
        norm = self._streaming(
            "post_attention_layernorm",
            batch,
            elements_read=tokens * hidden * (norm_factor - 1),
            elements_written=tokens * hidden,
            flops=5.0 * tokens * hidden,
            cluster_type=cluster_type,
        )
        return MLPTime(
            mlp_layer_up_proj_execution_time=up,
            mlp_layer_down_proj_execution_time=down,
            mlp_layer_act_execution_time=act,
            mlp_norm_time=norm,
        )

    @staticmethod
    def _discretize_expert_ratios(
        total_routed_tokens: int,
        allocation_ratios: Dict[int, float],
    ) -> Dict[int, int]:
        """Convert expert ratios to integer counts with strict conservation."""

        resolved_total = int(total_routed_tokens)
        if resolved_total < 0:
            raise ValueError("total_routed_tokens must be non-negative")
        if not allocation_ratios:
            if resolved_total == 0:
                return {}
            raise ValueError(
                "allocation_ratios must be non-empty when routed tokens exist"
            )
        if any(float(value) < 0.0 for value in allocation_ratios.values()):
            raise ValueError("allocation_ratios must be non-negative")

        ratio_sum = sum(float(value) for value in allocation_ratios.values())
        if ratio_sum <= 0.0:
            if resolved_total == 0:
                return {int(expert_id): 0 for expert_id in allocation_ratios}
            raise ValueError(
                "allocation_ratios must sum to a positive value"
            )

        base: Dict[int, int] = {}
        fractional: Dict[int, float] = {}
        normalized: Dict[int, float] = {}
        for expert_id in sorted(allocation_ratios):
            ratio = float(allocation_ratios[expert_id]) / ratio_sum
            exact = resolved_total * ratio
            base[int(expert_id)] = int(math.floor(exact))
            fractional[int(expert_id)] = exact - math.floor(exact)
            normalized[int(expert_id)] = ratio

        remainder = resolved_total - sum(base.values())
        ranked_experts = sorted(
            base,
            key=lambda expert_id: (
                -fractional[expert_id],
                -normalized[expert_id],
                expert_id,
            ),
        )
        for index in range(remainder):
            base[ranked_experts[index % len(ranked_experts)]] += 1

        if sum(base.values()) != resolved_total:
            raise ValueError(
                "MoE token conservation failed after ratio discretization"
            )
        return base

    def _get_global_per_expert_tokens(
        self,
        total_routed_tokens: int,
        layer_id: int,
    ) -> Dict[int, int]:
        """Apply the configured routing policy across all global experts."""

        resolved_total = int(total_routed_tokens)
        total_experts = int(self._replica_config.total_expert_num)
        if total_experts <= 0:
            raise ValueError(
                f"MoE routing requires num_experts > 0, got {total_experts}"
            )
        if resolved_total < 0:
            raise ValueError("total_routed_tokens must be non-negative")
        if resolved_total == 0:
            return {expert_id: 0 for expert_id in range(total_experts)}

        if self._moe_routing_mode == "uniform_random":
            rng = np.random.default_rng(
                self._moe_routing_seed + int(layer_id)
            )
            sampled_expert_ids = rng.integers(
                low=0,
                high=total_experts,
                size=resolved_total,
            )
            expert_counts = np.bincount(
                sampled_expert_ids,
                minlength=total_experts,
            )
            return {
                expert_id: int(expert_counts[expert_id])
                for expert_id in range(total_experts)
            }

        if self._moe_routing_mode == "uniform_legacy":
            ratios = {
                expert_id: 1.0 for expert_id in range(total_experts)
            }
            return self._discretize_expert_ratios(
                resolved_total,
                ratios,
            )

        distribution = self._moe_routing_distribution_type
        if distribution == "balanced":
            weights = np.ones(total_experts, dtype=float)
        elif distribution == "random":
            rng = np.random.default_rng(
                self._moe_routing_seed + int(layer_id)
            )
            weights = rng.uniform(0.1, 1.0, total_experts)
        elif distribution == "skewed":
            ranks = np.arange(1, total_experts + 1, dtype=float)
            weights = 1.0 / np.power(ranks, 0.35)
        elif distribution == "zipf":
            ranks = np.arange(1, total_experts + 1, dtype=float)
            weights = 1.0 / ranks
        else:  # pragma: no cover - guarded in __init__
            raise ValueError(
                f"Unsupported MoE routing distribution {distribution!r}"
            )

        return self._discretize_expert_ratios(
            resolved_total,
            {
                expert_id: float(weights[expert_id])
                for expert_id in range(total_experts)
            },
        )

    def _get_ep_lane_per_expert_tokens(
        self,
        batch: Batch,
        layer_id: int,
        cluster_type: ClusterType,
    ) -> Dict[int, Dict[int, int]]:
        """Partition global routed tokens by contiguous EP expert ownership."""

        ep_size = max(
            1,
            int(self._replica_config.moe_expert_parallel_size),
        )
        total_experts = int(self._replica_config.total_expert_num)
        if total_experts <= 0 or total_experts % ep_size != 0:
            raise ValueError(
                "MoE expert ownership requires num_experts to be positive and "
                "divisible by moe_expert_parallel_size; "
                f"got num_experts={total_experts}, ep_size={ep_size}"
            )

        topk = max(
            1,
            int(
                self._replica_config.router_topk
                or self._model_config.num_experts_per_tok
            ),
        )
        total_routed_tokens = self._tokens(batch, cluster_type) * topk
        global_allocation = self._get_global_per_expert_tokens(
            total_routed_tokens,
            layer_id,
        )
        experts_per_lane = total_experts // ep_size
        lane_allocations: Dict[int, Dict[int, int]] = {
            ep_id: {} for ep_id in range(ep_size)
        }
        for global_expert_id, token_count in global_allocation.items():
            ep_id = global_expert_id // experts_per_lane
            local_expert_id = global_expert_id % experts_per_lane
            lane_allocations[ep_id][local_expert_id] = int(token_count)

        if sum(
            token_count
            for allocation in lane_allocations.values()
            for token_count in allocation.values()
        ) != total_routed_tokens:
            raise ValueError(
                "MoE token conservation failed while partitioning EP lanes"
            )
        return lane_allocations

    def _uniform_local_expert_tokens(self, batch: Batch) -> Dict[int, int]:
        """Legacy uniform allocation for callers that explicitly request it."""

        topk = max(
            1,
            int(
                self._replica_config.router_topk
                or self._model_config.num_experts_per_tok
            ),
        )
        ep = max(1, int(self._replica_config.moe_expert_parallel_size))
        local_experts = max(
            1, int(self._model_config.num_experts) // ep
        )
        local_routed_tokens = math.ceil(
            self._tokens(batch, self._cluster_type) * topk / ep
        )
        base, remainder = divmod(local_routed_tokens, local_experts)
        return {
            expert_id: base + (1 if expert_id < remainder else 0)
            for expert_id in range(local_experts)
        }

    def _build_uniform_per_expert_tokens(
        self, total_routed_tokens: int
    ) -> Dict[int, int]:
        """Compatibility helper used by the shared-domain decode sync path."""

        resolved_tokens = int(total_routed_tokens)
        if resolved_tokens < 0:
            raise ValueError("total_routed_tokens must be non-negative")
        ep = max(1, int(self._replica_config.moe_expert_parallel_size))
        local_experts = max(1, int(self._model_config.num_experts) // ep)
        base, remainder = divmod(resolved_tokens, local_experts)
        return {
            expert_id: base + (1 if expert_id < remainder else 0)
            for expert_id in range(local_experts)
        }

    def _get_moe_tokens_input(
        self, batch: Batch, layer_id: int = 0
    ) -> Dict[int, int]:
        lane_allocations = self._get_ep_lane_per_expert_tokens(
            batch,
            layer_id,
            self._cluster_type,
        )
        return lane_allocations[0]

    def _predict_grouped_moe_projection_time(
        self,
        *,
        batch: Batch,
        allocation: Dict[int, int],
        k: int,
        n: int,
        cluster_type: ClusterType,
        projection: str,
        weight_multiplier: int = 1,
    ) -> float:
        """Predict one grouped projection over all active local experts."""

        active_allocation = {
            int(expert_id): int(token_count)
            for expert_id, token_count in allocation.items()
            if int(token_count) > 0
        }
        if not active_allocation:
            return 0.0

        precision = self._precision("moe_grouped_gemm", cluster_type)
        element_bytes = precision.bytes_per_element
        flops = 0.0
        hbm_bytes = 0.0
        for token_count in active_allocation.values():
            flops += (
                2.0
                * token_count
                * k
                * n
                * weight_multiplier
            )
            hbm_bytes += element_bytes * (
                token_count * k
                + weight_multiplier * k * n
                + weight_multiplier * token_count * n
            )

        token_shape = ",".join(
            f"e{expert_id}:{active_allocation[expert_id]}"
            for expert_id in sorted(active_allocation)
        )
        return self.predict_roofline_kernel(
            operator_name="moe_grouped_gemm",
            phase=self._phase(batch),
            local_shape=(
                f"projection={projection},groups={len(active_allocation)},"
                f"K={k},N={n},multiplicity={weight_multiplier},"
                f"tokens=[{token_shape}]"
            ),
            flops=flops,
            hbm_bytes=hbm_bytes,
            precision=precision,
            compute_efficiency=self._config.moe_compute_efficiency,
            memory_efficiency=self._config.moe_memory_efficiency,
            overlap_penalty=self._config.moe_overlap_penalty,
        )

    def predict_ep_lane_moe_times_ms(
        self,
        batch: Batch,
        layer_id: int,
        cluster_type: ClusterType,
    ) -> Dict[int, float]:
        """Predict local MoE compute time for every EP lane."""

        lane_allocations = self._get_ep_lane_per_expert_tokens(
            batch,
            layer_id,
            cluster_type,
        )
        return {
            ep_id: self.predict_moe_layer_time(
                batch,
                layer_id,
                cluster_type,
                per_expert_tokens=local_allocation,
            ).total_time()
            for ep_id, local_allocation in lane_allocations.items()
        }

    def predict_monolithic_decode_shared_domain_lane_moe_times_ms(
        self,
        batch: Batch,
        layer_id: int,
    ) -> Dict[int, float]:
        """Compatibility entry point for monolithic EP synchronization."""

        return self.predict_ep_lane_moe_times_ms(
            batch,
            layer_id,
            ClusterType.MONOLITHIC,
        )

    def predict_moe_layer_time(
        self,
        batch_or_group: Batch,
        layer_id: int,
        cluster_type: ClusterType,
        per_expert_tokens: Optional[Dict[int, int]] = None,
    ) -> MoETime:
        if not self._model_config.is_moe:
            raise ValueError("MoE prediction requested for a dense model")
        if cluster_type == ClusterType.DECODE_ATTN:
            return MoETime()
        batch = batch_or_group
        tokens = self._tokens(batch, cluster_type)
        hidden = int(self._model_config.embedding_dim)
        experts = int(self._model_config.num_experts)
        intermediate = int(self._model_config.mlp_hidden_dim)
        tp = max(1, int(self._replica_config.moe_tensor_parallel_size))
        local_intermediate = math.ceil(intermediate / tp)
        gated = 2 if self._model_config.use_gated_mlp else 1

        gating = self._gemm(
            "moe_gating_linear", batch, tokens, hidden, experts, cluster_type
        )
        routing_elements = tokens * experts
        routing = self._streaming(
            "moe_gating_routing_topk",
            batch,
            elements_read=routing_elements,
            elements_written=tokens
            * max(1, int(self._replica_config.router_topk)),
            flops=4.0 * routing_elements,
            cluster_type=cluster_type,
            operator_class="routing",
        )
        allocation = (
            dict(per_expert_tokens)
            if per_expert_tokens is not None
            else dict(getattr(batch, "per_expert_tokens", {}) or {})
        )
        if not allocation:
            allocation = self._get_moe_tokens_input(batch, layer_id)
        if any(int(value) < 0 for value in allocation.values()):
            raise ValueError("per_expert_tokens values must be non-negative")

        grouped_gemm = self._predict_grouped_moe_projection_time(
            batch=batch,
            allocation=allocation,
            k=hidden,
            n=local_intermediate,
            cluster_type=cluster_type,
            projection="up",
            weight_multiplier=gated,
        )
        grouped_gemm += self._predict_grouped_moe_projection_time(
            batch=batch,
            allocation=allocation,
            k=local_intermediate,
            n=hidden,
            cluster_type=cluster_type,
            projection="down",
        )

        routed_tokens = sum(int(value) for value in allocation.values())
        shuffling = self._streaming(
            "moe_shuffling",
            batch,
            elements_read=routed_tokens * hidden,
            elements_written=routed_tokens * hidden,
            flops=0.0,
            cluster_type=cluster_type,
        )
        norm_factor = 3 if self._model_config.uses_fused_add_norm else 2
        norm = self._streaming(
            "post_attention_layernorm",
            batch,
            elements_read=tokens * hidden * (norm_factor - 1),
            elements_written=tokens * hidden,
            flops=5.0 * tokens * hidden,
            cluster_type=cluster_type,
        )
        return MoETime(
            moe_grouped_gemm_time=grouped_gemm,
            moe_gating_linear_time=gating,
            moe_gating_routing_topk_time=routing,
            moe_shuffling_time=shuffling,
            mlp_norm_time=norm,
        )

    def _require_cc_backend(self):
        if self._cc_backend is None:
            raise RuntimeError(
                "analytical_roofline requires a CC backend for multi-device "
                "communication prediction"
            )
        return self._cc_backend

    def predict_allreduce_time(
        self,
        data_size_bytes: int,
        num_devices: int,
        cluster_type: ClusterType,
        comm_domain: Optional[str] = None,
    ) -> float:
        if num_devices <= 1 or data_size_bytes <= 0:
            return 0.0
        return self._require_cc_backend().predict_allreduce(
            data_size_bytes=data_size_bytes,
            num_devices=num_devices,
            cluster_type=cluster_type,
            comm_domain=comm_domain,
        )

    def predict_allgather_time(
        self,
        data_size_bytes: int,
        num_devices: int,
        cluster_type: ClusterType,
        comm_domain: Optional[str] = None,
    ) -> float:
        if num_devices <= 1 or data_size_bytes <= 0:
            return 0.0
        return self._require_cc_backend().predict_allgather(
            data_size_bytes=data_size_bytes,
            num_devices=num_devices,
            cluster_type=cluster_type,
            comm_domain=comm_domain,
        )

    def predict_alltoall_time(
        self,
        data_size_bytes: int,
        num_devices: int,
        cluster_type: ClusterType,
        comm_domain: Optional[str] = None,
    ) -> float:
        if num_devices <= 1 or data_size_bytes <= 0:
            return 0.0
        return self._require_cc_backend().predict_all_to_all(
            data_size_bytes=data_size_bytes,
            num_devices=num_devices,
            cluster_type=cluster_type,
            comm_domain=comm_domain,
        )

    def predict_p2p_time(
        self,
        data_size_bytes: int,
        cluster_type: ClusterType,
        comm_domain: Optional[str] = None,
    ) -> float:
        if data_size_bytes <= 0:
            return 0.0
        return self._require_cc_backend().predict_send_recv(
            data_size_bytes=data_size_bytes,
            cluster_type=cluster_type,
            comm_domain=comm_domain,
        )

    def _communication_payload_bytes(
        self, batch: Batch, cluster_type: ClusterType
    ) -> int:
        elements = self._tokens(batch, cluster_type) * int(
            self._model_config.embedding_dim
        )
        return int(
            math.ceil(
                elements
                * self._quantization_manager.get_bytes_per_element(
                    "allreduce", cluster_type
                )
            )
        )

    def _get_expert_parallel_communication_time(self, batch: Batch) -> float:
        ep = int(self._replica_config.moe_expert_parallel_size)
        if ep <= 1:
            return 0.0
        topk = max(1, int(self._replica_config.router_topk))
        payload = int(
            math.ceil(
                self._tokens(batch, self._cluster_type)
                * topk
                * int(self._model_config.embedding_dim)
                * self._quantization_manager.get_bytes_per_element(
                    "expert_parallel_communication", self._cluster_type
                )
            )
        )
        # Dispatch and combine are two separate all-to-all collectives.
        return 2.0 * self.predict_alltoall_time(
            payload, ep, self._cluster_type, comm_domain="EP"
        )

    def _residual_add_time(
        self, batch: Batch, cluster_type: ClusterType
    ) -> float:
        if self._model_config.uses_fused_add_norm:
            return 0.0
        elements = self._tokens(batch, cluster_type) * int(
            self._model_config.embedding_dim
        )
        return self._streaming(
            "add",
            batch,
            elements_read=2 * elements,
            elements_written=elements,
            flops=elements,
            cluster_type=cluster_type,
        )

    def predict_stage_execution_time(
        self,
        batch: Batch,
        stage_id: int,
        cluster_type: ClusterType,
        num_layers: int = 1,
        layer_id: int = 0,
    ) -> ExecutionTime:
        if self._enable_dummy_mode:
            return self._get_dummy_execution_time(batch, stage_id)
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got={num_layers}")

        attention = self.predict_attention_layer_time(
            batch, layer_id, cluster_type
        )
        mlp = MLPTime()
        moe = MoETime()
        if self._model_config.is_moe:
            moe = self.predict_moe_layer_time(
                batch, layer_id, cluster_type
            )
        else:
            mlp = self.predict_mlp_layer_time(
                batch, layer_id, cluster_type
            )

        payload = self._communication_payload_bytes(batch, cluster_type)
        attn_tp = self.predict_allreduce_time(
            payload,
            int(self._replica_config.attn_tensor_parallel_size),
            cluster_type,
            comm_domain="ATTN_TP",
        )
        ffn_tp_size = (
            int(self._replica_config.moe_tensor_parallel_size)
            if self._model_config.is_moe
            else int(self._replica_config.attn_tensor_parallel_size)
        )
        ffn_tp = self.predict_allreduce_time(
            payload,
            ffn_tp_size,
            cluster_type,
            comm_domain=("MOE_TP" if self._model_config.is_moe else "ATTN_TP"),
        )
        pp = 0.0
        if stage_id < int(self._replica_config.num_pipeline_stages) - 1:
            pp = self.predict_p2p_time(payload, cluster_type, comm_domain="PP")
        ep = (
            self._get_expert_parallel_communication_time(batch)
            if self._model_config.is_moe
            else 0.0
        )
        residual = self._residual_add_time(batch, cluster_type)

        return ExecutionTime(
            num_layers_per_pipeline_stage=num_layers,
            attention_rope_execution_time=attention.attention_rope_execution_time,
            attention_kv_cache_save_execution_time=(
                attention.attention_kv_cache_save_execution_time
            ),
            attention_decode_execution_time=(
                attention.attention_decode_execution_time
            ),
            attention_prefill_execution_time=(
                attention.attention_prefill_execution_time
            ),
            attention_layer_pre_proj_execution_time=(
                attention.attention_layer_pre_proj_execution_time
            ),
            attention_layer_post_proj_execution_time=(
                attention.attention_layer_post_proj_execution_time
            ),
            attn_norm_time=attention.attn_norm_time,
            mlp_norm_time=(
                moe.mlp_norm_time if self._model_config.is_moe else mlp.mlp_norm_time
            ),
            add_time=2.0 * residual,
            add_attn_residual_time=residual,
            add_ffn_residual_time=residual,
            tensor_parallel_communication_time=attn_tp + ffn_tp,
            attn_tensor_parallel_allreduce_time=attn_tp,
            moe_tensor_parallel_allreduce_time=ffn_tp,
            pipeline_parallel_communication_time=pp,
            expert_parallel_communication_time=ep,
            moe_gating_time=moe.moe_gating_time,
            moe_gating_linear_time=moe.moe_gating_linear_time,
            moe_gating_routing_topk_time=moe.moe_gating_routing_topk_time,
            moe_shuffling_time=moe.moe_shuffling_time,
            moe_grouped_gemm_time=moe.moe_grouped_gemm_time,
            mlp_layer_up_proj_execution_time=(
                mlp.mlp_layer_up_proj_execution_time
            ),
            mlp_layer_down_proj_execution_time=(
                mlp.mlp_layer_down_proj_execution_time
            ),
            mlp_layer_act_execution_time=mlp.mlp_layer_act_execution_time,
            schedule_time=0.0,
            sampler_e2e_time=0.0,
            prepare_inputs_e2e_time=0.0,
            process_model_outputs_time=0.0,
            ray_comm_time=0.0,
            is_moe=bool(self._model_config.is_moe),
        )

    # Legacy granular methods retained for scheduler compatibility.
    def _get_attention_layer_pre_proj_execution_time(self, batch: Batch) -> float:
        return self.predict_attention_layer_time(
            batch, 0, self._cluster_type
        ).attention_layer_pre_proj_execution_time

    def _get_attention_layer_post_proj_execution_time(self, batch: Batch) -> float:
        return self.predict_attention_layer_time(
            batch, 0, self._cluster_type
        ).attention_layer_post_proj_execution_time

    def _get_attention_rope_execution_time(self, batch: Batch) -> float:
        return self.predict_attention_layer_time(
            batch, 0, self._cluster_type
        ).attention_rope_execution_time

    def _get_attention_kv_cache_save_execution_time(self, batch: Batch) -> float:
        return self.predict_attention_layer_time(
            batch, 0, self._cluster_type
        ).attention_kv_cache_save_execution_time

    def _get_attention_decode_execution_time(self, batch: Batch) -> float:
        return self.predict_attention_layer_time(
            batch, 0, self._cluster_type
        ).attention_decode_execution_time

    def _get_attention_prefill_execution_time(self, batch: Batch) -> float:
        return self.predict_attention_layer_time(
            batch, 0, self._cluster_type
        ).attention_prefill_execution_time

    def _get_mlp_layer_up_proj_execution_time(self, batch: Batch) -> float:
        return self.predict_mlp_layer_time(
            batch, 0, self._cluster_type
        ).mlp_layer_up_proj_execution_time

    def _get_mlp_layer_down_proj_execution_time(self, batch: Batch) -> float:
        return self.predict_mlp_layer_time(
            batch, 0, self._cluster_type
        ).mlp_layer_down_proj_execution_time

    def _get_mlp_layer_act_execution_time(self, batch: Batch) -> float:
        return self.predict_mlp_layer_time(
            batch, 0, self._cluster_type
        ).mlp_layer_act_execution_time

    def _get_tensor_parallel_communication_time(self, batch: Batch) -> float:
        return self.predict_allreduce_time(
            self._communication_payload_bytes(batch, self._cluster_type),
            int(self._replica_config.attn_tensor_parallel_size),
            self._cluster_type,
            comm_domain="ATTN_TP",
        )

    def _get_pipeline_parallel_communication_time(self, batch: Batch) -> float:
        return self.predict_p2p_time(
            self._communication_payload_bytes(batch, self._cluster_type),
            self._cluster_type,
            comm_domain="PP",
        )

    def _get_schedule_time(self, batch: Batch) -> float:
        del batch
        return 0.0

    def _get_sampler_e2e_time(self, batch: Batch) -> float:
        del batch
        return 0.0

    def _get_prepare_inputs_e2e_time(self, batch: Batch) -> float:
        del batch
        return 0.0

    def _get_process_model_outputs_time(self, batch: Batch) -> float:
        del batch
        return 0.0

    def _get_ray_comm_time(self, batch: Batch) -> float:
        del batch
        return 0.0

    def _get_mlp_norm_layer_act_execution_time(self, batch: Batch) -> float:
        if self._model_config.is_moe:
            return self.predict_moe_layer_time(
                batch, 0, self._cluster_type
            ).mlp_norm_time
        return self.predict_mlp_layer_time(
            batch, 0, self._cluster_type
        ).mlp_norm_time

    def _get_attn_norm_layer_act_execution_time(self, batch: Batch) -> float:
        return self.predict_attention_layer_time(
            batch, 0, self._cluster_type
        ).attn_norm_time

    def _get_add_layer_act_execution_time(self, batch: Batch) -> float:
        return self._residual_add_time(batch, self._cluster_type)

    def to_dict(self) -> dict:
        return {
            "model_provider": str(self._config.get_type()),
            "device": self._replica_config.device,
            "cluster_type": str(self._cluster_type),
            "hbm_bandwidth_tbps": float(
                self._replica_config.device_config.hbm_bandwidth_tbps
            ),
            "fp32_tflops": float(self._replica_config.device_config.fp32_tflops),
            "fp16_tflops": float(self._replica_config.device_config.fp16_tflops),
            "fp8_tflops": float(self._replica_config.device_config.fp8_tflops),
            "fp4_tflops": float(self._replica_config.device_config.fp4_tflops),
            "diagnostic_count": len(self._diagnostics),
        }
