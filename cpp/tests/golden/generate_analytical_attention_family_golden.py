"""Generate Python/C++ analytical MLA and MFA attention parity data."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

logging.disable(logging.CRITICAL)

from frontier.config import (
    AnalyticalRooflineExecutionTimePredictorConfig,
    MetricsConfig,
    ReplicaConfig,
    VllmV1SchedulerConfig,
)
from frontier.execution_time_predictor.analytical_roofline_execution_time_predictor import (
    AnalyticalRooflineExecutionTimePredictor,
)
from frontier.types import ClusterType


class _NoCommunicationBackend:
    def predict_allreduce(self, **_kwargs):
        return 0.0

    def predict_allgather(self, **_kwargs):
        return 0.0

    def predict_all_to_all(self, **_kwargs):
        return 0.0

    def predict_send_recv(self, **_kwargs):
        return 0.0


class _Batch:
    def __init__(self, *, tokens: int, past: int, prefill: bool) -> None:
        self.requests = [
            SimpleNamespace(
                is_prefill_complete=not prefill,
                num_processed_tokens=past,
            )
        ]
        self.num_tokens = [tokens]
        self.total_num_tokens = tokens
        self.num_prefill_tokens = tokens if prefill else 0
        self.num_decode_tokens = 0 if prefill else tokens

    def get_effective_total_tokens_for_compute(self, _cluster_type=None):
        return self.total_num_tokens


def _predictor() -> AnalyticalRooflineExecutionTimePredictor:
    return AnalyticalRooflineExecutionTimePredictor(
        AnalyticalRooflineExecutionTimePredictorConfig(),
        ReplicaConfig(
            device="rubin",
            network_device="vera_rubin_nvl72_domain",
            attn_tensor_parallel_size=4,
        ),
        VllmV1SchedulerConfig(),
        MetricsConfig(),
        cluster_type=ClusterType.MONOLITHIC,
        cc_backend=_NoCommunicationBackend(),
    )


def _times(predictor, *, tokens: int, past: int, prefill: bool) -> dict:
    value = predictor.predict_attention_layer_time(
        _Batch(tokens=tokens, past=past, prefill=prefill),
        layer_id=0,
        cluster_type=ClusterType.MONOLITHIC,
    )
    return {
        "attention_pre_projection_ms": (
            value.attention_layer_pre_proj_execution_time
        ),
        "attention_post_projection_ms": (
            value.attention_layer_post_proj_execution_time
        ),
        "rope_ms": value.attention_rope_execution_time,
        "kv_cache_save_ms": value.attention_kv_cache_save_execution_time,
        "attention_norm_ms": value.attn_norm_time,
        "attention_inter_norm_ms": value.attn_inter_norm_time,
        "attention_wq_projection_ms": value.attn_wq_proj_time,
        "prefill_attention_ms": value.attention_prefill_execution_time,
        "decode_attention_ms": value.attention_decode_execution_time,
    }


def main() -> None:
    predictor = _predictor()
    predictor._model_config = SimpleNamespace(
        embedding_dim=7168,
        num_q_heads=64,
        num_kv_heads=64,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        qk_head_dim=192,
        v_head_dim=128,
        use_mfa=False,
        uses_mla=lambda: True,
        get_runtime_num_kv_heads=lambda: 1,
        get_runtime_head_size=lambda: 576,
        uses_fused_add_norm=True,
    )
    mla = _times(predictor, tokens=1, past=8192, prefill=False)

    predictor._model_config = SimpleNamespace(
        embedding_dim=7168,
        num_q_heads=64,
        num_kv_heads=1,
        share_q_dim=2048,
        use_mfa=True,
        uses_mla=lambda: False,
        get_runtime_num_kv_heads=lambda: 1,
        get_runtime_head_size=lambda: 256,
        uses_fused_add_norm=True,
    )
    mfa = _times(predictor, tokens=32, past=256, prefill=True)

    print(
        json.dumps(
            {
                "schema_version": 1,
                "precision": "fp16",
                "tensor_parallel_size": 4,
                "mla_decode_1_past_8192": mla,
                "mfa_prefill_32_past_256": mfa,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
