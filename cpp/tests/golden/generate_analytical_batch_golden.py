"""Generate analytical batch fixtures from production Python models."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

logging.disable(logging.CRITICAL)

from frontier.cc_backend.backends.analytical_cc_backend import (
    AnalyticalCCBackend,
)
from frontier.cc_backend.cc_backend_config import AnalyticalCCBackendConfig
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


class _Batch:
    def __init__(self, slices: list[dict[str, int | str]]) -> None:
        self.requests = [
            SimpleNamespace(
                is_prefill_complete=item["phase"] == "decode",
                num_processed_tokens=item["past_context"],
            )
            for item in slices
        ]
        self.num_tokens = [
            int(item["scheduled_tokens"]) for item in slices
        ]
        self.total_num_tokens = sum(self.num_tokens)
        self.num_prefill_tokens = sum(
            tokens
            for item, tokens in zip(slices, self.num_tokens)
            if item["phase"] == "prefill"
        )
        self.num_decode_tokens = (
            self.total_num_tokens - self.num_prefill_tokens
        )
        self.is_moe = False

    def get_effective_total_tokens_for_compute(
        self, _cluster_type=None
    ) -> int:
        return self.total_num_tokens


def _predictor() -> AnalyticalRooflineExecutionTimePredictor:
    replica = ReplicaConfig(
        model_name="meta-llama/Llama-2-7b-hf",
        device="rubin",
        network_device="vera_rubin_nvl72_domain",
        attn_tensor_parallel_size=8,
        num_pipeline_stages=1,
    )
    communication = AnalyticalCCBackend(
        AnalyticalCCBackendConfig(
            network_bandwidth_gbps=400.0,
            network_latency_us=1.0,
            intra_node_bandwidth_gbps=14_400.0,
        ),
        ClusterType.MONOLITHIC,
        "rubin",
        "vera_rubin_nvl72_domain",
        8,
    )
    return AnalyticalRooflineExecutionTimePredictor(
        AnalyticalRooflineExecutionTimePredictorConfig(),
        replica,
        VllmV1SchedulerConfig(),
        MetricsConfig(write_metrics=False, run_id="step2-analytical-batch"),
        cluster_type=ClusterType.MONOLITHIC,
        cc_backend=communication,
    )


def _case(
    predictor: AnalyticalRooflineExecutionTimePredictor,
    *,
    name: str,
    slices: list[dict[str, int | str]],
) -> dict:
    batch = _Batch(slices)
    attention = predictor.predict_attention_layer_time(
        batch, layer_id=0, cluster_type=ClusterType.MONOLITHIC
    )
    mlp = predictor.predict_mlp_layer_time(
        batch, layer_id=0, cluster_type=ClusterType.MONOLITHIC
    )
    payload = predictor._communication_payload_bytes(
        batch, ClusterType.MONOLITHIC
    )
    allreduce_ms = predictor.predict_allreduce_time(
        payload,
        8,
        ClusterType.MONOLITHIC,
        comm_domain="ATTN_TP",
    )
    component_values = [
        attention.attention_layer_pre_proj_execution_time,
        attention.attention_layer_post_proj_execution_time,
        attention.attention_rope_execution_time,
        attention.attention_kv_cache_save_execution_time,
        attention.attn_norm_time,
        attention.attention_prefill_execution_time,
        attention.attention_decode_execution_time,
        mlp.mlp_layer_up_proj_execution_time,
        mlp.mlp_layer_act_execution_time,
        mlp.mlp_layer_down_proj_execution_time,
        mlp.mlp_norm_time,
    ]
    dense_layer_compute_ms = sum(component_values)
    stage = predictor.predict_stage_execution_time(
        batch,
        stage_id=0,
        cluster_type=ClusterType.MONOLITHIC,
        num_layers=32,
    )
    layer_total_ms = dense_layer_compute_ms + 2.0 * allreduce_ms
    return {
        "name": name,
        "input": {"slices": slices},
        "expected": {
            "total_tokens": batch.total_num_tokens,
            "prefill_request_count": sum(
                item["phase"] == "prefill" for item in slices
            ),
            "decode_request_count": sum(
                item["phase"] == "decode" for item in slices
            ),
            "dense_layer_compute_ms": dense_layer_compute_ms,
            "tp_allreduce_ms": allreduce_ms,
            "dense_layer_total_ms": layer_total_ms,
            "num_layers": 32,
            "batch_duration_ms": stage.model_time_ms,
        },
    }


def main() -> None:
    predictor = _predictor()
    cases = [
        _case(
            predictor,
            name="single_prefill",
            slices=[
                {
                    "phase": "prefill",
                    "scheduled_tokens": 128,
                    "past_context": 0,
                }
            ],
        ),
        _case(
            predictor,
            name="single_decode_short_context",
            slices=[
                {
                    "phase": "decode",
                    "scheduled_tokens": 1,
                    "past_context": 128,
                }
            ],
        ),
        _case(
            predictor,
            name="single_decode_long_context",
            slices=[
                {
                    "phase": "decode",
                    "scheduled_tokens": 1,
                    "past_context": 4096,
                }
            ],
        ),
        _case(
            predictor,
            name="mixed_prefill_decode",
            slices=[
                {
                    "phase": "prefill",
                    "scheduled_tokens": 64,
                    "past_context": 32,
                },
                {
                    "phase": "decode",
                    "scheduled_tokens": 1,
                    "past_context": 1024,
                },
            ],
        ),
    ]
    print(
        json.dumps(
            {
                "schema_version": 1,
                "oracle": (
                    "production AnalyticalRooflineExecutionTimePredictor "
                    "with AnalyticalCCBackend"
                ),
                "cases": cases,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
