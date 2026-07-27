from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TRACE_FILE = REPO_ROOT / "examples" / "fixtures" / "session_prefix_multi_turn_trace.csv"
BLOCK_HASH_TRACE_FILE = (
    REPO_ROOT / "examples" / "fixtures" / "prefix_cache_shared_session_trace.csv"
)


def _common_args(
    output_root: Path,
    run_id: str,
    *,
    trace_file: Path = TRACE_FILE,
    key_mode: str = "session",
    cluster_scheduler_type: str = "sticky_round_robin",
    prefill_scale_factor: float = 1.0,
    decode_scale_factor: float = 1.0,
    time_scale_factor: float = 1.0,
) -> list[str]:
    return [
        "-m",
        "frontier.main",
        "--simulation_mode",
        "online",
        "--cc_backend_config_type",
        "analytical",
        "--cluster_scheduler_config_type",
        cluster_scheduler_type,
        "--replica_config_model_name",
        "Phi-tiny-MoE-instruct",
        "--replica_config_moe_routing_mode",
        "simulation",
        "--replica_config_moe_routing_seed",
        "42",
        "--replica_scheduler_config_type",
        "vllm_v1",
        "--decode_cuda_graph_mode",
        "full_decode_only",
        "--vllm_v1_scheduler_config_max_tokens_in_batch",
        "128",
        "--vllm_v1_scheduler_config_long_prefill_token_threshold",
        "16",
        "--vllm_v1_scheduler_config_block_size",
        "16",
        "--vllm_v1_scheduler_config_num_blocks",
        "128",
        "--vllm_v1_scheduler_config_enable_prefix_caching",
        "--vllm_v1_scheduler_config_prefix_caching_key_mode",
        key_mode,
        "--vllm_v1_scheduler_config_enable_chunked_prefill",
        "--request_generator_config_type",
        "trace_replay",
        "--trace_request_generator_config_trace_file",
        str(trace_file),
        "--trace_request_generator_config_prefill_scale_factor",
        str(prefill_scale_factor),
        "--trace_request_generator_config_decode_scale_factor",
        str(decode_scale_factor),
        "--trace_request_generator_config_time_scale_factor",
        str(time_scale_factor),
        "--trace_request_generator_config_max_tokens",
        "128",
        "--metrics_config_output_dir",
        str(output_root),
        "--metrics_config_run_id",
        run_id,
        "--metrics_config_write_metrics",
        "--metrics_config_store_request_metrics",
        "--metrics_config_store_batch_metrics",
        "--metrics_config_store_token_completion_metrics",
        "--metrics_config_store_utilization_metrics",
        "--no-metrics_config_store_plots",
        "--no-metrics_config_enable_chrome_trace",
        "--no-metrics_config_write_json_trace",
        "--random_forrest_execution_time_predictor_config_enable_dummy_mode",
        "--random_forrest_execution_time_predictor_config_dummy_execution_time_ms",
        "1.0",
    ]


def _colocation_args(*, dp_size: int = 1) -> list[str]:
    return [
        "--sys_arch",
        "co-location",
        "--cluster_config_num_replicas",
        "1",
        "--replica_config_attn_tensor_parallel_size",
        "1",
        "--replica_config_attn_data_parallel_size",
        str(dp_size),
        "--replica_config_moe_tensor_parallel_size",
        "1",
        "--replica_config_moe_expert_parallel_size",
        str(dp_size),
        "--replica_config_num_pipeline_stages",
        "1",
        "--replica_config_total_expert_num",
        "8",
        "--replica_config_router_topk",
        "2",
    ]


def _pdd_args() -> list[str]:
    args = [
        "--sys_arch",
        "pd-disaggregation",
        "--no-enable_parallel_clusters",
        "--cluster_config_prefill_cluster_num_replicas",
        "1",
        "--cluster_config_decode_cluster_num_replicas",
        "1",
        "--analytical_kv_cache_transfer_config_network_bandwidth_gbps",
        "200",
        "--analytical_kv_cache_transfer_config_network_latency_ms",
        "0.1",
    ]
    for cluster_name in ("prefill", "decode"):
        prefix = f"--cluster_config_{cluster_name}_replica_config_"
        args.extend(
            [
                f"{prefix}num_pipeline_stages",
                "1",
                f"{prefix}attn_tensor_parallel_size",
                "1",
                f"{prefix}attn_data_parallel_size",
                "1",
                f"{prefix}moe_tensor_parallel_size",
                "1",
                f"{prefix}moe_expert_parallel_size",
                "1",
                f"{prefix}total_expert_num",
                "8",
                f"{prefix}router_topk",
                "2",
            ]
        )
    return args


def _run_simulation(
    *,
    output_root: Path,
    run_id: str,
    architecture_args: list[str],
    trace_file: Path = TRACE_FILE,
    key_mode: str = "session",
    prefill_scale_factor: float = 1.0,
    decode_scale_factor: float = 1.0,
    time_scale_factor: float = 1.0,
) -> tuple[pd.DataFrame, dict]:
    command = [
        sys.executable,
        *_common_args(
            output_root,
            run_id,
            trace_file=trace_file,
            key_mode=key_mode,
            prefill_scale_factor=prefill_scale_factor,
            decode_scale_factor=decode_scale_factor,
            time_scale_factor=time_scale_factor,
        ),
        *architecture_args,
    ]
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(REPO_ROOT),
            "PYTHONDONTWRITEBYTECODE": "1",
            "WANDB_DISABLED": "true",
            "VIDUR_DISABLE_WANDB": "1",
        }
    )

    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout[-5000:]

    run_dir = output_root / "phi_tiny_moe_instruct" / "online_serving" / run_id
    request_metrics = pd.read_csv(run_dir / "request_metrics.csv")
    system_metrics = json.loads(
        (run_dir / "system_metrics.json").read_text(encoding="utf-8")
    )
    return request_metrics, system_metrics


@pytest.mark.parametrize(
    ("architecture", "architecture_args", "expected_cached_tokens"),
    [
        ("colocation", _colocation_args(), [0, 0, 48, 48]),
        ("pdd", _pdd_args(), [0, 0, 32, 32]),
    ],
)
def test_session_prefix_cache_runtime_metrics(
    tmp_path: Path,
    architecture: str,
    architecture_args: list[str],
    expected_cached_tokens: list[int],
) -> None:
    output_root = tmp_path / "metrics"
    run_id = f"session_prefix_{architecture}"
    request_metrics, system_metrics = _run_simulation(
        output_root=output_root,
        run_id=run_id,
        architecture_args=architecture_args,
    )

    assert request_metrics["request_session_id"].tolist() == [7, 8, 7, 8]
    assert request_metrics["request_cached_prefill_tokens"].tolist() == (
        expected_cached_tokens
    )
    assert request_metrics["request_prefix_cache_query_blocks"].tolist() == [
        2,
        2,
        3,
        3,
    ]
    assert request_metrics["request_prefix_cache_hit_blocks"].tolist() == [
        value // 16 for value in expected_cached_tokens
    ]
    assert system_metrics["prefix_cache_statistics"]["key_mode"] == "session"


def test_pdd_session_prefix_cache_rejects_sticky_lor_before_runtime(
    tmp_path: Path,
) -> None:
    command = [
        sys.executable,
        *_common_args(
            tmp_path / "metrics",
            "pdd_sticky_lor_rejected",
            cluster_scheduler_type="sticky_lor",
        ),
        *_pdd_args(),
    ]
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(REPO_ROOT),
            "PYTHONDONTWRITEBYTECODE": "1",
            "WANDB_DISABLED": "true",
            "VIDUR_DISABLE_WANDB": "1",
        }
    )

    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=120,
    )

    assert completed.returncode != 0
    assert "only supported for MONOLITHIC clusters" in completed.stdout
    assert "must use sticky_round_robin" in completed.stdout


def test_invalid_session_arrival_fails_before_simulation_events(
    tmp_path: Path,
) -> None:
    trace_file = tmp_path / "invalid_nan_arrival.csv"
    trace_file.write_text(
        "arrived_at,num_prefill_tokens,num_decode_tokens,session_id\n"
        "nan,32,8,7\n",
        encoding="utf-8",
    )
    command = [
        sys.executable,
        *_common_args(
            tmp_path / "metrics",
            "invalid_nan_arrival",
            trace_file=trace_file,
        ),
        *_colocation_args(),
    ]
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(REPO_ROOT),
            "PYTHONDONTWRITEBYTECODE": "1",
            "WANDB_DISABLED": "true",
            "VIDUR_DISABLE_WANDB": "1",
        }
    )

    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=120,
    )

    assert completed.returncode != 0
    assert "arrived_at must be finite" in completed.stdout
    assert "AssertionError" not in completed.stdout


def test_same_scheduling_window_does_not_publish_uncomputed_session_blocks(
    tmp_path: Path,
) -> None:
    trace_file = tmp_path / "same_window.csv"
    trace_file.write_text(
        "arrived_at,num_prefill_tokens,num_decode_tokens,session_id\n"
        "0.0,32,16,7\n"
        "0.0,8,8,7\n",
        encoding="utf-8",
    )

    request_metrics, _ = _run_simulation(
        output_root=tmp_path / "metrics",
        run_id="session_prefix_same_window",
        architecture_args=_colocation_args(),
        trace_file=trace_file,
    )

    assert request_metrics["request_cached_prefill_tokens"].tolist() == [0, 0]


def test_block_hash_runtime_regression_keeps_existing_behavior(
    tmp_path: Path,
) -> None:
    request_metrics, system_metrics = _run_simulation(
        output_root=tmp_path / "metrics",
        run_id="block_hash_prefix_regression",
        architecture_args=_colocation_args(),
        trace_file=BLOCK_HASH_TRACE_FILE,
        key_mode="block_hash",
    )

    assert request_metrics["request_cached_prefill_tokens"].tolist() == [0, 16]
    assert request_metrics["request_prefix_cache_hit_blocks"].tolist() == [0, 1]
    assert system_metrics["prefix_cache_statistics"]["key_mode"] == "block_hash"


def test_incremental_session_isl_materializes_thinking_rounds_at_runtime(
    tmp_path: Path,
) -> None:
    trace_file = tmp_path / "incremental_thinking.csv"
    trace_file.write_text(
        "arrived_at,num_prefill_tokens,num_decode_tokens,session_id,"
        "thinking_depth,thinking_round_plans_json\n"
        '0.0,32,8,7,2,"[{""num_prefill_tokens"":48,'
        '""num_decode_tokens"":16},{""num_prefill_tokens"":32,'
        '""num_decode_tokens"":8}]"\n',
        encoding="utf-8",
    )

    request_metrics, system_metrics = _run_simulation(
        output_root=tmp_path / "metrics",
        run_id="session_prefix_incremental_thinking",
        architecture_args=_colocation_args(),
        trace_file=trace_file,
    )

    assert request_metrics["request_num_prefill_tokens"].tolist() == [96]
    assert request_metrics["request_num_decode_tokens"].tolist() == [8]
    assert request_metrics["request_thinking_round_count"].tolist() == [1]
    assert request_metrics["request_cached_prefill_tokens"].tolist() == [64]
    assert request_metrics["request_prefix_cache_query_blocks"].tolist() == [9]
    assert request_metrics["request_prefix_cache_hit_blocks"].tolist() == [4]
    assert system_metrics["prefix_cache_statistics"]["total_query_blocks"] == 9
    assert system_metrics["prefix_cache_statistics"]["total_hit_blocks"] == 4
    assert (
        system_metrics["prefix_cache_statistics"]["total_cached_prefill_tokens"]
        == 64
    )


def test_incremental_thinking_round_scale_factors_apply_at_runtime(
    tmp_path: Path,
) -> None:
    trace_file = tmp_path / "scaled_incremental_thinking.csv"
    trace_file.write_text(
        "arrived_at,num_prefill_tokens,num_decode_tokens,session_id,"
        "thinking_depth,thinking_round_plans_json\n"
        '2.0,32,8,7,2,"[{""num_prefill_tokens"":48,'
        '""num_decode_tokens"":16},{""num_prefill_tokens"":32,'
        '""num_decode_tokens"":8}]"\n',
        encoding="utf-8",
    )

    request_metrics, system_metrics = _run_simulation(
        output_root=tmp_path / "metrics",
        run_id="session_prefix_scaled_incremental_thinking",
        architecture_args=_colocation_args(),
        trace_file=trace_file,
        prefill_scale_factor=0.5,
        decode_scale_factor=0.5,
        time_scale_factor=2.0,
    )

    assert request_metrics["request_num_prefill_tokens"].tolist() == [48]
    assert request_metrics["request_num_decode_tokens"].tolist() == [4]
    assert request_metrics["request_prefix_cache_query_blocks"].tolist() == [4]
    assert system_metrics["prefix_cache_statistics"]["total_query_blocks"] == 4


def test_session_prefix_cache_preserves_affinity_across_dp_lanes(
    tmp_path: Path,
) -> None:
    request_metrics, _ = _run_simulation(
        output_root=tmp_path / "metrics",
        run_id="session_prefix_dp2_affinity",
        architecture_args=_colocation_args(dp_size=2),
    )

    assert request_metrics["request_session_id"].tolist() == [7, 8, 7, 8]
    assert request_metrics["request_cached_prefill_tokens"].tolist() == [
        0,
        0,
        48,
        48,
    ]
