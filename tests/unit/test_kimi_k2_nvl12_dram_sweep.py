from pathlib import Path

import pytest

from frontier.config.model_config import BaseModelConfig
from scripts.generate_kimi_k2_probabilistic_workload import (
    WorkloadConfig,
    generate_rows,
)
from scripts.run_kimi_k2_nvl12_dram_sweep import (
    _collect_prefix_tier_metrics,
    _normalize_python_executable,
    build_trial,
    write_dashboard,
)
from scripts.run_kimi_k2_nvl12_pressure128k_server_sweep import (
    build_capacity_range,
)


def test_kimi_k2_frontier_config_matches_study_contract():
    model = BaseModelConfig.create_from_name("moonshotai/Kimi-K2-Instruct")

    assert model.num_layers == 61
    assert model.embedding_dim == 7168
    assert model.num_q_heads == 64
    assert model.num_experts == 384
    assert model.num_experts_per_tok == 8
    assert model.get_num_moe_layers() == 60
    assert model.is_moe_layer(0) is False
    assert model.is_moe_layer(1) is True
    assert model.share_expert_dim is None
    assert model.uses_mla() is True
    assert model.get_runtime_num_kv_heads() == 1
    assert model.get_runtime_head_size() == 576


def test_probabilistic_workload_is_deterministic_and_respects_64k():
    config = WorkloadConfig(sessions=12, seed=19)
    rows_a, manifest_a = generate_rows(config)
    rows_b, manifest_b = generate_rows(config)

    assert rows_a == rows_b
    assert manifest_a == manifest_b
    assert len(rows_a) >= config.sessions * 2
    assert max(
        int(row["_effective_prompt_tokens"]) + int(row["num_decode_tokens"])
        for row in rows_a
    ) <= 65_536
    assert {int(row["session_id"]) for row in rows_a} == set(
        range(config.sessions)
    )
    for session_id in range(config.sessions):
        session_rows = [
            row for row in rows_a if int(row["session_id"]) == session_id
        ]
        assert [int(row["turn_index"]) for row in session_rows] == list(
            range(len(session_rows))
        )
        assert session_rows[0]["arrived_at"] != ""
        assert float(session_rows[0]["think_time"]) == 0.0
        assert all(row["arrived_at"] == "" for row in session_rows[1:])
        assert all(float(row["think_time"]) > 0 for row in session_rows[1:])


def test_tool_seconds_128k_workload_uses_only_second_scale_gaps():
    config = WorkloadConfig(
        sessions=12,
        seed=19,
        max_context_tokens=131_072,
        initial_median=32_768,
        initial_min=16_384,
        initial_max=65_536,
        new_context_median=16_384,
        new_context_min=4_096,
        new_context_max=32_768,
        think_time_profile="tool_seconds",
    )

    rows, manifest = generate_rows(config)

    assert max(
        int(row["_effective_prompt_tokens"]) + int(row["num_decode_tokens"])
        for row in rows
    ) <= 131_072
    assert set(manifest["realized"]["gap_class_counts"]) <= {
        "fast_tool",
        "normal_tool",
        "slow_tool",
    }
    assert manifest["realized"]["think_time_seconds"]["max"] <= 60.0


def test_python_executable_normalization_preserves_venv_symlink(tmp_path):
    system_python = tmp_path / "system-python"
    system_python.touch()
    venv_python = tmp_path / "venv-python"
    try:
        venv_python.symlink_to(system_python)
    except OSError as error:
        pytest.skip(f"symlink creation is unavailable: {error}")

    normalized = Path(_normalize_python_executable(str(venv_python)))

    assert normalized == venv_python.absolute()
    assert normalized != venv_python.resolve()


def test_sweep_trial_maps_tp4_dp3_ep12_and_per_cpu_dram_to_per_gpu():
    trial = build_trial(
        capacity_gb_per_cpu=96,
        output_dir=Path("outputs/test"),
        trace_path=Path("trace.csv"),
        python_executable="python",
    )
    command = list(trial.command)

    assert command[
        command.index(
            "--cluster_config_prefill_replica_config_attn_tensor_parallel_size"
        )
        + 1
    ] == "4"
    assert command[
        command.index(
            "--cluster_config_prefill_replica_config_attn_data_parallel_size"
        )
        + 1
    ] == "3"
    assert command[
        command.index(
            "--cluster_config_prefill_replica_config_moe_expert_parallel_size"
        )
        + 1
    ] == "12"
    assert command[
        command.index("--cpu_kv_cache_config_capacity_bytes_per_gpu") + 1
    ] == "48000000000"
    assert "vera_rubin_nvl12_partition" in command
    assert command[
        command.index("--vllm_v1_scheduler_config_num_blocks_mode") + 1
    ] == "memory_planner"
    assert command[
        command.index("--vllm_v1_scheduler_config_num_blocks") + 1
    ] == "0"
    assert command[command.index("--simulation_mode") + 1] == "online"


def test_prefix_tier_metrics_separate_gpu_and_cpu_reuse(tmp_path: Path):
    metrics_file = tmp_path / "request_metrics.csv"
    metrics_file.write_text(
        "\n".join(
            [
                "Request Id,request_session_id,request_num_prefill_tokens,"
                "request_cached_prefill_tokens,"
                "request_gpu_prefix_cache_hit_blocks,"
                "request_cpu_prefix_cache_hit_blocks",
                "0,7,32,0,0,0",
                "1,7,64,0,1,1",
                "2,7,80,0,3,1",
            ]
        ),
        encoding="utf-8",
    )

    metrics = _collect_prefix_tier_metrics(metrics_file, block_size=16)

    assert metrics["reusable_prefix_blocks"] == 6
    assert metrics["gpu_prefix_hit_blocks"] == 4
    assert metrics["cpu_eligible_blocks"] == 2
    assert metrics["cpu_prefix_hit_blocks"] == 2
    assert metrics["cpu_conditional_hit_ratio"] == 1.0
    assert metrics["remaining_reuse_miss_blocks"] == 0
    assert metrics["combined_reuse_hit_ratio"] == 1.0
    assert metrics["total_materialized_prompt_tokens"] == 176
    assert metrics["ideal_incremental_prefill_tokens"] == 80
    assert metrics["actual_prefill_compute_tokens"] == 176
    assert metrics["avoidable_prefill_compute_tokens"] == 96
    assert metrics["prefill_compute_amplification"] == 2.2
    assert metrics["prefill_compute_overhead_pct"] == 120.0
    assert metrics["full_replay_amplification"] == 2.2


def test_server_capacity_range_is_inclusive_and_validated():
    assert build_capacity_range(0, 1000, 200) == [
        float(value) for value in range(0, 1001, 200)
    ]

    try:
        build_capacity_range(0, 950, 100)
    except ValueError as error:
        assert "land exactly" in str(error)
    else:
        raise AssertionError("non-divisible capacity range should fail")


def test_dashboard_writes_standalone_html(tmp_path: Path):
    rows = []
    for capacity in (0.0, 100.0):
        rows.append(
            {
                "capacity_gb_per_cpu": capacity,
                "capacity_gb": capacity,
                "ttft_mean_ms": 10.0,
                "ttft_p90_ms": 12.0,
                "ttft_p99_ms": 15.0,
                "tpot_mean_ms": 2.0,
                "tpot_p90_ms": 2.2,
                "tpot_p99_ms": 2.5,
                "requests_per_second": 1.0,
                "decode_tokens_per_second": 100.0,
                "gpu_reuse_hit_ratio": 0.5,
                "cpu_conditional_hit_ratio": 0.5,
                "combined_reuse_hit_ratio": 0.75,
                "prefill_compute_amplification": 1.5,
                "full_replay_amplification": 3.0,
                "cpu_peak_resident_gb": capacity,
                "cpu_restore_gb": capacity / 2,
                "cpu_offload_gb": capacity,
                "cpu_evicted_gb": 0.0,
            }
        )

    path = write_dashboard(tmp_path, rows)

    contents = path.read_text(encoding="utf-8")
    assert path.name == "sweep_dashboard.html"
    assert "<html>" in contents
    assert "Kimi K2 NVL12 CPU DRAM Sweep" in contents
