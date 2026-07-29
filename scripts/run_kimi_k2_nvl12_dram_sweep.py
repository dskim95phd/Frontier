#!/usr/bin/env python3
"""Run the Kimi K2 NVL12 prefill/decode CPU-DRAM capacity sweep."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

try:
    from scripts.generate_kimi_k2_probabilistic_workload import (
        WorkloadConfig,
        write_workload,
    )
except ModuleNotFoundError:
    from generate_kimi_k2_probabilistic_workload import (
        WorkloadConfig,
        write_workload,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "kimi_k2_nvl12_dram_sweep"
MODEL_NAME = "moonshotai/Kimi-K2-Instruct"
GPU_COUNT_PER_CLUSTER = 12
GPU_COUNT_PER_CPU = 2
CPU_COUNT_PER_CLUSTER = GPU_COUNT_PER_CLUSTER // GPU_COUNT_PER_CPU


@dataclass(frozen=True)
class Trial:
    capacity_gb_per_cpu: float
    run_id: str
    command: tuple[str, ...]
    log_path: Path


def parse_capacities(value: str) -> list[float]:
    try:
        capacities = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "--capacities-gb must be comma-separated numbers"
        ) from error
    if not capacities or any(capacity < 0 for capacity in capacities):
        raise argparse.ArgumentTypeError(
            "--capacities-gb must contain non-negative values"
        )
    if len(set(capacities)) != len(capacities):
        raise argparse.ArgumentTypeError("--capacities-gb must not contain duplicates")
    return capacities


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capacities-gb",
        type=parse_capacities,
        default=parse_capacities("0,200,400,600,800,1000"),
        help=(
            "Comma-separated decimal GB assigned to each Vera CPU. "
            "An NVL12 partition has six Vera CPUs and two Rubin GPUs per CPU."
        ),
    )
    parser.add_argument("--sessions", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--session-arrival-rate", type=float, default=0.5)
    parser.add_argument(
        "--workload-profile",
        choices=("standard", "pressure64k", "pressure128k"),
        default="standard",
        help=(
            "pressure64k uses longer prompts and outputs so the natural GPU KV "
            "capacity can be stressed without an artificial block limit."
        ),
    )
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Do not write the standalone Plotly sweep_dashboard.html.",
    )
    parser.add_argument(
        "--cprofile",
        action="store_true",
        help="Write a per-trial Python cProfile file under output-dir/profiles.",
    )
    return parser


def _capacity_label(capacity_gb: float) -> str:
    if float(capacity_gb).is_integer():
        return str(int(capacity_gb))
    return str(capacity_gb).replace(".", "p")


def _normalize_python_executable(value: str) -> str:
    """Return an absolute executable path without resolving venv symlinks."""
    expanded = os.path.expanduser(value)
    discovered = shutil.which(expanded)
    return os.path.abspath(discovered or expanded)


def build_trial(
    *,
    capacity_gb_per_cpu: float,
    output_dir: Path,
    trace_path: Path,
    python_executable: str,
    enable_cprofile: bool = False,
    max_context_tokens: int = 65_536,
    batch_size_cap: int = 128,
) -> Trial:
    run_id = f"dram_{_capacity_label(capacity_gb_per_cpu)}gb_per_cpu"
    log_path = output_dir / "logs" / f"{run_id}.log"
    command = [python_executable]
    if enable_cprofile:
        profile_path = output_dir / "profiles" / f"{run_id}.pstats"
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        command.extend(["-m", "cProfile", "-o", str(profile_path)])
    command.extend(
        [
        "-m", "frontier.main",
        "--simulation_mode",
        "online",
        "--sys_arch",
        "pd-disaggregation",
        "--no-enable_parallel_clusters",
        "--cluster_scheduler_config_type",
        "sticky_round_robin",
        "--cluster_config_prefill_cluster_num_replicas",
        "1",
        "--cluster_config_decode_cluster_num_replicas",
        "1",
        "--cluster_config_prefill_cluster_num_racks",
        "1",
        "--cluster_config_decode_cluster_num_racks",
        "1",
        "--cluster_config_prefill_replica_config_num_pipeline_stages",
        "1",
        "--cluster_config_decode_replica_config_num_pipeline_stages",
        "1",
        "--cluster_config_prefill_replica_config_attn_tensor_parallel_size",
        "4",
        "--cluster_config_decode_replica_config_attn_tensor_parallel_size",
        "4",
        "--cluster_config_prefill_replica_config_attn_data_parallel_size",
        "3",
        "--cluster_config_decode_replica_config_attn_data_parallel_size",
        "3",
        "--cluster_config_prefill_replica_config_moe_tensor_parallel_size",
        "1",
        "--cluster_config_decode_replica_config_moe_tensor_parallel_size",
        "1",
        "--cluster_config_prefill_replica_config_moe_expert_parallel_size",
        "12",
        "--cluster_config_decode_replica_config_moe_expert_parallel_size",
        "12",
        "--cluster_config_prefill_replica_config_total_expert_num",
        "384",
        "--cluster_config_decode_replica_config_total_expert_num",
        "384",
        "--cluster_config_prefill_replica_config_router_topk",
        "8",
        "--cluster_config_decode_replica_config_router_topk",
        "8",
        "--cluster_config_prefill_replica_config_device",
        "rubin",
        "--cluster_config_decode_replica_config_device",
        "rubin",
        "--cluster_config_prefill_replica_config_network_device",
        "vera_rubin_nvl12_partition",
        "--cluster_config_decode_replica_config_network_device",
        "vera_rubin_nvl12_partition",
        "--cluster_config_prefill_replica_config_memory_margin_fraction",
        "0.10",
        "--cluster_config_decode_replica_config_memory_margin_fraction",
        "0.10",
        "--replica_config_model_name",
        MODEL_NAME,
        "--replica_config_moe_routing_mode",
        "simulation",
        "--replica_config_moe_routing_seed",
        "20260728",
        "--replica_config_moe_routing_distribution_type",
        "balanced",
        "--replica_scheduler_config_type",
        "vllm_v1",
        "--decode_cuda_graph_mode",
        "none",
        "--vllm_v1_scheduler_config_enable_chunked_prefill",
        "--vllm_v1_scheduler_config_batch_size_cap",
        str(batch_size_cap),
        "--vllm_v1_scheduler_config_max_tokens_in_batch",
        "8192",
        "--vllm_v1_scheduler_config_long_prefill_token_threshold",
        "8192",
        "--vllm_v1_scheduler_config_block_size",
        "16",
        "--vllm_v1_scheduler_config_num_blocks_mode",
        "memory_planner",
        "--vllm_v1_scheduler_config_num_blocks",
        "0",
        "--vllm_v1_scheduler_config_gpu_memory_utilization",
        "0.90",
        "--vllm_v1_scheduler_config_enable_prefix_caching",
        "--vllm_v1_scheduler_config_prefix_caching_key_mode",
        "session",
        "--request_generator_config_type",
        "trace_replay",
        "--trace_request_generator_config_trace_file",
        str(trace_path),
        "--trace_request_generator_config_max_tokens",
        str(max_context_tokens),
        "--execution_time_predictor_config_type",
        "analytical_roofline",
        "--no-analytical_roofline_execution_time_predictor_config_keep_diagnostics",
        "--cc_backend_config_type",
        "astra_sim_analytical",
        "--astra_sim_analytical_cc_backend_config_intra_server_topology",
        "Switch",
        "--astra_sim_analytical_cc_backend_config_intra_server_bandwidth_gbps",
        "14400",
        "--astra_sim_analytical_cc_backend_config_intra_server_latency_us",
        "1.0",
        "--analytical_kv_cache_transfer_config_network_bandwidth_gbps",
        "38400",
        "--analytical_kv_cache_transfer_config_network_latency_ms",
        "0.02",
        "--metrics_config_output_dir",
        str(output_dir),
        "--metrics_config_run_id",
        run_id,
        "--metrics_config_write_metrics",
        "--metrics_config_store_request_metrics",
        "--metrics_config_enable_metrics_ground_truth_trace",
        "--no-metrics_config_store_batch_metrics",
        "--no-metrics_config_store_utilization_metrics",
        "--no-metrics_config_store_token_completion_metrics",
        "--no-metrics_config_store_plots",
        "--no-metrics_config_enable_chrome_trace",
        "--no-metrics_config_write_json_trace",
        "--no-metrics_config_store_frontier_stage_batch_ledger",
        ]
    )
    if capacity_gb_per_cpu > 0:
        per_gpu_bytes = int(
            round(
                capacity_gb_per_cpu
                * 1_000_000_000
                / GPU_COUNT_PER_CPU
            )
        )
        command.extend(
            [
                "--cpu_kv_cache_config_enable",
                "--cpu_kv_cache_config_static_slice_per_gpu",
                "--cpu_kv_cache_config_capacity_bytes_per_gpu",
                str(per_gpu_bytes),
                "--cpu_kv_cache_config_dram_bandwidth_gbps_per_gpu",
                "4800",
                "--cpu_kv_cache_config_c2c_bandwidth_gbps_per_gpu",
                "3600",
                "--cpu_kv_cache_config_capacity_pressure_policy",
                "prefix_fit",
            ]
        )
    return Trial(
        capacity_gb_per_cpu=capacity_gb_per_cpu,
        run_id=run_id,
        command=tuple(command),
        log_path=log_path,
    )


def _find_run_dir(output_dir: Path, run_id: str) -> Path | None:
    candidates = [
        path
        for path in output_dir.rglob(run_id)
        if path.is_dir()
        and (path / "system_metrics.json").is_file()
        and (path / "request_metrics.csv").is_file()
    ]
    if len(candidates) > 1:
        raise RuntimeError(f"multiple metrics directories found for {run_id}: {candidates}")
    return candidates[0] if candidates else None


def run_trial(trial: Trial, output_dir: Path, resume: bool) -> tuple[Trial, int]:
    if resume and _find_run_dir(output_dir, trial.run_id) is not None:
        return trial, 0
    trial.log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO_ROOT)
    environment["WANDB_DISABLED"] = "true"
    environment["VIDUR_DISABLE_WANDB"] = "1"
    environment["FRONTIER_LOG_LEVEL"] = "WARNING"
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("MKL_NUM_THREADS", "1")
    started_at = time.perf_counter()
    with trial.log_path.open("w", encoding="utf-8") as log:
        log.write(f"Command: {shlex.join(trial.command)}\n\n")
        completed = subprocess.run(
            trial.command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    elapsed_seconds = time.perf_counter() - started_at
    trial.log_path.with_suffix(".timing.json").write_text(
        json.dumps(
            {
                "run_id": trial.run_id,
                "wall_clock_seconds": elapsed_seconds,
                "return_code": completed.returncode,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return trial, completed.returncode


def _nested(metrics: dict[str, Any], *path: str, default: Any = 0.0) -> Any:
    current: Any = metrics
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _collect_prefix_tier_metrics(
    request_metrics_path: Path,
    *,
    block_size: int,
) -> dict[str, float | int]:
    with request_metrics_path.open("r", encoding="utf-8", newline="") as source:
        rows = sorted(
            csv.DictReader(source),
            key=lambda row: int(row["Request Id"]),
        )

    previous_prefill_tokens_by_session: dict[int, int] = {}
    reusable_blocks = 0
    gpu_hit_blocks = 0
    cpu_hit_blocks = 0
    total_prompt_tokens = 0
    actual_cached_tokens = 0
    ideal_cached_tokens = 0
    for row in rows:
        session_id = int(row["request_session_id"])
        prefill_tokens = int(row["request_num_prefill_tokens"])
        previous_prefill_tokens = previous_prefill_tokens_by_session.get(
            session_id
        )
        if previous_prefill_tokens is not None:
            reusable_blocks += previous_prefill_tokens // block_size
            ideal_cached_tokens += (
                previous_prefill_tokens // block_size
            ) * block_size
        gpu_hit_blocks += int(row["request_gpu_prefix_cache_hit_blocks"])
        cpu_hit_blocks += int(row["request_cpu_prefix_cache_hit_blocks"])
        total_prompt_tokens += prefill_tokens
        actual_cached_tokens += int(row["request_cached_prefill_tokens"])
        previous_prefill_tokens_by_session[session_id] = prefill_tokens

    cpu_eligible_blocks = max(reusable_blocks - gpu_hit_blocks, 0)
    remaining_miss_blocks = max(cpu_eligible_blocks - cpu_hit_blocks, 0)
    actual_prefill_compute_tokens = total_prompt_tokens - actual_cached_tokens
    ideal_incremental_prefill_tokens = total_prompt_tokens - ideal_cached_tokens
    avoidable_prefill_compute_tokens = max(
        actual_prefill_compute_tokens - ideal_incremental_prefill_tokens,
        0,
    )
    return {
        "reusable_prefix_blocks": reusable_blocks,
        "gpu_prefix_hit_blocks": gpu_hit_blocks,
        "gpu_reuse_hit_ratio": (
            gpu_hit_blocks / reusable_blocks if reusable_blocks else 0.0
        ),
        "cpu_eligible_blocks": cpu_eligible_blocks,
        "cpu_prefix_hit_blocks": cpu_hit_blocks,
        "cpu_conditional_hit_ratio": (
            cpu_hit_blocks / cpu_eligible_blocks
            if cpu_eligible_blocks
            else 0.0
        ),
        "cpu_assisted_reuse_ratio": (
            cpu_hit_blocks / reusable_blocks if reusable_blocks else 0.0
        ),
        "combined_reuse_hit_ratio": (
            min((gpu_hit_blocks + cpu_hit_blocks) / reusable_blocks, 1.0)
            if reusable_blocks
            else 0.0
        ),
        "remaining_reuse_miss_blocks": remaining_miss_blocks,
        "total_materialized_prompt_tokens": total_prompt_tokens,
        "actual_cached_prefill_tokens": actual_cached_tokens,
        "actual_prefill_compute_tokens": actual_prefill_compute_tokens,
        "ideal_incremental_prefill_tokens": ideal_incremental_prefill_tokens,
        "avoidable_prefill_compute_tokens": avoidable_prefill_compute_tokens,
        "prefill_compute_amplification": (
            actual_prefill_compute_tokens / ideal_incremental_prefill_tokens
            if ideal_incremental_prefill_tokens
            else 0.0
        ),
        "prefill_compute_overhead_pct": (
            100.0
            * avoidable_prefill_compute_tokens
            / ideal_incremental_prefill_tokens
            if ideal_incremental_prefill_tokens
            else 0.0
        ),
        "full_replay_amplification": (
            total_prompt_tokens / ideal_incremental_prefill_tokens
            if ideal_incremental_prefill_tokens
            else 0.0
        ),
    }


def collect_result(output_dir: Path, trial: Trial) -> dict[str, Any]:
    run_dir = _find_run_dir(output_dir, trial.run_id)
    if run_dir is None:
        raise FileNotFoundError(f"metrics not found for {trial.run_id}")
    system_metrics = json.loads(
        (run_dir / "system_metrics.json").read_text(encoding="utf-8")
    )
    cpu = system_metrics.get("cpu_kv_cache_statistics", {})
    prefix = system_metrics.get("prefix_cache_statistics", {})
    throughput = system_metrics.get("throughput_metrics", {})
    tier_metrics = _collect_prefix_tier_metrics(
        run_dir / "request_metrics.csv",
        block_size=int(prefix.get("block_size_tokens", 16)),
    )
    timing_path = trial.log_path.with_suffix(".timing.json")
    timing = (
        json.loads(timing_path.read_text(encoding="utf-8"))
        if timing_path.is_file()
        else {}
    )
    return {
        "capacity_gb_per_cpu": trial.capacity_gb_per_cpu,
        "capacity_tb_per_cpu": trial.capacity_gb_per_cpu / 1000.0,
        "total_cpu_dram_gb": (
            trial.capacity_gb_per_cpu * CPU_COUNT_PER_CLUSTER
        ),
        "vera_cpu_count": CPU_COUNT_PER_CLUSTER,
        # Compatibility aliases: capacities are now per Vera CPU.
        "capacity_gb": trial.capacity_gb_per_cpu,
        "capacity_tb": trial.capacity_gb_per_cpu / 1000.0,
        "run_id": trial.run_id,
        "wall_clock_seconds": timing.get("wall_clock_seconds"),
        "completed_requests": _nested(
            system_metrics, "simulation_metadata", "completed_requests"
        ),
        "ttft_mean_ms": _nested(system_metrics, "ttft_statistics", "mean"),
        "ttft_p50_ms": _nested(system_metrics, "ttft_statistics", "p50"),
        "ttft_p90_ms": _nested(system_metrics, "ttft_statistics", "p90"),
        "ttft_p99_ms": _nested(system_metrics, "ttft_statistics", "p99"),
        "tpot_mean_ms": _nested(system_metrics, "tpot_statistics", "mean"),
        "tpot_p50_ms": _nested(system_metrics, "tpot_statistics", "p50"),
        "tpot_p90_ms": _nested(system_metrics, "tpot_statistics", "p90"),
        "tpot_p99_ms": _nested(system_metrics, "tpot_statistics", "p99"),
        "requests_per_second": throughput.get("requests_per_second", 0.0),
        "tokens_per_second": throughput.get("tokens_per_second", 0.0),
        "decode_tokens_per_second": throughput.get(
            "decode_tokens_per_second", 0.0
        ),
        "prefix_hit_ratio": prefix.get("hit_ratio", 0.0),
        "cpu_hit_ratio": cpu.get("cpu_hit_ratio", 0.0),
        **tier_metrics,
        "cpu_restore_blocks": cpu.get("restore_blocks", 0),
        "cpu_offload_blocks": cpu.get("offload_blocks", 0),
        "sessions_with_cpu_hits": cpu.get("sessions_with_cpu_hits", 0),
        "cpu_peak_resident_gb": cpu.get("peak_resident_bytes", 0) / 1e9,
        "cpu_evicted_gb": cpu.get("evicted_bytes", 0) / 1e9,
        "cpu_restore_gb": cpu.get("restore_bytes", 0) / 1e9,
        "cpu_offload_gb": cpu.get("offload_bytes", 0) / 1e9,
        "cpu_restore_time_ms": cpu.get("restore_transfer_time_ms", 0.0),
        "cpu_restore_queue_time_ms": cpu.get("restore_queue_time_ms", 0.0),
        "cpu_offload_time_ms": cpu.get("offload_transfer_time_ms", 0.0),
        "cpu_offload_queue_time_ms": cpu.get("offload_queue_time_ms", 0.0),
        "cpu_evicted_sessions": cpu.get("evicted_sessions", 0),
        "gpu_weight_memory_gb_per_device": _nested(
            system_metrics,
            "model_weight_memory",
            "PREFILL",
            "total_memory_gb",
        ),
        "metrics_dir": str(run_dir),
    }


def write_results(output_dir: Path, results: list[dict[str, Any]]) -> None:
    results.sort(
        key=lambda row: float(
            row.get("capacity_gb_per_cpu", row.get("capacity_gb", 0.0))
        )
    )
    baseline = results[0]
    for result in results:
        result["ttft_mean_improvement_pct"] = (
            100.0
            * (baseline["ttft_mean_ms"] - result["ttft_mean_ms"])
            / baseline["ttft_mean_ms"]
        )
        result["tpot_mean_improvement_pct"] = (
            100.0
            * (baseline["tpot_mean_ms"] - result["tpot_mean_ms"])
            / baseline["tpot_mean_ms"]
        )
        result["throughput_improvement_pct"] = (
            100.0
            * (result["requests_per_second"] - baseline["requests_per_second"])
            / baseline["requests_per_second"]
        )

    json_path = output_dir / "sweep_results.json"
    csv_path = output_dir / "sweep_results.csv"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)


def write_dashboard(
    output_dir: Path,
    results: list[dict[str, Any]],
) -> Path:
    """Write a standalone interactive HTML dashboard for the sweep."""

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    ordered = sorted(
        results,
        key=lambda row: float(
            row.get("capacity_gb_per_cpu", row.get("capacity_gb", 0.0))
        ),
    )
    capacities = [
        float(row.get("capacity_gb_per_cpu", row.get("capacity_gb", 0.0)))
        for row in ordered
    ]

    figure = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "TTFT",
            "TPOT",
            "Throughput",
            "Reusable prefix hit rate",
            "Prefill compute amplification",
            "CPU KV-cache traffic and residency",
        ),
        specs=(
            ({}, {}),
            ({"secondary_y": True}, {}),
            ({}, {}),
        ),
        horizontal_spacing=0.10,
        vertical_spacing=0.12,
    )

    def add_line(
        *,
        key: str,
        name: str,
        row: int,
        col: int,
        scale: float = 1.0,
        secondary_y: bool | None = None,
    ) -> None:
        values = [float(item.get(key, 0.0) or 0.0) * scale for item in ordered]
        kwargs: dict[str, Any] = {"row": row, "col": col}
        if secondary_y is not None:
            kwargs["secondary_y"] = secondary_y
        figure.add_trace(
            go.Scatter(
                x=capacities,
                y=values,
                mode="lines+markers",
                name=name,
                hovertemplate=(
                    "CPU DRAM %{x:g} GB / Vera CPU<br>"
                    + name
                    + " %{y:,.3f}<extra></extra>"
                ),
            ),
            **kwargs,
        )

    for key, name in (
        ("ttft_mean_ms", "TTFT mean"),
        ("ttft_p90_ms", "TTFT p90"),
        ("ttft_p99_ms", "TTFT p99"),
    ):
        add_line(key=key, name=name, row=1, col=1)
    for key, name in (
        ("tpot_mean_ms", "TPOT mean"),
        ("tpot_p90_ms", "TPOT p90"),
        ("tpot_p99_ms", "TPOT p99"),
    ):
        add_line(key=key, name=name, row=1, col=2)

    add_line(
        key="requests_per_second",
        name="Requests/s",
        row=2,
        col=1,
        secondary_y=False,
    )
    add_line(
        key="decode_tokens_per_second",
        name="Decode tokens/s",
        row=2,
        col=1,
        secondary_y=True,
    )

    for key, name in (
        ("gpu_reuse_hit_ratio", "GPU reuse hit"),
        ("cpu_conditional_hit_ratio", "CPU conditional hit"),
        ("combined_reuse_hit_ratio", "GPU + CPU reuse hit"),
    ):
        add_line(key=key, name=name, row=2, col=2, scale=100.0)

    for key, name in (
        ("prefill_compute_amplification", "Actual prefill amplification"),
        ("full_replay_amplification", "Full replay amplification"),
    ):
        add_line(key=key, name=name, row=3, col=1)

    for key, name in (
        ("cpu_peak_resident_gb", "Peak CPU resident"),
        ("cpu_restore_gb", "CPU restored"),
        ("cpu_offload_gb", "CPU offloaded"),
        ("cpu_evicted_gb", "CPU evicted"),
    ):
        add_line(key=key, name=name, row=3, col=2)

    figure.update_xaxes(
        title_text="CPU DRAM capacity per Vera CPU (decimal GB)"
    )
    figure.update_yaxes(title_text="Latency (ms)", row=1, col=1)
    figure.update_yaxes(title_text="Latency (ms/token)", row=1, col=2)
    figure.update_yaxes(
        title_text="Requests/s",
        row=2,
        col=1,
        secondary_y=False,
    )
    figure.update_yaxes(
        title_text="Decode tokens/s",
        row=2,
        col=1,
        secondary_y=True,
    )
    figure.update_yaxes(title_text="Hit rate (%)", range=[0, 105], row=2, col=2)
    figure.update_yaxes(title_text="Actual / ideal", row=3, col=1)
    figure.update_yaxes(title_text="Data volume (GB)", row=3, col=2)
    figure.update_layout(
        title="Kimi K2 NVL12 CPU DRAM Sweep (per Vera CPU)",
        template="plotly_white",
        height=1200,
        hovermode="x unified",
        legend={"orientation": "h", "y": -0.08},
        margin={"l": 70, "r": 70, "t": 90, "b": 150},
    )

    dashboard_path = output_dir / "sweep_dashboard.html"
    figure.write_html(
        dashboard_path,
        include_plotlyjs=True,
        full_html=True,
    )
    return dashboard_path


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_workers <= 0:
        raise SystemExit("--max-workers must be positive")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "workload.csv"
    manifest_path = output_dir / "workload.manifest.json"
    workload_config = WorkloadConfig(
        sessions=args.sessions,
        seed=args.seed,
        session_arrival_rate=args.session_arrival_rate,
    )
    if args.workload_profile == "pressure64k":
        workload_config = WorkloadConfig(
            sessions=args.sessions,
            seed=args.seed,
            session_arrival_rate=args.session_arrival_rate,
            max_context_tokens=65_536,
            geometric_p=0.20,
            max_turns=10,
            initial_median=16_384,
            initial_sigma=0.55,
            initial_min=8_192,
            initial_max=32_768,
            output_median=128,
            output_sigma=0.70,
            output_min=16,
            output_max=512,
            new_context_median=8_192,
            new_context_sigma=0.60,
            new_context_min=2_048,
            new_context_max=16_384,
        )
    elif args.workload_profile == "pressure128k":
        workload_config = WorkloadConfig(
            sessions=args.sessions,
            seed=args.seed,
            session_arrival_rate=args.session_arrival_rate,
            max_context_tokens=131_072,
            geometric_p=0.20,
            max_turns=10,
            initial_median=32_768,
            initial_sigma=0.55,
            initial_min=16_384,
            initial_max=65_536,
            output_median=128,
            output_sigma=0.70,
            output_min=16,
            output_max=512,
            new_context_median=16_384,
            new_context_sigma=0.60,
            new_context_min=4_096,
            new_context_max=32_768,
            think_time_profile="tool_seconds",
        )
    write_workload(
        trace_path,
        manifest_path,
        workload_config,
    )
    trials = [
        build_trial(
            capacity_gb_per_cpu=capacity,
            output_dir=output_dir,
            trace_path=trace_path,
            python_executable=_normalize_python_executable(args.python),
            enable_cprofile=args.cprofile,
            max_context_tokens=workload_config.max_context_tokens,
            batch_size_cap=(
                8 if args.workload_profile == "pressure128k" else 128
            ),
        )
        for capacity in args.capacities_gb
    ]
    if args.dry_run:
        for trial in trials:
            print(f"\n[{trial.run_id}]\n{shlex.join(trial.command)}")
        return 0

    failures: list[Trial] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.max_workers
    ) as executor:
        futures = {
            executor.submit(run_trial, trial, output_dir, args.resume): trial
            for trial in trials
        }
        for future in concurrent.futures.as_completed(futures):
            trial, return_code = future.result()
            if return_code == 0:
                print(f"completed: {trial.run_id}")
            else:
                failures.append(trial)
                print(
                    f"failed: {trial.run_id}; inspect {trial.log_path}",
                    file=sys.stderr,
                )
    if failures:
        return 1

    results = [collect_result(output_dir, trial) for trial in trials]
    write_results(output_dir, results)
    print(f"wrote {output_dir / 'sweep_results.csv'}")
    if not args.no_dashboard:
        dashboard_path = write_dashboard(output_dir, results)
        print(f"wrote {dashboard_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
