#!/usr/bin/env python3
"""Run an offline PDD batch-cap sweep through the public MoE PDD example.

The default preset represents 128 logical H800 GPUs as two 64-GPU clusters:
eight PREFILL replicas and eight DECODE replicas, where every replica is
TP4 x DP2 for attention and MoE-TP1 x EP8 for MoE.  By default it uses the
profiled H800 ``Qwen3-30B-A3B-tiny`` model (8 layers, 16 experts, top-k 8),
not dummy execution time.

Each trial is a separate Frontier process.  ``--max-workers`` controls how
many trials run concurrently.  The first real-predictor run should use one
worker so its predictor caches can be trained without contention.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXAMPLE = REPO_ROOT / "examples" / "architecture" / "pdd" / "offline" / "moe_model_basic.sh"


@dataclass(frozen=True)
class Trial:
    batch_cap: int
    run_id: str
    command: tuple[str, ...]
    environment: dict[str, str]
    log_path: Path


def parse_batch_caps(value: str) -> list[int]:
    try:
        caps = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("--batch-caps must be comma-separated integers") from error
    if not caps or any(cap <= 0 for cap in caps):
        raise argparse.ArgumentTypeError("--batch-caps must contain one or more positive integers")
    if len(set(caps)) != len(caps):
        raise argparse.ArgumentTypeError("--batch-caps must not contain duplicates")
    return caps


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep per-replica batch caps for the profiled P64/D64 Qwen3-A3B-tiny MoE PDD preset."
    )
    parser.add_argument("--batch-caps", type=parse_batch_caps, default=parse_batch_caps("1,2,4,8"))
    parser.add_argument("--max-workers", type=int, default=1, help="Concurrent Frontier subprocesses.")
    parser.add_argument("--num-requests", type=int, default=256, help="Offline requests supplied at simulated time zero.")
    parser.add_argument("--chunk-size", type=int, default=64, help="Chunked-prefill tokens per request per iteration.")
    parser.add_argument("--isl", type=int, default=32768, help="Fixed input length in tokens.")
    parser.add_argument("--osl", type=int, default=8192, help="Fixed output length in tokens.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "pdd_h800_qwen3_a3b_tiny")
    parser.add_argument("--run-prefix", default="batchcap")
    parser.add_argument("--device", default="h800")
    parser.add_argument("--network-device", default="h800_dgx")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--num-blocks", type=int, default=0, help="0 derives KV capacity from the memory planner.")
    parser.add_argument("--nvlink-gbps", type=float, default=3200.0, help="Assumed intra-node link bandwidth for ASTRA analytical CC.")
    parser.add_argument("--ib-gbps", type=float, default=400.0, help="Assumed inter-node IB bandwidth and P-to-D KV-transfer bandwidth.")
    parser.add_argument("--kv-latency-ms", type=float, default=0.02, help="Assumed P-to-D KV-transfer latency.")
    parser.add_argument("--dummy", action="store_true", help="Use fixed dummy execution time instead of the included H800 tiny-model profiles.")
    parser.add_argument("--dummy-execution-time-ms", type=float, default=1.0)
    parser.add_argument("--python", default=sys.executable, help="Python executable passed to the example script.")
    parser.add_argument("--example-script", type=Path, default=DEFAULT_EXAMPLE)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.max_workers <= 0:
        raise ValueError("--max-workers must be positive")
    if args.num_requests <= 0 or args.chunk_size <= 0 or args.isl <= 0 or args.osl <= 0:
        raise ValueError("--num-requests, --chunk-size, --isl, and --osl must be positive")
    if args.num_blocks < 0:
        raise ValueError("--num-blocks must be non-negative")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("--gpu-memory-utilization must be in (0, 1]")
    if min(args.nvlink_gbps, args.ib_gbps, args.kv_latency_ms, args.dummy_execution_time_ms) <= 0:
        raise ValueError("Bandwidths, KV latency, and dummy execution time must be positive")
    if not args.example_script.resolve().is_file():
        raise ValueError(f"PDD example script does not exist: {args.example_script}")
    if shutil.which("bash") is None:
        raise ValueError("This driver requires bash because the public PDD example is a bash script")


def build_trial(args: argparse.Namespace, batch_cap: int, output_dir: Path) -> Trial:
    run_id = f"{args.run_prefix}_{batch_cap}"
    log_path = output_dir / "logs" / f"{run_id}.log"
    prefill_budget = batch_cap * args.chunk_size

    environment = os.environ.copy()
    environment.update(
        {
            "PYTHON_BIN": str(Path(args.python).resolve()),
            "MODEL_NAME": "Qwen3-30B-A3B-tiny",
            "PREFILL_REPLICAS": "8",
            "DECODE_REPLICAS": "8",
            "PREFILL_ATTN_TP": "4",
            "PREFILL_ATTN_DP": "2",
            "PREFILL_MOE_TP": "1",
            "PREFILL_MOE_EP": "8",
            "DECODE_ATTN_TP": "4",
            "DECODE_ATTN_DP": "2",
            "DECODE_MOE_TP": "1",
            "DECODE_MOE_EP": "8",
            "PREFILL_DEVICE": args.device,
            "DECODE_DEVICE": args.device,
            "PREFILL_MEMORY_MARGIN_FRACTION": str(1.0 - args.gpu_memory_utilization),
            "DECODE_MEMORY_MARGIN_FRACTION": str(1.0 - args.gpu_memory_utilization),
            "TOTAL_EXPERTS": "16",
            "ROUTER_TOPK": "8",
            "MOE_ROUTING_MODE": "uniform_random",
            "PREFILL_TOKENS": str(args.isl),
            "DECODE_TOKENS": str(args.osl),
            "NUM_REQUESTS": str(args.num_requests),
            "MAX_TOKENS_IN_BATCH": str(prefill_budget),
            "LONG_PREFILL_TOKEN_THRESHOLD": str(args.chunk_size),
            "NUM_BLOCKS": str(args.num_blocks),
            "KV_TRANSFER_BANDWIDTH_GBPS": str(args.ib_gbps),
            "KV_TRANSFER_LATENCY_MS": str(args.kv_latency_ms),
            "METRICS_OUTPUT_DIR": str(output_dir),
            "RUN_ID": run_id,
            "ENABLE_DUMMY_MODE": "true" if args.dummy else "false",
            "DUMMY_EXEC_TIME_MS": str(args.dummy_execution_time_ms),
            # Independent subprocesses otherwise tend to oversubscribe CPU cores.
            "OMP_NUM_THREADS": environment.get("OMP_NUM_THREADS", "1"),
            "MKL_NUM_THREADS": environment.get("MKL_NUM_THREADS", "1"),
        }
    )

    overrides = (
        "--cluster_config_prefill_replica_config_network_device", args.network_device,
        "--cluster_config_decode_replica_config_network_device", args.network_device,
        "--cluster_config_prefill_replica_scheduler_config_batch_size_cap", str(batch_cap),
        "--cluster_config_decode_replica_scheduler_config_batch_size_cap", str(batch_cap),
        "--cluster_config_prefill_replica_scheduler_config_max_tokens_in_batch", str(prefill_budget),
        "--cluster_config_decode_replica_scheduler_config_max_tokens_in_batch", str(batch_cap),
        "--vllm_v1_scheduler_config_num_blocks_mode", "memory_planner",
        "--vllm_v1_scheduler_config_gpu_memory_utilization", str(args.gpu_memory_utilization),
        "--metrics_config_keep_individual_batch_metrics",
        "--cc_backend_config_type", "astra_sim_analytical",
        "--astra_sim_analytical_cc_backend_config_placement_order", "TP,DP,EP,CP",
        "--astra_sim_analytical_cc_backend_config_intra_server_topology", "FullyConnected",
        "--astra_sim_analytical_cc_backend_config_inter_server_topology", "FullyConnected",
        "--astra_sim_analytical_cc_backend_config_intra_server_bandwidth_gbps", str(args.nvlink_gbps),
        "--astra_sim_analytical_cc_backend_config_inter_server_bandwidth_gbps", str(args.ib_gbps),
    )
    command = ("bash", str(args.example_script.resolve()), "--", *overrides)
    return Trial(batch_cap=batch_cap, run_id=run_id, command=command, environment=environment, log_path=log_path)


def run_trial(trial: Trial) -> tuple[int, str, Path]:
    trial.log_path.parent.mkdir(parents=True, exist_ok=True)
    with trial.log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"Command: {shlex.join(trial.command)}\n\n")
        completed = subprocess.run(
            trial.command,
            cwd=REPO_ROOT,
            env=trial.environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return completed.returncode, trial.run_id, trial.log_path


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validate_args(args)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    output_dir = args.output_dir.resolve()
    trials = [build_trial(args, batch_cap, output_dir) for batch_cap in args.batch_caps]

    if not args.dummy and args.max_workers > 1:
        print("warning: profile-based runs with multiple workers can duplicate predictor training and contend for cache files. Run once with --max-workers 1 first.")
    print(f"PDD batch-cap sweep: {len(trials)} trials, max_workers={args.max_workers}")
    print(f"Metrics root: {output_dir}")
    print("Note: public PDD runs sequentially (--no-enable_parallel_clusters).")

    if args.dry_run:
        for trial in trials:
            print(f"\n[{trial.run_id}]\n{shlex.join(trial.command)}")
        return 0

    failures: list[tuple[str, Path, int]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_trial = {executor.submit(run_trial, trial): trial for trial in trials}
        for future in concurrent.futures.as_completed(future_to_trial):
            return_code, run_id, log_path = future.result()
            if return_code == 0:
                print(f"completed: {run_id} ({log_path})")
            else:
                failures.append((run_id, log_path, return_code))
                print(f"failed: {run_id}, exit={return_code}; inspect {log_path}", file=sys.stderr)

    if failures:
        print(f"{len(failures)} trial(s) failed.", file=sys.stderr)
        return 1
    print("All trials completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
