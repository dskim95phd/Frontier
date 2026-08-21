#!/usr/bin/env python3
"""Run a short Kimi K3 CPU-4TB arrival-rate screening sweep.

The hardware and scheduling template is the Kimi K2 Vera/Rubin CPU-offload
experiment, expanded to P16/D32. Kimi K3 is loaded from its model
asset with FP8 KV, BF16 KDA snapshots, TP1/DP16/EP16 PREFILL, and two
TP4/DCP4/DP4/EP16 DECODE replicas.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from decimal import Decimal, InvalidOperation
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
K2_EXPERIMENT = HERE.parent / "kimi_k2_cpu_dram"
DEFAULT_TEMPLATE = (
    K2_EXPERIMENT
    / "configs"
    / "tracelab_vera_rubin_p8_d16_cache_aware_cpu4tb.json"
)
DEFAULT_BINARY = REPO_ROOT / "cpp" / "build" / "Release" / "frontier_sim.exe"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "outputs" / "kimi_k3_rubin_p16_d32_cpu4tb_rate_screen"
)
DEFAULT_RATES = "0.3,0.4,0.5,0.6,0.7"
DEFAULT_HORIZON_SECONDS = 3_600.0
DEFAULT_SEED = 20260803


def parse_rates(text: str) -> list[float]:
    rates: list[float] = []
    seen: set[Decimal] = set()
    for raw in text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            value = Decimal(raw)
        except InvalidOperation as error:
            raise argparse.ArgumentTypeError(f"invalid rate: {raw}") from error
        if not value.is_finite() or value <= 0:
            raise argparse.ArgumentTypeError(f"rate must be positive: {raw}")
        if value in seen:
            raise argparse.ArgumentTypeError(f"duplicate rate: {raw}")
        seen.add(value)
        rates.append(float(value))
    if not rates:
        raise argparse.ArgumentTypeError("at least one rate is required")
    return rates


def rate_label(rate: float) -> str:
    return "r" + format(Decimal(str(rate)).normalize(), "f").replace(".", "p")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_k3_config(template: dict[str, Any], rate: float) -> dict[str, Any]:
    config = copy.deepcopy(template)
    config["run_id"] = (
        f"kimi-k3-rubin-p16-d32-cpu4tb-{rate_label(rate)}"
    )
    cpu = config["cpu_kv_cache"]
    if not isinstance(cpu, dict):
        raise ValueError("template cpu_kv_cache must be an object")
    cpu["enabled"] = True
    # One 4 TB CPU is shared by two GPUs. Static slicing gives every PREFILL
    # TP1 cache target its local GPU's 2 TB slice (32 TB over 16 targets).
    cpu["capacity_bytes"] = 4_000_000_000_000
    cpu["capacity_bytes_per_gpu"] = 2_000_000_000_000
    config["model"] = "moonshotai/Kimi-K3"

    clusters = config.get("clusters")
    if not isinstance(clusters, dict):
        raise ValueError("template clusters must be an object")
    for role in ("prefill", "decode"):
        cluster = clusters.get(role)
        if not isinstance(cluster, dict):
            raise ValueError(f"template cluster {role} must be an object")
        cluster["profile"] = (
            "rubin-vera-16gpu" if role == "prefill" else "rubin-vera-32gpu"
        )
        for key in (
            "total_expert_num",
            "router_topk",
            "first_k_dense_replace",
            "num_shared_experts",
        ):
            cluster.pop(key, None)
        execution = cluster.get("execution_model")
        if not isinstance(execution, dict):
            raise ValueError(f"template {role}.execution_model must be an object")
        # Replace any K2 template policy with K3's named native mixed-precision
        # execution contract.
        execution.pop("precision", None)
        execution.pop("operator_precisions", None)
        execution["precision_profile"] = "kimi-k3-native"
        scheduler = cluster.get("scheduler")
        if not isinstance(scheduler, dict):
            raise ValueError(f"template {role}.scheduler must be an object")
        if role == "prefill":
            scheduler["max_tokens_in_batch"] = 16_384
            scheduler["enable_chunked_prefill"] = True
            scheduler["long_prefill_token_threshold"] = 512
        # K2's fixed block counts encode its all-MLA FP8 byte size. The named
        # cluster profile supplies the same 288 GB hardware and lets the
        # K3-aware planner derive new blocks.
        scheduler.pop("num_blocks", None)
        parallelism = cluster.get("parallelism")
        if not isinstance(parallelism, dict):
            raise ValueError(f"template {role}.parallelism must be an object")
        parallelism.pop("pipeline_stage_layer_counts", None)
        if role == "prefill":
            parallelism.update(
                {
                    "num_replicas": 1,
                    "tensor_parallel_size": 1,
                    "decode_context_parallel_size": 1,
                    "pipeline_parallel_size": 1,
                    "data_parallel_size": 16,
                    "moe_tensor_parallel_size": 1,
                    "moe_expert_parallel_size": 16,
                }
            )
        else:
            parallelism.update(
                {
                    "num_replicas": 2,
                    "tensor_parallel_size": 4,
                    "decode_context_parallel_size": 4,
                    "pipeline_parallel_size": 1,
                    "data_parallel_size": 4,
                    "moe_tensor_parallel_size": 1,
                    "moe_expert_parallel_size": 16,
                }
            )

    transfer = config.get("kv_cache_transfer")
    if not isinstance(transfer, dict):
        raise ValueError("template kv_cache_transfer must be an object")
    transfer["kv_cache_dtype_size_bytes"] = 1
    return config


def workload_paths(root: Path, rate: float) -> tuple[Path, Path, Path]:
    directory = root / rate_label(rate)
    stem = f"seed_{DEFAULT_SEED}"
    return (
        directory / f"{stem}.csv",
        directory / f"{stem}_manifest.csv",
        directory / f"{stem}_metadata.json",
    )


def ensure_workload(
    *,
    root: Path,
    rate: float,
    session_count: int,
    regenerate: bool,
    dry_run: bool,
) -> Path:
    workload, manifest, metadata = workload_paths(root, rate)
    if not regenerate and all(path.is_file() for path in (workload, manifest, metadata)):
        return workload
    six_turn_sessions = (2 * session_count) // 3
    command = [
        sys.executable,
        str(K2_EXPERIMENT / "generate_workload.py"),
        "--output",
        str(workload),
        "--manifest-output",
        str(manifest),
        "--metadata-output",
        str(metadata),
        "--seed",
        str(DEFAULT_SEED),
        "--sessions",
        str(session_count),
        "--six-turn-sessions",
        str(six_turn_sessions),
        "--arrival-rate",
        format(rate, ".12g"),
        "--arrival-process",
        "stratified",
        "--think-profile",
        "long_mixed",
    ]
    if dry_run:
        print("[workload] " + subprocess.list2cmdline(command), flush=True)
        return workload
    workload.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True, stdout=subprocess.DEVNULL)
    return workload


def complete(case_dir: Path) -> bool:
    run_path = case_dir / "run.json"
    if not run_path.is_file():
        return False
    try:
        run = json.loads(run_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        run.get("status") == "completed"
        and (case_dir / "summary.json").is_file()
        and (case_dir / "requests.csv").is_file()
        and (case_dir / "workload.normalized.csv").is_file()
    )


def run_case(
    *,
    binary: Path,
    output_root: Path,
    workload: Path,
    config_path: Path,
    rate: float,
    horizon_s: float,
    resume: bool,
    dry_run: bool,
) -> tuple[float, bool, float]:
    case_dir = output_root / rate_label(rate) / "cpu4000gb" / "r1"
    command = [
        str(binary),
        "--config",
        str(config_path),
        "--workload",
        str(workload),
        "--output-dir",
        str(case_dir),
        "--output-mode",
        "requests",
        "--runtime-validation",
        "false",
        "--gpu-kv-occupancy",
        "false",
        "--simulation-end-time-s",
        format(horizon_s, ".12g"),
    ]
    if resume and complete(case_dir):
        print(f"[{rate_label(rate)}] already complete; skipped", flush=True)
        return rate, True, 0.0
    if dry_run:
        print(f"[{rate_label(rate)}] " + subprocess.list2cmdline(command), flush=True)
        return rate, True, 0.0

    case_dir.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "model": "moonshotai/Kimi-K3",
        "session_arrival_rate_per_second": rate,
        "simulation_end_time_s": horizon_s,
        "physical_cpu_dram_capacity_bytes": 4_000_000_000_000,
        "cpu_cache_target_capacity_bytes": 2_000_000_000_000,
        "workload_csv": str(workload.resolve()),
        "config_input": str(config_path.resolve()),
        "command": command,
        "started_at_unix_s": time.time(),
    }
    write_json(case_dir / "run.json", record)
    started = time.perf_counter()
    with (case_dir / "simulator.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    wall_seconds = time.perf_counter() - started
    record.update(
        {
            "status": "completed" if result.returncode == 0 else "failed",
            "exit_code": result.returncode,
            "wall_clock_seconds": wall_seconds,
            "finished_at_unix_s": time.time(),
        }
    )
    write_json(case_dir / "run.json", record)
    print(
        f"[{rate_label(rate)}] {record['status']} in {wall_seconds:.1f}s",
        flush=True,
    )
    return rate, result.returncode == 0, wall_seconds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--config-template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--rates", type=parse_rates, default=parse_rates(DEFAULT_RATES))
    parser.add_argument(
        "--simulation-end-time-s", type=float, default=DEFAULT_HORIZON_SECONDS
    )
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--regenerate-workloads", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (
        not math.isfinite(args.simulation_end_time_s)
        or args.simulation_end_time_s <= 0
        or args.jobs <= 0
    ):
        raise SystemExit("simulation horizon and jobs must be positive")
    if not args.config_template.is_file():
        raise SystemExit(f"config template not found: {args.config_template}")
    if not args.dry_run and not args.binary.is_file():
        raise SystemExit(f"simulator binary not found: {args.binary}")

    output_root = args.output_root.resolve()
    workload_root = output_root / "workloads"
    # Every rate sees the same ordered session population. Lower rates simply
    # leave a longer suffix outside the fixed screening horizon.
    session_count = math.ceil(max(args.rates) * args.simulation_end_time_s) + 1
    template = json.loads(args.config_template.read_text(encoding="utf-8"))
    prepared: list[tuple[float, Path, Path]] = []
    for rate in args.rates:
        workload = ensure_workload(
            root=workload_root,
            rate=rate,
            session_count=session_count,
            regenerate=args.regenerate_workloads,
            dry_run=args.dry_run,
        )
        config_path = output_root / "configs" / f"{rate_label(rate)}.json"
        if not args.dry_run:
            write_json(config_path, build_k3_config(template, rate))
        prepared.append((rate, workload, config_path))

    if not args.dry_run:
        write_json(
            output_root / "sweep_plan.json",
            {
                "schema_version": 1,
                "study": "kimi_k3_rubin_p16_d32_cpu4tb_rate_screen",
                "model": "moonshotai/Kimi-K3",
                "hardware_template": str(args.config_template.resolve()),
                "rates": args.rates,
                "simulation_end_time_s": args.simulation_end_time_s,
                "session_count_per_workload": session_count,
                "workload_profile": "long_mixed; stratified root arrivals",
                "seed": DEFAULT_SEED,
                "jobs": args.jobs,
            },
        )

    failures: list[float] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(
                run_case,
                binary=args.binary.resolve(),
                output_root=output_root,
                workload=workload,
                config_path=config_path,
                rate=rate,
                horizon_s=args.simulation_end_time_s,
                resume=args.resume,
                dry_run=args.dry_run,
            ): rate
            for rate, workload, config_path in prepared
        }
        for future in as_completed(futures):
            rate, ok, _ = future.result()
            if not ok:
                failures.append(rate)
    if failures:
        raise SystemExit("failed rates: " + ",".join(format(rate, "g") for rate in failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
