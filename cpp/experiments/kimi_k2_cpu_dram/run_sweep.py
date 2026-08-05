#!/usr/bin/env python3
"""Run the Kimi K2 / GB300 CPU-DRAM capacity study matrix.

The runner is deliberately a thin process wrapper around ``frontier_sim``.
It writes one immutable config and one ``run.json`` record per
``(seed, capacity)`` pair, uses the same workload CSV for every capacity of a
seed, and defaults to the compact ``requests`` output mode.  No full event
trace is required for pilot, coarse, or fine sweeps.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Iterable, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "kimi_k2_cpu_dram"
DEFAULT_WORKLOAD_ROOT = HERE / "workloads"
BASE_CONFIG_PATH = HERE / "configs" / "base_pdd.json"

BLOCK_SIZE = 16
ONE_COPY_BYTES_PER_TOKEN = 39_040
ATTENTION_TP = 4
PREFILL_CONTEXT_PARALLEL_SIZE = 1
DECODE_CONTEXT_PARALLEL_SIZE = 4
PREFILL_RANK_LOCAL_BYTES_PER_TOKEN = ONE_COPY_BYTES_PER_TOKEN
DECODE_RANK_LOCAL_BYTES_PER_TOKEN = math.ceil(
    ONE_COPY_BYTES_PER_TOKEN / DECODE_CONTEXT_PARALLEL_SIZE
)
PREFILL_TARGET_PHYSICAL_BYTES_PER_TOKEN = (
    ONE_COPY_BYTES_PER_TOKEN
    * ATTENTION_TP
    // PREFILL_CONTEXT_PARALLEL_SIZE
)
DECODE_TARGET_PHYSICAL_BYTES_PER_TOKEN = (
    ONE_COPY_BYTES_PER_TOKEN
    * ATTENTION_TP
    // DECODE_CONTEXT_PARALLEL_SIZE
)
ONE_COPY_BYTES_PER_BLOCK = ONE_COPY_BYTES_PER_TOKEN * BLOCK_SIZE
PREFILL_RANK_LOCAL_BYTES_PER_BLOCK = (
    PREFILL_RANK_LOCAL_BYTES_PER_TOKEN * BLOCK_SIZE
)
DECODE_RANK_LOCAL_BYTES_PER_BLOCK = (
    DECODE_RANK_LOCAL_BYTES_PER_TOKEN * BLOCK_SIZE
)
PREFILL_TARGET_PHYSICAL_BYTES_PER_BLOCK = (
    PREFILL_TARGET_PHYSICAL_BYTES_PER_TOKEN * BLOCK_SIZE
)
DECODE_TARGET_PHYSICAL_BYTES_PER_BLOCK = (
    DECODE_TARGET_PHYSICAL_BYTES_PER_TOKEN * BLOCK_SIZE
)
GPU_KV_BUDGET_BYTES = 190_000_000_000
PREFILL_GPU_KV_BLOCKS = (
    GPU_KV_BUDGET_BYTES // PREFILL_RANK_LOCAL_BYTES_PER_BLOCK
)
DECODE_GPU_KV_BLOCKS = (
    GPU_KV_BUDGET_BYTES // DECODE_RANK_LOCAL_BYTES_PER_BLOCK
)
PREFILL_GPU_KV_TOKENS = PREFILL_GPU_KV_BLOCKS * BLOCK_SIZE
DECODE_GPU_KV_TOKENS = DECODE_GPU_KV_BLOCKS * BLOCK_SIZE
PREFILL_DP = 4
CPU_SLICES_PER_TARGET = ATTENTION_TP  # TP4 * PP1
DECIMAL_GB = 1_000_000_000

PILOT_CAPACITIES = ("off", "cpu128", "oracle_unbounded_cpu")
COARSE_CAPACITIES = ("off", "cpu32", "cpu64", "cpu128", "cpu256", "cpu500", "oracle_unbounded_cpu")
DEFAULT_FINE_CAPACITIES = ("cpu256", "cpu296", "cpu336", "cpu376", "cpu416", "cpu456", "cpu500")
DEFAULT_SEEDS = (20260803, 20260804, 20260805, 20260806, 20260807)


@dataclass(frozen=True)
class CapacityCase:
    label: str
    grace_cpu_dram_gb: float | None
    gpu_slice_dram_gb: float | None
    rho: float | None
    enabled: bool
    oracle: bool = False
    target_capacity_bytes: int = 0

    @property
    def target_capacity_gb(self) -> float:
        return self.target_capacity_bytes / DECIMAL_GB


def _positive_float(value: float, name: str) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError(f"{name} must be finite and positive")
    return value


def parse_seed_list(value: str) -> list[int]:
    try:
        result = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from error
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("seeds must be non-empty and unique")
    return result


def parse_capacity_list(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("capacities must be non-empty and unique")
    for item in result:
        if item in {"off", "oracle_unbounded_cpu"}:
            continue
        normalized = item[3:] if item.lower().startswith("cpu") else item
        try:
            numeric = float(normalized)
        except ValueError as error:
            raise argparse.ArgumentTypeError(f"invalid capacity label: {item}") from error
        if not math.isfinite(numeric) or numeric <= 0.0:
            raise argparse.ArgumentTypeError(f"capacity must be positive: {item}")
    return result


def _capacity_gb_from_label(label: str) -> float | None:
    if label in {"off", "oracle_unbounded_cpu"}:
        return None
    text = label[3:] if label.lower().startswith("cpu") else label
    return float(text)


def make_capacity_case(label: str, *, oracle_target_capacity_bytes: int = 10_000 * DECIMAL_GB) -> CapacityCase:
    """Materialize a user-facing Grace-CPU label into physical target bytes."""

    if label == "off":
        return CapacityCase(label, 0.0, 0.0, 0.0, False, False, 0)
    if label == "oracle_unbounded_cpu":
        if oracle_target_capacity_bytes <= 0:
            raise ValueError("oracle capacity must be positive")
        # Keep the static-slice contract: config capacity is per physical
        # slice and the C++ resolver multiplies it by TP*PP (=4).
        per_slice = math.ceil(oracle_target_capacity_bytes / CPU_SLICES_PER_TARGET)
        return CapacityCase(label, None, None, None, True, True, per_slice * CPU_SLICES_PER_TARGET)
    grace_gb = _capacity_gb_from_label(label)
    assert grace_gb is not None
    slice_gb = grace_gb / 2.0
    target_bytes = int(round(slice_gb * DECIMAL_GB)) * CPU_SLICES_PER_TARGET
    rho = slice_gb * DECIMAL_GB / GPU_KV_BUDGET_BYTES
    return CapacityCase(label, grace_gb, slice_gb, rho, True, False, target_bytes)


def estimate_oracle_capacity_bytes(manifest_csv: Path | None) -> int:
    """Estimate a direct unbounded target from final session snapshots."""

    if manifest_csv is None or not manifest_csv.is_file():
        return 10_000 * DECIMAL_GB
    import csv

    snapshots: dict[int, int] = {}
    with manifest_csv.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            session_id = int(row["session_id"])
            snapshots[session_id] = max(snapshots.get(session_id, 0), int(row["final_context_tokens"]))
    if not snapshots:
        return 10_000 * DECIMAL_GB
    target = sum(
        math.ceil(tokens / BLOCK_SIZE)
        * PREFILL_TARGET_PHYSICAL_BYTES_PER_BLOCK
        for tokens in snapshots.values()
    )
    # Add one block per session to make the no-eviction oracle insensitive to
    # block-frontier timing.
    target += len(snapshots) * PREFILL_TARGET_PHYSICAL_BYTES_PER_BLOCK
    return max(target, 10_000 * DECIMAL_GB)


def _load_generator_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("kimi_k2_cpu_dram_generator", HERE / "generate_workload.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load workload generator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def ensure_workload(seed: int, workload_root: Path) -> tuple[Path, Path, Path]:
    workload_root.mkdir(parents=True, exist_ok=True)
    workload = workload_root / f"seed_{seed}.csv"
    manifest = workload_root / f"seed_{seed}_manifest.csv"
    metadata = workload_root / f"seed_{seed}_metadata.json"
    if not workload.is_file() or not manifest.is_file():
        generator = _load_generator_module()
        rows = generator.generate_rows(seed=seed)
        generator.write_generation(
            workload,
            manifest,
            metadata,
            rows,
            seed=seed,
            parameters={
                "sessions": generator.SESSION_COUNT,
                "six_turn_sessions": generator.SIX_TURN_SESSIONS,
                "seven_turn_sessions": generator.SEVEN_TURN_SESSIONS,
                "context_mean": generator.CONTEXT_MEAN_TOKENS,
                "context_min": generator.CONTEXT_MIN_TOKENS,
                "context_max": generator.CONTEXT_MAX_TOKENS,
                "context_cv": generator.CONTEXT_CV,
                "dirichlet_alpha": generator.DIRICHLET_ALPHA,
                "arrival_rate": generator.ARRIVAL_RATE_PER_SECOND,
                "think_median": generator.THINK_MEDIAN_SECONDS,
                "think_log_sigma": generator.THINK_LOG_SIGMA,
                "think_min": generator.THINK_MIN_SECONDS,
                "think_max": generator.THINK_MAX_SECONDS,
                "block_size": generator.BLOCK_SIZE,
                "input_output_ratio": generator.INPUT_OUTPUT_RATIO,
            },
        )
    if not metadata.is_file():
        metadata.write_text(json.dumps({"seed": seed, "workload_csv": str(workload), "manifest_csv": str(manifest)}, indent=2) + "\n", encoding="utf-8")
    return workload, manifest, metadata


def _cluster_config(*, decode: bool) -> dict[str, object]:
    return {
        "parallelism": {
            "num_replicas": 1,
            "tensor_parallel_size": 4,
            "decode_context_parallel_size": (
                DECODE_CONTEXT_PARALLEL_SIZE
                if decode
                else PREFILL_CONTEXT_PARALLEL_SIZE
            ),
            "pipeline_parallel_size": 1,
            "data_parallel_size": 4,
            "moe_tensor_parallel_size": 1,
            "moe_expert_parallel_size": 16,
        },
        "scheduler": {
            "type": "vllm_v1",
            "scheduling_policy": "fcfs",
            "batch_size_cap": 64 if not decode else 32,
            "max_tokens_in_batch": 131_072 if not decode else 8_192,
            "enable_preemption": True,
            "enable_chunked_prefill": not decode,
            "long_prefill_token_threshold": 0,
            "block_size": BLOCK_SIZE,
            "num_blocks": (
                DECODE_GPU_KV_BLOCKS if decode else PREFILL_GPU_KV_BLOCKS
            ),
            "watermark_blocks_fraction": 0.0,
            "num_preallocate_tokens": 0,
        },
        "execution_model": {
            "type": "analytical",
            "device": "gb300",
            "precision": "fp8",
            "network_bandwidth_gbps": 38_400.0,
            "network_latency_us": 1.0,
            "intra_node_bandwidth_gbps": 14_400.0,
            "moe_layer_event_mode": "first_layer_scaled",
            "operator_precisions": {
                "attention": "fp8",
                "dense": "fp8",
                "moe_expert": "fp8",
                "moe_router": "fp8",
                "kv_cache": "fp8",
                "communication": "fp8",
                "attention_weight": "fp8",
                "attention_activation": "fp8",
                "dense_weight": "fp8",
                "dense_activation": "fp8",
                "moe_expert_weight": "fp8",
                "moe_expert_activation": "fp8",
                "moe_router_weight": "fp8",
                "moe_router_activation": "fp8",
                "lm_head": "fp8",
                "lm_head_weight": "fp8",
                "lm_head_activation": "fp8",
            },
        },
        "model_name": "moonshotai/Kimi-K2-Instruct",
        "total_expert_num": 384,
        "router_topk": 8,
        "moe_routing": {
            "mode": "simulation",
            "distribution": "balanced",
            "seed": 42,
        },
    }


def build_config(
    base: dict[str, object],
    *,
    run_id: str,
    capacity: CapacityCase,
    oracle_target_capacity_bytes: int,
) -> dict[str, object]:
    """Return one validated-study config without mutating the base object."""

    config = copy.deepcopy(base)
    config["run_id"] = run_id
    config["simulation_mode"] = "online"
    config["system_architecture"] = "pd-disaggregation"
    config["enable_parallel_clusters"] = False
    config["prefix_cache"] = {"enabled": True, "key_mode": "session"}
    config["cluster_scheduler"] = {
        "type": "sticky_round_robin",
        "prefill_type": "cache_aware",
        "decode_type": "vllm_queue_aware",
        "cache_threshold": 0.5,
        "balance_abs_threshold": 32,
        "balance_rel_threshold": 1.1,
    }
    config["clusters"] = {"prefill": _cluster_config(decode=False), "decode": _cluster_config(decode=True)}
    if capacity.oracle:
        target_bytes = oracle_target_capacity_bytes
    else:
        target_bytes = capacity.target_capacity_bytes
    if not capacity.enabled:
        target_bytes = max(target_bytes, DECIMAL_GB)
    per_slice_bytes = max(DECIMAL_GB, math.ceil(target_bytes / CPU_SLICES_PER_TARGET))
    config["cpu_kv_cache"] = {
        "enabled": capacity.enabled,
        "capacity_bytes": target_bytes if not capacity.oracle else target_bytes,
        "static_slice_per_gpu": True,
        "capacity_bytes_per_gpu": per_slice_bytes,
        "dram_bandwidth_gbps_per_gpu": 1_555.56,
        "c2c_bandwidth_gbps_per_gpu": 3_600.0,
        "write_bandwidth_gbps": 64.0,
        "write_latency_ms": 0.01,
        "read_bandwidth_gbps": 64.0,
        "read_latency_ms": 0.01,
        "eviction_policy": "session_lru_suffix",
        "capacity_pressure_policy": "prefix_fit",
        "transfer_concurrency": "full_duplex_serialized",
    }
    config["kv_cache_transfer"] = {
        "type": "analytical",
        "network_bandwidth_gbps": 38_400.0,
        "network_latency_ms": 0.02,
        "kv_cache_dtype_size_bytes": 1,
        "enable_compression": False,
    }
    return config


def _find_default_binary() -> Path | None:
    configured = os.environ.get("FRONTIER_CPP_BINARY")
    candidates = [Path(configured)] if configured else []
    candidates.extend(
        [
            REPO_ROOT / "cpp" / "build" / "Debug" / "frontier_sim.exe",
            REPO_ROOT / "cpp" / "build" / "Release" / "frontier_sim.exe",
            REPO_ROOT / "cpp" / "build" / "frontier_sim.exe",
            REPO_ROOT / "cpp" / "build" / "frontier_sim",
            REPO_ROOT / ".build" / "frontier_sim.exe",
            REPO_ROOT / ".build" / "frontier_sim",
        ]
    )
    return next((path.resolve() for path in candidates if path and path.is_file()), None)


def _phase_capacities(phase: str, explicit: Sequence[str] | None, selection_path: Path | None) -> list[str]:
    if explicit:
        return list(explicit)
    if phase == "pilot":
        return list(PILOT_CAPACITIES)
    if phase == "coarse":
        return list(COARSE_CAPACITIES)
    if phase == "fine":
        if selection_path and selection_path.is_file():
            try:
                selection = json.loads(selection_path.read_text(encoding="utf-8"))
                values = selection.get("fine_capacity_labels") or selection.get("fine_capacities")
                if values:
                    return parse_capacity_list(",".join(str(value) for value in values))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass
        return list(DEFAULT_FINE_CAPACITIES)
    raise ValueError("custom phase requires --capacities")


def _execution_order(phase: str, capacities: Sequence[str], seed: int) -> list[str]:
    """Return a reproducible per-seed order without changing the matrix."""

    order = list(capacities)
    if phase in {"coarse", "fine"}:
        random.Random(seed).shuffle(order)
    return order


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _run_one(
    *,
    binary: Path,
    config_path: Path,
    workload_path: Path,
    output_dir: Path,
    output_mode: str,
    timeout_seconds: float,
    cwd: Path,
) -> tuple[int, str, str, float]:
    command = [
        str(binary),
        "--config",
        str(config_path),
        "--workload",
        str(workload_path),
        "--output-dir",
        str(output_dir),
        "--output-mode",
        output_mode,
        "--runtime-validation",
        "false",
    ]
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    return completed.returncode, completed.stdout, completed.stderr, time.perf_counter() - started


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pilot", "coarse", "fine", "custom"), default="coarse")
    parser.add_argument("--capacities", type=parse_capacity_list)
    parser.add_argument("--fine-selection", type=Path)
    parser.add_argument("--seeds", type=parse_seed_list)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--base-config", type=Path, default=BASE_CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workload-dir", type=Path, default=DEFAULT_WORKLOAD_ROOT)
    parser.add_argument("--output-mode", choices=("summary", "requests", "full"), default="requests")
    parser.add_argument("--timeout-seconds", type=float, default=86_400.0)
    parser.add_argument("--oracle-capacity-gb", type=float, default=10_000.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Write configs and commands without launching frontier_sim.")
    parser.add_argument("--no-generate", action="store_true", help="Require existing seed CSV/manifest files.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.timeout_seconds <= 0.0 or not math.isfinite(args.timeout_seconds):
        raise SystemExit("--timeout-seconds must be finite and positive")
    if args.oracle_capacity_gb <= 0.0 or not math.isfinite(args.oracle_capacity_gb):
        raise SystemExit("--oracle-capacity-gb must be finite and positive")
    if args.phase == "custom" and not args.capacities:
        raise SystemExit("--phase custom requires --capacities")
    seeds = args.seeds or ((20260803,) if args.phase == "pilot" else DEFAULT_SEEDS)
    selection_path = args.fine_selection or (args.output_root / "fine" / "selection.json")
    capacities = _phase_capacities(args.phase, args.capacities, selection_path)
    if args.phase == "pilot" and args.seeds is None:
        seeds = [20260803]
    base_path = args.base_config.resolve()
    if not base_path.is_file():
        raise SystemExit(f"base config not found: {base_path}")
    try:
        base = json.loads(base_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"unable to read base config: {error}") from error
    binary = (args.binary or _find_default_binary())
    if not args.dry_run and (binary is None or not binary.is_file()):
        raise SystemExit("frontier_sim binary not found; pass --binary or set FRONTIER_CPP_BINARY (use --dry-run to plan without it)")
    if binary is not None:
        binary = binary.resolve()

    phase_root = args.output_root.resolve() / args.phase
    phase_root.mkdir(parents=True, exist_ok=True)
    root_manifest: dict[str, object] = {
        "study": "kimi_k2_cpu_dram",
        "phase": args.phase,
        "seeds": list(seeds),
        "capacity_labels": capacities,
        "execution_order_by_seed": {
            str(seed): _execution_order(args.phase, capacities, seed)
            for seed in seeds
        },
        "paired_workload": True,
        "output_mode": args.output_mode,
        "requires_full_trace": False,
        "topology": {
            "system_architecture": "pd-disaggregation",
            "enable_parallel_clusters": False,
            "prefill_gpus": 16,
            "decode_gpus": 16,
            "attention_tensor_parallel": 4,
            "prefill_context_parallel": PREFILL_CONTEXT_PARALLEL_SIZE,
            "decode_context_parallel": DECODE_CONTEXT_PARALLEL_SIZE,
            "pipeline_parallel": 1,
            "data_parallel": 4,
            "moe_tensor_parallel": 1,
            "expert_parallel": 16,
            "total_experts": 384,
            "router_topk": 8,
            "model": "moonshotai/Kimi-K2-Instruct",
            "device": "gb300",
        },
        "kv_contract": {
            "one_copy_bytes_per_token": ONE_COPY_BYTES_PER_TOKEN,
            "prefill_rank_local_bytes_per_token": (
                PREFILL_RANK_LOCAL_BYTES_PER_TOKEN
            ),
            "decode_rank_local_bytes_per_token": (
                DECODE_RANK_LOCAL_BYTES_PER_TOKEN
            ),
            "prefill_target_physical_bytes_per_token": (
                PREFILL_TARGET_PHYSICAL_BYTES_PER_TOKEN
            ),
            "decode_target_physical_bytes_per_token": (
                DECODE_TARGET_PHYSICAL_BYTES_PER_TOKEN
            ),
            "one_copy_bytes_per_block": ONE_COPY_BYTES_PER_BLOCK,
            "prefill_rank_local_bytes_per_block": (
                PREFILL_RANK_LOCAL_BYTES_PER_BLOCK
            ),
            "decode_rank_local_bytes_per_block": (
                DECODE_RANK_LOCAL_BYTES_PER_BLOCK
            ),
            "prefill_target_physical_bytes_per_block": (
                PREFILL_TARGET_PHYSICAL_BYTES_PER_BLOCK
            ),
            "decode_target_physical_bytes_per_block": (
                DECODE_TARGET_PHYSICAL_BYTES_PER_BLOCK
            ),
            "gpu_kv_budget_bytes": GPU_KV_BUDGET_BYTES,
            "prefill_gpu_kv_blocks_per_dp_target": PREFILL_GPU_KV_BLOCKS,
            "decode_gpu_kv_blocks_per_dp_target": DECODE_GPU_KV_BLOCKS,
            "prefill_gpu_kv_tokens_per_dp_target": PREFILL_GPU_KV_TOKENS,
            "decode_gpu_kv_tokens_per_dp_target": DECODE_GPU_KV_TOKENS,
        },
    }
    _write_json(phase_root / "phase_manifest.json", root_manifest)

    total = len(seeds) * len(capacities)
    completed = 0
    for seed in seeds:
        workload, manifest, metadata = ensure_workload(seed, args.workload_dir.resolve()) if not args.no_generate else (
            args.workload_dir.resolve() / f"seed_{seed}.csv",
            args.workload_dir.resolve() / f"seed_{seed}_manifest.csv",
            args.workload_dir.resolve() / f"seed_{seed}_metadata.json",
        )
        if not workload.is_file() or not manifest.is_file():
            raise SystemExit(f"missing workload/manifest for seed {seed}: {workload}, {manifest}")
        oracle_capacity_bytes = max(
            int(round(args.oracle_capacity_gb * DECIMAL_GB)),
            estimate_oracle_capacity_bytes(manifest),
        )
        for label in _execution_order(args.phase, capacities, seed):
            completed += 1
            capacity = make_capacity_case(label, oracle_target_capacity_bytes=oracle_capacity_bytes)
            case_dir = phase_root / f"seed_{seed}" / label
            case_dir.mkdir(parents=True, exist_ok=True)
            run_id = f"kimi-k2-cpu-dram-{args.phase}-seed{seed}-{label}"
            config = build_config(
                base,
                run_id=run_id,
                capacity=capacity,
                oracle_target_capacity_bytes=oracle_capacity_bytes,
            )
            config_path = case_dir / "config.input.json"
            _write_json(config_path, config)
            command = (
                [str(binary), "--config", str(config_path), "--workload", str(workload), "--output-dir", str(case_dir), "--output-mode", args.output_mode, "--runtime-validation", "false"]
                if binary is not None
                else ["<frontier_sim>", "--config", str(config_path), "--workload", str(workload), "--output-dir", str(case_dir), "--output-mode", args.output_mode, "--runtime-validation", "false"]
            )
            record_path = case_dir / "run.json"
            if args.resume and record_path.is_file() and (case_dir / "summary.json").is_file():
                try:
                    old = json.loads(record_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    old = {}
                if old.get("status") == "ok":
                    print(f"[{completed}/{total}] resume skip seed={seed} capacity={label}", flush=True)
                    continue
            record: dict[str, object] = {
                "study": "kimi_k2_cpu_dram",
                "phase": args.phase,
                "seed": seed,
                "capacity": asdict(capacity),
                "workload_csv": str(workload),
                "workload_manifest_csv": str(manifest),
                "workload_metadata_json": str(metadata),
                "config": str(config_path),
                "command": command,
                "output_mode": args.output_mode,
                "runtime_validation": False,
                "paired_seed": seed,
                "status": "planned" if args.dry_run else "running",
            }
            if args.dry_run:
                _write_json(record_path, record)
                print(f"[{completed}/{total}] plan seed={seed} capacity={label}: {' '.join(command)}", flush=True)
                continue
            assert binary is not None
            try:
                returncode, stdout, stderr, wall_seconds = _run_one(
                    binary=binary,
                    config_path=config_path,
                    workload_path=workload,
                    output_dir=case_dir,
                    output_mode=args.output_mode,
                    timeout_seconds=args.timeout_seconds,
                    cwd=REPO_ROOT,
                )
                record.update(
                    {
                        "status": "ok" if returncode == 0 else "failed",
                        "returncode": returncode,
                        "wall_clock_seconds": wall_seconds,
                        "stdout": stdout[-4_000:],
                        "stderr": stderr[-8_000:],
                        "summary_json": str(case_dir / "summary.json"),
                        "requests_csv": str(case_dir / "requests.csv") if args.output_mode in {"requests", "full"} else None,
                    }
                )
            except subprocess.TimeoutExpired as error:
                record.update({"status": "timeout", "returncode": None, "error": str(error)})
            except OSError as error:
                record.update({"status": "failed", "returncode": None, "error": str(error)})
            _write_json(record_path, record)
            print(f"[{completed}/{total}] {record['status']} seed={seed} capacity={label}", flush=True)
            if record["status"] != "ok":
                print(str(record.get("stderr", record.get("error", ""))), file=sys.stderr, flush=True)
    print(f"wrote sweep artifacts under {phase_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
