#!/usr/bin/env python3
"""Compare Kimi K2/K3 warm-prefix PREFILL latency on the same 8-GPU node."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
from typing import Any, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
WORKLOAD = HERE / "workload_128k_plus_1k.csv"
CONFIGS = {
    "kimi_k2": HERE / "configs" / "kimi_k2_8gpu_hbm500_batch1.json",
    "kimi_k3": HERE / "configs" / "kimi_k3_8gpu_hbm500_batch1.json",
}


def _default_binary() -> Path:
    candidates = (
        REPO_ROOT / "build-release" / "frontier_sim",
        REPO_ROOT / "cpp" / "build" / "Release" / "frontier_sim.exe",
    )
    return next((path for path in candidates if path.is_file()), candidates[0])


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_requests(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _number(row: dict[str, str], key: str, cast: type = float) -> Any:
    if key not in row or row[key] == "":
        raise ValueError(f"requests.csv is missing {key}")
    return cast(row[key])


def _run_one(binary: Path, output_root: Path, label: str, config: Path) -> dict[str, Any]:
    output_dir = output_root / label
    command = [
        str(binary),
        "--config",
        str(config),
        "--workload",
        str(WORKLOAD),
        "--output-dir",
        str(output_dir),
        "--output-mode",
        "requests",
        "--runtime-validation",
        "false",
        "--gpu-kv-occupancy",
        "false",
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)

    rows = _read_requests(output_dir / "requests.csv")
    if len(rows) != 2:
        raise ValueError(f"{label}: expected two completed requests, found {len(rows)}")
    warmup, measured = rows
    cached = _number(measured, "cached_prefill_tokens", int)
    scheduled = _number(measured, "scheduled_prefill_tokens", int)
    materialized = _number(measured, "num_prefill_tokens", int)
    if cached != 131_072 or scheduled != 1_024 or materialized != 132_096:
        raise ValueError(
            f"{label}: unexpected warm-prefix accounting: cached={cached}, "
            f"scheduled={scheduled}, materialized={materialized}"
        )

    normalized = _read_json(output_dir / "config.normalized.json")
    cluster = normalized["clusters"]["monolithic"]
    memory = cluster["gpu_memory"]
    return {
        "model": label,
        "gpu_count": 8,
        "hbm_gb_per_gpu": memory["capacity_bytes_per_gpu"] / 1e9,
        "model_weight_gb_per_gpu": memory["model_weight_bytes_per_gpu"] / 1e9,
        "kv_budget_gb_per_gpu": memory["kv_cache_budget_bytes_per_gpu"] / 1e9,
        "batch_size_cap": cluster["scheduler"]["batch_size_cap"],
        "max_tokens_in_batch": cluster["scheduler"]["max_tokens_in_batch"],
        "warmup_prefill_latency_ms": _number(warmup, "prefill_latency_ms"),
        "measured_prefill_latency_ms": _number(measured, "prefill_latency_ms"),
        "cached_context_tokens": cached,
        "new_prefill_tokens": scheduled,
        "materialized_context_tokens": materialized,
        "output_dir": str(output_dir.resolve()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=_default_binary())
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "kimi_prefill_latency_8gpu_hbm500",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    binary = args.binary.resolve()
    if not binary.is_file():
        raise SystemExit(f"missing Release simulator binary: {binary}")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    results = [
        _run_one(binary, output_root, label, config.resolve())
        for label, config in CONFIGS.items()
    ]
    by_model = {item["model"]: item for item in results}
    comparison = {
        "experiment": "8 GPUs, 500 GB HBM/GPU, batch 1, 128K cached + 1K PREFILL",
        "results": results,
        "k3_to_k2_latency_ratio": (
            by_model["kimi_k3"]["measured_prefill_latency_ms"]
            / by_model["kimi_k2"]["measured_prefill_latency_ms"]
        ),
    }
    output = output_root / "comparison.json"
    output.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(comparison, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
