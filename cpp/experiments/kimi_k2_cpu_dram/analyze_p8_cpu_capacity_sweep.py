#!/usr/bin/env python3
"""Aggregate request-level metrics for the P8 CPU DRAM capacity sweep."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "outputs"
    / "tracelab_vera_rubin_p8_d16_cpu_capacity_sweep_tiered_discard_20260808"
)


def ratio_percent(numerator: int | float, denominator: int | float) -> float | None:
    return 100.0 * numerator / denominator if denominator else None


def aggregate_case(case_dir: Path, capacity_gb: int) -> dict[str, Any]:
    summary = json.loads((case_dir / "summary.json").read_text(encoding="utf-8"))
    totals = {
        "query": 0,
        "combined_hit": 0,
        "gpu_hit": 0,
        "cpu_query": 0,
        "cpu_hit": 0,
        "restore_bytes": 0,
        "offload_bytes": 0,
        "restore_operations": 0,
        "offload_operations": 0,
        "migrations": 0,
        "repeated_requests": 0,
    }
    previous_target_by_session: dict[int, tuple[int, int]] = {}
    with (case_dir / "requests.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            totals["query"] += int(row["prefix_cache_query_blocks"])
            totals["combined_hit"] += int(row["prefix_cache_hit_blocks"])
            totals["gpu_hit"] += int(row["gpu_prefix_hit_blocks"])
            totals["cpu_query"] += int(row["cpu_prefix_query_blocks"])
            totals["cpu_hit"] += int(row["cpu_prefix_hit_blocks"])
            restore_bytes = int(row["cpu_restore_bytes"])
            offload_bytes = int(row["cpu_offload_bytes"])
            totals["restore_bytes"] += restore_bytes
            totals["offload_bytes"] += offload_bytes
            totals["restore_operations"] += restore_bytes > 0
            totals["offload_operations"] += offload_bytes > 0
            session = int(row["session_id"])
            target = (int(row["prefill_replica_id"]), int(row["prefill_dp_id"]))
            previous = previous_target_by_session.get(session)
            totals["repeated_requests"] += previous is not None
            totals["migrations"] += previous is not None and previous != target
            previous_target_by_session[session] = target

    latency = summary["latency_ms"]
    prefill_summary = summary["batch_summary_by_cluster"]["PREFILL"]
    query = totals["query"]
    combined_hit = totals["combined_hit"]
    result = {
        "physical_cpu_dram_gb": capacity_gb,
        "per_prefill_gpu_slice_gb": capacity_gb / 2.0 if capacity_gb else 0.0,
        "requests": int(summary["counts"]["requests"]),
        "simulation_window_s": float(summary["simulation_window_seconds"]),
        "wall_clock_s": float(summary["wall_clock_seconds"]),
        "ttft_mean_ms": float(latency["ttft"]["mean"]),
        "ttft_p50_ms": float(latency["ttft"]["p50"]),
        "ttft_p90_ms": float(latency["ttft"]["p90"]),
        "ttft_p99_ms": float(latency["ttft"]["p99"]),
        "tpot_mean_ms": float(latency["tpot"]["mean"]),
        "tpot_p50_ms": float(latency["tpot"]["p50"]),
        "tpot_p90_ms": float(latency["tpot"]["p90"]),
        "tpot_p99_ms": float(latency["tpot"]["p99"]),
        "requests_per_s": float(summary["throughput"]["requests_per_second"]),
        "prompt_tokens_per_s": float(summary["throughput"]["prompt_tokens_per_second"]),
        "decode_tokens_per_s": float(summary["throughput"]["decode_tokens_per_second"]),
        "total_tokens_per_s": float(summary["throughput"]["total_tokens_per_second"]),
        "gpu_hit_pct": ratio_percent(totals["gpu_hit"], query),
        "cpu_incremental_hit_pct": ratio_percent(totals["cpu_hit"], query),
        "cpu_conditional_hit_pct": ratio_percent(totals["cpu_hit"], totals["cpu_query"]),
        "combined_hit_pct": ratio_percent(combined_hit, query),
        "miss_pct": ratio_percent(query - combined_hit, query),
        "prefix_query_blocks": query,
        "gpu_hit_blocks": totals["gpu_hit"],
        "cpu_query_blocks": totals["cpu_query"],
        "cpu_hit_blocks": totals["cpu_hit"],
        "restore_tb": totals["restore_bytes"] / 1e12,
        "offload_tb": totals["offload_bytes"] / 1e12,
        "restore_operations": totals["restore_operations"],
        "offload_operations": totals["offload_operations"],
        "migrations": totals["migrations"],
        "migration_pct": ratio_percent(totals["migrations"], totals["repeated_requests"]),
        "prefill_scheduled_tokens": int(summary["prefill_work"]["scheduled_prefill_tokens"]),
        "prefill_attention_token_pairs": int(prefill_summary["prefill_attention_token_pairs"]),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = args.output_root.resolve()
    rows: list[dict[str, Any]] = []
    for case in sorted(root.glob("cpu*gb/r1")):
        if not (case / "summary.json").is_file() or not (case / "requests.csv").is_file():
            continue
        capacity_gb = int(case.parent.name.removeprefix("cpu").removesuffix("gb"))
        rows.append(aggregate_case(case, capacity_gb))
    if not rows:
        raise SystemExit(f"no completed cases under {root}")
    json_path = root / "sweep_metrics.json"
    csv_path = root / "sweep_metrics.csv"
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
