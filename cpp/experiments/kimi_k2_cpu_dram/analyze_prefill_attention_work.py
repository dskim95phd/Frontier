#!/usr/bin/env python3
"""Build a time series for context-dependent PREFILL attention work.

The demand side is reconstructed from request metrics.  For a request with
materialized prompt length ``N`` and cached prefix length ``H``, causal
attention work in token pairs is::

    (N * (N + 1) - H * (H + 1)) / 2

The service side is read from the compact batch time buckets emitted by the
C++ simulator.  It therefore accounts for the actual chunk schedule.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


KIMI_K2_ATTENTION_FLOPS_PER_PAIR = 2 * 64 * (192 + 128) * 61


@dataclass
class Bucket:
    start_s: float
    end_s: float
    arrived_requests: int = 0
    arrived_scheduled_tokens: int = 0
    arrived_attention_pairs: int = 0
    served_attention_pairs: int = 0
    served_prefill_tokens: int = 0
    prefill_predicted_execution_ms: float = 0.0
    ttft_ms: list[float] = field(default_factory=list)


def attention_token_pairs(num_prefill_tokens: int, cached_tokens: int) -> int:
    if num_prefill_tokens < 0 or cached_tokens < 0:
        raise ValueError("token counts must be nonnegative")
    if cached_tokens > num_prefill_tokens:
        raise ValueError("cached tokens cannot exceed the prompt length")
    return (
        num_prefill_tokens * (num_prefill_tokens + 1)
        - cached_tokens * (cached_tokens + 1)
    ) // 2


def percentile(values: Iterable[float], probability: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _as_int(row: dict[str, str], name: str) -> int:
    value = row.get(name)
    if value is None or value == "":
        raise ValueError(f"missing integer field: {name}")
    return int(value)


def _as_float(row: dict[str, str], name: str) -> float:
    value = row.get(name)
    if value is None or value == "":
        raise ValueError(f"missing numeric field: {name}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"nonfinite numeric field: {name}")
    return result


def load_arrival_work(
    requests_csv: Path, buckets: list[Bucket], bucket_seconds: float
) -> None:
    with requests_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            arrived_at = _as_float(row, "arrived_at_s")
            if arrived_at < 0:
                raise ValueError("request arrival time must be nonnegative")
            index = int(arrived_at // bucket_seconds)
            if index >= len(buckets):
                continue
            num_prefill = _as_int(row, "num_prefill_tokens")
            cached = _as_int(row, "cached_prefill_tokens")
            scheduled = _as_int(row, "scheduled_prefill_tokens")
            bucket = buckets[index]
            bucket.arrived_requests += 1
            bucket.arrived_scheduled_tokens += scheduled
            bucket.arrived_attention_pairs += attention_token_pairs(
                num_prefill, cached
            )
            bucket.ttft_ms.append(_as_float(row, "ttft_ms"))


def load_all_arrived_attention_work(
    summary: dict[str, Any], buckets: list[Bucket], bucket_seconds: float
) -> bool:
    values = summary.get("prefill_attention_token_pairs_by_arrival_time_bucket")
    if not isinstance(values, dict):
        return False
    source_bucket_seconds = float(summary.get("batch_time_bucket_seconds", 60))
    for bucket in buckets:
        bucket.arrived_attention_pairs = 0
    for raw_index, raw_pairs in values.items():
        start_s = int(raw_index) * source_bucket_seconds
        index = int(start_s // bucket_seconds)
        if index < len(buckets):
            buckets[index].arrived_attention_pairs += int(raw_pairs)
    return True


def load_service_work(
    summary: dict[str, Any], buckets: list[Bucket], bucket_seconds: float
) -> None:
    by_cluster = summary.get("batch_summary_by_cluster_time_bucket", {})
    prefill = by_cluster.get("PREFILL")
    if not isinstance(prefill, list):
        raise ValueError("summary is missing PREFILL batch time buckets")
    for item in prefill:
        start_s = float(item["start_time_s"])
        index = int(start_s // bucket_seconds)
        if index >= len(buckets):
            continue
        if "prefill_attention_token_pairs" not in item:
            raise ValueError(
                "summary lacks prefill_attention_token_pairs; rebuild and rerun "
                "the instrumented simulator"
            )
        bucket = buckets[index]
        bucket.served_attention_pairs += int(
            item["prefill_attention_token_pairs"]
        )
        bucket.served_prefill_tokens += int(
            item.get("prefill_scheduled_tokens", 0)
        )
        bucket.prefill_predicted_execution_ms += float(
            item.get("predicted_execution_ms", 0.0)
        )


def build_rows(
    buckets: list[Bucket],
    *,
    prefill_lanes: int,
    attention_flops_per_pair: int,
    arrival_attention_scope: str = "completed_requests_only",
) -> list[dict[str, int | float | str | None]]:
    if prefill_lanes <= 0:
        raise ValueError("prefill_lanes must be positive")
    cumulative_arrived = 0
    cumulative_served = 0
    rows: list[dict[str, int | float | str | None]] = []
    for bucket in buckets:
        duration = bucket.end_s - bucket.start_s
        cumulative_arrived += bucket.arrived_attention_pairs
        cumulative_served += bucket.served_attention_pairs
        mean_ttft = (
            sum(bucket.ttft_ms) / len(bucket.ttft_ms)
            if bucket.ttft_ms
            else None
        )
        rows.append(
            {
                "start_time_s": bucket.start_s,
                "end_time_s": bucket.end_s,
                "arrived_requests_completed_only": bucket.arrived_requests,
                "arrived_scheduled_prefill_tokens_completed_only": (
                    bucket.arrived_scheduled_tokens
                ),
                "arrival_attention_scope": arrival_attention_scope,
                "arrived_attention_token_pairs": bucket.arrived_attention_pairs,
                "served_prefill_tokens": bucket.served_prefill_tokens,
                "served_attention_token_pairs": bucket.served_attention_pairs,
                "arrival_attention_pflops_per_s": (
                    bucket.arrived_attention_pairs
                    * attention_flops_per_pair
                    / duration
                    / 1e15
                ),
                "served_attention_pflops_per_s": (
                    bucket.served_attention_pairs
                    * attention_flops_per_pair
                    / duration
                    / 1e15
                ),
                "prefill_predicted_execution_ms": (
                    bucket.prefill_predicted_execution_ms
                ),
                "prefill_execution_utilization": (
                    bucket.prefill_predicted_execution_ms
                    / (prefill_lanes * duration * 1000.0)
                ),
                "backlog_attention_token_pairs": (
                    cumulative_arrived - cumulative_served
                ),
                "ttft_count": len(bucket.ttft_ms),
                "ttft_mean_ms": mean_ttft,
                "ttft_p50_ms": percentile(bucket.ttft_ms, 0.50),
                "ttft_p90_ms": percentile(bucket.ttft_ms, 0.90),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bucket-seconds", type=float, default=300.0)
    parser.add_argument("--prefill-lanes", type=int, default=8)
    parser.add_argument(
        "--attention-flops-per-pair",
        type=int,
        default=KIMI_K2_ATTENTION_FLOPS_PER_PAIR,
    )
    parser.add_argument("--output-csv", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not math.isfinite(args.bucket_seconds) or args.bucket_seconds <= 0:
        raise ValueError("bucket_seconds must be finite and positive")
    summary_path = args.run_dir / "summary.json"
    requests_path = args.run_dir / "requests.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    simulation_window = float(summary["simulation_window_seconds"])
    count = int(math.ceil(simulation_window / args.bucket_seconds))
    buckets = [
        Bucket(
            start_s=index * args.bucket_seconds,
            end_s=min((index + 1) * args.bucket_seconds, simulation_window),
        )
        for index in range(count)
    ]
    load_arrival_work(requests_path, buckets, args.bucket_seconds)
    all_arrived_available = load_all_arrived_attention_work(
        summary, buckets, args.bucket_seconds
    )
    load_service_work(summary, buckets, args.bucket_seconds)
    rows = build_rows(
        buckets,
        prefill_lanes=args.prefill_lanes,
        attention_flops_per_pair=args.attention_flops_per_pair,
        arrival_attention_scope=(
            "all_arrived_requests"
            if all_arrived_available
            else "completed_requests_only"
        ),
    )
    output = args.output_csv or args.run_dir / "prefill_attention_work.csv"
    write_csv(output, rows)
    print(f"wrote {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
