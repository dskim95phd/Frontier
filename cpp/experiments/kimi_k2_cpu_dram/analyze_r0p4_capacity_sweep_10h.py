#!/usr/bin/env python3
"""Analyze the 10-hour r0p4 P8/D16 CPU-DRAM capacity sweep.

This report is deliberately read-only with respect to simulator output.  It
expects cases such as ``cpu2000gb/r1/{summary.json,requests.csv}`` under
``--output-root`` and uses each case's ``workload.normalized.csv`` when it is
available.  The fixed converter workload and metadata under ``--workload-dir``
are the fallback/canonical source for closed-loop session reconstruction.

The script writes a five-minute CSV, a JSON document with all source-derived
series and stability diagnostics, and a self-contained HTML comparison.  A
case can be absent while a sweep is still running: absent/incomplete cases are
reported in ``missing_cases`` and do not prevent completed cases from being
reported.  No latency, queue, cache, or transfer values are fabricated when a
source field is missing; the affected output is ``null`` and the case records
the missing scope.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

try:  # Allows both ``python path/to/script.py`` and package execution.
    from .analyze_cpu4tb_rate_sweep_10h import (
        DEFAULT_BACKLOG_GROWTH_RATE_PER_S,
        DEFAULT_BACKLOG_RELATIVE_GROWTH,
        DEFAULT_QUEUE_SLOPE_PER_HOUR,
        DEFAULT_TREND_WINDOW_HOURS,
        DEFAULT_TTFT_RELATIVE_GROWTH,
        DEFAULT_TTFT_SLOPE_MS_PER_HOUR,
        RATE_EPSILON,
        _active_session_metrics,
        _apply_arrival_attention_fallback,
        _build_bin_row,
        _finite_float,
        _finite_int,
        _flatten_csv_rows,
        _load_arrival_attention_map,
        _load_metadata,
        _load_service_metrics,
        _load_summary,
        _load_workload,
        _new_bins,
        _read_requests,
        _reconstruct_arrivals,
        _stability_summary,
        _write_csv,
    )
except ImportError:  # pragma: no cover - exercised by direct script runs.
    from analyze_cpu4tb_rate_sweep_10h import (
        DEFAULT_BACKLOG_GROWTH_RATE_PER_S,
        DEFAULT_BACKLOG_RELATIVE_GROWTH,
        DEFAULT_QUEUE_SLOPE_PER_HOUR,
        DEFAULT_TREND_WINDOW_HOURS,
        DEFAULT_TTFT_RELATIVE_GROWTH,
        DEFAULT_TTFT_SLOPE_MS_PER_HOUR,
        RATE_EPSILON,
        _active_session_metrics,
        _apply_arrival_attention_fallback,
        _build_bin_row,
        _finite_float,
        _finite_int,
        _flatten_csv_rows,
        _load_arrival_attention_map,
        _load_metadata,
        _load_service_metrics,
        _load_summary,
        _load_workload,
        _new_bins,
        _read_requests,
        _reconstruct_arrivals,
        _stability_summary,
        _write_csv,
    )


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "outputs"
    / "tracelab_vera_rubin_p8_d16_cpu_capacity_sweep_continuous_10h_r0p4_20260808"
)
DEFAULT_WORKLOAD_DIR = (
    REPO_ROOT
    / "outputs"
    / "datasets"
    / "tracelab"
    / "v0.0.2"
    / "frontier"
    / "epoch1000_continuous_10h"
    / "r0p4"
)
DEFAULT_CAPACITIES_GB = (0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000)
DEFAULT_BUCKET_SECONDS = 300.0
DEFAULT_PREFILL_LANES = 8
DEFAULT_BUSY_SATURATION_PCT = 90.0
DEFAULT_PERSISTENT_POSITIVE_BIN_FRACTION = 0.60
DEFAULT_MIN_NET_INCREASE = 1.0
# Explicit QoS gates for the operational recommendation.  These are
# diagnostics, not simulator physics; every value is exposed and can be
# overridden from the CLI.  The defaults are intentionally visible because a
# different service SLO should produce a different recommendation.
DEFAULT_MAX_FINAL_HOUR_TTFT_P90_MS = 60_000.0
DEFAULT_MAX_FINAL_HOUR_QUEUE_COUNT = 1_000.0
DEFAULT_MAX_FINAL_HOUR_BACKLOG_REQUESTS = 1_000.0
SIMULATION_RATE = 0.4
CONTEXT_LENGTH_BUCKETS: tuple[tuple[str, str, int | None], ...] = (
    ("context_le_40k", "≤40K", 40_000),
    ("context_40k_to_80k", "40–80K", 80_000),
    ("context_80k_to_120k", "80–120K", 120_000),
    ("context_120k_to_160k", "120–160K", 160_000),
    ("context_160k_to_200k", "160–200K", 200_000),
    ("context_gt_200k", ">200K", None),
)


def _safe_capacity_name(capacity_gb: int) -> str:
    return f"cpu{capacity_gb:04d}gb"


def _parse_capacities(raw: str) -> list[int]:
    values: list[int] = []
    for token in raw.split(","):
        token = token.strip().lower()
        if not token:
            continue
        match = re.fullmatch(r"(?:cpu)?(\d+)(?:gb)?", token)
        if not match:
            raise ValueError(f"invalid capacity: {token!r}; use values such as 0,500,4000")
        value = int(match.group(1))
        if value < 0:
            raise ValueError(f"capacity must be nonnegative: {token!r}")
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError("at least one capacity is required")
    return sorted(values)


def _report_identity(case_dir: Path) -> dict[str, Any]:
    """Derive a human-readable model and topology label from normalized config."""

    config_path = case_dir / "config.normalized.json"
    if not config_path.is_file():
        return {
            "model_name": "LLM",
            "prefill_gpu_count": None,
            "decode_gpu_count": None,
            "prefill_only": False,
        }
    config = json.loads(config_path.read_text(encoding="utf-8"))
    clusters = config.get("clusters", {})

    def gpu_count(cluster: Any) -> int | None:
        if not isinstance(cluster, Mapping):
            return None
        parallelism = cluster.get("parallelism", {})
        if not isinstance(parallelism, Mapping):
            return None
        return math.prod(
            int(parallelism.get(key, 1))
            for key in (
                "num_replicas",
                "tensor_parallel_size",
                "pipeline_parallel_size",
                "data_parallel_size",
            )
        )

    prefill = clusters.get("prefill", {}) if isinstance(clusters, Mapping) else {}
    decode = clusters.get("decode", {}) if isinstance(clusters, Mapping) else {}
    raw_model_name = (
        str(prefill.get("model_name", "LLM"))
        if isinstance(prefill, Mapping)
        else "LLM"
    )
    kimi_match = re.search(r"Kimi[-_ ]?(K\d+)", raw_model_name, re.IGNORECASE)
    model_name = (
        f"Kimi {kimi_match.group(1).upper()}"
        if kimi_match is not None
        else raw_model_name.rsplit("/", 1)[-1]
    )
    return {
        "model_name": model_name,
        "model_asset_name": raw_model_name,
        "prefill_gpu_count": gpu_count(prefill),
        "decode_gpu_count": gpu_count(decode),
        "prefill_only": isinstance(config.get("prefill_only"), Mapping),
    }


def _discover_capacities(root: Path, requested: Sequence[int]) -> list[int]:
    """Include valid extra case directories while retaining requested gaps."""

    capacities = set(requested)
    for case in root.glob("cpu*gb/r1"):
        match = re.fullmatch(r"cpu(\d+)gb", case.parent.name.lower())
        if match:
            capacities.add(int(match.group(1)))
    return sorted(capacities)


def _rate_tag(rate: float) -> str:
    return f"r{rate:g}".replace(".", "p")


def _resolve_workload_dir(path: Path, simulation_rate: float) -> Path:
    """Accept either a rate directory or its epoch parent for convenience."""

    path = path.resolve()
    if path.is_dir() and (path / "seed_20260803.csv").is_file():
        return path
    candidate = path / _rate_tag(simulation_rate)
    return candidate if candidate.is_dir() else path


def _context_length_bucket_index(tokens: int) -> int:
    value = max(0, tokens)
    for index, (_, _, upper) in enumerate(CONTEXT_LENGTH_BUCKETS):
        if upper is None or value <= upper:
            return index
    return len(CONTEXT_LENGTH_BUCKETS) - 1


def _arrival_context_distribution_4tb(
    cases: Sequence[Mapping[str, Any]],
    *,
    bucket_seconds: float,
    metadata_path: Path | None,
) -> dict[str, Any] | None:
    """Reconstruct 4 TB closed-loop arrivals and bucket full input context."""

    source = next(
        (case for case in cases if int(case.get("capacity_gb", -1)) == 4_000),
        None,
    )
    if source is None or not math.isfinite(bucket_seconds) or bucket_seconds <= 0:
        return None
    horizon = _finite_float(source.get("horizon_s"))
    if horizon is None or horizon <= 0:
        return None
    run_dir = Path(str(source.get("run_dir", "")))
    requests_path = run_dir / "requests.csv"
    workload_path = Path(str(source.get("workload_path", "")))
    if not requests_path.is_file() or not workload_path.is_file():
        return None

    metadata_roots, _ = _load_metadata(metadata_path)
    plans = _load_workload(workload_path, metadata_roots)
    contexts: dict[int, dict[int, int]] = {}
    with workload_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            sid = _finite_int(row.get("session_id"))
            turn = _finite_int(row.get("session_turn_index"))
            tokens = _finite_int(row.get("num_prefill_tokens"))
            if sid is None or turn is None or turn < 0 or tokens is None:
                continue
            contexts.setdefault(sid, {})[turn] = max(0, tokens)

    count = max(1, int(math.ceil(horizon / bucket_seconds)))
    rows = [
        {
            "start_time_s": index * bucket_seconds,
            "end_time_s": min(horizon, (index + 1) * bucket_seconds),
            **{key: 0 for key, _, _ in CONTEXT_LENGTH_BUCKETS},
        }
        for index in range(count)
    ]
    completed: dict[int, list[tuple[float, float]]] = {}

    def add(arrived_s: float, tokens: int) -> bool:
        if not math.isfinite(arrived_s) or arrived_s < 0 or arrived_s >= horizon:
            return False
        index = min(count - 1, int(arrived_s // bucket_seconds))
        key = CONTEXT_LENGTH_BUCKETS[_context_length_bucket_index(tokens)][0]
        rows[index][key] += 1
        return True

    with requests_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            sid = _finite_int(row.get("session_id"))
            arrived = _finite_float(row.get("arrived_at_s"))
            completed_at = _finite_float(row.get("completed_at_s"))
            tokens = _finite_int(row.get("num_prefill_tokens"))
            if sid is None or arrived is None or completed_at is None or tokens is None:
                continue
            completed.setdefault(sid, []).append((arrived, completed_at))
            add(arrived, tokens)

    reconstructed = 0
    for sid, plan in plans.items():
        records = sorted(completed.get(sid, ()), key=lambda item: item[0])
        next_turn = len(records)
        if plan.turn_count <= 0 or next_turn >= plan.turn_count:
            continue
        if next_turn == 0:
            next_arrival = plan.root_arrival_s
        elif next_turn - 1 < len(records):
            next_arrival = records[next_turn - 1][1] + plan.think_times.get(
                next_turn, 0.0
            )
        else:
            next_arrival = None
        tokens = contexts.get(sid, {}).get(next_turn)
        if next_arrival is not None and tokens is not None and add(next_arrival, tokens):
            reconstructed += 1

    for row in rows:
        row["total_requests"] = sum(
            int(row[key]) for key, _, _ in CONTEXT_LENGTH_BUCKETS
        )
    return {
        "capacity_gb": 4_000,
        "context_definition": "full num_prefill_tokens at closed-loop request arrival",
        "bucket_seconds": bucket_seconds,
        "reconstructed_arrivals": reconstructed,
        "context_buckets": [
            {"key": key, "label": label, "upper_tokens_inclusive": upper}
            for key, label, upper in CONTEXT_LENGTH_BUCKETS
        ],
        "bins": rows,
    }
def _window_indices(rows: Sequence[Mapping[str, Any]], hours: float) -> list[int]:
    if not rows:
        return []
    start_s = float(rows[-1]["end_time_s"]) - hours * 3600.0
    # Use bucket starts.  A run ending at 35999.99 s must still produce exactly
    # 12/24 complete 5-minute bins for the final 1/2-hour windows.
    return [
        index
        for index, row in enumerate(rows)
        if float(row["start_time_s"]) >= start_s
    ]


def _add_window_audit(
    stability: dict[str, Any], rows: Sequence[Mapping[str, Any]], hours: float, bucket_seconds: float
) -> dict[str, Any]:
    expected = int(round(hours * 3600.0 / bucket_seconds))
    indices = _window_indices(rows, hours)
    stability = dict(stability)
    stability["expected_bin_count"] = expected
    stability["observed_bin_count"] = len(indices)
    stability["exact_bin_count"] = len(indices) == expected
    stability["window_selection"] = "bucket_start_at_or_after_(window_end-window_duration)"
    if len(indices) != expected:
        stability["classification"] = "unknown"
        reason = "expected exactly " + str(expected) + " complete bins, observed " + str(len(indices))
        stability["reason"] = f"{stability.get('reason', '')}; {reason}".strip("; ")
    return stability


def _window_qos(rows: Sequence[Mapping[str, Any]], hours: float) -> dict[str, Any]:
    """Return conservative final-window QoS observables.

    The simulator summary does not retain request-level TTFT samples per
    five-minute bucket.  ``ttft_p90_max_ms`` is therefore the maximum of the
    source-provided per-bucket p90 values, which is conservative and remains
    explicitly scoped rather than pretending to be a pooled percentile.
    """

    indices = _window_indices(rows, hours)
    selected = [rows[index] for index in indices]
    ttft_p90 = [
        float(row["ttft_p90_ms"])
        for row in selected
        if row.get("ttft_p90_ms") is not None and math.isfinite(float(row["ttft_p90_ms"]))
    ]
    queue = [
        float(row["prefill_waiting_queue_count_time_weighted"])
        for row in selected
        if row.get("prefill_waiting_queue_count_time_weighted") is not None
        and math.isfinite(float(row["prefill_waiting_queue_count_time_weighted"]))
    ]
    backlog = [
        float(row["cumulative_backlog_requests"])
        for row in selected
        if row.get("cumulative_backlog_requests") is not None
        and math.isfinite(float(row["cumulative_backlog_requests"]))
    ]
    return {
        "window_hours": hours,
        "ttft_p90_max_ms": max(ttft_p90) if ttft_p90 else None,
        "ttft_p90_observed_bins": len(ttft_p90),
        "waiting_queue_count_max": max(queue) if queue else None,
        "waiting_queue_observed_bins": len(queue),
        "cumulative_backlog_max_requests": max(backlog) if backlog else None,
        "cumulative_backlog_observed_bins": len(backlog),
        "scope": "max_of_source_five_minute_values; TTFT p90 is not pooled across bins",
    }


def _window_statistics(
    rows: Sequence[Mapping[str, Any]],
    hours: float,
    *,
    prefill_lanes: int,
) -> dict[str, Any]:
    """Aggregate auditable measurements for one trailing report window.

    Means for latency are weighted by their source sample counts. Utilization,
    active-session, and waiting-queue means are weighted by bucket duration.
    Percentiles cannot be pooled from the retained five-minute summaries, so
    the report exposes the maximum source-bin p90 and labels it accordingly.
    """

    selected = [rows[index] for index in _window_indices(rows, hours)]

    def duration(row: Mapping[str, Any]) -> float:
        value = _finite_float(row.get("duration_s"))
        if value is not None and value > 0:
            return value
        start = _finite_float(row.get("start_time_s"))
        end = _finite_float(row.get("end_time_s"))
        return max(0.0, end - start) if start is not None and end is not None else 0.0

    def total(key: str) -> float | None:
        values = [_finite_float(row.get(key)) for row in selected]
        observed = [value for value in values if value is not None]
        return sum(observed) if observed else None

    def maximum(key: str) -> float | None:
        values = [_finite_float(row.get(key)) for row in selected]
        observed = [value for value in values if value is not None]
        return max(observed) if observed else None

    def duration_weighted_mean(key: str) -> float | None:
        numerator = 0.0
        denominator = 0.0
        for row in selected:
            value = _finite_float(row.get(key))
            weight = duration(row)
            if value is None or weight <= 0:
                continue
            numerator += value * weight
            denominator += weight
        return numerator / denominator if denominator > 0 else None

    def sample_weighted_mean(value_key: str, count_key: str) -> float | None:
        numerator = 0.0
        denominator = 0.0
        for row in selected:
            value = _finite_float(row.get(value_key))
            count = _finite_float(row.get(count_key))
            if value is None or count is None or count <= 0:
                continue
            numerator += value * count
            denominator += count
        return numerator / denominator if denominator > 0 else None

    def ratio_percent(numerator: float | None, denominator: float | None) -> float | None:
        return 100.0 * numerator / denominator if numerator is not None and denominator else None

    window_duration_s = sum(duration(row) for row in selected)
    arrivals = total("request_arrivals")
    completions = total("request_completions")
    query_blocks = total("prefix_cache_query_blocks")
    gpu_hit_blocks = total("gpu_prefix_hit_blocks")
    cpu_query_blocks = total("cpu_prefix_query_blocks")
    cpu_hit_blocks = total("cpu_prefix_hit_blocks")
    combined_hit_blocks = total("prefix_cache_hit_blocks")
    transfer_bytes = total("transfer_bytes")

    backlog_start: float | None = None
    backlog_end: float | None = None
    if selected:
        first_backlog_end = _finite_float(selected[0].get("cumulative_backlog_requests"))
        first_delta = _finite_float(selected[0].get("backlog_delta_requests"))
        if first_delta is None:
            first_arrivals = _finite_float(selected[0].get("request_arrivals"))
            first_completions = _finite_float(selected[0].get("request_completions"))
            if first_arrivals is not None and first_completions is not None:
                first_delta = first_arrivals - first_completions
        if first_backlog_end is not None and first_delta is not None:
            backlog_start = first_backlog_end - first_delta
        backlog_end = _finite_float(selected[-1].get("cumulative_backlog_requests"))

    active_sessions = duration_weighted_mean("active_sessions_time_weighted")
    waiting_queue = duration_weighted_mean(
        "prefill_waiting_queue_count_time_weighted"
    )
    per_lane = float(prefill_lanes) if prefill_lanes > 0 else None
    return {
        "window_hours": hours,
        "window_start_s": (
            _finite_float(selected[0].get("start_time_s")) if selected else None
        ),
        "window_end_s": (
            _finite_float(selected[-1].get("end_time_s")) if selected else None
        ),
        "window_duration_s": window_duration_s,
        "observed_bin_count": len(selected),
        "request_arrivals": arrivals,
        "request_completions": completions,
        "completion_to_arrival_pct": ratio_percent(completions, arrivals),
        "completion_requests_per_s": (
            completions / window_duration_s
            if completions is not None and window_duration_s > 0
            else None
        ),
        "completion_requests_per_s_per_prefill_gpu": (
            completions / window_duration_s / per_lane
            if completions is not None and window_duration_s > 0 and per_lane
            else None
        ),
        "prefill_busy_pct_time_weighted": duration_weighted_mean("pool_busy_pct"),
        "active_sessions_time_weighted": active_sessions,
        "active_sessions_per_prefill_gpu": (
            active_sessions / per_lane
            if active_sessions is not None and per_lane
            else None
        ),
        "waiting_queue_count_time_weighted": waiting_queue,
        "waiting_queue_count_per_prefill_gpu": (
            waiting_queue / per_lane if waiting_queue is not None and per_lane else None
        ),
        "waiting_queue_count_max": maximum(
            "prefill_waiting_queue_count_time_weighted"
        ),
        "cumulative_backlog_start_requests": backlog_start,
        "cumulative_backlog_end_requests": backlog_end,
        "cumulative_backlog_max_requests": maximum("cumulative_backlog_requests"),
        "ttft_mean_ms": sample_weighted_mean("ttft_mean_ms", "ttft_count"),
        "ttft_p90_max_ms": maximum("ttft_p90_ms"),
        "tpot_mean_ms": sample_weighted_mean("tpot_mean_ms", "tpot_count"),
        "tpot_p90_max_ms": maximum("tpot_p90_ms"),
        "gpu_hit_pct": ratio_percent(gpu_hit_blocks, query_blocks),
        "cpu_conditional_hit_pct": ratio_percent(cpu_hit_blocks, cpu_query_blocks),
        "combined_hit_pct": ratio_percent(combined_hit_blocks, query_blocks),
        "cpu_transfer_gbps": (
            transfer_bytes / window_duration_s / 1e9
            if transfer_bytes is not None and window_duration_s > 0
            else None
        ),
        "scope": (
            "exact trailing window; means are duration- or sample-count-weighted; "
            "p90 values are maxima of source five-minute p90 values"
        ),
    }


def _integrate_step_series(
    events: Sequence[tuple[float, float]], start_s: float, end_s: float
) -> tuple[float, float]:
    """Return the time-weighted mean and maximum of a delta event series."""

    if end_s <= start_s:
        raise ValueError("working-set window must have positive duration")
    ordered = sorted(events)
    current = 0.0
    position = 0
    while position < len(ordered) and ordered[position][0] <= start_s:
        current += ordered[position][1]
        position += 1

    previous = start_s
    area = 0.0
    maximum = current
    while position < len(ordered) and ordered[position][0] < end_s:
        event_time = ordered[position][0]
        area += current * (event_time - previous)
        while position < len(ordered) and ordered[position][0] == event_time:
            current += ordered[position][1]
            position += 1
        previous = event_time
        maximum = max(maximum, current)
    area += current * (end_s - previous)
    return area / (end_s - start_s), maximum


def _hot_session_hbm_statistics(
    case_dir: Path,
    summary: Mapping[str, Any],
    *,
    horizon_s: float,
    prefill_lanes: int,
) -> dict[str, Any]:
    """Measure KV needed by sessions that still have a future request.

    The metric is an oracle capacity demand, not observed cache residency.  A
    session enters or updates the working set when PREFILL completes and is
    removed when its final planned request completes PREFILL.  KDA snapshots
    share the same accounting pool and are included when configured.
    """

    config_path = case_dir / "config.normalized.json"
    workload_path = case_dir / "workload.normalized.csv"
    requests_path = case_dir / "requests.csv"
    for path in (config_path, workload_path, requests_path):
        if not path.is_file():
            raise FileNotFoundError(f"hot-session HBM source is missing: {path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    prefill = config["clusters"]["prefill"]
    parallelism = prefill["parallelism"]
    configured_gpus = (
        int(parallelism.get("num_replicas", 1))
        * int(parallelism.get("tensor_parallel_size", 1))
        * int(parallelism.get("pipeline_parallel_size", 1))
        * int(parallelism.get("data_parallel_size", 1))
    )
    gpu_count = configured_gpus if configured_gpus > 0 else prefill_lanes
    if gpu_count <= 0:
        raise ValueError("PREFILL GPU count must be positive")

    gpu_memory = prefill["gpu_memory"]
    hbm_bytes_per_gpu = int(gpu_memory["capacity_bytes_per_gpu"])
    kv_budget_bytes_per_gpu = int(gpu_memory["kv_cache_budget_bytes_per_gpu"])
    model_weight_bytes_per_gpu = int(gpu_memory["model_weight_bytes_per_gpu"])
    reserve_bytes_per_gpu = int(
        hbm_bytes_per_gpu * float(gpu_memory.get("runtime_reserve_fraction", 0.0))
        + int(gpu_memory.get("runtime_reserve_bytes", 0))
    )

    cpu_cache = summary["cpu_kv_cache"]
    ordinary_block_bytes = int(cpu_cache["bytes_per_block"])
    snapshot_bytes_per_session = int(
        cpu_cache.get("kda_snapshot_bytes_per_session", 0)
    )
    planned_turns: Counter[int] = Counter()
    with workload_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            planned_turns[int(row["session_id"])] += 1

    completed: defaultdict[int, list[tuple[float, int]]] = defaultdict(list)
    with requests_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            completion_s = _finite_float(row.get("prefill_completed_at_s"))
            prompt_tokens = _finite_int(row.get("num_prefill_tokens"))
            session_id = _finite_int(row.get("session_id"))
            if completion_s is None or prompt_tokens is None or session_id is None:
                continue
            completed[session_id].append((completion_s, prompt_tokens))

    total_events: list[tuple[float, float]] = []
    ordinary_events: list[tuple[float, float]] = []
    snapshot_events: list[tuple[float, float]] = []
    session_events: list[tuple[float, float]] = []
    for session_id, turns in completed.items():
        planned_count = planned_turns.get(session_id)
        if not planned_count:
            continue
        old_ordinary = 0
        old_snapshot = 0
        active = False
        for completed_index, (completion_s, prompt_tokens) in enumerate(
            sorted(turns), start=1
        ):
            has_future_turn = completed_index < planned_count
            new_ordinary = (
                math.ceil(prompt_tokens / 16) * ordinary_block_bytes
                if has_future_turn
                else 0
            )
            new_snapshot = snapshot_bytes_per_session if has_future_turn else 0
            total_events.append(
                (
                    completion_s,
                    float(new_ordinary + new_snapshot - old_ordinary - old_snapshot),
                )
            )
            ordinary_events.append(
                (completion_s, float(new_ordinary - old_ordinary))
            )
            snapshot_events.append(
                (completion_s, float(new_snapshot - old_snapshot))
            )
            if has_future_turn != active:
                session_events.append((completion_s, 1.0 if has_future_turn else -1.0))
            old_ordinary = new_ordinary
            old_snapshot = new_snapshot
            active = has_future_turn

    start_s = max(0.0, horizon_s - 3600.0)
    average_bytes, maximum_bytes = _integrate_step_series(
        total_events, start_s, horizon_s
    )
    average_ordinary, _ = _integrate_step_series(
        ordinary_events, start_s, horizon_s
    )
    average_snapshot, _ = _integrate_step_series(
        snapshot_events, start_s, horizon_s
    )
    average_sessions, maximum_sessions = _integrate_step_series(
        session_events, start_s, horizon_s
    )
    kv_budget_total = gpu_count * kv_budget_bytes_per_gpu
    hbm_total = gpu_count * hbm_bytes_per_gpu
    fixed_hbm_total = gpu_count * (model_weight_bytes_per_gpu + reserve_bytes_per_gpu)
    return {
        "available": True,
        "definition": "KV of sessions with at least one future turn; terminal state removed at PREFILL completion",
        "scope": "post-hoc oracle capacity demand, not observed physical residency",
        "window_start_s": start_s,
        "window_end_s": horizon_s,
        "prefill_gpu_count": gpu_count,
        "average_hot_sessions": average_sessions,
        "maximum_hot_sessions": maximum_sessions,
        "average_ordinary_kv_gb_per_gpu": average_ordinary / gpu_count / 1e9,
        "average_kda_snapshot_gb_per_gpu": average_snapshot / gpu_count / 1e9,
        "average_hot_kv_gb_per_gpu": average_bytes / gpu_count / 1e9,
        "maximum_hot_kv_gb_per_gpu": maximum_bytes / gpu_count / 1e9,
        "kv_budget_gb_per_gpu": kv_budget_bytes_per_gpu / 1e9,
        "average_hot_kv_pct_of_budget": 100.0 * average_bytes / kv_budget_total,
        "maximum_hot_kv_pct_of_budget": 100.0 * maximum_bytes / kv_budget_total,
        "average_total_hbm_demand_gb_per_gpu": (
            fixed_hbm_total + average_bytes
        )
        / gpu_count
        / 1e9,
        "maximum_total_hbm_demand_gb_per_gpu": (
            fixed_hbm_total + maximum_bytes
        )
        / gpu_count
        / 1e9,
        "average_total_hbm_demand_pct": 100.0
        * (fixed_hbm_total + average_bytes)
        / hbm_total,
        "maximum_total_hbm_demand_pct": 100.0
        * (fixed_hbm_total + maximum_bytes)
        / hbm_total,
    }


def _primary_metrics_complete(stability: Mapping[str, Any], expected_bins: int) -> bool:
    """Require backlog and TTFT observations before recommending a capacity."""

    backlog = stability.get("backlog", {})
    ttft = stability.get("ttft", {})
    return bool(
        stability.get("exact_bin_count")
        and (int(backlog.get("observed_bins", 0) or 0) >= 2)
        and (int(ttft.get("observed_bins", 0) or 0) >= 2)
        and expected_bins > 0
    )


def _analyze_capacity_case(
    *,
    capacity_gb: int,
    simulation_rate: float,
    case_dir: Path,
    workload_path: Path,
    metadata_path: Path | None,
    bucket_seconds: float,
    prefill_lanes: int,
    busy_saturation_pct: float,
    positive_fraction: float,
    min_net_increase: float,
    max_final_hour_ttft_p90_ms: float,
    max_final_hour_queue_count: float,
    max_final_hour_backlog_requests: float,
) -> dict[str, Any]:
    """Analyze one capacity, allowing the fixed workload as a fallback."""

    summary_path = case_dir / "summary.json"
    requests_path = case_dir / "requests.csv"
    summary = _load_summary(summary_path)
    horizon = _finite_float(summary.get("simulation_window_seconds"))
    if horizon is None or horizon <= 0:
        raise ValueError(f"summary has no positive simulation_window_seconds: {summary_path}")
    bins = _new_bins(horizon, bucket_seconds)
    _load_service_metrics(summary, bins, bucket_seconds)
    metadata_roots, metadata = _load_metadata(metadata_path)
    plans = _load_workload(workload_path, metadata_roots)
    completed_by_session: dict[int, list[Any]] = {}
    _read_requests(requests_path, bins, completed_by_session, horizon_s=horizon)
    arrival_attention_available = _load_arrival_attention_map(summary, bins, bucket_seconds)
    if not arrival_attention_available:
        _apply_arrival_attention_fallback(bins)
    root_arrivals, final_completion_times, incomplete_count = _reconstruct_arrivals(
        plans, completed_by_session, bins, horizon_s=horizon
    )
    backlog = 0
    for target in bins:
        backlog += target.arrival_count - target.completion_count
        target.cumulative_backlog = backlog
    _active_session_metrics(
        bins,
        root_arrivals,
        final_completion_times,
        incomplete_count,
        horizon_s=horizon,
    )
    rows = [
        _build_bin_row(target, prefill_lanes=prefill_lanes, bucket_seconds=bucket_seconds)
        for target in bins
    ]
    label = _safe_capacity_name(capacity_gb)
    for row in rows:
        row.update(
            {
                "capacity_gb": capacity_gb,
                "capacity_label": label,
                "rate": simulation_rate,
                "rate_label": label,
            }
        )
        # Explicit alias for consumers that distinguish mean full context from
        # aggregate queued-context load.
        row["prefill_waiting_mean_full_context_tokens_time_weighted"] = row.get(
            "prefill_waiting_context_tokens_time_weighted"
        )
    final_hour = _add_window_audit(
        _stability_summary(
            rows,
            hours=1.0,
            busy_saturation_pct=busy_saturation_pct,
            positive_fraction=positive_fraction,
            min_net_increase=min_net_increase,
        ),
        rows,
        1.0,
        bucket_seconds,
    )
    last_two_hours = _add_window_audit(
        _stability_summary(
            rows,
            hours=DEFAULT_TREND_WINDOW_HOURS,
            busy_saturation_pct=busy_saturation_pct,
            positive_fraction=positive_fraction,
            min_net_increase=min_net_increase,
        ),
        rows,
        DEFAULT_TREND_WINDOW_HOURS,
        bucket_seconds,
    )
    final_hour["qos"] = _window_qos(rows, 1.0)
    final_hour["statistics"] = _window_statistics(
        rows, 1.0, prefill_lanes=prefill_lanes
    )
    try:
        hot_session_hbm = _hot_session_hbm_statistics(
            case_dir,
            summary,
            horizon_s=horizon,
            prefill_lanes=prefill_lanes,
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        hot_session_hbm = {
            "available": False,
            "reason": str(exc),
        }
    last_two_hours["qos"] = _window_qos(rows, DEFAULT_TREND_WINDOW_HOURS)
    final_qos = final_hour["qos"]
    final_qos["thresholds"] = {
        "max_ttft_p90_ms": max_final_hour_ttft_p90_ms,
        "max_waiting_queue_count": max_final_hour_queue_count,
        "max_cumulative_backlog_requests": max_final_hour_backlog_requests,
    }
    final_qos["passes"] = {
        "ttft_p90": (
            final_qos["ttft_p90_max_ms"] is not None
            and final_qos["ttft_p90_max_ms"] <= max_final_hour_ttft_p90_ms
        ),
        "waiting_queue": (
            final_qos["waiting_queue_count_max"] is not None
            and final_qos["waiting_queue_count_max"] <= max_final_hour_queue_count
        ),
        "cumulative_backlog": (
            final_qos["cumulative_backlog_max_requests"] is not None
            and final_qos["cumulative_backlog_max_requests"] <= max_final_hour_backlog_requests
        ),
    }
    final_qos["all_pass"] = all(final_qos["passes"].values())
    missing_metrics: list[str] = []
    if not any(row.get("pool_busy_pct") is not None for row in rows):
        missing_metrics.append("PREFILL busy (summary time buckets)")
    if not any(row.get("prefill_attention_token_pairs") is not None for row in rows):
        missing_metrics.append("PREFILL attention")
    if not any(row.get("ttft_count", 0) for row in rows):
        missing_metrics.append("TTFT")
    if not any(row.get("tpot_count", 0) for row in rows):
        missing_metrics.append("TPOT")
    if not metadata and not plans:
        missing_metrics.append("session workload/metadata for reconstruction")
    overall = _overall_with_totals(rows, summary)
    return {
        "capacity_gb": capacity_gb,
        "capacity_label": label,
        "rate": simulation_rate,
        "rate_label": label,
        "run_dir": str(case_dir),
        "metadata_path": str(metadata_path) if metadata_path else None,
        "metadata_available": metadata is not None,
        "metadata_root_count": len(metadata_roots),
        "workload_path": str(workload_path),
        "workload_available": workload_path.is_file(),
        "horizon_s": horizon,
        "prefill_lanes": prefill_lanes,
        "workload_session_count": len(plans),
        "completed_session_count": len(completed_by_session),
        "reconstructed_incomplete_final_session_count": incomplete_count,
        "arrival_attention_scope": (
            "summary_arrival_time_map" if arrival_attention_available else "completed_requests_fallback"
        ),
        "missing_metrics": missing_metrics,
        "overall": overall,
        "hot_session_hbm": hot_session_hbm,
        "stability": {"final_hour": final_hour, "last_2_hours": last_two_hours},
        "bins": rows,
    }


def _overall_with_totals(rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]) -> dict[str, Any]:
    """Retain summary latency and expose aggregate source counts for a case."""

    latency = summary.get("latency_ms", {})
    throughput = summary.get("throughput", {})

    def nested(name: str, key: str) -> float | None:
        value = latency.get(name, {}) if isinstance(latency, Mapping) else {}
        return _finite_float(value.get(key)) if isinstance(value, Mapping) else None

    def total(key: str) -> float:
        return sum(float(row.get(key) or 0.0) for row in rows)

    def ratio_percent(numerator: float, denominator: float) -> float | None:
        return 100.0 * numerator / denominator if denominator > 0 else None

    query = total("prefix_cache_query_blocks")
    combined_hit = total("prefix_cache_hit_blocks")
    gpu_hit = total("gpu_prefix_hit_blocks")
    cpu_query = total("cpu_prefix_query_blocks")
    cpu_hit = total("cpu_prefix_hit_blocks")

    return {
        "summary_request_count": _finite_int(
            summary.get("counts", {}).get("requests")
            if isinstance(summary.get("counts"), Mapping)
            else None
        ),
        "simulation_window_seconds": _finite_float(summary.get("simulation_window_seconds")),
        "summary_ttft_mean_ms": nested("ttft", "mean"),
        "summary_ttft_p50_ms": nested("ttft", "p50"),
        "summary_ttft_p90_ms": nested("ttft", "p90"),
        "summary_ttft_p99_ms": nested("ttft", "p99"),
        "summary_tpot_mean_ms": nested("tpot", "mean"),
        "summary_tpot_p50_ms": nested("tpot", "p50"),
        "summary_tpot_p90_ms": nested("tpot", "p90"),
        "summary_tpot_p99_ms": nested("tpot", "p99"),
        "requests_per_second": _finite_float(
            throughput.get("requests_per_second") if isinstance(throughput, Mapping) else None
        ),
        "prompt_tokens_per_second": _finite_float(
            throughput.get("prompt_tokens_per_second") if isinstance(throughput, Mapping) else None
        ),
        "decode_tokens_per_second": _finite_float(
            throughput.get("decode_tokens_per_second") if isinstance(throughput, Mapping) else None
        ),
        "request_arrivals": int(total("request_arrivals")),
        "reconstructed_arrivals": int(total("reconstructed_arrivals")),
        "request_completions": int(total("request_completions")),
        "final_cumulative_backlog_requests": rows[-1].get("cumulative_backlog_requests") if rows else None,
        "final_active_sessions": rows[-1].get("active_sessions_time_weighted") if rows else None,
        "prefix_cache_query_blocks": int(query),
        "gpu_prefix_hit_blocks": int(gpu_hit),
        "cpu_prefix_query_blocks": int(cpu_query),
        "cpu_prefix_hit_blocks": int(cpu_hit),
        "prefix_cache_hit_blocks": int(combined_hit),
        "prefix_cache_miss_blocks": int(max(0.0, query - combined_hit)),
        "gpu_hit_pct": ratio_percent(gpu_hit, query),
        "cpu_conditional_hit_pct": ratio_percent(cpu_hit, cpu_query),
        "combined_hit_pct": ratio_percent(combined_hit, query),
        "combined_miss_pct": ratio_percent(max(0.0, query - combined_hit), query),
        "cpu_restore_bytes": int(total("cpu_restore_bytes")),
        "cpu_offload_bytes": int(total("cpu_offload_bytes")),
        "cpu_transfer_bytes": int(total("transfer_bytes")),
        "summary_prefill_attention_token_pairs": (
            _finite_float(
                summary.get("batch_summary_by_cluster", {})
                .get("PREFILL", {})
                .get("prefill_attention_token_pairs")
            )
            if isinstance(summary.get("batch_summary_by_cluster"), Mapping)
            and isinstance(summary.get("batch_summary_by_cluster", {}).get("PREFILL"), Mapping)
            else None
        ),
        "rows_prefill_attention_token_pairs": total("prefill_attention_token_pairs"),
        "rows_transfer_bw_gbps_time_sum": total("transfer_bw_gbps"),
    }


def _stable_classification(value: Any) -> bool:
    return value in {"underloaded-stable", "saturated-stable"}


def _recommendations(
    cases: Sequence[Mapping[str, Any]],
    *,
    bucket_seconds: float,
    max_final_hour_ttft_p90_ms: float,
    max_final_hour_queue_count: float,
    max_final_hour_backlog_requests: float,
) -> dict[str, Any]:
    ordered = sorted(cases, key=lambda case: int(case.get("capacity_gb", 0)))
    clear: list[int] = []
    operational: list[int] = []
    eligibility: list[dict[str, Any]] = []
    for case in ordered:
        stability = case.get("stability", {})
        final = stability.get("final_hour", {})
        two = stability.get("last_2_hours", {})
        final_primary = _primary_metrics_complete(final, int(round(3600 / bucket_seconds)))
        two_primary = _primary_metrics_complete(
            two, int(round(DEFAULT_TREND_WINDOW_HOURS * 3600 / bucket_seconds))
        )
        both_windows_complete = bool(final_primary and two_primary)
        final_class = final.get("classification")
        two_class = two.get("classification")
        # "Clearly stable" is the dynamic/stability answer.  It deliberately
        # permits saturated-stable: utilization at the line is not overload
        # when the primary backlog/TTFT trends remain flat.
        clear_ok = bool(
            both_windows_complete
            and _stable_classification(final_class)
            and _stable_classification(two_class)
        )
        operational_ok = bool(
            clear_ok
            and bool(final.get("qos", {}).get("all_pass"))
        )
        if clear_ok:
            clear.append(int(case["capacity_gb"]))
        if operational_ok:
            operational.append(int(case["capacity_gb"]))
        eligibility.append(
            {
                "capacity_gb": int(case["capacity_gb"]),
                "final_hour_classification": final_class,
                "last_2_hours_classification": two_class,
                "primary_metrics_complete_both_windows": both_windows_complete,
                "clearly_stable": clear_ok,
                "operationally_acceptable": operational_ok,
                "final_hour_qos": final.get("qos", {}),
            }
        )
    return {
        "minimum_clearly_stable_capacity_gb": min(clear) if clear else None,
        "minimum_operationally_acceptable_capacity_gb": min(operational) if operational else None,
        "eligibility_by_capacity": eligibility,
        "definitions": {
            "clearly_stable": (
                "Both exact final 1h (12-bin) and last 2h (24-bin) windows are "
                "underloaded-stable or saturated-stable, with backlog and TTFT observed."
            ),
            "operationally_acceptable": (
                "Clearly stable plus all final-hour QoS gates pass: max source-bin TTFT p90, "
                "max time-weighted waiting queue, and max cumulative backlog."
            ),
            "primary_signals": "Reconstructed cumulative backlog and TTFT; queue and active-session trends are secondary and right-censored.",
            "clear_busy_requirement": f"classification uses {DEFAULT_BUSY_SATURATION_PCT:g}% to distinguish underloaded-stable from saturated-stable; both are dynamically stable",
            "operational_busy_requirement": f"busy may be at/over {DEFAULT_BUSY_SATURATION_PCT:g}% only when backlog/TTFT remain stable",
            "max_final_hour_ttft_p90_ms": max_final_hour_ttft_p90_ms,
            "max_final_hour_waiting_queue_count": max_final_hour_queue_count,
            "max_final_hour_cumulative_backlog_requests": max_final_hour_backlog_requests,
        },
    }


def _fmt(value: Any) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, str):
        return html.escape(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return html.escape(str(value))
    if not math.isfinite(number):
        return "—"
    if abs(number) >= 1000:
        return f"{number:,.1f}"
    if abs(number) >= 10:
        return f"{number:.2f}"
    return f"{number:.3f}"


def _svg_chart(
    cases: Sequence[Mapping[str, Any]],
    metric: str,
    title: str,
    unit: str = "",
    y_max_limit: float | None = None,
) -> str:
    """Small inline SVG chart; no external JS/CSS/assets are needed."""

    width, height = 960, 280
    left, right, top, bottom = 64, 18, 30, 42
    plot_w, plot_h = width - left - right, height - top - bottom
    values = [
        float(row[metric])
        for case in cases
        for row in case.get("bins", [])
        if row.get(metric) is not None and math.isfinite(float(row[metric]))
    ]
    if not values:
        return f"<h3>{html.escape(title)}</h3><p class='muted'>No source data.</p>"
    y_min = min(0.0, min(values))
    clipped = False
    if y_max_limit is not None:
        if not math.isfinite(y_max_limit) or y_max_limit <= y_min:
            raise ValueError("y_max_limit must be finite and greater than the chart minimum")
        y_max = y_max_limit
        clipped = any(value > y_max for value in values)
    else:
        y_max = max(values)
        if math.isclose(y_min, y_max):
            y_max = y_min + 1.0
        elif y_max > 0:
            y_max *= 1.08
    n = max((len(case.get("bins", [])) for case in cases), default=1)
    colors = ["#2563eb", "#ea580c", "#059669", "#7c3aed", "#dc2626", "#0891b2", "#ca8a04", "#db2777", "#475569"]

    def sx(index: int) -> float:
        return left + index / max(1, n - 1) * plot_w

    def sy(value: float) -> float:
        visible = min(y_max, max(y_min, value))
        return top + (y_max - visible) / (y_max - y_min) * plot_h

    pieces = [
        f"<h3>{html.escape(title)}</h3>",
        (
            "<p class='muted'>Y-axis capped at 5 s; values above the cap are clipped.</p>"
            if clipped and math.isclose(y_max, 5_000.0)
            else (
                "<p class='muted'>Y-axis capped at 4,000 sessions; values above the cap are clipped.</p>"
                if clipped
                and metric == "active_sessions_time_weighted"
                and math.isclose(y_max, 4_000.0)
                else ""
            )
        ),
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{html.escape(title)}'>",
    ]
    for tick in range(5):
        value = y_min + (y_max - y_min) * tick / 4
        y = sy(value)
        display_value = value
        display_unit = unit
        if (
            metric in {"ttft_mean_ms", "ttft_p90_ms"}
            and math.isclose(y_max, 5_000.0)
        ):
            display_value = value / 1_000.0
            display_unit = " s"
        pieces.append(f"<line x1='{left}' y1='{y:.1f}' x2='{width-right}' y2='{y:.1f}' class='grid'/>")
        pieces.append(f"<text x='{left-8}' y='{y+4:.1f}' text-anchor='end' class='tick'>{display_value:.3g}{html.escape(display_unit)}</text>")
    pieces.append(f"<line x1='{left}' y1='{top+plot_h}' x2='{width-right}' y2='{top+plot_h}' class='axis'/>")
    pieces.append(f"<line x1='{left}' y1='{top}' x2='{left}' y2='{top+plot_h}' class='axis'/>")
    for index in range(0, n, max(1, n // 10)):
        end_time = None
        if cases and index < len(cases[0].get("bins", [])):
            end_time = _finite_float(cases[0]["bins"][index].get("end_time_s"))
        label = f"{end_time / 3600.0:.1f}h" if end_time is not None else str(index)
        pieces.append(f"<text x='{sx(index):.1f}' y='{height-14}' text-anchor='middle' class='tick'>{label}</text>")
    for case_index, case in enumerate(cases):
        path: list[str] = []
        previous_valid = False
        for index, row in enumerate(case.get("bins", [])):
            value = row.get(metric)
            valid = value is not None and math.isfinite(float(value))
            if not valid:
                previous_valid = False
                continue
            command = "L" if previous_valid else "M"
            path.append(f"{command}{sx(index):.1f},{sy(float(value)):.1f}")
            previous_valid = True
        if path:
            pieces.append(
                f"<path d='{' '.join(path)}' fill='none' stroke='{colors[case_index % len(colors)]}' stroke-width='2'/>")
    legend = " ".join(
        f"<span><i style='background:{colors[index % len(colors)]}'></i>{html.escape(str(case.get('capacity_label', case.get('capacity_gb'))))}</span>"
        for index, case in enumerate(cases)
    )
    pieces.append("</svg><div class='legend'>" + legend + "</div>")
    return "".join(pieces)


def _stacked_context_chart(distribution: Mapping[str, Any] | None) -> str:
    if not isinstance(distribution, Mapping):
        return ""
    rows = [
        row
        for row in distribution.get("bins", [])
        if isinstance(row, Mapping)
    ]
    if not rows:
        return ""
    width, height = 1_200, 350
    left, right, top, bottom = 76, 22, 34, 52
    plot_w, plot_h = width - left - right, height - top - bottom
    totals = [
        sum(int(row.get(key, 0) or 0) for key, _, _ in CONTEXT_LENGTH_BUCKETS)
        for row in rows
    ]
    y_max = max(totals, default=0)
    if y_max <= 0:
        return ""
    y_max *= 1.06
    colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2", "#e45756", "#7f3c8d"]

    def sx(index: int) -> float:
        return left + index / max(1, len(rows) - 1) * plot_w

    def sy(value: float) -> float:
        return top + (y_max - value) / y_max * plot_h

    cumulative = [0] * len(rows)
    areas: list[str] = []
    for category_index, (key, _, _) in enumerate(CONTEXT_LENGTH_BUCKETS):
        lower = list(cumulative)
        upper = [
            lower[index] + int(row.get(key, 0) or 0)
            for index, row in enumerate(rows)
        ]
        cumulative = upper
        upper_points = " ".join(
            f"L{sx(index):.1f},{sy(value):.1f}"
            for index, value in enumerate(upper[1:], start=1)
        )
        lower_points = " ".join(
            f"L{sx(index):.1f},{sy(lower[index]):.1f}"
            for index in range(len(rows) - 1, -1, -1)
        )
        areas.append(
            f"<path d='M{sx(0):.1f},{sy(upper[0]):.1f} {upper_points} "
            f"{lower_points} Z' fill='{colors[category_index]}' fill-opacity='0.88'/>"
        )

    pieces = [
        "<h3>Incoming request full-context distribution — 4 TB</h3>",
        "<p class='muted'>Five-minute arrival counts; full context is num_prefill_tokens.</p>",
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='Incoming request full-context distribution at 4 TB over simulation time'>",
        "<title>Incoming request full-context distribution at 4 TB</title>",
        "<desc>Stacked five-minute request counts grouped into forty-thousand-token context intervals, with a final interval above two hundred thousand tokens.</desc>",
    ]
    for tick in range(5):
        value = y_max * tick / 4
        y = sy(value)
        pieces.append(
            f"<line x1='{left}' y1='{y:.1f}' x2='{width-right}' y2='{y:.1f}' class='grid'/>"
        )
        pieces.append(
            f"<text x='{left-9}' y='{y+4:.1f}' text-anchor='end' class='tick'>{value:.0f}</text>"
        )
    pieces.extend(areas)
    pieces.append(
        f"<line x1='{left}' y1='{top+plot_h}' x2='{width-right}' y2='{top+plot_h}' class='axis'/>"
    )
    pieces.append(
        f"<line x1='{left}' y1='{top}' x2='{left}' y2='{top+plot_h}' class='axis'/>"
    )
    pieces.append(
        f"<text transform='translate(18 {top + plot_h / 2:.1f}) rotate(-90)' text-anchor='middle' class='axis-label'>Requests / 5 min</text>"
    )
    for index in range(0, len(rows), max(1, len(rows) // 10)):
        end = _finite_float(rows[index].get("end_time_s"))
        label = f"{end / 3600.0:.1f}h" if end is not None else str(index)
        pieces.append(
            f"<text x='{sx(index):.1f}' y='{height-17}' text-anchor='middle' class='tick'>{label}</text>"
        )
    legend = " ".join(
        f"<span><i class='context-swatch' style='background:{colors[index]}'></i>{html.escape(label)}</span>"
        for index, (_, label, _) in enumerate(CONTEXT_LENGTH_BUCKETS)
    )
    pieces.append("</svg><div class='legend'>" + legend + "</div>")
    return "".join(pieces)


def _html_report(document: Mapping[str, Any]) -> str:
    simulation_rate = float(document.get("simulation_rate_per_s", SIMULATION_RATE))
    rate_tag = _rate_tag(simulation_rate)
    cases = list(document.get("cases", []))
    thresholds = document.get("heuristic_thresholds", {})
    recommendations = document.get("recommendations", {})
    identity = document.get("report_identity", {})
    if not isinstance(identity, Mapping):
        identity = {}
    model_name = str(identity.get("model_name") or "LLM")
    prefill_gpu_count = _finite_int(identity.get("prefill_gpu_count"))
    decode_gpu_count = _finite_int(identity.get("decode_gpu_count"))
    prefill_only = bool(identity.get("prefill_only"))
    if prefill_only:
        topology_label = (
            f"PREFILL-only, {prefill_gpu_count} PREFILL GPUs"
            if prefill_gpu_count is not None
            else "PREFILL-only"
        )
    else:
        topology_parts = []
        if prefill_gpu_count is not None:
            topology_parts.append(f"{prefill_gpu_count} PREFILL GPUs")
        if decode_gpu_count is not None:
            topology_parts.append(f"{decode_gpu_count} DECODE GPUs")
        topology_label = ", ".join(topology_parts)
    report_name = model_name + (f" — {topology_label}" if topology_label else "")
    rows: list[str] = []
    for case in cases:
        final = case.get("stability", {}).get("final_hour", {})
        two = case.get("stability", {}).get("last_2_hours", {})
        overall = case.get("overall", {})
        backlog = final.get("backlog", {})
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(case.get('capacity_label')))}</td>"
            f"<td>{_fmt(final.get('classification'))}</td>"
            f"<td>{_fmt(two.get('classification'))}</td>"
            f"<td>{_fmt(final.get('final_window_pool_busy_pct_time_weighted'))}%</td>"
            f"<td>{_fmt(backlog.get('net_change'))}</td>"
            f"<td>{_fmt(final.get('ttft', {}).get('slope_per_hour'))} ms/h</td>"
            f"<td>{_fmt(overall.get('summary_ttft_mean_ms'))} ms</td>"
            f"<td>{_fmt(overall.get('summary_tpot_mean_ms'))} ms</td>"
            f"<td>{_fmt(overall.get('combined_hit_pct'))}%</td>"
            f"<td>{_fmt(overall.get('final_cumulative_backlog_requests'))}</td>"
            "</tr>"
        )
    final_hour_details = ""
    if bool(document.get("report_options", {}).get("include_final_hour_details")):
        detail_rows: list[str] = []
        for case in cases:
            final = case.get("stability", {}).get("final_hour", {})
            stats = final.get("statistics", {})
            start_s = _finite_float(stats.get("window_start_s"))
            end_s = _finite_float(stats.get("window_end_s"))
            window = (
                f"{start_s / 3600.0:.2f}–{end_s / 3600.0:.2f} h"
                if start_s is not None and end_s is not None
                else "—"
            )
            observed_bins = stats.get("observed_bin_count")
            expected_bins = final.get("expected_bin_count")
            bins = (
                f"{observed_bins}/{expected_bins}"
                if observed_bins is not None and expected_bins is not None
                else "—"
            )
            detail_rows.append(
                "<tr>"
                f"<td>{html.escape(str(case.get('capacity_label')))}</td>"
                f"<td>{window}</td>"
                f"<td>{bins}</td>"
                f"<td>{_fmt(stats.get('request_arrivals'))}</td>"
                f"<td>{_fmt(stats.get('request_completions'))}</td>"
                f"<td>{_fmt(stats.get('completion_to_arrival_pct'))}%</td>"
                f"<td>{_fmt(stats.get('completion_requests_per_s_per_prefill_gpu'))}</td>"
                f"<td>{_fmt(stats.get('prefill_busy_pct_time_weighted'))}%</td>"
                f"<td>{_fmt(stats.get('active_sessions_per_prefill_gpu'))}</td>"
                f"<td>{_fmt(stats.get('waiting_queue_count_per_prefill_gpu'))}</td>"
                f"<td>{_fmt(stats.get('waiting_queue_count_max'))}</td>"
                f"<td>{_fmt(stats.get('cumulative_backlog_start_requests'))}</td>"
                f"<td>{_fmt(stats.get('cumulative_backlog_end_requests'))}</td>"
                f"<td>{_fmt(stats.get('cumulative_backlog_max_requests'))}</td>"
                f"<td>{_fmt(stats.get('ttft_mean_ms'))} ms</td>"
                f"<td>{_fmt(stats.get('ttft_p90_max_ms'))} ms</td>"
                f"<td>{_fmt(stats.get('tpot_mean_ms'))} ms</td>"
                f"<td>{_fmt(stats.get('tpot_p90_max_ms'))} ms</td>"
                f"<td>{_fmt(stats.get('gpu_hit_pct'))}%</td>"
                f"<td>{_fmt(stats.get('cpu_conditional_hit_pct'))}%</td>"
                f"<td>{_fmt(stats.get('combined_hit_pct'))}%</td>"
                f"<td>{_fmt(stats.get('cpu_transfer_gbps'))} GB/s</td>"
                f"<td>{_fmt(final.get('classification'))}</td>"
                "</tr>"
            )
        final_hour_details = (
            "<h2>Final-hour detailed statistics</h2>"
            "<p class='muted'>All values use the exact trailing one-hour window. "
            "Rates and per-GPU counts use the configured PREFILL GPU count. "
            "Latency means are request-sample weighted; p90 max is the maximum "
            "of the retained five-minute p90 values, not a pooled percentile. "
            "Cache-hit percentages are block weighted.</p>"
            "<div class='table-wrap'><table><thead><tr>"
            "<th>capacity / PREFILL GPU</th><th>window</th><th>bins</th>"
            "<th>arrivals</th><th>completions</th><th>completion / arrival</th>"
            "<th>completed req/s/GPU</th><th>PREFILL busy</th>"
            "<th>active sessions/GPU</th><th>waiting/GPU</th><th>waiting max</th>"
            "<th>backlog start</th><th>backlog end</th><th>backlog max</th>"
            "<th>TTFT mean</th><th>TTFT p90 max</th>"
            "<th>TPOT mean</th><th>TPOT p90 max</th>"
            "<th>GPU hit</th><th>CPU hit</th><th>combined hit</th>"
            "<th>CPU transfer</th><th>final 1h class</th>"
            f"</tr></thead><tbody>{''.join(detail_rows)}</tbody></table></div>"
        )
    hot_session_rows: list[str] = []
    unavailable_hot_session_cases: list[str] = []
    for case in cases:
        metric = case.get("hot_session_hbm", {})
        if not isinstance(metric, Mapping) or not metric.get("available"):
            unavailable_hot_session_cases.append(str(case.get("capacity_label")))
            continue
        start_s = _finite_float(metric.get("window_start_s"))
        end_s = _finite_float(metric.get("window_end_s"))
        window = (
            f"{start_s / 3600.0:.2f}–{end_s / 3600.0:.2f} h"
            if start_s is not None and end_s is not None
            else "—"
        )
        hot_session_rows.append(
            "<tr>"
            f"<td>{html.escape(str(case.get('capacity_label')))}</td>"
            f"<td>{window}</td>"
            f"<td>{_fmt(metric.get('average_hot_sessions'))}</td>"
            f"<td>{_fmt(metric.get('maximum_hot_sessions'))}</td>"
            f"<td>{_fmt(metric.get('average_ordinary_kv_gb_per_gpu'))} GB</td>"
            f"<td>{_fmt(metric.get('average_kda_snapshot_gb_per_gpu'))} GB</td>"
            f"<td>{_fmt(metric.get('average_hot_kv_gb_per_gpu'))} GB</td>"
            f"<td>{_fmt(metric.get('maximum_hot_kv_gb_per_gpu'))} GB</td>"
            f"<td>{_fmt(metric.get('kv_budget_gb_per_gpu'))} GB</td>"
            f"<td>{_fmt(metric.get('average_hot_kv_pct_of_budget'))}%</td>"
            f"<td>{_fmt(metric.get('maximum_hot_kv_pct_of_budget'))}%</td>"
            f"<td>{_fmt(metric.get('average_total_hbm_demand_gb_per_gpu'))} GB "
            f"({_fmt(metric.get('average_total_hbm_demand_pct'))}%)</td>"
            f"<td>{_fmt(metric.get('maximum_total_hbm_demand_gb_per_gpu'))} GB "
            f"({_fmt(metric.get('maximum_total_hbm_demand_pct'))}%)</td>"
            "</tr>"
        )
    hot_session_hbm_details = ""
    if hot_session_rows:
        unavailable_note = (
            "<p class='muted'>Unavailable cases: "
            + html.escape(", ".join(unavailable_hot_session_cases))
            + ".</p>"
            if unavailable_hot_session_cases
            else ""
        )
        hot_session_hbm_details = (
            "<h2>Final-hour hot-session HBM working set</h2>"
            "<p class='muted'>A hot session has at least one future request. "
            "Its KV is removed at terminal-request PREFILL completion. KDA "
            "snapshot bytes are included. Values are post-hoc capacity demand, "
            "not observed physical residency; values above 100% mean the full "
            "hot working set cannot fit in the configured KV HBM budget.</p>"
            "<div class='table-wrap'><table><thead><tr>"
            "<th>capacity / PREFILL GPU</th><th>window</th>"
            "<th>hot sessions avg</th><th>hot sessions max</th>"
            "<th>ordinary KV avg/GPU</th><th>KDA snapshot avg/GPU</th>"
            "<th>hot KV avg/GPU</th><th>hot KV max/GPU</th>"
            "<th>KV HBM budget/GPU</th><th>avg / budget</th>"
            "<th>max / budget</th><th>total HBM avg/GPU</th>"
            "<th>total HBM max/GPU</th>"
            f"</tr></thead><tbody>{''.join(hot_session_rows)}</tbody></table></div>"
            + unavailable_note
        )
    context_chart = _stacked_context_chart(
        document.get("arrival_context_distribution_4tb")
    )
    charts = (
        (f"<section class='chart context-chart'>{context_chart}</section>" if context_chart else "")
        + "".join(
        f"<section class='chart'>{_svg_chart(cases, metric, title, unit, 5_000.0 if metric in {'ttft_mean_ms', 'ttft_p90_ms'} else (4_000.0 if metric == 'active_sessions_time_weighted' else None))}</section>"
        for metric, title, unit in (
            ("pool_busy_pct", "PREFILL busy", "%"),
            ("prefill_attention_token_pairs_trillion", "PREFILL attention", " T"),
            ("request_arrivals_per_s", "Reconstructed request arrivals", " /s"),
            ("request_completions_per_s", "Request completions", " /s"),
            ("cumulative_backlog_requests", "Cumulative outstanding backlog", ""),
            ("ttft_mean_ms", "TTFT mean", " ms"),
            ("ttft_p90_ms", "TTFT p90", " ms"),
            ("tpot_mean_ms", "TPOT mean", " ms"),
            ("gpu_hit_pct", "GPU prefix hit (block-weighted)", "%"),
            ("cpu_conditional_hit_pct", "Conditional CPU prefix hit (block-weighted)", "%"),
            ("combined_hit_pct", "Combined prefix hit (block-weighted)", "%"),
            ("combined_miss_pct", "Combined prefix miss (block-weighted)", "%"),
            ("transfer_bw_gbps", "CPU transfer bandwidth", " GB/s"),
            ("active_sessions_time_weighted", "Active sessions", ""),
            ("prefill_waiting_queue_count_time_weighted", "PREFILL waiting queue count", ""),
            ("prefill_waiting_mean_full_context_tokens_time_weighted", "Mean full context per queued request", " tokens"),
        )
        )
    )
    details: list[str] = []
    headers = [
        "start_time_s", "end_time_s", "pool_busy_pct", "prefill_attention_token_pairs_trillion",
        "request_arrivals", "reconstructed_arrivals", "request_completions", "cumulative_backlog_requests",
        "ttft_mean_ms", "ttft_p90_ms", "tpot_mean_ms", "tpot_p90_ms", "gpu_hit_pct",
        "cpu_conditional_hit_pct", "combined_hit_pct", "combined_miss_pct", "transfer_bw_gbps",
        "active_sessions_time_weighted", "prefill_waiting_queue_count_time_weighted",
        "prefill_waiting_mean_full_context_tokens_time_weighted",
    ]
    for case in cases:
        head = "".join(f"<th>{html.escape(key)}</th>" for key in headers)
        body = "".join(
            "<tr>" + "".join(f"<td>{_fmt(row.get(key))}</td>" for key in headers) + "</tr>"
            for row in case.get("bins", [])
        )
        details.append(
            f"<details><summary>{html.escape(str(case.get('capacity_label')))} five-minute data</summary>"
            f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div></details>"
        )
    missing = document.get("missing_cases", [])
    missing_html = (
        "<p><strong>Missing/incomplete cases:</strong> " + html.escape(json.dumps(missing)) + "</p>"
        if missing
        else "<p><strong>Missing/incomplete cases:</strong> none</p>"
    )
    return f"""<!doctype html>
<html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>{rate_tag} {html.escape(report_name)} CPU-DRAM capacity sweep</title>
<style>
:root {{ color-scheme:light dark; --bg:#f7f8fb; --panel:#fff; --fg:#172033; --muted:#5b6475; --border:#d8dde8; --grid:#e7eaf0; }}
@media (prefers-color-scheme:dark) {{ :root {{ --bg:#0f1219; --panel:#171b24; --fg:#edf1f7; --muted:#a6afbf; --border:#31394a; --grid:#252c39; }} }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:var(--bg); color:var(--fg); font:14px/1.45 system-ui,sans-serif; }} main {{ max-width:1600px; margin:auto; padding:26px; }}
h1 {{ margin:0 0 6px; }} h2 {{ margin:28px 0 12px; }} h3 {{ margin:0 0 8px; font-size:15px; }} .muted {{ color:var(--muted); }}
.table-wrap {{ overflow-x:auto; background:var(--panel); border:1px solid var(--border); border-radius:10px; }} table {{ width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }} th,td {{ padding:7px 9px; border-bottom:1px solid var(--border); text-align:right; white-space:nowrap; }} th:first-child,td:first-child {{ text-align:left; }} th {{ color:var(--muted); font-size:12px; }}
.charts {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; }} .chart {{ min-width:0; padding:13px; background:var(--panel); border:1px solid var(--border); border-radius:10px; }} .context-chart {{ grid-column:1/-1; }} svg {{ width:100%; height:auto; display:block; }} .axis {{ stroke:var(--muted); stroke-width:1; }} .axis-label {{ fill:var(--fg); font-size:12px; }} .grid {{ stroke:var(--grid); stroke-width:1; }} .tick {{ fill:var(--muted); font-size:10px; }} .legend {{ display:flex; gap:12px; flex-wrap:wrap; color:var(--muted); font-size:12px; }} .legend i {{ display:inline-block; width:13px; height:3px; margin-right:4px; vertical-align:middle; }} .legend .context-swatch {{ height:10px; }}
details {{ margin-top:14px; }} summary {{ cursor:pointer; font-weight:600; }} .notes {{ padding:14px 16px; background:var(--panel); border-left:4px solid #2563eb; border-radius:4px; }} code {{ font-family:ui-monospace,Consolas,monospace; }}
@media (max-width:900px) {{ main{{padding:16px}} .charts{{grid-template-columns:1fr}} }}
</style></head><body><main>
<h1>{rate_tag} {html.escape(report_name)} CPU-per-PREFILL-GPU capacity sweep</h1>
<p class='muted'>Five-minute source-derived metrics; simulation rate {simulation_rate:g} sessions/s. Output root: <code>{html.escape(str(document.get('output_root')))}</code></p>
<div class='notes'><strong>Recommendations</strong><p>Minimum clearly stable: <code>{_fmt(recommendations.get('minimum_clearly_stable_capacity_gb'))} GB</code>. Minimum operationally acceptable: <code>{_fmt(recommendations.get('minimum_operationally_acceptable_capacity_gb'))} GB</code>.</p>
<p><code>underloaded-stable</code> is below the {thresholds.get('busy_saturation_pct')}% time-weighted PREFILL busy line with non-material reconstructed-backlog and TTFT trends. <code>saturated-stable</code> is at/over that line but remains flat on those primary signals. <code>overloaded</code> requires a material backlog or TTFT trend. Queue and active-session trends are secondary/right-censored diagnostics and cannot trigger overload alone. Stability requires exact 12-bin final-hour and 24-bin final-2-hour windows plus observed backlog and TTFT.</p>
<p>Backlog growth thresholds: both {thresholds.get('backlog_relative_growth')} of window arrivals and {thresholds.get('backlog_growth_rate_per_s')} requests/s, with at least {thresholds.get('positive_bin_fraction')} positive steps. TTFT trend thresholds: {thresholds.get('ttft_slope_ms_per_hour')} ms/hour and {thresholds.get('ttft_relative_growth')} relative growth. Operational QoS gates are final-hour max source-bin TTFT p90 ≤ {thresholds.get('max_final_hour_ttft_p90_ms')} ms, max time-weighted waiting queue ≤ {thresholds.get('max_final_hour_waiting_queue_count')}, and max cumulative backlog ≤ {thresholds.get('max_final_hour_cumulative_backlog_requests')}. CPU bandwidth is arrival-binned restore+offload. Waiting context is sum(context&nbsp;×&nbsp;queue-duration)/sum(queue-duration), not aggregate context load.</p>{missing_html}</div>
<h2>Capacity summary</h2><div class='table-wrap'><table><thead><tr><th>capacity</th><th>final 1h class</th><th>last 2h class</th><th>final busy</th><th>backlog Δ</th><th>TTFT slope</th><th>TTFT mean</th><th>TPOT mean</th><th>combined hit</th><th>final backlog</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
{final_hour_details}
{hot_session_hbm_details}
<h2>Five-minute comparison</h2><div class='charts'>{charts}</div>
<h2>Raw five-minute tables</h2>{''.join(details)}
<p class='muted'>Generated by <code>analyze_r0p4_capacity_sweep_10h.py</code>. JSON and CSV beside this report contain complete machine-readable output.</p>
</main></body></html>"""


def _serialize_number(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _serialize_number(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_number(item) for item in value]
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--simulation-rate",
        type=float,
        default=SIMULATION_RATE,
        help="source-session injection rate in sessions/s (default: 0.4)",
    )
    parser.add_argument(
        "--workload-dir",
        "--workload-root",
        dest="workload_dir",
        type=Path,
        default=DEFAULT_WORKLOAD_DIR,
        help="rate directory containing seed_20260803.csv and metadata (or its epoch parent)",
    )
    parser.add_argument("--metadata", type=Path, help="metadata JSON; defaults to workload-dir/seed_20260803_metadata.json")
    parser.add_argument(
        "--capacities",
        default=",".join(str(value) for value in DEFAULT_CAPACITIES_GB),
        help="comma-separated CPU DRAM capacities in GB; default includes 0..4000 in 500 GB steps",
    )
    parser.add_argument(
        "--include-discovered-capacities",
        action="store_true",
        help="also analyze valid cpu*gb/r1 directories not listed by --capacities",
    )
    parser.add_argument("--bucket-seconds", type=float, default=DEFAULT_BUCKET_SECONDS)
    parser.add_argument("--prefill-lanes", type=int, default=DEFAULT_PREFILL_LANES, help="PREFILL DP lanes (P8 defaults to 8)")
    parser.add_argument("--busy-saturation-pct", type=float, default=DEFAULT_BUSY_SATURATION_PCT)
    parser.add_argument("--persistent-positive-bin-fraction", type=float, default=DEFAULT_PERSISTENT_POSITIVE_BIN_FRACTION)
    parser.add_argument("--min-net-increase", type=float, default=DEFAULT_MIN_NET_INCREASE)
    parser.add_argument("--max-final-hour-ttft-p90-ms", type=float, default=DEFAULT_MAX_FINAL_HOUR_TTFT_P90_MS, help="operational QoS gate; max source-bin TTFT p90 in final hour")
    parser.add_argument("--max-final-hour-queue-count", type=float, default=DEFAULT_MAX_FINAL_HOUR_QUEUE_COUNT, help="operational QoS gate; max time-weighted PREFILL waiting queue count")
    parser.add_argument("--max-final-hour-backlog-requests", type=float, default=DEFAULT_MAX_FINAL_HOUR_BACKLOG_REQUESTS, help="operational QoS gate; max cumulative outstanding backlog in final hour")
    parser.add_argument(
        "--include-final-hour-details",
        action="store_true",
        help="include an exact trailing-one-hour statistics table in the HTML report",
    )
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-html", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not math.isfinite(args.bucket_seconds) or args.bucket_seconds <= 0:
        raise SystemExit("--bucket-seconds must be finite and positive")
    if not math.isfinite(args.simulation_rate) or args.simulation_rate <= 0:
        raise SystemExit("--simulation-rate must be finite and positive")
    if args.prefill_lanes <= 0:
        raise SystemExit("--prefill-lanes must be positive")
    if not math.isfinite(args.busy_saturation_pct) or args.busy_saturation_pct <= 0:
        raise SystemExit("--busy-saturation-pct must be finite and positive")
    if not 0 < args.persistent_positive_bin_fraction <= 1:
        raise SystemExit("--persistent-positive-bin-fraction must be in (0,1]")
    if not math.isfinite(args.min_net_increase) or args.min_net_increase < 0:
        raise SystemExit("--min-net-increase must be finite and nonnegative")
    for option_name, option_value in (
        ("--max-final-hour-ttft-p90-ms", args.max_final_hour_ttft_p90_ms),
        ("--max-final-hour-queue-count", args.max_final_hour_queue_count),
        ("--max-final-hour-backlog-requests", args.max_final_hour_backlog_requests),
    ):
        if not math.isfinite(option_value) or option_value < 0:
            raise SystemExit(f"{option_name} must be finite and nonnegative")
    try:
        requested = _parse_capacities(args.capacities)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    root = args.output_root.resolve()
    workload_dir = _resolve_workload_dir(args.workload_dir, args.simulation_rate)
    workload_path = workload_dir / "seed_20260803.csv"
    metadata_path = (args.metadata or workload_dir / "seed_20260803_metadata.json").resolve()
    capacities = _discover_capacities(root, requested) if args.include_discovered_capacities else requested
    cases: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for capacity_gb in capacities:
        case_dir = root / _safe_capacity_name(capacity_gb) / "r1"
        required = [case_dir / "summary.json", case_dir / "requests.csv"]
        absent = [str(path.name) for path in required if not path.is_file()]
        if absent:
            missing.append({"capacity_gb": capacity_gb, "case_dir": str(case_dir), "missing_files": absent})
            continue
        case_workload = case_dir / "workload.normalized.csv"
        selected_workload = case_workload if case_workload.is_file() else workload_path
        if not selected_workload.is_file():
            missing.append(
                {
                    "capacity_gb": capacity_gb,
                    "case_dir": str(case_dir),
                    "missing_files": ["workload.normalized.csv and fixed seed_20260803.csv"],
                }
            )
            continue
        try:
            case = _analyze_capacity_case(
                capacity_gb=capacity_gb,
                simulation_rate=args.simulation_rate,
                case_dir=case_dir,
                workload_path=selected_workload,
                metadata_path=metadata_path if metadata_path.is_file() else None,
                bucket_seconds=args.bucket_seconds,
                prefill_lanes=args.prefill_lanes,
                busy_saturation_pct=args.busy_saturation_pct,
                positive_fraction=args.persistent_positive_bin_fraction,
                min_net_increase=args.min_net_increase,
                max_final_hour_ttft_p90_ms=args.max_final_hour_ttft_p90_ms,
                max_final_hour_queue_count=args.max_final_hour_queue_count,
                max_final_hour_backlog_requests=args.max_final_hour_backlog_requests,
            )
        except (OSError, ValueError, csv.Error, json.JSONDecodeError) as exc:
            errors.append({"capacity_gb": capacity_gb, "case_dir": str(case_dir), "error": str(exc)})
            continue
        cases.append(case)
    if not cases:
        raise SystemExit(
            "no completed cases found; expected summary.json and requests.csv under "
            f"{root}. Missing: {json.dumps(missing)} Errors: {json.dumps(errors)}"
        )
    recommendations = _recommendations(
        cases,
        bucket_seconds=args.bucket_seconds,
        max_final_hour_ttft_p90_ms=args.max_final_hour_ttft_p90_ms,
        max_final_hour_queue_count=args.max_final_hour_queue_count,
        max_final_hour_backlog_requests=args.max_final_hour_backlog_requests,
    )
    nominal_horizon_s = max(float(case.get("horizon_s", 0) or 0) for case in cases)
    document: dict[str, Any] = {
        "schema_version": 1,
        "analysis": f"{_rate_tag(args.simulation_rate)}_cpu_capacity_sweep",
        "report_identity": _report_identity(Path(cases[0]["run_dir"])),
        "output_root": str(root),
        "workload_dir": str(workload_dir),
        "workload_path": str(workload_path),
        "metadata_path": str(metadata_path),
        "simulation_rate_per_s": args.simulation_rate,
        "nominal_horizon_s": nominal_horizon_s,
        "bucket_seconds": args.bucket_seconds,
        "prefill_lanes": args.prefill_lanes,
        "report_options": {
            "include_final_hour_details": args.include_final_hour_details,
        },
        "requested_capacities_gb": requested,
        "analyzed_capacities_gb": [int(case["capacity_gb"]) for case in sorted(cases, key=lambda item: item["capacity_gb"])],
        "missing_cases": missing,
        "errors": errors,
        "heuristic_thresholds": {
            "busy_saturation_pct": args.busy_saturation_pct,
            "persistent_positive_bin_fraction": args.persistent_positive_bin_fraction,
            "positive_bin_fraction": args.persistent_positive_bin_fraction,
            "min_net_increase": args.min_net_increase,
            "backlog_relative_growth": DEFAULT_BACKLOG_RELATIVE_GROWTH,
            "backlog_growth_rate_per_s": DEFAULT_BACKLOG_GROWTH_RATE_PER_S,
            "ttft_slope_ms_per_hour": DEFAULT_TTFT_SLOPE_MS_PER_HOUR,
            "ttft_relative_growth": DEFAULT_TTFT_RELATIVE_GROWTH,
            "queue_slope_per_hour": DEFAULT_QUEUE_SLOPE_PER_HOUR,
            "trend_window_hours": DEFAULT_TREND_WINDOW_HOURS,
            "final_hour_expected_bins": int(round(3600 / args.bucket_seconds)),
            "last_2_hours_expected_bins": int(round(DEFAULT_TREND_WINDOW_HOURS * 3600 / args.bucket_seconds)),
            "max_final_hour_ttft_p90_ms": args.max_final_hour_ttft_p90_ms,
            "max_final_hour_waiting_queue_count": args.max_final_hour_queue_count,
            "max_final_hour_cumulative_backlog_requests": args.max_final_hour_backlog_requests,
            "classification_basis": "backlog and TTFT primary; queue and active sessions secondary/right-censored",
        },
        "recommendations": recommendations,
        "cases": sorted(cases, key=lambda item: item["capacity_gb"]),
    }
    document["arrival_context_distribution_4tb"] = (
        _arrival_context_distribution_4tb(
            document["cases"],
            bucket_seconds=args.bucket_seconds,
            metadata_path=metadata_path if metadata_path.is_file() else None,
        )
    )
    document = _serialize_number(document)
    rate_tag = _rate_tag(args.simulation_rate)
    csv_path = (args.output_csv or root / f"{rate_tag}_capacity_sweep_10h_5min.csv").resolve()
    json_path = (args.output_json or root / f"{rate_tag}_capacity_sweep_10h.json").resolve()
    html_path = (args.output_html or root / f"{rate_tag}_capacity_sweep_10h.html").resolve()
    _write_csv(csv_path, _flatten_csv_rows(document["cases"]))
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(_html_report(document), encoding="utf-8")
    print(
        json.dumps(
            {
                "cases": len(cases),
                "analyzed_capacities_gb": document["analyzed_capacities_gb"],
                "missing_cases": missing,
                "errors": errors,
                "minimum_clearly_stable_capacity_gb": recommendations["minimum_clearly_stable_capacity_gb"],
                "minimum_operationally_acceptable_capacity_gb": recommendations["minimum_operationally_acceptable_capacity_gb"],
                "csv": str(csv_path),
                "json": str(json_path),
                "html": str(html_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
