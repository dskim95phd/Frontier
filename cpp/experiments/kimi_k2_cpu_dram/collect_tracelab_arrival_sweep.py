#!/usr/bin/env python3
"""Collect detailed metrics from a cache-aware TraceLab arrival-rate sweep.

The runner writes one directory per session-arrival rate (normally ``r0p05``
through ``r0p10``).  A run directory may contain ``summary.json`` and, when
``--output-mode requests`` was used, ``requests.csv``.  This collector is
intentionally independent of the runner: it can be copied to a server and
run after a sweep, and it keeps rows for missing or partial runs instead of
failing the whole report.

Examples
--------

.. code-block:: bash

   python collect_tracelab_arrival_sweep.py \
       --output-root outputs/tracelab_cache_aware_arrival_sweep

   # Explicitly collect a different grid and write outside the run root.
   python collect_tracelab_arrival_sweep.py --rates 0.05,0.06,0.07,0.08,0.09,0.10 \
       --output-root /scratch/arrival_sweep --report-prefix /scratch/reports/arrival_sweep

The main CSV has one row per requested rate.  ``*_lanes.csv`` has one row per
PREFILL/DECODE lane and is useful for checking load balance.  The Markdown
report contains compact tables plus all metric columns grouped by topic.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any, Iterable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "tracelab_cache_aware_arrival_sweep"
DEFAULT_RATES = tuple(round(0.05 + 0.01 * index, 2) for index in range(6))
BLOCK_SIZE = 16


def _finite_float(value: Any, default: float | None = None) -> float | None:
    """Return a finite float, preserving ``None`` for missing values."""

    if value is None or value == "":
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _number(value: Any, default: float = 0.0) -> float:
    result = _finite_float(value)
    return default if result is None else result


def _integer(value: Any, default: int = 0) -> int:
    result = _finite_float(value)
    if result is None:
        return default
    try:
        return int(result)
    except (OverflowError, ValueError):
        return default


def _clean_number(value: float | int | None) -> float | int | str:
    """CSV-friendly representation; missing measurements become blank."""

    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _iter_csv_rows(
    path: Path | None,
    *,
    required_field: str | None = "request_id",
) -> Iterable[tuple[dict[str, str] | None, bool]]:
    """Yield request rows without retaining the whole requests.csv in RAM.

    The request artifact can contain millions of rows.  The collector only
    needs aggregate counters, per-lane vectors, and the previous DP id per
    session, so streaming keeps peak memory bounded by the manifest and the
    small per-lane TTFT/E2E vectors.
    """

    if path is None or not path.is_file():
        return
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                yield None, True
                return
            for raw in reader:
                if not raw or all(value in (None, "") for value in raw.values()):
                    continue
                if required_field is not None and raw.get(required_field, "") in (None, ""):
                    yield None, True
                    continue
                invalid = None in raw
                yield {
                    str(key): ("" if value is None else str(value))
                    for key, value in raw.items()
                    if key is not None
                }, invalid
    except (OSError, UnicodeError, csv.Error):
        yield None, True


def _rate_value(value: Any) -> float | None:
    """Parse a rate from run metadata or a directory label."""

    parsed = _finite_float(value)
    if parsed is not None and 0.0 < parsed <= 1000.0:
        return parsed
    if value is None:
        return None
    text = str(value).lower()
    # Labels used by the sweep include r0p05, rate_0p05, 0.05, and
    # session_arrival_rate_0p05.  Avoid interpreting a duration such as 144000
    # as an arrival rate by requiring a decimal marker for directory labels.
    match = re.search(r"(?<!\d)(?:rate[_-]?)?(0(?:[p.]\d+)|1(?:[p.]0+))(?!\d)", text)
    if not match:
        return None
    try:
        return float(match.group(1).replace("p", "."))
    except ValueError:
        return None


def _rate_key(value: float) -> str:
    return f"{value:.8f}"


def _format_rate(value: float) -> str:
    return f"{value:.2f}"


def parse_rate_list(value: str) -> list[float]:
    rates: list[float] = []
    for token in value.split(","):
        rate = _rate_value(token.strip())
        if rate is None or rate <= 0.0:
            raise argparse.ArgumentTypeError(f"invalid positive arrival rate: {token!r}")
        if not any(math.isclose(rate, item, rel_tol=0.0, abs_tol=1e-9) for item in rates):
            rates.append(rate)
    if not rates:
        raise argparse.ArgumentTypeError("--rates must contain at least one rate")
    return rates


def _resolve_reference(value: Any, base: Path) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    if path.is_file():
        return path
    if not path.is_absolute():
        for candidate in (base / path, Path.cwd() / path):
            if candidate.is_file():
                return candidate
    return path


def _candidate_score(path: Path) -> tuple[int, int, float]:
    """Prefer complete requests artifacts over summary-only duplicates."""

    requests = (path / "requests.csv").is_file()
    run = (path / "run.json").is_file()
    workload = (path / "workload.normalized.csv").is_file()
    try:
        modified = (path / "summary.json").stat().st_mtime
    except OSError:
        modified = 0.0
    return (100 * int(requests) + 20 * int(run) + 2 * int(workload), int(requests), modified)


def _metadata_rate(path: Path) -> float | None:
    run = _read_json(path / "run.json")
    for key in (
        "session_arrival_rate",
        "session_arrival_rate_per_second",
        "arrival_rate",
        "rate",
    ):
        parsed = _rate_value(run.get(key))
        if parsed is not None:
            return parsed
    nested = run.get("workload")
    if isinstance(nested, Mapping):
        for key in ("session_arrival_rate", "arrival_rate", "rate"):
            parsed = _rate_value(nested.get(key))
            if parsed is not None:
                return parsed
    return None


def _find_rate_artifact(root: Path, rate: float) -> tuple[Path | None, Path | None, int]:
    """Return ``(best_run_dir, rate_dir, candidate_count)`` for one rate."""

    if not root.is_dir():
        return None, None, 0
    direct_names = {
        f"r{_format_rate(rate).replace('.', 'p')}",
        f"rate_{_format_rate(rate).replace('.', 'p')}",
        f"rate_{_format_rate(rate)}",
        _format_rate(rate),
        f"{rate:g}",
    }
    rate_dirs = [child for child in root.iterdir() if child.is_dir() and child.name in direct_names]
    candidates: list[Path] = []
    selected_rate_dir: Path | None = rate_dirs[0] if rate_dirs else None
    if rate_dirs:
        for rate_dir in rate_dirs:
            candidates.extend(path.parent for path in rate_dir.rglob("summary.json"))
    else:
        # Fallback for a runner that uses a custom name.  A candidate is
        # considered only when its run metadata or an ancestor label matches
        # the requested rate.
        for summary in root.rglob("summary.json"):
            run_dir = summary.parent
            metadata = _metadata_rate(run_dir)
            path_rate = next((_rate_value(part) for part in reversed(run_dir.relative_to(root).parts) if _rate_value(part) is not None), None)
            candidate_rate = metadata if metadata is not None else path_rate
            if candidate_rate is not None and math.isclose(candidate_rate, rate, rel_tol=0.0, abs_tol=1e-9):
                candidates.append(run_dir)
                if selected_rate_dir is None:
                    selected_rate_dir = run_dir
        # Include an otherwise empty rate directory, so a missing summary is
        # visible in the final report.
        if selected_rate_dir is None:
            for child in root.iterdir():
                if child.is_dir() and any(
                    math.isclose(_rate_value(part) or -1.0, rate, rel_tol=0.0, abs_tol=1e-9)
                    for part in child.parts
                ):
                    selected_rate_dir = child
                    break
    if not candidates:
        # A direct rate directory can itself be the run directory even when no
        # summary exists yet.  Keep it as an artifact marker.
        return None, selected_rate_dir, 0
    unique = sorted(set(candidates), key=_candidate_score, reverse=True)
    return unique[0], selected_rate_dir or unique[0], len(unique)


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return None
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _stats(values: Sequence[float], prefix: str) -> dict[str, float | int | str]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    result: dict[str, float | int | str] = {f"{prefix}_count": len(finite)}
    if not finite:
        for suffix in ("mean", "p50", "p90", "p99", "min", "max", "stddev", "cv"):
            result[f"{prefix}_{suffix}"] = ""
        return result
    mean = statistics.fmean(finite)
    stddev = statistics.pstdev(finite) if len(finite) > 1 else 0.0
    result.update(
        {
            f"{prefix}_mean": mean,
            f"{prefix}_p50": _percentile(finite, 0.50),
            f"{prefix}_p90": _percentile(finite, 0.90),
            f"{prefix}_p99": _percentile(finite, 0.99),
            f"{prefix}_min": min(finite),
            f"{prefix}_max": max(finite),
            f"{prefix}_stddev": stddev,
            f"{prefix}_cv": stddev / mean if mean else "",
        }
    )
    return result


def _flatten_latency(summary: Mapping[str, Any], row: dict[str, Any]) -> None:
    latency = summary.get("latency_ms")
    if not isinstance(latency, Mapping):
        return
    for metric in ("scheduling_delay", "prefill", "ttft", "tpot", "e2e"):
        values = latency.get(metric)
        if not isinstance(values, Mapping):
            continue
        for statistic in ("count", "mean", "p50", "p90", "p99", "min", "max"):
            if statistic in values:
                row[f"{metric}_{statistic}_ms"] = values[statistic]


def _histogram_stats(histogram: Mapping[str, Any]) -> dict[str, float | int | str]:
    weighted: list[float] = []
    for key, count in histogram.items():
        size = _integer(key, -1)
        occurrences = _integer(count)
        if size < 0 or occurrences < 0:
            continue
        weighted.extend([float(size)] * occurrences)
    return {
        "count": len(weighted),
        "p50": _percentile(weighted, 0.50) if weighted else "",
        "p90": _percentile(weighted, 0.90) if weighted else "",
        "p99": _percentile(weighted, 0.99) if weighted else "",
        "max": max(weighted) if weighted else "",
    }


def _flatten_batches(summary: Mapping[str, Any], row: dict[str, Any]) -> None:
    batches = summary.get("batch_summary_by_cluster")
    if not isinstance(batches, Mapping):
        return
    buckets = summary.get("batch_summary_by_cluster_time_bucket")
    for phase in ("PREFILL", "DECODE"):
        value = batches.get(phase)
        if not isinstance(value, Mapping):
            continue
        prefix = phase.lower()
        for source, target in (
            ("batch_count", "batch_count"),
            ("mean_batch_size", "batch_size_mean"),
            ("predicted_execution_ms", "predicted_execution_ms"),
            ("prefill_scheduled_tokens", "prefill_scheduled_tokens"),
            ("preemption_recomputed_prefill_tokens", "preemption_recomputed_prefill_tokens"),
        ):
            if source in value:
                row[f"{prefix}_{target}"] = value[source]
        histogram = value.get("batch_size_histogram")
        if isinstance(histogram, Mapping):
            for statistic, statistic_value in _histogram_stats(histogram).items():
                row[f"{prefix}_batch_size_{statistic}"] = statistic_value
            for size, count in histogram.items():
                row[f"{prefix}_batch_size_hist_{size}"] = count
        components = value.get("execution_time_components_ms")
        if isinstance(components, Mapping):
            for component, component_value in components.items():
                row[f"{prefix}_component_{component}_ms"] = component_value
        if isinstance(buckets, Mapping):
            bucket_values = buckets.get(phase)
            if isinstance(bucket_values, Sequence) and not isinstance(bucket_values, (str, bytes)):
                row[f"{prefix}_time_bucket_count"] = len(bucket_values)
                means = [_number(item.get("mean_batch_size")) for item in bucket_values if isinstance(item, Mapping)]
                executions = [_number(item.get("predicted_execution_ms")) for item in bucket_values if isinstance(item, Mapping)]
                row[f"{prefix}_time_bucket_mean_batch_size"] = statistics.fmean(means) if means else ""
                row[f"{prefix}_time_bucket_peak_mean_batch_size"] = max(means) if means else ""
                row[f"{prefix}_time_bucket_peak_execution_ms"] = max(executions) if executions else ""


def _flatten_config(config: Mapping[str, Any], row: dict[str, Any]) -> None:
    for key in ("simulation_mode", "system_architecture", "enable_parallel_clusters"):
        if key in config:
            row[f"config_{key}"] = config[key]
    for section, prefix in (("prefix_cache", "config_prefix_cache"), ("cpu_kv_cache", "config_cpu_kv_cache"), ("cluster_scheduler", "config_cluster_scheduler")):
        value = config.get(section)
        if isinstance(value, Mapping):
            for key in value:
                if isinstance(value[key], (str, int, float, bool)) or value[key] is None:
                    row[f"{prefix}_{key}"] = value[key]
    clusters = config.get("clusters")
    if not isinstance(clusters, Mapping):
        return
    for phase in ("prefill", "decode"):
        value = clusters.get(phase)
        if not isinstance(value, Mapping):
            continue
        for section in ("parallelism", "scheduler"):
            nested = value.get(section)
            if not isinstance(nested, Mapping):
                continue
            for key, item in nested.items():
                if isinstance(item, (str, int, float, bool)) or item is None:
                    row[f"config_{phase}_{section}_{key}"] = item


def _collect_gpu_occupancy(path: Path | None, row: dict[str, Any]) -> None:
    """Summarize the optional per-sample GPU KV occupancy artifact."""

    if path is None or not path.is_file():
        row["gpu_occupancy_sample_count"] = ""
        row["gpu_occupancy_invalid_rows"] = ""
        return
    by_phase: dict[str, dict[str, list[float]]] = {}
    sample_count = 0
    invalid_rows = 0
    for raw, invalid in _iter_csv_rows(path, required_field=None):
        if invalid or raw is None:
            invalid_rows += 1
            continue
        phase = str(raw.get("cluster_type", "unknown") or "unknown").upper()
        values = by_phase.setdefault(phase, {})
        sample_count += 1
        for source, name in (
            ("active_blocks", "active_blocks"),
            ("capacity_blocks", "capacity_blocks"),
            ("active_bytes_per_gpu", "active_bytes_per_gpu"),
            ("hbm_fraction", "hbm_fraction"),
            ("active_fraction_of_kv_budget", "active_kv_budget_fraction"),
            ("active_fraction_of_total_hbm", "total_hbm_fraction"),
        ):
            value = _finite_float(raw.get(source))
            if value is not None:
                values.setdefault(name, []).append(value)
    row["gpu_occupancy_sample_count"] = sample_count
    row["gpu_occupancy_invalid_rows"] = invalid_rows
    for phase, metrics in by_phase.items():
        prefix = f"gpu_occupancy_{phase.lower()}"
        for metric, values in metrics.items():
            row.update(_stats(values, f"{prefix}_{metric}"))
            # Explicit aliases make the two most useful peak values easy to
            # grep from a CSV/Markdown report.
            if metric in {"active_kv_budget_fraction", "total_hbm_fraction"}:
                row[f"{prefix}_peak_{metric}"] = max(values) if values else ""


def _manifest_rows(path: Path | None) -> dict[int, dict[str, str]]:
    result: dict[int, dict[str, str]] = {}
    needed = {
        "request_id",
        "session_id",
        "theoretical_min_prefill_tokens",
        "implementation_min_prefill_tokens",
        "materialized_prefill_tokens",
    }
    for row, invalid in _iter_csv_rows(path):
        if invalid or row is None:
            continue
        request_id = _integer(row.get("request_id"), -1)
        if request_id >= 0:
            result[request_id] = {key: row.get(key, "") for key in needed}
    return result


class _Lane:
    def __init__(self, rate: float, phase: str, lane_id: str, replica_id: str, dp_id: str) -> None:
        self.rate = rate
        self.phase = phase
        self.lane_id = lane_id
        self.replica_id = replica_id
        self.dp_id = dp_id
        self.requests = 0
        self.sessions: set[str] = set()
        self.prefill_tokens = 0.0
        self.scheduled_tokens = 0.0
        self.decode_tokens = 0.0
        self.cached_tokens = 0.0
        self.query_blocks = 0.0
        self.gpu_hit_blocks = 0.0
        self.ttft: list[float] = []
        self.e2e: list[float] = []

    def add(self, raw: Mapping[str, str], request_id: int, manifest: Mapping[str, str] | None) -> None:
        self.requests += 1
        session = raw.get("session_id") or (manifest or {}).get("session_id")
        if session not in (None, ""):
            self.sessions.add(str(session))
        self.prefill_tokens += _number(raw.get("num_prefill_tokens"))
        self.scheduled_tokens += _number(raw.get("scheduled_prefill_tokens"))
        self.decode_tokens += _number(raw.get("num_decode_tokens"))
        self.cached_tokens += _number(raw.get("cached_prefill_tokens"))
        self.query_blocks += _number(raw.get("prefix_cache_query_blocks"))
        self.gpu_hit_blocks += _number(raw.get("gpu_prefix_hit_blocks"))
        ttft = _finite_float(raw.get("ttft_ms"))
        e2e = _finite_float(raw.get("e2e_ms"))
        if ttft is not None:
            self.ttft.append(ttft)
        if e2e is not None:
            self.e2e.append(e2e)

    def as_dict(self, load_total: float) -> dict[str, Any]:
        load = self.scheduled_tokens if self.phase == "PREFILL" else self.decode_tokens
        row: dict[str, Any] = {
            "rate": self.rate,
            "phase": self.phase,
            "lane_id": self.lane_id,
            "replica_id": self.replica_id,
            "dp_id": self.dp_id,
            "request_count": self.requests,
            "session_count": len(self.sessions),
            "num_prefill_tokens": self.prefill_tokens,
            "scheduled_prefill_tokens": self.scheduled_tokens,
            "num_decode_tokens": self.decode_tokens,
            "cached_prefill_tokens": self.cached_tokens,
            "prefix_query_blocks": self.query_blocks,
            "gpu_prefix_hit_blocks": self.gpu_hit_blocks,
            "gpu_prefix_hit_rate": self.gpu_hit_blocks / self.query_blocks if self.query_blocks else "",
            "load_tokens": load,
            "load_share": load / load_total if load_total else "",
        }
        row.update(_stats(self.ttft, "ttft_ms"))
        row.update(_stats(self.e2e, "e2e_ms"))
        return row


def _lane_stats(lanes: Iterable[_Lane], row: dict[str, Any], phase: str) -> None:
    values = list(lanes)
    prefix = phase.lower()
    loads = [lane.scheduled_tokens if phase == "PREFILL" else lane.decode_tokens for lane in values]
    requests = [float(lane.requests) for lane in values]
    row[f"{prefix}_lane_count"] = len(values)
    # Keep stable, readable names (prefill_lane_load_min, rather than the
    # accidental prefill_lane_prefill_lane_load_tokens_min form).
    for name, data in (("load", loads), ("request_count", requests)):
        row.update(_stats(data, f"{prefix}_lane_{name}"))
    if loads:
        row[f"{prefix}_lane_load_imbalance_ratio"] = max(loads) / min(loads) if min(loads) > 0 else ""
        row[f"{prefix}_lane_load_max_minus_min"] = max(loads) - min(loads)
        row[f"{prefix}_lane_load_total"] = sum(loads)
    else:
        row[f"{prefix}_lane_load_imbalance_ratio"] = ""
        row[f"{prefix}_lane_load_max_minus_min"] = ""
        row[f"{prefix}_lane_load_total"] = 0


def _collect_request_metrics(
    requests_path: Path | None,
    manifest_path: Path | None,
    rate: float,
    row: dict[str, Any],
    configured_lane_counts: Mapping[str, int] | None = None,
) -> list[dict[str, Any]]:
    if requests_path is None or not requests_path.is_file():
        # Keep summary-only runs distinct from an empty, successfully written
        # requests.csv.  In particular, zero sessions must not look like a
        # measured zero-load result.
        row.update(
            {
                "requests_csv_rows": "",
                "requests_csv_invalid_rows": "",
                "manifest_rows": "",
                "manifest_unmatched_requests": "",
                "request_count_from_csv": "",
                "session_count": "",
            }
        )
        return []
    manifest = _manifest_rows(manifest_path)
    request_count = 0
    invalid_rows = 0
    row["requests_csv_rows"] = 0
    row["requests_csv_invalid_rows"] = 0
    row["manifest_rows"] = len(manifest)
    row["manifest_unmatched_requests"] = 0
    lanes: dict[tuple[str, str, str], _Lane] = {}
    session_ids: set[str] = set()
    session_last_dp: dict[str, str] = {}
    session_dps: dict[str, set[str]] = {}
    migration_transitions = 0
    migration_requests = 0
    ttft_values: list[float] = []
    e2e_values: list[float] = []
    prefill_tokens = scheduled_tokens = cached_tokens = 0.0
    decode_tokens = 0.0
    gpu_query_blocks = gpu_hit_blocks = prefix_query_blocks = prefix_hit_blocks = 0.0
    cpu_query_blocks = cpu_hit_blocks = 0.0
    request_prefill_recomputed = 0.0
    theoretical = implementation = materialized = 0.0
    affected_request_count = 0
    cpu_sum: dict[str, float] = {}
    lane_rows: list[dict[str, Any]] = []
    for raw, invalid in _iter_csv_rows(requests_path):
        if invalid or raw is None:
            invalid_rows += 1
            continue
        request_count += 1
        request_id = _integer(raw.get("request_id"), -1)
        manifest_row = manifest.get(request_id)
        if manifest and manifest_row is None:
            row["manifest_unmatched_requests"] += 1
        session = str(raw.get("session_id") or (manifest_row or {}).get("session_id") or "")
        if session:
            session_ids.add(session)
        dp = raw.get("prefill_dp_id", "")
        if session and dp != "":
            session_dps.setdefault(session, set()).add(dp)
            previous = session_last_dp.get(session)
            if previous is not None and previous != dp:
                migration_transitions += 1
                migration_requests += 1
            session_last_dp[session] = dp
        num_prefill = _number(raw.get("num_prefill_tokens"), _number((manifest_row or {}).get("materialized_prefill_tokens")))
        cached = _number(raw.get("cached_prefill_tokens"))
        recomputed = _number(raw.get("preemption_recomputed_prefill_tokens"))
        if raw.get("scheduled_prefill_tokens", "") not in (None, ""):
            scheduled = _number(raw.get("scheduled_prefill_tokens"))
        else:
            scheduled = max(0.0, num_prefill - cached) + recomputed
        prefill_tokens += num_prefill
        scheduled_tokens += scheduled
        cached_tokens += cached
        decode_tokens += _number(raw.get("num_decode_tokens"))
        request_prefill_recomputed += recomputed
        prefix_query_blocks += _number(raw.get("prefix_cache_query_blocks"))
        prefix_hit_blocks += _number(raw.get("prefix_cache_hit_blocks"))
        gpu_query_blocks += _number(raw.get("prefix_cache_query_blocks"))
        gpu_hit_blocks += _number(raw.get("gpu_prefix_hit_blocks"))
        cpu_query_blocks += _number(raw.get("cpu_prefix_query_blocks"))
        cpu_hit_blocks += _number(raw.get("cpu_prefix_hit_blocks"))
        for key, value in raw.items():
            if key.startswith("cpu_") and key not in {"cpu_prefix_query_blocks", "cpu_prefix_hit_blocks"}:
                parsed = _finite_float(value)
                if parsed is not None:
                    cpu_sum[key] = cpu_sum.get(key, 0.0) + parsed
        if manifest_row:
            theoretical += _number(manifest_row.get("theoretical_min_prefill_tokens"))
            implementation += _number(manifest_row.get("implementation_min_prefill_tokens"), _number(manifest_row.get("theoretical_min_prefill_tokens")))
            materialized += _number(manifest_row.get("materialized_prefill_tokens"), num_prefill)
            if scheduled > _number(manifest_row.get("implementation_min_prefill_tokens"), scheduled):
                affected_request_count += 1
        ttft = _finite_float(raw.get("ttft_ms"))
        e2e = _finite_float(raw.get("e2e_ms"))
        if ttft is not None:
            ttft_values.append(ttft)
        if e2e is not None:
            e2e_values.append(e2e)
        for phase, replica_key, dp_key in (
            ("PREFILL", "prefill_replica_id", "prefill_dp_id"),
            ("DECODE", "decode_replica_id", "decode_dp_id"),
        ):
            replica = raw.get(replica_key, "")
            lane_dp = raw.get(dp_key, "")
            if replica == "" and lane_dp == "":
                continue
            lane_key = (phase, replica, lane_dp)
            lane = lanes.setdefault(lane_key, _Lane(rate, phase, f"{replica}:{lane_dp}", replica, lane_dp))
            lane.add(raw, request_id, manifest_row)
    row["requests_csv_rows"] = request_count
    row["requests_csv_invalid_rows"] = invalid_rows
    row.update(
        {
            "request_count_from_csv": request_count,
            "session_count": len(session_ids),
            "prefill_input_tokens_from_csv": prefill_tokens,
            "prefill_scheduled_tokens_from_csv": scheduled_tokens,
            "prefill_cached_tokens_from_csv": cached_tokens,
            "prefill_cache_saved_fraction": cached_tokens / prefill_tokens if prefill_tokens else "",
            "prefill_scheduled_to_input_ratio": scheduled_tokens / prefill_tokens if prefill_tokens else "",
            "decode_tokens_from_csv": decode_tokens,
            "gpu_prefix_query_blocks_from_csv": gpu_query_blocks,
            "gpu_prefix_hit_blocks_from_csv": gpu_hit_blocks,
            "gpu_prefix_hit_rate_from_csv": gpu_hit_blocks / gpu_query_blocks if gpu_query_blocks else "",
            "prefix_query_blocks_from_csv": prefix_query_blocks,
            "prefix_hit_blocks_from_csv": prefix_hit_blocks,
            "prefix_hit_rate_from_csv": prefix_hit_blocks / prefix_query_blocks if prefix_query_blocks else "",
            "cpu_prefix_query_blocks_from_csv": cpu_query_blocks,
            "cpu_prefix_hit_blocks_from_csv": cpu_hit_blocks,
            "cpu_prefix_hit_rate_from_csv": cpu_hit_blocks / cpu_query_blocks if cpu_query_blocks else "",
            "prefill_recomputed_tokens_from_csv": request_prefill_recomputed,
            "migration_sessions_count": sum(1 for values in session_dps.values() if len(values) > 1),
            "migration_session_rate": (
                sum(1 for values in session_dps.values() if len(values) > 1) / len(session_ids)
                if session_ids
                else ""
            ),
            "migration_transitions": migration_transitions,
            "migration_request_rate": migration_requests / request_count if request_count else "",
            "manifest_theoretical_min_prefill_tokens": theoretical if manifest else "",
            "manifest_implementation_min_prefill_tokens": implementation if manifest else "",
            "manifest_materialized_prefill_tokens": materialized if manifest else "",
            "manifest_avoidable_recompute_tokens": max(0.0, scheduled_tokens - implementation) if manifest else "",
            "manifest_avoidable_recompute_fraction": (
                max(0.0, scheduled_tokens - implementation) / scheduled_tokens
                if manifest and scheduled_tokens
                else ""
            ),
            "manifest_affected_request_count": affected_request_count if manifest else "",
        }
    )
    row.update(_stats(ttft_values, "csv_ttft_ms"))
    row.update(_stats(e2e_values, "csv_e2e_ms"))
    for key, value in cpu_sum.items():
        row[f"cpu_requests_{key}"] = value
    for phase in ("PREFILL", "DECODE"):
        # Add configured but idle lanes so max/min and CV account for zero
        # load.  Requests may omit such lanes entirely.
        configured = _integer((configured_lane_counts or {}).get(phase), 0)
        existing = [lane for lane in lanes.values() if lane.phase == phase]
        if configured > len(existing):
            for ordinal in range(configured):
                # The common topology is one replica with DP lanes.  For
                # multiple replicas, ordinal still gives a stable lane label;
                # populated lanes retain their exact replica:dp labels.
                replica = "0"
                dp = str(ordinal)
                key = (phase, replica, dp)
                lanes.setdefault(key, _Lane(rate, phase, f"{replica}:{dp}", replica, dp))
        phase_lanes = [lane for lane in lanes.values() if lane.phase == phase]
        total_load = sum(lane.scheduled_tokens if phase == "PREFILL" else lane.decode_tokens for lane in phase_lanes)
        _lane_stats(phase_lanes, row, phase)
        for lane in phase_lanes:
            lane_rows.append(lane.as_dict(total_load))
    return lane_rows


def _collect_one(rate: float, run_dir: Path | None, rate_dir: Path | None, candidate_count: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    row: dict[str, Any] = {
        "rate": rate,
        "rate_label": _format_rate(rate),
        "status": "missing_artifact",
        "candidate_count": candidate_count,
        "artifact_dir": str(run_dir or rate_dir or ""),
        "rate_dir": str(rate_dir or ""),
    }
    if run_dir is None:
        if rate_dir is not None:
            run = _read_json(rate_dir / "run.json")
            row["run_status"] = run.get("status", "")
        return row, []
    summary_path = run_dir / "summary.json"
    requests_path = run_dir / "requests.csv"
    config_path = run_dir / "config.normalized.json"
    if not config_path.is_file():
        config_path = run_dir / "config.input.json"
    run = _read_json(run_dir / "run.json")
    summary = _read_json(summary_path)
    config = _read_json(config_path)
    row.update(
        {
            "summary_path": str(summary_path),
            "requests_path": str(requests_path) if requests_path.is_file() else "",
            "config_path": str(config_path) if config_path.is_file() else "",
            "run_status": run.get("status", ""),
            "run_returncode": run.get("returncode", ""),
            "run_wall_clock_seconds": run.get("wall_clock_seconds", run.get("process_wall_clock_seconds", "")),
            "run_process_wall_clock_seconds": run.get("process_wall_clock_seconds", ""),
            "simulation_end_time_s": run.get("simulation_end_time_s", ""),
            "workload_csv": run.get("workload_csv", ""),
            "workload_manifest_csv": run.get("workload_manifest", run.get("workload_manifest_csv", "")),
        }
    )
    if not summary:
        row["status"] = "missing_summary"
        return row, []
    row["status"] = "complete" if requests_path.is_file() else "summary_only"
    row["summary_schema_version"] = summary.get("schema_version", "")
    row["simulation_window_seconds"] = summary.get("simulation_window_seconds", "")
    row["wall_clock_seconds"] = summary.get("wall_clock_seconds", "")
    row["simulation_speedup"] = (
        _number(summary.get("simulation_window_seconds")) / _number(summary.get("wall_clock_seconds"))
        if _number(summary.get("wall_clock_seconds")) > 0
        else ""
    )
    counts = summary.get("counts") if isinstance(summary.get("counts"), Mapping) else {}
    throughput = summary.get("throughput") if isinstance(summary.get("throughput"), Mapping) else {}
    prefill_work = summary.get("prefill_work") if isinstance(summary.get("prefill_work"), Mapping) else {}
    for source, prefix in ((counts, "count"), (throughput, "throughput"), (prefill_work, "prefill_work")):
        for key, value in source.items():
            row[f"{prefix}_{key}"] = value
    prefix = summary.get("prefix_cache") if isinstance(summary.get("prefix_cache"), Mapping) else {}
    for key, value in prefix.items():
        row[f"summary_prefix_cache_{key}"] = value
    cpu = summary.get("cpu_kv_cache") if isinstance(summary.get("cpu_kv_cache"), Mapping) else {}
    for key, value in cpu.items():
        row[f"cpu_cache_{key}"] = value
    transfer = summary.get("kv_cache_transfer") if isinstance(summary.get("kv_cache_transfer"), Mapping) else {}
    for key, value in transfer.items():
        if isinstance(value, Mapping):
            for nested_key, nested_value in value.items():
                row[f"kv_transfer_{key}_{nested_key}"] = nested_value
        else:
            row[f"kv_transfer_{key}"] = value
    _flatten_latency(summary, row)
    _flatten_batches(summary, row)
    _flatten_config(config, row)
    occupancy_path = run_dir / "gpu_kv_occupancy.csv"
    _collect_gpu_occupancy(occupancy_path if occupancy_path.is_file() else None, row)
    manifest_reference = run.get("workload_manifest", run.get("workload_manifest_csv"))
    manifest_path = _resolve_reference(manifest_reference, run_dir)
    if manifest_path is None:
        # The direct rate runner often stores the manifest beside its workload
        # or in the experiment's checked-in workloads directory.
        workload = _resolve_reference(run.get("workload_csv"), run_dir)
        if workload is not None:
            guess = workload.with_name(workload.stem + "_manifest.csv")
            if guess.is_file():
                manifest_path = guess
    configured_lane_counts: dict[str, int] = {}
    clusters = config.get("clusters") if isinstance(config.get("clusters"), Mapping) else {}
    for phase, label in (("prefill", "PREFILL"), ("decode", "DECODE")):
        cluster = clusters.get(phase) if isinstance(clusters, Mapping) else None
        parallelism = cluster.get("parallelism") if isinstance(cluster, Mapping) else None
        if isinstance(parallelism, Mapping):
            replicas = _integer(parallelism.get("num_replicas"), 1)
            data_parallel = _integer(parallelism.get("data_parallel_size"), 0)
            if replicas > 0 and data_parallel > 0:
                configured_lane_counts[label] = replicas * data_parallel
    lanes = _collect_request_metrics(
        requests_path if requests_path.is_file() else None,
        manifest_path,
        rate,
        row,
        configured_lane_counts,
    )
    if requests_path.is_file() and row.get("request_count_from_csv") != _integer(counts.get("requests"), row.get("request_count_from_csv", 0)):
        row["status"] = "partial_requests"
        row["request_count_mismatch"] = True
    else:
        row["request_count_mismatch"] = False
    run_session_rate = _finite_float(run.get("session_arrival_rate", run.get("arrival_rate")))
    row["run_session_arrival_rate"] = run_session_rate if run_session_rate is not None else ""
    return row, lanes


def collect(output_root: Path, rates: Sequence[float]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    lanes: list[dict[str, Any]] = []
    for rate in rates:
        run_dir, rate_dir, candidate_count = _find_rate_artifact(output_root, rate)
        row, lane_rows = _collect_one(rate, run_dir, rate_dir, candidate_count)
        rows.append(row)
        lanes.extend(lane_rows)
    return rows, lanes


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _clean_number(row.get(key)) for key in fields})


def _md(value: Any, digits: int = 3) -> str:
    if value in (None, ""):
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return "-"
        if abs(float(value)) >= 1_000_000:
            return f"{float(value):,.0f}"
        return f"{float(value):,.{digits}f}" if isinstance(value, float) else f"{value:,}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[tuple[str, str]]) -> str:
    lines = ["| " + " | ".join(label for _, label in columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_md(row.get(key)) for key, _ in columns) + " |")
    return "\n".join(lines)


def render_report(output_root: Path, rows: Sequence[Mapping[str, Any]], lanes: Sequence[Mapping[str, Any]]) -> str:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# TraceLab arrival-rate sweep report",
        "",
        f"- Output root: `{output_root}`",
        f"- Generated: `{generated}`",
        "- A dash (`-`) means the corresponding artifact or field was not available.",
        "",
        "## Rate summary",
        "",
        _markdown_table(
            rows,
            (
                ("rate", "Rate (sessions/s)"),
                ("status", "Status"),
                ("count_requests", "Requests"),
                ("session_count", "Sessions"),
                ("throughput_requests_per_second", "Observed req/s"),
                ("ttft_mean_ms", "TTFT mean ms"),
                ("ttft_p50_ms", "TTFT p50 ms"),
                ("ttft_p90_ms", "TTFT p90 ms"),
                ("ttft_p99_ms", "TTFT p99 ms"),
                ("e2e_p99_ms", "E2E p99 ms"),
                ("prefill_batch_size_mean", "PREFILL batch mean"),
                ("decode_batch_size_mean", "DECODE batch mean"),
                ("prefill_work_scheduled_prefill_tokens", "PREFILL scheduled"),
                ("gpu_prefix_hit_rate_from_csv", "GPU block hit"),
                ("migration_sessions_count", "Migrated sessions"),
                ("prefill_lane_load_imbalance_ratio", "PREFILL lane max/min"),
                ("wall_clock_seconds", "Wall s"),
            ),
        ),
        "",
        "## Throughput and latency",
        "",
        _markdown_table(
            rows,
            tuple((key, label) for key, label in (
                ("rate", "Rate"), ("status", "Status"),
                ("count_requests", "Requests"), ("count_batches", "Batches"),
                ("throughput_requests_per_second", "Requests/s"),
                ("throughput_prompt_tokens_per_second", "Prompt tokens/s"),
                ("throughput_decode_tokens_per_second", "Decode tokens/s"),
                ("throughput_total_tokens_per_second", "Total tokens/s"),
                ("scheduling_delay_mean_ms", "Schedule mean ms"), ("scheduling_delay_p50_ms", "Schedule p50"),
                ("scheduling_delay_p90_ms", "Schedule p90"), ("scheduling_delay_p99_ms", "Schedule p99"),
                ("prefill_mean_ms", "PREFILL mean ms"), ("prefill_p50_ms", "PREFILL p50"),
                ("prefill_p90_ms", "PREFILL p90"), ("prefill_p99_ms", "PREFILL p99"),
                ("ttft_mean_ms", "TTFT mean ms"), ("ttft_p50_ms", "TTFT p50"),
                ("ttft_p90_ms", "TTFT p90"), ("ttft_p99_ms", "TTFT p99"),
                ("tpot_mean_ms", "TPOT mean ms"), ("tpot_p50_ms", "TPOT p50"),
                ("tpot_p90_ms", "TPOT p90"), ("tpot_p99_ms", "TPOT p99"),
                ("e2e_mean_ms", "E2E mean ms"), ("e2e_p50_ms", "E2E p50"),
                ("e2e_p90_ms", "E2E p90"), ("e2e_p99_ms", "E2E p99"),
                ("simulation_window_seconds", "Sim window s"), ("wall_clock_seconds", "Wall s"), ("simulation_speedup", "Sim/wall"),
            )),
        ),
        "",
        "## PREFILL work, cache, and migrations",
        "",
        _markdown_table(
            rows,
            tuple((key, label) for key, label in (
                ("rate", "Rate"), ("prefill_input_tokens_from_csv", "Materialized input"),
                ("prefill_scheduled_tokens_from_csv", "Scheduled PREFILL"),
                ("prefill_cached_tokens_from_csv", "Cached tokens"),
                ("prefill_cache_saved_fraction", "Cache saved fraction"),
                ("prefill_scheduled_to_input_ratio", "Scheduled/input"),
                ("gpu_prefix_query_blocks_from_csv", "GPU queries"), ("gpu_prefix_hit_blocks_from_csv", "GPU hits"),
                ("gpu_prefix_hit_rate_from_csv", "GPU hit rate"),
                ("cpu_prefix_query_blocks_from_csv", "CPU queries"), ("cpu_prefix_hit_blocks_from_csv", "CPU hits"),
                ("cpu_prefix_hit_rate_from_csv", "CPU hit rate"),
                ("migration_sessions_count", "Migrated sessions"), ("migration_session_rate", "Migration session rate"),
                ("migration_transitions", "DP transitions"), ("migration_request_rate", "Transition/request"),
                ("manifest_theoretical_min_prefill_tokens", "Theoretical min"),
                ("manifest_implementation_min_prefill_tokens", "Implementation min"),
                ("manifest_avoidable_recompute_tokens", "Avoidable recompute"),
                ("manifest_avoidable_recompute_fraction", "Avoidable fraction"),
                ("manifest_affected_request_count", "Affected requests"),
            )),
        ),
        "",
        "## Batch sizes and lane balance",
        "",
        _markdown_table(
            rows,
            tuple((key, label) for key, label in (
                ("rate", "Rate"), ("prefill_batch_count", "PREFILL batches"), ("prefill_batch_size_mean", "PREFILL mean"),
                ("prefill_batch_size_p50", "PREFILL p50"), ("prefill_batch_size_p90", "PREFILL p90"),
                ("prefill_batch_size_p99", "PREFILL p99"), ("prefill_batch_size_max", "PREFILL max"),
                ("prefill_lane_count", "PREFILL lanes"), ("prefill_lane_load_min", "PREFILL load min"),
                ("prefill_lane_load_max", "PREFILL load max"), ("prefill_lane_load_imbalance_ratio", "PREFILL max/min"),
                ("decode_batch_count", "DECODE batches"), ("decode_batch_size_mean", "DECODE mean"),
                ("decode_batch_size_p50", "DECODE p50"), ("decode_batch_size_p90", "DECODE p90"),
                ("decode_batch_size_p99", "DECODE p99"), ("decode_batch_size_max", "DECODE max"),
                ("decode_lane_count", "DECODE lanes"), ("decode_lane_load_min", "DECODE load min"),
                ("decode_lane_load_max", "DECODE load max"), ("decode_lane_load_imbalance_ratio", "DECODE max/min"),
            )),
        ),
        "",
        "### Per-lane distribution",
        "",
        _markdown_table(
            sorted(lanes, key=lambda item: (_number(item.get("rate")), str(item.get("phase")), str(item.get("lane_id")))),
            tuple((key, label) for key, label in (
                ("rate", "Rate"), ("phase", "Phase"), ("lane_id", "Lane"), ("request_count", "Requests"),
                ("session_count", "Sessions"), ("load_tokens", "Load tokens"), ("load_share", "Load share"),
                ("ttft_ms_mean", "TTFT mean"), ("ttft_ms_p90", "TTFT p90"),
                ("e2e_ms_mean", "E2E mean"), ("gpu_prefix_hit_rate", "GPU hit"),
            )),
        ) if lanes else "No lane rows were available (requests.csv was missing or empty).",
        "",
        "## CPU KV-cache fields",
        "",
    ]
    cpu_columns = [(key, key) for key in sorted({key for row in rows for key in row if key.startswith("cpu_cache_") or key.startswith("cpu_requests_")})]
    lines.append(_markdown_table(rows, [("rate", "Rate"), ("status", "Status"), *cpu_columns]) if cpu_columns else "No CPU KV-cache fields were present.")
    occupancy_columns = [("rate", "Rate"), ("status", "Status"), ("gpu_occupancy_sample_count", "Samples")]
    for phase in ("prefill", "decode"):
        occupancy_columns.extend(
            [
                (f"gpu_occupancy_{phase}_peak_active_kv_budget_fraction", f"{phase.upper()} peak KV-budget"),
                (f"gpu_occupancy_{phase}_peak_total_hbm_fraction", f"{phase.upper()} peak total HBM"),
                (f"gpu_occupancy_{phase}_active_kv_budget_fraction_mean", f"{phase.upper()} sample-mean KV-budget"),
                (f"gpu_occupancy_{phase}_total_hbm_fraction_mean", f"{phase.upper()} sample-mean total HBM"),
            ]
        )
    lines.extend(
        [
            "",
            "## GPU KV occupancy",
            "",
            _markdown_table(rows, occupancy_columns),
            "",
            "## Artifact and completeness status",
            "",
            _markdown_table(
                rows,
                (("rate", "Rate"), ("status", "Status"), ("run_status", "Runner status"), ("candidate_count", "Candidates"), ("request_count_mismatch", "Request count mismatch"), ("requests_csv_invalid_rows", "Invalid CSV rows"), ("gpu_occupancy_invalid_rows", "Invalid occupancy rows"), ("artifact_dir", "Artifact directory")),
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _output_paths(root: Path, report_prefix: Path | None) -> tuple[Path, Path, Path]:
    if report_prefix is None:
        return root / "arrival_sweep_summary.csv", root / "arrival_sweep_lanes.csv", root / "arrival_sweep_report.md"
    prefix = report_prefix
    if prefix.suffix:
        prefix = prefix.with_suffix("")
    return prefix.parent / f"{prefix.name}_summary.csv", prefix.parent / f"{prefix.name}_lanes.csv", prefix.parent / f"{prefix.name}_report.md"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Directory containing r0p05, r0p06, ... run directories.")
    parser.add_argument("--rates", type=parse_rate_list, default=list(DEFAULT_RATES), help="Comma-separated sessions/second rates (default: 0.05,0.06,...,0.10).")
    parser.add_argument("--report-prefix", type=Path, help="Prefix for <prefix>_summary.csv, <prefix>_lanes.csv, and <prefix>_report.md.")
    parser.add_argument("--summary-csv", type=Path, help="Explicit summary CSV path (overrides --report-prefix).")
    parser.add_argument("--lanes-csv", type=Path, help="Explicit lane CSV path (overrides --report-prefix).")
    parser.add_argument("--report", type=Path, help="Explicit Markdown report path (overrides --report-prefix).")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = args.output_root.resolve()
    rows, lanes = collect(output_root, args.rates)
    default_summary, default_lanes, default_report = _output_paths(output_root, args.report_prefix)
    summary_path = (args.summary_csv or default_summary).resolve()
    lanes_path = (args.lanes_csv or default_lanes).resolve()
    report_path = (args.report or default_report).resolve()
    _write_csv(summary_path, rows)
    _write_csv(lanes_path, lanes)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(output_root, rows, lanes) + "\n", encoding="utf-8")
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
    print(f"wrote {summary_path}")
    print(f"wrote {lanes_path}")
    print(f"wrote {report_path}")
    print("statuses: " + ", ".join(f"{key}={value}" for key, value in sorted(status_counts.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
