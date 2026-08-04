#!/usr/bin/env python3
"""Analyze Kimi K2 CPU-DRAM sweep artifacts and choose a fine grid.

The analyzer consumes compact ``summary.json`` plus ``requests.csv`` files
written by ``frontier_sim --output-dir ... --output-mode requests``.  It does
not require ``trace.json``.  It also accepts small synthetic run directories,
which makes the threshold and fine-grid logic easy to smoke-test without a
multi-hour simulator sweep.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import statistics
from typing import Iterable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "kimi_k2_cpu_dram"
DEFAULT_WORKLOAD_ROOT = HERE / "workloads"
BLOCK_SIZE = 16
PHYSICAL_MAX_GRACE_CPU_DRAM_GB = 500.0

THRESHOLDS = {
    "cpu_extension_hit_rate_delta": 0.05,
    "scheduled_prefill_reduction": 0.05,
    "successor_ttft_p90_improvement": 0.03,
    "eviction_or_truncation_rate_delta": 0.10,
}


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _as_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _load_manifest(path: Path | None) -> dict[int, dict[str, object]]:
    if path is None or not path.is_file():
        return {}
    result: dict[int, dict[str, object]] = {}
    for raw in _read_csv(path):
        request_id = _as_int(raw.get("request_id"))
        row: dict[str, object] = dict(raw)
        for key in (
            "session_id",
            "session_turn_index",
            "num_turns",
            "new_input_tokens",
            "decode_tokens",
            "prior_context_tokens",
            "prior_decode_tokens",
            "theoretical_min_prefill_tokens",
            "implementation_min_prefill_tokens",
            "materialized_prefill_tokens",
            "final_context_tokens",
        ):
            if raw.get(key, "") != "":
                row[key] = _as_int(raw[key])
        result[request_id] = row
    return result


def _scheduled_tokens(output: Mapping[str, object], manifest: Mapping[str, object]) -> int:
    for key in (
        "actual_scheduled_prefill_tokens",
        "scheduled_prefill_tokens",
        "prefill_scheduled_tokens",
    ):
        if output.get(key) not in (None, ""):
            return max(0, _as_int(output[key]))
    materialized = _as_int(manifest.get("materialized_prefill_tokens"), _as_int(manifest.get("new_input_tokens")))
    cached = _as_int(output.get("cached_prefill_tokens"))
    scheduled = max(0, materialized - cached)
    if output.get("preemption_recomputed_prefill_tokens") not in (None, ""):
        scheduled += _as_int(output.get("preemption_recomputed_prefill_tokens"))
    return scheduled


def _capacity_from_record(record: Mapping[str, object], path: Path) -> tuple[str, float | None, float | None, float | None, bool]:
    capacity = record.get("capacity")
    if isinstance(capacity, Mapping):
        label = str(capacity.get("label", ""))
        grace = capacity.get("grace_cpu_dram_gb")
        gpu_slice = capacity.get("gpu_slice_dram_gb")
        rho = capacity.get("rho")
        oracle = bool(capacity.get("oracle", label == "oracle_unbounded_cpu"))
        return label or path.parent.name, None if grace is None else _as_float(grace), None if gpu_slice is None else _as_float(gpu_slice), None if rho is None else _as_float(rho), oracle
    for candidate in (path.parent.name, path.parent.parent.name):
        if candidate == "off":
            return candidate, 0.0, 0.0, 0.0, False
        if candidate == "oracle_unbounded_cpu":
            return candidate, None, None, None, True
        text = candidate[3:] if candidate.lower().startswith("cpu") else ""
        try:
            grace = float(text)
        except ValueError:
            continue
        return candidate, grace, grace / 2.0, (grace / 2.0 * 1_000_000_000) / 190_000_000_000, False
    return path.parent.name, None, None, None, False


def _summary_path(record_path: Path, record: Mapping[str, object]) -> Path:
    value = record.get("summary_json")
    return Path(str(value)) if value else record_path.parent / "summary.json"


def _requests_path(record_path: Path, record: Mapping[str, object]) -> Path:
    value = record.get("requests_csv")
    return Path(str(value)) if value else record_path.parent / "requests.csv"


def _manifest_path(record_path: Path, record: Mapping[str, object], workload_root: Path) -> Path | None:
    value = record.get("workload_manifest_csv")
    if value:
        return Path(str(value))
    seed = record.get("seed")
    if seed is not None:
        return workload_root / f"seed_{_as_int(seed)}_manifest.csv"
    return None


def summarize_run(record_path: Path, *, workload_root: Path = DEFAULT_WORKLOAD_ROOT) -> dict[str, object] | None:
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(record, Mapping) or record.get("status") not in {"ok", "planned", None}:
        return None
    summary_path = _summary_path(record_path, record)
    if not summary_path.is_file():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(summary, Mapping):
        return None
    label, grace, gpu_slice, rho, oracle = _capacity_from_record(record, record_path)
    manifest = _load_manifest(_manifest_path(record_path, record, workload_root))
    requests_path = _requests_path(record_path, record)
    output_rows: list[dict[str, object]] = []
    if requests_path.is_file():
        for raw in _read_csv(requests_path):
            request_id = _as_int(raw.get("request_id"))
            joined: dict[str, object] = dict(raw)
            joined.update(manifest.get(request_id, {}))
            joined["request_id"] = request_id
            joined["session_turn_index"] = _as_int(joined.get("session_turn_index"))
            joined["session_id"] = _as_int(joined.get("session_id"))
            joined["new_input_tokens"] = _as_int(joined.get("new_input_tokens"), _as_int(raw.get("num_prefill_tokens")))
            joined["scheduled_prefill_tokens"] = _scheduled_tokens(raw, joined)
            joined["ttft_ms"] = _as_float(raw.get("ttft_ms"))
            joined["cached_prefill_tokens"] = _as_int(raw.get("cached_prefill_tokens"))
            output_rows.append(joined)
    cpu = summary.get("cpu_kv_cache") if isinstance(summary.get("cpu_kv_cache"), Mapping) else {}
    prefix = summary.get("prefix_cache") if isinstance(summary.get("prefix_cache"), Mapping) else {}
    if output_rows:
        new_input = sum(_as_int(row.get("new_input_tokens")) for row in output_rows)
        scheduled = sum(_as_int(row.get("scheduled_prefill_tokens")) for row in output_rows)
        successor_ttft = [_as_float(row.get("ttft_ms")) for row in output_rows if _as_int(row.get("session_turn_index")) > 0]
        session_ids = {row.get("session_id") for row in output_rows}
        successor_mean = statistics.fmean(successor_ttft) if successor_ttft else 0.0
        successor_p50 = percentile(successor_ttft, 0.50)
        successor_p90 = percentile(successor_ttft, 0.90)
        successor_p99 = percentile(successor_ttft, 0.99)
        session_successors: dict[int, list[float]] = {}
        for row in output_rows:
            if _as_int(row.get("session_turn_index")) > 0:
                session_successors.setdefault(_as_int(row.get("session_id")), []).append(
                    _as_float(row.get("ttft_ms"))
                )
        session_means = [statistics.fmean(values) for values in session_successors.values() if values]
        session_equal_mean = statistics.fmean(session_means) if session_means else 0.0
        session_equal_p50 = percentile(session_means, 0.50)
        session_equal_p90 = percentile(session_means, 0.90)
        session_equal_p99 = percentile(session_means, 0.99)
        cpu_query_rows = sum(_as_int(row.get("cpu_prefix_query_blocks")) for row in output_rows)
        cpu_hit_rows = sum(_as_int(row.get("cpu_prefix_hit_blocks")) for row in output_rows)
        gpu_hit_rows = sum(_as_int(row.get("gpu_prefix_hit_blocks")) for row in output_rows)
        restorable_prefix_blocks = sum(
            max(
                0,
                (_as_int(row.get("prior_context_tokens")) - _as_int(row.get("prior_decode_tokens")))
                // BLOCK_SIZE,
            )
            for row in output_rows
            if _as_int(row.get("session_turn_index")) > 0
        )
        cpu_eligible_blocks = max(0, restorable_prefix_blocks - gpu_hit_rows)
        cpu_restorable_hit_rate = (
            cpu_hit_rows / cpu_eligible_blocks if cpu_eligible_blocks else 0.0
        )
        combined_restorable_hit_rate = (
            min(1.0, (gpu_hit_rows + cpu_hit_rows) / restorable_prefix_blocks)
            if restorable_prefix_blocks
            else 0.0
        )
        cpu_hit_rate = cpu_hit_rows / cpu_query_rows if cpu_query_rows else _as_float(cpu.get("hit_rate"))
        preemptions = sum(_as_int(row.get("preemption_count")) for row in output_rows)
    else:
        new_input = _as_int(summary.get("throughput", {}).get("prompt_tokens_per_second")) if isinstance(summary.get("throughput"), Mapping) else 0
        scheduled = new_input
        latency = summary.get("latency_ms", {}) if isinstance(summary.get("latency_ms"), Mapping) else {}
        ttft = latency.get("ttft", {}) if isinstance(latency.get("ttft"), Mapping) else {}
        successor_mean = _as_float(ttft.get("mean"))
        successor_p50 = _as_float(ttft.get("p50"))
        successor_p90 = _as_float(ttft.get("p90"))
        successor_p99 = _as_float(ttft.get("p99"))
        session_equal_mean = 0.0
        session_equal_p50 = 0.0
        session_equal_p90 = 0.0
        session_equal_p99 = 0.0
        session_ids = set()
        cpu_hit_rate = _as_float(cpu.get("hit_rate"))
        gpu_hit_rows = _as_int(prefix.get("hit_blocks"))
        restorable_prefix_blocks = 0
        cpu_eligible_blocks = 0
        cpu_restorable_hit_rate = 0.0
        combined_restorable_hit_rate = 0.0
        preemptions = _as_int(summary.get("counts", {}).get("preemptions")) if isinstance(summary.get("counts"), Mapping) else 0
    evicted_blocks = _as_float(cpu.get("evicted_blocks"))
    offload_blocks = _as_float(cpu.get("offload_blocks"))
    truncated_offloads = _as_float(cpu.get("truncated_offloads"))
    offload_operations = _as_float(cpu.get("offload_operations"))
    eviction_block_rate = evicted_blocks / offload_blocks if offload_blocks else 0.0
    truncation_operation_rate = (
        truncated_offloads / offload_operations if offload_operations else 0.0
    )
    eviction_rate = max(eviction_block_rate, truncation_operation_rate)
    return {
        "seed": _as_int(record.get("seed")),
        "capacity": label,
        "grace_cpu_dram_gb": grace,
        "gpu_slice_dram_gb": gpu_slice,
        "rho": rho,
        "oracle": oracle,
        "request_count": len(output_rows) or _as_int(summary.get("counts", {}).get("requests")) if isinstance(summary.get("counts"), Mapping) else len(output_rows),
        "session_count": len(session_ids),
        "new_input_tokens": new_input,
        "scheduled_prefill_tokens": scheduled,
        "actual_extra_ratio": (scheduled - new_input) / new_input if new_input else 0.0,
        "cpu_prefix_hit_rate": cpu_hit_rate,
        "restorable_prefix_blocks": restorable_prefix_blocks,
        "gpu_prefix_hit_blocks": gpu_hit_rows,
        "cpu_eligible_prefix_blocks": cpu_eligible_blocks,
        "cpu_restorable_hit_rate": cpu_restorable_hit_rate,
        "combined_restorable_hit_rate": combined_restorable_hit_rate,
        "cpu_query_blocks": _as_int(cpu.get("query_blocks")),
        "cpu_hit_blocks": _as_int(cpu.get("hit_blocks")),
        "cpu_resident_bytes": _as_int(cpu.get("resident_bytes")),
        "cpu_peak_resident_bytes": _as_int(cpu.get("peak_resident_bytes")),
        "cpu_offload_blocks": _as_int(cpu.get("offload_blocks")),
        "cpu_restore_blocks": _as_int(cpu.get("restore_blocks")),
        "d2h_queue_time_ms": _as_float(cpu.get("d2h_queue_time_ms")),
        "d2h_service_time_ms": _as_float(cpu.get("d2h_service_time_ms")),
        "h2d_queue_time_ms": _as_float(cpu.get("h2d_queue_time_ms")),
        "h2d_service_time_ms": _as_float(cpu.get("h2d_service_time_ms")),
        "source_gpu_hold_time_ms": _as_float(cpu.get("source_gpu_hold_time_ms")),
        "pending_restore_operations": _as_int(cpu.get("pending_restore_operations")),
        "staged_restore_payloads": _as_int(cpu.get("staged_restore_payloads")),
        "active_restore_leases": _as_int(cpu.get("active_restore_leases")),
        "active_offload_reservations": _as_int(cpu.get("active_offload_reservations")),
        "eviction_or_truncation_rate": eviction_rate,
        "eviction_block_rate": eviction_block_rate,
        "truncation_operation_rate": truncation_operation_rate,
        "evicted_blocks": _as_int(cpu.get("evicted_blocks")),
        "truncated_offloads": _as_int(cpu.get("truncated_offloads")),
        "successor_ttft_mean_ms": successor_mean,
        "successor_ttft_p50_ms": successor_p50,
        "successor_ttft_p90_ms": successor_p90,
        "successor_ttft_p99_ms": successor_p99,
        "session_equal_successor_ttft_mean_ms": session_equal_mean,
        "session_equal_successor_ttft_p50_ms": session_equal_p50,
        "session_equal_successor_ttft_p90_ms": session_equal_p90,
        "session_equal_successor_ttft_p99_ms": session_equal_p99,
        "ttft_p99_ms": _as_float(summary.get("latency_ms", {}).get("ttft", {}).get("p99")) if isinstance(summary.get("latency_ms"), Mapping) and isinstance(summary.get("latency_ms", {}).get("ttft"), Mapping) else 0.0,
        "preemptions": preemptions,
        "wall_clock_seconds": _as_float(summary.get("wall_clock_seconds")),
        "run_path": str(record_path.parent),
        "request_rows": output_rows,
    }


def iter_run_records(root: Path) -> Iterable[Path]:
    yield from sorted(root.rglob("run.json"))


def collect_runs(root: Path, *, workload_root: Path = DEFAULT_WORKLOAD_ROOT) -> list[dict[str, object]]:
    return [row for path in iter_run_records(root) if (row := summarize_run(path, workload_root=workload_root)) is not None]


def aggregate_by_capacity(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    groups: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        groups.setdefault(str(row.get("capacity", "")), []).append(row)
    result: list[dict[str, object]] = []
    numeric = lambda label: 0.0 if label == "off" else (float("inf") if label == "oracle_unbounded_cpu" else _as_float(label[3:] if label.lower().startswith("cpu") else label))
    for label, values in sorted(groups.items(), key=lambda item: numeric(item[0])):
        def mean(key: str) -> float:
            numbers = [_as_float(value.get(key)) for value in values]
            return statistics.fmean(numbers) if numbers else 0.0
        first = values[0]
        result.append({
            "capacity": label,
            "grace_cpu_dram_gb": first.get("grace_cpu_dram_gb"),
            "gpu_slice_dram_gb": first.get("gpu_slice_dram_gb"),
            "rho": first.get("rho"),
            "oracle": bool(first.get("oracle")),
            "seed_count": len(values),
            "request_count_mean": mean("request_count"),
            "scheduled_prefill_tokens": mean("scheduled_prefill_tokens"),
            "new_input_tokens": mean("new_input_tokens"),
            "actual_extra_ratio": mean("actual_extra_ratio"),
            "cpu_prefix_hit_rate": mean("cpu_prefix_hit_rate"),
            "restorable_prefix_blocks": mean("restorable_prefix_blocks"),
            "gpu_prefix_hit_blocks": mean("gpu_prefix_hit_blocks"),
            "cpu_eligible_prefix_blocks": mean("cpu_eligible_prefix_blocks"),
            "cpu_restorable_hit_rate": mean("cpu_restorable_hit_rate"),
            "combined_restorable_hit_rate": mean("combined_restorable_hit_rate"),
            "cpu_query_blocks": mean("cpu_query_blocks"),
            "cpu_hit_blocks": mean("cpu_hit_blocks"),
            "cpu_resident_bytes": mean("cpu_resident_bytes"),
            "cpu_peak_resident_bytes": mean("cpu_peak_resident_bytes"),
            "cpu_offload_blocks": mean("cpu_offload_blocks"),
            "cpu_restore_blocks": mean("cpu_restore_blocks"),
            "d2h_queue_time_ms": mean("d2h_queue_time_ms"),
            "d2h_service_time_ms": mean("d2h_service_time_ms"),
            "h2d_queue_time_ms": mean("h2d_queue_time_ms"),
            "h2d_service_time_ms": mean("h2d_service_time_ms"),
            "source_gpu_hold_time_ms": mean("source_gpu_hold_time_ms"),
            "pending_restore_operations": mean("pending_restore_operations"),
            "staged_restore_payloads": mean("staged_restore_payloads"),
            "active_restore_leases": mean("active_restore_leases"),
            "active_offload_reservations": mean("active_offload_reservations"),
            "eviction_or_truncation_rate": mean("eviction_or_truncation_rate"),
            "eviction_block_rate": mean("eviction_block_rate"),
            "truncation_operation_rate": mean("truncation_operation_rate"),
            "evicted_blocks": mean("evicted_blocks"),
            "truncated_offloads": mean("truncated_offloads"),
            "successor_ttft_mean_ms": mean("successor_ttft_mean_ms"),
            "successor_ttft_p50_ms": mean("successor_ttft_p50_ms"),
            "successor_ttft_p90_ms": mean("successor_ttft_p90_ms"),
            "successor_ttft_p99_ms": mean("successor_ttft_p99_ms"),
            "session_equal_successor_ttft_mean_ms": mean("session_equal_successor_ttft_mean_ms"),
            "session_equal_successor_ttft_p50_ms": mean("session_equal_successor_ttft_p50_ms"),
            "session_equal_successor_ttft_p90_ms": mean("session_equal_successor_ttft_p90_ms"),
            "session_equal_successor_ttft_p99_ms": mean("session_equal_successor_ttft_p99_ms"),
            "ttft_p99_ms": mean("ttft_p99_ms"),
            "preemptions": mean("preemptions"),
            "wall_clock_seconds": mean("wall_clock_seconds"),
        })
    return result


def _numeric_capacity(label: str) -> float:
    if label == "off":
        return 0.0
    if label == "oracle_unbounded_cpu":
        return float("inf")
    text = label[3:] if label.lower().startswith("cpu") else label
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _transition(lower: Mapping[str, object], upper: Mapping[str, object]) -> dict[str, object]:
    lower_hit = _as_float(lower.get("cpu_prefix_hit_rate"))
    upper_hit = _as_float(upper.get("cpu_prefix_hit_rate"))
    lower_scheduled = _as_float(lower.get("scheduled_prefill_tokens"))
    upper_scheduled = _as_float(upper.get("scheduled_prefill_tokens"))
    lower_ttft = _as_float(lower.get("successor_ttft_p90_ms"))
    upper_ttft = _as_float(upper.get("successor_ttft_p90_ms"))
    lower_eviction = _as_float(lower.get("eviction_or_truncation_rate"))
    upper_eviction = _as_float(upper.get("eviction_or_truncation_rate"))
    prefill_reduction = (lower_scheduled - upper_scheduled) / lower_scheduled if lower_scheduled > 0 else 0.0
    ttft_improvement = (lower_ttft - upper_ttft) / lower_ttft if lower_ttft > 0 else 0.0
    eviction_delta = abs(upper_eviction - lower_eviction)
    triggers: list[str] = []
    if upper_hit - lower_hit >= THRESHOLDS["cpu_extension_hit_rate_delta"]:
        triggers.append("cpu_extension_hit_rate_delta")
    if prefill_reduction >= THRESHOLDS["scheduled_prefill_reduction"]:
        triggers.append("scheduled_prefill_reduction")
    if ttft_improvement >= THRESHOLDS["successor_ttft_p90_improvement"]:
        triggers.append("successor_ttft_p90_improvement")
    if eviction_delta >= THRESHOLDS["eviction_or_truncation_rate_delta"]:
        triggers.append("eviction_or_truncation_rate_delta")
    return {
        "lower": lower.get("capacity"),
        "upper": upper.get("capacity"),
        "cpu_extension_hit_rate_delta": upper_hit - lower_hit,
        "scheduled_prefill_reduction": prefill_reduction,
        "successor_ttft_p90_improvement": ttft_improvement,
        "eviction_or_truncation_rate_delta": eviction_delta,
        "triggers": triggers,
        "crosses_documented_threshold": bool(triggers),
    }


def propose_fine_grid(c_low: float, c_high: float, *, points: int = 7, unit_gb: float = 8.0) -> list[float]:
    """Return 5--7 inclusive Grace-CPU points aligned to 8 GB."""

    if not math.isfinite(c_low) or not math.isfinite(c_high) or c_high <= c_low:
        raise ValueError("fine-grid bounds must be finite and increasing")
    points = max(5, min(7, int(points)))
    raw = [c_low + (c_high - c_low) * index / (points - 1) for index in range(points)]
    values = [round(value / unit_gb) * unit_gb for value in raw]
    values[0] = c_low
    values[-1] = c_high
    unique: list[float] = []
    for value in values:
        if not unique or value > unique[-1]:
            unique.append(value)
    # Very narrow intervals can collapse after alignment.  Fill between the
    # endpoints using the requested 8 GB granularity, then retain 5--7 points.
    cursor = c_low + unit_gb
    while len(unique) < 5 and cursor < c_high:
        if cursor not in unique:
            unique.insert(-1, cursor)
        cursor += unit_gb
    return unique[:7]


def select_transition(aggregate: Sequence[Mapping[str, object]]) -> dict[str, object]:
    capacities = sorted(
        [row for row in aggregate if not bool(row.get("oracle")) and _numeric_capacity(str(row.get("capacity"))) > 0],
        key=lambda row: _numeric_capacity(str(row.get("capacity"))),
    )
    physical = [
        row
        for row in capacities
        if _numeric_capacity(str(row.get("capacity"))) <= PHYSICAL_MAX_GRACE_CPU_DRAM_GB
    ]
    transitions = [_transition(lower, upper) for lower, upper in zip(physical, physical[1:])]
    selected = next((item for item in transitions if item["crosses_documented_threshold"]), None)
    physical_transition = selected is not None
    analytical_extension_transition = False
    if selected is None:
        extension_transitions = [
            _transition(lower, upper)
            for lower, upper in zip(capacities, capacities[1:])
            if _numeric_capacity(str(upper.get("capacity"))) > PHYSICAL_MAX_GRACE_CPU_DRAM_GB
        ]
        selected = next(
            (item for item in extension_transitions if item["crosses_documented_threshold"]),
            None,
        )
        analytical_extension_transition = selected is not None
        if selected is not None:
            selected["non_physical_analytical_transition"] = True
    if selected is None:
        oracle = next((row for row in aggregate if bool(row.get("oracle"))), None)
        if capacities and oracle is not None:
            selected = _transition(capacities[-1], oracle)
            selected["non_physical_oracle_transition"] = bool(selected["crosses_documented_threshold"])
        elif len(physical) >= 2:
            selected = _transition(physical[-2], physical[-1])
            selected["non_physical_oracle_transition"] = False
        else:
            selected = {"lower": None, "upper": None, "triggers": [], "crosses_documented_threshold": False}
    low_label = selected.get("lower")
    high_label = selected.get("upper")
    low = _numeric_capacity(str(low_label)) if low_label else float("nan")
    high = _numeric_capacity(str(high_label)) if high_label else float("nan")
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        fine_values: list[float] = []
    else:
        fine_values = propose_fine_grid(low, high)
    labels = [f"cpu{int(value) if value.is_integer() else value:g}" for value in fine_values]
    return {
        "thresholds": THRESHOLDS,
        "physical_transition": physical_transition,
        "analytical_extension_transition": analytical_extension_transition,
        "selected_transition": selected,
        "C_low": low_label,
        "C_high": high_label,
        "C_low_grace_cpu_dram_gb": low if math.isfinite(low) else None,
        "C_high_grace_cpu_dram_gb": high if math.isfinite(high) else None,
        "fine_capacity_gb": fine_values,
        "fine_capacity_labels": labels,
        "fine_point_count": len(fine_values),
        "oracle_transition_only": not physical_transition and not analytical_extension_transition and bool(selected.get("non_physical_oracle_transition", False)),
        "interpretation": (
            "a physical coarse transition was found"
            if physical_transition
            else (
                "no physical transition through 500 GB/CPU; a non-physical analytical extension transition was found"
                if analytical_extension_transition
                else "no documented threshold crossed in physical coarse capacities; oracle/analytical extension is reported separately"
            )
        ),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields and key != "request_rows":
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _session_summary(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for run in rows:
        request_rows = run.get("request_rows")
        if not isinstance(request_rows, Sequence):
            continue
        sessions: dict[int, list[Mapping[str, object]]] = {}
        for row in request_rows:
            if isinstance(row, Mapping):
                sessions.setdefault(_as_int(row.get("session_id")), []).append(row)
        for session_id, values in sorted(sessions.items()):
            successors = [row for row in values if _as_int(row.get("session_turn_index")) > 0]
            ttft = [_as_float(row.get("ttft_ms")) for row in successors]
            new_input = sum(_as_int(row.get("new_input_tokens")) for row in values)
            scheduled = sum(_as_int(row.get("scheduled_prefill_tokens")) for row in values)
            result.append({
                "seed": run.get("seed"),
                "capacity": run.get("capacity"),
                "session_id": session_id,
                "num_turns": len(values),
                "final_context_tokens": _as_int(values[-1].get("final_context_tokens")) if values else 0,
                "successor_ttft_mean_ms": statistics.fmean(ttft) if ttft else 0.0,
                "successor_ttft_p90_ms": percentile(ttft, 0.90),
                "new_input_tokens": new_input,
                "scheduled_prefill_tokens": scheduled,
                "actual_extra_ratio": (scheduled - new_input) / new_input if new_input else 0.0,
            })
    return result


def _turn_summary(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], list[Mapping[str, object]]] = {}
    for run in rows:
        request_rows = run.get("request_rows")
        if not isinstance(request_rows, Sequence):
            continue
        for row in request_rows:
            if isinstance(row, Mapping):
                grouped.setdefault(
                    (str(run.get("capacity")), _as_int(row.get("session_turn_index"))),
                    [],
                ).append(row)
    result: list[dict[str, object]] = []
    for (capacity, turn_index), values in sorted(
        grouped.items(), key=lambda item: (_numeric_capacity(item[0][0]), item[0][1])
    ):
        ttft = [_as_float(row.get("ttft_ms")) for row in values]
        cached = [_as_int(row.get("cached_prefill_tokens")) for row in values]
        scheduled = [_as_int(row.get("scheduled_prefill_tokens")) for row in values]
        cached_fraction = [
            c / (c + s) if c + s else 0.0 for c, s in zip(cached, scheduled)
        ]
        result.append({
            "capacity": capacity,
            "turn_index": turn_index,
            "request_count": len(values),
            "ttft_mean_ms": statistics.fmean(ttft) if ttft else 0.0,
            "ttft_p50_ms": percentile(ttft, 0.50),
            "ttft_p90_ms": percentile(ttft, 0.90),
            "ttft_p99_ms": percentile(ttft, 0.99),
            "cached_token_fraction_mean": statistics.fmean(cached_fraction) if cached_fraction else 0.0,
        })
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workload-dir", type=Path, default=DEFAULT_WORKLOAD_ROOT)
    parser.add_argument("--selection-output", type=Path)
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument("--session-summary-csv", type=Path)
    parser.add_argument("--turn-summary-csv", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--print-runs", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.output_root.resolve()
    rows = collect_runs(root, workload_root=args.workload_dir.resolve())
    aggregate = aggregate_by_capacity(rows)
    selection = select_transition(aggregate)
    selection["run_count"] = len(rows)
    selection["aggregate"] = aggregate
    _write_csv(args.summary_csv or (root / "summary.csv"), aggregate)
    _write_csv(args.session_summary_csv or (root / "session_summary.csv"), _session_summary(rows))
    _write_csv(args.turn_summary_csv or (root / "turn_summary.csv"), _turn_summary(rows))
    selection_path = args.selection_output or (root / "fine" / "selection.json")
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    if args.print_runs:
        print(json.dumps(rows, indent=2, default=str))
    print(json.dumps(selection, indent=2))
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
