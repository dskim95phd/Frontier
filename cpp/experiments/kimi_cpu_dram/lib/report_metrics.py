#!/usr/bin/env python3
"""Analyze the ten-hour CPU-DRAM arrival-rate sweep.

The analyzer is intentionally read-only with respect to simulator outputs.  It
expects one completed run per rate under ``--output-root``::

    r0p5/cpu4000gb/r1/{summary.json,requests.csv,workload.normalized.csv}

and (when available) the converter metadata under ``--workload-root``.  It
emits one five-minute time-series CSV, a JSON document containing all series
and stability diagnostics, and a self-contained HTML comparison report.  No
values are synthesized when a source metric is absent; the corresponding
field is ``null`` and the scope is called out in the report.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "outputs"
    / "tracelab_vera_rubin_p8_d16_cpu4tb_arrival_sweep_continuous_10h_20260808"
)
DEFAULT_WORKLOAD_ROOT = (
    REPO_ROOT
    / "outputs"
    / "datasets"
    / "tracelab"
    / "v0.0.2"
    / "frontier"
    / "epoch1000_continuous_10h"
)
DEFAULT_RATES = (0.5, 0.4, 0.3, 0.2)
DEFAULT_BUCKET_SECONDS = 300.0
DEFAULT_PREFILL_LANES = 8
DEFAULT_BUSY_SATURATION_PCT = 90.0
DEFAULT_PERSISTENT_POSITIVE_BIN_FRACTION = 0.60
DEFAULT_MIN_NET_INCREASE = 1.0
DEFAULT_TREND_WINDOW_HOURS = 2.0
DEFAULT_BACKLOG_RELATIVE_GROWTH = 0.01
DEFAULT_BACKLOG_GROWTH_RATE_PER_S = 0.10
DEFAULT_TTFT_SLOPE_MS_PER_HOUR = 100.0
DEFAULT_TTFT_RELATIVE_GROWTH = 0.01
DEFAULT_QUEUE_SLOPE_PER_HOUR = 0.10
RATE_EPSILON = 1e-9


def _finite_float(value: Any, default: float | None = None) -> float | None:
    """Convert a value to a finite float, returning ``default`` on failure."""

    if value is None or value == "":
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _finite_int(value: Any, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    try:
        result = int(float(value))
    except (TypeError, ValueError, OverflowError):
        return default
    return result


def _ratio_percent(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return 100.0 * numerator / denominator


def _percentile(values: Iterable[float], probability: float) -> float | None:
    ordered = sorted(value for value in values if math.isfinite(value))
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


def _safe_name(rate: float) -> str:
    """Format a rate in the directory convention (``r0p5``)."""

    text = f"{rate:g}"
    return "r" + text.replace(".", "p")


def _parse_rate_name(name: str) -> float | None:
    match = re.fullmatch(r"r(\d+(?:p\d+)?|\d*\.\d+)", name)
    if not match:
        return None
    try:
        return float(match.group(1).replace("p", "."))
    except ValueError:
        return None


@dataclass
class Bin:
    """Accumulators for one five-minute interval."""

    start_s: float
    end_s: float
    busy_ms: float = 0.0
    busy_observed: bool = False
    attention_pairs: float = 0.0
    attention_observed: bool = False
    arrival_attention_pairs: float | None = None
    arrival_attention_fallback: float = 0.0
    arrival_count: int = 0
    reconstructed_arrival_count: int = 0
    completion_count: int = 0
    ttft_ms: list[float] = field(default_factory=list)
    tpot_ms: list[float] = field(default_factory=list)
    query_blocks: int = 0
    gpu_hit_blocks: int = 0
    cpu_query_blocks: int = 0
    cpu_hit_blocks: int = 0
    combined_hit_blocks: int = 0
    restore_bytes: int = 0
    offload_bytes: int = 0
    kv_transfer_bytes: int = 0
    kv_transfer_count: int = 0
    waiting_seconds: float = 0.0
    waiting_context_seconds: float = 0.0
    waiting_observations: int = 0
    active_session_seconds: float = 0.0
    active_sessions_start: int = 0
    active_sessions_end: int = 0
    incomplete_final_sessions: int = 0
    cumulative_backlog: int = 0

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


@dataclass
class SessionPlan:
    session_id: int
    root_arrival_s: float | None = None
    think_times: dict[int, float] = field(default_factory=dict)

    @property
    def turn_count(self) -> int:
        return max(self.think_times, default=-1) + 1


def _new_bins(horizon_s: float, bucket_seconds: float) -> list[Bin]:
    count = max(1, int(math.ceil(horizon_s / bucket_seconds)))
    return [
        Bin(
            start_s=index * bucket_seconds,
            end_s=min(horizon_s, (index + 1) * bucket_seconds),
        )
        for index in range(count)
    ]


def _bin_index(time_s: float, bins: Sequence[Bin]) -> int | None:
    if not math.isfinite(time_s) or not bins:
        return None
    if time_s < bins[0].start_s or time_s >= bins[-1].end_s:
        return None
    # Bins are contiguous, and this avoids a second floating-point division
    # when the horizon is a non-integral simulator end time.
    index = int(time_s // (bins[1].start_s - bins[0].start_s)) if len(bins) > 1 else 0
    return min(max(index, 0), len(bins) - 1)


def _add_interval(
    bins: Sequence[Bin],
    start_s: float,
    end_s: float,
    callback: Any,
) -> None:
    """Call ``callback(bin, overlap_seconds)`` for each overlapping bin."""

    if not math.isfinite(start_s) or not math.isfinite(end_s) or end_s <= start_s:
        return
    if len(bins) > 1:
        bucket_seconds = bins[1].start_s - bins[0].start_s
    else:
        bucket_seconds = bins[0].duration_s
    if bucket_seconds <= 0:
        return
    first = max(0, int(start_s // bucket_seconds))
    # Subtract a tiny epsilon so an interval ending exactly on a boundary does
    # not visit the following bucket.
    last = min(len(bins) - 1, int((end_s - RATE_EPSILON) // bucket_seconds))
    for item in bins[first : last + 1]:
        overlap = min(end_s, item.end_s) - max(start_s, item.start_s)
        if overlap > 0:
            callback(item, overlap)


def _load_summary(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"summary must be a JSON object: {path}")
    return value


def _load_metadata(path: Path | None) -> tuple[dict[int, float], dict[str, Any] | None]:
    """Return simulator session root arrivals and the raw metadata document."""

    if path is None or not path.is_file():
        return {}, None
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}, None
    if not isinstance(value, dict):
        return {}, value if isinstance(value, dict) else None
    roots: dict[int, float] = {}
    sessions = value.get("session_map")
    if isinstance(sessions, list):
        for item in sessions:
            if not isinstance(item, Mapping):
                continue
            sid = _finite_int(item.get("numeric_session_id"))
            root = _finite_float(item.get("segment_root_arrival_seconds"))
            if sid is not None and root is not None:
                roots[sid] = root
    return roots, value


def _load_workload(path: Path, metadata_roots: Mapping[int, float]) -> dict[int, SessionPlan]:
    """Load only fields needed for closed-loop arrival reconstruction.

    The normalized workload can be very large.  We retain one small mapping of
    ``turn_index -> think_time`` per simulator session rather than materializing
    the complete CSV in memory.
    """

    plans: dict[int, SessionPlan] = {}
    if not path.is_file():
        return plans
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            sid = _finite_int(row.get("session_id"))
            turn = _finite_int(row.get("session_turn_index"))
            if sid is None or turn is None or turn < 0:
                continue
            plan = plans.setdefault(sid, SessionPlan(session_id=sid))
            root = _finite_float(row.get("session_start_at"))
            if turn == 0 and root is not None:
                plan.root_arrival_s = root
            think = _finite_float(row.get("think_time"), 0.0)
            if think is not None:
                plan.think_times[turn] = max(0.0, think)
    for sid, root in metadata_roots.items():
        plan = plans.setdefault(sid, SessionPlan(session_id=sid))
        # Metadata is the converter's canonical root timing, while the
        # normalized CSV is the canonical turn count and think-time sequence.
        # Prefer metadata whenever it is available; this keeps active-session
        # reconstruction tied to the source workload's segment anchors.
        plan.root_arrival_s = root
    return plans


@dataclass
class Completed:
    arrived_s: float
    completed_s: float
    first_scheduled_s: float | None


def _summary_prefill_buckets(summary: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    source = summary.get("batch_summary_by_cluster_time_bucket", {})
    if not isinstance(source, Mapping):
        return []
    values = source.get("PREFILL")
    if isinstance(values, list):
        return [item for item in values if isinstance(item, Mapping)]
    if isinstance(values, Mapping):
        return [item for item in values.values() if isinstance(item, Mapping)]
    return []


def _load_service_metrics(
    summary: Mapping[str, Any], bins: list[Bin], bucket_seconds: float
) -> None:
    """Project summary source buckets into five-minute bins.

    A source bucket's service and attention work are distributed in proportion
    to overlap.  This preserves totals at the 5-minute boundaries even if the
    simulator's source bucket size is not a divisor of the requested output
    bucket size.
    """

    source_bucket_seconds = _finite_float(summary.get("batch_time_bucket_seconds"), 60.0) or 60.0
    for item in _summary_prefill_buckets(summary):
        start = _finite_float(item.get("start_time_s"))
        if start is None:
            continue
        end = _finite_float(item.get("end_time_s"), start + source_bucket_seconds)
        if end is None or end <= start:
            continue
        busy_ms = _finite_float(item.get("predicted_execution_ms"), 0.0) or 0.0
        attention = _finite_float(item.get("prefill_attention_token_pairs"))
        if attention is None:
            components = item.get("execution_time_components_ms")
            if isinstance(components, Mapping):
                attention = _finite_float(components.get("prefill_attention_token_pairs"))
        _add_interval(
            bins,
            start,
            end,
            lambda target, overlap: _accumulate_service(
                target,
                overlap / (end - start),
                busy_ms,
                attention,
            ),
        )

    # The arrival-time attention map is demand-side data and is kept separate
    # from service-side ``attention_pairs`` when it exists.


def _accumulate_service(
    target: Bin, fraction: float, busy_ms: float, attention: float | None
) -> None:
    target.busy_ms += busy_ms * fraction
    target.busy_observed = True
    if attention is not None:
        target.attention_pairs += attention * fraction
        target.attention_observed = True


def _load_arrival_attention_map(
    summary: Mapping[str, Any], bins: list[Bin], bucket_seconds: float
) -> bool:
    source = summary.get("prefill_attention_token_pairs_by_arrival_time_bucket")
    if not isinstance(source, Mapping):
        return False
    source_seconds = _finite_float(summary.get("batch_time_bucket_seconds"), 60.0) or 60.0
    for raw_index, raw_pairs in source.items():
        index = _finite_int(raw_index)
        pairs = _finite_float(raw_pairs)
        if index is None or pairs is None:
            continue
        start = index * source_seconds
        target_index = int(start // bucket_seconds)
        if 0 <= target_index < len(bins):
            existing = bins[target_index].arrival_attention_pairs or 0.0
            bins[target_index].arrival_attention_pairs = existing + pairs
    return True


def _attention_pairs(num_prefill: int, cached: int) -> int:
    if num_prefill < 0 or cached < 0:
        return 0
    cached = min(cached, num_prefill)
    return (
        num_prefill * (num_prefill + 1) - cached * (cached + 1)
    ) // 2


def _read_requests(
    path: Path,
    bins: list[Bin],
    completed_by_session: dict[int, list[Completed]],
    *,
    horizon_s: float,
) -> None:
    if not path.is_file():
        return
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            sid = _finite_int(row.get("session_id"))
            arrived = _finite_float(row.get("arrived_at_s"))
            completed = _finite_float(row.get("completed_at_s"))
            if sid is None or arrived is None or completed is None or arrived < 0:
                continue
            first_scheduled = _finite_float(row.get("first_scheduled_at_s"))
            completed_by_session.setdefault(sid, []).append(
                Completed(arrived, completed, first_scheduled)
            )
            arrival_index = _bin_index(arrived, bins)
            if arrival_index is not None:
                target = bins[arrival_index]
                target.arrival_count += 1
                target.query_blocks += _finite_int(row.get("prefix_cache_query_blocks"), 0) or 0
                target.gpu_hit_blocks += _finite_int(row.get("gpu_prefix_hit_blocks"), 0) or 0
                target.cpu_query_blocks += _finite_int(row.get("cpu_prefix_query_blocks"), 0) or 0
                target.cpu_hit_blocks += _finite_int(row.get("cpu_prefix_hit_blocks"), 0) or 0
                target.combined_hit_blocks += _finite_int(row.get("prefix_cache_hit_blocks"), 0) or 0
                target.restore_bytes += _finite_int(row.get("cpu_restore_bytes"), 0) or 0
                target.offload_bytes += _finite_int(row.get("cpu_offload_bytes"), 0) or 0
                target.kv_transfer_bytes += _finite_int(row.get("kv_cache_transfer_size_bytes"), 0) or 0
                target.kv_transfer_count += 1
                ttft = _finite_float(row.get("ttft_ms"))
                if ttft is not None:
                    target.ttft_ms.append(ttft)
                decode_tokens = _finite_int(row.get("num_decode_tokens"), 0) or 0
                first_token = _finite_float(row.get("first_token_completed_at_s"))
                if decode_tokens > 1 and first_token is not None:
                    tail_ms = (completed - first_token) * 1000.0
                    if math.isfinite(tail_ms) and tail_ms >= 0:
                        target.tpot_ms.append(tail_ms / (decode_tokens - 1))
                # A request's input work is only a fallback.  The summary map
                # is preferred because it includes arrivals not present in the
                # completion-only requests.csv.
                target.arrival_attention_fallback += _attention_pairs(
                    _finite_int(row.get("num_prefill_tokens"), 0) or 0,
                    _finite_int(row.get("cached_prefill_tokens"), 0) or 0,
                )
            if first_scheduled is not None and first_scheduled >= arrived:
                _add_interval(
                    bins,
                    arrived,
                    min(first_scheduled, horizon_s),
                    lambda target, overlap: _accumulate_waiting(
                        target,
                        overlap,
                        _finite_int(row.get("num_prefill_tokens"), 0) or 0,
                    ),
                )
            completion_index = _bin_index(completed, bins)
            if completion_index is not None:
                bins[completion_index].completion_count += 1


def _accumulate_waiting(target: Bin, overlap: float, context_tokens: int) -> None:
    target.waiting_seconds += overlap
    target.waiting_context_seconds += overlap * max(0, context_tokens)
    target.waiting_observations += 1


def _reconstruct_arrivals(
    plans: Mapping[int, SessionPlan],
    completed_by_session: Mapping[int, list[Completed]],
    bins: list[Bin],
    *,
    horizon_s: float,
) -> tuple[dict[int, float], dict[int, float], int]:
    """Add closed-loop arrivals not represented by completion-only requests.

    Returns ``(root_arrivals, final_completion_times, incomplete_session_count)``.
    The returned maps are also used to reconstruct active-session occupancy.
    """

    root_arrivals: dict[int, float] = {}
    final_completion_times: dict[int, float] = {}
    incomplete_sessions = 0
    for sid, plan in plans.items():
        root = plan.root_arrival_s
        if root is None or root < 0 or root >= horizon_s:
            continue
        root_arrivals[sid] = root
        records = sorted(completed_by_session.get(sid, ()), key=lambda item: item.arrived_s)
        expected_turns = plan.turn_count
        completed_count = len(records)
        if expected_turns > 0 and completed_count >= expected_turns:
            final_completion_times[sid] = max(item.completed_s for item in records[:expected_turns])
        elif expected_turns > 0:
            incomplete_sessions += 1

        # Completion rows already contributed their measured arrival events.
        # Infer only the first missing turn (and continue while a predecessor
        # is known), which prevents inventing requests after an unobserved turn.
        if expected_turns <= 0 or completed_count >= expected_turns:
            continue
        next_turn = completed_count
        if next_turn == 0:
            next_arrival = root
        elif next_turn - 1 < len(records):
            think = plan.think_times.get(next_turn, 0.0)
            next_arrival = records[next_turn - 1].completed_s + think
        else:
            next_arrival = float("inf")
        if math.isfinite(next_arrival) and 0 <= next_arrival < horizon_s:
            index = _bin_index(next_arrival, bins)
            if index is not None:
                bins[index].arrival_count += 1
                bins[index].reconstructed_arrival_count += 1
    return root_arrivals, final_completion_times, incomplete_sessions


def _active_session_metrics(
    bins: list[Bin],
    root_arrivals: Mapping[int, float],
    final_completion_times: Mapping[int, float],
    incomplete_session_count: int,
    *,
    horizon_s: float,
) -> None:
    intervals: list[tuple[float, float]] = []
    incomplete_starts: list[float] = []
    for sid, start in root_arrivals.items():
        end = final_completion_times.get(sid, horizon_s)
        end = min(max(end, start), horizon_s)
        if end > start:
            intervals.append((start, end))
        if sid not in final_completion_times:
            incomplete_starts.append(start)
    for target in bins:
        target.active_sessions_start = sum(start <= target.start_s < end for start, end in intervals)
        # Intervals are [start, end).  At the exact simulation horizon an
        # unfinished session remains active for the left-limit used by the
        # final bucket; at interior boundaries a session completing exactly at
        # the boundary is no longer active in the following bucket.
        if abs(target.end_s - horizon_s) <= RATE_EPSILON:
            target.active_sessions_end = sum(start < target.end_s <= end for start, end in intervals)
        else:
            target.active_sessions_end = sum(start < target.end_s < end for start, end in intervals)
        target.active_session_seconds = sum(
            max(0.0, min(end, target.end_s) - max(start, target.start_s))
            for start, end in intervals
        )
        target.incomplete_final_sessions = sum(
            start < target.end_s for start in incomplete_starts
        )
    # ``incomplete_session_count`` is deliberately not used to fill bins: the
    # interval reconstruction above is the source of truth.  Keeping the
    # argument documents that the count is based on final-turn completion.
    _ = incomplete_session_count


def _apply_arrival_attention_fallback(bins: list[Bin]) -> None:
    for target in bins:
        if target.arrival_attention_pairs is None and target.arrival_attention_fallback:
            target.arrival_attention_pairs = target.arrival_attention_fallback


def _slope_per_hour(values: Sequence[float | None], bins: Sequence[Bin]) -> float | None:
    pairs = [
        ((bins[index].end_s / 3600.0), float(value))
        for index, value in enumerate(values)
        if value is not None and math.isfinite(float(value))
    ]
    if len(pairs) < 2:
        return None
    xs = [pair[0] for pair in pairs]
    ys = [pair[1] for pair in pairs]
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    return sum((x - x_mean) * (y - y_mean) for x, y in pairs) / denominator if denominator else 0.0


def _serialize_number(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _serialize_number(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_number(item) for item in value]
    return value


def _build_bin_row(target: Bin, *, prefill_lanes: int, bucket_seconds: float) -> dict[str, Any]:
    duration = target.duration_s
    query = target.query_blocks
    combined = target.combined_hit_blocks
    tpot_mean = statistics.fmean(target.tpot_ms) if target.tpot_ms else None
    busy_pct = (
        100.0 * target.busy_ms / (duration * 1000.0 * prefill_lanes)
        if duration > 0 and target.busy_observed and prefill_lanes > 0
        else None
    )
    return {
        "start_time_s": target.start_s,
        "end_time_s": target.end_s,
        "duration_s": duration,
        "pool_busy_pct": busy_pct,
        "prefill_attention_token_pairs": target.attention_pairs if target.attention_observed else None,
        "prefill_attention_token_pairs_trillion": (
            target.attention_pairs / 1e12 if target.attention_observed else None
        ),
        "arrival_attention_token_pairs": target.arrival_attention_pairs,
        "arrival_attention_token_pairs_trillion": (
            target.arrival_attention_pairs / 1e12
            if target.arrival_attention_pairs is not None
            else None
        ),
        "request_arrivals": target.arrival_count,
        "reconstructed_arrivals": target.reconstructed_arrival_count,
        "request_arrivals_per_s": target.arrival_count / duration if duration else None,
        "request_completions": target.completion_count,
        "request_completions_per_s": target.completion_count / duration if duration else None,
        "backlog_delta_requests": target.arrival_count - target.completion_count,
        "backlog_delta_per_s": (
            (target.arrival_count - target.completion_count) / duration if duration else None
        ),
        "ttft_count": len(target.ttft_ms),
        "ttft_mean_ms": statistics.fmean(target.ttft_ms) if target.ttft_ms else None,
        "ttft_p90_ms": _percentile(target.ttft_ms, 0.90),
        "tpot_count": len(target.tpot_ms),
        "tpot_mean_ms": tpot_mean,
        "tpot_p90_ms": _percentile(target.tpot_ms, 0.90),
        "prefix_cache_query_blocks": query,
        "gpu_prefix_hit_blocks": target.gpu_hit_blocks,
        "cpu_prefix_query_blocks": target.cpu_query_blocks,
        "cpu_prefix_hit_blocks": target.cpu_hit_blocks,
        "prefix_cache_hit_blocks": combined,
        "prefix_cache_miss_blocks": max(0, query - combined),
        "gpu_hit_pct": _ratio_percent(target.gpu_hit_blocks, query),
        "cpu_conditional_hit_pct": _ratio_percent(target.cpu_hit_blocks, target.cpu_query_blocks),
        "combined_hit_pct": _ratio_percent(combined, query),
        "combined_miss_pct": _ratio_percent(max(0, query - combined), query),
        "cpu_restore_bytes": target.restore_bytes,
        "cpu_offload_bytes": target.offload_bytes,
        "transfer_bytes": target.restore_bytes + target.offload_bytes,
        "cpu_restore_bw_gbps": target.restore_bytes / duration / 1e9 if duration else None,
        "cpu_offload_bw_gbps": target.offload_bytes / duration / 1e9 if duration else None,
        "transfer_bw_gbps": (target.restore_bytes + target.offload_bytes) / duration / 1e9 if duration else None,
        "kv_transfer_bw_gbps": target.kv_transfer_bytes / duration / 1e9 if duration else None,
        "kv_transfer_count": target.kv_transfer_count,
        "transfer_bw_scope": "request_arrival_bucket_cpu_restore_plus_offload",
        "prefill_waiting_queue_count_time_weighted": target.waiting_seconds / duration if duration else None,
        # Mean full context of requests while they occupy the PREFILL waiting
        # queue.  Divide by queued-request seconds, not wall-clock bucket
        # seconds; the latter would be the aggregate queued-context load.
        "prefill_waiting_context_tokens_time_weighted": (
            target.waiting_context_seconds / target.waiting_seconds
            if target.waiting_seconds > 0
            else None
        ),
        "prefill_waiting_observations": target.waiting_observations,
        "active_sessions_time_weighted": target.active_session_seconds / duration if duration else None,
        "active_sessions_start": target.active_sessions_start,
        "active_sessions_end": target.active_sessions_end,
        "incomplete_final_sessions": target.incomplete_final_sessions,
        "cumulative_backlog_requests": target.cumulative_backlog,
        "waiting_queue_scope": "completed_requests_only",
        "bucket_seconds": bucket_seconds,
    }


def _overall_summary(rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]) -> dict[str, Any]:
    latency = summary.get("latency_ms", {})
    throughput = summary.get("throughput", {})
    counts = summary.get("counts", {})
    def nested(name: str, key: str) -> float | None:
        value = latency.get(name, {}) if isinstance(latency, Mapping) else {}
        return _finite_float(value.get(key)) if isinstance(value, Mapping) else None
    return {
        "summary_request_count": _finite_int(counts.get("requests")) if isinstance(counts, Mapping) else None,
        "simulation_window_seconds": _finite_float(summary.get("simulation_window_seconds")),
        "summary_ttft_mean_ms": nested("ttft", "mean"),
        "summary_ttft_p90_ms": nested("ttft", "p90"),
        "summary_tpot_mean_ms": nested("tpot", "mean"),
        "summary_tpot_p90_ms": nested("tpot", "p90"),
        "requests_per_second": _finite_float(throughput.get("requests_per_second")) if isinstance(throughput, Mapping) else None,
        "prompt_tokens_per_second": _finite_float(throughput.get("prompt_tokens_per_second")) if isinstance(throughput, Mapping) else None,
        "rows_request_arrivals": sum(int(row["request_arrivals"]) for row in rows),
        "rows_request_completions": sum(int(row["request_completions"]) for row in rows),
    }


def _trend_metric(
    rows: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    key: str,
    *,
    positive_fraction: float,
    slope_threshold: float = 0.0,
    relative_growth_threshold: float = 0.0,
    min_net_increase: float = 0.0,
    observation_key: str | None = None,
) -> dict[str, Any]:
    selected: list[Mapping[str, Any]] = []
    for index in indices:
        row = rows[index]
        if observation_key is not None and not (_finite_int(row.get(observation_key), 0) or 0) > 0:
            continue
        value = _finite_float(row.get(key))
        if value is not None:
            selected.append(row)
    values = [float(row[key]) for row in selected]
    if not values:
        return {
            "key": key,
            "observed_bins": 0,
            "start": None,
            "end": None,
            "net_change": None,
            "relative_growth": None,
            "growth_rate_per_s": None,
            "slope_per_hour": None,
            "positive_step_fraction": None,
            "material_positive": None,
        }
    proxies = [type("Proxy", (), {"end_s": float(row["end_time_s"])})() for row in selected]
    deltas = [right - left for left, right in zip(values, values[1:])]
    positive = sum(delta > RATE_EPSILON for delta in deltas) / len(deltas) if deltas else 0.0
    net = values[-1] - values[0]
    duration_s = max(1.0, float(selected[-1]["end_time_s"]) - float(selected[0]["end_time_s"]))
    relative = net / max(abs(values[0]), 1.0)
    slope = _slope_per_hour(values, proxies)
    material = bool(
        len(values) >= 2
        and slope is not None
        and slope > slope_threshold
        and net > max(min_net_increase, abs(values[0]) * relative_growth_threshold)
        and positive >= positive_fraction
    )
    return {
        "key": key,
        "observed_bins": len(values),
        "start": values[0],
        "end": values[-1],
        "net_change": net,
        "relative_growth": relative,
        "growth_rate_per_s": net / duration_s,
        "slope_per_hour": slope,
        "positive_step_fraction": positive,
        "material_positive": material,
        # Kept explicit in JSON so a reader can audit the classification.
        "slope_threshold": slope_threshold,
        "relative_growth_threshold": relative_growth_threshold,
        "min_net_increase": min_net_increase,
    }


def _stability_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    hours: float,
    busy_saturation_pct: float,
    positive_fraction: float,
    min_net_increase: float,
) -> dict[str, Any]:
    if not rows:
        return {"window_hours": hours, "classification": "unknown", "reason": "no time buckets"}
    start_s = float(rows[-1]["end_time_s"]) - hours * 3600.0
    # The simulator horizon can finish a few milliseconds before the nominal
    # ten-hour boundary.  Selecting by ``end_time > start_s`` would then pull
    # in the preceding five-minute bucket for only that millisecond-scale
    # overlap and count the whole bucket.  Use bucket starts so the final
    # one-/two-hour windows contain the intended 12/24 complete buckets.
    indices = [
        index
        for index, row in enumerate(rows)
        if float(row["start_time_s"]) >= start_s
    ]
    selected = [rows[index] for index in indices]
    busy_values = [row.get("pool_busy_pct") for row in selected]
    weights = [float(row.get("duration_s") or 0.0) for row in selected]
    weighted = [
        (float(value), weight)
        for value, weight in zip(busy_values, weights)
        if value is not None and math.isfinite(float(value)) and weight > 0
    ]
    busy_mean = (
        sum(value * weight for value, weight in weighted) / sum(weight for _, weight in weighted)
        if weighted
        else None
    )
    arrivals = sum(int(row.get("request_arrivals") or 0) for row in selected)
    # Backlog growth is considered material only when it exceeds both the
    # relative 1% allowance and 0.1 request/s.  This avoids flagging tiny
    # stochastic drifts as overload while still exposing absolute growth.
    backlog = _trend_metric(
        rows,
        indices,
        "cumulative_backlog_requests",
        positive_fraction=positive_fraction,
        slope_threshold=0.0,
        relative_growth_threshold=DEFAULT_BACKLOG_RELATIVE_GROWTH,
        min_net_increase=max(min_net_increase, arrivals * DEFAULT_BACKLOG_RELATIVE_GROWTH, hours * 3600.0 * DEFAULT_BACKLOG_GROWTH_RATE_PER_S),
    )
    backlog["window_arrivals"] = arrivals
    backlog["relative_to_window_arrivals"] = (
        backlog["net_change"] / max(arrivals, 1)
        if backlog.get("net_change") is not None
        else None
    )
    backlog["material_threshold_requests"] = max(
        min_net_increase,
        arrivals * DEFAULT_BACKLOG_RELATIVE_GROWTH,
        hours * 3600.0 * DEFAULT_BACKLOG_GROWTH_RATE_PER_S,
    )
    ttft = _trend_metric(
        rows,
        indices,
        "ttft_mean_ms",
        positive_fraction=positive_fraction,
        slope_threshold=DEFAULT_TTFT_SLOPE_MS_PER_HOUR,
        relative_growth_threshold=DEFAULT_TTFT_RELATIVE_GROWTH,
        min_net_increase=min_net_increase,
    )
    queue = _trend_metric(
        rows,
        indices,
        "prefill_waiting_queue_count_time_weighted",
        positive_fraction=positive_fraction,
        slope_threshold=DEFAULT_QUEUE_SLOPE_PER_HOUR,
        relative_growth_threshold=DEFAULT_BACKLOG_RELATIVE_GROWTH,
        min_net_increase=min_net_increase,
        observation_key="prefill_waiting_observations",
    )
    active = _trend_metric(
        rows,
        indices,
        "active_sessions_end",
        positive_fraction=positive_fraction,
        slope_threshold=0.0,
        relative_growth_threshold=0.0,
        min_net_increase=min_net_increase,
    )
    busy_saturated = busy_mean is not None and busy_mean >= busy_saturation_pct
    # Queue observations stop at each request's first schedule and are
    # right-censored at the simulation horizon.  Queue growth therefore cannot
    # independently label a case overloaded; it remains a secondary diagnostic
    # alongside the primary reconstructed-backlog and TTFT signals.
    primary_positive = any(
        trend.get("material_positive") is True for trend in (backlog, ttft)
    )
    # Active sessions are a secondary diagnostic: a warm-up population can
    # grow without indicating overload, so active-session growth alone never
    # changes the class.
    if busy_mean is None or backlog.get("observed_bins", 0) < 2:
        classification = "unknown"
    elif primary_positive:
        classification = "overloaded"
    elif busy_saturated:
        classification = "saturated-stable"
    else:
        classification = "underloaded-stable"
    reasons: list[str] = []
    if busy_mean is None:
        reasons.append("missing PREFILL busy data")
    if backlog.get("material_positive"):
        reasons.append("reconstructed backlog growth is material")
    if ttft.get("material_positive"):
        reasons.append("TTFT trend is materially positive")
    if queue.get("material_positive"):
        reasons.append("PREFILL waiting-queue trend is materially positive (secondary, right-censored)")
    if active.get("material_positive"):
        reasons.append("active-session trend is positive (secondary only)")
    if not reasons and classification != "unknown":
        reasons.append("backlog/TTFT/queue trends are near-flat under stated thresholds")
    return {
        "window_hours": hours,
        "window_start_s": start_s,
        "window_end_s": float(rows[-1]["end_time_s"]),
        "final_window_pool_busy_pct_time_weighted": busy_mean,
        "busy_below_saturation": (not busy_saturated) if busy_mean is not None else None,
        "saturation_state": (
            "under_saturation" if busy_mean is not None and not busy_saturated else
            "at_or_over_saturation" if busy_mean is not None else "unknown"
        ),
        "arrivals_in_window": arrivals,
        "backlog": backlog,
        "ttft": ttft,
        "queue": queue,
        "active_sessions": active,
        "queue_scope": "completed_requests_only_right_censored_near_horizon",
        "classification": classification,
        "reason": "; ".join(reasons),
    }


def _analyze_case(
    *,
    rate: float,
    case_dir: Path,
    metadata_path: Path | None,
    bucket_seconds: float,
    prefill_lanes: int,
    busy_saturation_pct: float,
    positive_fraction: float,
    min_net_increase: float,
) -> dict[str, Any]:
    summary_path = case_dir / "summary.json"
    requests_path = case_dir / "requests.csv"
    workload_path = case_dir / "workload.normalized.csv"
    summary = _load_summary(summary_path)
    horizon = _finite_float(summary.get("simulation_window_seconds"))
    if horizon is None or horizon <= 0:
        raise ValueError(f"summary has no positive simulation_window_seconds: {summary_path}")
    bins = _new_bins(horizon, bucket_seconds)
    _load_service_metrics(summary, bins, bucket_seconds)
    metadata_roots, metadata = _load_metadata(metadata_path)
    plans = _load_workload(workload_path, metadata_roots)
    completed_by_session: dict[int, list[Completed]] = {}
    _read_requests(requests_path, bins, completed_by_session, horizon_s=horizon)
    arrival_attention_available = _load_arrival_attention_map(summary, bins, bucket_seconds)
    if not arrival_attention_available:
        _apply_arrival_attention_fallback(bins)
    root_arrivals, final_completion_times, incomplete_count = _reconstruct_arrivals(
        plans,
        completed_by_session,
        bins,
        horizon_s=horizon,
    )
    # Cumulative request backlog uses all measured + reconstructed arrivals.
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
    # Keep only the fields used for trend calculations here.  The raw rows are
    # also the CSV/JSON output and remain source-derived throughout.
    for row in rows:
        row["rate"] = rate
        row["rate_label"] = _safe_name(rate)
    final_hour = _stability_summary(
        rows,
        hours=1.0,
        busy_saturation_pct=busy_saturation_pct,
        positive_fraction=positive_fraction,
        min_net_increase=min_net_increase,
    )
    last_two_hours = _stability_summary(
        rows,
        hours=DEFAULT_TREND_WINDOW_HOURS,
        busy_saturation_pct=busy_saturation_pct,
        positive_fraction=positive_fraction,
        min_net_increase=min_net_increase,
    )
    return {
        "rate": rate,
        "rate_label": _safe_name(rate),
        "run_dir": str(case_dir),
        "metadata_path": str(metadata_path) if metadata_path else None,
        "metadata_available": metadata is not None,
        "metadata_root_count": len(metadata_roots),
        "workload_path": str(workload_path),
        "horizon_s": horizon,
        "prefill_lanes": prefill_lanes,
        "workload_session_count": len(plans),
        "completed_session_count": len(completed_by_session),
        "reconstructed_incomplete_final_session_count": incomplete_count,
        "arrival_attention_scope": "summary_arrival_time_map" if arrival_attention_available else "completed_requests_fallback",
        "overall": _overall_summary(rows, summary),
        "stability": {"final_hour": final_hour, "last_2_hours": last_two_hours},
        "bins": rows,
    }


def _flatten_csv_rows(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        rows.extend(dict(row) for row in case.get("bins", []))
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _svg_chart(cases: Sequence[Mapping[str, Any]], metric: str, title: str, unit: str = "") -> str:
    width, height = 960, 280
    left, right, top, bottom = 64, 18, 30, 42
    plot_w, plot_h = width - left - right, height - top - bottom
    all_values = [
        float(row[metric])
        for case in cases
        for row in case.get("bins", [])
        if row.get(metric) is not None and math.isfinite(float(row[metric]))
    ]
    if not all_values:
        return f"<h3>{html.escape(title)}</h3><p class='muted'>No source data.</p>"
    y_min = min(0.0, min(all_values))
    y_max = max(all_values)
    if math.isclose(y_min, y_max):
        y_max = y_min + 1.0
    else:
        y_max *= 1.08
    n = max((len(case.get("bins", [])) for case in cases), default=1)
    colors = ["#2563eb", "#ea580c", "#059669", "#7c3aed", "#dc2626", "#0891b2"]
    def sx(index: int) -> float:
        return left + (index / max(1, n - 1)) * plot_w
    def sy(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h
    pieces = [f"<h3>{html.escape(title)}</h3><svg viewBox='0 0 {width} {height}' role='img' aria-label='{html.escape(title)}'>"]
    for tick in range(5):
        value = y_min + (y_max - y_min) * tick / 4
        y = sy(value)
        pieces.append(f"<line x1='{left}' y1='{y:.1f}' x2='{width-right}' y2='{y:.1f}' class='grid'/>")
        pieces.append(f"<text x='{left-8}' y='{y+4:.1f}' text-anchor='end' class='tick'>{value:.3g}{html.escape(unit)}</text>")
    pieces.append(f"<line x1='{left}' y1='{top+plot_h}' x2='{width-right}' y2='{top+plot_h}' class='axis'/>")
    pieces.append(f"<line x1='{left}' y1='{top}' x2='{left}' y2='{top+plot_h}' class='axis'/>")
    for index in range(0, n, max(1, n // 10)):
        x = sx(index)
        end_time = None
        if cases and index < len(cases[0].get("bins", [])):
            end_time = _finite_float(cases[0]["bins"][index].get("end_time_s"))
        label = f"{end_time / 3600.0:.1f}h" if end_time is not None else f"{index}"
        pieces.append(f"<text x='{x:.1f}' y='{height-14}' text-anchor='middle' class='tick'>{label}</text>")
    for case_index, case in enumerate(cases):
        path: list[str] = []
        for index, row in enumerate(case.get("bins", [])):
            value = row.get(metric)
            if value is None or not math.isfinite(float(value)):
                continue
            command = "M" if not path or (index and case["bins"][index-1].get(metric) is None) else "L"
            path.append(f"{command}{sx(index):.1f},{sy(float(value)):.1f}")
        if path:
            pieces.append(f"<path d='{' '.join(path)}' fill='none' stroke='{colors[case_index % len(colors)]}' stroke-width='2'/>")
    legend = " ".join(
        f"<span><i style='background:{colors[index % len(colors)]}'></i>{html.escape(str(case.get('rate_label', case.get('rate'))))}</span>"
        for index, case in enumerate(cases)
    )
    pieces.append("</svg><div class='legend'>" + legend + "</div>")
    return "".join(pieces)


def _html_report(document: Mapping[str, Any]) -> str:
    cases = document.get("cases", [])
    thresholds = document.get("heuristic_thresholds", {})
    summary_rows: list[str] = []
    for case in cases:
        overall = case.get("overall", {})
        final = case.get("stability", {}).get("final_hour", {})
        two = case.get("stability", {}).get("last_2_hours", {})
        backlog = final.get("backlog", {})
        active = final.get("active_sessions", {})
        summary_rows.append(
            "<tr>"
            f"<td>{html.escape(str(case.get('rate_label')))}</td>"
            f"<td>{_fmt(final.get('classification'))}</td>"
            f"<td>{_fmt(final.get('saturation_state'))}</td>"
            f"<td>{_fmt(final.get('final_window_pool_busy_pct_time_weighted'))}%</td>"
            f"<td>{_fmt(backlog.get('net_change'))}</td>"
            f"<td>{_fmt(active.get('net_change'))}</td>"
            f"<td>{_fmt(overall.get('summary_ttft_mean_ms'))} ms</td>"
            f"<td>{_fmt(overall.get('summary_ttft_p90_ms'))} ms</td>"
            f"<td>{_fmt(overall.get('summary_tpot_mean_ms'))} ms</td>"
            f"<td>{html.escape(str(two.get('classification')))}</td>"
            "</tr>"
        )
    charts = "".join(
        f"<section class='chart'>{_svg_chart(cases, metric, title, unit)}</section>"
        for metric, title, unit in (
            ("pool_busy_pct", "PREFILL pool busy", "%"),
            ("prefill_attention_token_pairs_trillion", "PREFILL attention token pairs", " T"),
            ("request_arrivals_per_s", "Request arrivals", " /s"),
            ("request_completions_per_s", "Request completions", " /s"),
            ("cumulative_backlog_requests", "Cumulative request backlog", ""),
            ("active_sessions_time_weighted", "Active sessions", ""),
            ("ttft_mean_ms", "TTFT mean", " ms"),
            ("ttft_p90_ms", "TTFT p90", " ms"),
            ("tpot_mean_ms", "TPOT mean", " ms"),
            ("gpu_hit_pct", "GPU prefix hit", "%"),
            ("cpu_conditional_hit_pct", "Conditional CPU prefix hit", "%"),
            ("combined_hit_pct", "Combined prefix hit", "%"),
            ("transfer_bw_gbps", "CPU transfer bandwidth", " GB/s"),
            ("prefill_waiting_queue_count_time_weighted", "PREFILL waiting queue count", ""),
            ("prefill_waiting_context_tokens_time_weighted", "Waiting context length", " tokens"),
        )
    )
    details: list[str] = []
    for case in cases:
        headers = [
            "end_time_s", "pool_busy_pct", "request_arrivals", "request_completions", "cumulative_backlog_requests",
            "active_sessions_time_weighted", "ttft_mean_ms", "ttft_p90_ms", "tpot_mean_ms", "tpot_p90_ms",
            "gpu_hit_pct", "cpu_conditional_hit_pct", "combined_hit_pct", "combined_miss_pct", "transfer_bw_gbps",
            "prefill_waiting_queue_count_time_weighted", "prefill_waiting_context_tokens_time_weighted",
        ]
        head = "".join(f"<th>{html.escape(key)}</th>" for key in headers)
        body = "".join(
            "<tr>" + "".join(f"<td>{_fmt(row.get(key))}</td>" for key in headers) + "</tr>"
            for row in case.get("bins", [])
        )
        details.append(
            f"<details><summary>{html.escape(str(case.get('rate_label')))} five-minute data</summary>"
            f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div></details>"
        )
    return f"""<!doctype html>
<html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>
<title>10-hour CPU4TB arrival-rate sweep</title>
<style>
:root {{ color-scheme: light dark; --bg:#f7f8fb; --panel:#fff; --fg:#172033; --muted:#5b6475; --border:#d8dde8; --grid:#e7eaf0; }}
@media (prefers-color-scheme:dark) {{ :root {{ --bg:#0f1219; --panel:#171b24; --fg:#edf1f7; --muted:#a6afbf; --border:#31394a; --grid:#252c39; }} }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:var(--bg); color:var(--fg); font:14px/1.45 system-ui,sans-serif; }} main {{ max-width:1500px; margin:auto; padding:26px; }}
h1 {{ margin:0 0 6px; }} h2 {{ margin:28px 0 12px; }} h3 {{ margin:0 0 8px; font-size:15px; }} .muted {{ color:var(--muted); }}
.table-wrap {{ overflow-x:auto; background:var(--panel); border:1px solid var(--border); border-radius:10px; }} table {{ width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }} th,td {{ padding:7px 9px; border-bottom:1px solid var(--border); text-align:right; white-space:nowrap; }} th:first-child,td:first-child {{ text-align:left; }} th {{ color:var(--muted); font-size:12px; }}
.charts {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; }} .chart {{ min-width:0; padding:13px; background:var(--panel); border:1px solid var(--border); border-radius:10px; }} svg {{ width:100%; height:auto; display:block; }} .axis {{ stroke:var(--muted); stroke-width:1; }} .grid {{ stroke:var(--grid); stroke-width:1; }} .tick {{ fill:var(--muted); font-size:10px; }} .legend {{ display:flex; gap:12px; flex-wrap:wrap; color:var(--muted); font-size:12px; }} .legend i {{ display:inline-block; width:13px; height:3px; margin-right:4px; vertical-align:middle; }}
details {{ margin-top:14px; }} summary {{ cursor:pointer; font-weight:600; }} .notes {{ padding:14px 16px; background:var(--panel); border-left:4px solid #2563eb; border-radius:4px; }} code {{ font-family:ui-monospace,Consolas,monospace; }}
@media (max-width:900px) {{ main{{padding:16px}} .charts{{grid-template-columns:1fr}} }}
</style></head><body><main>
<h1>10-hour CPU4TB arrival-rate sweep</h1>
<p class='muted'>Rates: {html.escape(', '.join(str(case.get('rate')) for case in cases))}. Five-minute bins. Source-derived metrics only.</p>
<div class='notes'><strong>Heuristic classification</strong><p><code>underloaded-stable</code> means final-window PREFILL busy is below <code>{thresholds.get('busy_saturation_pct')}%</code> and the primary reconstructed-backlog/TTFT trends are not materially positive. <code>saturated-stable</code> means busy is at/above that saturation line but those primary signals remain near-flat. <code>overloaded</code> requires a materially positive primary backlog or TTFT trend. Backlog growth is material only when it exceeds both <code>{thresholds.get('backlog_relative_growth')}</code> of window arrivals and <code>{thresholds.get('backlog_growth_rate_per_s')} req/s</code>, with positive changes in at least <code>{thresholds.get('positive_bin_fraction')}</code> of consecutive bins. TTFT uses a <code>{thresholds.get('ttft_slope_ms_per_hour')} ms/hour</code> slope and <code>{thresholds.get('ttft_relative_growth')}</code> relative-growth threshold. Queue (<code>{thresholds.get('queue_slope_per_hour')} entries/hour</code>) and active-session slopes are secondary diagnostics only; neither can trigger overload by itself. These are diagnostic thresholds, not capacity guarantees.</p><p>Waiting queue/context metrics use completed requests with finite <code>arrived_at_s</code> and <code>first_scheduled_at_s</code>. Pending requests have no fabricated schedule time, so this signal is right-censored near the 10-hour horizon and is never based on only the final bin. CPU transfer bandwidth is binned by request arrival and sums restore+offload bytes.</p></div>
<h2>Comparison summary</h2><div class='table-wrap'><table><thead><tr><th>rate</th><th>final-hour class</th><th>saturation state</th><th>final-hour busy</th><th>backlog Δ</th><th>active Δ</th><th>TTFT mean</th><th>TTFT p90</th><th>TPOT mean</th><th>last-2-hour class</th></tr></thead><tbody>{''.join(summary_rows)}</tbody></table></div>
<h2>Five-minute time series</h2><div class='charts'>{charts}</div>
<h2>Raw five-minute tables</h2>{''.join(details)}
<p class='muted'>Generated by the shared Kimi CPU-DRAM report pipeline. JSON and CSV beside this report contain the complete machine-readable output.</p>
</main></body></html>"""


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workload-root", type=Path, default=DEFAULT_WORKLOAD_ROOT)
    parser.add_argument("--rates", type=str, default=",".join(str(rate) for rate in DEFAULT_RATES), help="Comma-separated rates; default: 0.5,0.4,0.3,0.2")
    parser.add_argument("--bucket-seconds", type=float, default=DEFAULT_BUCKET_SECONDS)
    parser.add_argument("--prefill-lanes", type=int, default=DEFAULT_PREFILL_LANES, help="PREFILL DP lanes (P8 defaults to 8)")
    parser.add_argument("--busy-saturation-pct", type=float, default=DEFAULT_BUSY_SATURATION_PCT)
    parser.add_argument("--persistent-positive-bin-fraction", type=float, default=DEFAULT_PERSISTENT_POSITIVE_BIN_FRACTION)
    parser.add_argument("--min-net-increase", type=float, default=DEFAULT_MIN_NET_INCREASE)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-html", type=Path)
    return parser


def _parse_rates(raw: str) -> list[float]:
    values: list[float] = []
    for token in raw.split(","):
        value = _finite_float(token.strip())
        if value is None or value <= 0:
            raise ValueError(f"rates must be finite and positive: {token!r}")
        if not any(math.isclose(value, existing, rel_tol=0.0, abs_tol=RATE_EPSILON) for existing in values):
            values.append(value)
    if not values:
        raise ValueError("at least one rate is required")
    return values


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not math.isfinite(args.bucket_seconds) or args.bucket_seconds <= 0:
        raise SystemExit("--bucket-seconds must be finite and positive")
    if args.prefill_lanes <= 0:
        raise SystemExit("--prefill-lanes must be positive")
    if not math.isfinite(args.busy_saturation_pct) or args.busy_saturation_pct <= 0:
        raise SystemExit("--busy-saturation-pct must be finite and positive")
    if not 0 < args.persistent_positive_bin_fraction <= 1:
        raise SystemExit("--persistent-positive-bin-fraction must be in (0, 1]")
    if args.min_net_increase < 0 or not math.isfinite(args.min_net_increase):
        raise SystemExit("--min-net-increase must be finite and nonnegative")
    rates = _parse_rates(args.rates)
    root = args.output_root.resolve()
    workload_root = args.workload_root.resolve()
    cases: list[dict[str, Any]] = []
    missing: list[str] = []
    for rate in rates:
        rate_dir = root / _safe_name(rate)
        case_dir = rate_dir / "cpu4000gb" / "r1"
        metadata_path = workload_root / _safe_name(rate) / "seed_20260803_metadata.json"
        required = [case_dir / "summary.json", case_dir / "requests.csv", case_dir / "workload.normalized.csv"]
        if not all(path.is_file() for path in required):
            missing.append(str(case_dir))
            continue
        cases.append(
            _analyze_case(
                rate=rate,
                case_dir=case_dir,
                metadata_path=metadata_path if metadata_path.is_file() else None,
                bucket_seconds=args.bucket_seconds,
                prefill_lanes=args.prefill_lanes,
                busy_saturation_pct=args.busy_saturation_pct,
                positive_fraction=args.persistent_positive_bin_fraction,
                min_net_increase=args.min_net_increase,
            )
        )
    if not cases:
        raise SystemExit(
            "no completed cases found; expected summary.json, requests.csv, and "
            f"workload.normalized.csv under {root}. Missing: {', '.join(missing)}"
        )
    document: dict[str, Any] = {
        "schema_version": 1,
        "output_root": str(root),
        "workload_root": str(workload_root),
        "bucket_seconds": args.bucket_seconds,
        "requested_rates": rates,
        "missing_cases": missing,
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
            "classification_basis": "window busy saturation plus reconstructed backlog/TTFT/queue trends; active sessions secondary",
        },
        "cases": cases,
    }
    document = _serialize_number(document)
    csv_path = (args.output_csv or root / "cpu4tb_rate_sweep_10h_5min.csv").resolve()
    json_path = (args.output_json or root / "cpu4tb_rate_sweep_10h.json").resolve()
    html_path = (args.output_html or root / "cpu4tb_rate_sweep_10h.html").resolve()
    _write_csv(csv_path, _flatten_csv_rows(cases))
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(_html_report(document), encoding="utf-8")
    print(json.dumps({"cases": len(cases), "csv": str(csv_path), "json": str(json_path), "html": str(html_path), "missing_cases": missing}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
