#!/usr/bin/env python3
"""Build a per-PREFILL-GPU CPU-DRAM sizing report from 10-hour sweeps."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "outputs"
    / "tracelab_vera_rubin_p8_d16_per_gpu_cpu_dram_scaling_20260811"
)
DEFAULT_REPORTS = (
    REPO_ROOT
    / "outputs"
    / "tracelab_vera_rubin_p8_d16_cpu_capacity_knee_rebuild_10h_r0p30_20260811"
    / "r0p3_capacity_sweep_10h.json",
    REPO_ROOT
    / "outputs"
    / "tracelab_vera_rubin_p8_d16_cpu_capacity_knee_10h_r0p325_20260810"
    / "r0p325_capacity_sweep_10h.json",
    REPO_ROOT
    / "outputs"
    / "tracelab_vera_rubin_p8_d16_cpu_capacity_knee_rebuild_10h_r0p35_20260811"
    / "r0p35_capacity_sweep_10h.json",
    REPO_ROOT
    / "outputs"
    / "tracelab_vera_rubin_p8_d16_cpu_capacity_knee_10h_r0p375_20260810"
    / "r0p375_capacity_sweep_10h.json",
    REPO_ROOT
    / "outputs"
    / "tracelab_vera_rubin_p8_d16_cpu_capacity_knee_rebuild_10h_r0p40_20260811"
    / "r0p4_capacity_sweep_10h.json",
    REPO_ROOT
    / "outputs"
    / "tracelab_vera_rubin_p8_d16_cpu_capacity_knee_10h_r0p425_20260810"
    / "r0p425_capacity_sweep_10h.json",
    REPO_ROOT
    / "outputs"
    / "tracelab_vera_rubin_p8_d16_cpu_capacity_knee_rebuild_10h_r0p45_20260811"
    / "r0p45_capacity_sweep_10h.json",
)
PREFILL_GPUS = 8
GPUS_PER_CPU_POOL = 2
FINAL_WINDOW_SECONDS = 3_600.0
DEFAULT_TTFT_P90_SLO_MS = 5_000.0
DEFAULT_MIN_COMPLETION_RATIO = 0.99
DEFAULT_MAX_WAITING_QUEUE = 1_000.0
DEFAULT_MAX_BACKLOG_REQUESTS = 1_000.0
STABLE_CLASSES = {"underloaded-stable", "saturated-stable"}


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _duration(row: Mapping[str, Any]) -> float:
    start = _finite(row.get("start_time_s"))
    end = _finite(row.get("end_time_s"))
    if start is None or end is None:
        return 0.0
    return max(0.0, end - start)


def _weighted_mean(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    numerator = 0.0
    denominator = 0.0
    for row in rows:
        value = _finite(row.get(key))
        weight = _duration(row)
        if value is None or weight <= 0:
            continue
        numerator += value * weight
        denominator += weight
    return numerator / denominator if denominator else None


def _maximum(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [_finite(row.get(key)) for row in rows]
    finite = [value for value in values if value is not None]
    return max(finite) if finite else None


def _sum(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return sum(_finite(row.get(key)) or 0.0 for row in rows)


def _final_rows(case: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = [row for row in case.get("bins", []) if isinstance(row, Mapping)]
    if not rows:
        return []
    horizon = max(_finite(row.get("end_time_s")) or 0.0 for row in rows)
    start = horizon - FINAL_WINDOW_SECONDS
    return [
        row
        for row in rows
        if (_finite(row.get("start_time_s")) or -math.inf) >= start - 1e-6
    ]


def extract_points(
    report_paths: Iterable[Path],
    *,
    ttft_p90_slo_ms: float = DEFAULT_TTFT_P90_SLO_MS,
    min_completion_ratio: float = DEFAULT_MIN_COMPLETION_RATIO,
    max_waiting_queue: float = DEFAULT_MAX_WAITING_QUEUE,
    max_backlog_requests: float = DEFAULT_MAX_BACKLOG_REQUESTS,
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for path in report_paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        rate = float(document["simulation_rate_per_s"])
        for case in document.get("cases", []):
            final = _final_rows(case)
            if not final:
                continue
            duration_s = sum(_duration(row) for row in final)
            arrivals = _sum(final, "request_arrivals")
            completions = _sum(final, "request_completions")
            ratio = completions / arrivals if arrivals > 0 else None
            classification = str(
                case.get("stability", {})
                .get("final_hour", {})
                .get("classification", "unknown")
            )
            last_two_hour_classification = str(
                case.get("stability", {})
                .get("last_2_hours", {})
                .get("classification", "unknown")
            )
            ttft_p90_max = _maximum(final, "ttft_p90_ms")
            source_stable = (
                classification in STABLE_CLASSES
                and last_two_hour_classification in STABLE_CLASSES
            )
            ttft_pass = (
                ttft_p90_max is not None and ttft_p90_max <= ttft_p90_slo_ms
            )
            completion_pass = ratio is not None and ratio >= min_completion_ratio
            queue_max = _maximum(final, "prefill_waiting_queue_count_time_weighted")
            backlog_max = _maximum(final, "cumulative_backlog_requests")
            queue_pass = queue_max is not None and queue_max <= max_waiting_queue
            backlog_pass = (
                backlog_max is not None and backlog_max <= max_backlog_requests
            )
            verified = (
                source_stable
                and ttft_pass
                and completion_pass
                and queue_pass
                and backlog_pass
            )
            if verified:
                status = "verified"
            elif ttft_pass and completion_pass:
                status = "trend-flagged"
            elif not ttft_pass:
                status = "ttft-fail"
            else:
                status = "throughput-fail"

            physical_pool_gb = int(case["capacity_gb"])
            active = _weighted_mean(final, "active_sessions_time_weighted")
            queue = _weighted_mean(
                final, "prefill_waiting_queue_count_time_weighted"
            )
            point = {
                "session_injection_rate_per_s": rate,
                "physical_cpu_pool_gb_per_two_prefill_gpus": physical_pool_gb,
                "cpu_dram_gb_per_prefill_gpu": physical_pool_gb
                / GPUS_PER_CPU_POOL,
                "aggregate_cpu_dram_gb_for_p8": physical_pool_gb
                * PREFILL_GPUS
                / GPUS_PER_CPU_POOL,
                "active_sessions_per_prefill_gpu": (
                    active / PREFILL_GPUS if active is not None else None
                ),
                "arrival_requests_per_s_per_prefill_gpu": (
                    arrivals / duration_s / PREFILL_GPUS if duration_s else None
                ),
                "completion_requests_per_s_per_prefill_gpu": (
                    completions / duration_s / PREFILL_GPUS if duration_s else None
                ),
                "completion_to_arrival_ratio": ratio,
                "prefill_waiting_requests_per_gpu": (
                    queue / PREFILL_GPUS if queue is not None else None
                ),
                "prefill_waiting_queue_max": queue_max,
                "backlog_requests_max": backlog_max,
                "final_backlog_requests_per_gpu": (
                    (_finite(final[-1].get("cumulative_backlog_requests")) or 0.0)
                    / PREFILL_GPUS
                ),
                "ttft_p90_max_ms": ttft_p90_max,
                "ttft_mean_ms": _weighted_mean(final, "ttft_mean_ms"),
                "tpot_p90_max_ms": _maximum(final, "tpot_p90_ms"),
                "combined_hit_pct": _weighted_mean(final, "combined_hit_pct"),
                "prefill_busy_pct": _weighted_mean(final, "pool_busy_pct"),
                "source_classification": classification,
                "source_last_2h_classification": last_two_hour_classification,
                "source_ttft_slope_ms_per_hour": _finite(
                    case.get("stability", {})
                    .get("final_hour", {})
                    .get("ttft", {})
                    .get("slope_per_hour")
                ),
                "source_stable": source_stable,
                "ttft_slo_pass": ttft_pass,
                "completion_ratio_pass": completion_pass,
                "waiting_queue_pass": queue_pass,
                "backlog_pass": backlog_pass,
                "verified_sustainable": verified,
                "report_status": status,
                "source_report": str(path.resolve()),
            }
            points.append(point)
    return sorted(
        points,
        key=lambda point: (
            point["session_injection_rate_per_s"],
            point["cpu_dram_gb_per_prefill_gpu"],
        ),
    )


def build_load_envelope(
    points: Sequence[Mapping[str, Any]],
    *,
    verified_key: str = "verified_sustainable",
) -> list[dict[str, Any]]:
    envelope: list[dict[str, Any]] = []
    rates = sorted({float(point["session_injection_rate_per_s"]) for point in points})
    for rate in rates:
        candidates = [
            point
            for point in points
            if float(point["session_injection_rate_per_s"]) == rate
            and point.get(verified_key)
        ]
        if not candidates:
            envelope.append(
                {
                    "session_injection_rate_per_s": rate,
                    "verified": False,
                }
            )
            continue
        selected = min(
            candidates, key=lambda point: float(point["cpu_dram_gb_per_prefill_gpu"])
        )
        lower_points = [
            point
            for point in points
            if float(point["session_injection_rate_per_s"]) == rate
            and float(point["cpu_dram_gb_per_prefill_gpu"])
            < float(selected["cpu_dram_gb_per_prefill_gpu"])
        ]
        lower = max(
            (
                float(point["cpu_dram_gb_per_prefill_gpu"])
                for point in lower_points
                if not point.get(verified_key)
            ),
            default=None,
        )
        higher_failures = [
            float(point["cpu_dram_gb_per_prefill_gpu"])
            for point in points
            if float(point["session_injection_rate_per_s"]) == rate
            and float(point["cpu_dram_gb_per_prefill_gpu"])
            > float(selected["cpu_dram_gb_per_prefill_gpu"])
            and not point.get(verified_key)
        ]
        envelope.append(
            {
                "session_injection_rate_per_s": rate,
                "verified": True,
                "lower_failed_dram_gb_per_gpu": lower,
                "verified_dram_gb_per_gpu": selected[
                    "cpu_dram_gb_per_prefill_gpu"
                ],
                "physical_pool_gb_per_two_prefill_gpus": selected[
                    "physical_cpu_pool_gb_per_two_prefill_gpus"
                ],
                "active_sessions_per_prefill_gpu": selected[
                    "active_sessions_per_prefill_gpu"
                ],
                "completion_requests_per_s_per_prefill_gpu": selected[
                    "completion_requests_per_s_per_prefill_gpu"
                ],
                "ttft_p90_max_ms": selected["ttft_p90_max_ms"],
                "tpot_p90_max_ms": selected["tpot_p90_max_ms"],
                "combined_hit_pct": selected["combined_hit_pct"],
                "higher_capacity_nonverified_gb_per_gpu": higher_failures,
            }
        )
    return envelope


def build_capacity_summary(points: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    capacities = sorted({float(point["cpu_dram_gb_per_prefill_gpu"]) for point in points})
    for capacity in capacities:
        candidates = [
            point
            for point in points
            if float(point["cpu_dram_gb_per_prefill_gpu"]) == capacity
            and point.get("verified_sustainable")
        ]
        best = (
            max(candidates, key=lambda point: float(point["session_injection_rate_per_s"]))
            if candidates
            else None
        )
        summary.append(
            {
                "cpu_dram_gb_per_prefill_gpu": capacity,
                "physical_pool_gb_per_two_prefill_gpus": capacity
                * GPUS_PER_CPU_POOL,
                "maximum_tested_verified_rate_per_s": (
                    best["session_injection_rate_per_s"] if best else None
                ),
                "active_sessions_per_prefill_gpu": (
                    best["active_sessions_per_prefill_gpu"] if best else None
                ),
                "completion_requests_per_s_per_prefill_gpu": (
                    best["completion_requests_per_s_per_prefill_gpu"]
                    if best
                    else None
                ),
                "ttft_p90_max_ms": best["ttft_p90_max_ms"] if best else None,
                "is_lower_bound": best is not None,
            }
        )
    return summary


def _fmt(value: Any, digits: int = 2) -> str:
    number = _finite(value)
    if number is None:
        return "—"
    if abs(number) >= 1_000:
        return f"{number:,.0f}"
    return f"{number:,.{digits}f}"


def _line_chart(
    rows: Sequence[Mapping[str, Any]],
    *,
    y_key: str,
    title: str,
    y_label: str,
) -> str:
    valid = [
        row
        for row in rows
        if row.get("verified")
        and _finite(row.get("verified_dram_gb_per_gpu")) is not None
        and _finite(row.get(y_key)) is not None
    ]
    if not valid:
        return ""
    width, height = 650, 330
    left, right, top, bottom = 70, 24, 32, 56
    plot_w, plot_h = width - left - right, height - top - bottom
    x_values = [float(row["verified_dram_gb_per_gpu"]) for row in valid]
    y_values = [float(row[y_key]) for row in valid]
    x_max = max(x_values) * 1.08
    y_max = max(y_values) * 1.15

    def sx(value: float) -> float:
        return left + value / x_max * plot_w

    def sy(value: float) -> float:
        return top + (y_max - value) / y_max * plot_h

    pieces = [
        f"<section class='plot'><h3>{html.escape(title)}</h3>",
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{html.escape(title)}'>",
        f"<title>{html.escape(title)}</title>",
        f"<desc>Verified ten-hour operating points by CPU DRAM per PREFILL GPU.</desc>",
    ]
    for tick in range(5):
        x_value = x_max * tick / 4
        x = sx(x_value)
        y_value = y_max * tick / 4
        y = sy(y_value)
        pieces.append(
            f"<line x1='{left}' y1='{y:.1f}' x2='{width-right}' y2='{y:.1f}' class='grid'/>"
        )
        pieces.append(
            f"<text x='{left-8}' y='{y+4:.1f}' text-anchor='end' class='tick'>{_fmt(y_value, 1)}</text>"
        )
        pieces.append(
            f"<text x='{x:.1f}' y='{height-22}' text-anchor='middle' class='tick'>{_fmt(x_value, 0)}</text>"
        )
    path = " ".join(
        ("M" if index == 0 else "L")
        + f"{sx(x):.1f},{sy(y):.1f}"
        for index, (x, y) in enumerate(zip(x_values, y_values))
    )
    pieces.append(
        f"<path d='{path}' fill='none' stroke='#2563eb' stroke-width='2.5'/>"
    )
    for row, x, y in zip(valid, x_values, y_values):
        rate = float(row["session_injection_rate_per_s"])
        pieces.append(
            f"<circle cx='{sx(x):.1f}' cy='{sy(y):.1f}' r='5' fill='#2563eb'><title>rate {rate:.2f}/s; {_fmt(x,0)} GB/GPU; {_fmt(y,2)} {html.escape(y_label)}</title></circle>"
        )
        pieces.append(
            f"<text x='{sx(x)+8:.1f}' y='{sy(y)-8:.1f}' class='label'>{rate:.2f}/s</text>"
        )
    pieces.extend(
        [
            f"<line x1='{left}' y1='{top+plot_h}' x2='{width-right}' y2='{top+plot_h}' class='axis'/>",
            f"<line x1='{left}' y1='{top}' x2='{left}' y2='{top+plot_h}' class='axis'/>",
            f"<text x='{left+plot_w/2:.1f}' y='{height-4}' text-anchor='middle' class='axis-label'>CPU DRAM per PREFILL GPU (GB)</text>",
            f"<text transform='translate(18 {top+plot_h/2:.1f}) rotate(-90)' text-anchor='middle' class='axis-label'>{html.escape(y_label)}</text>",
            "</svg></section>",
        ]
    )
    return "".join(pieces)


def _status_label(point: Mapping[str, Any]) -> str:
    return {
        "verified": "Verified",
        "trend-flagged": "QoS pass / trend flag",
        "ttft-fail": "TTFT fail",
        "throughput-fail": "Throughput fail",
    }.get(str(point.get("report_status")), "Unknown")


def _html_report(document: Mapping[str, Any]) -> str:
    points = list(document["points"])
    envelope = list(document["load_envelope"])
    capacity_summary = list(document["capacity_summary"])
    ttft_slo_s = float(document["criteria"]["ttft_p90_slo_ms"]) / 1_000.0

    envelope_rows = []
    for row in envelope:
        if not row.get("verified"):
            envelope_rows.append(
                f"<tr><td>{float(row['session_injection_rate_per_s']):.2f}</td><td colspan='8'>No verified point in the tested grid</td></tr>"
            )
            continue
        lower = row.get("lower_failed_dram_gb_per_gpu")
        bracket = (
            f"({_fmt(lower,0)}, {_fmt(row['verified_dram_gb_per_gpu'],0)}]"
            if lower is not None
            else f"≤ {_fmt(row['verified_dram_gb_per_gpu'],0)}"
        )
        anomalies = row.get("higher_capacity_nonverified_gb_per_gpu", [])
        envelope_rows.append(
            "<tr>"
            f"<td>{float(row['session_injection_rate_per_s']):.2f}</td>"
            f"<td>{_fmt(row['active_sessions_per_prefill_gpu'],1)}</td>"
            f"<td>{_fmt(row['completion_requests_per_s_per_prefill_gpu'],3)}</td>"
            f"<td>{bracket}</td>"
            f"<td>{_fmt(row['physical_pool_gb_per_two_prefill_gpus'],0)}</td>"
            f"<td>{_fmt(row['ttft_p90_max_ms']/1000.0,2)} s</td>"
            f"<td>{_fmt(row['tpot_p90_max_ms'],2)} ms</td>"
            f"<td>{_fmt(row['combined_hit_pct'],1)}%</td>"
            f"<td>{html.escape(', '.join(_fmt(v,0) for v in anomalies) or 'none')}</td>"
            "</tr>"
        )

    capacity_rows = []
    for row in capacity_summary:
        capacity_rows.append(
            "<tr>"
            f"<td>{_fmt(row['cpu_dram_gb_per_prefill_gpu'],0)}</td>"
            f"<td>{_fmt(row['physical_pool_gb_per_two_prefill_gpus'],0)}</td>"
            f"<td>{_fmt(row['maximum_tested_verified_rate_per_s'],2)}</td>"
            f"<td>{_fmt(row['active_sessions_per_prefill_gpu'],1)}</td>"
            f"<td>{_fmt(row['completion_requests_per_s_per_prefill_gpu'],3)}</td>"
            f"<td>{_fmt((_finite(row['ttft_p90_max_ms']) or math.nan)/1000.0,2)} s</td>"
            "</tr>"
        )

    rates = sorted({float(point["session_injection_rate_per_s"]) for point in points})
    capacities = sorted({float(point["cpu_dram_gb_per_prefill_gpu"]) for point in points})
    by_key = {
        (float(point["session_injection_rate_per_s"]), float(point["cpu_dram_gb_per_prefill_gpu"])): point
        for point in points
    }
    matrix_rows = []
    for rate in rates:
        cells = [f"<th>{rate:.2f}/s</th>"]
        for capacity in capacities:
            point = by_key.get((rate, capacity))
            if point is None:
                cells.append("<td class='status missing'>Not tested</td>")
                continue
            status = str(point["report_status"])
            cells.append(
                f"<td class='status {status}'><strong>{_status_label(point)}</strong><br>"
                f"{_fmt(point['active_sessions_per_prefill_gpu'],1)} sess/GPU<br>"
                f"{_fmt(point['completion_requests_per_s_per_prefill_gpu'],3)} req/s/GPU<br>"
                f"TTFT p90 max {_fmt(point['ttft_p90_max_ms']/1000.0,2)} s</td>"
            )
        matrix_rows.append("<tr>" + "".join(cells) + "</tr>")

    point_headers = [
        "rate",
        "GB/GPU",
        "active/GPU",
        "completed req/s/GPU",
        "completion/arrival",
        "TTFT p90 max",
        "TPOT p90 max",
        "hit",
        "busy",
        "source class (1h / 2h)",
        "report status",
    ]
    point_rows = []
    for point in points:
        values = [
            f"{float(point['session_injection_rate_per_s']):.2f}",
            _fmt(point["cpu_dram_gb_per_prefill_gpu"], 0),
            _fmt(point["active_sessions_per_prefill_gpu"], 1),
            _fmt(point["completion_requests_per_s_per_prefill_gpu"], 3),
            _fmt(100.0 * point["completion_to_arrival_ratio"], 2) + "%",
            _fmt(point["ttft_p90_max_ms"] / 1000.0, 2) + " s",
            _fmt(point["tpot_p90_max_ms"], 2) + " ms",
            _fmt(point["combined_hit_pct"], 1) + "%",
            _fmt(point["prefill_busy_pct"], 1) + "%",
            f"{point['source_classification']} / {point['source_last_2h_classification']}",
            _status_label(point),
        ]
        point_rows.append(
            "<tr>" + "".join(f"<td>{html.escape(value)}</td>" for value in values) + "</tr>"
        )

    charts = _line_chart(
        envelope,
        y_key="active_sessions_per_prefill_gpu",
        title="Verified active-session density",
        y_label="Active sessions per PREFILL GPU",
    ) + _line_chart(
        envelope,
        y_key="completion_requests_per_s_per_prefill_gpu",
        title="Verified request throughput density",
        y_label="Completed requests/s per PREFILL GPU",
    )

    verified_rows = [row for row in envelope if row.get("verified")]
    strongest = max(
        verified_rows,
        key=lambda row: float(row["session_injection_rate_per_s"]),
        default=None,
    )
    if strongest is None:
        strongest_html = "No rate has a verified operating point in this grid."
    else:
        rate = float(strongest["session_injection_rate_per_s"])
        dram = float(strongest["verified_dram_gb_per_gpu"])
        active = float(strongest["active_sessions_per_prefill_gpu"])
        throughput = float(strongest["completion_requests_per_s_per_prefill_gpu"])
        strongest_html = (
            f"The highest tested rate with a passing point is {rate:.3f} session/s: "
            f"about {_fmt(active,1)} active sessions/GPU and {_fmt(throughput,3)} "
            f"completed req/s/GPU at {_fmt(dram,0)} GB/GPU. On P8/D16 this is "
            f"about {_fmt(active * PREFILL_GPUS,0)} active sessions, "
            f"{_fmt(throughput * PREFILL_GPUS,2)} completed req/s, and "
            f"{_fmt(dram * PREFILL_GPUS / 1_000.0,2)} TB aggregate CPU DRAM."
        )
    exception_rows = [
        row
        for row in verified_rows
        if row.get("higher_capacity_nonverified_gb_per_gpu")
    ]
    if exception_rows:
        exception_text = "; ".join(
            f"{float(row['session_injection_rate_per_s']):.3f}/s: "
            + ", ".join(
                f"{float(value):g} GB/GPU"
                for value in row["higher_capacity_nonverified_gb_per_gpu"]
            )
            for row in exception_rows
        )
        exception_html = (
            "Higher-capacity non-verified observations remain in the sparse grid ("
            + html.escape(exception_text)
            + "), so the envelope is an observed operating boundary rather than a monotonic capacity law."
        )
    else:
        exception_html = (
            "No higher-capacity exception appears among the tested points, but untested cells prevent a universal monotonic-capacity claim."
        )
    rate_min, rate_max = min(rates), max(rates)

    return f"""<!doctype html>
<html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Per-GPU CPU-DRAM sizing from TraceLab P8/D16</title>
<style>
:root {{ color-scheme:light dark; --bg:#f7f8fb; --panel:#fff; --fg:#172033; --muted:#5b6475; --border:#d8dde8; --grid:#e7eaf0; --ok:#dcfce7; --ok-fg:#14532d; --warn:#fef3c7; --warn-fg:#78350f; --bad:#fee2e2; --bad-fg:#7f1d1d; }}
@media (prefers-color-scheme:dark) {{ :root {{ --bg:#0f1219; --panel:#171b24; --fg:#edf1f7; --muted:#a6afbf; --border:#31394a; --grid:#252c39; --ok:#173e2a; --ok-fg:#bbf7d0; --warn:#493716; --warn-fg:#fde68a; --bad:#4b2024; --bad-fg:#fecaca; }} }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:var(--bg); color:var(--fg); font:14px/1.48 system-ui,sans-serif; }} main {{ max-width:1560px; margin:auto; padding:28px; }}
h1 {{ margin:0 0 6px; }} h2 {{ margin:30px 0 12px; }} h3 {{ margin:0 0 8px; }} .muted {{ color:var(--muted); }} .note {{ padding:14px 16px; border-left:4px solid #2563eb; background:var(--panel); }}
.charts {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; }} .plot {{ min-width:0; background:var(--panel); padding:14px; border:1px solid var(--border); border-radius:10px; }} svg {{ width:100%; height:auto; display:block; }} .grid {{ stroke:var(--grid); }} .axis {{ stroke:var(--muted); }} .tick {{ fill:var(--muted); font-size:10px; }} .axis-label,.label {{ fill:var(--fg); font-size:11px; }}
.table-wrap {{ overflow-x:auto; background:var(--panel); border:1px solid var(--border); border-radius:10px; }} table {{ width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }} th,td {{ padding:8px 9px; border-bottom:1px solid var(--border); text-align:right; white-space:nowrap; }} th {{ color:var(--muted); font-size:12px; }} th:first-child,td:first-child {{ text-align:left; }}
.matrix td {{ white-space:normal; min-width:145px; text-align:left; vertical-align:top; }} .status.verified {{ background:var(--ok); color:var(--ok-fg); }} .status.trend-flagged {{ background:var(--warn); color:var(--warn-fg); }} .status.ttft-fail,.status.throughput-fail {{ background:var(--bad); color:var(--bad-fg); }} details {{ margin-top:14px; }} summary {{ cursor:pointer; font-weight:600; }} code {{ font-family:ui-monospace,Consolas,monospace; }}
@media (max-width:900px) {{ main {{ padding:16px; }} .charts {{ grid-template-columns:1fr; }} }}
</style></head><body><main>
<h1>Per-GPU CPU-DRAM sizing from the 10-hour TraceLab sweeps</h1>
<p class='muted'>Kimi K2 · P8/D16 · {len(rates)} session-injection rates from {rate_min:.3f} to {rate_max:.3f}/s · 8 PREFILL GPUs · 16 DECODE GPUs</p>
<div class='note'><strong>Interpretation.</strong> The report normalizes the measured system by PREFILL GPU. One physical CPU-DRAM pool serves two PREFILL GPUs, so a 4 TB experiment label means 2 TB/GPU and 16 TB aggregate CPU DRAM across P8. Proportional extrapolation assumes PREFILL GPUs, DECODE GPUs, CPU pools, memory bandwidth, and routing are all scaled together while preserving the TraceLab session mix.</div>
<div class='note'><strong>Fresh-run provenance.</strong> The original 0.30/0.35/0.40/0.45 source artifacts were unavailable after the pre-K3 checkout, so their boundary cases were rerun with the restored official TraceLab v0.0.2 database, seed 20260803, a 1,000-source-session sample repeated for 20 epochs, and the same pre-K3 simulator used for the new intermediate rates. Treat this report as one internally consistent fresh-run dataset; do not splice individual rows into the older archived report.</div>

<h2>Verified sizing envelope</h2>
<p>“Verified” uses one unified service criterion: stable source classifications in both the final one-hour and two-hour windows, at least {100*float(document['criteria']['min_completion_ratio']):.0f}% completions/arrivals, final-hour waiting queue and backlog no greater than 1,000 requests, and every five-minute TTFT p90 in the final hour at or below {ttft_slo_s:g} seconds. The memory value is the smallest passing point among the tested capacities, not an exact analytical minimum.</p>
<div class='charts'>{charts}</div>
<div class='table-wrap'><table><thead><tr><th>Session injection</th><th>Active sessions/GPU</th><th>Completed req/s/GPU</th><th>Observed DRAM bracket (GB/GPU)</th><th>Passing pool GB/2 GPUs</th><th>TTFT p90 max</th><th>TPOT p90 max</th><th>Combined hit</th><th>Higher-capacity exceptions (GB/GPU)</th></tr></thead><tbody>{''.join(envelope_rows)}</tbody></table></div>

<h2>How to scale the claim</h2>
<p>For a target tier, multiply the per-GPU active-session and request-rate values by the number of PREFILL GPUs. Provision <code>DRAM/GPU × PREFILL GPUs</code> aggregate CPU memory, implemented as one pool of <code>2 × DRAM/GPU</code> for each two-GPU pair. Keep the measured P8:D16 ratio, so doubling users from a verified point means P16/D32 and twice the aggregate CPU DRAM.</p>
<p>{strongest_html}</p>

<h2>Maximum tested verified load by installed DRAM</h2>
<p>These are lower bounds: a listed rate means the configuration passed at that tested rate; it does not locate the exact failure rate between sampled rows.</p>
<div class='table-wrap'><table><thead><tr><th>DRAM GB/GPU</th><th>Pool GB/2 GPUs</th><th>Max tested verified injection</th><th>Active sessions/GPU</th><th>Completed req/s/GPU</th><th>TTFT p90 max</th></tr></thead><tbody>{''.join(capacity_rows)}</tbody></table></div>

<h2>All observed operating points</h2>
<p>Amber cells met the {ttft_slo_s:g}-second TTFT and throughput guardrails but were flagged by the source trend classifier. Red cells missed an operational guardrail. The non-monotonic amber/red exceptions at higher DRAM are why this report states tested lower bounds rather than a universal monotone capacity law.</p>
<div class='table-wrap'><table class='matrix'><thead><tr><th>Injection</th>{''.join(f'<th>{_fmt(capacity,0)} GB/GPU</th>' for capacity in capacities)}</tr></thead><tbody>{''.join(matrix_rows)}</tbody></table></div>

<details><summary>Exact final-hour measurements</summary><div class='table-wrap'><table><thead><tr>{''.join(f'<th>{html.escape(header)}</th>' for header in point_headers)}</tr></thead><tbody>{''.join(point_rows)}</tbody></table></div></details>

<h2>Claim boundary and follow-up</h2>
<p>Use the envelope as a trace-conditioned sizing statement, not a model-independent constant. {exception_html}</p>
<p>Capacity comparisons within each injection rate use an identical workload. Across rates, the source sample and seed are shared, but arrival spacing and the final-hour session subsequence differ. Historical run records do not contain an executable hash or Git revision, so exact binary identity across every old case cannot be certified; future publication runs should persist both.</p>
<p class='muted'>Machine-readable JSON and CSV are stored beside this report. Generated from {len(document['source_reports'])} ten-hour capacity-sweep reports.</p>
</main></body></html>"""


def _serialize(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    return value


def write_report(
    report_paths: Sequence[Path],
    output_dir: Path,
    *,
    ttft_p90_slo_ms: float,
    min_completion_ratio: float,
) -> tuple[Path, Path, Path]:
    points = extract_points(
        report_paths,
        ttft_p90_slo_ms=ttft_p90_slo_ms,
        min_completion_ratio=min_completion_ratio,
    )
    document = {
        "study": "per_prefill_gpu_cpu_dram_scaling",
        "prefill_gpus": PREFILL_GPUS,
        "decode_gpus": 16,
        "gpus_per_cpu_pool": GPUS_PER_CPU_POOL,
        "criteria": {
            "final_window_seconds": FINAL_WINDOW_SECONDS,
            "ttft_p90_slo_ms": ttft_p90_slo_ms,
            "min_completion_ratio": min_completion_ratio,
            "max_waiting_queue": DEFAULT_MAX_WAITING_QUEUE,
            "max_backlog_requests": DEFAULT_MAX_BACKLOG_REQUESTS,
            "requires_source_stable_classification_in_final_1h_and_2h": True,
        },
        "source_reports": [str(path.resolve()) for path in report_paths],
        "comparability": {
            "within_rate": "identical workload across CPU-DRAM capacities",
            "across_rates": "same TraceLab source sample and seed, but arrival spacing, repetition count, and final-hour session subsequence differ",
            "binary_identity": "historical run.json files record a binary path but not an executable hash or Git revision",
        },
        "load_envelope": build_load_envelope(points),
        "capacity_summary": build_capacity_summary(points),
        "points": points,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "per_gpu_cpu_dram_scaling_report.json"
    csv_path = output_dir / "per_gpu_cpu_dram_scaling_points.csv"
    html_path = output_dir / "per_gpu_cpu_dram_scaling_report.html"
    json_path.write_text(
        json.dumps(_serialize(document), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(points[0]) if points else [])
        if points:
            writer.writeheader()
            writer.writerows(points)
    html_path.write_text(_html_report(document), encoding="utf-8")
    return html_path, json_path, csv_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", type=Path, default=list(DEFAULT_REPORTS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--ttft-p90-slo-ms", type=float, default=DEFAULT_TTFT_P90_SLO_MS
    )
    parser.add_argument(
        "--min-completion-ratio", type=float, default=DEFAULT_MIN_COMPLETION_RATIO
    )
    args = parser.parse_args()
    if args.ttft_p90_slo_ms <= 0:
        raise SystemExit("--ttft-p90-slo-ms must be positive")
    if not 0 < args.min_completion_ratio <= 1:
        raise SystemExit("--min-completion-ratio must be in (0, 1]")
    missing = [path for path in args.reports if not path.is_file()]
    if missing:
        raise SystemExit("missing source reports: " + ", ".join(map(str, missing)))
    html_path, json_path, csv_path = write_report(
        [path.resolve() for path in args.reports],
        args.output_dir.resolve(),
        ttft_p90_slo_ms=args.ttft_p90_slo_ms,
        min_completion_ratio=args.min_completion_ratio,
    )
    print(html_path)
    print(json_path)
    print(csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
