#!/usr/bin/env python3
"""Generate one existing-style HTML report per completed dense-sweep rate.

The output directory is the source of truth.  Every completed
``r*/cpu*gb/r1`` case with both ``summary.json`` and ``requests.csv`` is
included, even when later split runner invocations replaced ``sweep_plan.json``
with a smaller plan.  The script is safe to run while a sweep is partial and
can be rerun after ``run_dense_cpu_capacity_rate_sweep.py --resume``.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
K2_EXPERIMENT = HERE.parent / "kimi_k2_cpu_dram"
if str(K2_EXPERIMENT) not in sys.path:
    sys.path.insert(0, str(K2_EXPERIMENT))

import analyze_r0p4_capacity_sweep_10h as capacity_report  # noqa: E402


DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "outputs" / "tracelab_k3_p24_d64_cpu_per_gpu_capacity_10h"
)

RATE_DIRECTORY = re.compile(r"^r(?P<integer>\d+)p(?P<fraction>\d+)$")
CAPACITY_DIRECTORY = re.compile(r"^cpu(?P<capacity>\d+)gb$")


def _json_read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def completed_cases_by_rate(plan: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Retain the legacy plan-only helper for callers outside this wrapper."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    cases = plan.get("cases", [])
    if not isinstance(cases, list):
        raise ValueError("sweep_plan.json cases must be a list")
    for raw in cases:
        if not isinstance(raw, Mapping):
            continue
        case = dict(raw)
        case_dir = Path(str(case.get("output_dir", "")))
        if not (case_dir / "summary.json").is_file() or not (case_dir / "requests.csv").is_file():
            continue
        label = str(case.get("rate_label", ""))
        if not label:
            continue
        grouped.setdefault(label, []).append(case)
    for cases_for_rate in grouped.values():
        cases_for_rate.sort(key=lambda case: int(case["capacity_gb"]))
    return dict(sorted(grouped.items(), key=lambda item: float(item[1][0]["rate"])))


def _rate_from_directory(label: str) -> float:
    match = RATE_DIRECTORY.fullmatch(label)
    if match is None:
        raise ValueError(f"invalid rate directory: {label}")
    return float(f"{match.group('integer')}.{match.group('fraction')}")


def _plan_cases_by_output_dir(plan: Mapping[str, Any] | None) -> dict[Path, dict[str, Any]]:
    if plan is None:
        return {}
    raw_cases = plan.get("cases", [])
    if not isinstance(raw_cases, list):
        raise ValueError("sweep_plan.json cases must be a list")
    indexed: dict[Path, dict[str, Any]] = {}
    for raw in raw_cases:
        if not isinstance(raw, Mapping) or not raw.get("output_dir"):
            continue
        indexed[Path(str(raw["output_dir"])).resolve()] = dict(raw)
    return indexed


def _prefill_gpu_count(case_dir: Path) -> int | None:
    for name in ("config.normalized.json", "config.input.json"):
        path = case_dir / name
        if not path.is_file():
            continue
        try:
            config = _json_read(path)
            parallelism = config["clusters"]["prefill"]["parallelism"]
            return (
                int(parallelism.get("num_replicas", 1))
                * int(parallelism.get("tensor_parallel_size", 1))
                * int(parallelism.get("pipeline_parallel_size", 1))
                * int(parallelism.get("data_parallel_size", 1))
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return None


def discover_completed_cases_by_rate(
    output_root: Path,
    plan: Mapping[str, Any] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Discover every completed canonical case below ``output_root``.

    Per-case ``run.json`` metadata wins over the latest sweep plan.  The plan
    remains a fallback for old result directories that predate ``run.json``.
    A case explicitly marked non-completed in ``run.json`` is excluded even if
    partial result files exist.
    """

    output_root = output_root.resolve()
    planned = _plan_cases_by_output_dir(plan)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for rate_root in sorted(output_root.iterdir() if output_root.is_dir() else []):
        if not rate_root.is_dir() or RATE_DIRECTORY.fullmatch(rate_root.name) is None:
            continue
        for capacity_root in sorted(rate_root.iterdir()):
            match = CAPACITY_DIRECTORY.fullmatch(capacity_root.name)
            if not capacity_root.is_dir() or match is None:
                continue
            case_dir = (capacity_root / "r1").resolve()
            if not (case_dir / "summary.json").is_file() or not (
                case_dir / "requests.csv"
            ).is_file():
                continue

            case = dict(planned.get(case_dir, {}))
            run_path = case_dir / "run.json"
            run: dict[str, Any] = {}
            if run_path.is_file():
                run = _json_read(run_path)
                status = run.get("status")
                if status is not None and status != "completed":
                    continue

            rate = float(
                run.get(
                    "session_arrival_rate_per_second",
                    case.get("rate", _rate_from_directory(rate_root.name)),
                )
            )
            capacity_gb = int(match.group("capacity"))
            case.update(
                {
                    "rate": rate,
                    "rate_label": rate_root.name,
                    "capacity_gb": capacity_gb,
                    "output_dir": str(case_dir),
                    "workload_metadata": run.get(
                        "workload_metadata", case.get("workload_metadata", "")
                    ),
                    "simulation_end_time_s": run.get(
                        "simulation_end_time_s", case.get("simulation_end_time_s")
                    ),
                    "prefill_gpus": _prefill_gpu_count(case_dir),
                }
            )
            grouped.setdefault(rate_root.name, []).append(case)

    for cases_for_rate in grouped.values():
        cases_for_rate.sort(key=lambda item: int(item["capacity_gb"]))
    return dict(sorted(grouped.items(), key=lambda item: float(item[1][0]["rate"])))


def _horizon_tag(cases: Sequence[Mapping[str, Any]]) -> str:
    horizons = {
        float(case["simulation_end_time_s"])
        for case in cases
        if case.get("simulation_end_time_s") is not None
        and float(case["simulation_end_time_s"]) > 0
    }
    if len(horizons) != 1:
        return "mixed_horizons" if horizons else "custom_horizon"
    horizon_hours = next(iter(horizons)) / 3600.0
    return (
        f"{int(round(horizon_hours))}h"
        if abs(horizon_hours - round(horizon_hours)) < 1e-9
        else "custom_horizon"
    )


def _index_html(
    output_root: Path,
    reports: Sequence[Mapping[str, Any]],
    latest_planned: int,
    completed: int,
    identity: Mapping[str, Any],
) -> str:
    model_name = str(identity.get("model_name") or "LLM")
    prefill_gpus = identity.get("prefill_gpu_count")
    decode_gpus = identity.get("decode_gpu_count")
    if bool(identity.get("prefill_only")):
        topology = (
            f"PREFILL-only, {int(prefill_gpus)} PREFILL GPUs"
            if prefill_gpus is not None
            else "PREFILL-only"
        )
    else:
        parts = []
        if prefill_gpus is not None:
            parts.append(f"{int(prefill_gpus)} PREFILL GPUs")
        if decode_gpus is not None:
            parts.append(f"{int(decode_gpus)} DECODE GPUs")
        topology = ", ".join(parts)
    report_name = model_name + (f" — {topology}" if topology else "")
    rows = []
    for report in reports:
        capacities = ", ".join(str(value) for value in report["capacities_gb"])
        href = Path(str(report["html"])).relative_to(output_root).as_posix()
        rows.append(
            "<tr>"
            f"<td>{float(report['rate']):.2f}/s</td>"
            f"<td>{len(report['capacities_gb'])}</td>"
            f"<td>{html.escape(capacities)}</td>"
            f"<td><a href='{html.escape(href)}'>open report</a></td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>{html.escape(report_name)} CPU-per-GPU capacity sweep</title>
<style>body{{font:14px/1.5 system-ui,sans-serif;max-width:1200px;margin:auto;padding:28px;color:#172033;background:#f7f8fb}}table{{width:100%;border-collapse:collapse;background:#fff}}th,td{{padding:9px 11px;border:1px solid #d8dde8;text-align:left}}th{{background:#eef2f8}}code{{font-family:ui-monospace,monospace}}</style>
</head><body><h1>{html.escape(report_name)} CPU-per-GPU capacity sweep</h1>
<p>Discovered {completed} completed points under this output directory. The latest sweep plan contains {latest_planned} points and is informational only. Each link contains every completed capacity found for that session-injection rate, including final-hour measurements and five-minute time series.</p>
<table><thead><tr><th>session rate</th><th>completed points</th><th>CPU capacity per PREFILL GPU (GB)</th><th>report</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<p>Capacity is the configured CPU DRAM slice per PREFILL GPU. Aggregate CPU
capacity is <code>capacity &times; 24</code>; zero is the CPU-off baseline.</p>
</body></html>"""


def generate_reports(
    output_root: Path,
    *,
    max_final_hour_ttft_p90_ms: float = 5_000.0,
    max_final_hour_queue_count: float = 1_000.0,
    max_final_hour_backlog_requests: float = 1_000.0,
) -> dict[str, Any]:
    output_root = output_root.resolve()
    plan_path = output_root / "sweep_plan.json"
    plan = _json_read(plan_path) if plan_path.is_file() else None
    grouped = discover_completed_cases_by_rate(output_root, plan)
    if not grouped:
        raise FileNotFoundError(
            "no completed r*/cpu*gb/r1 cases with summary.json and requests.csv "
            f"under {output_root}"
        )
    first_case_dir = Path(str(next(iter(grouped.values()))[0]["output_dir"]))
    report_identity = capacity_report._report_identity(first_case_dir)
    reports: list[dict[str, Any]] = []
    for rate_tag, cases in grouped.items():
        rate = float(cases[0]["rate"])
        rate_root = output_root / rate_tag
        capacities = [int(case["capacity_gb"]) for case in cases]
        metadata_candidates = [
            Path(str(case["workload_metadata"])).resolve()
            for case in cases
            if case.get("workload_metadata")
        ]
        metadata = next(
            (candidate for candidate in metadata_candidates if candidate.is_file()),
            rate_root / "missing_workload_metadata.json",
        )
        workload_dir = metadata.parent if metadata.is_file() else rate_root
        lane_counts = {
            int(case["prefill_gpus"])
            for case in cases
            if case.get("prefill_gpus") is not None
        }
        if len(lane_counts) > 1:
            raise ValueError(
                f"completed cases for {rate_tag} use different PREFILL GPU counts: "
                f"{sorted(lane_counts)}"
            )
        prefill_lanes = next(
            iter(lane_counts), int((plan or {}).get("prefill_gpus", 24) or 24)
        )
        horizon_tag = _horizon_tag(cases)
        output_csv = rate_root / f"{rate_tag}_capacity_sweep_{horizon_tag}_5min.csv"
        output_json = rate_root / f"{rate_tag}_capacity_sweep_{horizon_tag}.json"
        output_html = rate_root / f"{rate_tag}_capacity_sweep_{horizon_tag}.html"
        status = capacity_report.main(
            [
                "--output-root",
                str(rate_root),
                "--simulation-rate",
                format(rate, ".12g"),
                "--workload-dir",
                str(workload_dir),
                "--metadata",
                str(metadata),
                "--capacities",
                ",".join(str(value) for value in capacities),
                "--prefill-lanes",
                str(prefill_lanes),
                "--include-final-hour-details",
                "--max-final-hour-ttft-p90-ms",
                format(max_final_hour_ttft_p90_ms, "g"),
                "--max-final-hour-queue-count",
                format(max_final_hour_queue_count, "g"),
                "--max-final-hour-backlog-requests",
                format(max_final_hour_backlog_requests, "g"),
                "--output-csv",
                str(output_csv),
                "--output-json",
                str(output_json),
                "--output-html",
                str(output_html),
            ]
        )
        if status != 0:
            raise RuntimeError(f"report generation failed for {rate_tag} with status {status}")
        reports.append(
            {
                "rate": rate,
                "rate_label": rate_tag,
                "capacities_gb": capacities,
                "html": str(output_html.resolve()),
                "json": str(output_json.resolve()),
                "csv": str(output_csv.resolve()),
            }
        )
    completed = sum(len(cases) for cases in grouped.values())
    latest_planned = len((plan or {}).get("cases", []))
    index_path = output_root / "index.html"
    index_path.write_text(
        _index_html(
            output_root,
            reports,
            latest_planned,
            completed,
            report_identity,
        ),
        encoding="utf-8",
    )
    summary = {
        "schema_version": 1,
        "discovery": "completed r*/cpu*gb/r1 directories under output_root",
        "sweep_plan": str(plan_path) if plan_path.is_file() else None,
        "planned_cases": latest_planned,
        "completed_cases": completed,
        "report_identity": report_identity,
        "rate_reports": reports,
        "index_html": str(index_path),
    }
    (output_root / "report_manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-final-hour-ttft-p90-ms", type=float, default=5_000.0)
    parser.add_argument("--max-final-hour-queue-count", type=float, default=1_000.0)
    parser.add_argument("--max-final-hour-backlog-requests", type=float, default=1_000.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for name in (
        "max_final_hour_ttft_p90_ms",
        "max_final_hour_queue_count",
        "max_final_hour_backlog_requests",
    ):
        if getattr(args, name) < 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be nonnegative")
    print(
        json.dumps(
            generate_reports(
                args.output_root,
                max_final_hour_ttft_p90_ms=args.max_final_hour_ttft_p90_ms,
                max_final_hour_queue_count=args.max_final_hour_queue_count,
                max_final_hour_backlog_requests=args.max_final_hour_backlog_requests,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
