#!/usr/bin/env python3
"""Generate one existing-style HTML report per completed dense-sweep rate.

Only cases with both ``summary.json`` and ``requests.csv`` are included.  The
script is safe to run while a sweep is partial and can be rerun after
``run_dense_cpu_capacity_rate_sweep.py --resume``.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from . import analyze_r0p4_capacity_sweep_10h as capacity_report
except ImportError:  # pragma: no cover - direct script execution
    import analyze_r0p4_capacity_sweep_10h as capacity_report


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "outputs" / "tracelab_vera_rubin_p8_d16_cpu_capacity_rate_dense_10h"
)


def _json_read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def completed_cases_by_rate(plan: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
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


def _index_html(output_root: Path, reports: Sequence[Mapping[str, Any]],
                planned: int, completed: int) -> str:
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
<title>Kimi K2 dense CPU capacity/rate sweep</title>
<style>body{{font:14px/1.5 system-ui,sans-serif;max-width:1200px;margin:auto;padding:28px;color:#172033;background:#f7f8fb}}table{{width:100%;border-collapse:collapse;background:#fff}}th,td{{padding:9px 11px;border:1px solid #d8dde8;text-align:left}}th{{background:#eef2f8}}code{{font-family:ui-monospace,monospace}}</style>
</head><body><h1>Kimi K2 P8/D16 dense CPU capacity/rate sweep</h1>
<p>Completed {completed} of {planned} planned points. Each link contains only completed capacities for that session-injection rate, including final-hour measurements and five-minute time series.</p>
<table><thead><tr><th>session rate</th><th>completed points</th><th>physical CPU capacities (GB)</th><th>report</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<p><code>capacity / 2</code> is the configured CPU DRAM slice per PREFILL GPU.</p>
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
    if not plan_path.is_file():
        raise FileNotFoundError(f"missing sweep plan: {plan_path}")
    plan = _json_read(plan_path)
    grouped = completed_cases_by_rate(plan)
    reports: list[dict[str, Any]] = []
    for rate_tag, cases in grouped.items():
        rate = float(cases[0]["rate"])
        rate_root = output_root / rate_tag
        capacities = [int(case["capacity_gb"]) for case in cases]
        metadata = Path(str(cases[0]["workload_metadata"]))
        workload_dir = metadata.parent
        output_csv = rate_root / f"{rate_tag}_capacity_sweep_10h_5min.csv"
        output_json = rate_root / f"{rate_tag}_capacity_sweep_10h.json"
        output_html = rate_root / f"{rate_tag}_capacity_sweep_10h.html"
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
    index_path = output_root / "index.html"
    index_path.write_text(
        _index_html(output_root, reports, len(plan.get("cases", [])), completed),
        encoding="utf-8",
    )
    summary = {
        "schema_version": 1,
        "sweep_plan": str(plan_path),
        "planned_cases": len(plan.get("cases", [])),
        "completed_cases": completed,
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
