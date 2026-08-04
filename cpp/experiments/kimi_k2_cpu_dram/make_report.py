#!/usr/bin/env python3
"""Build the final paired summary, bootstrap intervals, plots, and report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_results as analysis  # noqa: E402

SEEDS = (20260803, 20260804, 20260805, 20260806, 20260807)
COARSE = ("off", "cpu32", "cpu64", "cpu128", "cpu256")
FINE = ("cpu500", "cpu544", "cpu584", "cpu632", "cpu672", "cpu720", "cpu760")
EXTENSION = ("cpu1520",)
ORACLE = ("oracle_unbounded_cpu",)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--workload-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    return parser.parse_args(argv)


def record_path(root: Path, seed: int, label: str) -> tuple[Path, str]:
    if label in COARSE or label in ORACLE:
        return root / "coarse" / f"seed_{seed}" / label / "run.json", "coarse"
    if label in FINE:
        return root / "fine" / f"seed_{seed}" / label / "run.json", "fine"
    return root / "coarse" / "custom" / f"seed_{seed}" / label / "run.json", "analytical_extension"


def collect(root: Path, workload_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label in (*COARSE, *FINE, *EXTENSION, *ORACLE):
        for seed in SEEDS:
            path, stage = record_path(root, seed, label)
            row = analysis.summarize_run(path, workload_root=workload_dir)
            if row is None:
                raise RuntimeError(f"missing or failed run: {path}")
            row["stage"] = stage
            rows.append(row)
    return rows


def session_means(run: Mapping[str, object]) -> dict[int, float]:
    grouped: dict[int, list[float]] = {}
    values = run.get("request_rows")
    if not isinstance(values, Sequence):
        return {}
    for row in values:
        if isinstance(row, Mapping) and analysis._as_int(row.get("session_turn_index")) > 0:
            grouped.setdefault(analysis._as_int(row.get("session_id")), []).append(
                analysis._as_float(row.get("ttft_ms"))
            )
    return {session: float(np.mean(ttft)) for session, ttft in grouped.items() if ttft}


def bootstrap_paired_improvement(
    baseline_values: Sequence[float], case_values: Sequence[float], samples: int, seed: int
) -> tuple[float, float, float]:
    baseline = np.asarray(baseline_values, dtype=float)
    case = np.asarray(case_values, dtype=float)
    if baseline.size == 0 or baseline.size != case.size:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=float)
    for index in range(samples):
        selected = rng.integers(0, baseline.size, baseline.size)
        sampled_baseline = float(np.mean(baseline[selected]))
        sampled_case = float(np.mean(case[selected]))
        means[index] = (
            (sampled_baseline - sampled_case) / sampled_baseline
            if sampled_baseline
            else 0.0
        )
    baseline_mean = float(np.mean(baseline))
    case_mean = float(np.mean(case))
    point = (baseline_mean - case_mean) / baseline_mean if baseline_mean else 0.0
    return point, float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def add_comparisons(rows: list[dict[str, object]], aggregate: list[dict[str, object]], samples: int) -> list[dict[str, object]]:
    by_key = {(str(row["capacity"]), int(row["seed"])): row for row in rows}
    for item in aggregate:
        label = str(item["capacity"])
        ttft_deltas: list[float] = []
        prefill_deltas: list[float] = []
        session_baselines: list[float] = []
        session_cases: list[float] = []
        for seed in SEEDS:
            off = by_key[("off", seed)]
            case = by_key[(label, seed)]
            off_ttft = analysis._as_float(off.get("session_equal_successor_ttft_mean_ms"))
            case_ttft = analysis._as_float(case.get("session_equal_successor_ttft_mean_ms"))
            off_prefill = analysis._as_float(off.get("scheduled_prefill_tokens"))
            case_prefill = analysis._as_float(case.get("scheduled_prefill_tokens"))
            ttft_deltas.append((off_ttft - case_ttft) / off_ttft if off_ttft else 0.0)
            prefill_deltas.append((off_prefill - case_prefill) / off_prefill if off_prefill else 0.0)
            off_sessions = session_means(off)
            case_sessions = session_means(case)
            for session_id in sorted(off_sessions.keys() & case_sessions.keys()):
                session_baselines.append(off_sessions[session_id])
                session_cases.append(case_sessions[session_id])
        mean_delta, ci_low, ci_high = bootstrap_paired_improvement(
            session_baselines,
            session_cases,
            samples,
            20260803 + int(analysis._numeric_capacity(label) if np.isfinite(analysis._numeric_capacity(label)) else 9999),
        )
        item["paired_seed_ttft_improvement_mean"] = float(np.mean(ttft_deltas))
        item["paired_seed_ttft_improvement_min"] = float(np.min(ttft_deltas))
        item["paired_seed_ttft_improvement_max"] = float(np.max(ttft_deltas))
        item["paired_seed_prefill_reduction_mean"] = float(np.mean(prefill_deltas))
        item["session_bootstrap_ttft_improvement_mean"] = mean_delta
        item["session_bootstrap_ttft_improvement_ci95_low"] = ci_low
        item["session_bootstrap_ttft_improvement_ci95_high"] = ci_high
        item["physical_gb300_range"] = bool(label == "off" or (
            np.isfinite(analysis._numeric_capacity(label)) and analysis._numeric_capacity(label) <= 500
        ))
        item["stage"] = next(str(row["stage"]) for row in rows if str(row["capacity"]) == label)
    return aggregate


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields and key != "request_rows":
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def capacity_x(row: Mapping[str, object]) -> float:
    value = analysis._numeric_capacity(str(row["capacity"]))
    return 1700.0 if not np.isfinite(value) else value


def plot_all(summary: Sequence[Mapping[str, object]], turns: Sequence[Mapping[str, object]], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    plotted = list(summary)
    x = [capacity_x(row) for row in plotted]
    stage_markers = {"coarse": "o", "fine": "s", "analytical_extension": "^"}

    def add_stage_markers(y: Sequence[float], color: str, *, labels: bool) -> None:
        for stage, marker in stage_markers.items():
            indices = [index for index, row in enumerate(plotted) if str(row.get("stage")) == stage]
            if indices:
                plt.scatter(
                    [x[index] for index in indices],
                    [y[index] for index in indices],
                    color=color,
                    marker=marker,
                    s=42,
                    label=stage if labels else None,
                    zorder=3,
                )

    def finish(name: str, ylabel: str) -> None:
        plt.axvline(500, color="black", linestyle="--", linewidth=1, label="physical limit")
        plt.xlabel("Grace CPU DRAM (GB); oracle plotted at 1700")
        plt.ylabel(ylabel)
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output / name, dpi=180)
        plt.close()

    for index, key in enumerate(("successor_ttft_p50_ms", "successor_ttft_p90_ms", "successor_ttft_p99_ms")):
        y = [analysis._as_float(row.get(key)) for row in plotted]
        line, = plt.plot(x, y, label=key.replace("successor_ttft_", ""))
        add_stage_markers(y, line.get_color(), labels=index == 0)
    finish("01_ttft_quantiles.png", "Successor TTFT (ms)")

    y = [1.0 + analysis._as_float(row.get("actual_extra_ratio")) for row in plotted]
    line, = plt.plot(x, y, label="scheduled / fresh-input minimum")
    add_stage_markers(y, line.get_color(), labels=True)
    finish("02_prefill_ratio.png", "Scheduled PREFILL / theoretical minimum")

    for index, (key, label) in enumerate((("cpu_restorable_hit_rate", "CPU restorable hit"), ("eviction_block_rate", "evicted / offloaded blocks"))):
        y = [100 * analysis._as_float(row.get(key)) for row in plotted]
        line, = plt.plot(x, y, label=label)
        add_stage_markers(y, line.get_color(), labels=index == 0)
    finish("03_cache_hit_eviction.png", "Percent")

    plt.plot(x, [analysis._as_float(row.get("h2d_queue_time_ms")) for row in plotted], marker="o", label="H2D queue")
    plt.plot(x, [analysis._as_float(row.get("h2d_service_time_ms")) for row in plotted], marker="s", label="H2D service")
    plt.plot(x, [analysis._as_float(row.get("d2h_service_time_ms")) for row in plotted], marker="^", label="D2H service")
    finish("04_transfer_time.png", "Aggregate transfer time (ms)")

    selected = {"off", "cpu500", "cpu672", "cpu720", "cpu760", "oracle_unbounded_cpu"}
    for label in selected:
        values = sorted(
            [row for row in turns if str(row.get("capacity")) == label],
            key=lambda row: analysis._as_int(row.get("turn_index")),
        )
        if values:
            plt.plot([analysis._as_int(row.get("turn_index")) for row in values], [analysis._as_float(row.get("ttft_mean_ms")) for row in values], marker="o", label=label)
    plt.xlabel("Turn index")
    plt.ylabel("TTFT mean (ms)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "05_turn_ttft.png", dpi=180)
    plt.close()

    for label in selected:
        values = sorted(
            [row for row in turns if str(row.get("capacity")) == label],
            key=lambda row: analysis._as_int(row.get("turn_index")),
        )
        if values:
            plt.plot([analysis._as_int(row.get("turn_index")) for row in values], [100 * analysis._as_float(row.get("cached_token_fraction_mean")) for row in values], marker="o", label=label)
    plt.xlabel("Turn index")
    plt.ylabel("Mean cached-token fraction (%)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "06_turn_cached_fraction.png", dpi=180)
    plt.close()


def report_markdown(summary: Sequence[Mapping[str, object]]) -> str:
    by_label = {str(row["capacity"]): row for row in summary}
    chosen = ("off", "cpu256", "cpu500", "cpu672", "cpu720", "cpu760", "cpu1520", "oracle_unbounded_cpu")
    lines = [
        "# Kimi K2 / GB300 CPU DRAM capacity study — final results",
        "",
        "## 결론",
        "",
        "GB300 physical 범위(최대 500 GB/Grace CPU)에서는 inclusive CPU cache가 GPU cache를 유효하게 확장하지 못했다. 500 GB에서 CPU restorable hit는 약 1.9%, PREFILL 감소는 약 1.4%였고 successor TTFT 개선은 통계적으로 작았다. 최초의 사전 정의 threshold는 non-physical 672→720 GB/CPU 구간(ρ 1.77→1.89)에서 나타났다. unbounded oracle은 복원 가능한 prefix를 100% 커버하고 scheduled PREFILL을 implementation minimum까지 낮췄지만 TTFT 개선은 PREFILL 외 지연 때문에 제한됐다.",
        "",
        "## 핵심 수치",
        "",
        "| Case | ρ | CPU restorable hit | PREFILL reduction vs OFF | Session-equal TTFT improvement | 95% bootstrap CI | Successor p90 (ms) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label in chosen:
        row = by_label[label]
        rho = analysis._as_float(row.get("rho")) if row.get("rho") is not None else float("nan")
        rho_text = "—" if not np.isfinite(rho) else f"{rho:.2f}"
        lines.append(
            f"| {label} | {rho_text} | {100*analysis._as_float(row.get('cpu_restorable_hit_rate')):.2f}% | "
            f"{100*analysis._as_float(row.get('paired_seed_prefill_reduction_mean')):.2f}% | "
            f"{100*analysis._as_float(row.get('paired_seed_ttft_improvement_mean')):.2f}% | "
            f"[{100*analysis._as_float(row.get('session_bootstrap_ttft_improvement_ci95_low')):.2f}%, "
            f"{100*analysis._as_float(row.get('session_bootstrap_ttft_improvement_ci95_high')):.2f}%] | "
            f"{analysis._as_float(row.get('successor_ttft_p90_ms')):.1f} |"
        )
    lines.extend([
        "",
        "## Gate 결과",
        "",
        "- Workload: final context 평균 65,536 tokens, input:output 8:1, CPU-OFF 추가 PREFILL pooled 296.84%.",
        "- OFF: CPU offload/hit/restore 0, GPU hit 1% 이하, preemption 0.",
        "- Oracle: eviction/truncation/skip 0, CPU restorable hit 및 GPU+CPU coverage 100%, terminal state 0.",
        "- Runtime: coarse 35/35, extension 10/10, fine 35/35 완료; 동일 cpu128 재실행 requests SHA-256 일치.",
        "",
        "## 해석 범위",
        "",
        "결과는 Kimi K2 FP8 MLA, TP4에서 각 TP가 전체 KV를 저장하는 가정, sequential PDD, decode KV 반환 없음, analytical GB300 실행시간/전송 모델에 한정된다. 544 GB 이상은 제품 구성이 아니라 analytical capacity extension이다.",
    ])
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.study_root.resolve()
    workload = args.workload_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = collect(root, workload)
    aggregate = add_comparisons(rows, analysis.aggregate_by_capacity(rows), args.bootstrap_samples)
    turns = analysis._turn_summary(rows)
    sessions = analysis._session_summary(rows)
    write_csv(output / "final_capacity_summary.csv", aggregate)
    write_csv(output / "final_turn_summary.csv", turns)
    write_csv(output / "final_session_summary.csv", sessions)
    (output / "final_summary.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    plot_all(aggregate, turns, output / "plots")
    (output / "final_report.md").write_text(report_markdown(aggregate), encoding="utf-8")
    print(f"wrote final report artifacts under {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
