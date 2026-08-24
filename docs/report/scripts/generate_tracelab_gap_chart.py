#!/usr/bin/env python3
"""Export the canonical TraceLab v0.0.2 turn-gap ECDF as a report SVG."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPORT = (
    ROOT / "outputs/datasets/tracelab/v0.0.2/analysis/v3/report.html"
)
DEFAULT_OUTPUT = ROOT / "docs/presentation_assets/tracelab-turn-gap-ecdf.svg"
DEFAULT_DATA = ROOT / "docs/report/data/tracelab-turn-gap-ecdf.csv"

COLORS = {
    "human": "#3f7f5f",
    "tool": "#6d5aa7",
    "navy": "#17324d",
    "gray": "#6d7782",
    "grid": "#dce3e8",
}


def _json_array_after_marker(source: str, marker: str) -> list[dict[str, object]]:
    marker_at = source.find(marker)
    if marker_at < 0:
        raise ValueError(f"marker not found in TraceLab report: {marker}")
    start = source.find("[", marker_at + len(marker))
    if start < 0:
        raise ValueError("Plotly trace array not found")

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(source)):
        char = source[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return json.loads(source[start : index + 1])
    raise ValueError("unterminated Plotly trace array")


def _extract_counts(source: str) -> dict[str, int]:
    section_match = re.search(r"<h2>2\. Turn gap.*?</section>", source, re.S)
    if not section_match:
        raise ValueError("TraceLab turn-gap section not found")
    section = section_match.group(0)
    result: dict[str, int] = {}
    for name in ("human", "tool"):
        match = re.search(
            rf"<tr><td>{name}</td><td>([\d,]+)</td>", section
        )
        if not match:
            raise ValueError(f"TraceLab count not found for {name}")
        result[name] = int(match.group(1).replace(",", ""))
    return result


def load_gap_curves(report_path: Path) -> dict[str, dict[str, object]]:
    source = report_path.read_text(encoding="utf-8")
    traces = _json_array_after_marker(source, '"chart-gaps",')
    counts = _extract_counts(source)
    curves: dict[str, dict[str, object]] = {}
    for trace in traces:
        name = str(trace.get("name", ""))
        if name not in counts:
            continue
        gaps = np.asarray(trace["customdata"], dtype=float)
        percentiles = np.asarray(trace["y"], dtype=float)
        if len(gaps) != len(percentiles) or len(gaps) == 0:
            raise ValueError(f"invalid TraceLab curve for {name}")
        curves[name] = {
            "n": counts[name],
            "gaps": gaps,
            "percentiles": percentiles,
        }
    if set(curves) != {"human", "tool"}:
        raise ValueError(f"expected human/tool curves, found {sorted(curves)}")
    return curves


def write_data(path: Path, curves: dict[str, dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("trigger_type", "n", "percentile", "gap_seconds"))
        for name in ("human", "tool"):
            values = curves[name]
            for percentile, gap in zip(
                values["percentiles"], values["gaps"], strict=True
            ):
                writer.writerow(
                    (name, values["n"], f"{float(percentile):.3f}", f"{float(gap):.9f}")
                )


def format_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds:.3g}초"
    if seconds < 60:
        return f"{seconds:.1f}초"
    if seconds < 3_600:
        return f"{seconds / 60:.1f}분"
    if seconds < 86_400:
        return f"{seconds / 3_600:.1f}시간"
    return f"{seconds / 86_400:.1f}일"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Malgun Gothic", "DejaVu Sans", "sans-serif"],
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "font.size": 12,
        }
    )


def plot_chart(
    curves: dict[str, dict[str, object]], output_path: Path, preview_path: Path | None
) -> None:
    configure_style()
    fig, axis = plt.subplots(figsize=(12.8, 6.8))

    labels = {
        "human": "사람 후속 입력",
        "tool": "도구 결과 후속 입력",
    }
    markers = {"human": "o", "tool": "s"}
    floor_seconds = 0.001

    for name in ("tool", "human"):
        values = curves[name]
        gaps = np.maximum(np.asarray(values["gaps"], dtype=float), floor_seconds)
        percentiles = np.asarray(values["percentiles"], dtype=float)
        axis.plot(
            gaps,
            percentiles,
            color=COLORS[name],
            linewidth=2.6,
            label=f"{labels[name]} (n={int(values['n']):,})",
        )

        for percentile, vertical_offset in ((0.50, 12), (0.90, -19), (0.99, -19)):
            index = int(np.argmin(np.abs(percentiles - percentile)))
            gap = float(np.asarray(values["gaps"])[index])
            axis.scatter(
                [max(gap, floor_seconds)],
                [percentile],
                color=COLORS[name],
                marker=markers[name],
                s=42,
                zorder=4,
            )
            axis.annotate(
                f"{name} p{int(percentile * 100)}  {format_duration(gap)}",
                xy=(max(gap, floor_seconds), percentile),
                xytext=(7, vertical_offset),
                textcoords="offset points",
                color=COLORS[name],
                fontsize=10.5,
                fontweight="bold",
            )

    axis.set_xscale("log")
    axis.set_xlim(floor_seconds, 30 * 86_400)
    axis.set_ylim(0, 1.025)
    ticks = [0.001, 0.01, 0.1, 1, 10, 60, 600, 3_600, 21_600, 86_400, 604_800, 2_592_000]
    tick_labels = [
        "1ms", "10ms", "0.1초", "1초", "10초", "1분", "10분",
        "1시간", "6시간", "1일", "7일", "30일",
    ]
    axis.set_xticks(ticks, tick_labels)
    axis.tick_params(axis="x", labelrotation=22)
    axis.set_yticks(
        [0, 0.25, 0.50, 0.75, 0.90, 1.0],
        ["0%", "25%", "50%", "75%", "90%", "100%"],
    )
    axis.grid(True, which="major", color=COLORS["grid"], linewidth=0.9)
    axis.grid(True, which="minor", axis="x", color=COLORS["grid"], linewidth=0.45, alpha=0.45)
    for spine in axis.spines.values():
        spine.set_color("#c9d1d8")

    axis.set_title(
        "TraceLab에서 관측한 에이전틱 요청 간격",
        color=COLORS["navy"],
        fontsize=20,
        fontweight="bold",
        pad=18,
    )
    axis.set_xlabel("이전 모델 출력부터 다음 입력까지의 시간 (log scale)", fontsize=12.5)
    axis.set_ylabel("누적 비율", fontsize=12.5)
    axis.legend(loc="lower right", frameon=False, fontsize=11.5)

    fig.text(
        0.01,
        0.012,
        "자료: TraceLab v0.0.2, canonical v3 analysis. 세션 root 제외; 음수 관측 gap은 0초로 보정(그래프에서는 1ms 위치에 표시).",
        fontsize=9.5,
        color=COLORS["gray"],
    )
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.18, top=0.88)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    if preview_path is not None:
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-output", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--preview", type=Path)
    args = parser.parse_args()

    curves = load_gap_curves(args.report)
    write_data(args.data_output, curves)
    plot_chart(curves, args.output, args.preview)
    for name in ("human", "tool"):
        values = curves[name]
        gaps = np.asarray(values["gaps"], dtype=float)
        percentiles = np.asarray(values["percentiles"], dtype=float)
        summary = {
            f"p{int(p * 100)}": float(gaps[int(np.argmin(np.abs(percentiles - p)))])
            for p in (0.50, 0.90, 0.99)
        }
        print(name, f"n={int(values['n']):,}", summary)


if __name__ == "__main__":
    main()
