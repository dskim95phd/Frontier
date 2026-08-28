#!/usr/bin/env python3
"""Generate the TraceLab workload-distribution figure used in Chapter 3."""

from __future__ import annotations

import csv
import math
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = (
    REPO_ROOT
    / "outputs/datasets/tracelab/v0.0.2/frontier/all_sessions_continuous_10h"
    / "r0p30/seed_20260803_manifest.csv"
)
DEFAULT_OUTPUT = REPO_ROOT / "docs/report/figures/tracelab-workload-distributions.svg"


def load_values(path: Path) -> dict[str, list[float]]:
    values = {"isl": [], "osl": [], "gap": []}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            isl = float(row["new_input_tokens"])
            osl = float(row["decode_tokens"])
            gap = float(row["think_time"] or 0.0)
            if isl > 0:
                values["isl"].append(isl)
            if osl > 0:
                values["osl"].append(osl)
            values["gap"].append(max(gap, 0.001))
    return values


def percentile(ordered: list[float], fraction: float) -> float:
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def number_label(value: float, unit: str) -> str:
    if unit == "tokens":
        if value >= 1_000_000:
            return f"{value / 1_000_000:g}M"
        if value >= 1_000:
            return f"{value / 1_000:g}K"
        return f"{value:g}"
    if value < 1:
        return f"{value * 1_000:g} ms"
    if value < 1_000:
        return f"{value:g} s"
    if value < 1_000_000:
        return f"{value / 1_000:g} ks"
    return f"{value / 1_000_000:g} Ms"


def summary_label(value: float, unit: str) -> str:
    if unit == "tokens":
        return number_label(value, unit)
    if value < 1:
        return f"{value * 1_000:.3g} ms"
    if value < 60:
        return f"{value:.3g} s"
    if value < 3_600:
        return f"{value / 60:.3g} min"
    if value < 86_400:
        return f"{value / 3_600:.3g} h"
    return f"{value / 86_400:.3g} d"


def panel(
    title: str,
    unit: str,
    values: list[float],
    x0: float,
    y0: float,
    width: float,
    height: float,
) -> str:
    ordered = sorted(values)
    log_min = math.log10(ordered[0])
    log_max = math.log10(ordered[-1])
    if log_min == log_max:
        log_max += 1.0

    left = x0 + 70
    right = x0 + width - 22
    top = y0 + 48
    bottom = y0 + height - 58

    def x(value: float) -> float:
        return left + (math.log10(value) - log_min) / (log_max - log_min) * (right - left)

    def y(fraction: float) -> float:
        return bottom - fraction * (bottom - top)

    point_count = min(900, len(ordered))
    indices = sorted(
        {
            round(index * (len(ordered) - 1) / (point_count - 1))
            for index in range(point_count)
        }
    )
    path = " ".join(
        ("M" if position == 0 else "L")
        + f" {x(ordered[index]):.2f} {y((index + 1) / len(ordered)):.2f}"
        for position, index in enumerate(indices)
    )

    pieces = [
        f'<text x="{x0 + width / 2:.1f}" y="{y0 + 25:.1f}" text-anchor="middle" class="panel-title">{title}</text>',
        f'<rect x="{left:.1f}" y="{top:.1f}" width="{right-left:.1f}" height="{bottom-top:.1f}" class="frame"/>',
    ]

    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        yy = y(fraction)
        pieces.append(
            f'<line x1="{left:.1f}" y1="{yy:.1f}" x2="{right:.1f}" y2="{yy:.1f}" class="grid"/>'
        )
        pieces.append(
            f'<text x="{left-10:.1f}" y="{yy+4:.1f}" text-anchor="end" class="tick">{fraction*100:.0f}%</text>'
        )

    for exponent in range(math.ceil(log_min), math.floor(log_max) + 1):
        value = 10.0**exponent
        xx = x(value)
        pieces.append(
            f'<line x1="{xx:.1f}" y1="{bottom:.1f}" x2="{xx:.1f}" y2="{bottom+6:.1f}" class="axis"/>'
        )
        pieces.append(
            f'<text x="{xx:.1f}" y="{bottom+23:.1f}" text-anchor="middle" class="tick">{number_label(value, unit)}</text>'
        )

    pieces.extend(
        [
            f'<path d="{path}" class="cdf"/>',
            f'<text x="{(left+right)/2:.1f}" y="{y0+height-14:.1f}" text-anchor="middle" class="axis-title">{unit}</text>',
            f'<text x="{x0+18:.1f}" y="{(top+bottom)/2:.1f}" text-anchor="middle" transform="rotate(-90 {x0+18:.1f} {(top+bottom)/2:.1f})" class="axis-title">Cumulative fraction</text>',
        ]
    )

    summary = ", ".join(
        f"p{int(fraction*100)} {summary_label(percentile(ordered, fraction), unit)}"
        for fraction in (0.5, 0.9, 0.99)
    )
    pieces.append(
        f'<text x="{x0+width/2:.1f}" y="{top+18:.1f}" text-anchor="middle" class="summary">{summary}</text>'
    )
    return "\n".join(pieces)


def generate(input_path: Path, output_path: Path) -> None:
    values = load_values(input_path)
    width = 1500
    height = 520
    panel_width = 480
    panel_height = 410
    panels = [
        panel(
            "Input sequence length (ISL)",
            "tokens",
            values["isl"],
            10,
            70,
            panel_width,
            panel_height,
        ),
        panel(
            "Output sequence length (OSL)",
            "tokens",
            values["osl"],
            510,
            70,
            panel_width,
            panel_height,
        ),
        panel(
            "Inter-turn gap",
            "seconds",
            values["gap"],
            1010,
            70,
            panel_width,
            panel_height,
        ),
    ]
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
  <title id="title">TraceLab workload distributions</title>
  <desc id="desc">Empirical cumulative distributions of input tokens, output tokens, and inter-turn gaps in the TraceLab workload used for the experiment.</desc>
  <style>
    text {{ font-family: Inter, "Noto Sans KR", "Segoe UI", Arial, sans-serif; fill: #17324d; }}
    .title {{ font-size: 28px; font-weight: 700; }}
    .panel-title {{ font-size: 19px; font-weight: 700; }}
    .summary {{ font-size: 12px; fill: #436176; }}
    .axis-title {{ font-size: 13px; font-weight: 600; }}
    .tick {{ font-size: 11px; fill: #526a7c; }}
    .frame {{ fill: #ffffff; stroke: #9fb4c4; stroke-width: 1.2; }}
    .grid {{ stroke: #e2e9ee; stroke-width: 1; }}
    .axis {{ stroke: #71889a; stroke-width: 1; }}
    .cdf {{ fill: none; stroke: #087f8c; stroke-width: 3; stroke-linejoin: round; stroke-linecap: round; }}
  </style>
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="750" y="38" text-anchor="middle" class="title">TraceLab workload distributions</text>
  {panels[0]}
  {panels[1]}
  {panels[2]}
  <text x="1490" y="510" text-anchor="end" class="tick">Log-scaled x-axes; zero-second gaps shown at 1 ms</text>
</svg>
'''
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    generate(DEFAULT_INPUT, DEFAULT_OUTPUT)
