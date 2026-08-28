#!/usr/bin/env python3
"""Generate the Chapter 3 CPU-DRAM capacity result figure."""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT = REPO_ROOT / "docs/report/figures/cpu-dram-capacity-throughput.svg"

CAPACITIES_GB = [250, 375, 500, 625, 750, 875, 1000]
RATE_LABELS = [
    "r0p05",
    "r0p10",
    "r0p15",
    "r0p20",
    "r0p25",
    "r0p30",
    "r0p35",
    "r0p40",
    "r0p45",
    "r0p50",
]

EXPERIMENTS = {
    "Kimi K2": {
        "root": REPO_ROOT
        / "outputs/tracelab_k2_p8_d16_prefill_only_cpu_per_gpu_capacity_12h",
        "color": "#087f8c",
        "prefill_gpus": 8,
    },
    "Kimi K3": {
        "root": REPO_ROOT
        / "outputs/tracelab_k3_p24_d64_prefill_only_batch8192_cpu_per_gpu_capacity_12h",
        "color": "#d97706",
        "prefill_gpus": 24,
    },
}


def load_points(root: Path) -> list[dict[str, float]]:
    points: list[dict[str, float]] = []
    for capacity in CAPACITIES_GB:
        best: dict[str, float] | None = None
        for rate_label in RATE_LABELS:
            report_path = root / rate_label / f"{rate_label}_capacity_sweep_12h.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            eligibility = next(
                item
                for item in report["recommendations"]["eligibility_by_capacity"]
                if item["capacity_gb"] == capacity
            )
            if not eligibility["operationally_acceptable"]:
                continue
            case = next(
                item for item in report["cases"] if item["capacity_gb"] == capacity
            )
            statistics = case["stability"]["final_hour"]["statistics"]
            best = {
                "capacity_gb": float(capacity),
                "rate": float(case["rate"]),
                "requests_per_second": float(statistics["completion_requests_per_s"]),
            }
        if best is None:
            raise RuntimeError(f"no operational point for {capacity} GB in {root}")
        points.append(best)
    return points


def generate(output_path: Path) -> None:
    series = {
        name: {**metadata, "points": load_points(metadata["root"])}
        for name, metadata in EXPERIMENTS.items()
    }

    width, height = 1200, 680
    left, right, top, bottom = 105, 1145, 90, 545
    y_max = 22.0

    def x(value: float) -> float:
        return left + (value - CAPACITIES_GB[0]) / (
            CAPACITIES_GB[-1] - CAPACITIES_GB[0]
        ) * (right - left)

    def y(value: float) -> float:
        return bottom - value / y_max * (bottom - top)

    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '  <title id="title">Maximum verified request throughput by CPU DRAM capacity</title>',
        '  <desc id="desc">Kimi K2 throughput increases with CPU DRAM capacity and plateaus at 750 GB per PREFILL GPU, while Kimi K3 remains nearly flat from 250 to 1000 GB per PREFILL GPU.</desc>',
        "  <style>",
        '    text { font-family: Inter, "Noto Sans KR", "Segoe UI", Arial, sans-serif; fill: #17324d; }',
        "    .title { font-size: 27px; font-weight: 700; }",
        "    .subtitle { font-size: 14px; fill: #526a7c; }",
        "    .axis-title { font-size: 15px; font-weight: 600; }",
        "    .tick { font-size: 13px; fill: #526a7c; }",
        "    .grid { stroke: #e2e9ee; stroke-width: 1; }",
        "    .axis { stroke: #71889a; stroke-width: 1.2; }",
        "    .value { font-size: 12px; font-weight: 600; }",
        "    .legend { font-size: 14px; font-weight: 600; }",
        "    .note { font-size: 12px; fill: #526a7c; }",
        "  </style>",
        '  <rect width="100%" height="100%" fill="#ffffff"/>',
        '  <text x="600" y="38" text-anchor="middle" class="title">Maximum verified request throughput</text>',
        '  <text x="600" y="64" text-anchor="middle" class="subtitle">Trailing 1 h; every 5 min TTFT p90 ≤ 5 s</text>',
    ]

    for tick in range(0, 23, 2):
        yy = y(float(tick))
        pieces.append(
            f'  <line x1="{left}" y1="{yy:.2f}" x2="{right}" y2="{yy:.2f}" class="grid"/>'
        )
        pieces.append(
            f'  <text x="{left - 14}" y="{yy + 5:.2f}" text-anchor="end" class="tick">{tick}</text>'
        )

    pieces.extend(
        [
            f'  <line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" class="axis"/>',
            f'  <line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" class="axis"/>',
        ]
    )
    for capacity in CAPACITIES_GB:
        xx = x(float(capacity))
        pieces.append(
            f'  <line x1="{xx:.2f}" y1="{bottom}" x2="{xx:.2f}" y2="{bottom + 7}" class="axis"/>'
        )
        pieces.append(
            f'  <text x="{xx:.2f}" y="{bottom + 27}" text-anchor="middle" class="tick">{capacity}</text>'
        )

    pieces.extend(
        [
            f'  <text x="{(left + right) / 2:.1f}" y="{bottom + 62}" text-anchor="middle" class="axis-title">CPU DRAM per PREFILL GPU (GB)</text>',
            f'  <text x="28" y="{(top + bottom) / 2:.1f}" text-anchor="middle" transform="rotate(-90 28 {(top + bottom) / 2:.1f})" class="axis-title">Maximum completed requests/s</text>',
        ]
    )

    for name, metadata in series.items():
        points = metadata["points"]
        color = metadata["color"]
        path = " ".join(
            ("M" if index == 0 else "L")
            + f" {x(point['capacity_gb']):.2f} {y(point['requests_per_second']):.2f}"
            for index, point in enumerate(points)
        )
        pieces.append(
            f'  <path d="{path}" fill="none" stroke="{color}" stroke-width="4" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        for point in points:
            xx = x(point["capacity_gb"])
            yy = y(point["requests_per_second"])
            label_y = yy - 13 if name == "Kimi K2" else yy + 24
            pieces.append(
                f'  <circle cx="{xx:.2f}" cy="{yy:.2f}" r="6" fill="#ffffff" stroke="{color}" stroke-width="4"/>'
            )
            pieces.append(
                f'  <text x="{xx:.2f}" y="{label_y:.2f}" text-anchor="middle" class="value" fill="{color}">{point["requests_per_second"]:.2f}</text>'
            )

    legend_x = 835
    for index, (name, metadata) in enumerate(series.items()):
        yy = 105 + index * 27
        pieces.append(
            f'  <line x1="{legend_x}" y1="{yy}" x2="{legend_x + 38}" y2="{yy}" stroke="{metadata["color"]}" stroke-width="4"/>'
        )
        pieces.append(
            f'  <text x="{legend_x + 50}" y="{yy + 5}" class="legend">{name}</text>'
        )

    pieces.extend(
        [
            '  <text x="105" y="632" class="note">Aggregate CPU range: Kimi K2 2–8 TB (8 PREFILL GPUs); Kimi K3 6–24 TB (24 PREFILL GPUs).</text>',
            '  <text x="105" y="653" class="note">Each point is the highest tested session-injection rate that passed the operational stability and TTFT gates.</text>',
            "</svg>",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(pieces) + "\n", encoding="utf-8")


if __name__ == "__main__":
    generate(OUTPUT)
