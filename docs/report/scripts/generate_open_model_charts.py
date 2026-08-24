#!/usr/bin/env python3
"""Generate report charts for open-weight model and KV-cache trends."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = ROOT / "docs" / "report"
DATA_DIR = REPORT_DIR / "data"
DEFAULT_OUTPUT_DIR = ROOT / "docs" / "presentation_assets"

COLORS = {
    "navy": "#17324d",
    "teal": "#087f8c",
    "orange": "#d9772f",
    "red": "#b54343",
    "purple": "#6d5aa7",
    "green": "#3f7f5f",
    "gold": "#a87812",
    "gray": "#6d7782",
    "light_gray": "#c9d1d8",
    "grid": "#dce3e8",
}


@dataclass(frozen=True)
class ReleasePoint:
    release_date: datetime
    model: str
    architecture: str
    total_params_b: float
    active_params_b: float | None
    advertised_context: int
    trained_context: int | None


SHORT_NAMES = {
    "Llama 2 70B": "Llama 2 70B",
    "Mistral 7B v0.1": "Mistral 7B",
    "Mixtral 8x7B v0.1": "Mixtral 8×7B",
    "Llama 3 70B": "Llama 3 70B",
    "DeepSeek-V2": "DeepSeek-V2",
    "Qwen2-72B": "Qwen2 72B",
    "Llama 3.1 405B": "Llama 3.1 405B",
    "Mistral Large 2": "Mistral Large 2",
    "Qwen2.5-72B": "Qwen2.5 72B",
    "DeepSeek-V3": "DeepSeek-V3",
    "Gemma 3 27B": "Gemma 3 27B",
    "Llama 4 Scout": "Llama 4 Scout",
    "Qwen3-235B-A22B": "Qwen3 235B",
    "Kimi K2": "Kimi K2",
    "DeepSeek-V4-Pro": "DeepSeek-V4 Pro",
    "Kimi K3": "Kimi K3",
}


def _optional_float(value: str) -> float | None:
    return float(value) if value else None


def _optional_int(value: str) -> int | None:
    return int(value) if value else None


def load_release_points() -> list[ReleasePoint]:
    path = DATA_DIR / "open-model-release-timeline.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        ReleasePoint(
            release_date=datetime.strptime(row["release_date"], "%Y-%m-%d"),
            model=row["model"],
            architecture=row["architecture"],
            total_params_b=float(row["total_params_b"]),
            active_params_b=_optional_float(row["active_params_b"]),
            advertised_context=int(row["advertised_context_tokens"]),
            trained_context=_optional_int(row["trained_or_native_context_tokens"]),
        )
        for row in rows
    ]


def human_tokens(value: float) -> str:
    canonical = {
        4096: "4K",
        8192: "8K",
        32768: "32K",
        131072: "128K",
        1048576: "1M",
        10485760: "10M",
    }
    if int(value) in canonical:
        return canonical[int(value)]
    if value >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if value >= 1_000:
        return f"{value / 1_000:g}K"
    return f"{value:g}"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Malgun Gothic", "DejaVu Sans", "sans-serif"],
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "font.size": 10.5,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
        }
    )


def style_axis(axis: plt.Axes) -> None:
    axis.grid(True, which="major", color=COLORS["grid"], linewidth=0.8)
    axis.grid(True, which="minor", color=COLORS["grid"], linewidth=0.45, alpha=0.55)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_color(COLORS["light_gray"])
        spine.set_linewidth(0.8)


def generate_release_chart(output_dir: Path, preview_dir: Path | None) -> None:
    points = load_release_points()
    fig, (ax_size, ax_context) = plt.subplots(
        2,
        1,
        figsize=(14.5, 9.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1], "hspace": 0.12},
    )
    fig.suptitle(
        "주요 Open-weight 모델의 규모와 Context Length 변화",
        fontsize=17,
        fontweight="semibold",
        color=COLORS["navy"],
        y=0.985,
    )

    dense = [point for point in points if point.architecture == "dense"]
    moe = [point for point in points if point.architecture == "MoE"]

    ax_size.scatter(
        [point.release_date for point in dense],
        [point.total_params_b for point in dense],
        s=56,
        marker="o",
        color=COLORS["navy"],
        label="Dense: total = active",
        zorder=4,
    )
    ax_size.scatter(
        [point.release_date for point in moe],
        [point.total_params_b for point in moe],
        s=66,
        marker="D",
        color=COLORS["teal"],
        label="MoE: total parameters",
        zorder=4,
    )

    for point in moe:
        if point.active_params_b is None:
            continue
        ax_size.plot(
            [point.release_date, point.release_date],
            [point.active_params_b, point.total_params_b],
            color=COLORS["light_gray"],
            linewidth=1.1,
            zorder=2,
        )
        ax_size.scatter(
            point.release_date,
            point.active_params_b,
            s=42,
            marker="o",
            facecolor=COLORS["orange"],
            edgecolor="none",
            zorder=5,
        )

    ax_size.scatter([], [], s=42, color=COLORS["orange"], label="MoE: active parameters")

    label_offsets = {
        "Qwen2-72B": (-4, -16),
        "Llama 3.1 405B": (-24, 9),
        "Mistral Large 2": (24, -17),
        "Qwen2.5-72B": (0, -16),
        "Gemma 3 27B": (-12, -17),
        "Llama 4 Scout": (8, 9),
        "Qwen3-235B-A22B": (18, -17),
        "DeepSeek-V4-Pro": (-12, 9),
        "Kimi K3": (0, 9),
    }
    for point in points:
        offset = label_offsets.get(point.model, (0, 8))
        ax_size.annotate(
            SHORT_NAMES[point.model],
            (point.release_date, point.total_params_b),
            xytext=offset,
            textcoords="offset points",
            ha="center",
            va="bottom" if offset[1] >= 0 else "top",
            fontsize=8.6,
            color=COLORS["navy"],
        )

    ax_size.set_yscale("log")
    ax_size.set_ylim(5, 5000)
    ax_size.set_ylabel("파라미터 수 (B, log scale)")
    ax_size.set_title("모델 규모: MoE는 총 파라미터와 활성 파라미터를 분리", loc="left")
    ax_size.legend(loc="upper left", ncol=3, frameon=False)
    style_axis(ax_size)

    for point in points:
        if point.trained_context and point.trained_context != point.advertised_context:
            ax_context.plot(
                [point.release_date, point.release_date],
                [point.trained_context, point.advertised_context],
                color=COLORS["light_gray"],
                linestyle="--",
                linewidth=1.1,
                zorder=2,
            )

    ax_context.scatter(
        [point.release_date for point in points],
        [point.advertised_context for point in points],
        s=54,
        marker="o",
        color=COLORS["red"],
        label="Advertised / supported context",
        zorder=4,
    )
    trained = [point for point in points if point.trained_context is not None]
    ax_context.scatter(
        [point.release_date for point in trained],
        [point.trained_context for point in trained],
        s=52,
        marker="^",
        color=COLORS["navy"],
        label="공개된 trained / native context",
        zorder=5,
    )

    context_label_models = {
        "Llama 2 70B",
        "Mistral 7B v0.1",
        "Llama 3 70B",
        "DeepSeek-V2",
        "Llama 4 Scout",
        "DeepSeek-V4-Pro",
        "Kimi K3",
    }
    for point in points:
        if point.model not in context_label_models:
            continue
        ax_context.annotate(
            human_tokens(point.advertised_context),
            (point.release_date, point.advertised_context),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.4,
            color=COLORS["red"],
        )

    ax_context.set_yscale("log")
    ax_context.set_ylim(2_000, 30_000_000)
    ax_context.set_ylabel("Context length (tokens, log scale)")
    ax_context.set_title(
        "Context 확대: advertised 길이와 학습·native 길이는 동일하지 않을 수 있음",
        loc="left",
    )
    ax_context.legend(loc="upper left", ncol=2, frameon=False)
    style_axis(ax_context)

    ax_context.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax_context.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_context.set_xlabel("출시일")
    plt.setp(ax_context.get_xticklabels(), rotation=35, ha="right")

    fig.text(
        0.01,
        0.012,
        "주: Open-weight 기준. 총/활성 파라미터와 context 정의는 각 공식 발표·모델 카드 기준이며, 미공개 trained length는 표시하지 않음.",
        fontsize=8.8,
        color=COLORS["gray"],
    )
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.12, top=0.90, hspace=0.14)

    output = output_dir / "open-model-release-trends.svg"
    fig.savefig(
        output,
        format="svg",
        bbox_inches="tight",
        metadata={
            "Title": "주요 Open-weight 모델의 규모와 Context Length 변화",
            "Description": "2023년부터 2026년까지 주요 open-weight 모델의 총·활성 파라미터와 context length를 출시일에 따라 비교한 그래프",
            "Creator": "Frontier report chart generator",
        },
    )
    if preview_dir:
        fig.savefig(preview_dir / "open-model-release-trends.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _series(start: int, end: int, formula) -> tuple[np.ndarray, np.ndarray]:
    x = np.geomspace(start, end, 180)
    y = np.asarray([formula(value) for value in x], dtype=float) / (1024**3)
    return x, y


def _plot_curve(
    axis: plt.Axes,
    name: str,
    maximum: int,
    formula,
    color: str,
    linestyle: str = "-",
    linewidth: float = 2.0,
) -> tuple[float, float]:
    x, y = _series(1024, maximum, formula)
    axis.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth, label=name)
    axis.scatter([x[-1]], [y[-1]], s=26, color=color, zorder=5)
    return float(x[-1]), float(y[-1])


def generate_kv_chart(output_dir: Path, preview_dir: Path | None) -> None:
    fig, (ax_full, ax_efficient) = plt.subplots(1, 2, figsize=(15.2, 7.8), sharex=True, sharey=True)
    fig.suptitle(
        "Context Length에 따른 세션당 논리 KV Cache",
        fontsize=17,
        fontweight="semibold",
        color=COLORS["navy"],
        y=0.985,
    )

    full_series = [
        ("Llama 2 70B", 4096, lambda n: n * 2 * 80 * 8 * 128 * 2, COLORS["navy"]),
        ("Mixtral 8×7B", 32768, lambda n: n * 2 * 32 * 8 * 128 * 2, COLORS["teal"]),
        ("Llama 3 70B", 8192, lambda n: n * 2 * 80 * 8 * 128 * 2, COLORS["orange"]),
        ("Llama 3.1 405B", 131072, lambda n: n * 2 * 126 * 8 * 128 * 2, COLORS["red"]),
        ("Qwen2.5 72B", 131072, lambda n: n * 2 * 80 * 8 * 128 * 2, COLORS["purple"]),
        ("Qwen3 235B-A22B", 131072, lambda n: n * 2 * 94 * 4 * 128 * 2, COLORS["green"]),
    ]
    for name, maximum, formula, color in full_series:
        _plot_curve(ax_full, name, maximum, formula, color)

    efficient_series = [
        (
            "Mistral 7B (4K rolling)",
            32768,
            lambda n: min(n, 4096) * 2 * 32 * 8 * 128 * 2,
            COLORS["navy"],
        ),
        (
            "DeepSeek-V3 (MLA)",
            131072,
            lambda n: n * 61 * (512 + 64) * 2,
            COLORS["teal"],
        ),
        (
            "Gemma 3 27B (local/global)",
            131072,
            lambda n: 2 * 16 * 128 * 2 * (10 * n + 52 * min(n, 1024)),
            COLORS["orange"],
        ),
        (
            "Kimi K3 (KDA+MLA)",
            1048576,
            lambda n: 232_316_928 + n * 24 * (512 + 64) * 2,
            COLORS["purple"],
        ),
        (
            "Llama 4 Scout (chunked/full)",
            10485760,
            lambda n: 2 * 8 * 128 * 2 * (12 * n + 36 * min(n, 8192)),
            COLORS["red"],
        ),
    ]
    for name, maximum, formula, color in efficient_series:
        _plot_curve(ax_efficient, name, maximum, formula, color)

    _plot_curve(
        ax_efficient,
        "Mistral 7B full-retention 상한",
        32768,
        lambda n: n * 2 * 32 * 8 * 128 * 2,
        COLORS["gray"],
        linestyle="--",
        linewidth=1.4,
    )

    for axis, title in (
        (ax_full, "Full attention 계열"),
        (ax_efficient, "Window · MLA · Hybrid 계열"),
    ):
        axis.set_xscale("log", base=2)
        axis.set_yscale("log", base=2)
        axis.set_xlim(1024, 16_777_216)
        axis.set_ylim(0.03, 3072)
        axis.set_title(title, loc="left")
        axis.set_xlabel("Context length (tokens, log scale)")
        axis.legend(loc="upper left", frameon=False, fontsize=9)
        style_axis(axis)

    ticks_x = [1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216]
    tick_labels_x = ["1K", "4K", "16K", "64K", "256K", "1M", "4M", "16M"]
    for axis in (ax_full, ax_efficient):
        axis.set_xticks(ticks_x, tick_labels_x)

    ticks_y = [0.03125, 0.125, 0.5, 2, 8, 32, 128, 512, 2048]
    tick_labels_y = ["0.03", "0.125", "0.5", "2", "8", "32", "128", "512", "2,048"]
    ax_full.set_yticks(ticks_y, tick_labels_y)
    ax_full.set_ylabel("세션당 논리 KV cache (GiB, BF16, log scale)")

    ax_efficient.annotate(
        "DeepSeek-V4 Pro\n1M에서 V3.2 대비 KV 10%\n(공식 상대값; 절대 곡선 제외)",
        xy=(1048576, 0.12),
        xytext=(270000, 0.055),
        fontsize=8.7,
        color=COLORS["gray"],
        arrowprops={"arrowstyle": "-", "color": COLORS["light_gray"], "linewidth": 1.0},
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#f5f7f8", "edgecolor": COLORS["light_gray"]},
    )

    fig.text(
        0.01,
        0.012,
        "가정: 1 session, BF16 KV, TP·PP·DP 물리 복제와 allocator/block overhead 제외. Rolling/chunked cache가 과거 KV를 실제 폐기한다고 가정.",
        fontsize=8.8,
        color=COLORS["gray"],
    )
    fig.text(
        0.99,
        0.012,
        "Kimi K2의 BF16 정규화 곡선은 DeepSeek-V3와 거의 중첩되어 생략",
        fontsize=8.8,
        color=COLORS["gray"],
        ha="right",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.955), w_pad=2.6)

    output = output_dir / "open-model-kv-cache-curves.svg"
    fig.savefig(
        output,
        format="svg",
        bbox_inches="tight",
        metadata={
            "Title": "Context Length에 따른 세션당 논리 KV Cache",
            "Description": "주요 open-weight 모델의 attention 구조별 BF16 논리 KV cache 증가 곡선",
            "Creator": "Frontier report chart generator",
        },
    )
    if preview_dir:
        fig.savefig(preview_dir / "open-model-kv-cache-curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.preview_dir:
        args.preview_dir.mkdir(parents=True, exist_ok=True)

    configure_style()
    generate_release_chart(args.output_dir, args.preview_dir)
    generate_kv_chart(args.output_dir, args.preview_dir)


if __name__ == "__main__":
    main()
