#!/usr/bin/env python3
"""Generate a deterministic, closed-loop Kimi K2 session workload.

The CSV is consumed directly by Frontier's session-prefix trace replay path.
Each row stores only newly added prompt tokens; Frontier materializes the full
prompt as prior context + prior output + the new prompt delta.

Only the first turn of each session has an absolute ``arrived_at``. Every
subsequent turn has a ``think_time`` and is released at:

    previous turn completion + think_time
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class WorkloadConfig:
    sessions: int = 24
    seed: int = 20260728
    session_arrival_rate: float = 0.75
    max_context_tokens: int = 65_536
    geometric_p: float = 0.25
    max_turns: int = 12
    initial_median: int = 8_192
    initial_sigma: float = 0.8
    initial_min: int = 2_048
    initial_max: int = 32_768
    output_median: int = 8
    output_sigma: float = 0.7
    output_min: int = 4
    output_max: int = 16
    new_context_median: int = 2_048
    new_context_sigma: float = 0.9
    new_context_min: int = 128
    new_context_max: int = 8_192
    think_time_profile: str = "mixed_human"


def _bounded_lognormal(
    rng: np.random.Generator,
    *,
    median: float,
    sigma: float,
    lower: int,
    upper: int,
) -> int:
    sample = math.exp(rng.normal(math.log(median), sigma))
    return int(min(upper, max(lower, round(sample))))


def _sample_gap_seconds(
    rng: np.random.Generator,
    profile: str,
) -> tuple[float, str]:
    draw = float(rng.random())
    if profile == "tool_seconds":
        if draw < 0.10:
            label, median, sigma, lower, upper = (
                "fast_tool",
                3.0,
                0.45,
                1.0,
                6.0,
            )
        elif draw < 0.80:
            label, median, sigma, lower, upper = (
                "normal_tool",
                12.0,
                0.5,
                4.0,
                25.0,
            )
        else:
            label, median, sigma, lower, upper = (
                "slow_tool",
                30.0,
                0.5,
                12.0,
                60.0,
            )
    elif profile != "mixed_human":
        raise ValueError(
            "think_time_profile must be 'mixed_human' or 'tool_seconds', "
            f"got={profile!r}"
        )
    elif draw < 0.65:
        label, median, sigma, lower, upper = (
            "fast_tool",
            0.2,
            0.8,
            0.01,
            2.0,
        )
    elif draw < 0.90:
        label, median, sigma, lower, upper = (
            "slow_tool",
            5.0,
            1.0,
            0.5,
            60.0,
        )
    else:
        label, median, sigma, lower, upper = (
            "human_thinking",
            120.0,
            0.8,
            10.0,
            600.0,
        )
    value = math.exp(rng.normal(math.log(median), sigma))
    return float(min(upper, max(lower, value))), label


def generate_rows(config: WorkloadConfig) -> tuple[list[dict[str, object]], dict]:
    if config.sessions <= 0:
        raise ValueError("sessions must be positive")
    if config.session_arrival_rate <= 0:
        raise ValueError("session_arrival_rate must be positive")
    if config.max_context_tokens <= 0:
        raise ValueError("max_context_tokens must be positive")

    rng = np.random.default_rng(config.seed)
    session_start = 0.0
    rows: list[dict[str, object]] = []
    effective_prompt_lengths: list[int] = []
    output_lengths: list[int] = []
    realized_turns: list[int] = []
    final_context_lengths: list[int] = []
    think_times: list[float] = []
    gap_counts: dict[str, int] = {}

    for session_id in range(config.sessions):
        session_start += float(rng.exponential(1.0 / config.session_arrival_rate))
        planned_turns = min(
            config.max_turns,
            2 + int(rng.geometric(config.geometric_p)),
        )
        context = 0
        turns = 0
        pending_think_time = 0.0
        pending_gap_label: str | None = None

        for turn_index in range(planned_turns):
            if turn_index == 0:
                new_prompt = _bounded_lognormal(
                    rng,
                    median=config.initial_median,
                    sigma=config.initial_sigma,
                    lower=config.initial_min,
                    upper=config.initial_max,
                )
            else:
                new_prompt = _bounded_lognormal(
                    rng,
                    median=config.new_context_median,
                    sigma=config.new_context_sigma,
                    lower=config.new_context_min,
                    upper=config.new_context_max,
                )
            output = _bounded_lognormal(
                rng,
                median=config.output_median,
                sigma=config.output_sigma,
                lower=config.output_min,
                upper=config.output_max,
            )

            effective_prompt = context + new_prompt
            if effective_prompt + output > config.max_context_tokens:
                break

            rows.append(
                {
                    "arrived_at": (
                        round(session_start, 9) if turn_index == 0 else ""
                    ),
                    "think_time": round(pending_think_time, 9),
                    "turn_index": turn_index,
                    "num_prefill_tokens": new_prompt,
                    "num_decode_tokens": output,
                    "session_id": session_id,
                    "cohort": "kimi_k2_64k",
                    "_effective_prompt_tokens": effective_prompt,
                }
            )
            if turn_index > 0:
                think_times.append(pending_think_time)
                assert pending_gap_label is not None
                gap_counts[pending_gap_label] = (
                    gap_counts.get(pending_gap_label, 0) + 1
                )
            effective_prompt_lengths.append(effective_prompt)
            output_lengths.append(output)
            context = effective_prompt + output
            turns += 1

            if turn_index < planned_turns - 1:
                pending_think_time, pending_gap_label = _sample_gap_seconds(
                    rng,
                    config.think_time_profile,
                )

        realized_turns.append(turns)
        final_context_lengths.append(context)

    def distribution(values: list[int | float]) -> dict[str, float]:
        array = np.asarray(values, dtype=float)
        if array.size == 0:
            return {"count": 0}
        return {
            "count": int(array.size),
            "mean": float(array.mean()),
            "p50": float(np.percentile(array, 50)),
            "p90": float(np.percentile(array, 90)),
            "p99": float(np.percentile(array, 99)),
            "max": float(array.max()),
        }

    manifest = {
        "schema_version": 2,
        "description": (
            "Closed-loop, append-only, session-affine Kimi K2 workload with "
            "Poisson first-turn arrivals, post-completion think times, and a "
            f"hard {config.max_context_tokens:,}-token context ceiling."
        ),
        "config": asdict(config),
        "realized": {
            "sessions": config.sessions,
            "requests": len(rows),
            "session_start_horizon_seconds": float(session_start),
            "turns_per_session": distribution(realized_turns),
            "effective_prompt_tokens": distribution(effective_prompt_lengths),
            "output_tokens": distribution(output_lengths),
            "final_context_tokens": distribution(final_context_lengths),
            "think_time_seconds": distribution(think_times),
            "gap_class_counts": gap_counts,
        },
    }
    return rows, manifest


def write_workload(
    csv_path: Path,
    manifest_path: Path,
    config: WorkloadConfig,
) -> dict:
    rows, manifest = generate_rows(config)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=(
                "arrived_at",
                "think_time",
                "turn_index",
                "num_prefill_tokens",
                "num_decode_tokens",
                "session_id",
                "cohort",
            ),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: row[key]
                    for key in writer.fieldnames
                }
            )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--sessions", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--session-arrival-rate", type=float, default=0.75)
    parser.add_argument("--max-context-tokens", type=int, default=65_536)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    manifest = write_workload(
        args.output.resolve(),
        manifest_path.resolve(),
        WorkloadConfig(
            sessions=args.sessions,
            seed=args.seed,
            session_arrival_rate=args.session_arrival_rate,
            max_context_tokens=args.max_context_tokens,
        ),
    )
    print(
        f"wrote {manifest['realized']['requests']} requests across "
        f"{manifest['realized']['sessions']} sessions to {args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
