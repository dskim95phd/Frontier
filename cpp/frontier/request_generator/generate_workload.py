#!/usr/bin/env python3
"""Generate a normalized workload CSV for the C++ Frontier simulator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

if __package__:
    from .workload_generator import (
        generate_requests,
        interval_distribution_from_config,
        length_distribution_from_config,
        write_workload_csv,
    )
else:
    from workload_generator import (
        generate_requests,
        interval_distribution_from_config,
        length_distribution_from_config,
        write_workload_csv,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def generate_from_config(config: object, output: Path) -> None:
    if not isinstance(config, dict):
        raise ValueError("workload config must be a JSON object")
    schema_version = config.get("schema_version")
    if isinstance(schema_version, bool) or schema_version != 1:
        raise ValueError("workload config schema_version must be 1")

    num_requests = config.get("num_requests")
    seed = config.get("seed", 42)
    if isinstance(num_requests, bool) or not isinstance(num_requests, int):
        raise ValueError("num_requests must be a positive integer")
    if num_requests <= 0:
        raise ValueError("num_requests must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    length_config = config.get("length")
    interval_config = config.get("interval")
    if not isinstance(length_config, dict):
        raise ValueError("length must be a JSON object")
    if not isinstance(interval_config, dict):
        raise ValueError("interval must be a JSON object")

    first_arrival_at_zero = config.get("first_arrival_at_zero", True)
    if not isinstance(first_arrival_at_zero, bool):
        raise ValueError("first_arrival_at_zero must be a boolean")

    requests = generate_requests(
        num_requests=num_requests,
        seed=seed,
        length_distribution=length_distribution_from_config(
            length_config, seed=seed
        ),
        interval_distribution=interval_distribution_from_config(interval_config),
        first_arrival_at_zero=first_arrival_at_zero,
    )
    write_workload_csv(output, requests)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        generate_from_config(config, args.output)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise SystemExit(f"generate_workload: error: {error}") from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
