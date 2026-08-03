#!/usr/bin/env python3
"""Run one of the documented C++ Frontier examples and print key results."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


EXAMPLES = {
    "hello": ("00_hello_colocation_fixed.json", "00_tiny.csv"),
    "dense-analytical": ("01_online_dense_analytical.json", "01_online.csv"),
    "kv-pressure": ("02_kv_pressure.json", "02_pressure.csv"),
    "pdd": ("03_sequential_pdd.json", "00_tiny.csv"),
    "moe": ("04_moe_expert_parallel.json", "00_tiny.csv"),
    "prefix-cache": ("05_session_prefix_cache.json", "05_sessions.csv"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("example", choices=EXAMPLES)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/cpp_examples")
    )
    parser.add_argument(
        "--output-mode",
        choices=("summary", "requests", "full"),
        default="requests",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = Path(__file__).resolve().parent
    config_name, workload_name = EXAMPLES[args.example]
    output_dir = args.output_root / args.example
    command = [
        str(args.binary.resolve()),
        "--config",
        str(root / "configs" / config_name),
        "--workload",
        str(root / "workloads" / workload_name),
        "--output-dir",
        str(output_dir),
        "--output-mode",
        args.output_mode,
    ]
    subprocess.run(command, check=True)

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    latency = summary["latency_ms"]
    throughput = summary["throughput"]
    print(f"example: {args.example}")
    print(f"output:  {output_dir}")
    print(f"requests/s: {throughput['requests_per_second']:.3f}")
    print(
        "latency mean (ms): "
        f"prefill={latency['prefill']['mean']:.3f}, "
        f"ttft={latency['ttft']['mean']:.3f}, "
        f"tpot={latency['tpot']['mean']:.3f}, "
        f"e2e={latency['e2e']['mean']:.3f}"
    )
    print(f"preemptions: {summary['counts']['preemptions']}")
    print(f"prefix hit rate: {summary['prefix_cache']['hit_rate']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
