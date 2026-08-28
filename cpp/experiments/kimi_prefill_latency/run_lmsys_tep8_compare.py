#!/usr/bin/env python3
"""Compare Frontier Kimi K3 TEP8 PREFILL with the LMSYS Day-0 results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
from typing import Any, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
WARM_PREFIX_WORKLOAD = HERE / "workload_128k_plus_1k.csv"
DP8_CONFIG = HERE / "configs" / "kimi_k3_8gpu_hbm500_batch1.json"
TP8_EP8_CONFIG = (
    HERE / "configs" / "kimi_k3_8gpu_hbm500_tp8_ep8_batch1.json"
)
LMSYS_TEP8_CONFIG = HERE / "configs" / "kimi_k3_gb300_tep8_8k_prefill.json"

LMSYS_SOURCE = "https://www.lmsys.org/blog/2026-07-27-kimi-k3-day0-support/"
LMSYS_TEP8_TOKENS_PER_SECOND_PER_GPU = {
    1: 2118.0,
    4: 2456.0,
    16: 3761.0,
    64: 3642.0,
    128: 3562.0,
    256: 3554.0,
}
LMSYS_TEP8_COST_MS_PER_1K_TOKENS = 31.4
LMSYS_TEP8_COMPUTE_MS_PER_1K_TOKENS = 27.2
LMSYS_TEP8_EXPOSED_COMM_MS_PER_1K_TOKENS = 4.18


def _default_binary() -> Path:
    candidates = (
        REPO_ROOT / "build-release" / "frontier_sim",
        REPO_ROOT / "cpp" / "build" / "Release" / "frontier_sim.exe",
    )
    return next((path for path in candidates if path.is_file()), candidates[0])


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _run(
    binary: Path, config: Path, workload: Path, output_dir: Path
) -> None:
    if output_dir.exists():
        raise FileExistsError(
            f"output directory already exists; choose a new --output-root: {output_dir}"
        )
    subprocess.run(
        [
            str(binary),
            "--config",
            str(config),
            "--workload",
            str(workload),
            "--output-dir",
            str(output_dir),
            "--output-mode",
            "requests",
            "--runtime-validation",
            "false",
            "--gpu-kv-occupancy",
            "false",
        ],
        cwd=REPO_ROOT,
        check=True,
    )


def _request_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _run_warm_prefix_topology(
    binary: Path, output_root: Path, label: str, config: Path
) -> dict[str, Any]:
    output_dir = output_root / label
    _run(binary, config, WARM_PREFIX_WORKLOAD, output_dir)
    rows = _request_rows(output_dir / "requests.csv")
    if len(rows) != 2:
        raise ValueError(f"{label}: expected two requests, found {len(rows)}")
    measured = rows[1]
    if (
        int(measured["cached_prefill_tokens"]) != 131_072
        or int(measured["scheduled_prefill_tokens"]) != 1_024
    ):
        raise ValueError(f"{label}: warm-prefix accounting did not match")
    normalized = _read_json(output_dir / "config.normalized.json")
    cluster = normalized["clusters"]["monolithic"]
    return {
        "topology": label,
        "parallelism": cluster["parallelism"],
        "model_weight_gb_per_gpu": (
            cluster["gpu_memory"]["model_weight_bytes_per_gpu"] / 1e9
        ),
        "prefill_latency_ms": float(measured["prefill_latency_ms"]),
    }


def _write_8k_workload(path: Path, concurrency: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "session_start_at",
                "think_time",
                "num_prefill_tokens",
                "num_decode_tokens",
                "session_id",
                "session_turn_index",
            ]
        )
        writer.writerows(
            (0, 0, 8192, 1, session_id, 0)
            for session_id in range(concurrency)
        )


def _run_lmsys_point(
    binary: Path, output_root: Path, concurrency: int
) -> dict[str, Any]:
    workload = output_root / "workloads" / f"8k_c{concurrency}.csv"
    output_dir = output_root / f"gb300_tep8_c{concurrency}"
    _write_8k_workload(workload, concurrency)
    _run(binary, LMSYS_TEP8_CONFIG, workload, output_dir)

    summary = _read_json(output_dir / "summary.json")
    rows = _request_rows(output_dir / "requests.csv")
    if len(rows) != concurrency or any(
        int(row["scheduled_prefill_tokens"]) != 8192 for row in rows
    ):
        raise ValueError(f"concurrency {concurrency}: incomplete 8K PREFILL run")

    total_tokens = concurrency * 8192
    elapsed_s = float(summary["simulation_window_seconds"])
    components = summary["batch_summary_by_cluster"]["MONOLITHIC"][
        "execution_time_components_ms"
    ]
    communication_ms = sum(
        float(components[key])
        for key in (
            "tp_communication_ms",
            "pp_communication_ms",
            "moe_tp_communication_ms",
            "ep_dispatch_ms",
            "ep_combine_ms",
            "dp_input_communication_ms",
            "dp_output_communication_ms",
            "synchronization_wait_ms",
        )
    )
    component_scale = 1000.0 / total_tokens
    simulated_cost_ms_per_1k = elapsed_s / total_tokens * 1e6
    simulated_per_gpu = (
        float(summary["throughput"]["prompt_tokens_per_second"]) / 8.0
    )
    published_per_gpu = LMSYS_TEP8_TOKENS_PER_SECOND_PER_GPU[concurrency]
    return {
        "concurrency": concurrency,
        "simulated_tokens_per_second_per_gpu": simulated_per_gpu,
        "published_tokens_per_second_per_gpu": published_per_gpu,
        "simulated_to_published_ratio": simulated_per_gpu / published_per_gpu,
        "relative_error_percent": 100.0
        * (simulated_per_gpu - published_per_gpu)
        / published_per_gpu,
        "simulated_cost_ms_per_1k_tokens": simulated_cost_ms_per_1k,
        "simulated_compute_ms_per_1k_tokens": (
            simulated_cost_ms_per_1k - communication_ms * component_scale
        ),
        "simulated_communication_ms_per_1k_tokens": (
            communication_ms * component_scale
        ),
        "mean_prefill_latency_ms": float(summary["latency_ms"]["prefill"]["mean"]),
        "p99_prefill_latency_ms": float(summary["latency_ms"]["prefill"]["p99"]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=_default_binary())
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "kimi_k3_lmsys_tep8_compare",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    binary = args.binary.resolve()
    if not binary.is_file():
        raise SystemExit(f"missing Release simulator binary: {binary}")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    topology_results = [
        _run_warm_prefix_topology(binary, output_root, "dp8_ep8", DP8_CONFIG),
        _run_warm_prefix_topology(
            binary, output_root, "tp8_ep8", TP8_EP8_CONFIG
        ),
    ]
    by_topology = {item["topology"]: item for item in topology_results}
    sweep = [
        _run_lmsys_point(binary, output_root, concurrency)
        for concurrency in LMSYS_TEP8_TOKENS_PER_SECOND_PER_GPU
    ]
    saturated = next(item for item in sweep if item["concurrency"] == 16)
    comparison = {
        "experiment": "Kimi K3 TEP8 PREFILL comparison",
        "published_source": LMSYS_SOURCE,
        "published_contract": {
            "hardware": "2x4 GB300",
            "input_tokens_per_request": 8192,
            "topology": "TP8 attention + EP8 MoE (TEP8)",
            "cost_ms_per_1k_tokens": LMSYS_TEP8_COST_MS_PER_1K_TOKENS,
            "compute_ms_per_1k_tokens": LMSYS_TEP8_COMPUTE_MS_PER_1K_TOKENS,
            "exposed_communication_ms_per_1k_tokens": (
                LMSYS_TEP8_EXPOSED_COMM_MS_PER_1K_TOKENS
            ),
        },
        "warm_prefix_topology_ablation": topology_results,
        "tp8_ep8_to_dp8_ep8_latency_ratio": (
            by_topology["tp8_ep8"]["prefill_latency_ms"]
            / by_topology["dp8_ep8"]["prefill_latency_ms"]
        ),
        "gb300_8k_prefill_sweep": sweep,
        "c16_cost_comparison": {
            "simulated_ms_per_1k_tokens": saturated[
                "simulated_cost_ms_per_1k_tokens"
            ],
            "published_ms_per_1k_tokens": LMSYS_TEP8_COST_MS_PER_1K_TOKENS,
            "simulated_compute_ms_per_1k_tokens": saturated[
                "simulated_compute_ms_per_1k_tokens"
            ],
            "published_compute_ms_per_1k_tokens": (
                LMSYS_TEP8_COMPUTE_MS_PER_1K_TOKENS
            ),
            "simulated_communication_ms_per_1k_tokens": saturated[
                "simulated_communication_ms_per_1k_tokens"
            ],
            "published_communication_ms_per_1k_tokens": (
                LMSYS_TEP8_EXPOSED_COMM_MS_PER_1K_TOKENS
            ),
        },
        "modeling_notes": [
            "The published concurrency curve and cost bars were digitized from fig-pp-prefill.svg.",
            "The simulator uses a 131072-token batch budget inferred from the published TEP8 saturation at concurrency 16.",
            "Vision preprocessing and SGLang host/runtime overhead are outside the simulator model.",
        ],
    }
    output = output_root / "comparison.json"
    output.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(comparison, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
