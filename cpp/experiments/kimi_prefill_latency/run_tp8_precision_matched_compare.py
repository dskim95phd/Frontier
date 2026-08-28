#!/usr/bin/env python3
"""Compare K2/K3 TP8+EP8 with K2-equivalent precision and FlashKDA."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
from typing import Any, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
WORKLOADS = {
    1: HERE / "workload_128k_plus_1k.csv",
    16: HERE / "workload_batch16_128k_plus_1k.csv",
}
CONFIGS = {
    1: {
        "kimi_k2": (
            HERE / "configs" / "kimi_k2_8gpu_hbm500_tp8_ep8_batch1.json"
        ),
        "kimi_k3": (
            HERE
            / "configs"
            / "kimi_k3_8gpu_hbm500_tp8_ep8_k2_precision_flashkda.json"
        ),
    },
    16: {
        "kimi_k2": (
            HERE / "configs" / "kimi_k2_8gpu_hbm500_tp8_ep8_batch16.json"
        ),
        "kimi_k3": (
            HERE
            / "configs"
            / "kimi_k3_8gpu_hbm500_tp8_ep8_k2_precision_flashkda_batch16.json"
        ),
    },
}

PRECISION_DIAGNOSTICS = (
    "attention_weight_element_bytes",
    "attention_activation_element_bytes",
    "dense_weight_element_bytes",
    "dense_activation_element_bytes",
    "routed_expert_weight_element_bytes",
    "routed_expert_activation_element_bytes",
    "latent_moe_projection_weight_element_bytes",
    "latent_moe_projection_activation_element_bytes",
    "shared_expert_weight_element_bytes",
    "shared_expert_activation_element_bytes",
    "router_weight_storage_element_bytes",
    "router_activation_storage_element_bytes",
    "router_compute_element_bytes",
    "kda_snapshot_element_bytes",
    "kv_cache_element_bytes",
)


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


def _read_requests(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _run_one(
    binary: Path,
    output_root: Path,
    label: str,
    config: Path,
    workload: Path,
    batch_size: int,
) -> dict[str, Any]:
    output_dir = output_root / label
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
            "full",
            "--runtime-validation",
            "false",
            "--gpu-kv-occupancy",
            "false",
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    rows = _read_requests(output_dir / "requests.csv")
    if len(rows) != 2 * batch_size:
        raise ValueError(
            f"{label}: expected {2 * batch_size} requests, found {len(rows)}"
        )
    measured_rows = [
        row
        for row in rows
        if int(row["cached_prefill_tokens"]) == 131_072
        and int(row["scheduled_prefill_tokens"]) == 1_024
    ]
    if len(measured_rows) != batch_size:
        raise ValueError(
            f"{label}: expected {batch_size} warm-prefix requests, "
            f"found {len(measured_rows)}"
        )

    trace = _read_json(output_dir / "trace.json")
    stage = trace["batch_stages"][-1]
    diagnostics = trace["analytical_diagnostics"][-1]["values"]
    if (
        int(diagnostics["prefill_request_count"]) != batch_size
        or int(diagnostics["total_tokens"]) != batch_size * 1_024
    ):
        raise ValueError(f"{label}: measured batch shape did not match")
    precision = {key: float(diagnostics[key]) for key in PRECISION_DIAGNOSTICS}
    moe_compute_ms = sum(
        float(stage[key])
        for key in (
            "moe_gating_linear_ms",
            "moe_gating_routing_topk_ms",
            "moe_grouped_gemm_ms",
            "moe_shuffling_ms",
            "moe_post_attention_norm_ms",
        )
    )
    communication_ms = sum(
        float(stage[key])
        for key in (
            "tp_communication_ms",
            "moe_tp_communication_ms",
            "ep_dispatch_ms",
            "ep_combine_ms",
            "dp_input_communication_ms",
            "dp_output_communication_ms",
        )
    )
    normalized = _read_json(output_dir / "config.normalized.json")
    cluster = normalized["clusters"]["monolithic"]
    average_prefill_latency_ms = sum(
        float(row["prefill_latency_ms"]) for row in measured_rows
    ) / batch_size
    batch_duration_ms = float(stage["duration_ms"])
    return {
        "model": label,
        "batch_size": batch_size,
        "total_prefill_tokens": batch_size * 1_024,
        "prefill_latency_ms": average_prefill_latency_ms,
        "average_prefill_latency_ms": average_prefill_latency_ms,
        "batch_duration_ms": batch_duration_ms,
        "batch_prefill_throughput_tokens_per_s": (
            batch_size * 1_024 * 1_000 / batch_duration_ms
        ),
        "prefill_throughput_tokens_per_s_per_gpu": (
            batch_size * 1_024 * 1_000 / batch_duration_ms / 8
        ),
        "dense_attention_ms": float(stage["dense_compute_ms"]),
        "moe_compute_ms": moe_compute_ms,
        "communication_ms": communication_ms,
        "lm_head_ms": float(stage["lm_head_ms"]),
        "kda_projection_ms": float(diagnostics["kda_projection_ms"]),
        "kda_short_conv_ms": float(diagnostics["kda_short_conv_ms"]),
        "kda_recurrent_ms": float(diagnostics["kda_recurrent_ms"]),
        "kda_gate_norm_ms": float(diagnostics["kda_gate_norm_ms"]),
        "model_weight_gb_per_gpu": (
            cluster["gpu_memory"]["model_weight_bytes_per_gpu"] / 1e9
        ),
        "precision_element_bytes": precision,
        "kernel_profile": cluster["execution_model"].get(
            "kernel_profile", "generic"
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=_default_binary())
    parser.add_argument("--batch-size", type=int, choices=sorted(CONFIGS), default=1)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            REPO_ROOT / "outputs" / "kimi_tp8_precision_matched_compare"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    binary = args.binary.resolve()
    if not binary.is_file():
        raise SystemExit(f"missing Release simulator binary: {binary}")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    results = [
        _run_one(
            binary,
            output_root,
            label,
            config.resolve(),
            WORKLOADS[args.batch_size].resolve(),
            args.batch_size,
        )
        for label, config in CONFIGS[args.batch_size].items()
    ]
    by_model = {item["model"]: item for item in results}
    if (
        by_model["kimi_k2"]["precision_element_bytes"]
        != by_model["kimi_k3"]["precision_element_bytes"]
    ):
        raise ValueError("K2 and K3 effective precision diagnostics differ")

    comparison = {
        "experiment": (
            f"TP8/DP1/EP8, 500 GB HBM/GPU, batch {args.batch_size}, "
            "128K cached + 1K PREFILL, K2-equivalent precision"
        ),
        "results": results,
        "k3_to_k2_latency_ratio": (
            by_model["kimi_k3"]["average_prefill_latency_ms"]
            / by_model["kimi_k2"]["average_prefill_latency_ms"]
        ),
        "precision_match_validated": True,
        "notes": [
            "K3 uses k3_flashkda_prefill, which changes only KDA prefill.",
            "No other k3_sglang_mxfp4 efficiency or fusion priors are enabled.",
        ],
    }
    output = output_root / "comparison.json"
    output.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(comparison, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
