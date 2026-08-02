#!/usr/bin/env python3
"""Run a single-turn load, batch-size, and KV-capacity sweep.

Request sizes follow the first-turn distribution from main's
generate_kimi_k2_probabilistic_workload.py. Both the historical Llama-2-7B
baseline and a Kimi K2 / GB300 PDD rack profile are available.
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
REQUEST_GENERATOR_DIR = REPO_ROOT / "cpp" / "frontier" / "request_generator"
sys.path.insert(0, str(REQUEST_GENERATOR_DIR))

from workload_generator import (  # noqa: E402
    BoundedLognormalLengthDistribution,
    PoissonIntervalDistribution,
    RequestShape,
    generate_request_shapes,
    materialize_requests,
    write_workload_csv,
)


BASE_CONFIG = (
    REPO_ROOT
    / "cpp"
    / "tests"
    / "fixtures"
    / "config"
    / "analytical_parallel_colocation.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "cpp_single_turn_load_sweep"
BLOCK_SIZE = 16
GIB = 1024**3
PRECISION_BYTES = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
    "fp4": 0.5,
    "int4": 0.5,
}
PRECISION_PROFILES = {
    "fp16": {
        "attention": "fp16",
        "dense": "fp16",
        "moe_expert": "fp16",
        "moe_router": "fp16",
        "kv_cache": "fp16",
        "communication": "fp16",
    },
    "fp8-fp4-mixed": {
        "attention": "fp8",
        "dense": "fp8",
        "moe_expert": "fp8",
        "moe_expert_weight": "fp4",
        "moe_expert_activation": "fp8",
        "moe_router": "fp8",
        "lm_head": "fp8",
        "kv_cache": "fp8",
        "communication": "fp8",
    },
    "fp8": {
        "attention": "fp8",
        "dense": "fp8",
        "moe_expert": "fp8",
        "moe_expert_weight": "fp8",
        "moe_expert_activation": "fp8",
        "moe_router": "fp8",
        "lm_head": "fp8",
        "kv_cache": "fp8",
        "communication": "fp8",
    },
}


@dataclass(frozen=True)
class TopologyConfig:
    name: str
    system_architecture: str
    tensor_parallel_size: int
    pipeline_parallel_size: int
    data_parallel_size: int = 2
    num_replicas: int = 1
    cluster_count: int = 1
    prefill_tensor_parallel_size: int | None = None
    prefill_pipeline_parallel_size: int | None = None
    prefill_data_parallel_size: int | None = None
    prefill_num_replicas: int | None = None
    moe_tensor_parallel_size: int = 1
    moe_expert_parallel_size: int = 1
    prefill_moe_tensor_parallel_size: int | None = None
    prefill_moe_expert_parallel_size: int | None = None
    model_name: str = "meta-llama/Llama-2-7b-hf"
    device: str = "rubin"
    moe_layer_event_mode: str = "detailed"
    moe_routing_distribution: str | None = None

    def parallelism(self, *, decode: bool) -> tuple[int, int, int, int]:
        if decode or self.system_architecture == "co-location":
            return (
                self.num_replicas,
                self.tensor_parallel_size,
                self.pipeline_parallel_size,
                self.data_parallel_size,
            )
        return (
            self.prefill_num_replicas or self.num_replicas,
            self.prefill_tensor_parallel_size or self.tensor_parallel_size,
            self.prefill_pipeline_parallel_size or self.pipeline_parallel_size,
            self.prefill_data_parallel_size or self.data_parallel_size,
        )

    def moe_parallelism(self, *, decode: bool) -> tuple[int, int]:
        if decode or self.system_architecture == "co-location":
            return (
                self.moe_tensor_parallel_size,
                self.moe_expert_parallel_size,
            )
        return (
            self.prefill_moe_tensor_parallel_size
            or self.moe_tensor_parallel_size,
            self.prefill_moe_expert_parallel_size
            or self.moe_expert_parallel_size,
        )

    @property
    def gpu_count_per_cluster(self) -> int:
        return math.prod(self.parallelism(decode=True))

    @property
    def decode_gpu_count(self) -> int:
        return math.prod(self.parallelism(decode=True))

    @property
    def prefill_gpu_count(self) -> int:
        if self.system_architecture == "co-location":
            return 0
        return math.prod(self.parallelism(decode=False))

    @property
    def physical_gpu_count(self) -> int:
        if self.system_architecture == "co-location":
            return self.decode_gpu_count
        return self.prefill_gpu_count + self.decode_gpu_count

    def kv_bytes_per_token_per_gpu(self, dtype_size_bytes: float) -> int:
        if self.model_name == "moonshotai/Kimi-K2-Instruct":
            # Latent MLA cache: 512 FP8 latent elements plus 64 FP16 RoPE
            # elements per token and layer. Account for the largest PP stage.
            _, _, pipeline_parallel, _ = self.parallelism(decode=True)
            layers_per_stage = math.ceil(61 / pipeline_parallel)
            return round(
                layers_per_stage * (512 * dtype_size_bytes + 64 * 2.0)
            )
        # Llama-2-7B: 32 layers, 32 KV heads, head_dim 128, K+V.
        byte_count = (
            (32 // self.pipeline_parallel_size)
            * (32 // self.tensor_parallel_size)
            * 128
            * 2
            * dtype_size_bytes
        )
        if not float(byte_count).is_integer():
            raise ValueError("KV bytes per token must be integral")
        return int(byte_count)


TOPOLOGIES = {
    "colocation": TopologyConfig(
        name="colocation",
        system_architecture="co-location",
        tensor_parallel_size=8,
        pipeline_parallel_size=4,
    ),
    "pdd-pp-half": TopologyConfig(
        name="pdd-pp-half",
        system_architecture="pd-disaggregation",
        tensor_parallel_size=8,
        pipeline_parallel_size=2,
        cluster_count=2,
    ),
    "pdd-tp-half": TopologyConfig(
        name="pdd-tp-half",
        system_architecture="pd-disaggregation",
        tensor_parallel_size=4,
        pipeline_parallel_size=4,
        cluster_count=2,
    ),
    "pdd-decode-tradeoff": TopologyConfig(
        name="pdd-decode-tradeoff",
        system_architecture="pd-disaggregation",
        tensor_parallel_size=4,
        pipeline_parallel_size=4,
        data_parallel_size=2,
        cluster_count=2,
        prefill_tensor_parallel_size=4,
        prefill_pipeline_parallel_size=4,
        prefill_data_parallel_size=8,
    ),
    "kimi-k2-gb300-pdd-64": TopologyConfig(
        name="kimi-k2-gb300-pdd-64",
        system_architecture="pd-disaggregation",
        tensor_parallel_size=4,
        pipeline_parallel_size=4,
        data_parallel_size=4,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=16,
        num_replicas=1,
        cluster_count=2,
        prefill_tensor_parallel_size=4,
        prefill_pipeline_parallel_size=4,
        prefill_data_parallel_size=1,
        prefill_num_replicas=4,
        prefill_moe_tensor_parallel_size=1,
        prefill_moe_expert_parallel_size=4,
        model_name="moonshotai/Kimi-K2-Instruct",
        device="gb300",
        moe_layer_event_mode="first_layer_scaled",
        moe_routing_distribution="balanced",
    ),
}


@dataclass(frozen=True)
class WorkloadConfig:
    requests: int = 64
    seed: int = 20260728
    prompt_median: int = 8_192
    prompt_sigma: float = 0.8
    prompt_min: int = 2_048
    prompt_max: int = 32_768
    output_median: int = 1_024
    output_sigma: float = 0.7
    output_min: int = 256
    output_max: int = 4_096


@dataclass(frozen=True)
class WorkloadProfile:
    name: str
    prompt_median: int
    prompt_sigma: float
    prompt_min: int
    prompt_max: int
    output_median: int
    output_sigma: float
    output_min: int
    output_max: int


WORKLOAD_PROFILES = {
    "kimi-short": WorkloadProfile(
        name="kimi-short",
        prompt_median=8_192,
        prompt_sigma=0.8,
        prompt_min=2_048,
        prompt_max=32_768,
        output_median=8,
        output_sigma=0.7,
        output_min=4,
        output_max=16,
    ),
    "prefill-heavy": WorkloadProfile(
        name="prefill-heavy",
        prompt_median=8_192,
        prompt_sigma=0.8,
        prompt_min=2_048,
        prompt_max=32_768,
        output_median=256,
        output_sigma=0.7,
        output_min=64,
        output_max=1_024,
    ),
    "balanced-8k-1k": WorkloadProfile(
        name="balanced-8k-1k",
        prompt_median=8_192,
        prompt_sigma=0.8,
        prompt_min=2_048,
        prompt_max=32_768,
        output_median=1_024,
        output_sigma=0.7,
        output_min=256,
        output_max=4_096,
    ),
    "balanced-4k-4k": WorkloadProfile(
        name="balanced-4k-4k",
        # These medians produce 4,096.117 ISL and 4,096.000 OSL means for
        # the canonical 1,000-request seed while retaining stochastic lengths.
        prompt_median=3_414,
        prompt_sigma=0.55,
        prompt_min=1_024,
        prompt_max=16_384,
        output_median=3_512,
        output_sigma=0.55,
        output_min=1_024,
        output_max=16_384,
    ),
}


def generate_shapes(config: WorkloadConfig) -> list[RequestShape]:
    return generate_request_shapes(
        num_requests=config.requests,
        seed=config.seed,
        length_distribution=BoundedLognormalLengthDistribution(
            prefill_median=config.prompt_median,
            prefill_sigma=config.prompt_sigma,
            prefill_min=config.prompt_min,
            prefill_max=config.prompt_max,
            decode_median=config.output_median,
            decode_sigma=config.output_sigma,
            decode_min=config.output_min,
            decode_max=config.output_max,
        ),
        interval_distribution=PoissonIntervalDistribution(qps=1.0),
    )


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(
        ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
    )


def parse_int_list(value: str) -> list[int]:
    try:
        parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected comma-separated integers"
        ) from error
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("all values must be positive")
    if len(parsed) != len(set(parsed)):
        raise argparse.ArgumentTypeError("values must be unique")
    return parsed


def parse_float_list(value: str) -> list[float]:
    try:
        parsed = [
            float(item.strip()) for item in value.split(",") if item.strip()
        ]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected comma-separated numbers"
        ) from error
    if not parsed or any(not math.isfinite(item) or item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("all values must be finite and positive")
    if len(parsed) != len(set(parsed)):
        raise argparse.ArgumentTypeError("values must be unique")
    return parsed


def write_workload(
    path: Path,
    shapes: Sequence[RequestShape],
    *,
    arrival_rate: float | None,
    isolated_gap_seconds: float = 10.0,
) -> None:
    requests = materialize_requests(
        shapes,
        interval_scale=1.0 if arrival_rate is None else 1.0 / arrival_rate,
        fixed_interval_seconds=(
            isolated_gap_seconds if arrival_rate is None else None
        ),
    )
    write_workload_csv(path, requests, arrival_decimal_places=12)


def write_config(
    path: Path,
    *,
    run_id: str,
    prefill_batch_size_cap: int,
    decode_batch_size_cap: int,
    max_tokens_in_batch: int,
    prefill_chunk_tokens: int,
    num_blocks: int,
    topology: TopologyConfig,
    operator_precisions: dict[str, str],
    closed_loop_max_concurrency: int = 0,
) -> None:
    config = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    config["run_id"] = run_id
    config["simulation_mode"] = "online"
    if closed_loop_max_concurrency > 0:
        config["closed_loop_max_concurrency"] = closed_loop_max_concurrency
    base_cluster = config["clusters"]["monolithic"]

    def make_cluster(*, decode: bool) -> dict:
        cluster = copy.deepcopy(base_cluster)
        replicas, tensor_parallel, pipeline_parallel, data_parallel = (
            topology.parallelism(decode=decode)
        )
        moe_tensor_parallel, moe_expert_parallel = topology.moe_parallelism(
            decode=decode
        )
        cluster["parallelism"].update(
            {
                "num_replicas": replicas,
                "tensor_parallel_size": tensor_parallel,
                "pipeline_parallel_size": pipeline_parallel,
                "data_parallel_size": data_parallel,
                "moe_tensor_parallel_size": moe_tensor_parallel,
                "moe_expert_parallel_size": moe_expert_parallel,
            }
        )
        cluster["scheduler"].update(
            {
                "batch_size_cap": (
                    decode_batch_size_cap
                    if decode or topology.system_architecture == "co-location"
                    else prefill_batch_size_cap
                ),
                "max_tokens_in_batch": max_tokens_in_batch,
                "enable_preemption": True,
                "enable_chunked_prefill": not decode,
                "long_prefill_token_threshold": (
                    0 if decode else prefill_chunk_tokens
                ),
                "block_size": BLOCK_SIZE,
                "num_blocks": num_blocks,
                "watermark_blocks_fraction": 0.0,
                "num_preallocate_tokens": 0,
            }
        )
        cluster["execution_model"]["operator_precisions"] = dict(
            operator_precisions
        )
        cluster["execution_model"]["precision"] = operator_precisions[
            "attention"
        ]
        cluster["execution_model"]["device"] = topology.device
        cluster["execution_model"]["moe_layer_event_mode"] = (
            topology.moe_layer_event_mode
        )
        if topology.moe_routing_distribution is not None:
            cluster["moe_routing"]["distribution"] = (
                topology.moe_routing_distribution
            )
        cluster["model_name"] = topology.model_name
        cluster.pop("total_expert_num", None)
        cluster.pop("router_topk", None)
        return cluster

    config["system_architecture"] = topology.system_architecture
    if topology.system_architecture == "co-location":
        config["clusters"] = {"monolithic": make_cluster(decode=False)}
    else:
        config["clusters"] = {
            "prefill": make_cluster(decode=False),
            "decode": make_cluster(decode=True),
        }
        # Match the high-bandwidth NVL12 assumption used by main's Kimi K2
        # sweep so that the experiment measures the compute split instead of
        # an arbitrary low-bandwidth transfer bottleneck.
        config["kv_cache_transfer"] = {
            "type": "analytical",
            "network_bandwidth_gbps": 38_400.0,
            "network_latency_ms": 0.02,
            "kv_cache_dtype_size_bytes": PRECISION_BYTES[
                operator_precisions["kv_cache"]
            ],
            "enable_compression": False,
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def run_simulator(
    binary: Path,
    config_path: Path,
    workload_path: Path,
    *,
    timeout_seconds: float,
) -> tuple[dict, float]:
    started_at = time.perf_counter()
    summary_path = config_path.with_suffix(".summary.json")
    summary_path.unlink(missing_ok=True)
    completed = subprocess.run(
        [
            str(binary),
            "--config",
            str(config_path),
            "--workload",
            str(workload_path),
            "--output",
            str(summary_path),
        ],
        cwd=REPO_ROOT,
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    process_seconds = time.perf_counter() - started_at
    if completed.returncode != 0:
        raise RuntimeError(
            f"simulator failed ({completed.returncode}): "
            f"{completed.stderr.strip()}"
        )
    if not summary_path.is_file():
        raise RuntimeError("simulator did not write its summary output")
    try:
        return (
            json.loads(summary_path.read_text(encoding="utf-8")),
            process_seconds,
        )
    finally:
        summary_path.unlink(missing_ok=True)


def summarize_case(
    output: dict,
    shapes: Sequence[RequestShape],
    *,
    offered_concurrency: float,
    arrival_rate: float,
    prefill_batch_size_cap: int,
    decode_batch_size_cap: int,
    decode_batch_cap_enabled: bool,
    max_tokens_in_batch: int,
    prefill_chunk_tokens: int,
    measurement_trim_fraction: float,
    num_blocks: int,
    topology: TopologyConfig,
    kv_bytes_per_token_per_gpu: int,
    process_seconds: float,
) -> dict[str, float | int | str]:
    requests = output["requests"]
    batch_sizes = [
        int(batch_size)
        for batch_size, count in output["batch_size_histogram"].items()
        for _ in range(int(count))
    ]
    batch_histograms = output.get("batch_size_histogram_by_cluster", {})

    def materialize_batch_sizes(cluster_type: str) -> list[int]:
        return [
            int(batch_size)
            for batch_size, count in batch_histograms.get(
                cluster_type, {}
            ).items()
            for _ in range(int(count))
        ]

    prefill_batch_sizes = materialize_batch_sizes("PREFILL")
    decode_batch_sizes = materialize_batch_sizes("DECODE")
    batch_summaries = output.get("batch_summary_by_cluster", {})
    transfers = output.get("kv_cache_transfers", [])
    if len(requests) != len(shapes):
        raise RuntimeError(
            f"expected {len(shapes)} requests, got {len(requests)}"
        )
    ordered_requests = sorted(
        requests,
        key=lambda row: int(row["request_id"]),
    )
    trim_count = int(len(ordered_requests) * measurement_trim_fraction)
    measurement_requests = ordered_requests[
        trim_count : len(ordered_requests) - trim_count
    ]
    if not measurement_requests:
        raise RuntimeError("measurement window contains no requests")

    first_arrival = min(float(row["arrived_at_s"]) for row in requests)
    last_completion = max(float(row["completed_at_s"]) for row in requests)
    simulation_window = last_completion - first_arrival
    if simulation_window <= 0.0:
        raise RuntimeError("simulation window must be positive")

    ttft = [float(row["ttft_ms"]) for row in requests]
    measurement_ttft = [
        float(row["ttft_ms"]) for row in measurement_requests
    ]
    first_token_latency = [
        (
            float(row["first_token_completed_at_s"])
            - float(row["arrived_at_s"])
        )
        * 1_000.0
        for row in requests
    ]
    prefill_service = [
        (
            float(row["prefill_completed_at_s"])
            - float(row["first_scheduled_at_s"])
        )
        * 1_000.0
        for row in requests
    ]
    tpot = []
    user_decode_tps = []
    for row in requests:
        request_id = int(row["request_id"])
        output_tokens = shapes[request_id].output_tokens
        if output_tokens <= 1:
            continue
        decode_tail_ms = (
            float(row["completed_at_s"])
            - float(row["first_token_completed_at_s"])
        ) * 1_000.0
        tpot.append(decode_tail_ms / (output_tokens - 1))
    for row in measurement_requests:
        request_id = int(row["request_id"])
        output_tokens = shapes[request_id].output_tokens
        decode_tail_seconds = float(row["completed_at_s"]) - float(
            row["first_token_completed_at_s"]
        )
        if output_tokens > 1 and decode_tail_seconds > 0.0:
            user_decode_tps.append(
                (output_tokens - 1) / decode_tail_seconds
            )
    e2e_seconds = [float(row["e2e_ms"]) / 1_000.0 for row in requests]
    scheduling = [float(row["scheduling_delay_ms"]) for row in requests]
    total_prompt = sum(shape.prompt_tokens for shape in shapes)
    total_decode = sum(shape.output_tokens for shape in shapes)
    total_tokens = total_prompt + total_decode
    total_preemptions = sum(
        int(row["preemption_count"]) for row in requests
    )
    transfer_times = [
        (
            float(row["completed_at_s"])
            - float(row["started_at_s"])
        )
        * 1_000.0
        for row in transfers
    ]

    # Integral of the in-flight request count divided by the total window.
    realized_mean_concurrency = sum(e2e_seconds) / simulation_window
    kv_token_capacity_per_target = num_blocks * BLOCK_SIZE
    kv_gib_per_gpu = (
        kv_token_capacity_per_target
        * kv_bytes_per_token_per_gpu
        / GIB
    )

    return {
        "status": "ok",
        "topology": topology.name,
        "offered_concurrency": offered_concurrency,
        "arrival_rate_rps": arrival_rate,
        "realized_mean_concurrency": realized_mean_concurrency,
        "batch_size_cap": decode_batch_size_cap,
        "prefill_batch_size_cap": prefill_batch_size_cap,
        "decode_batch_size_cap": decode_batch_size_cap,
        "decode_batch_cap_enabled": decode_batch_cap_enabled,
        "max_tokens_in_batch": max_tokens_in_batch,
        "prefill_chunk_tokens": prefill_chunk_tokens,
        "measurement_trim_fraction": measurement_trim_fraction,
        "measurement_requests": len(measurement_requests),
        "num_blocks_per_dp_target": num_blocks,
        "kv_token_capacity_per_dp_target": kv_token_capacity_per_target,
        "kv_capacity_gib_per_gpu": kv_gib_per_gpu,
        "requests": len(requests),
        "simulation_window_s": simulation_window,
        "throughput_requests_per_s": len(requests) / simulation_window,
        "total_tokens_per_s": total_tokens / simulation_window,
        "prompt_tokens_per_s": total_prompt / simulation_window,
        "decode_tokens_per_s": total_decode / simulation_window,
        "decode_tokens_per_gpu_s": (
            total_decode / simulation_window / topology.decode_gpu_count
        ),
        "ttft_mean_ms": statistics.fmean(ttft),
        "ttft_p50_ms": percentile(ttft, 0.50),
        "ttft_p90_ms": percentile(ttft, 0.90),
        "ttft_p99_ms": percentile(ttft, 0.99),
        "measurement_ttft_mean_ms": statistics.fmean(measurement_ttft),
        "measurement_ttft_p50_ms": percentile(measurement_ttft, 0.50),
        "measurement_ttft_p90_ms": percentile(measurement_ttft, 0.90),
        "measurement_ttft_p99_ms": percentile(measurement_ttft, 0.99),
        "first_token_latency_mean_ms": statistics.fmean(first_token_latency),
        "first_token_latency_p50_ms": percentile(first_token_latency, 0.50),
        "first_token_latency_p90_ms": percentile(first_token_latency, 0.90),
        "first_token_latency_p99_ms": percentile(first_token_latency, 0.99),
        "prefill_service_mean_ms": statistics.fmean(prefill_service),
        "prefill_service_p90_ms": percentile(prefill_service, 0.90),
        "tpot_mean_ms": statistics.fmean(tpot) if tpot else 0.0,
        "tpot_p50_ms": percentile(tpot, 0.50),
        "tpot_p90_ms": percentile(tpot, 0.90),
        "tpot_p99_ms": percentile(tpot, 0.99),
        "user_decode_tps_mean": (
            statistics.fmean(user_decode_tps) if user_decode_tps else 0.0
        ),
        "user_decode_tps_p50": percentile(user_decode_tps, 0.50),
        "user_decode_tps_p10": percentile(user_decode_tps, 0.10),
        "e2e_mean_ms": statistics.fmean(e2e_seconds) * 1_000.0,
        "scheduling_delay_mean_ms": statistics.fmean(scheduling),
        "preemptions": total_preemptions,
        "requests_preempted": sum(
            int(int(row["preemption_count"]) > 0) for row in requests
        ),
        "kv_transfers": len(transfers),
        "kv_transfer_total_gib": (
            sum(int(row["size_bytes"]) for row in transfers) / GIB
        ),
        "kv_transfer_mean_ms": (
            statistics.fmean(transfer_times) if transfer_times else 0.0
        ),
        "kv_transfer_p90_ms": percentile(transfer_times, 0.90),
        "batches": len(batch_sizes),
        "mean_realized_batch_size": (
            statistics.fmean(batch_sizes) if batch_sizes else 0.0
        ),
        "batch_size_p90": percentile(batch_sizes, 0.90),
        "prefill_batches": len(prefill_batch_sizes),
        "mean_prefill_batch_size": (
            statistics.fmean(prefill_batch_sizes)
            if prefill_batch_sizes
            else 0.0
        ),
        "prefill_batch_size_p90": percentile(prefill_batch_sizes, 0.90),
        "decode_batches": len(decode_batch_sizes),
        "mean_decode_batch_size": (
            statistics.fmean(decode_batch_sizes)
            if decode_batch_sizes
            else 0.0
        ),
        "decode_batch_size_p50": percentile(decode_batch_sizes, 0.50),
        "decode_batch_size_p90": percentile(decode_batch_sizes, 0.90),
        "decode_time_weighted_mean_batch_size": float(
            batch_summaries.get("DECODE", {}).get(
                "execution_time_weighted_mean_batch_size", 0.0
            )
        ),
        "events": int(output["counts"]["events"]),
        "batch_stages": int(output["counts"]["batch_stages"]),
        "simulator_core_wall_s": float(output["wall_clock_seconds"]),
        "process_wall_s": process_seconds,
    }


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_tradeoff_html(path: Path, rows: Sequence[dict]) -> None:
    successful = sorted(
        (
            row
            for row in rows
            if row.get("status") == "ok"
            and "mean_decode_batch_size" in row
        ),
        key=lambda row: float(row["mean_decode_batch_size"]),
    )
    if not successful:
        return

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    decode_batch = [float(row["mean_decode_batch_size"]) for row in successful]
    offered = [float(row["offered_concurrency"]) for row in successful]
    customdata = [
        [
            float(row["offered_concurrency"]),
            float(row["realized_mean_concurrency"]),
            float(row["decode_batch_size_p90"]),
        ]
        for row in successful
    ]
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=("User decode throughput", "Decode GPU throughput"),
    )
    figure.add_trace(
        go.Scatter(
            x=decode_batch,
            y=[float(row["user_decode_tps_mean"]) for row in successful],
            mode="lines+markers+text",
            text=[f"c={value:g}" for value in offered],
            textposition="top center",
            customdata=customdata,
            name="User TPS",
            hovertemplate=(
                "Mean decode batch %{x:.2f}<br>"
                "User TPS %{y:.2f}<br>"
                "Offered concurrency %{customdata[0]:g}<br>"
                "Realized concurrency %{customdata[1]:.2f}<br>"
                "Decode batch p90 %{customdata[2]:.1f}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=decode_batch,
            y=[
                float(row["decode_tokens_per_gpu_s"])
                for row in successful
            ],
            mode="lines+markers+text",
            text=[f"c={value:g}" for value in offered],
            textposition="top center",
            customdata=customdata,
            name="Decode tokens/s/GPU",
            hovertemplate=(
                "Mean decode batch %{x:.2f}<br>"
                "Decode tokens/s/GPU %{y:.2f}<br>"
                "Offered concurrency %{customdata[0]:g}<br>"
                "Realized concurrency %{customdata[1]:.2f}<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )
    figure.update_xaxes(title_text="Mean realized Decode batch size", row=2, col=1)
    figure.update_yaxes(title_text="tokens/s/user", row=1, col=1)
    figure.update_yaxes(title_text="tokens/s/GPU", row=2, col=1)
    figure.update_layout(
        title="Decode batching tradeoff under increasing offered load",
        height=760,
        showlegend=False,
        template="plotly_white",
        hovermode="closest",
    )
    figure.write_html(path, include_plotlyjs=True, full_html=True)


def distribution(values: Iterable[int]) -> dict[str, float | int]:
    materialized = list(values)
    return {
        "count": len(materialized),
        "mean": statistics.fmean(materialized),
        "p50": percentile(materialized, 0.50),
        "p90": percentile(materialized, 0.90),
        "p99": percentile(materialized, 0.99),
        "max": max(materialized),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--topology",
        choices=tuple(TOPOLOGIES),
        default="colocation",
    )
    parser.add_argument(
        "--workload-profile",
        choices=tuple(WORKLOAD_PROFILES),
        default="balanced-8k-1k",
    )
    parser.add_argument(
        "--precision-profile",
        choices=tuple(PRECISION_PROFILES),
        default="fp16",
        help=(
            "Operator precision mapping. fp8-fp4-mixed uses FP8 for "
            "attention, dense, router, KV cache, and communication, with "
            "W4A8 MoE experts."
        ),
    )
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument(
        "--calibration-requests",
        type=int,
        help=(
            "Use this many representative request shapes for isolated-service "
            "calibration. Defaults to all requests."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument(
        "--offered-concurrency",
        type=parse_float_list,
        default=parse_float_list("1,2,4,8,16,32"),
    )
    parser.add_argument(
        "--closed-loop-concurrency",
        action="store_true",
        help=(
            "Interpret --offered-concurrency values as exact total in-flight "
            "request targets. Release one replacement whenever a request "
            "completes instead of generating Poisson arrivals."
        ),
    )
    parser.add_argument(
        "--batch-sizes",
        type=parse_int_list,
        default=parse_int_list("1,4,8,16,32"),
    )
    parser.add_argument(
        "--unbounded-decode-batch",
        action="store_true",
        help=(
            "Do not sweep Decode batch caps. Use the request count as a "
            "non-binding safety cap and let offered load determine batch size."
        ),
    )
    parser.add_argument(
        "--prefill-batch-size-cap",
        type=int,
        default=64,
        help="Fixed Prefill batch-size cap for PDD runs.",
    )
    parser.add_argument(
        "--prefill-chunk-tokens",
        type=int,
        default=8_192,
        help=(
            "Maximum prefill tokens scheduled per request and iteration."
        ),
    )
    parser.add_argument(
        "--max-tokens-in-batch",
        type=int,
        default=8_192,
        help="Maximum total scheduled tokens in one batch.",
    )
    parser.add_argument(
        "--measurement-trim-fraction",
        type=float,
        default=0.0,
        help=(
            "Exclude this fraction of request IDs from both the beginning "
            "and end when calculating measurement_ttft_* fields."
        ),
    )
    capacity_group = parser.add_mutually_exclusive_group()
    capacity_group.add_argument(
        "--kv-capacities-gib",
        type=parse_float_list,
    )
    capacity_group.add_argument(
        "--num-blocks",
        type=parse_int_list,
    )
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument(
        "--arrival-reference-service-seconds",
        type=float,
        help=(
            "Use this isolated service time to convert offered concurrency "
            "to arrival rate. Set the same value across topologies for a "
            "paired load comparison."
        ),
    )
    parser.add_argument(
        "--moe-layer-event-mode",
        choices=("detailed", "first_layer_scaled"),
        help=(
            "Override the topology's analytical MoE event mode; useful for "
            "paired detailed-versus-scaled validation."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    topology = TOPOLOGIES[args.topology]
    if args.moe_layer_event_mode is not None:
        topology = replace(
            topology, moe_layer_event_mode=args.moe_layer_event_mode
        )
    workload_profile = WORKLOAD_PROFILES[args.workload_profile]
    operator_precisions = PRECISION_PROFILES[args.precision_profile]
    kv_dtype_size_bytes = PRECISION_BYTES[operator_precisions["kv_cache"]]
    kv_bytes_per_token_per_gpu = topology.kv_bytes_per_token_per_gpu(
        kv_dtype_size_bytes
    )
    binary = args.binary.resolve()
    if not binary.is_file():
        raise SystemExit(f"binary not found: {binary}")
    if args.requests <= 0:
        raise SystemExit("--requests must be positive")
    if args.calibration_requests is not None and (
        args.calibration_requests <= 0
        or args.calibration_requests > args.requests
    ):
        raise SystemExit(
            "--calibration-requests must be in [1, --requests]"
        )
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be positive")
    if args.prefill_chunk_tokens <= 0:
        raise SystemExit("--prefill-chunk-tokens must be positive")
    if args.max_tokens_in_batch <= 0:
        raise SystemExit("--max-tokens-in-batch must be positive")
    if args.prefill_batch_size_cap <= 0:
        raise SystemExit("--prefill-batch-size-cap must be positive")
    if args.prefill_chunk_tokens > args.max_tokens_in_batch:
        raise SystemExit(
            "--prefill-chunk-tokens cannot exceed --max-tokens-in-batch"
        )
    if (
        not math.isfinite(args.measurement_trim_fraction)
        or args.measurement_trim_fraction < 0.0
        or args.measurement_trim_fraction >= 0.5
    ):
        raise SystemExit(
            "--measurement-trim-fraction must be finite and in [0, 0.5)"
        )
    if (
        args.arrival_reference_service_seconds is not None
        and (
            not math.isfinite(args.arrival_reference_service_seconds)
            or args.arrival_reference_service_seconds <= 0.0
        )
    ):
        raise SystemExit(
            "--arrival-reference-service-seconds must be finite and positive"
        )

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else DEFAULT_OUTPUT.with_name(
            f"{DEFAULT_OUTPUT.name}_{topology.name}_"
            f"{workload_profile.name}"
        )
    ).resolve()
    if args.num_blocks is not None:
        num_blocks_values = args.num_blocks
    else:
        kv_capacities_gib = (
            args.kv_capacities_gib
            if args.kv_capacities_gib is not None
            else [4.0, 16.0]
        )
        num_blocks_values = []
        for capacity_gib in kv_capacities_gib:
            blocks = capacity_gib * GIB / (
                BLOCK_SIZE * kv_bytes_per_token_per_gpu
            )
            rounded_blocks = round(blocks)
            if not math.isclose(blocks, rounded_blocks):
                raise SystemExit(
                    "KV capacity cannot be represented by an integral block "
                    f"count: {capacity_gib:g} GiB"
                )
            num_blocks_values.append(rounded_blocks)
    inputs_dir = output_dir / "inputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    workload_config = WorkloadConfig(
        requests=args.requests,
        seed=args.seed,
        prompt_median=workload_profile.prompt_median,
        prompt_sigma=workload_profile.prompt_sigma,
        prompt_min=workload_profile.prompt_min,
        prompt_max=workload_profile.prompt_max,
        output_median=workload_profile.output_median,
        output_sigma=workload_profile.output_sigma,
        output_min=workload_profile.output_min,
        output_max=workload_profile.output_max,
    )
    shapes = generate_shapes(workload_config)

    if args.closed_loop_concurrency and any(
        not float(value).is_integer() for value in args.offered_concurrency
    ):
        raise SystemExit(
            "closed-loop concurrency values must be positive integers"
        )
    calibration_request_count = (
        0
        if args.closed_loop_concurrency
        else (args.calibration_requests or len(shapes))
    )
    prompt_mean = statistics.fmean(shape.prompt_tokens for shape in shapes)
    output_mean = statistics.fmean(shape.output_tokens for shape in shapes)
    calibration_shapes = sorted(
        shapes,
        key=lambda shape: (
            abs(shape.prompt_tokens - prompt_mean)
            + abs(shape.output_tokens - output_mean)
        ),
    )[:calibration_request_count]

    if args.closed_loop_concurrency:
        isolated_service_seconds = 0.0
        arrival_reference_service_seconds = 0.0
        print("closed-loop concurrency mode; calibration skipped", flush=True)
    else:
        calibration_workload = inputs_dir / "calibration.csv"
        calibration_config = inputs_dir / "calibration.json"
        write_workload(
            calibration_workload, calibration_shapes, arrival_rate=None
        )
        write_config(
            calibration_config,
            run_id="single-turn-load-calibration",
            prefill_batch_size_cap=1,
            decode_batch_size_cap=1,
            max_tokens_in_batch=args.max_tokens_in_batch,
            prefill_chunk_tokens=args.prefill_chunk_tokens,
            num_blocks=max(num_blocks_values),
            topology=topology,
            operator_precisions=operator_precisions,
        )
        print("calibration start", flush=True)
        calibration, calibration_process_seconds = run_simulator(
            binary,
            calibration_config,
            calibration_workload,
            timeout_seconds=args.timeout_seconds,
        )
        isolated_service_seconds = statistics.fmean(
            float(row["e2e_ms"]) / 1_000.0
            for row in calibration["requests"]
        )
        if isolated_service_seconds <= 0.0:
            raise RuntimeError("isolated mean service time must be positive")
        print(
            f"calibration mean_service_s={isolated_service_seconds:.9f} "
            f"process_s={calibration_process_seconds:.3f}",
            flush=True,
        )
        arrival_reference_service_seconds = (
            args.arrival_reference_service_seconds
            if args.arrival_reference_service_seconds is not None
            else isolated_service_seconds
        )
        print(
            "arrival reference_service_s="
            f"{arrival_reference_service_seconds:.9f}",
            flush=True,
        )

    workloads: dict[float, Path] = {}
    arrival_rates: dict[float, float] = {}
    for offered_concurrency in args.offered_concurrency:
        arrival_rate = (
            0.0
            if args.closed_loop_concurrency
            else offered_concurrency / arrival_reference_service_seconds
        )
        workload_path = (
            inputs_dir / f"load_c{offered_concurrency:g}.csv"
        )
        write_workload(
            workload_path,
            shapes,
            arrival_rate=1.0 if args.closed_loop_concurrency else arrival_rate,
        )
        workloads[offered_concurrency] = workload_path
        arrival_rates[offered_concurrency] = arrival_rate

    results_path = output_dir / "sweep_results.json"
    rows: list[dict] = []
    batch_size_values: list[int | None] = (
        [None] if args.unbounded_decode_batch else list(args.batch_sizes)
    )
    completed_keys: set[tuple[float, str, int]] = set()
    if args.resume and results_path.is_file():
        rows = json.loads(results_path.read_text(encoding="utf-8"))
        completed_keys = {
            (
                float(row["offered_concurrency"]),
                (
                    "unbounded"
                    if not bool(row.get("decode_batch_cap_enabled", True))
                    else str(int(row["decode_batch_size_cap"]))
                ),
                int(row["num_blocks_per_dp_target"]),
            )
            for row in rows
        }

    total_cases = (
        len(args.offered_concurrency)
        * len(batch_size_values)
        * len(num_blocks_values)
    )
    case_index = 0
    for num_blocks in num_blocks_values:
        for batch_size in batch_size_values:
            for offered_concurrency in args.offered_concurrency:
                case_index += 1
                batch_key = "unbounded" if batch_size is None else str(batch_size)
                key = (offered_concurrency, batch_key, num_blocks)
                if key in completed_keys:
                    print(
                        f"[{case_index}/{total_cases}] resume skip {key}",
                        flush=True,
                    )
                    continue
                run_id = (
                    f"single-turn-{topology.name}-{workload_profile.name}-"
                    f"c{offered_concurrency:g}-"
                    f"b{batch_key}-kv{num_blocks}"
                )
                config_path = inputs_dir / f"{run_id}.json"
                decode_batch_size_cap = (
                    args.requests if batch_size is None else batch_size
                )
                write_config(
                    config_path,
                    run_id=run_id,
                    prefill_batch_size_cap=args.prefill_batch_size_cap,
                    decode_batch_size_cap=decode_batch_size_cap,
                    max_tokens_in_batch=args.max_tokens_in_batch,
                    prefill_chunk_tokens=args.prefill_chunk_tokens,
                    num_blocks=num_blocks,
                    topology=topology,
                    operator_precisions=operator_precisions,
                    closed_loop_max_concurrency=(
                        int(offered_concurrency)
                        if args.closed_loop_concurrency
                        else 0
                    ),
                )
                print(
                    f"[{case_index}/{total_cases}] {run_id}",
                    flush=True,
                )
                try:
                    output, process_seconds = run_simulator(
                        binary,
                        config_path,
                        workloads[offered_concurrency],
                        timeout_seconds=args.timeout_seconds,
                    )
                    row = summarize_case(
                        output,
                        shapes,
                        offered_concurrency=offered_concurrency,
                        arrival_rate=arrival_rates[offered_concurrency],
                        prefill_batch_size_cap=args.prefill_batch_size_cap,
                        decode_batch_size_cap=decode_batch_size_cap,
                        decode_batch_cap_enabled=batch_size is not None,
                        max_tokens_in_batch=args.max_tokens_in_batch,
                        prefill_chunk_tokens=args.prefill_chunk_tokens,
                        measurement_trim_fraction=(
                            args.measurement_trim_fraction
                        ),
                        num_blocks=num_blocks,
                        topology=topology,
                        kv_bytes_per_token_per_gpu=(
                            kv_bytes_per_token_per_gpu
                        ),
                        process_seconds=process_seconds,
                    )
                    print(
                        "  "
                        f"actual_c={row['realized_mean_concurrency']:.3f} "
                        f"decode_b={row['mean_decode_batch_size']:.2f} "
                        f"rps={row['throughput_requests_per_s']:.3f} "
                        f"ttft_p90={row['ttft_p90_ms']:.3f}ms "
                        f"preemptions={row['preemptions']}",
                        flush=True,
                    )
                except (RuntimeError, subprocess.TimeoutExpired) as error:
                    row = {
                        "status": "failed",
                        "error": str(error),
                        "topology": topology.name,
                        "offered_concurrency": offered_concurrency,
                        "arrival_rate_rps": arrival_rates[offered_concurrency],
                        "batch_size_cap": decode_batch_size_cap,
                        "prefill_batch_size_cap": args.prefill_batch_size_cap,
                        "decode_batch_size_cap": decode_batch_size_cap,
                        "decode_batch_cap_enabled": batch_size is not None,
                        "max_tokens_in_batch": args.max_tokens_in_batch,
                        "prefill_chunk_tokens": args.prefill_chunk_tokens,
                        "num_blocks_per_dp_target": num_blocks,
                        "kv_token_capacity_per_dp_target": (
                            num_blocks * BLOCK_SIZE
                        ),
                        "kv_capacity_gib_per_gpu": (
                            num_blocks
                            * BLOCK_SIZE
                            * kv_bytes_per_token_per_gpu
                            / GIB
                        ),
                    }
                    print(f"  failed: {error}", flush=True)
                rows.append(row)
                results_path.write_text(
                    json.dumps(rows, indent=2) + "\n",
                    encoding="utf-8",
                )
                write_csv(output_dir / "sweep_results.csv", rows)

    manifest = {
        "description": (
            "Single-turn load sweep using probabilistic request lengths and "
            "a configurable analytical model topology."
        ),
        "workload": asdict(
            workload_config
        ),
        "workload_profile": workload_profile.name,
        "precision_profile": args.precision_profile,
        "operator_precisions": operator_precisions,
        "realized_workload": {
            "prompt_tokens": distribution(
                shape.prompt_tokens for shape in shapes
            ),
            "output_tokens": distribution(
                shape.output_tokens for shape in shapes
            ),
        },
        "isolated_mean_service_seconds": isolated_service_seconds,
        "calibration_requests": calibration_request_count,
        "arrival_reference_service_seconds": (
            arrival_reference_service_seconds
        ),
        "offered_concurrency": args.offered_concurrency,
        "arrival_rates_rps": arrival_rates,
        "concurrency_mode": (
            "closed_loop" if args.closed_loop_concurrency else "poisson"
        ),
        "batch_sizes": (
            [] if args.unbounded_decode_batch else args.batch_sizes
        ),
        "decode_batch_cap_mode": (
            "unbounded" if args.unbounded_decode_batch else "sweep"
        ),
        "prefill_batch_size_cap": args.prefill_batch_size_cap,
        "max_tokens_in_batch": args.max_tokens_in_batch,
        "prefill_chunk_tokens": args.prefill_chunk_tokens,
        "measurement_trim_fraction": args.measurement_trim_fraction,
        "num_blocks_per_dp_target": num_blocks_values,
        "topology": {
            "name": topology.name,
            "system_architecture": topology.system_architecture,
            "model": topology.model_name,
            "device": topology.device,
            "precision": operator_precisions["attention"],
            "operator_precisions": operator_precisions,
            "cluster_count": topology.cluster_count,
            "num_replicas_per_cluster": topology.num_replicas,
            "prefill_num_replicas": topology.parallelism(decode=False)[0],
            "decode_num_replicas": topology.parallelism(decode=True)[0],
            "tensor_parallel_size": topology.tensor_parallel_size,
            "pipeline_parallel_size": topology.pipeline_parallel_size,
            "data_parallel_size": topology.data_parallel_size,
            "gpu_count_per_cluster": topology.gpu_count_per_cluster,
            "prefill_parallelism": topology.parallelism(decode=False),
            "decode_parallelism": topology.parallelism(decode=True),
            "prefill_moe_parallelism": topology.moe_parallelism(
                decode=False
            ),
            "decode_moe_parallelism": topology.moe_parallelism(decode=True),
            "prefill_gpu_count": topology.prefill_gpu_count,
            "decode_gpu_count": topology.decode_gpu_count,
            "physical_gpu_count": topology.physical_gpu_count,
            "kv_bytes_per_token_per_gpu": (
                kv_bytes_per_token_per_gpu
            ),
            "max_tokens_in_batch": args.max_tokens_in_batch,
            "prefill_chunk_tokens": args.prefill_chunk_tokens,
            "block_size": BLOCK_SIZE,
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    tradeoff_path = output_dir / "decode_batch_tradeoff.html"
    write_tradeoff_html(tradeoff_path, rows)
    print(f"wrote {output_dir / 'sweep_results.csv'}", flush=True)
    if tradeoff_path.is_file():
        print(f"wrote {tradeoff_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
