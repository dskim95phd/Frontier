"""Step 2 analytical oracle backed by Frontier's full Python Simulator."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import csv
import io
import json
import logging
import math
from pathlib import Path
import sys
import tempfile
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.disable(logging.CRITICAL)

from frontier.cc_backend.cc_backend_config import AnalyticalCCBackendConfig
from frontier.config import (
    AnalyticalRooflineExecutionTimePredictorConfig,
    ClusterConfig,
    FixedRequestLengthGeneratorConfig,
    MetricsConfig,
    ReplicaConfig,
    SimulationConfig,
    StaticRequestIntervalGeneratorConfig,
    SyntheticRequestGeneratorConfig,
    VllmV1SchedulerConfig,
)
from frontier.entities import Batch, Request
from frontier.simulator import Simulator
from frontier.types import ClusterType


class Step2AnalyticalOracleError(RuntimeError):
    pass


def _read_inputs(
    config_path: Path, workload_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 2:
        raise Step2AnalyticalOracleError("schema_version=2 is required")
    if config.get("system_architecture") != "co-location":
        raise Step2AnalyticalOracleError("co-location is required")
    if config.get("enable_parallel_clusters") is not False:
        raise Step2AnalyticalOracleError("sequential execution is required")
    if config.get("execution_model", {}).get("type") != "analytical":
        raise Step2AnalyticalOracleError(
            "analytical execution_model is required"
        )
    with workload_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise Step2AnalyticalOracleError(
            "full Simulator oracle fixture requires exactly one request"
        )
    row = rows[0]
    request = {
        "arrived_at": float(row["arrived_at"]),
        "num_prefill_tokens": int(row["num_prefill_tokens"]),
        "num_decode_tokens": int(row["num_decode_tokens"]),
    }
    if (
        not math.isfinite(request["arrived_at"])
        or request["arrived_at"] != 0.0
        or request["num_prefill_tokens"] < 2
        or request["num_decode_tokens"] < 1
    ):
        raise Step2AnalyticalOracleError(
            "oracle fixture requires a zero-time valid request"
        )
    return config, request


def _build_python_config(
    config: dict[str, Any],
    request: dict[str, Any],
    output_dir: str,
) -> SimulationConfig:
    scheduler = config["scheduler"]
    execution = config["execution_model"]
    replica_scheduler = VllmV1SchedulerConfig(
        batch_size_cap=int(scheduler["batch_size_cap"]),
        block_size=int(scheduler["block_size"]),
        watermark_blocks_fraction=float(
            scheduler["watermark_blocks_fraction"]
        ),
        max_tokens_in_batch=int(scheduler["max_tokens_in_batch"]),
        scheduling_policy=str(scheduler["scheduling_policy"]),
        enable_preemption=bool(scheduler["enable_preemption"]),
        enable_chunked_prefill=bool(
            scheduler["enable_chunked_prefill"]
        ),
        enable_prefix_caching=False,
        prefix_caching_key_mode="session",
        num_preallocate_tokens=int(
            scheduler["num_preallocate_tokens"]
        ),
        long_prefill_token_threshold=int(
            scheduler["long_prefill_token_threshold"]
        ),
        num_blocks=int(scheduler["num_blocks"]),
        num_blocks_mode="explicit",
    )
    replica = ReplicaConfig(
        model_name="meta-llama/Llama-2-7b-hf",
        device=str(execution["device"]),
        network_device="vera_rubin_nvl72_domain",
        attn_tensor_parallel_size=int(
            execution["tensor_parallel_size"]
        ),
        attn_data_parallel_size=1,
        num_pipeline_stages=1,
    )
    cluster = ClusterConfig(
        replica_scheduler_config=replica_scheduler,
        execution_time_predictor_config=(
            AnalyticalRooflineExecutionTimePredictorConfig()
        ),
        cc_backend_config=AnalyticalCCBackendConfig(
            network_bandwidth_gbps=float(
                execution["network_bandwidth_gbps"]
            ),
            network_latency_us=float(execution["network_latency_us"]),
            intra_node_bandwidth_gbps=float(
                execution["intra_node_bandwidth_gbps"]
            ),
        ),
        num_replicas=1,
        replica_config=replica,
    )
    request_generator = SyntheticRequestGeneratorConfig(
        length_generator_config=FixedRequestLengthGeneratorConfig(
            prefill_tokens=request["num_prefill_tokens"],
            decode_tokens=request["num_decode_tokens"],
        ),
        interval_generator_config=StaticRequestIntervalGeneratorConfig(),
        num_requests=1,
    )
    metrics = MetricsConfig(
        write_metrics=False,
        write_json_trace=False,
        enable_chrome_trace=False,
        store_plots=False,
        output_dir=output_dir,
        run_id="step2-analytical-full-simulator",
    )
    return SimulationConfig(
        simulation_mode=str(config["simulation_mode"]),
        sys_arch="co-location",
        decode_cuda_graph_mode="none",
        cluster_config=cluster,
        request_generator_config=request_generator,
        metrics_config=metrics,
        enable_parallel_clusters=False,
        log_level="critical",
    )


def run_oracle(
    config_path: Path, workload_path: Path
) -> dict[str, Any]:
    config, request_input = _read_inputs(config_path, workload_path)
    Request._id = -1
    Batch._id = -1
    with tempfile.TemporaryDirectory(
        prefix="frontier-step2-analytical-"
    ) as output_dir:
        with redirect_stdout(io.StringIO()):
            python_config = _build_python_config(
                config, request_input, output_dir
            )
            simulator = Simulator(python_config)
            simulator.run()
    if len(simulator._all_requests) != 1:
        raise Step2AnalyticalOracleError(
            "full Simulator did not retain exactly one request"
        )
    request = simulator._all_requests[0]
    predictor = simulator._predictors[ClusterType.MONOLITHIC]
    diagnostics = predictor.get_diagnostics()
    return {
        "oracle": "frontier.simulator.Simulator",
        "request": {
            "request_id": int(request.id),
            "arrived_at_s": float(request.arrived_at),
            "first_scheduled_at_s": float(request.scheduled_at),
            "prefill_completed_at_s": float(
                request.prefill_completed_at
            ),
            "completed_at_s": float(request.completed_at),
            "ttft_ms": float(request.ttft * 1000.0),
            "e2e_ms": float(request.e2e_time * 1000.0),
            "num_processed_tokens": int(
                request.num_processed_tokens
            ),
        },
        "simulation_completed_at_s": float(simulator._time),
        "analytical_operator_diagnostic_count": len(diagnostics),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run_oracle(arguments.config, arguments.workload),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
