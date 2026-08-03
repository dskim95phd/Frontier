"""Production-Python/C++ MoE routing and event-contract differential gates."""

from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from frontier.config import (
    AnalyticalRooflineExecutionTimePredictorConfig,
    MetricsConfig,
    ReplicaConfig,
    VllmV1SchedulerConfig,
)
from frontier.execution_time_predictor.analytical_roofline_execution_time_predictor import (
    AnalyticalRooflineExecutionTimePredictor,
)
from frontier.types import ClusterType


REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = REPO_ROOT / "cpp" / "tests" / "fixtures"
STEP25_SIMULATOR_ORACLE = (
    REPO_ROOT / "cpp" / "tests" / "parity"
    / "step25_simulator_oracle.py"
)
STEP3_SIMULATOR_ORACLE = (
    REPO_ROOT / "cpp" / "tests" / "parity"
    / "step3_simulator_oracle.py"
)


def _binary() -> Path:
    configured = os.environ.get("FRONTIER_CPP_BINARY")
    if not configured:
        pytest.skip("FRONTIER_CPP_BINARY is not set")
    return Path(configured)


def _run_path(
    config_path: Path,
    workload: str | Path = "step3_pdd_small.csv",
) -> tuple[dict, dict]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    workload_path = (
        workload
        if isinstance(workload, Path)
        else FIXTURES / "workloads" / workload
    )
    completed = subprocess.run(
        [
            str(_binary()),
            "--config",
            str(config_path),
            "--workload",
            str(workload_path),
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return config, json.loads(completed.stdout)


def _run(config_name: str) -> tuple[dict, dict]:
    return _run_path(FIXTURES / "config" / config_name)


def _run_production_oracle(
    script: Path,
    config_path: Path,
    workload: str | Path = "step3_pdd_small.csv",
) -> dict:
    workload_path = (
        workload
        if isinstance(workload, Path)
        else FIXTURES / "workloads" / workload
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            str(config_path),
            "--workload",
            str(workload_path),
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    result = json.loads(completed.stdout)
    assert result.pop("oracle") == "frontier.simulator.Simulator"
    return result


def _assert_nested_parity(expected: object, actual: object) -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        for key, expected_value in expected.items():
            assert key in actual
            _assert_nested_parity(expected_value, actual[key])
        return
    if isinstance(expected, list):
        assert isinstance(actual, list)
        assert len(actual) == len(expected)
        for expected_value, actual_value in zip(expected, actual):
            _assert_nested_parity(expected_value, actual_value)
        return
    if isinstance(expected, float):
        assert isinstance(actual, (int, float))
        assert actual == pytest.approx(
            expected,
            rel=1e-12,
            abs=1e-12,
        )
        return
    assert type(actual) is type(expected)
    assert actual == expected


def _assert_stage_parity(
    python_output: dict,
    cpp_output: dict,
) -> None:
    batches = {
        batch["batch_id"]: batch for batch in cpp_output["batches"]
    }
    actual_stages = []
    for stage in cpp_output["batch_stages"]:
        batch = batches[stage["batch_id"]]
        actual_stages.append(
            {
                "replica_id": stage["replica_id"],
                "dp_id": stage["dp_id"],
                "stage_id": stage["stage_id"],
                "request_ids": batch["request_ids"],
                "request_num_tokens": batch["scheduled_tokens"],
                "started_at_s": stage["started_at_s"],
                "completed_at_s": stage["completed_at_s"],
                "duration_ms": stage["duration_ms"],
            }
        )
    expected_stages = [
        {
            key: stage[key]
            for key in (
                "replica_id",
                "dp_id",
                "stage_id",
                "request_ids",
                "request_num_tokens",
                "started_at_s",
                "completed_at_s",
                "duration_ms",
            )
        }
        for stage in python_output["batch_stages"]
    ]
    _assert_nested_parity(expected_stages, actual_stages)


def _assert_full_simulator_parity(
    python_output: dict,
    cpp_output: dict,
    *,
    include_transfers: bool,
) -> None:
    _assert_request_timeline_parity(python_output, cpp_output)
    _assert_event_trace_parity(python_output, cpp_output)
    _assert_nested_parity(
        python_output["scheduler_trace"],
        cpp_output["scheduler_trace"],
    )
    _assert_stage_parity(python_output, cpp_output)
    assert max(
        request["completed_at_s"]
        for request in cpp_output["requests"]
    ) == pytest.approx(
        python_output["simulation_completed_at_s"],
        rel=1e-12,
        abs=1e-12,
    )
    if include_transfers:
        _assert_nested_parity(
            python_output["kv_cache_transfers"],
            cpp_output["kv_cache_transfers"],
        )


def _write_workload(
    path: Path,
    rows: list[tuple[float, int, int]],
) -> Path:
    lines = [
        "session_start_at,think_time,num_prefill_tokens,num_decode_tokens",
        *[
            f"{arrival:.12f},0,{prefill},{decode}"
            for arrival, prefill, decode in rows
        ],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _analytical_execution() -> dict:
    return {
        "type": "analytical",
        "device": "rubin",
        "precision": "bf16",
        "network_bandwidth_gbps": 400.0,
        "network_latency_us": 1.0,
        "intra_node_bandwidth_gbps": 14_400.0,
    }


def _assert_routing_allocations(
    config: dict,
    cpp_output: dict,
) -> None:
    assert cpp_output["moe_routing"]
    for record in cpp_output["moe_routing"]:
        if config["system_architecture"] == "co-location":
            cluster = config["clusters"]["monolithic"]
        else:
            cluster = config["clusters"][
                record["cluster_type"].lower()
            ]
        expected = _python_global_route(
            total=record["routed_tokens"],
            layer=record["layer_id"],
            total_expert_num=cluster["total_expert_num"],
            routing=cluster["moe_routing"],
        )
        assert record["global_expert_tokens"] == expected
        assert sum(expected) == record["routed_tokens"]
        ep_size = cluster["parallelism"][
            "moe_expert_parallel_size"
        ]
        experts_per_lane = len(expected) // ep_size
        assert record["lane_expert_tokens"] == [
            expected[
                lane * experts_per_lane :
                (lane + 1) * experts_per_lane
            ]
            for lane in range(ep_size)
        ]


def _assert_request_timeline_parity(
    python_output: dict,
    cpp_output: dict,
) -> None:
    cpp_by_id = {
        request["request_id"]: request
        for request in cpp_output["requests"]
    }
    assert set(cpp_by_id) == {
        request["request_id"]
        for request in python_output["requests"]
    }
    exact_fields = {
        "request_id",
        "num_processed_tokens",
        "preemption_count",
        "replica_id",
        "dp_id",
        "prefill_replica_id",
        "prefill_dp_id",
        "decode_replica_id",
        "decode_dp_id",
    }
    for expected in python_output["requests"]:
        actual = cpp_by_id[expected["request_id"]]
        for field, expected_value in expected.items():
            if field not in actual:
                continue
            if field in exact_fields:
                assert type(actual[field]) is type(expected_value)
                assert actual[field] == expected_value
            elif field.endswith("_s"):
                assert actual[field] == pytest.approx(
                    expected_value,
                    rel=1e-12,
                    abs=1e-12,
                )


def _assert_event_trace_parity(
    python_output: dict,
    cpp_output: dict,
) -> None:
    expected_events = python_output["event_trace"]
    actual_events = cpp_output["event_trace"]
    assert len(actual_events) == len(expected_events)
    sync_types = {
        "prefill_sync",
        "prefill_sync_collective",
        "decode_sync",
        "decode_sync_collective",
    }
    semantic_fields = {
        "request_id",
        "replica_id",
        "dp_id",
        "stage_id",
        "layer_id",
        "sync_group_id",
    }

    def signature(
        event: dict,
        fields_by_type: dict[str, set[str]],
    ) -> tuple:
        fields = fields_by_type[event["type"]]
        values = [event["type"]]
        for field in sorted(semantic_fields | {"cluster_type"}):
            values.append(
                event.get(field) if field in fields else None
            )
        values.append(
            event.get("sync_stage", event.get("sync_phase"))
            if "sync_stage" in fields
            else None
        )
        # Idle Batch objects are allocated lazily and their IDs depend on
        # same-timestamp heap insertion order.  They are not part of the
        # production event contract; real request/batch identity is checked
        # separately through request timelines and non-sync events.
        values.append(
            None
            if (
                event["type"] in sync_types
                or "batch_id" not in fields
            )
            else event.get("batch_id")
        )
        return tuple(values)

    start = 0
    while start < len(expected_events):
        expected_time = expected_events[start]["time_s"]
        end = start + 1
        while (
            end < len(expected_events)
            and expected_events[end]["time_s"]
            == pytest.approx(expected_time, rel=1e-12, abs=1e-12)
        ):
            end += 1

        expected_group = expected_events[start:end]
        actual_group = actual_events[start:end]
        assert all(
            event["time_s"]
            == pytest.approx(expected_time, rel=1e-12, abs=1e-12)
            for event in actual_group
        )
        fields_by_type = {
            event_type: {
                field
                for event in expected_group
                if event["type"] == event_type
                for field in event
            }
            for event_type in {
                event["type"] for event in expected_group
            }
        }
        assert Counter(
            signature(event, fields_by_type) for event in actual_group
        ) == Counter(
            signature(event, fields_by_type) for event in expected_group
        )
        start = end


def _python_global_route(
    *,
    total: int,
    layer: int,
    total_expert_num: int,
    routing: dict,
) -> list[int]:
    predictor = object.__new__(AnalyticalRooflineExecutionTimePredictor)
    predictor._replica_config = SimpleNamespace(
        total_expert_num=total_expert_num
    )
    predictor._moe_routing_mode = routing["mode"]
    predictor._moe_routing_distribution_type = routing["distribution"]
    predictor._moe_routing_seed = routing["seed"]
    allocation = predictor._get_global_per_expert_tokens(total, layer)
    return [allocation[index] for index in range(len(allocation))]


def _python_analytical_predictor(
    cluster: dict,
) -> AnalyticalRooflineExecutionTimePredictor:
    parallelism = cluster["parallelism"]
    routing = cluster["moe_routing"]
    execution = cluster["execution_model"]
    replica = ReplicaConfig(
        model_name=cluster["model_name"],
        device=execution["device"],
        network_device="vera_rubin_nvl72_domain",
        attn_tensor_parallel_size=parallelism[
            "tensor_parallel_size"
        ],
        attn_data_parallel_size=parallelism[
            "data_parallel_size"
        ],
        data_parallel_size=parallelism["data_parallel_size"],
        num_pipeline_stages=parallelism[
            "pipeline_parallel_size"
        ],
        moe_tensor_parallel_size=parallelism[
            "moe_tensor_parallel_size"
        ],
        moe_expert_parallel_size=parallelism[
            "moe_expert_parallel_size"
        ],
        total_expert_num=cluster["total_expert_num"],
        router_topk=cluster["router_topk"],
        moe_routing_mode=routing["mode"],
        moe_routing_distribution_type=routing["distribution"],
        moe_routing_seed=routing["seed"],
    )
    return AnalyticalRooflineExecutionTimePredictor(
        AnalyticalRooflineExecutionTimePredictorConfig(),
        replica,
        VllmV1SchedulerConfig(),
        MetricsConfig(write_metrics=False),
        cluster_type=ClusterType.MONOLITHIC,
    )


def _python_lane_times(
    *,
    predictor: AnalyticalRooflineExecutionTimePredictor,
    record: dict,
) -> list[float]:
    batch = SimpleNamespace(
        num_prefill_tokens=record["input_tokens"],
        num_decode_tokens=0,
        get_effective_total_tokens_for_compute=lambda _: record[
            "input_tokens"
        ],
    )
    return [
        predictor.predict_moe_layer_time(
            batch,
            record["layer_id"],
            ClusterType.MONOLITHIC,
            per_expert_tokens={
                expert: tokens
                for expert, tokens in enumerate(lane)
            },
        ).total_time()
        for lane in record["lane_expert_tokens"]
    ]


@pytest.mark.parametrize(
    "config_name",
    [
        "analytical_moe_ep4_colocation.json",
        "fixed_moe_sequential_pdd.json",
    ],
)
def test_moe_integer_routing_matches_production_python(
    config_name: str,
) -> None:
    config, output = _run(config_name)
    assert output["moe_routing"]

    for record in output["moe_routing"]:
        if config["system_architecture"] == "co-location":
            cluster = config["clusters"]["monolithic"]
        else:
            cluster_name = record["cluster_type"].lower()
            cluster = config["clusters"][cluster_name]

        expected = _python_global_route(
            total=record["routed_tokens"],
            layer=record["layer_id"],
            total_expert_num=cluster["total_expert_num"],
            routing=cluster["moe_routing"],
        )
        assert record["global_expert_tokens"] == expected
        assert sum(expected) == record["routed_tokens"]

        ep_size = cluster["parallelism"]["moe_expert_parallel_size"]
        experts_per_lane = len(expected) // ep_size
        assert record["lane_expert_tokens"] == [
            expected[
                lane * experts_per_lane : (lane + 1) * experts_per_lane
            ]
            for lane in range(ep_size)
        ]


def test_monolithic_decode_ids_and_idle_compaction_match_python_rule() -> None:
    config, output = _run("analytical_moe_ep4_colocation.json")
    ep_size = config["clusters"]["monolithic"]["parallelism"][
        "moe_expert_parallel_size"
    ]
    decode_arrivals = [
        event
        for event in output["event_trace"]
        if event["type"] == "decode_sync" and not event["is_idle"]
    ]
    assert decode_arrivals
    assert all(
        event["sync_group_id"] % ep_size == event["dp_id"]
        for event in decode_arrivals
    )
    assert not any(
        event["type"] == "decode_sync"
        and event["is_idle"]
        and event["sync_phase"] == "pre_moe"
        for event in output["event_trace"]
    )
    assert any(
        event["type"] == "decode_sync"
        and event["is_idle"]
        and event["sync_phase"] == "post_moe"
        for event in output["event_trace"]
    )


def test_phi_pdd_transfer_bytes_match_production_model_shape() -> None:
    _, output = _run("fixed_moe_sequential_pdd.json")
    expected_per_token = 32 * 4 * 128 * 2 * 2
    assert [
        transfer["size_bytes"] for transfer in output["kv_cache_transfers"]
    ] == [4 * expected_per_token, 6 * expected_per_token]


def test_analytical_colocation_matches_full_production_simulator() -> None:
    config_path = (
        FIXTURES
        / "config"
        / "analytical_moe_ep4_colocation.json"
    )
    _, cpp_output = _run_path(config_path)
    python_output = _run_production_oracle(
        STEP25_SIMULATOR_ORACLE,
        config_path,
    )
    _assert_full_simulator_parity(
        python_output,
        cpp_output,
        include_transfers=False,
    )


def test_analytical_pdd_matches_full_production_simulator(
    tmp_path: Path,
) -> None:
    config = json.loads(
        (
            FIXTURES
            / "config"
            / "fixed_moe_sequential_pdd.json"
        ).read_text(encoding="utf-8")
    )
    for cluster_name in ("prefill", "decode"):
        config["clusters"][cluster_name]["execution_model"] = {
            "type": "analytical",
            "device": "rubin",
            "precision": "bf16",
            "network_bandwidth_gbps": 400.0,
            "network_latency_us": 1.0,
            "intra_node_bandwidth_gbps": 14_400.0,
        }
    config_path = tmp_path / "analytical_moe_pdd.json"
    config_path.write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    _, cpp_output = _run_path(config_path)
    python_output = _run_production_oracle(
        STEP3_SIMULATOR_ORACLE,
        config_path,
    )
    _assert_full_simulator_parity(
        python_output,
        cpp_output,
        include_transfers=True,
    )


COLOCATION_PRODUCTION_MATRIX = [
    {
        "name": "local_online_balanced_top1",
        "mode": "online",
        "topology": (1, 1, 1, 1, 1, 1),
        "routing": ("simulation", "balanced", 3),
        "topk": 1,
        "scheduler": {
            "batch_size_cap": 3,
            "max_tokens_in_batch": 9,
        },
        "workload": [
            (0.0, 1, 1),
            (0.0, 7, 3),
            (0.0004, 13, 2),
        ],
    },
    {
        "name": "moe_tp_offline_uniform_legacy",
        "mode": "offline",
        "topology": (1, 2, 1, 2, 1, 2),
        "routing": ("uniform_legacy", "balanced", 17),
        "topk": 2,
        "scheduler": {
            "batch_size_cap": 4,
            "max_tokens_in_batch": 11,
        },
        "workload": [
            (0.0, 2, 4),
            (0.0, 9, 1),
            (0.0, 17, 3),
            (0.0002, 5, 2),
        ],
    },
    {
        "name": "ep_pp4_online_skewed",
        "mode": "online",
        "topology": (1, 2, 1, 1, 2, 4),
        "routing": ("simulation", "skewed", 29),
        "topk": 1,
        "scheduler": {
            "batch_size_cap": 4,
            "max_tokens_in_batch": 13,
        },
        "workload": [
            (0.0, 3, 2),
            (0.00003, 21, 4),
            (0.00003, 8, 1),
            (0.0007, 33, 2),
        ],
    },
    {
        "name": "dp2_ep4_offline_zipf_pressure",
        "mode": "offline",
        "topology": (1, 2, 2, 1, 4, 2),
        "routing": ("simulation", "zipf", 41),
        "topk": 2,
        "scheduler": {
            "batch_size_cap": 2,
            "max_tokens_in_batch": 8,
            "block_size": 4,
            "num_blocks": 2,
        },
        "workload": [
            (0.0, 3, 3),
            (0.0, 4, 2),
            (0.0, 3, 3),
            (0.0, 4, 2),
        ],
    },
    {
        "name": "replicas2_staggered_random",
        "mode": "online",
        "topology": (2, 2, 1, 1, 2, 2),
        "routing": ("simulation", "random", 53),
        "topk": 2,
        "scheduler": {
            "batch_size_cap": 3,
            "max_tokens_in_batch": 10,
        },
        "workload": [
            (0.0, 4, 2),
            (0.00001, 6, 3),
            (0.00002, 11, 1),
            (0.0005, 2, 4),
            (0.00051, 15, 2),
            (0.001, 7, 3),
        ],
    },
    {
        "name": "tp_ep_pp2_uniform_random_long",
        "mode": "offline",
        "topology": (1, 4, 1, 2, 2, 2),
        "routing": ("uniform_random", "balanced", 67),
        "topk": 2,
        "scheduler": {
            "batch_size_cap": 6,
            "max_tokens_in_batch": 16,
            "block_size": 8,
            "num_blocks": 128,
        },
        "workload": [
            (0.0, 48, 2),
            (0.0, 79, 4),
            (0.0001, 33, 3),
        ],
    },
]


@pytest.mark.parametrize(
    "case",
    COLOCATION_PRODUCTION_MATRIX,
    ids=lambda case: case["name"],
)
def test_colocation_production_configuration_workload_matrix(
    tmp_path: Path,
    case: dict,
) -> None:
    config = json.loads(
        (
            FIXTURES
            / "config"
            / "analytical_moe_ep4_colocation.json"
        ).read_text(encoding="utf-8")
    )
    config["run_id"] = f"moe-production-{case['name']}"
    config["simulation_mode"] = case["mode"]
    replicas, attn_tp, attn_dp, moe_tp, moe_ep, pp = (
        case["topology"]
    )
    cluster = config["clusters"]["monolithic"]
    cluster["parallelism"].update(
        {
            "num_replicas": replicas,
            "tensor_parallel_size": attn_tp,
            "data_parallel_size": attn_dp,
            "moe_tensor_parallel_size": moe_tp,
            "moe_expert_parallel_size": moe_ep,
            "pipeline_parallel_size": pp,
        }
    )
    cluster["scheduler"].update(case["scheduler"])
    routing_mode, distribution, seed = case["routing"]
    cluster["moe_routing"].update(
        {
            "mode": routing_mode,
            "distribution": distribution,
            "seed": seed,
        }
    )
    cluster["router_topk"] = case["topk"]
    config_path = tmp_path / f"{case['name']}.json"
    config_path.write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )
    workload_path = _write_workload(
        tmp_path / f"{case['name']}.csv",
        case["workload"],
    )

    _, cpp_output = _run_path(config_path, workload_path)
    python_output = _run_production_oracle(
        STEP25_SIMULATOR_ORACLE,
        config_path,
        workload_path,
    )
    _assert_full_simulator_parity(
        python_output,
        cpp_output,
        include_transfers=False,
    )
    _assert_routing_allocations(config, cpp_output)


PDD_PRODUCTION_MATRIX = [
    {
        "name": "local_online_balanced",
        "mode": "online",
        "prefill": (1, 1, 1, 1, 1, 2),
        "decode": (1, 1, 1, 1, 1, 1),
        "prefill_routing": ("simulation", "balanced", 5),
        "decode_routing": ("simulation", "balanced", 7),
        "topk": 1,
        "workload": [
            (0.0, 2, 2),
            (0.0003, 7, 3),
            (0.0008, 11, 1),
        ],
    },
    {
        "name": "asymmetric_dp_ep_offline",
        "mode": "offline",
        "prefill": (1, 2, 1, 1, 2, 4),
        "decode": (1, 2, 2, 1, 4, 1),
        "prefill_routing": ("simulation", "skewed", 13),
        "decode_routing": ("simulation", "zipf", 19),
        "topk": 2,
        "workload": [
            (0.0, 4, 3),
            (0.0, 9, 2),
            (0.0002, 17, 4),
            (0.0006, 5, 1),
        ],
    },
    {
        "name": "moe_tp_to_tp_ep_online",
        "mode": "online",
        "prefill": (1, 2, 1, 2, 1, 2),
        "decode": (1, 4, 1, 2, 2, 1),
        "prefill_routing": (
            "uniform_legacy",
            "balanced",
            23,
        ),
        "decode_routing": (
            "uniform_random",
            "balanced",
            29,
        ),
        "topk": 2,
        "workload": [
            (0.0, 3, 1),
            (0.00001, 12, 3),
            (0.00002, 25, 2),
        ],
    },
    {
        "name": "replicas2_staggered",
        "mode": "online",
        "prefill": (2, 2, 1, 1, 2, 2),
        "decode": (2, 2, 1, 1, 2, 1),
        "prefill_routing": ("simulation", "random", 31),
        "decode_routing": ("simulation", "skewed", 37),
        "topk": 1,
        "workload": [
            (0.0, 4, 2),
            (0.00001, 6, 3),
            (0.00002, 8, 1),
            (0.0005, 10, 4),
            (0.00051, 12, 2),
            (0.001, 14, 3),
        ],
    },
    {
        "name": "decode_pressure_slow_transfer",
        "mode": "online",
        "prefill": (1, 2, 1, 1, 2, 2),
        "decode": (1, 2, 2, 1, 4, 1),
        "prefill_routing": ("simulation", "zipf", 43),
        "decode_routing": ("simulation", "random", 47),
        "topk": 2,
        "prefill_scheduler": {
            "batch_size_cap": 2,
            "max_tokens_in_batch": 8,
            "block_size": 4,
            "num_blocks": 8,
        },
        "decode_scheduler": {
            "batch_size_cap": 2,
            "max_tokens_in_batch": 8,
            "block_size": 4,
            "num_blocks": 3,
        },
        "transfer": {
            "network_bandwidth_gbps": 2.0,
            "network_latency_ms": 1.5,
        },
        "workload": [
            (0.0, 3, 3),
            (0.0, 4, 2),
            (0.0001, 3, 3),
            (0.0001, 4, 2),
        ],
    },
]


@pytest.mark.parametrize(
    "case",
    PDD_PRODUCTION_MATRIX,
    ids=lambda case: case["name"],
)
def test_pdd_production_configuration_workload_matrix(
    tmp_path: Path,
    case: dict,
) -> None:
    config = json.loads(
        (
            FIXTURES
            / "config"
            / "fixed_moe_sequential_pdd.json"
        ).read_text(encoding="utf-8")
    )
    config["run_id"] = f"moe-pdd-production-{case['name']}"
    config["simulation_mode"] = case["mode"]
    config["kv_cache_transfer"].update(case.get("transfer", {}))
    for cluster_name in ("prefill", "decode"):
        cluster = config["clusters"][cluster_name]
        cluster["execution_model"] = _analytical_execution()
        (
            replicas,
            attn_tp,
            attn_dp,
            moe_tp,
            moe_ep,
            pp,
        ) = case[cluster_name]
        cluster["parallelism"].update(
            {
                "num_replicas": replicas,
                "tensor_parallel_size": attn_tp,
                "data_parallel_size": attn_dp,
                "moe_tensor_parallel_size": moe_tp,
                "moe_expert_parallel_size": moe_ep,
                "pipeline_parallel_size": pp,
            }
        )
        cluster["scheduler"].update(
            case.get(f"{cluster_name}_scheduler", {})
        )
        routing_mode, distribution, seed = case[
            f"{cluster_name}_routing"
        ]
        cluster["moe_routing"].update(
            {
                "mode": routing_mode,
                "distribution": distribution,
                "seed": seed,
            }
        )
        cluster["router_topk"] = case["topk"]
    config_path = tmp_path / f"{case['name']}.json"
    config_path.write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )
    workload_path = _write_workload(
        tmp_path / f"{case['name']}.csv",
        case["workload"],
    )

    _, cpp_output = _run_path(config_path, workload_path)
    python_output = _run_production_oracle(
        STEP3_SIMULATOR_ORACLE,
        config_path,
        workload_path,
    )
    _assert_full_simulator_parity(
        python_output,
        cpp_output,
        include_transfers=True,
    )
    _assert_routing_allocations(config, cpp_output)


@pytest.mark.parametrize(
    (
        "name",
        "attn_tp",
        "attn_dp",
        "moe_tp",
        "moe_ep",
        "pp",
        "replicas",
        "distribution",
        "topk",
    ),
    [
        ("local", 1, 1, 1, 1, 1, 1, "balanced", 1),
        ("moe_tp", 2, 1, 2, 1, 1, 1, "random", 2),
        ("ep", 2, 1, 1, 2, 1, 1, "skewed", 1),
        ("tp_ep", 4, 1, 2, 2, 2, 1, "zipf", 2),
        ("dp_ep", 2, 2, 1, 4, 2, 1, "random", 2),
        ("pp4", 2, 1, 1, 2, 4, 1, "skewed", 2),
        ("replicas2", 2, 1, 1, 2, 2, 2, "balanced", 2),
    ],
)
def test_moe_topology_and_distribution_matrix(
    tmp_path: Path,
    name: str,
    attn_tp: int,
    attn_dp: int,
    moe_tp: int,
    moe_ep: int,
    pp: int,
    replicas: int,
    distribution: str,
    topk: int,
) -> None:
    config = json.loads(
        (
            FIXTURES
            / "config"
            / "analytical_moe_ep4_colocation.json"
        ).read_text(encoding="utf-8")
    )
    config["run_id"] = f"moe-matrix-{name}"
    cluster = config["clusters"]["monolithic"]
    cluster["parallelism"].update(
        {
            "num_replicas": replicas,
            "tensor_parallel_size": attn_tp,
            "pipeline_parallel_size": pp,
            "data_parallel_size": attn_dp,
            "moe_tensor_parallel_size": moe_tp,
            "moe_expert_parallel_size": moe_ep,
        }
    )
    cluster["moe_routing"].update(
        {
            "distribution": distribution,
            "seed": 100 + attn_tp + 10 * moe_ep,
        }
    )
    cluster["router_topk"] = topk
    config_path = tmp_path / f"{name}.json"
    config_path.write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    _, output = _run_path(config_path)
    assert len(output["completed_request_ids"]) == 2
    assert output["moe_routing"]
    assert output["batch_stages"]
    python_predictor = _python_analytical_predictor(cluster)

    for record in output["moe_routing"]:
        expected = _python_global_route(
            total=record["routed_tokens"],
            layer=record["layer_id"],
            total_expert_num=cluster["total_expert_num"],
            routing=cluster["moe_routing"],
        )
        assert record["global_expert_tokens"] == expected
        assert sum(expected) == record["routed_tokens"]
        assert record["lane_times_ms"] == pytest.approx(
            _python_lane_times(
                predictor=python_predictor,
                record=record,
            ),
            rel=1e-12,
            abs=1e-12,
        )
        assert record["critical_lane"] == max(
            range(len(record["lane_times_ms"])),
            key=record["lane_times_ms"].__getitem__,
        )

    stages = output["batch_stages"]
    assert all(
        stage["attention_tensor_parallel_size"] == attn_tp
        and stage["attention_data_parallel_size"] == attn_dp
        and stage["moe_tensor_parallel_size"] == moe_tp
        and stage["moe_expert_parallel_size"] == moe_ep
        for stage in stages
    )
    assert any(stage["tp_communication_ms"] > 0 for stage in stages) == (
        attn_tp > 1
    )
    assert any(
        stage["moe_tp_communication_ms"] > 0 for stage in stages
    ) == (moe_tp > 1)
    assert any(stage["ep_dispatch_ms"] > 0 for stage in stages) == (
        moe_ep > 1
    )
    # The production analytical predictor does not put DP gather/scatter
    # costs in monolithic ExecutionTime records.  PDD's decode scheduler
    # accounts for those transfers explicitly.
    assert not any(
        stage["dp_input_communication_ms"] > 0 for stage in stages
    )
    assert any(stage["pp_communication_ms"] > 0 for stage in stages) == (
        pp > 1
    )

    sync_types = {
        event["type"]
        for event in output["event_trace"]
        if event["type"]
        in {
            "prefill_sync",
            "prefill_sync_collective",
            "decode_sync",
            "decode_sync_collective",
        }
    }
    assert bool(sync_types) == (attn_dp > 1 or moe_ep > 1)
    if sync_types:
        assert sync_types == {
            "prefill_sync",
            "prefill_sync_collective",
            "decode_sync",
            "decode_sync_collective",
        }
