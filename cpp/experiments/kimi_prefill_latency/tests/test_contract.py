from __future__ import annotations

import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
REPO_ROOT = HERE.parents[2]


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_workload_builds_exact_128k_prefix_then_adds_1k() -> None:
    with (HERE / "workload_128k_plus_1k.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))

    assert len(rows) == 2
    assert int(rows[0]["num_prefill_tokens"]) + int(
        rows[0]["num_decode_tokens"]
    ) == 131_072
    assert int(rows[1]["num_prefill_tokens"]) == 1_024
    assert rows[0]["session_id"] == rows[1]["session_id"]
    assert [int(row["session_turn_index"]) for row in rows] == [0, 1]


def test_batch16_workload_has_sixteen_independent_warm_prefix_sessions() -> None:
    with (HERE / "workload_batch16_128k_plus_1k.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))

    assert len(rows) == 32
    for index in range(0, len(rows), 2):
        initial, followup = rows[index : index + 2]
        assert int(initial["num_prefill_tokens"]) + int(
            initial["num_decode_tokens"]
        ) == 131_072
        assert int(followup["num_prefill_tokens"]) == 1_024
        assert initial["session_id"] == followup["session_id"]
        assert [
            int(initial["session_turn_index"]),
            int(followup["session_turn_index"]),
        ] == [0, 1]


def test_both_models_share_the_exact_8gpu_batch1_contract() -> None:
    configs = [
        _json(HERE / "configs" / "kimi_k2_8gpu_hbm500_batch1.json"),
        _json(HERE / "configs" / "kimi_k3_8gpu_hbm500_batch1.json"),
    ]

    assert {config["model"] for config in configs} == {
        "moonshotai/Kimi-K2-Instruct",
        "moonshotai/Kimi-K3",
    }
    for config in configs:
        cluster = config["clusters"]["monolithic"]
        parallelism = cluster["parallelism"]
        scheduler = cluster["scheduler"]
        assert cluster["profile"] == "rubin-500gb-vera-8gpu"
        assert parallelism["tensor_parallel_size"] == 1
        assert parallelism["data_parallel_size"] == 8
        assert parallelism["moe_tensor_parallel_size"] == 1
        assert parallelism["moe_expert_parallel_size"] == 8
        assert scheduler["batch_size_cap"] == 1
        assert scheduler["max_tokens_in_batch"] == 1_024
        assert scheduler["long_prefill_token_threshold"] == 1_024


def test_hypothetical_rubin_profile_has_500gb_hbm_and_eight_gpus() -> None:
    gpu = _json(REPO_ROOT / "data/config/gpus/rubin-500gb.json")
    cluster = _json(
        REPO_ROOT / "data/config/clusters/rubin-500gb-vera-8gpu.json"
    )

    assert gpu["memory"]["capacity_bytes_per_gpu"] == 500_000_000_000
    assert cluster["gpu"] == {"profile": "rubin-500gb", "count": 8}
    assert cluster["gpu_memory"]["runtime_reserve_fraction"] == 0.1


def test_k3_tep8_configs_use_tp8_dp1_and_ep8() -> None:
    configs = [
        _json(HERE / "configs" / "kimi_k3_8gpu_hbm500_tp8_ep8_batch1.json"),
        _json(HERE / "configs" / "kimi_k3_gb300_tep8_8k_prefill.json"),
    ]
    for config in configs:
        parallelism = config["clusters"]["monolithic"]["parallelism"]
        assert parallelism["tensor_parallel_size"] == 8
        assert parallelism["data_parallel_size"] == 1
        assert parallelism["moe_tensor_parallel_size"] == 1
        assert parallelism["moe_expert_parallel_size"] == 8

    gb300 = configs[1]["clusters"]["monolithic"]
    assert gb300["profile"] == "gb300-tracelab-8gpu"
    assert gb300["execution_model"]["kernel_profile"] == "k3_sglang_mxfp4"
    assert gb300["scheduler"]["max_tokens_in_batch"] == 16 * 8192


def test_precision_matched_configs_use_tp8_ep8_and_flashkda() -> None:
    k2 = _json(
        HERE / "configs" / "kimi_k2_8gpu_hbm500_tp8_ep8_batch1.json"
    )
    k3 = _json(
        HERE
        / "configs"
        / "kimi_k3_8gpu_hbm500_tp8_ep8_k2_precision_flashkda.json"
    )
    for config in (k2, k3):
        parallelism = config["clusters"]["monolithic"]["parallelism"]
        assert parallelism["tensor_parallel_size"] == 8
        assert parallelism["data_parallel_size"] == 1
        assert parallelism["moe_tensor_parallel_size"] == 1
        assert parallelism["moe_expert_parallel_size"] == 8

    execution = k3["clusters"]["monolithic"]["execution_model"]
    assert execution["precision"] == "fp8"
    assert execution["kernel_profile"] == "k3_flashkda_prefill"
    operators = execution["operator_precisions"]
    assert operators["attention_weight"] == "fp8"
    assert operators["attention_activation"] == "fp8"
    assert operators["routed_expert_weight"] == "fp4"
    assert operators["routed_expert_activation"] == "fp8"
    assert operators["kda_snapshot"] == "fp8"


def test_precision_matched_batch16_configs_schedule_16_one_k_chunks() -> None:
    configs = [
        _json(HERE / "configs" / "kimi_k2_8gpu_hbm500_tp8_ep8_batch16.json"),
        _json(
            HERE
            / "configs"
            / "kimi_k3_8gpu_hbm500_tp8_ep8_k2_precision_flashkda_batch16.json"
        ),
    ]
    for config in configs:
        cluster = config["clusters"]["monolithic"]
        scheduler = cluster["scheduler"]
        parallelism = cluster["parallelism"]
        assert scheduler["batch_size_cap"] == 16
        assert scheduler["max_tokens_in_batch"] == 16 * 1_024
        assert scheduler["long_prefill_token_threshold"] == 1_024
        assert parallelism["tensor_parallel_size"] == 8
        assert parallelism["data_parallel_size"] == 1
        assert parallelism["moe_expert_parallel_size"] == 8
