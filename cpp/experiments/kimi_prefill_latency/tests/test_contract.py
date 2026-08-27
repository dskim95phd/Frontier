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
