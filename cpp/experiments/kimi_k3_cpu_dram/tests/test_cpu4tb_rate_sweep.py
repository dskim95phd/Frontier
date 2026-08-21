import json

from cpp.experiments.kimi_k3_cpu_dram import run_cpu4tb_rate_sweep as runner


def test_k3_cpu4tb_config_overrides_prefill_batching_contract() -> None:
    template = json.loads(runner.DEFAULT_TEMPLATE.read_text(encoding="utf-8"))
    template["clusters"]["prefill"]["parallelism"][
        "pipeline_stage_layer_counts"
    ] = [4] * 23 + [1]
    template["clusters"]["decode"]["parallelism"][
        "pipeline_stage_layer_counts"
    ] = [93]

    config = runner.build_k3_config(template, 0.3)
    prefill = config["clusters"]["prefill"]["scheduler"]
    decode = config["clusters"]["decode"]["scheduler"]

    assert prefill["max_tokens_in_batch"] == 16_384
    assert prefill["enable_chunked_prefill"] is True
    assert prefill["long_prefill_token_threshold"] == 512
    assert decode["max_tokens_in_batch"] == 8_192
    assert decode["long_prefill_token_threshold"] == 0
    assert all(
        "pipeline_stage_layer_counts" not in cluster["parallelism"]
        for cluster in config["clusters"].values()
    )
