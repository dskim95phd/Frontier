from pathlib import Path

from cpp.experiments.kimi_cpu_dram import generate_report as reports


def test_index_hides_decode_topology_for_prefill_only(tmp_path: Path) -> None:
    rendered = reports._index_html(
        tmp_path,
        [],
        latest_planned=0,
        completed=0,
        identity={
            "model_name": "Kimi K3",
            "prefill_gpu_count": 24,
            "decode_gpu_count": 32,
            "prefill_only": True,
        },
    )

    assert "Kimi K3 — PREFILL-only, 24 PREFILL GPUs" in rendered
    assert "DECODE GPUs" not in rendered
    assert "capacity &times; 24" in rendered


def test_index_uses_configured_k2_prefill_gpu_count(tmp_path: Path) -> None:
    rendered = reports._index_html(
        tmp_path,
        [],
        latest_planned=0,
        completed=0,
        identity={
            "model_name": "Kimi K2",
            "prefill_gpu_count": 8,
            "decode_gpu_count": 16,
            "prefill_only": True,
        },
    )

    assert "Kimi K2 — PREFILL-only, 8 PREFILL GPUs" in rendered
    assert "capacity &times; 8" in rendered
