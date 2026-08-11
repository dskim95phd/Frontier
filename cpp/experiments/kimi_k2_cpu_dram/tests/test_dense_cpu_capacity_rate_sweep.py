from __future__ import annotations

import json
from pathlib import Path

import pytest

from cpp.experiments.kimi_k2_cpu_dram import generate_dense_cpu_capacity_rate_reports as reports
from cpp.experiments.kimi_k2_cpu_dram import run_dense_cpu_capacity_rate_sweep as runner


def test_decimal_matrix_is_inclusive_and_exact(tmp_path: Path) -> None:
    cases = runner.build_matrix(
        {500: ("0.30", "0.32"), 750: ("0.31", "0.32")},
        step="0.01",
        workload_root=tmp_path / "workloads",
        output_root=tmp_path / "runs",
        seed=7,
    )

    assert [(case.capacity_gb, str(case.rate)) for case in cases] == [
        (500, "0.30"),
        (500, "0.31"),
        (750, "0.31"),
        (500, "0.32"),
        (750, "0.32"),
    ]
    assert cases[0].rate_label == "r0p30"
    assert cases[0].capacity_label == "cpu0500gb"
    assert cases[0].output_dir == tmp_path / "runs" / "r0p30" / "cpu0500gb" / "r1"


def test_decimal_matrix_rejects_misaligned_range(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not aligned"):
        runner.build_matrix(
            {500: ("0.305", "0.32")},
            step="0.01",
            workload_root=tmp_path / "workloads",
            output_root=tmp_path / "runs",
            seed=7,
        )


def test_workload_compatibility_requires_current_converter_version(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.json"
    payload = {
        "converter_version": runner.CONVERTER_VERSION,
        "sampling": {
            "sample_sessions": None,
            "seed": runner.SEED,
            "session_repetitions": 3,
            "session_arrival_rate_per_second": 0.3,
        },
    }
    metadata.write_text(json.dumps(payload), encoding="utf-8")
    assert runner._workload_is_compatible(metadata, runner.Decimal("0.3"), 3)

    payload["converter_version"] = "tracelab-v0.0.2-frontier-csv-v2"
    metadata.write_text(json.dumps(payload), encoding="utf-8")
    assert not runner._workload_is_compatible(metadata, runner.Decimal("0.3"), 3)


def test_all_session_epoch_count_is_derived_from_highest_rate(tmp_path: Path) -> None:
    cases = runner.build_matrix(
        {500: ("0.30", "0.50")},
        step="0.01",
        workload_root=tmp_path / "workloads",
        output_root=tmp_path / "runs",
        seed=7,
    )

    assert runner.resolve_session_repetitions(
        cases, source_sessions_per_epoch=1000, configured_repetitions=None
    ) == 18
    with pytest.raises(ValueError, match="use at least 18"):
        runner.resolve_session_repetitions(
            cases, source_sessions_per_epoch=1000, configured_repetitions=17
        )


def test_all_session_converter_command_omits_sampling_flag(tmp_path: Path) -> None:
    args = type(
        "Args",
        (),
        {
            "python": tmp_path / "python",
            "tracelab_db": tmp_path / "trace.duckdb",
        },
    )()
    command = runner._converter_command(
        args,
        runner.Decimal("0.30"),
        tmp_path / "workload.csv",
        tmp_path / "manifest.csv",
        tmp_path / "metadata.json",
        3,
    )

    assert "--sample-sessions" not in command
    assert command[command.index("--session-repetitions") + 1] == "3"


def test_case_start_message_prints_complete_experiment_condition(tmp_path: Path) -> None:
    case = runner.build_matrix(
        {500: ("0.30", "0.30")},
        step="0.01",
        workload_root=tmp_path / "workloads",
        output_root=tmp_path / "runs",
        seed=7,
    )[0]

    message = runner._case_start_message(
        case,
        source_sessions_per_epoch=8041,
        session_repetitions=3,
    )

    assert message == (
        "[running] r0p30/cpu0500gb rate=0.30/s physical_cpu=500GB "
        "per_prefill_gpu=250GB source_sessions=8041 epochs=3 horizon=10h"
    )


def test_case_progress_message_reports_simulated_hours(tmp_path: Path) -> None:
    case = runner.build_matrix(
        {4000: ("0.45", "0.45")},
        step="0.01",
        workload_root=tmp_path / "workloads",
        output_root=tmp_path / "runs",
        seed=7,
    )[0]

    assert runner._case_progress_message(
        case,
        simulation_time_s=3 * 3600,
        wall_seconds=42.125,
    ) == "[progress] r0p45/cpu4000gb simulated=3/10h (30%) wall=42.1s"


def test_report_discovery_includes_only_completed_points(tmp_path: Path) -> None:
    completed = tmp_path / "r0p3" / "cpu0500gb" / "r1"
    incomplete = tmp_path / "r0p3" / "cpu0750gb" / "r1"
    completed.mkdir(parents=True)
    incomplete.mkdir(parents=True)
    (completed / "summary.json").write_text("{}", encoding="utf-8")
    (completed / "requests.csv").write_text("request_id\n", encoding="utf-8")
    (incomplete / "summary.json").write_text("{}", encoding="utf-8")
    metadata = tmp_path / "workloads" / "r0p3" / "seed_7_metadata.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(json.dumps({"sampling": {}}), encoding="utf-8")
    plan = {
        "cases": [
            {
                "rate": 0.3,
                "rate_label": "r0p3",
                "capacity_gb": 500,
                "output_dir": str(completed),
                "workload_metadata": str(metadata),
            },
            {
                "rate": 0.3,
                "rate_label": "r0p3",
                "capacity_gb": 750,
                "output_dir": str(incomplete),
                "workload_metadata": str(metadata),
            },
        ]
    }

    grouped = reports.completed_cases_by_rate(plan)

    assert list(grouped) == ["r0p3"]
    assert [case["capacity_gb"] for case in grouped["r0p3"]] == [500]
