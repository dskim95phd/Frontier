from __future__ import annotations

import json
from pathlib import Path

import pytest

from cpp.experiments.kimi_k3_cpu_dram import generate_dense_cpu_capacity_rate_reports as reports
from cpp.experiments.kimi_k3_cpu_dram import run_dense_cpu_capacity_rate_sweep as runner


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


def test_default_k3_matrix_has_off_plus_seven_per_gpu_capacity_points(
    tmp_path: Path,
) -> None:
    cases = runner.build_rate_capacity_matrix(
        runner.SESSION_RATE_CAPACITIES_GB,
        workload_root=tmp_path / "workloads",
        output_root=tmp_path / "runs",
        seed=7,
    )

    assert [(case.capacity_gb, str(case.rate)) for case in cases] == [
        (capacity_gb, runner.PILOT_SESSION_RATE)
        for capacity_gb in (0, 250, 375, 500, 625, 750, 875, 1000)
    ]
    assert cases[0].capacity_label == "cpu0000gb"


def test_configured_matrix_can_select_capacities_per_session_rate(
    tmp_path: Path,
) -> None:
    cases = runner.build_rate_capacity_matrix(
        {
            "0.50": (250, 500, 1000),
            "0.70": (500, 1000),
            "0.90": (1000,),
        },
        workload_root=tmp_path / "workloads",
        output_root=tmp_path / "runs",
        seed=7,
    )

    assert [(str(case.rate), case.capacity_gb) for case in cases] == [
        ("0.50", 250),
        ("0.50", 500),
        ("0.50", 1000),
        ("0.70", 500),
        ("0.70", 1000),
        ("0.90", 1000),
    ]


def test_session_rate_cli_is_a_uniform_override_for_all_declared_capacities() -> None:
    args = runner.build_parser().parse_args(["--session-rate", "0.42"])
    rate_capacities = runner.resolve_rate_capacities(
        args.session_rate,
        {"0.30": (0, 250, 500), "0.50": (500, 750, 1000)},
    )

    assert rate_capacities == {
        runner.Decimal("0.42"): (0, 250, 500, 750, 1000)
    }


def test_omitting_session_rate_uses_configured_rate_matrix() -> None:
    args = runner.build_parser().parse_args([])

    assert args.session_rate is None
    assert runner.resolve_rate_capacities(
        args.session_rate,
        {"0.50": (250, 500), "0.70": (750, 1000)},
    ) == {
        runner.Decimal("0.50"): (250, 500),
        runner.Decimal("0.70"): (750, 1000),
    }


def test_twelve_hour_endpoint_cli_contract() -> None:
    args = runner.build_parser().parse_args(
        [
            "--session-rate",
            "0.5",
            "--simulation-hours",
            "12",
            "--capacities-gb",
            "250,1000",
        ]
    )

    assert args.session_rate == "0.5"
    assert args.simulation_hours == 12
    assert args.capacities_gb == {250, 1000}


def test_simulator_command_enables_one_minute_wall_progress_by_default(
    tmp_path: Path,
) -> None:
    args = runner.build_parser().parse_args([])
    case = runner.build_matrix(
        {1000: ("0.50", "0.50")},
        step="0.01",
        workload_root=tmp_path / "workloads",
        output_root=tmp_path / "runs",
        seed=7,
    )[0]

    command = runner._simulator_command(args, case)

    option_index = command.index("--wall-progress-interval-s")
    assert command[option_index + 1] == "60"


def test_simulator_command_uses_performance_safe_diagnostic_options_by_default(
    tmp_path: Path,
) -> None:
    args = runner.build_parser().parse_args([])
    case = runner.build_matrix(
        {1000: ("0.50", "0.50")},
        step="0.01",
        workload_root=tmp_path / "workloads",
        output_root=tmp_path / "runs",
        seed=7,
    )[0]

    command = runner._simulator_command(args, case)

    assert command[command.index("--runtime-validation") + 1] == "false"
    assert command[command.index("--gpu-kv-occupancy") + 1] == "false"


def test_expensive_diagnostics_require_explicit_acknowledgement() -> None:
    args = runner.build_parser().parse_args(["--runtime-validation"])
    with pytest.raises(SystemExit, match="--allow-expensive-diagnostics"):
        runner._require_expensive_diagnostics_opt_in(args)

    acknowledged = runner.build_parser().parse_args(
        ["--runtime-validation", "--allow-expensive-diagnostics"]
    )
    runner._require_expensive_diagnostics_opt_in(acknowledged)


def test_simulator_command_can_disable_wall_progress(tmp_path: Path) -> None:
    args = runner.build_parser().parse_args(["--wall-progress-interval-s", "0"])
    case = runner.build_matrix(
        {1000: ("0.50", "0.50")},
        step="0.01",
        workload_root=tmp_path / "workloads",
        output_root=tmp_path / "runs",
        seed=7,
    )[0]

    assert "--wall-progress-interval-s" not in runner._simulator_command(
        args, case
    )


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

    payload["converter_version"] = "definitely-not-the-current-version"
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
        "[running] r0p30/cpu0500gb rate=0.30/s per_prefill_gpu=500GB "
        "aggregate_cpu=12000GB source_sessions=8041 epochs=3 horizon=10h"
    )


def test_k3_config_uses_static_per_gpu_capacity_and_off_baseline() -> None:
    assert runner.DEFAULT_CONFIG.name == (
        "tracelab_k3_p24_d32_cpu_sweep_exact_benchmark.json"
    )
    template = json.loads(runner.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    cases = runner.build_matrix(
        {0: ("0.30", "0.30"), 250: ("0.30", "0.30")},
        step="0.01",
        workload_root=Path("workloads"),
        output_root=Path("runs"),
        seed=7,
    )

    off = runner.build_config(template, cases[0])
    enabled = runner.build_config(template, cases[1])
    assert off["cpu_kv_cache"]["enabled"] is False
    assert enabled["cpu_kv_cache"] == {
        **template["cpu_kv_cache"],
        "enabled": True,
        "static_slice_per_gpu": True,
        "capacity_bytes_per_gpu": 250_000_000_000,
        "capacity_bytes": 6_000_000_000_000,
    }
    assert runner._topology_gpu_counts(enabled) == (24, 32)
    assert enabled["run_id"] == "kimi-k3-tracelab-p24-d32-r0p30-cpu0250gb"
    prefill_scheduler = enabled["clusters"]["prefill"]["scheduler"]
    assert prefill_scheduler["max_tokens_in_batch"] == 16_384
    assert prefill_scheduler["enable_chunked_prefill"] is True
    assert (
        prefill_scheduler["long_prefill_token_threshold"] == 512
    )
    assert enabled["clusters"]["prefill"]["parallelism"] == {
        "num_replicas": 1,
        "tensor_parallel_size": 1,
        "decode_context_parallel_size": 1,
        "pipeline_parallel_size": 24,
        "data_parallel_size": 1,
        "moe_tensor_parallel_size": 1,
        "moe_expert_parallel_size": 1,
        "pipeline_exclusive": True,
    }
    assert (
        enabled["clusters"]["decode"]["parallelism"]
        == template["clusters"]["decode"]["parallelism"]
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


def _write_discoverable_report_case(
    root: Path,
    *,
    rate_label: str,
    rate: float,
    capacity_gb: int,
    metadata: Path,
    status: str = "completed",
) -> Path:
    case_dir = root / rate_label / f"cpu{capacity_gb:04d}gb" / "r1"
    case_dir.mkdir(parents=True)
    (case_dir / "summary.json").write_text("{}", encoding="utf-8")
    (case_dir / "requests.csv").write_text("request_id\n", encoding="utf-8")
    (case_dir / "run.json").write_text(
        json.dumps(
            {
                "status": status,
                "session_arrival_rate_per_second": rate,
                "simulation_end_time_s": 12 * 3600,
                "workload_metadata": str(metadata),
            }
        ),
        encoding="utf-8",
    )
    (case_dir / "config.normalized.json").write_text(
        json.dumps(
            {
                "clusters": {
                    "prefill": {
                        "parallelism": {
                            "num_replicas": 1,
                            "tensor_parallel_size": 1,
                            "pipeline_parallel_size": 24,
                            "data_parallel_size": 1,
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return case_dir


def test_report_generation_discovers_split_runs_beyond_latest_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "split-sweep"
    metadata = tmp_path / "workloads" / "r0p50" / "seed_7_metadata.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(json.dumps({"sampling": {}}), encoding="utf-8")
    first = _write_discoverable_report_case(
        output_root,
        rate_label="r0p50",
        rate=0.5,
        capacity_gb=500,
        metadata=metadata,
    )
    second = _write_discoverable_report_case(
        output_root,
        rate_label="r0p50",
        rate=0.5,
        capacity_gb=1000,
        metadata=metadata,
    )
    _write_discoverable_report_case(
        output_root,
        rate_label="r0p50",
        rate=0.5,
        capacity_gb=750,
        metadata=metadata,
        status="failed",
    )
    # Simulate a later split invocation replacing the plan with only its own
    # capacity.  Directory discovery must retain the earlier completed point.
    (output_root / "sweep_plan.json").write_text(
        json.dumps(
            {
                "prefill_gpus": 24,
                "cases": [
                    {
                        "rate": 0.5,
                        "rate_label": "r0p50",
                        "capacity_gb": 1000,
                        "output_dir": str(second),
                        "workload_metadata": str(metadata),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    calls: list[list[str]] = []

    def fake_capacity_report(argv: list[str]) -> int:
        calls.append(argv)
        for option in ("--output-csv", "--output-json", "--output-html"):
            path = Path(argv[argv.index(option) + 1])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("generated", encoding="utf-8")
        return 0

    monkeypatch.setattr(reports.capacity_report, "main", fake_capacity_report)

    result = reports.generate_reports(output_root)

    assert len(calls) == 1
    assert calls[0][calls[0].index("--capacities") + 1] == "500,1000"
    assert calls[0][calls[0].index("--prefill-lanes") + 1] == "24"
    assert "--include-final-hour-details" in calls[0]
    assert result["planned_cases"] == 1
    assert result["completed_cases"] == 2
    assert result["rate_reports"][0]["capacities_gb"] == [500, 1000]
    assert first.is_dir()
    index = (output_root / "index.html").read_text(encoding="utf-8")
    assert "Discovered 2 completed points" in index
    assert "500, 1000" in index


def test_report_directory_discovery_does_not_require_sweep_plan(
    tmp_path: Path,
) -> None:
    metadata = tmp_path / "workloads" / "r0p30" / "seed_7_metadata.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text("{}", encoding="utf-8")
    _write_discoverable_report_case(
        tmp_path,
        rate_label="r0p30",
        rate=0.3,
        capacity_gb=250,
        metadata=metadata,
    )

    grouped = reports.discover_completed_cases_by_rate(tmp_path)

    assert list(grouped) == ["r0p30"]
    assert [case["capacity_gb"] for case in grouped["r0p30"]] == [250]
