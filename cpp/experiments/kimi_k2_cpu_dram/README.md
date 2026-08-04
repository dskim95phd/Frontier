# Kimi K2 / GB300 CPU-DRAM capacity study

This directory is a self-contained experiment harness for the study described
in [`docs/experiments/kimi-k2-gb300-cpu-dram-capacity-study.md`](../../../docs/experiments/kimi-k2-gb300-cpu-dram-capacity-study.md).
It does not change the simulator.  The harness uses the current C++ CLI and
keeps every capacity case paired to the same generated workload for a seed.

The frozen topology in `configs/base_pdd.json` is sequential online PDD:

| side | GPUs | replicas | attention TP | PP | DP | MoE TP | EP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PREFILL | 16 | 1 | 4 | 1 | 4 | 1 | 16 |
| DECODE | 16 | 1 | 4 | 1 | 4 | 1 | 16 |

It uses `moonshotai/Kimi-K2-Instruct`, analytical `gb300`, FP8 KV, balanced
MoE routing, `first_layer_scaled`, sticky session prefix caching, and
`enable_parallel_clusters=false`.  The byte contract is recorded in every
phase manifest: 39,040 one-copy bytes/token, 156,160 target-physical
bytes/token, 2,498,560 target-physical bytes/block, and 304,175 GPU KV blocks
per DP target from the 190 GB one-copy budget.

## 1. Generate paired workloads

The generator uses only the standard library, so it works from PowerShell
without installing the Python simulator:

```powershell
python .\cpp\experiments\kimi_k2_cpu_dram\generate_workload.py `
  --output-dir .\cpp\experiments\kimi_k2_cpu_dram\workloads `
  --seed 20260803
```

The frozen `v3` defaults are 1,728 sessions (1,152 six-turn and 576
seven-turn), a bounded
log-normal final context with mean 65,536 tokens, input:output 8:1, Poisson
first-turn starts at 100 sessions/s, Dirichlet turn shares (`alpha=20`), and
clipped log-normal successor think times (median 1,000 s, sigma 0.001,
[990, 1,010]
s).  The command writes:

* `seed_<seed>.csv`: the six-column CSV accepted by `frontier_sim`;
* `seed_<seed>_manifest.csv`: request-level join fields; and
* `seed_<seed>_metadata.json`: parameters, deterministic statistics, and the
  generator version.

The manifest distinguishes the fresh-input theoretical minimum from the
implementation minimum (`new input + previous decode + one 16-token block`).
The generated materialized prompt is recorded as a calibration baseline; it is
replaced with actual scheduled PREFILL when a simulator requests output is
joined.

## 2. CPU-OFF calibration (Phase W)

Run the pilot OFF case first.  `--dry-run` is useful for checking paths and
configs without a binary; normal runs use the compact `requests` output mode,
not a full event trace:

```powershell
python .\cpp\experiments\kimi_k2_cpu_dram\run_sweep.py `
  --phase pilot `
  --binary .\cpp\build\Release\frontier_sim.exe `
  --output-root .\outputs\kimi_k2_cpu_dram
```

Every capacity in a seed uses the exact same CSV.  A run writes
`summary.json`, `requests.csv`, normalized inputs, and `run.json`; the latter
also records the exact Windows command and paired seed.

Compute the actual extra-PREFILL ratio from the compact request output:

```powershell
python .\cpp\experiments\kimi_k2_cpu_dram\generate_workload.py `
  --calibrate-manifest .\cpp\experiments\kimi_k2_cpu_dram\workloads\seed_20260803_manifest.csv `
  --calibrate-requests .\outputs\kimi_k2_cpu_dram\pilot\seed_20260803\off\requests.csv `
  --calibrate-output .\outputs\kimi_k2_cpu_dram\workload_calibration\iteration_1\seed_20260803_manifest.csv
```

The calculation is exactly
`(sum(scheduled_prefill_tokens) - sum(new_input_tokens)) /
sum(new_input_tokens)`.  Current C++ requests output emits
`scheduled_prefill_tokens` and `preemption_recomputed_prefill_tokens`; older
outputs are supported by reconstructing the materialized prompt from the
manifest and subtracting `cached_prefill_tokens`.  Freeze a workload only
after the pooled OFF ratio is 2.80--3.20 (and document any think-time,
arrival-rate, or session-count adjustment in the iteration directory).

The frozen parameters passed validation-only CPU-OFF runs on seeds 20260803,
20260804, and 20260805.  Their actual extra-PREFILL ratios were 2.96276,
2.97222, and 2.97014; the pooled ratio was 2.96837.  Pooled GPU-prefix hit
rate was 0.7852% and no request was preempted.  The experiment runner disables
expensive per-event structural validation; unit/integration tests and final
CPU-cache diagnostics retain full invariant validation.

## 3. Pilot, coarse, and fine sweeps

The phase defaults are:

* `pilot`: `off`, `cpu128`, `oracle_unbounded_cpu` (one seed);
* `coarse`: `off`, `cpu32`, `cpu64`, `cpu128`, `cpu256`, `cpu500`, and
  `oracle_unbounded_cpu` (five paired seeds);
* `fine`: the analyzer's selected 5--7 labels, or the documented seven-point
  fallback `cpu256,cpu296,cpu336,cpu376,cpu416,cpu456,cpu500`; and
* `custom`: labels passed with `--capacities`.

```powershell
# Coarse matrix: 7 cases x 5 paired seeds = 35 runs
python .\cpp\experiments\kimi_k2_cpu_dram\run_sweep.py `
  --phase coarse `
  --binary .\cpp\build\Release\frontier_sim.exe `
  --output-root .\outputs\kimi_k2_cpu_dram `
  --resume

# Plan a custom/fine matrix without launching the simulator
python .\cpp\experiments\kimi_k2_cpu_dram\run_sweep.py `
  --phase custom `
  --capacities cpu256,cpu296,cpu336,cpu376,cpu416,cpu456,cpu500 `
  --seeds 20260803,20260804 `
  --dry-run
```

`cpu<N>` denotes `<N>` GB per Grace CPU, `<N>/2` GB per GPU slice, and four
physical TP4 slices per `(replica, dp)` CPU target.  `cpu500` therefore resolves
to 1,000,000,000,000 bytes per target and 6,222.24 Gbps of serialized
DRAM/C2C bandwidth.  The oracle computes a direct no-eviction target from the
session snapshots (with a conservative block margin) and retains cpu500
bandwidth.

## 4. Analyze and choose `C_low`/`C_high`

```powershell
python .\cpp\experiments\kimi_k2_cpu_dram\analyze_results.py `
  --output-root .\outputs\kimi_k2_cpu_dram `
  --workload-dir .\cpp\experiments\kimi_k2_cpu_dram\workloads
```

The analyzer writes `summary.csv`, `session_summary.csv`, and
`fine\selection.json`.  It scans adjacent physical coarse capacities and
selects the first pair crossing any documented threshold: CPU extension hit
rate +5 percentage points, scheduled PREFILL -5%, successor TTFT p90 -3%, or
eviction/truncation change 10 percentage points.  It proposes 5--7 inclusive
Grace-CPU points, aligned to 8 GB (`fine step ~= (C_high-C_low)/6`).  If no
physical pair crosses a threshold, an oracle-only/non-physical transition is
reported separately rather than being presented as a GB300 product result.

Run the selected fine sweep with the generated selection file:

```powershell
python .\cpp\experiments\kimi_k2_cpu_dram\run_sweep.py `
  --phase fine `
  --fine-selection .\outputs\kimi_k2_cpu_dram\fine\selection.json `
  --binary .\cpp\build\Release\frontier_sim.exe `
  --output-root .\outputs\kimi_k2_cpu_dram
```

No command assumes POSIX separators or a shell pipeline; paths are passed as
`Path` arguments to `subprocess.run`, so the same scripts work in Windows
PowerShell and in a POSIX shell.  Use `--output-mode summary` when per-request
metrics are not needed; `requests` is the recommended mode for successor-turn
TTFT and calibration, while `full` is reserved for a small diagnostic pilot.
