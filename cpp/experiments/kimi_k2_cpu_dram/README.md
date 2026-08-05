# Kimi K2 / GB300 CPU-DRAM capacity study

This directory is a self-contained experiment harness for the study described
in [`docs/experiments/kimi-k2-gb300-cpu-dram-capacity-study.md`](../../../docs/experiments/kimi-k2-gb300-cpu-dram-capacity-study.md).
It does not change the simulator.  The harness uses the current C++ CLI and
keeps every capacity case paired to the same generated workload for a seed.

The separate TraceLab cache-aware arrival-rate experiment uses 8 PREFILL and
24 DECODE GPUs and sweeps 0.10 down to 0.05 new source sessions/s.  Its frozen
configuration, server commands, resume behavior, measurement horizons, and
detailed result collector are documented in
[`docs/experiments/tracelab-cache-aware-arrival-sweep.md`](../../../docs/experiments/tracelab-cache-aware-arrival-sweep.md).

The frozen topology in `configs/base_pdd.json` is sequential online PDD:

| side | GPUs | replicas | attention TP | DCP | PP | DP | MoE TP | EP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PREFILL | 16 | 1 | 4 | 1 | 1 | 4 | 1 | 16 |
| DECODE | 16 | 1 | 4 | 4 | 1 | 4 | 1 | 16 |

It uses `moonshotai/Kimi-K2-Instruct`, analytical `gb300`, FP8 KV, balanced
MoE routing, `first_layer_scaled`, actual-GPU-cache-aware PREFILL routing,
vLLM-like least-outstanding DECODE routing, and
`enable_parallel_clusters=false`. PREFILL retains session affinity while its
actual GPU-prefix hit ratio is at least 0.5 and the target loads are balanced.
It switches to the least-loaded target when both the absolute load gap exceeds
32 and the relative gap exceeds 1.1, or when the GPU hit ratio is below 0.5.
Migration discards all old-target GPU KV for that session. PREFILL is TP-only
(`DCP=1`); only DECODE uses token-interleaved DCP. The byte contract is
recorded in every phase manifest. PREFILL uses 39,040 rank-local and 156,160 target-physical
bytes/token, 2,498,560 target-physical bytes/block, and 304,175 GPU KV blocks
per DP target. DECODE uses 9,760 rank-local and 39,040 target-physical
bytes/token, 624,640 target-physical bytes/block, and 1,216,700 GPU KV blocks
per DP target. Both capacities use the same 190 GB per-GPU budget.

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

### TraceLab v0.0.2 workload

`convert_tracelab_workload.py` converts the released TraceLab DuckDB directly
to the same six-column C++ workload format.  One TraceLab row is one agent/LLM
step (including tool-result continuations), not necessarily one human turn.
The default mapping is:

* `--session-arrival-rate` is required explicitly (there is no converter
  default); retained source sessions are seed-shuffled and roots are placed in
  deterministic strata at `index / session_arrival_rate` seconds;
* successor `think_time` is the time from the previous step's last model-output
  event to the current step's first input event;
* `OSL = output_tokens`; and
* successor `ISL = current_input_total - previous_input_total -
  previous_output_tokens`.  This is the logical new input expected by
  Frontier's prefix-cache materializer, rather than TraceLab's provider-side
  cache-accounting field `newly_append_tokens`.

TraceLab context-max and tokenizer metadata are intentionally ignored.  When a
context reduction/compaction makes that ISL nonpositive or timing is missing,
the current row begins a new numeric simulator session and its full observed
input becomes the new root ISL.  A negative timestamp gap is clamped to
`think_time=0` while retaining the session.  At the first non-adjacent
`round_index`, that row and all later rows in the source session are discarded:
the missing requests make their compute time and context evolution
unobservable.

Split segment roots are not reset to `t=0`.  They are placed at the shuffled
source-root arrival plus source-relative elapsed time from the original first
input to the segment's first input.  If that input timestamp is missing, the
previous segment's final model-output time is used as a conservative fallback.
Frontier has no completion dependency between different simulator sessions,
so this preserves the observed chronology as closely as the workload contract
allows but cannot force a split root to wait for predecessor completion under
simulator queueing.  Metadata records the anchor and any fallback.

Using the isolated TraceLab environment and the v0.0.2 dataset downloaded
under `outputs/`:

```powershell
# A deterministic 100-source-session sample
.\outputs\tools\tracelab-venv\Scripts\python.exe `
  .\cpp\experiments\kimi_k2_cpu_dram\convert_tracelab_workload.py `
  --db .\outputs\datasets\tracelab\v0.0.2\syfi_coding_trace.duckdb `
  --output .\outputs\datasets\tracelab\v0.0.2\frontier\sample100.csv `
  --sample-sessions 100 --seed 20260804 `
  --session-arrival-rate 1.0

# Validate without running a simulation
.\cpp\build\Release\frontier_sim.exe --normalize-workload `
  .\outputs\datasets\tracelab\v0.0.2\frontier\sample100.csv
```

Omit `--sample-sessions` for the complete trace.  Exact `--provider`,
`--model`, and `--session-id` filters can be repeated or comma-separated.  A
metadata JSON and Kimi-study-compatible manifest CSV are written beside the
workload by default (override the manifest with `--manifest-output`).  The
metadata separates pre-sampling and post-sampling session/round counters and
records the seeded shuffle, root-arrival rate, segment anchors, and timing
fallbacks.  The manifest includes `final_context_tokens`, allowing the shared
capacity runner to calculate an unbounded-oracle capacity directly.

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

Those calibration numbers were produced by the earlier all-stage
`sticky_round_robin` configuration. They are retained as historical evidence,
not as results for the new `cache_aware` PREFILL / `vllm_queue_aware` DECODE
configuration. Re-run Phase W and re-check the 2.80--3.20 gate before using
the updated routing policy for a capacity sweep.

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
