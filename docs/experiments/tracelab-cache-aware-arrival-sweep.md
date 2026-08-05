# TraceLab cache-aware arrival-rate sweep

This experiment finds the sustainable TraceLab source-session arrival rate for
the cache-aware PREFILL routing policy.  It runs the same seeded 3,000-source-
session sample twice at six rates from 0.10 down to 0.05 sessions/s, then
collects detailed system, latency, cache, migration, and per-lane balance
metrics into CSV and Markdown tables.

## Frozen experiment configuration

The default config is
`cpp/experiments/kimi_k2_cpu_dram/configs/tracelab_p8_d24_cache_aware.json`.

| Item | Setting |
| --- | --- |
| Model | `moonshotai/Kimi-K2-Instruct` |
| Device and precision | GB300 analytical model, FP8 |
| Architecture | sequential online PDD, one replica |
| PREFILL | 8 GPUs, TP1 / DCP1 / DP8 / EP8, chunked PREFILL |
| DECODE | 24 GPUs, TP8 / DCP8 / DP3 / EP24 |
| Scheduler cap | 64 requests |
| PREFILL token cap | 131,072 tokens/batch |
| DECODE token cap | 8,192 tokens/batch |
| GPU KV | session prefix cache, 16-token blocks |
| CPU KV offload | disabled |
| PREFILL routing | actual GPU-cache-aware |
| DECODE routing | vLLM-like least-outstanding |
| Cache threshold | 0.5 |
| Absolute load-gap threshold | 8 outstanding requests |
| Relative load threshold | 1.5× |
| Migration | discard the old PREFILL target's session KV |
| TraceLab sample | seed 20260803, 3,000 source sessions |
| Repetitions | two independently reshuffled injection epochs |
| Rates | 0.10, 0.09, 0.08, 0.07, 0.06, 0.05 sessions/s |

The load check uses `running requests + waiting requests` per PREFILL DP lane.
The cache target is abandoned when both `max_load - min_load > 8` and
`max_load > 1.5 * min_load`, or when the mapped GPU owns less than 50% of the
request prefix.

## Measurement horizon

One injection epoch lasts `3,000 / arrival_rate` simulated seconds.  The
runner includes two complete injection epochs and a settling interval:

```text
epoch_seconds = 3000 / rate
drain_seconds = max(15000, 0.4 * epoch_seconds)
simulation_end = 2 * epoch_seconds + drain_seconds
```

| Session arrival rate | Simulation end time |
| ---: | ---: |
| 0.10 | 75,000 s |
| 0.09 | 81,666.667 s |
| 0.08 | 90,000 s |
| 0.07 | 102,857.143 s |
| 0.06 | 120,000 s |
| 0.05 | 144,000 s |

This reproduces the previously used horizons at 0.05, 0.08, and 0.10.  Change
`--drain-fraction` or `--minimum-drain-seconds` only when intentionally
changing the measurement contract.

## Build on the server

From the repository root on Linux:

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build --target frontier_sim -j
```

On Windows PowerShell:

```powershell
cmake -S .\cpp -B .\cpp\build
cmake --build .\cpp\build --config Release --target frontier_sim -j
```

The simulator and sweep runner need no Python simulator environment.  The
TraceLab converter needs the `duckdb` Python package only when a requested rate
does not already have a converted workload:

```bash
python -m pip install duckdb
```

## Prepare and inspect the matrix

The dry run validates existing workload metadata, prints the three missing
rate-conversion commands when necessary, and shows every simulator command
without launching it:

```bash
python cpp/experiments/kimi_k2_cpu_dram/run_tracelab_arrival_sweep.py \
  --binary cpp/build/frontier_sim \
  --tracelab-db outputs/datasets/tracelab/v0.0.2/syfi_coding_trace.duckdb \
  --output-root outputs/tracelab_cache_aware_arrival_sweep \
  --dry-run
```

If TraceLab conversion must run in a separate environment, provide its Python
executable:

```bash
python cpp/experiments/kimi_k2_cpu_dram/run_tracelab_arrival_sweep.py \
  --converter-python /path/to/tracelab-venv/bin/python \
  --dry-run
```

Before the full matrix, a short end-to-end smoke run is recommended:

```bash
python cpp/experiments/kimi_k2_cpu_dram/run_tracelab_arrival_sweep.py \
  --binary cpp/build/frontier_sim \
  --rates 0.05 \
  --simulation-end-time-s 300 \
  --output-root outputs/tracelab_cache_aware_arrival_sweep_smoke
```

`--simulation-end-time-s` is a diagnostic override and must be omitted from
the full experiment.

## Run or resume the sweep

The safe default is one simulation at a time:

```bash
python cpp/experiments/kimi_k2_cpu_dram/run_tracelab_arrival_sweep.py \
  --binary cpp/build/frontier_sim \
  --tracelab-db outputs/datasets/tracelab/v0.0.2/syfi_coding_trace.duckdb \
  --output-root outputs/tracelab_cache_aware_arrival_sweep \
  --resume
```

Use `--jobs 2` or higher only when the server has enough memory.  The 0.05
run observed locally used roughly 8 GB near completion and took about 14
minutes; concurrent runs multiply that memory requirement.  Every rate writes
its own `simulator.log`, so parallel output does not interleave.

To run a subset or override the routing thresholds:

```bash
python cpp/experiments/kimi_k2_cpu_dram/run_tracelab_arrival_sweep.py \
  --binary cpp/build/frontier_sim \
  --rates 0.08,0.07,0.06 \
  --cache-threshold 0.5 \
  --balance-abs-threshold 8 \
  --balance-rel-threshold 1.5 \
  --output-root outputs/tracelab_cache_aware_arrival_sweep \
  --resume
```

Useful safety and control flags:

- `--resume`: skip a rate only when `run.json` says `completed` and required
  result artifacts exist.
- `--no-generate-workloads`: fail instead of querying the TraceLab database
  when a workload is missing.
- `--regenerate-workloads`: intentionally replace converted workloads after
  validating that a new conversion is desired.
- `--jobs N`: run N independent rates concurrently; default 1.
- `--output-mode requests`: required for the full detailed report and used by
  default.
- `--runtime-validation`: enable expensive per-event invariant validation for
  diagnostics; disabled for the full matrix by default.
- `--no-gpu-kv-occupancy`: omit occupancy sampling only for a deliberate
  reduced-output run.

Each rate directory contains:

```text
<output-root>/r0p05/
├── config.input.json
├── config.normalized.json
├── run.json
├── simulator.log
├── summary.json
├── requests.csv
├── workload.normalized.csv
└── gpu_kv_occupancy.csv
```

The output root also contains `sweep_plan.json`, which records the complete
matrix and measurement contract before simulations launch.

`run.json` records the exact command, workload/manifest paths, thresholds,
simulation horizon, process wall time, exit code, and completion status.

## Collect detailed results

After all or some rates finish, run:

```bash
python cpp/experiments/kimi_k2_cpu_dram/collect_tracelab_arrival_sweep.py \
  --output-root outputs/tracelab_cache_aware_arrival_sweep
```

The collector tolerates missing or failed rates and reports their status.  It
writes machine-readable CSV plus a Markdown report with, where available:

- simulated and process wall time, request/event/batch counts, throughput;
- TTFT, TPOT, scheduling delay, PREFILL, and end-to-end latency percentiles;
- PREFILL/DECODE batch counts, mean sizes, histograms, and execution-time
  components;
- scheduled/recomputed PREFILL and theoretical/implementation-minimum work;
- request-level GPU block hit, miss, CPU hit/restore/offload, and eviction
  metrics;
- PREFILL lane request/work distribution and imbalance ratios;
- session migration request count, migrated-session count, migration rate,
  and hit/TTFT behavior of migrated versus sticky requests;
- GPU KV occupancy peaks when occupancy output is present.

Open the generated Markdown file directly or paste its tables into the
experiment report.  The CSV is intended for plotting and cross-run regression
checks.
