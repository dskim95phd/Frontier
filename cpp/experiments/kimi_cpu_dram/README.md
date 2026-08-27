# Shared Kimi CPU-DRAM sweep

This directory is the single maintained experiment workflow for Kimi K2 and
Kimi K3. Model-specific directories contain simulator configs only.

## Maintained tools

- `convert_tracelab_workload.py` converts the TraceLab DuckDB to deterministic
  session workloads. A nonpositive logical ISL is retained as compaction only
  after at least a 75% context reduction; ambiguous tails are discarded.
- `run_sweep.py` generates workloads, runs the rate/capacity Cartesian product,
  resumes completed cases, and invokes the report generator.
- `generate_report.py` creates one HTML/JSON/CSV report per rate plus the root
  `index.html`. Private `lib/` modules contain report calculations and are not
  separate experiment entry points.

The `SESSION_RATES_BY_CAPACITY_GB` mapping near the top of `run_sweep.py` is
the server experiment matrix. Each CPU capacity can have a different tuple of
session rates. The checked-in matrix uses 0.05 through 0.50 sessions/s for
each of 250/375/500/625/750/875/1000 decimal GB per PREFILL GPU. For example:

```python
SESSION_RATES_BY_CAPACITY_GB = {
    250: ("0.05", "0.10", "0.15"),
    500: ("0.15", "0.20", "0.25"),
    1000: ("0.25", "0.30", "0.35"),
}
```

The default simulation horizon is 12 hours. CLI values can override the
script mapping without editing it:

```bash
python cpp/experiments/kimi_cpu_dram/run_sweep.py \
  --config path/to/simulator-config.json \
  --binary build-release/frontier_sim \
  --output-root outputs/my-sweep \
  --session-rates 0.10,0.15,0.20 \
  --capacities-gb 250,500,1000 \
  --simulation-hours 12 \
  --jobs 4 \
  --report-jobs 4 \
  --regenerate-workloads \
  --resume
```

The TraceLab database defaults to
`outputs/datasets/tracelab/v0.0.2/syfi_coding_trace.duckdb`. Use a Release
binary for performance experiments. `--runtime-validation` and
`--gpu-kv-occupancy` remain off unless explicitly enabled with
`--allow-expensive-diagnostics`.

Reports can be rebuilt independently:

```bash
python cpp/experiments/kimi_cpu_dram/generate_report.py \
  --output-root outputs/my-sweep \
  --report-jobs 4
```

Report jobs are separate processes, one session-rate report per process. The
default is up to four processes. Every completed canonical case already under
the output root is included by default, even if it is absent from the latest
`sweep_plan.json`. Use `--no-include-existing-results` only when the report
must be restricted to that latest plan.
