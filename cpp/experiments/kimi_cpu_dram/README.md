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

The default grid is 0.05 through 0.50 source sessions/s in 0.05 increments,
250/375/500/625/750/875/1000 decimal GB per PREFILL GPU, and a 12-hour
simulation. Override it without editing code:

```bash
python cpp/experiments/kimi_cpu_dram/run_sweep.py \
  --config path/to/simulator-config.json \
  --binary build-release/frontier_sim \
  --output-root outputs/my-sweep \
  --session-rates 0.10,0.15,0.20 \
  --capacities-gb 250,500,1000 \
  --simulation-hours 12 \
  --jobs 4 \
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
  --output-root outputs/my-sweep
```
