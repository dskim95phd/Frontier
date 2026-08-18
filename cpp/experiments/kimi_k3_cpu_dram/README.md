# Kimi K3 CPU KV-cache experiments

This directory contains the Kimi K3 CPU-offload experiment runners, their
K3-specific configuration, reporting wrapper, and tests. Shared TraceLab
conversion and capacity-report analysis utilities remain in
`../kimi_k2_cpu_dram/`.

## CPU-per-GPU capacity sweep

`run_dense_cpu_capacity_rate_sweep.py` is the asymmetric PDD capacity runner.
The intended P24/D64 topology is PREFILL TP1/PP24/DP1 with
pipeline-exclusive collapsed events and DECODE TP4/PP1/DP16/EP64 with the
exact event path. Check the selected JSON configuration before each run,
because diagnostic P24/D32 variants have also been used.

The default capacity grid contains eight points:

```text
CPU off, then 250, 375, 500, 625, 750, 875, and 1000 decimal GB
per PREFILL GPU.
```

With 24 PREFILL GPU slices, the enabled points resolve to 6, 9, 12, 15, 18,
21, and 24 TB of aggregate CPU KV capacity. Directory names such as
`cpu0250gb` retain analyzer compatibility; the number means capacity per
PREFILL GPU, not a shared two-GPU CPU pool.

The checked-in `PILOT_SESSION_RATE=0.30` is a runnable starting point and must
be calibrated before interpreting a long K3 run as a capacity-knee result.
The runner enables wall-clock progress output every 60 seconds by default;
pass `--wall-progress-interval-s 0` to disable it or another positive value to
change the interval.

### Performance-run safety contract

Kimi K3 CPU-offload performance runs must disable full runtime validation and
the GPU KV occupancy stream.  In the P24/D32 exact workload, runtime validation
made a 180-second probe about 26 times slower because it repeatedly scanned a
CPU KV cache containing hundreds of thousands of materialized blocks.  The
Python runner therefore always emits the following explicit simulator options,
prints their state at startup, and records them in both `sweep_plan.json` and
each case's `run.json`:

```text
--runtime-validation false --gpu-kv-occupancy false
```

Enabling either diagnostic option through the runner additionally requires
`--allow-expensive-diagnostics`, so an accidental flag cannot silently turn a
benchmark into a validation run.  When invoking `frontier_sim` directly for a
performance comparison, include both explicit `false` options; do not rely on
the simulator CLI defaults.

```powershell
# Inspect the eight commands without running them.
python .\cpp\experiments\kimi_k3_cpu_dram\run_dense_cpu_capacity_rate_sweep.py `
  --dry-run --no-generate-reports

# Run or resume the complete eight-point pilot.
python .\cpp\experiments\kimi_k3_cpu_dram\run_dense_cpu_capacity_rate_sweep.py `
  --binary .\cpp\build\Release\frontier_sim.exe --session-rate 0.30 `
  --jobs 4 --resume
```

For a 12-hour endpoint comparison at 0.5 sessions/s:

```powershell
python .\cpp\experiments\kimi_k3_cpu_dram\run_dense_cpu_capacity_rate_sweep.py `
  --binary .\cpp\build\Release\frontier_sim.exe `
  --session-rate 0.5 --simulation-hours 12 `
  --capacities-gb 250,1000 --jobs 2 --resume
```

Report generation discovers completed results from the output directory, not
only from the most recently written `sweep_plan.json`.  This makes split runs
additive: invoke the runner several times with the same `--output-root` and
different `--capacities-gb` subsets, and each regenerated HTML/JSON/CSV report
will contain every completed `r*/cpu*gb/r1` point already present there.
Incomplete or failed cases are ignored.  Reports can also be refreshed
directly:

```powershell
python .\cpp\experiments\kimi_k3_cpu_dram\generate_dense_cpu_capacity_rate_reports.py `
  --output-root .\outputs\my-k3-capacity-sweep
```

## Fixed 4 TB rate screen

`run_cpu4tb_rate_sweep.py` runs the shorter fixed-capacity arrival-rate screen
used while calibrating a sustainable K3 session rate.
