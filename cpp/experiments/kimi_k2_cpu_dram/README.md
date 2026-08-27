# Kimi K2 experiment config

This directory intentionally contains only the current Kimi K2 PREFILL-only
simulator config. Workload conversion, rate/capacity sweeping, analysis, and
HTML reporting are shared with Kimi K3 under
[`../kimi_cpu_dram/`](../kimi_cpu_dram/).

Use
[`configs/tracelab_vera_rubin_p8_d16_prefill_only_cpu_per_gpu.json`](configs/tracelab_vera_rubin_p8_d16_prefill_only_cpu_per_gpu.json)
with the shared runner. CPU capacities passed to the runner are decimal GB per
PREFILL GPU.

```bash
python cpp/experiments/kimi_cpu_dram/run_sweep.py \
  --config cpp/experiments/kimi_k2_cpu_dram/configs/tracelab_vera_rubin_p8_d16_prefill_only_cpu_per_gpu.json \
  --binary build-release/frontier_sim \
  --output-root outputs/kimi_k2_prefill_only_v3_12h \
  --regenerate-workloads --resume
```
