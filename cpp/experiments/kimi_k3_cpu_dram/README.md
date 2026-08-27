# Kimi K3 experiment config

This directory intentionally contains only the current Kimi K3 PREFILL-only
simulator config. Workload conversion, rate/capacity sweeping, analysis, and
HTML reporting are shared with Kimi K2 under
[`../kimi_cpu_dram/`](../kimi_cpu_dram/).

Use
[`configs/tracelab_k3_p24_d64_prefill_only_batch8192_cpu_per_gpu.json`](configs/tracelab_k3_p24_d64_prefill_only_batch8192_cpu_per_gpu.json)
with the shared runner. It keeps chunked PREFILL enabled with a 512-token
threshold and an 8,192-token PREFILL batch budget.

```bash
python cpp/experiments/kimi_cpu_dram/run_sweep.py \
  --config cpp/experiments/kimi_k3_cpu_dram/configs/tracelab_k3_p24_d64_prefill_only_batch8192_cpu_per_gpu.json \
  --binary build-release/frontier_sim \
  --output-root outputs/kimi_k3_prefill_only_v3_12h \
  --regenerate-workloads --resume
```
