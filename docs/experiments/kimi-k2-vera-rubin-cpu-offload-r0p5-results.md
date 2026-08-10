# Kimi K2 / Vera Rubin CPU KV offload r0.5 results

## Experiment contract

- Workload: TraceLab v0.0.2, seed 20260803, 1,000 sampled source sessions,
  two injection epochs, 0.5 new source sessions/s
- Measurement horizon: 19,000 simulated seconds
- Model: `moonshotai/Kimi-K2-Instruct`
- PREFILL: 8 Rubin GPUs, TP1 / DP8 / EP8, 1,024-token chunking
- DECODE: 24 Rubin GPUs, TP4 / DCP4 / DP6 / EP24
- Batch cap: 256
- Precision: FP8 except FP4 MoE expert weights; FP8 KV
- Routing: cache-aware PREFILL (`0.5`, `8`, `1.5`) and vLLM-like
  queue-aware DECODE
- GPU HBM: 288 GB/GPU with a 10% runtime reserve and automatic KV-block
  calculation

The automatic memory planner resolved 296,780 PREFILL blocks/GPU and
1,496,237 DECODE blocks/GPU. CPU4TB means an analytical 4 TB CPU attached to
two GPUs, hence a static 2 TB/GPU slice. With PREFILL TP1/DP8 this is 2 TB per
cache target and 16 TB over all eight targets. Bandwidth remains the Rubin
Vera contract: 600 GB/s DRAM and 900 GB/s C2C per GPU slice, so DRAM is the
transfer bottleneck. The 4 TB capacity is analytical; physical Vera capacity
is 1.5 TB/CPU.

## Aggregate result

| Metric | CPU off | CPU 4 TB | Change |
| --- | ---: | ---: | ---: |
| Completed requests | 95,154 | 95,152 | -2 |
| Request throughput | 5.0081 req/s | 5.0086 req/s | +0.01% |
| Mean TTFT | 323.706 ms | 313.284 ms | **-3.22%** |
| p50 TTFT | 94.714 ms | 94.378 ms | -0.36% |
| p90 TTFT | 495.385 ms | 487.061 ms | -1.68% |
| p99 TTFT | 5,023.905 ms | 4,807.450 ms | -4.31% |
| Mean TPOT | 8.915 ms | 8.921 ms | +0.07% |
| p90 TPOT | 9.964 ms | 9.977 ms | +0.13% |
| Mean scheduling delay | 6.930 ms | 6.874 ms | -0.81% |
| Scheduled PREFILL tokens | 569,356,393 | 549,975,755 | **-3.40%** |
| Mean PREFILL batch | 1.072 | 1.072 | unchanged |
| Mean DECODE batch | 4.320 | 4.326 | +0.14% |
| Simulator wall clock | 396.14 s | 406.35 s | +10.21 s |

The paired intersection contains 95,151 requests. On this intersection, CPU
offload reduces scheduled PREFILL by 3.39% and mean TTFT by 10.41 ms.

## Cache result

Cache statistics below are streamed directly from `requests.csv`; the current
top-level `summary.json` cache counters remain zero even in request output
mode and must not be used for this experiment.

| Metric | CPU off | CPU 4 TB |
| --- | ---: | ---: |
| Prefix query blocks | 745,079,195 | 745,066,192 |
| GPU direct-hit blocks | 709,539,055 | 709,288,101 |
| GPU direct-hit rate | **95.2300%** | **95.1980%** |
| CPU-added hit blocks | 0 | 1,449,240 |
| CPU-added hit percentage points | 0 | **0.1945 pp** |
| Combined hit rate | 95.2300% | **95.3925%** |
| Conditional CPU hit among GPU-miss queries | 0 | 4.0506% |
| Requests with any CPU hit | 0 | 257 / 95,152 |
| CPU-restored tokens consumed | 0 | 23,187,840 |
| CPU restore bytes | 0 | 909.38 GB |
| Cumulative CPU offload bytes | 0 | 21.44 TB |

The 257 requests that receive CPU hits have mean TTFT 4,689.76 ms in the
paired CPU-off run and 1,318.95 ms with CPU4TB, an average improvement of
3,370.81 ms. They save about 74,250 PREFILL tokens per affected request.

## Interpretation

The workload is already a GPU-cache-friendly regime: roughly 95.2% of queried
prefix blocks hit HBM without CPU offload. CPU4TB adds only 0.1945 percentage
points to the global hit rate and affects 0.27% of completed requests.
Consequently median TTFT and throughput barely move, while mean and p99 TTFT
improve because the small set of long-context misses avoids very expensive
re-PREFILL. TPOT is unchanged, as expected, because CPU KV offload changes
PREFILL reuse rather than DECODE execution.

## Artifacts

- CPU off: `outputs/tracelab_vera_rubin_r0p5_s1000_auto_hbm_cpuoff_detailed/`
- CPU 4 TB: `outputs/tracelab_vera_rubin_r0p5_s1000_auto_hbm_cpu4tb_detailed/`
- CPU 4 TB config:
  `cpp/experiments/kimi_k2_cpu_dram/configs/tracelab_vera_rubin_p8_d24_cache_aware_cpu4tb.json`

