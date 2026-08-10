# Kimi K2 / Vera Rubin P16-D16 r1 rep4 result

## Contract

- CPU KV offload: disabled
- TraceLab: 1,000 sampled source sessions, seed 20260803, four repetitions
- Root injection: 1 session/s for 4,000 seconds, 4,000 roots total
- Potential requests: 315,744
- Drain: 15,000 seconds; total horizon 19,000 seconds
- PREFILL: 16 Rubin GPUs, TP1/DP16/EP16
- DECODE: 16 Rubin GPUs, TP4/DCP4/DP4/EP16
- Precision: FP8 except FP4 MoE expert weights; FP8 KV
- HBM: 288 GB/GPU with 10% runtime reserve and automatic block sizing
- Resolved capacity: 347,544 PREFILL and 1,428,552 DECODE blocks/GPU

## Aggregate result

| Metric | Result |
| --- | ---: |
| Completed requests | 189,653 / 315,744 potential (60.07%) |
| Throughput over full horizon | 9.982 req/s |
| Mean / p50 TTFT | 298.58 ms / 95.16 ms |
| p90 / p99 TTFT | 459.53 ms / 4,455.48 ms |
| Mean / p90 scheduling delay | 5.96 ms / 20.68 ms |
| Mean / p90 TPOT | 13.123 ms / 16.696 ms |
| Mean PREFILL batch | 1.06 |
| Mean DECODE batch | 16.24 |
| Estimated PREFILL utilization | 14.84% |
| Estimated DECODE utilization | 97.03% |
| Overall GPU prefix hit | 95.37% |
| Scheduled PREFILL tokens | 1.097 billion |
| Peak PREFILL KV-budget occupancy | 31.27% |
| Peak DECODE KV-budget occupancy | 43.73% |
| Simulator wall clock | 271.28 s |

## Selected 300-second buckets

| Time (s) | PREFILL util | DECODE util | PREFILL batch | DECODE batch | Mean TTFT | p90 TTFT | p90 sched. delay | GPU hit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0–300 | 8.8% | 99.1% | 1.03 | 11.83 | 156.6 ms | 337.0 ms | 0.0 ms | 88.32% |
| 600–900 | 15.7% | 99.4% | 1.06 | 28.63 | 165.7 ms | 325.4 ms | 8.8 ms | 92.69% |
| 1,200–1,500 | 21.1% | 98.5% | 1.07 | 34.83 | 190.7 ms | 341.1 ms | 14.6 ms | 93.50% |
| 1,800–2,100 | 24.3% | 98.8% | 1.07 | 40.87 | 223.8 ms | 416.0 ms | 17.0 ms | 92.99% |
| 2,400–2,700 | 29.8% | 97.3% | 1.10 | 51.16 | 243.1 ms | 423.1 ms | 24.0 ms | 93.71% |
| 3,000–3,300 | 36.2% | 97.7% | 1.12 | 63.66 | 286.3 ms | 474.1 ms | 32.0 ms | 93.74% |
| 3,300–3,600 | **40.7%** | 96.7% | 1.13 | 67.11 | 334.1 ms | 503.7 ms | 38.8 ms | 93.47% |
| 3,600–3,900 | 39.0% | 97.4% | 1.13 | **75.15** | 292.6 ms | 485.6 ms | 33.9 ms | 94.04% |
| 3,900–4,200 | 34.4% | 97.0% | 1.11 | 66.63 | 308.3 ms | 479.0 ms | 33.0 ms | 94.78% |
| 6,000–6,300 | 22.7% | 95.9% | 1.06 | 23.64 | 379.6 ms | 489.4 ms | 34.7 ms | 96.87% |
| 9,000–9,300 | 12.1% | 96.5% | 1.03 | 11.32 | 359.2 ms | 538.1 ms | 1.3 ms | 96.41% |
| 15,000–15,300 | 4.9% | 97.1% | 1.01 | 4.16 | 260.7 ms | 492.5 ms | 0.0 ms | 97.35% |

## DECODE batch size and TPOT

Request TPOT is assigned to the 300-second bucket in which its first decode
token completes. Against the DECODE batch size for the same buckets, the
Pearson correlation is 0.992 during the 0–4,200-second injection/boundary
period and 0.906 over the full 19,000 seconds.

| Time (s) | Mean DECODE batch | Mean TPOT | p90 TPOT |
| --- | ---: | ---: | ---: |
| 0–300 | 11.83 | 9.25 ms | 9.60 ms |
| 600–900 | 28.63 | 10.84 ms | 11.14 ms |
| 1,200–1,500 | 34.83 | 11.70 ms | 12.09 ms |
| 1,800–2,100 | 40.87 | 12.48 ms | 12.93 ms |
| 2,400–2,700 | 51.16 | 14.12 ms | 14.47 ms |
| 3,000–3,300 | 63.66 | 15.92 ms | 16.34 ms |
| 3,300–3,600 | 67.11 | 16.47 ms | 17.31 ms |
| 3,600–3,900 | 75.15 | 17.37 ms | 17.93 ms |
| 4,200–4,500 | 47.00 | 15.92 ms | 16.38 ms |
| 6,000–6,300 | 23.64 | 13.75 ms | 14.30 ms |
| 9,000–9,300 | 11.32 | 11.21 ms | 11.62 ms |
| 15,000–15,300 | 4.16 | 9.65 ms | 10.11 ms |
| 18,000–18,300 | 3.70 | 9.50 ms | 9.88 ms |

Batch size grows 6.35x from the first bucket to its injection-period peak,
while TPOT grows 1.88x. The increase is sublinear because batching amortizes
fixed and parallel work.

Per decode batch, total modeled time grows from 9.17 ms at 0–300 seconds to
17.39 ms at 3,600–3,900 seconds. The combined attention/dense component
accounts for 4.57→12.06 ms, or about 7.49 ms of the 8.23 ms increase. MoE
grouped GEMM is nearly flat at 2.85→2.93 ms; synchronization wait grows from
0.05 to 0.50 ms. Thus the current TPOT slope is dominated by the
attention/dense path, with a smaller contribution from lane synchronization,
not by MoE expert GEMM.

## Saturation interpretation

This run is very different from P8/D24 r1.5 rep6:

- PREFILL does not saturate. Its busiest 300-second bucket is about 41%, mean
  batch size stays close to one, and PREFILL HBM peaks at only 31% of its KV
  budget.
- DECODE is continuously busy at roughly 96–99%. Its batch size rises from 12
  to 75 as the four injection epochs accumulate, then falls after root
  injection stops.
- Despite high DECODE utilization, TTFT and scheduling delay do not diverge.
  Completed-subset arrivals and completions remain closely matched and the
  observed lower-bound backlog returns to zero at the horizon.
- DECODE HBM peaks at only 44%; this is compute saturation rather than KV
  memory saturation.

The system is therefore stable but decode-limited and close to its throughput
boundary. P16 removes PREFILL pressure, while reducing DECODE from six to four
lanes raises mean TPOT to 13.1 ms. A modest additional load or longer OSL may
push DECODE into unstable queue growth even though PREFILL and HBM still have
substantial headroom.

The overall 95.37% GPU hit also indicates that CPU KV offload would affect only
a small fraction of requests in this particular stable configuration.

## Artifacts

- Result: `outputs/tracelab_vera_rubin_p16_d16_r1_s1000_rep4_auto_hbm_cpuoff_detailed/`
- Full 300-second analysis:
  `outputs/tracelab_vera_rubin_p16_d16_r1_s1000_rep4_auto_hbm_cpuoff_detailed/time_buckets_300s.csv`
- Workload: `outputs/datasets/tracelab/v0.0.2/frontier/epoch1000_rep4/r1/`
- Config:
  `cpp/experiments/kimi_k2_cpu_dram/configs/tracelab_vera_rubin_p16_d16_cache_aware.json`
