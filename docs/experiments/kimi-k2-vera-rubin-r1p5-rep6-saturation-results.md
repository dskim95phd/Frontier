# Kimi K2 / Vera Rubin r1.5 rep6 saturation result

## Contract

- CPU KV offload: disabled
- TraceLab sample: 1,000 source sessions, seed 20260803
- Repetitions: 6, producing 6,000 source-session root injections and 473,616
  potential requests
- Root rate: 1.5 sessions/s
- Root injection interval: 0–4,000 seconds
- Drain: 15,000 seconds; simulation horizon 19,000 seconds
- Model/topology: Kimi K2, Rubin PREFILL 8 GPUs TP1/DP8/EP8 and DECODE
  24 GPUs TP4/DCP4/DP6/EP24
- HBM: 288 GB/GPU, automatic capacity of 296,780 PREFILL and 1,496,237
  DECODE blocks/GPU

## Aggregate result

| Metric | Result |
| --- | ---: |
| Completed requests | 132,899 / 473,616 potential (28.06%) |
| Completed throughput over full horizon | 6.995 req/s |
| Mean / p50 TTFT | 518.05 s / 4.961 s |
| p90 / p99 TTFT | 1,602.71 s / 1,963.32 s |
| Mean scheduling delay | 392.23 s |
| p90 scheduling delay | 1,266.30 s |
| Mean / p90 TPOT | 9.295 ms / 11.078 ms |
| Mean PREFILL batch | 23.56 |
| Mean DECODE batch | 5.74 |
| Estimated PREFILL utilization | 96.51% |
| Estimated DECODE utilization | 98.52% |
| Scheduled PREFILL tokens | 7.070 billion |
| Overall GPU prefix block hit | 45.08% |
| Simulator wall clock | 414.90 s |

Only completed requests are retained in `requests.csv`; arrival-bucket latency
below therefore describes the completed subset. It is not a count of every
request waiting at the horizon.

## Selected 300-second buckets

Latency and GPU hit are grouped by request arrival time. Utilization and batch
size come from the simulator's 60-second batch buckets aggregated to 300
seconds.

| Time (s) | PREFILL util | DECODE util | PREFILL batch | DECODE batch | Mean TTFT | p90 TTFT | p90 sched. delay | GPU hit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0–300 | 28.7% | 98.9% | 1.11 | 11.30 | 0.18 s | 0.41 s | 0.02 s | 89.0% |
| 600–900 | 53.6% | 98.6% | 1.23 | 24.65 | 0.26 s | 0.50 s | 0.04 s | 91.6% |
| 1,200–1,500 | 85.8% | 97.9% | 1.57 | 32.85 | 0.49 s | 0.89 s | 0.06 s | 89.8% |
| 1,500–1,800 | **95.0%** | 97.4% | 2.38 | 34.97 | 0.95 s | 1.59 s | 0.12 s | 88.8% |
| 1,800–2,100 | **100.1%** | 98.0% | 12.13 | 32.69 | 4.10 s | 9.26 s | 0.34 s | 84.6% |
| 2,100–2,400 | 100.0% | 97.1% | 35.41 | 28.92 | 13.10 s | 35.44 s | 0.81 s | 81.9% |
| 2,400–2,700 | 100.2% | 99.0% | 89.40 | 12.84 | 66.96 s | 194.57 s | 5.46 s | 54.7% |
| 2,700–3,000 | 100.1% | 99.5% | 116.33 | 5.64 | 215.52 s | 362.10 s | 114.53 s | 0.0% |
| 3,000–3,300 | 99.9% | 99.5% | 116.52 | 5.37 | 304.36 s | 443.58 s | 208.10 s | 0.0% |
| 3,600–3,900 | 99.9% | 99.7% | 114.21 | 5.17 | 494.09 s | 649.64 s | 396.19 s | 0.0% |
| 3,900–4,200 | 100.1% | 99.8% | 114.36 | 5.19 | 609.70 s | 779.30 s | 485.15 s | 0.0% |
| 6,000–6,300 | 100.1% | 99.5% | 105.87 | 4.33 | 1,116.33 s | 1,326.07 s | 920.56 s | 0.0% |
| 9,000–9,300 | 99.8% | 99.8% | 101.28 | 3.36 | 1,468.08 s | 1,706.61 s | 1,269.63 s | 0.0% |
| 12,000–12,300 | 99.9% | 98.6% | 99.99 | 2.70 | 1,614.92 s | 1,882.56 s | 1,434.87 s | 0.0% |
| 15,000–15,300 | 99.9% | 97.8% | 102.64 | 2.41 | 1,691.68 s | 1,921.59 s | 1,406.89 s | 0.0% |

Values slightly above 100% are bucket-boundary/aggregation effects and mean
the cluster is effectively continuously busy.

## HBM saturation

All eight PREFILL lanes reached 95% of their usable KV-block budgets between
2,486 and 2,586 seconds, and 99% between 2,509 and 2,603 seconds. Every lane
reached 100% at least once. At the end of the 19,000-second horizon PREFILL
still held 2,329,770 of 2,374,240 total blocks, or 98.13%.

DECODE HBM is not the memory bottleneck: its busiest lane peaked at only 17.75%
of the DECODE KV budget. The maximum active-KV share of physical HBM was
64.37% on PREFILL, which is exactly the usable KV portion left after Kimi K2
weights and the 10% runtime reserve.

## Saturation interpretation

This is an unstable overload, not merely a late steady-state plateau:

1. DECODE compute is already about 99% utilized in the first 300 seconds.
2. PREFILL compute crosses 95% around 1,500 seconds and reaches continuous
   saturation around 1,800 seconds.
3. PREFILL HBM fills around 2,500–2,600 seconds.
4. GPU prefix hit then collapses; the completed-arrival buckets after 2,700
   seconds show no reuse and PREFILL batches rise above 110.
5. Root injection stops at 4,000 seconds, but both clusters remain essentially
   saturated for the full 15,000-second drain. PREFILL HBM is still 98% full
   at the end and the system has not returned to an idle state.

The rep6 setup therefore overshoots a useful saturation-boundary experiment.
The next useful sweep should keep the same 4,000-second injection window but
try intermediate repetition counts, especially rep3, rep4, and rep5, and use
these same time buckets to locate the highest stable configuration.

## Artifacts

- Result: `outputs/tracelab_vera_rubin_r1p5_s1000_rep6_auto_hbm_cpuoff_detailed/`
- Full 300-second analysis:
  `outputs/tracelab_vera_rubin_r1p5_s1000_rep6_auto_hbm_cpuoff_detailed/time_buckets_300s.csv`
- Workload: `outputs/datasets/tracelab/v0.0.2/frontier/epoch1000_rep6/r1p5/`
