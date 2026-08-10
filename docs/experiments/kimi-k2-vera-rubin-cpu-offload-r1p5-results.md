# Kimi K2 / Vera Rubin CPU KV offload r1.5 results

This experiment triples the source-session root arrival rate from 0.5 to
1.5 sessions/s while retaining the same TraceLab sample, seed, two injection
epochs, Rubin P8/D24 topology, scheduler, quantization, HBM planner, and CPU
offload contract described by the r0.5 experiment.

The 2,000 source-session roots are injected over 1,333.33 seconds. The fixed
drain rule gives a 16,333.33-second measurement horizon.

## Aggregate result

| Metric | CPU off | CPU 4 TB | Change |
| --- | ---: | ---: | ---: |
| Completed requests | 93,571 | 93,573 | +2 |
| Request throughput | 5.7298 req/s | 5.7291 req/s | effectively unchanged |
| Mean TTFT | 361.341 ms | 338.686 ms | **-6.27%** |
| p50 TTFT | 110.098 ms | 106.936 ms | -2.87% |
| p90 TTFT | 562.226 ms | 538.165 ms | **-4.28%** |
| p99 TTFT | 5,510.046 ms | 5,111.066 ms | **-7.24%** |
| Mean TPOT | 9.283 ms | 9.286 ms | +0.03% |
| p90 TPOT | 10.416 ms | 10.430 ms | +0.14% |
| Mean scheduling delay | 10.132 ms | 9.355 ms | -7.67% |
| Scheduled PREFILL tokens | 598,696,251 | 563,532,218 | **-5.87%** |
| Mean PREFILL batch | 1.124 | 1.119 | -0.40% |
| Mean DECODE batch | 4.972 | 4.975 | +0.05% |
| Estimated PREFILL lane utilization | 19.37% | 18.24% | -1.13 pp |
| Estimated DECODE lane utilization | 91.99% | 91.93% | -0.06 pp |
| Simulator wall clock | 347.72 s | 357.60 s | +9.88 s |

The utilization estimate is total predicted cluster execution time divided by
the number of DP lanes and measurement time. PREFILL has eight lanes and
DECODE has six TP4/DCP4 lanes.

The paired intersection contains 93,571 requests. On that intersection,
CPU4TB improves mean TTFT by 22.65 ms and reduces scheduled PREFILL by 5.87%.

## Cache result

| Metric | CPU off | CPU 4 TB |
| --- | ---: | ---: |
| Prefix query blocks | 729,869,354 | 729,893,117 |
| GPU direct-hit blocks | 692,494,748 | 692,659,956 |
| GPU direct-hit rate | 94.8793% | 94.8988% |
| CPU-added hit blocks | 0 | 2,056,308 |
| CPU-added hit percentage points | 0 | **0.2817 pp** |
| Combined hit rate | 94.8793% | **95.1805%** |
| Conditional CPU hit among GPU-miss queries | 0 | 5.5228% |
| Requests with any CPU hit | 0 | 395 / 93,573 |
| CPU-restored tokens consumed | 0 | 32,900,928 |
| CPU restore bytes | 0 | 1.294 TB |
| Cumulative CPU offload bytes | 0 | 21.963 TB |

The 395 paired requests with CPU hits have mean TTFT 3,902.60 ms without
offload and 1,056.00 ms with CPU4TB, an average improvement of 2,846.59 ms.

## Comparison with r0.5

| Metric | r0.5 CPU off | r1.5 CPU off |
| --- | ---: | ---: |
| Completed request throughput | 5.0081 req/s | 5.7298 req/s |
| Mean TTFT | 323.706 ms | 361.341 ms |
| p90 TTFT | 495.385 ms | 562.226 ms |
| Mean TPOT | 8.915 ms | 9.283 ms |
| GPU direct-hit rate | 95.2300% | 94.8793% |
| Mean PREFILL batch | 1.072 | 1.124 |
| Mean DECODE batch | 4.320 | 4.972 |
| PREFILL utilization | 16.33% | 19.37% |
| DECODE utilization | 91.69% | 91.99% |
| CPU4TB mean-TTFT reduction | 3.22% | **6.27%** |

Tripling the root-session arrival parameter does not triple completed request
throughput. DECODE was already near saturation at r0.5 and remains around 92%
utilized. TraceLab successors are completion-relative, so slower completion
holds back later turns; r1.5 therefore completes fewer requests in the shorter
fixed horizon even though roots arrive three times faster. A true 3x open-loop
request-rate experiment would require materializing successor arrivals
independently of simulated completion.

The higher working set lowers the no-offload GPU hit rate by 0.35 percentage
points and increases CPU-hit requests from 257 to 395. CPU4TB therefore saves
more PREFILL work and its mean-TTFT benefit grows from 3.22% to 6.27%, while
TPOT and throughput remain decode-limited.

## Artifacts

- CPU off: `outputs/tracelab_vera_rubin_r1p5_s1000_auto_hbm_cpuoff_detailed/`
- CPU 4 TB: `outputs/tracelab_vera_rubin_r1p5_s1000_auto_hbm_cpu4tb_detailed/`
- Workload: `outputs/datasets/tracelab/v0.0.2/frontier/epoch1000/r1p5/`

