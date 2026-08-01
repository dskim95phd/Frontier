# Single-turn load sweep

This benchmark studies how offered load, request batch-size limits, KV-cache
capacity, and co-location versus PDD topology affect the C++ simulator.

## Scope

The request generator starts from the random **first-turn** method used by the
Kimi K2 CPU-offload experiment on `main`:

- Poisson arrivals (exponentially distributed inter-arrival gaps)
- bounded log-normal prompt lengths: median 8,192, sigma 0.8, range
  2,048-32,768 tokens
- deterministic PCG seed `20260728`

Every configuration reuses the same request lengths and unit arrival gaps. This
makes comparisons paired and deterministic. Multi-turn/session behavior is
intentionally excluded.

The distribution and CSV-writing implementation lives in
`cpp/frontier/request_generator/workload_generator.py`. The benchmark imports
that standalone C++-input tool instead of maintaining a separate generator.

Three output-length profiles are available:

| Profile | OSL median | OSL range | Purpose |
| --- | ---: | ---: | --- |
| `balanced-8k-1k` (default) | 1,024 | 256-4,096 | Long-context generation |
| `prefill-heavy` | 256 | 64-1,024 | Prefill-dominant serving |
| `kimi-short` | 8 | 4-16 | Original Kimi script compatibility |

This default sweep is **not** a Kimi K2 model-performance study. C++ can load
Kimi K2's Python-compatible asset and analytically model its latent MLA cache,
but the checked-in sweep deliberately retains the established Llama-2-7B FP16
baseline on Rubin so historical results remain comparable:

- one replica
- TP8, PP4, DP2 (64 simulated GPUs)
- 8,192-token scheduler budget by default
- 16-token KV blocks

The PDD comparison preserves the same 64-GPU total and uses a 1:1
Prefill:Decode split:

- `pdd-pp-half`: Prefill and Decode each use TP8, PP2, DP2 = 32 GPUs.
- `pdd-tp-half`: Prefill and Decode each use TP4, PP4, DP2 = 32 GPUs.

Both clusters contain a full model copy. PDD KV transfer uses the Kimi K2 NVL12
sweep assumption of 38,400 Gbit/s aggregate bandwidth and 0.02 ms latency.

`offered_concurrency` is converted to an arrival rate using
`arrival_rate = offered_concurrency / isolated_mean_service_time`. Under
saturation, queued requests cause the measured time-averaged concurrency to be
larger; both values are written to the result.

## Run

Build the Release benchmark:

```powershell
cmake --build .build-step35-release --config Release `
  --target frontier_load_benchmark_summary
```

Run the default 60-case sweep:

```powershell
python cpp/benchmarks/run_single_turn_load_sweep.py `
  --binary .build-step35-release/frontier_load_benchmark_summary.exe
```

The default remains the historical all-FP16 baseline. To run FP8 attention,
dense, router, KV cache, and communication with FP4 MoE experts, add:

```powershell
  --precision-profile fp8-fp4-mixed
```

The mixed profile also changes PDD KV-transfer bytes and logical KV-capacity
conversion to the FP8 byte width.

The default grid contains 64 requests per case, offered concurrency
`1,2,4,8,16,32`, batch caps `1,4,8,16,32`, and 4/16 GiB of logical KV
capacity per GPU. Results and generated inputs are written under
`outputs/cpp_single_turn_load_sweep_colocation_balanced-8k-1k/`.

`--prefill-chunk-tokens` controls the maximum Prefill tokens scheduled for
one request in one iteration. It defaults to 8,192.
`--max-tokens-in-batch` independently controls the total scheduler token
budget and also defaults to 8,192. For example, the 256-token chunk experiment
uses:

```powershell
python cpp/benchmarks/run_single_turn_load_sweep.py `
  --binary .build-step35-release/frontier_load_benchmark_summary.exe `
  --prefill-chunk-tokens 256
```

## Decode batching tradeoff

The `pdd-decode-tradeoff` topology overprovisions Prefill with TP4/PP4/DP8
(128 GPUs) and holds Decode at TP4/PP4/DP2 (32 GPUs). In unbounded Decode mode,
the request count is used only as a non-binding safety cap; offered load, the
token budget, and KV capacity determine the realized Decode batch size.

```powershell
python cpp/benchmarks/run_single_turn_load_sweep.py `
  --binary .build-step35-release/frontier_load_benchmark_summary.exe `
  --output-dir outputs/cpp_decode_batch_tradeoff_pdd_balanced `
  --topology pdd-decode-tradeoff `
  --workload-profile balanced-8k-1k `
  --requests 256 `
  --offered-concurrency 1,2,4,8,16,32,64,128,256,512 `
  --unbounded-decode-batch `
  --prefill-batch-size-cap 64 `
  --kv-capacities-gib 64 `
  --measurement-trim-fraction 0.1
```

The result includes cluster-specific batch distributions,
`user_decode_tps_*`, `decode_tokens_per_gpu_s`, and an interactive
`decode_batch_tradeoff.html` plot.

To run one high-load tail-TTFT case instead of a grid, provide one value for
the offered concurrency, batch cap, and KV capacity:

```powershell
python cpp/benchmarks/run_single_turn_load_sweep.py `
  --binary .build-step35-release/frontier_load_benchmark_summary.exe `
  --output-dir outputs/manual_c128_b128_chunk1024 `
  --topology colocation `
  --workload-profile prefill-heavy `
  --requests 1024 `
  --offered-concurrency 128 `
  --batch-sizes 128 `
  --prefill-chunk-tokens 1024 `
  --max-tokens-in-batch 32768 `
  --kv-capacities-gib 64 `
  --measurement-trim-fraction 0.1 `
  --arrival-reference-service-seconds 0.6098279083830135
```

This runs one calibration plus one measured case. The fixed arrival reference
matches the paired 1,024-request tail-TTFT experiment; omit it to derive the
arrival rate from that run's own calibration.

For paired topology runs, first read `isolated_mean_service_seconds` from the
co-location `manifest.json`, then pass it to both PDD runs. The value below is
the reference produced by the checked-in seed and default 64-request 8K/1K
profile:

```powershell
python cpp/benchmarks/run_single_turn_load_sweep.py `
  --binary .build-step35-release/frontier_load_benchmark_summary.exe `
  --topology pdd-pp-half `
  --arrival-reference-service-seconds 2.5587908414683014

python cpp/benchmarks/run_single_turn_load_sweep.py `
  --binary .build-step35-release/frontier_load_benchmark_summary.exe `
  --topology pdd-tp-half `
  --arrival-reference-service-seconds 2.5587908414683014
```

The benchmark executable uses the production simulator and emits a compact
request/batch summary instead of serializing the full output contract. The
simulator still retains its detailed parity trace internally. Very small KV
capacities combined with large batch caps can create enough preemption events
to consume substantial host RAM; use a smaller request count for such
diagnostic runs.

## Original short-output 128-request snapshot

The completed experiment covered 108 configurations, or 13,824 request
simulations. The realized prompt length was 10,774.7 tokens on average
(p50 7,459, p90 25,988.5, max 32,768); output length averaged 9.76 tokens.
This section uses the legacy `kimi-short` profile and is retained only as a
prefill-dominated reference.

At offered concurrency 16 with 16 GiB KV per GPU:

| Batch cap | Requests/s | Total tokens/s | TTFT p90 (ms) |
| ---: | ---: | ---: | ---: |
| 1 | 27.858 | 300,432 | 3,815.2 |
| 2 | 40.592 | 437,763 | 2,505.1 |
| 4 | 55.197 | 595,264 | 1,748.2 |
| 8 | 75.271 | 811,751 | 1,165.9 |
| 16 | 102.585 | 1,106,319 | 773.1 |
| 32 | 132.991 | 1,434,225 | 498.5 |

For the same offered concurrency 16 and 16 GiB KV case, TPOT changes in the
opposite direction:

| Batch cap | TPOT mean (ms) | TPOT p50 (ms) | TPOT p90 (ms) | TPOT p99 (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.845 | 1.829 | 1.921 | 1.955 |
| 2 | 3.141 | 2.468 | 5.423 | 8.219 |
| 4 | 5.656 | 5.422 | 8.401 | 14.457 |
| 8 | 8.533 | 8.495 | 13.354 | 15.777 |
| 16 | 12.141 | 12.886 | 15.730 | 17.551 |
| 32 | 13.818 | 14.462 | 16.091 | 18.275 |

TPOT follows Frontier's Python definition:
`(request completion - first decode-token completion) / (output tokens - 1)`.
Larger batches increase aggregate throughput and improve queueing-dominated
TTFT here, but each decode iteration takes longer, so inter-token latency
increases.

The 16 GiB and 64 GiB grids were identical, so 16 GiB was already non-binding
for this workload. At batch cap 32 and offered concurrency 32, reducing KV
capacity from 16 GiB to 4 GiB caused 15 preemptions:

| KV/GPU | Requests/s | Total tokens/s | TTFT p90 (ms) | Preemptions |
| ---: | ---: | ---: | ---: | ---: |
| 16 GiB | 132.991 | 1,434,225 | 665.5 | 0 |
| 4 GiB | 117.555 | 1,267,759 | 791.3 | 15 |

That is an 11.6% throughput/TPS reduction and an 18.9% p90 TTFT increase.

A separate 1 GiB diagnostic made the collapse more pronounced. With batch cap
8, the completed 64-request cases ranged from 100 to 296 preemptions and
0.69–2.19 seconds p90 TTFT. Larger 1 GiB cases were stopped when detailed
trace retention, not simulated GPU memory, made host-memory usage excessive.

For a non-pressure reference case (offered concurrency 1, batch cap 8,
16 GiB), the production Python and C++ outputs matched across the complete
event, scheduler, batch-stage, batch, and request traces. A high-pressure
Python oracle run did not complete within the practical timeout, so the
pressure results are C++ measurements and are not claimed as cross-language
parity evidence.

## 64-GPU PDD 1:1 split result

The paired PDD experiment reused the exact 128 request shapes, arrival gaps,
arrival rates, and 4/16/64 GiB-per-GPU KV budgets from the co-location sweep.
At offered concurrency 16, batch cap 32, and 16 GiB KV per GPU:

| Topology | GPU layout | Requests/s | Total tokens/s | TTFT p90 (ms) | First-token p90 (ms) | TPOT p90 (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Co-location | TP8, PP4, DP2 = 64 | 132.991 | 1,434,225 | 498.5 | 498.5 | 16.091 |
| PDD PP-half | 2 × (TP8, PP2, DP2) = 64 | 105.037 | 1,132,765 | 751.9 | 754.3 | 1.953 |
| PDD TP-half | 2 × (TP4, PP4, DP2) = 64 | 140.316 | 1,513,229 | 471.0 | 473.5 | 2.196 |

The TP-half PDD split is the strongest of these three configurations: versus
co-location it improves request/token throughput by 5.5%, improves p90 TTFT by
5.5%, and reduces p90 TPOT by 86.4%. The PP-half split loses 21.0% throughput
and raises p90 TTFT by 50.8%, although its dedicated Decode cluster still gives
low TPOT.

PDD saturates at batch cap 8 in this workload. Prefill and Decode have
independent scheduler caps, and the short outputs leave the dedicated Decode
cluster lightly loaded. The mean KV transfer time is 1.197 ms per request.
Frontier's canonical TTFT ends at prefill completion, before PDD KV transfer;
the `first_token_latency_*` columns include transfer and the first Decode step.

The 4, 16, and 64 GiB PDD grids are identical and have no preemptions. A
separate 2 GiB diagnostic remains feasible; TP-half at batch cap 8 experiences
two preemptions. At 1.25 GiB, higher-load cases quiesce with incomplete state,
and 1 GiB cannot hold this workload's maximum 32,768-token prompt plus output
on the halved TP/PP layouts. These sub-2-GiB cases expose a low-capacity
progress limitation and are not included as valid performance measurements.

## Long-output workload results

The updated study uses 64 requests, offered concurrency
`1,2,4,8,16,32`, batch caps `1,4,8,16,32`, and 4/16 GiB KV per GPU.
Each profile covers 60 cases per topology and 11,520 paired request
simulations across the three topologies.

The realized deterministic workload distributions are:

| Profile | ISL mean / p50 / p90 | OSL mean / p50 / p90 / max |
| --- | --- | --- |
| Prefill-heavy | 10,598 / 7,459 / 24,645 | 344 / 300 / 629 / 1,024 |
| 8K/1K | 10,598 / 7,459 / 24,645 | 1,376 / 1,200 / 2,518 / 4,096 |

The configured log-normal medians are 8,192 ISL and 256 or 1,024 OSL;
the finite 64-request sample medians differ slightly.

At offered concurrency 32, batch cap 32, and 16 GiB KV per GPU, the
prefill-heavy profile gives:

| Topology | Requests/s | Total tokens/s | TTFT p90 (ms) | First-token p90 (ms) | TPOT p90 (ms) | Mean E2E (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Co-location | 15.109 | 165,329 | 282.4 | 282.4 | 4.187 | 1,274 |
| PDD PP-half | **17.408** | **190,490** | 37.5 | 43.3 | **2.448** | **773** |
| PDD TP-half | 16.556 | 181,165 | **33.2** | **39.7** | 2.881 | 880 |

The 8K/1K profile gives:

| Topology | Requests/s | Total tokens/s | TTFT p90 (ms) | First-token p90 (ms) | TPOT p90 (ms) | Mean E2E (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Co-location | 4.125 | 49,401 | 51.0 | 51.0 | 2.786 | 3,535 |
| PDD PP-half | **4.385** | **52,514** | **29.6** | **35.2** | **2.279** | **2,926** |
| PDD TP-half | 4.174 | 49,985 | 31.6 | 37.5 | 2.641 | 3,319 |

With meaningful Decode work, PP-half becomes the best overall 1:1 PDD split.
Its TP8 Decode cluster outweighs TP-half's PP4 Prefill advantage. At 8K/1K,
PP-half improves throughput by 6.3%, p90 TTFT by 41.9%, p90 TPOT by 18.2%,
and mean E2E by 17.2% versus co-location.

The 4 GiB regime is much harsher for PDD because only half of the 64 GPUs
belong to Decode, so the Decode cluster has half the aggregate KV capacity of
co-location at the same per-GPU budget. At 8K/1K, concurrency 32, and batch
cap 32:

| Topology | KV/GPU | Requests/s | TTFT p90 (ms) | First-token p90 (ms) | TPOT p90 (ms) | Preemptions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Co-location | 4 GiB | 3.631 | 834.9 | 834.9 | 6.241 | 160 |
| PDD PP-half | 4 GiB | 1.133 | 29.6 | 206.8 | 2.208 | 7 |
| PDD TP-half | 4 GiB | 0.766 | 31.6 | 766.8 | 14.925 | 10 |

Canonical PDD TTFT still ends before transfer and Decode admission, so it
remains low even when the Decode side is memory-starved. First-token latency,
TPOT, E2E, and throughput reveal the actual degradation.
