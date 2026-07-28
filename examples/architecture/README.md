# Architecture Examples

## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-05 | Added profiling-independent dummy smoke matrix runner documentation. |
| 2026-06-22 | Removed legacy split-decode terminology from the public PDD surface. |
| 2026-06-14 | Added PDD pd-disaggregation script list, configuration contract, and validation criteria for local PR preparation. |

This directory contains one-click architecture entrypoints for Frontier's release-supported runtime layouts.

## Release Scope

`pre-release-v0.2` foregrounds **PDD / `pd-disaggregation`** examples. Prefill runs in the `PREFILL` cluster, decode runs in the unified `DECODE` cluster, and KV cache is transferred between them. The public PDD example path uses the sequential simulator mode through `--no-enable_parallel_clusters`.

`co-location` examples remain available as baseline comparison recipes and v0.1-compatible architecture references. Additional disaggregated research prototypes outside the PDD path are not exposed as release examples.

## Scripts

| Path | Scenario | Notes |
|------|----------|-------|
| `run_dummy_smoke_matrix.sh` | Profiling-independent dummy smoke matrix | Runs dense/MoE across co-location/PDD and offline/online; does not consume profiling CSV datasets |
| `co-location/run_all.sh` | Full co-location suite | Runs all five offline cases and all five online cases; pass extra Frontier CLI flags after `--` |
| `co-location/offline/dense_model_basic.sh` | Offline dense co-location baseline | Analytical backend by default, dummy execution time, `decode_cuda_graph_mode=full_decode_only`, Chunked Prefill, CSV/JSON metrics |
| `co-location/offline/vera_rubin_nvl72_analytical.sh` | Rubin NVL72 analytical co-location | Configurable cluster-exclusive racks, whole-replica packing, and profile-free roofline timing |
| `co-location/offline/moe_model_basic.sh` | Offline MoE co-location baseline | Analytical backend by default, dummy execution time, shared-domain MoE invariant, Chunked Prefill, CSV/JSON metrics |
| `co-location/offline/thinking_mode_basic.sh` | Offline Thinking Mode v1 co-location | Analytical backend; one hidden round plus one final round; CSV/JSON metrics |
| `co-location/offline/moe_spec_dec.sh` | Offline MoE Speculative Decoding / MTP | Speculative Decoding / MTP enabled; uses `decode_cuda_graph_mode=none` to avoid the current conflict |
| `co-location/offline/moe_prefix_caching.sh` | Offline MoE Prefix Caching | Prefix Caching enabled with `examples/fixtures/prefix_cache_shared_session_trace.csv` |
| `co-location/online/dense_model_basic_online.sh` | Online dense co-location baseline | Mirrors dense offline settings with analytical backend and `--simulation_mode online` |
| `co-location/online/moe_model_basic_online.sh` | Online MoE co-location baseline | Mirrors MoE offline settings with analytical backend and `--simulation_mode online` |
| `co-location/online/thinking_mode_basic_online.sh` | Online Thinking Mode v1 co-location | Mirrors Thinking Mode offline settings with `--simulation_mode online` |
| `co-location/online/moe_spec_dec_online.sh` | Online MoE Speculative Decoding / MTP | Mirrors Speculative Decoding offline settings with `--simulation_mode online` |
| `co-location/online/moe_prefix_caching_online.sh` | Online MoE Prefix Caching | Replays the same prefix-cache fixture with `--simulation_mode online` |
| `pdd/run_all.sh` | Full PDD suite | Runs all six offline PDD cases and all six online PDD cases; pass extra Frontier CLI flags after `--` |
| `pdd/offline/dense_model_basic.sh` | Offline dense PDD baseline | Sequential `pd-disaggregation`, analytical backend, dummy execution time, Chunked Prefill, CSV/JSON metrics |
| `pdd/offline/moe_model_basic.sh` | Offline MoE PDD baseline | Sequential `pd-disaggregation`, reference-runnable shared-domain MoE topology, Chunked Prefill, CSV/JSON metrics |
| `pdd/offline/thinking_mode_basic.sh` | Offline Thinking Mode v1 PDD | Thinking Mode with two KV transfer handoffs for the one-request smoke configuration |
| `pdd/offline/moe_spec_dec.sh` | Offline MoE PDD Speculative Decoding / MTP | Speculative Decoding enabled; Prefix Caching intentionally disabled; `DECODE_CUDA_GRAPH_MODE=none` |
| `pdd/offline/moe_prefix_caching.sh` | Offline MoE PDD Prefix Caching | Sticky scheduler with `examples/fixtures/prefix_cache_shared_session_trace.csv` |
| `pdd/offline/cpu_kv_offloading.sh` | Offline prefill CPU KV offloading | Session-affine PDD with a finite five-block prefill GPU cache and analytical D2H/H2D copies |
| `pdd/offline/vera_rubin_nvl72_cpu_offload_analytical.sh` | Rubin NVL72 analytical PDD | Cluster-exclusive prefill/decode racks with configurable rack counts, profile-free roofline timing, and static per-GPU Vera CPU slices |
| `pdd/online/dense_model_basic_online.sh` | Online dense PDD baseline | Mirrors dense offline settings with `--simulation_mode online` |
| `pdd/online/moe_model_basic_online.sh` | Online MoE PDD baseline | Mirrors MoE offline settings with `--simulation_mode online` |
| `pdd/online/thinking_mode_basic_online.sh` | Online Thinking Mode v1 PDD | Mirrors Thinking Mode offline settings with `--simulation_mode online` |
| `pdd/online/moe_spec_dec_online.sh` | Online MoE PDD Speculative Decoding / MTP | Mirrors Speculative Decoding offline settings with `--simulation_mode online` |
| `pdd/online/moe_prefix_caching_online.sh` | Online MoE PDD Prefix Caching | Replays the same prefix-cache fixture with `--simulation_mode online` |
| `pdd/online/cpu_kv_offloading_online.sh` | Online prefill CPU KV offloading | Online counterpart using `examples/fixtures/cpu_kv_offload_session_trace.csv` |

All four Prefix Caching recipes accept
`PREFIX_CACHING_KEY_MODE=session` together with
`examples/fixtures/session_prefix_multi_turn_trace.csv`. Session mode derives
cache identity from `session_id` and block position, so the trace does not need
`block_hash_ids`:

```bash
TRACE_FILE=examples/fixtures/session_prefix_multi_turn_trace.csv \
PREFIX_CACHING_KEY_MODE=session \
EXPECTED_TRACE_REQUESTS=4 \
bash examples/architecture/co-location/online/moe_prefix_caching_online.sh
```

The ISL column contains only the new input tokens added by each turn. Frontier
accumulates those values with the prior session context to materialize the
effective full prompt. `thinking_round_plans_json` uses the same incremental
rule, scale factors, and integerization as the top-level CSV values for every
round. Session arrivals and token lengths are validated as finite,
nonnegative/positive values and are never silently clipped. Expanded context
must remain within `max_tokens`. Context truncation and conversation branching
under one session ID are not supported. Final request/system cache metrics sum
hidden and final Thinking-round lookups without counting same-round
preemption twice. Co-location may reuse full decode blocks from the prior
turn; PDD only reuses blocks previously processed by its prefill cluster
because decode KV is not returned automatically. Prefix Caching recipes use
`sticky_round_robin` because every `(replica_id, dp_id)` owns an independent
cache; affinity is required whenever replicas multiplied by DP lanes exceeds
one. `sticky_lor` is supported for co-location only in this release;
sequential PDD Prefix Caching rejects it during validation and requires
`sticky_round_robin`.

### Prefill-side CPU KV offloading

The CPU examples add a DRAM tier local to every prefill
`(replica_id, dp_id)` cache target. After prefill finishes, full reusable
blocks are copied to CPU while the normal prefill-to-decode transfer proceeds
independently. Decode can begin as soon as its transfer finishes, but the
prefill GPU allocation remains pinned until both exports are terminal.

Only prefill-created KV is copied. If turn 1 has prompt `A` and produces
decode output `B`, turn 2 restores `A` from prefill GPU or CPU and recomputes
`B` plus the new input. Once that expanded prompt passes through prefill, its
full blocks become eligible for the next CPU snapshot.

The main controls are:

- `--cpu_kv_cache_config_enable`
- `--cpu_kv_cache_config_capacity_bytes`
- `--cpu_kv_cache_config_static_slice_per_gpu` derives the target capacity and
  bandwidth from `attention TP × pipeline parallel size`; use it with
  `capacity_bytes_per_gpu`, `dram_bandwidth_gbps_per_gpu`, and
  `c2c_bandwidth_gbps_per_gpu`. The slower of DRAM and C2C determines each
  GPU slice's effective per-direction transfer bandwidth.
- `--cpu_kv_cache_config_write_bandwidth_gbps` and
  `--cpu_kv_cache_config_write_latency_ms`
- `--cpu_kv_cache_config_read_bandwidth_gbps` and
  `--cpu_kv_cache_config_read_latency_ms`
- `--cpu_kv_cache_config_capacity_pressure_policy prefix_fit|skip_offload`
- `--vllm_v1_scheduler_config_kv_cache_dtype` with `auto`, `fp32`, `fp16`,
  `bf16`, `fp8`, `int8`, `fp4`, or `int4`

The capacity and direction-specific bandwidth values apply to one aggregate
prefill `(replica_id, dp_id)` CPU cache target. All attention TP workers in
that DP lane contribute their physical KV shard bytes to each CPU block. A
pipeline-parallel target owns physical GPU slices in every stage, even though
the aggregate CPU block already represents the full model and therefore is
not multiplied by PP a second time.

The MVP requires sequential `pd-disaggregation`, `vllm_v1`, session prefix
caching, and `sticky_round_robin`; Thinking Mode is not supported in this
MVP because its overlapping rounds require epoch-scoped export ownership.
CPU eviction chooses the least-recently used unpinned session and removes the
required highest-index suffix in one bulk operation, preserving the longest
useful prefix.

## PDD Configuration Contract

All PDD scripts use these release-supported defaults unless overridden from the shell:

- `--sys_arch pd-disaggregation`
- `--no-enable_parallel_clusters`
- explicit `PREFILL` and unified `DECODE` cluster settings
- `--cc_backend_config_type analytical`
- dummy execution-time prediction enabled by default
- CSV/JSON metrics enabled by default through `--metrics_config_write_metrics` and `--metrics_config_store_request_metrics`
- plots, Chrome trace, and JSON event trace disabled for lightweight one-click artifacts

MoE PDD scripts also enforce that each role's attention and MoE parallel domains match before launching Frontier. This fail-fast check prevents known non-runnable MoE topology combinations from entering the simulator.

## Thinking Mode v1

The Thinking Mode examples use:

- `--enable_thinking_mode`
- `--thinking_depth 2`
- one explicit hidden round via `--thinking_round_prefill_tokens` and `--thinking_round_decode_tokens`
- `--tool_call_latency 0.001`
- explicit `vllm_v1` scheduler settings
- `--cc_backend_config_type analytical` so the one-click smoke run works on a minimal single-replica layout
- CSV/JSON metrics enabled by default, with plots, Chrome trace, and JSON event trace disabled for lightweight artifacts

Under PDD, one user request can produce multiple prefill-to-decode KV handoffs. The default Thinking Mode smoke case completes one request and records two KV transfers.

## Recommended Start Order

```bash
# Profiling-independent smoke matrix: dense/MoE across co-location/PDD and offline/online.
# This does not consume profiling CSV datasets because it forces dummy execution-time prediction.
bash examples/architecture/run_dummy_smoke_matrix.sh

# Full PDD suite for pre-release-v0.2.
bash examples/architecture/pdd/run_all.sh

# PDD offline cases.
bash examples/architecture/pdd/offline/dense_model_basic.sh
bash examples/architecture/pdd/offline/moe_model_basic.sh
bash examples/architecture/pdd/offline/thinking_mode_basic.sh
bash examples/architecture/pdd/offline/moe_spec_dec.sh
bash examples/architecture/pdd/offline/moe_prefix_caching.sh
bash examples/architecture/pdd/offline/cpu_kv_offloading.sh

# PDD online cases.
bash examples/architecture/pdd/online/dense_model_basic_online.sh
bash examples/architecture/pdd/online/moe_model_basic_online.sh
bash examples/architecture/pdd/online/thinking_mode_basic_online.sh
bash examples/architecture/pdd/online/moe_spec_dec_online.sh
bash examples/architecture/pdd/online/moe_prefix_caching_online.sh
bash examples/architecture/pdd/online/cpu_kv_offloading_online.sh

# Full co-location comparison suite.
bash examples/architecture/co-location/run_all.sh

# Offline cases.
bash examples/architecture/co-location/offline/dense_model_basic.sh
bash examples/architecture/co-location/offline/moe_model_basic.sh
bash examples/architecture/co-location/offline/thinking_mode_basic.sh
bash examples/architecture/co-location/offline/moe_spec_dec.sh
bash examples/architecture/co-location/offline/moe_prefix_caching.sh

# Online cases.
bash examples/architecture/co-location/online/dense_model_basic_online.sh
bash examples/architecture/co-location/online/moe_model_basic_online.sh
bash examples/architecture/co-location/online/thinking_mode_basic_online.sh
bash examples/architecture/co-location/online/moe_spec_dec_online.sh
bash examples/architecture/co-location/online/moe_prefix_caching_online.sh
```

Use the dense baseline scripts first, then use the Thinking Mode, Speculative Decoding / MTP, and Prefix Caching recipes as advanced cases.

## Cross-validation Criteria

For each offline/online pair:

1. Confirm the script exits with code `0`.
2. Confirm `request_metrics.csv` and `system_metrics.json` exist in the metrics output directory.
3. Record expected request count, actual request rows, completed request rows, total input tokens, total output tokens, mean TTFT, mean latency, and request throughput when present.
4. Confirm offline outputs include the `offline_batch` taxonomy segment and online outputs include `online_serving`.
5. Treat latency differences as expected when online mode preserves request arrival times; investigate only if counts, token totals, output files, or finite numeric metrics diverge unexpectedly.

For every PDD script, the release gate should additionally record:

1. The script exits with code `0`.
2. `request_metrics.csv` and `system_metrics.json` exist in the metrics output directory.
3. Request row count, `total_requests`, and `completed_requests` match the expected case size.
4. KV transfer count, total KV bytes, and KV transfer time are present and positive.
5. Request-level `ttft`, `tpot`, `request_e2e_time`, and `transfer_kv_cache` are finite and positive.
