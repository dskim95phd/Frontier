# CLI User Guide

## Scope

This guide covers the public CLI surface for the `pre-release-v0.1` branch. The supported runtime architecture is `co-location`, where prefill and decode run in one monolithic cluster.

The parser still exposes older architecture fields for compatibility, but the release guard rejects these paths at startup:

- `pd-disaggregation`
- `pd-af-disaggregation`
- disaggregated cluster-specific CLI fields such as `--cluster_config_prefill_*`
- transfer config fields such as `--kv_cache_transfer_config_*` and `--m2n_transfer_config_*`

Use the examples first. They set the required flags, disable optional services, and write metrics to a predictable location.

## Environment

From the repository root:

```bash
conda env create -f environment.yml
conda activate frontier
python -m pip install -e ".[test]"

export PYTHONPATH=$PWD
export WANDB_DISABLED=true
export VIDUR_DISABLE_WANDB=1
```

If you already have the environment, update the editable install before running new code:

```bash
conda activate frontier
python -m pip install -e ".[test]"
export PYTHONPATH=$PWD
```

The co-location example suite defaults to `--cc_backend_config_type analytical` for one-click smoke runs and does not require the optional `collective_sim` submodule. Build `collective_sim` only when you explicitly select it:

```bash
git submodule update --init --recursive frontier/cc_backend/backends/collective-sim
cd frontier/cc_backend/backends/collective-sim/sim
make -j"$(nproc)"
```

Use `--cc_backend_config_type analytical` for the release co-location smoke examples. Use `--cc_backend_config_type astra_sim_analytical` when you intentionally want the ASTRA-Sim-inspired lightweight topology model.

## Recommended Entry Points

Run the release examples before writing a custom command:

```bash
# Run all five offline cases and all five online cases.
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

The baseline dense, MoE, and Thinking Mode scripts enable these runtime settings by default:

- `--decode_cuda_graph_mode full_decode_only`
- `--vllm_v1_scheduler_config_enable_chunked_prefill`
- CSV/JSON metrics output
- dummy execution-time predictor mode for fast smoke tests

The advanced MoE scripts are available in both `offline/` and `online/`:

- `moe_spec_dec.sh` / `moe_spec_dec_online.sh`: Speculative Decoding / MTP. They use `decode_cuda_graph_mode=none` because Speculative Decoding and decode CUDA Graph modeling currently conflict unless a diagnostic opt-in is used.
- `moe_prefix_caching.sh` / `moe_prefix_caching_online.sh`: Prefix Caching. They replay `examples/fixtures/prefix_cache_shared_session_trace.csv` so repeated prompt blocks produce cache-hit behavior.

## Running `frontier.main` Directly

CLI entry point:

```bash
python -m frontier.main --help
```

A small dense co-location command:

```bash
python -m frontier.main \
  --simulation_mode offline \
  --sys_arch co-location \
  --cluster_config_num_replicas 1 \
  --replica_config_model_name meta-llama/Llama-2-7b-hf \
  --replica_config_attn_tensor_parallel_size 2 \
  --replica_config_num_pipeline_stages 1 \
  --replica_config_attn_data_parallel_size 1 \
  --cc_backend_config_type astra_sim_analytical \
  --replica_scheduler_config_type vllm_v1 \
  --decode_cuda_graph_mode full_decode_only \
  --vllm_v1_scheduler_config_enable_chunked_prefill \
  --request_generator_config_type synthetic \
  --synthetic_request_generator_config_num_requests 4 \
  --length_generator_config_type fixed \
  --fixed_request_length_generator_config_prefill_tokens 128 \
  --fixed_request_length_generator_config_decode_tokens 32 \
  --interval_generator_config_type poisson \
  --poisson_request_interval_generator_config_qps 1.0 \
  --random_forrest_execution_time_predictor_config_enable_dummy_mode \
  --metrics_config_output_dir outputs/examples/co-location \
  --metrics_config_run_id cli_dense_smoke \
  --metrics_config_write_metrics \
  --metrics_config_store_request_metrics \
  --no-metrics_config_store_plots \
  --no-metrics_config_enable_chrome_trace \
  --no-metrics_config_write_json_trace
```

Dummy predictor mode is useful for smoke tests. For latency studies, disable dummy mode and point the predictor at profiling CSVs under `data/profiling/compute/<device>/<model>/`.

## Core CLI Groups

### Simulation mode and architecture

| Option | Use |
|--------|-----|
| `--simulation_mode offline` | Generate or replay requests inside the simulator. |
| `--simulation_mode online` | Run online mode where supported by the selected scheduler path. |
| `--sys_arch co-location` | Release-supported architecture. |

### Model and parallelism

| Option | Use |
|--------|-----|
| `--replica_config_model_name` | Model name used to resolve model config and output taxonomy. |
| `--cluster_config_num_replicas` | Number of monolithic replicas. |
| `--replica_config_attn_tensor_parallel_size` | Attention tensor parallel size. |
| `--replica_config_attn_data_parallel_size` | Attention data parallel size. |
| `--replica_config_num_pipeline_stages` | Pipeline parallel stages. |
| `--replica_config_moe_tensor_parallel_size` | MoE tensor parallel size. |
| `--replica_config_moe_expert_parallel_size` | MoE expert parallel size. |
| `--replica_config_total_expert_num` | Total expert count for MoE models. |
| `--replica_config_router_topk` | Number of routed experts per token. |
| `--replica_config_moe_routing_mode` | MoE routing mode: `simulation`, `uniform_legacy`, or `uniform_random`. |

### Runtime optimization

| Option | Use |
|--------|-----|
| `--decode_cuda_graph_mode full_decode_only` | Model decode CUDA Graph behavior for decode-only batches. |
| `--decode_cuda_graph_mode none` | Disable decode CUDA Graph modeling. Required by the Speculative Decoding example. |
| `--vllm_v1_scheduler_config_enable_chunked_prefill` | Enable Chunked Prefill on the `vllm_v1` scheduler. |
| `--vllm_v1_scheduler_config_enable_prefix_caching` | Enable Prefix Caching on supported scheduler paths. |
| `--vllm_v1_scheduler_config_prefix_caching_key_mode block_hash` | Use explicit `block_hash_ids` from the trace. This is the default. |
| `--vllm_v1_scheduler_config_prefix_caching_key_mode session` | Derive replica-local prefix block keys from `session_id` and block position; trace ISLs are newly appended input tokens. |
| `--speculative_decoding_config_enabled` | Enable Speculative Decoding / MTP modeling. |

### Workload generation

Synthetic fixed-length workload:

```bash
--request_generator_config_type synthetic
--synthetic_request_generator_config_num_requests 16
--length_generator_config_type fixed
--fixed_request_length_generator_config_prefill_tokens 512
--fixed_request_length_generator_config_decode_tokens 128
--interval_generator_config_type poisson
--poisson_request_interval_generator_config_qps 1.0
```

Trace replay workload:

```bash
--request_generator_config_type trace_replay
--trace_request_generator_config_trace_file examples/fixtures/prefix_cache_shared_session_trace.csv
--trace_request_generator_config_max_tokens 128
```

Use trace replay when you need shared-prefix or known-arrival behavior.

Session-scoped prefix caching accepts a length-only, append-only multi-turn
trace:

```csv
arrived_at,num_prefill_tokens,num_decode_tokens,session_id
0.0,32,16,7
10.0,8,8,7
```

In session mode, `num_prefill_tokens` is the number of new input tokens added
by that turn. Frontier materializes the effective full-prompt length before
creating each request:

```text
effective_ISL = prior_session_context + new_ISL
next_session_context = effective_ISL + OSL
```

The example therefore materializes to effective prompt lengths 32 and 56.
Entries in `thinking_round_plans_json` use the same incremental ISL rule and
are accumulated round by round. Each round also uses
`trace_request_generator_config_prefill_scale_factor` and
`trace_request_generator_config_decode_scale_factor` with the same
`int(raw * factor)` conversion as the top-level CSV values. Scaling happens
before the final round is compared with the row and before contexts are
accumulated.

Session traces are strict: raw and scaled arrivals must be finite and
nonnegative; raw and scaled token counts must be finite and positive; and
integerized token counts must remain at least one. Invalid values fail with a
row/field-specific `ValueError` and are not clipped to one. The expanded
context must remain within `trace_request_generator_config_max_tokens`.
Context truncation and branching under one session ID are not supported.
Request and batch metrics continue to report the materialized effective prompt
length, not the raw incremental ISL from the CSV.

KV cache state is local to each `(replica_id, dp_id)` target. When
`num_replicas * data_parallel_size > 1`, Prefix Caching therefore requires a
sticky cluster scheduler even if there is only one physical replica.
In this release, `sticky_lor` is supported only for co-location /
`MONOLITHIC`. Sequential PDD Prefix Caching must use
`--cluster_scheduler_config_type sticky_round_robin`; `sticky_lor` is rejected
during configuration validation.

Run the public session fixture in co-location mode:

```bash
TRACE_FILE=examples/fixtures/session_prefix_multi_turn_trace.csv \
PREFIX_CACHING_KEY_MODE=session \
EXPECTED_TRACE_REQUESTS=4 \
bash examples/architecture/co-location/online/moe_prefix_caching_online.sh
```

For offline mode, the prefix-cache recipe automatically enables
`--offline_use_generated_request_arrivals` when
`PREFIX_CACHING_KEY_MODE=session`. This prevents all turns from being forced to
arrive at `t=0`.

In co-location, full decode blocks may remain available for a later turn. In
PDD, the prefill cache does not automatically receive decode output KV, so the
first follow-up turn can only hit the prefix previously processed by the
prefill cluster.

Prefill-side CPU KV-cache offloading extends that PDD behavior without
changing the decode boundary:

```bash
bash examples/architecture/pdd/offline/cpu_kv_offloading.sh
bash examples/architecture/pdd/online/cpu_kv_offloading_online.sh
```

The feature requires sequential `pd-disaggregation`, `vllm_v1`,
`sticky_round_robin`, `enable_prefix_caching`, and
`prefix_caching_key_mode=session`; Thinking Mode is rejected for this MVP.
Full blocks produced by prefill are copied
to the local CPU tier with direction-specific analytical bandwidth and
latency. Decode-produced KV is not copied back; those tokens are recomputed
by the next turn's prefill and only then can join a later CPU snapshot.

`capacity_bytes` is the aggregate capacity of one prefill
`(replica_id, dp_id)` target, not a per-GPU value. Its block accounting sums
the physical KV shards of all attention TP workers in that target. The
configured read/write bandwidths are likewise aggregate effective bandwidths
for the target. `--vllm_v1_scheduler_config_kv_cache_dtype` controls the
canonical GPU/CPU KV precision; `auto` follows the model `torch_dtype`.

Relevant flat CLI fields are
`--cpu_kv_cache_config_enable`,
`--cpu_kv_cache_config_capacity_bytes`,
`--cpu_kv_cache_config_static_slice_per_gpu`,
`--cpu_kv_cache_config_capacity_bytes_per_gpu`,
`--cpu_kv_cache_config_dram_bandwidth_gbps_per_gpu`,
`--cpu_kv_cache_config_c2c_bandwidth_gbps_per_gpu`,
`--cpu_kv_cache_config_write_bandwidth_gbps`,
`--cpu_kv_cache_config_write_latency_ms`,
`--cpu_kv_cache_config_read_bandwidth_gbps`,
`--cpu_kv_cache_config_read_latency_ms`, and
`--cpu_kv_cache_config_capacity_pressure_policy`. KV precision is selected
with `--vllm_v1_scheduler_config_kv_cache_dtype` using `auto`, `fp32`,
`fp16`, `bf16`, `fp8`, `int8`, `fp4`, or `int4`.

With `static_slice_per_gpu`, the number of physical slices for one target is
`attention TP × pipeline parallel size`. The effective per-direction transfer
bandwidth of each GPU slice is
`min(dram_bandwidth_gbps_per_gpu, c2c_bandwidth_gbps_per_gpu)`.

For `network_device=vera_rubin_nvl72_domain`, rack-local placement can be
configured with `--cluster_config_num_racks` in co-location mode or
`--cluster_config_prefill_cluster_num_racks` and
`--cluster_config_decode_cluster_num_racks` in sequential PDD. Each rack is
owned by one cluster, replicas never span racks, and unused rack GPUs remain
idle. Omitting a rack count selects the minimum number required to pack all
whole replicas.

Request metrics distinguish GPU hits, CPU queries/hits/restores, and
D2H/H2D byte and time costs. `system_metrics.json` includes a separate
`cpu_kv_cache_statistics` object for capacity, occupancy, eviction,
offload/restore, and CPU hit information.

CPU restore uses the same deferred target-allocation abstraction as the PDD
prefill-to-decode transfer. H2D completion places the transferred block
contents in a logical staging state; it does not reserve scheduler-managed
prefill GPU KV pages. The prefill scheduler revalidates current GPU hits and
materializes the usable staged prefix only when the request is admitted.

This abstraction preserves normal waiting-request scheduling semantics, but it
does not model a finite receiver staging buffer, destination-GPU transfer
backpressure, or an additional materialization latency at admission. An H2D
operation can therefore complete while the prefill GPU KV pool is full, and
cache churn can make some transferred blocks unusable before admission.
`restore_blocks` and restore-byte metrics describe transferred traffic;
request cached-token and CPU-hit metrics describe the prefix actually consumed
at admission.

When cluster event logging is enabled, scheduler diagnostics report
`prefix_cache_admissions`, `prefix_cache_queries`, and `prefix_cache_hits` with
`prefix_cache_metric_semantics=successful_admission_block_level`. These
counters are committed only after allocation succeeds, so repeated lookup
attempts caused by token-budget or allocation checks do not inflate them.
For Thinking Mode, request CSV and `prefix_cache_statistics` accumulate cached
tokens, query blocks, and hit blocks across hidden and final rounds.
Preemption/re-admission inside one round restores runtime state without
double-counting that round.

### Execution-time predictor

| Option | Use |
|--------|-----|
| `--random_forrest_execution_time_predictor_config_enable_dummy_mode` | Use fixed dummy execution time. Good for smoke tests only. |
| `--no-random_forrest_execution_time_predictor_config_enable_dummy_mode` | Use profiling-backed ML predictors. |
| `--random_forrest_execution_time_predictor_config_linear_op_input_file` | Path to `linear_op.csv`. |
| `--random_forrest_execution_time_predictor_config_atten_input_file` | Path to `attention.csv`. |
| `--random_forrest_execution_time_predictor_config_moe_input_file` | Path to `moe.csv`. Required for MoE non-dummy runs. |
| `--random_forrest_execution_time_predictor_config_skip_cpu_overhead_modeling` | Skip CPU overhead predictor modeling for lightweight CSV smoke runs. |

When dummy mode is disabled, Frontier checks the predictor cache. If a needed model is missing, it trains from the configured CSVs and writes the trained estimator into the cache directory.

### Metrics output

| Option | Use |
|--------|-----|
| `--metrics_config_output_dir` | Metrics output root. The simulator appends `<model>/<workload>/<run_id>/`. |
| `--metrics_config_run_id` | Stable run id. Use this for reproducible output paths. |
| `--metrics_config_write_metrics` | Write metrics artifacts. |
| `--metrics_config_store_request_metrics` | Write request-level CSV metrics. |
| `--metrics_config_store_batch_metrics` | Write batch-level CSV metrics. |
| `--metrics_config_store_token_completion_metrics` | Write token completion metrics. |
| `--metrics_config_store_utilization_metrics` | Write utilization metrics. |
| `--no-metrics_config_store_plots` | Skip optional Plotly plot export. |
| `--no-metrics_config_enable_chrome_trace` | Skip Chrome trace output. |
| `--no-metrics_config_write_json_trace` | Skip JSON event trace output. |

Output path format:

```text
<metrics_config_output_dir>/<model_type>/<offline_batch|online_serving>/<run_id>/
```

Common files include:

- `config.json`
- `system_metrics.json`
- `request_metrics.csv` when request metrics are enabled
- `<cluster>_batch_metrics.csv` when batch metrics are enabled
- utilization CSVs when utilization metrics are enabled

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Release guard exits for `pd-disaggregation` or `pd-af-disaggregation`. | Disaggregated architectures are not included in `pre-release-v0.1`. | Use `--sys_arch co-location`. |
| `htsim_ndp` is missing after selecting `--cc_backend_config_type collective_sim`. | The optional `collective_sim` submodule binary has not been built. | Build `frontier/cc_backend/backends/collective-sim/sim`, or use the default co-location example `analytical` backend. |
| W&B tries to initialize. | Environment variables are not set. | Set `WANDB_DISABLED=true` and `VIDUR_DISABLE_WANDB=1`. |
| Non-dummy run fails on a missing CSV or schema mismatch. | Predictor training needs matching profiling data. | Use the profiling guide and keep CSVs under `data/profiling/compute/<device>/<model>/`. |
| Plot export warns about `kaleido`. | PNG export is optional. | Keep `--no-metrics_config_store_plots` for smoke runs, or install `kaleido` if PNGs are needed. |
