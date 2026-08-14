# Frontier: LLM Inference Simulator

Frontier is a modular **discrete-event simulator (DES)** for large language
model inference. It models a serving system as clusters and replicas
processing requests over time, and reports request latency, throughput,
scheduling behavior, and memory/KV-cache pressure.

This document is the authoritative entry point for the repository.

## Which implementation to use

The repository holds two simulators. **The C++ core under `cpp/` is the
primary implementation and the one to use for new work.**

| | C++ core (`cpp/`) | Python simulator (`frontier/`) |
| --- | --- | --- |
| Status | Active | Maintenance; retained for reference and profiling |
| Entry point | `frontier_sim` binary | `python -m frontier.main` |
| Input | Strict JSON config + CSV workload | Flattened dataclass CLI |
| Determinism | Deterministic, contract-tested | Best-effort |

The C++ core began as a port of the Python simulator. That port is complete,
and the C++ surface has since moved ahead of it. Features that exist only in
C++ include Kimi K3 / KDA support, MLA with decode context parallelism,
tiered CPU KV-cache offload, MXFP4/MXFP8 operator precisions, automatic KV
block sizing from physical HBM, the collapsed pipeline-parallel event
calendar, and the pipeline-exclusive topology contract.

Because the port goal is met, **do not add code to the C++ core whose only
purpose is to reproduce Python behavior.** Match Python only where its
behavior is itself the modeling contract being simulated — for example vLLM
V1 scheduling semantics — and say so in a comment when you do.

Features still exclusive to the Python path are speculative decoding / MTP,
sklearn-trained execution-time predictors, GPU profiling, topology-aware
communication backends (`collective_sim`, `astra_sim_analytical`), the
SGLang-style replica scheduler, and Thinking Mode.

## C++ core

### Supported surface

- Architectures: `co-location` (single monolithic cluster) and sequential
  `pd-disaggregation` (separate PREFILL/DECODE clusters with KV transfer).
- Modes: `offline` and `online` for both architectures.
- Parallelism: TP, PP, DP, MoE TP/EP, and decode context parallelism (DCP).
- Scheduling: hierarchical global → cluster → replica → pipeline stage;
  FCFS vLLM V1 continuous batching with chunked prefill, KV-block
  accounting, and recompute preemption.
- Execution models: fixed per-stage latencies, or an analytical roofline
  model with `rubin`/`gb300`/`custom` device ceilings and per-operator
  precisions.
- Caching: session prefix caching, and finite CPU KV-cache offload/restore
  for sequential PDD.

Not implemented in C++: `pd-af-disaggregation`, parallel PDD clusters
(`enable_parallel_clusters` must be `false`), and block-hash prefix caching.

### Build and test

The build is CMake + Ninja, C++17, GCC 11+ or Clang 14+. `nlohmann/json`
3.11.3 is the only dependency.

```bash
cmake -S cpp -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

From Windows, build inside WSL against the Windows worktree and keep the
build directory WSL-native:

```bash
cd /mnt/c/Users/<user>/path/to/Frontier-cxx-port
export PATH="$HOME/frontier-tools/bin:$PATH"
cmake -S cpp -B "$HOME/frontier-build/cxx-port" -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON
cmake --build "$HOME/frontier-build/cxx-port"
ctest --test-dir "$HOME/frontier-build/cxx-port" --output-on-failure
```

Tests are plain executables registered with CTest; there is no external test
framework. Stress and matrix tests carry the `stress` label and run serially.

### Run

```bash
build/frontier_sim \
  --config cpp/examples/configs/03_sequential_pdd.json \
  --workload cpp/examples/workloads/00_tiny.csv \
  --output-dir outputs/my-run \
  --output-mode requests
```

Without `--output-dir`, the full deterministic JSON trace goes to stdout.
See [`cpp/README.md`](cpp/README.md) for the complete CLI, configuration,
workload, and output contracts, and `cpp/examples/` for runnable recipes.

### Architecture

Four scheduling layers mirror real serving systems:

1. **Global scheduler** receives request arrivals and routes them to a
   cluster.
2. **Cluster scheduler** selects a `(replica_id, dp_id)` target.
   Implementations: `round_robin`, `sticky_round_robin`, `cache_aware`,
   `kv_aware`, `vllm_queue_aware`.
3. **Replica scheduler** owns batching policy, KV-block allocation,
   preemption, prefix-cache admission, and CPU offload. Implementation:
   `vllm_v1`.
4. **Replica stage scheduler** owns one pipeline stage's execution calendar
   and calls the execution-time predictor.

State changes are driven exclusively by typed events. Event payloads are a
`std::variant`; ID types are distinct phantom types (`RequestId`, `BatchId`,
…) with an invalid `-1` sentinel, as is `SimTime`.

Key source directories under `cpp/frontier/`:

| Directory | Contents |
| --- | --- |
| `config/` | JSON parsing, normalization, serialization, model assets |
| `core/` | `Event`, `EventQueue`, strong ID types |
| `entities/` | `Request`, `Batch`, `BatchStage`, `Cluster`, `Replica` |
| `events/` | One handler per event type |
| `scheduler/` | The four scheduling layers |
| `execution_time_predictor/` | Fixed and analytical roofline models |
| `kv_cache/` | GPU and CPU KV-cache managers |
| `metrics/` | Metrics store and output contract |
| `simulator/` | Event loop, entity arena |

## Repository layout

| Path | Contents |
| --- | --- |
| `cpp/` | C++ core: source, tests, examples, experiments |
| `frontier/` | Python simulator package |
| `data/` | Model config assets and profiling datasets |
| `docs/design/` | Design decision records for implemented C++ features |
| `docs/roadmap/` | Historical porting plans; superseded by the code |
| `docs/cli/`, `docs/profiling/`, `docs/training/` | Python-path guides |
| `examples/` | Python-path shell recipes |
| `tests/` | Python-path test scripts |

## Development gates

Any new feature, refactor, or module adjustment must pass these before
delivery:

- Keep the design modular and integrated with existing boundaries. No
  hard-coded logic and no locally reinvented duplicates of existing helpers.
- Preserve existing behavior: unchanged features keep their results and key
  intermediate outputs.
- Preserve simulator fidelity: refactors must not change numeric results
  unless an explicitly approved fidelity fix explains the difference. When a
  change does alter results, say so in the commit message and name the
  configurations affected.
- Validate with a case matrix covering dense/MoE, offline/online, varied
  request lengths and counts, varied arrival rates, feature toggles, and
  model configs.
- Run the full CTest suite. It must be green before delivery.

## Contributing

- Events drive state changes; keep schedulers layered.
- Prefer incremental, backward-compatible changes. New configuration fields
  should be optional and normalize to the existing behavior when absent.
- Add a runnable test scenario with each feature, and update `cpp/README.md`
  when a contract changes.

## Python simulator (reference)

The Python simulator remains in `frontier/` for the capabilities the C++
core does not cover, and as the source of the profiling and predictor
training workflows.

```bash
conda env create -f environment.yml
conda activate frontier
python -m pip install -e ".[test]"
export PYTHONPATH=$PWD

bash examples/architecture/co-location/offline/dense_model_basic.sh
```

Configuration is nested dataclasses (`frontier/config/config.py`) flattened
into a large `--<field>` CLI surface, so the reliable workflow is to start
from a script in `examples/` and change only what you need. The example
suites default to `--cc_backend_config_type analytical` and do not require
the optional `collective_sim` submodule; sequential PDD additionally
requires `--no-enable_parallel_clusters`. `pd-af-disaggregation` fails fast
with the guarded disaggregation error.

Metrics land under `outputs/metrics/<model_type>/<workload_type>/<run_id>/`
as `request_metrics.csv` and `system_metrics.json`. Latency fields are in
milliseconds. Plotting imports `plotly` unconditionally.

GPU profiling needs the heavier `environment_profiling.yml` environment,
which adds `vllm`, `flashinfer-python`, `torch`, `triton`, and `cuda-nvcc`.

Detailed guides:

- [`docs/cli/README.md`](docs/cli/README.md) — CLI usage and metrics output
- [`docs/profiling/README.md`](docs/profiling/README.md) — profiling
  wrappers and simulator CSV smokes
- [`docs/training/README.md`](docs/training/README.md) — predictor training
  and cache management
- [`examples/README.md`](examples/README.md) — example catalog

### Canonical TTFT contract

Both implementations define TTFT as **queue-visible request arrival → first
generated token completion**. Arrival-to-prefill-completion is reported
separately as `prefill_latency` (Python) / `prefill_latency_ms` (C++). In
PDD, TTFT therefore includes KV-cache transfer and the first decode
execution. Historical comparison artifacts that reconstructed TTFT from
prefill completion are `prefill_latency` baselines, not canonical TTFT.

## License

Frontier is released under the MIT License. See [`LICENSE`](LICENSE).
