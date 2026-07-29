# C++ Porting Step 1: Contracts, Parity, and Simulator Foundation

## Status

Complete on the `cxx-port` branch.

Current progress:

- [x] Step 1A: minimal build skeleton
- [x] Step 1B: deterministic event core
- [x] Step 1C: versioned input and output contracts
- [x] Step 1D: analytical model foundation
- [x] Step 1E: minimal lifecycle smoke
- [x] Step 1F: Python/C++ differential harness

This document expands Step 1 of the
[C++ Simulator Porting Plan](cpp-porting-session-prefix-plan.md). Its purpose is
to establish a buildable, testable, deterministic foundation before any
vLLM-style scheduling, KV-cache admission, session prefix caching, or PDD
behavior is implemented.

## Fixed Decisions

- The C++ implementation lives under `cpp/` in the existing
  `Frontier-cxx-port` worktree.
- Headers and implementation files live together in functional directories.
- The canonical initial development environment is WSL Ubuntu with a modern
  GCC toolchain.
- The source remains in the existing Windows worktree; it is not copied into a
  second WSL checkout.
- Build artifacts live outside the worktree in the WSL filesystem.
- The initial language level is C++20.
- Step 1 begins without third-party C++ dependencies.
- Python remains the behavioral and numerical oracle.
- Only session-derived prefix keys are in scope for the C++ MVP.
  `block_hash_ids` and `prefix_caching_key_mode=block_hash` remain unsupported.

## Development Environment

The existing source worktree is visible at:

```text
Windows:
C:\Users\jklpr\Desktop\project\Frontier-cxx-port

WSL:
/mnt/c/Users/jklpr/Desktop/project/Frontier-cxx-port
```

Use a separate WSL-native build directory:

```text
~/frontier-build/cxx-port
```

The current WSL Ubuntu environment has GCC 13.3. The supported Step 1 baseline
is GCC 11 or newer (or Clang 14 or newer), CMake 3.24 or newer, and Ninja 1.10
or newer. CMake and Ninja must be installed before Step 1A can be accepted:

```bash
sudo apt update
sudo apt install -y cmake ninja-build
```

When WSL `sudo` is unavailable, use an isolated user-space tool environment:

```bash
python3 -m venv ~/frontier-tools
~/frontier-tools/bin/python -m pip install \
  cmake==3.31.10 \
  ninja==1.13.0
export PATH="$HOME/frontier-tools/bin:$PATH"
```

Canonical configure, build, and test commands:

```bash
cd /mnt/c/Users/jklpr/Desktop/project/Frontier-cxx-port
export PATH="$HOME/frontier-tools/bin:$PATH"

cmake -S cpp \
  -B ~/frontier-build/cxx-port \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug

cmake --build ~/frontier-build/cxx-port

ctest \
  --test-dir ~/frontier-build/cxx-port \
  --output-on-failure
```

The build must not create generated files under `cpp/`.

## Step 1A: Minimal Build Skeleton

### Goal

Create the smallest dependency-free C++ project that configures, builds, runs,
and is exercised through CTest.

### Initial layout

```text
cpp/
  CMakeLists.txt
  frontier/
    main.cc
```

Do not create placeholder headers or empty future feature directories.
Directories are added only when their vertical slice begins.

### Build contract

- Require CMake 3.24 or newer.
- Set `CMAKE_CXX_STANDARD` to 20 and disable compiler extensions.
- Build an executable named `frontier_sim`.
- Implement `frontier_sim --version`.
- Enable useful GCC/Clang/MSVC warnings without making compiler-specific
  options leak into source files.
- Register a CTest smoke test that runs `frontier_sim --version`.
- Keep the skeleton free of JSON, testing-framework, logging, and CLI-library
  dependencies.

### Acceptance

```bash
cmake -S cpp -B ~/frontier-build/cxx-port -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug
cmake --build ~/frontier-build/cxx-port
~/frontier-build/cxx-port/frontier_sim --version
ctest --test-dir ~/frontier-build/cxx-port --output-on-failure
```

All commands must succeed from a clean build directory.

## Step 1B: Deterministic Event Core

### Goal

Implement the minimum reusable DES primitives independently of request
scheduling.

### Target layout

```text
cpp/frontier/core/
  ids.h
  event.h
  event_queue.h
  event_queue.cc

cpp/tests/core/
  event_queue_test.cc
```

### Required behavior

- Define strong or otherwise non-interchangeable IDs for event sequence,
  request, batch, replica, DP target, and generation/epoch values.
- Represent simulation time explicitly as `SimTime`, initially backed by
  `double`.
- Store events as compact values with:
  - simulation time;
  - monotonically increasing creation sequence;
  - event type;
  - ID-based payload; and
  - generation/epoch where stale-event detection is required.
- Order the min-heap by `(time, sequence)`.
- Do not use epsilon comparison in event-queue ordering.
- Reject nonfinite event times at the queue boundary.
- Keep event payload ownership out of the event queue.

### Tests

- earlier time precedes later time;
- equal times preserve creation sequence;
- insertion order does not override an earlier timestamp;
- nonfinite times are rejected;
- empty-queue access is safe and explicit;
- stale generation/epoch matching can be evaluated without object pointers;
- a repeated deterministic fixture produces the same ordered event trace.

### Acceptance

The core tests pass independently of config parsing, analytical models, the
scheduler, and metrics output.

## Step 1C: Versioned Input and Output Contracts

### Goal

Freeze the small machine-readable surface shared by the Python oracle and C++
simulator.

### Configuration contract

- Use normalized JSON rather than reproducing the full flattened Python CLI.
- Require a top-level `schema_version`.
- Define explicit defaults only in one layer.
- Reject unknown enum values and unsupported feature combinations.
- Reject unsupported architectures and `prefix_caching_key_mode=block_hash`
  with clear messages.

### Workload contract

CSV input supports:

- `arrived_at`;
- `num_prefill_tokens`;
- `num_decode_tokens`;
- optional `session_id`; and
- optional `session_turn_index`.

The reader rejects:

- nonfinite or negative arrival times;
- nonfinite or nonpositive token counts;
- malformed integer fields;
- duplicate or missing required columns; and
- `block_hash_ids`.

### Output contract

Define versioned JSON/CSV outputs for:

- run metadata;
- completed request IDs;
- request arrival and completion timestamps;
- TTFT and E2E latency;
- event trace records needed for early parity; and
- analytical-model diagnostics.

The schema must specify units, float serialization precision, required fields,
optional fields, and stable ordering.

### Dependency decision

Step 1A remains dependency-free. Step 1C uses `nlohmann/json` 3.11.3. CMake
first accepts an exact installed package or
`FRONTIER_NLOHMANN_JSON_SOURCE_DIR`; otherwise it downloads the pinned release
archive when `FRONTIER_FETCH_DEPENDENCIES=ON`. The archive is verified with:

```text
SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
```

Both the pinned-download build and the fetch-disabled local-source build are
part of the Step 1C acceptance check. See `cpp/README.md` for commands and the
frozen v1 contract examples.

### Tests

- valid minimal config and workload;
- all required validation failures;
- stable JSON/CSV field ordering where byte comparisons are required;
- read/write round-trip for normalized fixtures; and
- explicit rejection of block-hash prefix input.

## Step 1D: Analytical Model Foundation

### Goal

Port only the pure formulas needed by the MVP and validate them before they are
called by the DES.

### Python references

- `frontier/execution_time_predictor/analytical_roofline_execution_time_predictor.py`;
- `frontier/cc_backend/backends/analytical_cc_backend.py`;
- `frontier/kv_cache_transfer/analytical_kv_cache_transfer_predictor.py`.

The selected Step 1 communication surface is the release example default,
`analytical`, exercised with 72 participants. The topology-aware
`astra_sim_analytical` placement and routing model is not needed by the first
dense vertical slice and is deferred until a later milestone selects it.

### Rules

- Keep execution, communication, and KV-transfer formulas in separate
  functional modules.
- Use side-effect-free inputs and outputs.
- Do not read global simulator state from analytical functions.
- Preserve units explicitly in type and field names.
- Reject invalid hardware/model parameters rather than silently repairing
  them.
- Port only formulas exercised by the selected dense model and NVL72-oriented
  topology.

### Validation

- Generate small golden fixtures from Python.
- Compare individual intermediate terms as well as final latency.
- Use `1e-12` absolute and relative tolerances for execution,
  communication, and transfer latency values; compare byte counts exactly.
- Include boundary cases such as zero communication volume where valid,
  minimal token counts, and the selected 72-participant communication domain.

### Implemented surface

- Rubin FP32/FP16/BF16/FP8/INT8/FP4/INT4 roofline ceilings;
- dense Llama-2-7B TP8 per-layer attention and MLP components;
- simple analytical point-to-point and collective communication formulas;
- dense KV-cache byte sizing and analytical transfer latency; and
- a Python-generated v1 golden fixture covering prefill, long-context decode,
  a non-power-of-two 72-device collective, and KV transfer.

The analytical modules are pure value-in/value-out functions and remain
independent of the event loop.

## Step 1E: Minimal Lifecycle Smoke

### Goal

Exercise the event loop, one request record, and metrics serialization without
implementing the Step 2 scheduler.

```text
arrival -> foundation completion event -> metrics output
```

This is a foundation smoke only. It must not introduce continuous batching,
token budgeting, KV-block admission, chunked prefill, preemption, prefix
caching, or PDD behavior.

### Validation

- one request arrives and completes exactly once;
- the event trace is deterministic;
- request timestamps use the frozen units;
- TTFT/E2E placeholder semantics are clearly separated from the canonical
  scheduler metrics introduced in Step 2; and
- normalized output is consumable by the parity harness.

### Implemented surface

- `frontier/simulator/simulator.h` exposes the scheduler-free lifecycle as a
  pure input/output boundary over normalized config and workload values.
- Each request creates exactly one arrival and one foundation-completion
  event. Equal-time events retain `(time, sequence)` ordering.
- The default fixed service time is 1 ms and is deliberately independent of
  token counts. Foundation `prefill_completed_at` and `completed_at` are the
  same event time.
- Output run metadata uses
  `metrics_semantics=foundation-placeholder`; Step 2 scheduler output will use
  `canonical`.
- `frontier_sim --config <config.json> --workload <workload.csv>` emits the
  normalized v1 JSON result to stdout for Step 1F.
- The foundation runner accepts only co-location with prefix caching disabled.
  PDD and prefix-cache behavior remain explicit later milestones rather than
  silently producing placeholder results.

## Step 1F: Python/C++ Differential Harness

### Goal

Make parity an executable gate before scheduler development begins.

### Harness responsibilities

- Run Python and C++ against the same normalized fixture.
- Normalize paths, float formatting, and intentionally implementation-specific
  metadata.
- Compare fields and report focused, field-level differences.
- Preserve raw Python and C++ artifacts for failed cases.
- Support exact comparison for IDs, counts, and equal-time event ordering.
- Support documented tolerances for analytical floating-point results.

The harness should be driven by `pytest` so it remains integrated with the
existing Python test workflow. C++ unit tests remain driven by CTest.

### Initial comparisons

- configuration acceptance and rejection;
- workload parsing;
- analytical model golden values;
- completed request ID/count for the minimal lifecycle smoke; and
- equal-time event ordering.

### Implemented harness

- `cpp/tests/parity/python_oracle.py` independently implements the normalized
  Step 1 config, workload, and foundation-lifecycle contracts.
- `cpp/tests/parity/test_differential.py` invokes Python and the built C++
  executable against the same fixtures.
- JSON is parsed before comparison, normalizing field order and float text
  formatting. Numeric values use `1e-12` absolute and relative tolerances;
  IDs, counts, strings, booleans, and array/event order compare exactly.
- `FRONTIER_CPP_RUNNER` supports a JSON command prefix, and
  `FRONTIER_CPP_PATH_STYLE=wsl` translates Windows fixture paths for a WSL
  binary.
- A failed comparison writes Python output, C++ stdout/stderr, and the
  field-level difference under pytest's temporary directory. Set
  `FRONTIER_PARITY_ARTIFACT_DIR` to preserve them at a chosen location.
- The analytical gate regenerates the fixture with the actual Python
  predictor and runs the C++ analytical golden test against that same checked
  fixture.

Windows Python invoking the canonical WSL build:

```powershell
$env:FRONTIER_CPP_BINARY = "/home/dskim/frontier-build/cxx-port/frontier_sim"
$env:FRONTIER_CPP_RUNNER = '["wsl","-d","Ubuntu","--"]'
$env:FRONTIER_CPP_PATH_STYLE = "wsl"
python -m pytest cpp/tests/parity -q
```

Native Linux/WSL Python:

```bash
export FRONTIER_CPP_BINARY="$HOME/frontier-build/cxx-port/frontier_sim"
python -m pytest cpp/tests/parity -q
```

## Step 1 Exit Criteria

Step 1 is complete only when:

- a clean WSL out-of-tree configure/build/test succeeds;
- `frontier_sim --version` works;
- deterministic event-core unit tests pass;
- versioned config, workload, and output schemas are documented and tested;
- the selected analytical formulas pass Python-generated golden tests;
- the minimal lifecycle smoke produces deterministic machine-readable output;
- the Python/C++ differential harness passes all Step 1 fixtures;
- unsupported modes, including block-hash prefix caching, fail clearly; and
- no Step 2 scheduler, KV-cache policy, session affinity, or PDD behavior has
  leaked into the foundation.

## Suggested Change Sequence

Keep changes reviewable and independently verifiable:

1. `build(cpp): add minimal CMake and test skeleton`
2. `feat(cpp): add deterministic event queue`
3. `feat(cpp): add versioned input and output schemas`
4. `feat(cpp): port analytical model foundation`
5. `feat(cpp): add minimal lifecycle smoke`
6. `test(cpp): add Python differential harness`

Every change must include the tests required to establish its own acceptance
criteria.
