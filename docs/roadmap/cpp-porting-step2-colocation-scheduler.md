# C++ Porting Step 2: Co-location vLLM V1 Scheduler

## Status

Complete on the `cxx-port` branch.

Step 2 starts from the completed Step 1 foundation commit:

```text
0750a72 feat(cpp): complete simulator foundation
```

Implemented slices:

- [x] Step 2A: canonical single-request lifecycle
- [x] Step 2B: multiple arrivals and continuous batch formation
- [x] Step 2C: token budget and ordinary KV-block admission
- [x] Step 2D: monolithic decode iterations
- [x] Step 2E: chunked prefill
- [x] Step 2F: memory-pressure preemption and recovery
- [x] Step 2G: analytical batch timing integration

Completion evidence (2026-07-29):

- Schema v1 remains on the foundation path; schema v2 selects the canonical
  co-location scheduler and fixed or analytical batch execution.
- The differential probe calls the production Python Request, Batch, and
  `VLLMv1EngineReplicaScheduler`. Its fixed-latency matrix covers same-time and
  mid-flight arrivals, batch and token caps, unchunked skipping, allocation
  pressure, watermark admission, decode iterations, chunked/thresholded
  prefill, and recompute preemption.
- Python-generated analytical golden data covers single prefill, short- and
  long-context decode, and mixed prefill/decode batches at `1e-12` absolute and
  relative tolerance.
- A separate analytical parity gate runs the production sequential Python
  `Simulator`; its request completion timestamp, TTFT, E2E, and batch duration
  match the C++ simulator.
- A clean WSL GCC 13.3 build passes all 16 CTest targets. The same suite passes
  under AddressSanitizer and UndefinedBehaviorSanitizer. The differential
  suite passes 28 pytest cases, and the focused Python analytical/session
  regression set passes 49 cases.
- Recompute preemption exposed and fixed a production Python state reset bug:
  resetting token frontiers now also clears `is_prefill_complete`, allowing
  monolithic requests to re-enter prefill recomputation.

The completed Step 2 scope remains intentionally limited to one replica, one
DP target, and one pipeline stage. The follow-up
[Step 2.5 plan](cpp-porting-step2-5-colocation-parallelism-events.md) restores
the Python event boundaries and adds dense TP/PP/DP plus multi-replica
execution without redefining Step 2's frozen schema v2 contract.

This document expands Step 2 of the
[C++ Simulator Porting Plan](cpp-porting-session-prefix-plan.md).

## Goal

Replace the Step 1 fixed foundation-completion path with a canonical
co-location scheduler that mirrors the selected Python vLLM V1 behavior:

```text
arrival
  -> waiting queue
  -> two-phase scheduler decision
  -> batch execution
  -> request token-state update
  -> completion or next scheduler iteration
```

Step 2 ends with deterministic dense co-location runs supporting:

- FCFS vLLM V1-style waiting/running queues;
- continuous batching;
- per-iteration token budgeting;
- ordinary, non-prefix KV-block accounting;
- monolithic prefill and decode progression;
- chunked prefill;
- recompute-style preemption; and
- fixed or analytical batch timing.

## Fixed Scope

### Included

- `co-location` / `MONOLITHIC`;
- one dense replica;
- one data-parallel target;
- one pipeline stage;
- Llama-2-7B TP8 on Rubin for analytical fixtures;
- vLLM V1 scheduler;
- FCFS scheduling;
- explicit KV-block capacity;
- normalized CSV workloads;
- both `offline` and `online` metadata modes over the same trace-driven event
  semantics; and
- session IDs as passive request metadata.

### Excluded

- PDD and every AFD surface;
- more than one replica, DP target, or pipeline stage;
- MoE, EP synchronization, and expert imbalance;
- priority scheduling and Thinking Mode fast lanes;
- speculative decoding / MTP;
- CUDA Graph timing;
- prefix-cache lookup, block hashes, session affinity, and cache eviction;
- CPU KV-cache offload/restore; and
- automatic GPU memory planning.

Unsupported combinations must fail during configuration or simulator
initialization. They must not silently degrade to the Step 1 foundation
lifecycle.

## Python Behavioral Oracle

The primary reference is:

- `frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py`;
- `frontier/entities/request.py`;
- `frontier/entities/batch.py`;
- `frontier/events/request_arrival_event.py`;
- `frontier/events/cluster_batch_end_event.py`;
- `frontier/events/global_batch_end_event.py`; and
- `frontier/simulator.py` sequential mode.

The Step 2 parity harness must call the actual Python
`VLLMv1EngineReplicaScheduler`. It must not reimplement the scheduling
algorithm in a second Python oracle.

Two oracle modes are required:

1. **Scheduler probe**
   - Uses the production Python Request, Batch, and vLLM V1 scheduler.
   - Injects a deterministic fixed batch-completion latency.
   - Emits normalized decisions and request/batch state after each iteration.
   - Drives Steps 2A through 2F without predictor noise.
2. **Full simulator analytical run**
   - Uses the production sequential Python Simulator and analytical predictor.
   - Validates the Step 2G end-to-end timestamps and analytical diagnostics.

The existing
`FRONTIER_VLLM_V1_SCHED_DECISION_LOG_PATH` stream provides iteration IDs,
decision results, request IDs, token budgets, available block counts, request
counts, batch order, and per-request scheduled tokens. The parity normalizer
must drop its wall-clock timestamp and compare simulation-time fields.

## Contract Evolution

Step 1 input and output schema v1 is frozen and remains available for the
foundation tests. Step 2 introduces schema v2 instead of adding scheduler
fields to v1.

### Normalized configuration schema v2

The initial fixed-timing shape is:

```json
{
  "schema_version": 2,
  "run_id": "step2-single-request",
  "simulation_mode": "offline",
  "system_architecture": "co-location",
  "enable_parallel_clusters": false,
  "prefix_cache": {
    "enabled": false,
    "key_mode": "session"
  },
  "scheduler": {
    "type": "vllm_v1",
    "scheduling_policy": "fcfs",
    "batch_size_cap": 8,
    "max_tokens_in_batch": 128,
    "enable_preemption": false,
    "enable_chunked_prefill": false,
    "long_prefill_token_threshold": 0,
    "block_size": 16,
    "num_blocks": 128,
    "watermark_blocks_fraction": 0.0,
    "num_preallocate_tokens": 0
  },
  "execution_model": {
    "type": "fixed",
    "batch_latency_ms": 1.0
  }
}
```

The analytical execution-model variant requires:

```json
{
  "type": "analytical",
  "device": "rubin",
  "model": "llama2-7b",
  "precision": "fp16",
  "tensor_parallel_size": 8,
  "num_layers": 32,
  "network_bandwidth_gbps": 400.0,
  "network_latency_us": 1.0,
  "intra_node_bandwidth_gbps": 14400.0
}
```

Validation rules:

- `schema_version=1` continues to select only the Step 1 foundation path.
- `schema_version=2` selects the canonical scheduler path.
- Step 2 v2 requires co-location, sequential execution, and prefix caching
  disabled.
- `type=vllm_v1` and `scheduling_policy=fcfs` are the only accepted scheduler
  values.
- Batch size, token budget, block size, and block count must be positive.
- `watermark_blocks_fraction` must be finite and in `[0, 1)`.
- `long_prefill_token_threshold` must be nonnegative and requires chunked
  prefill when nonzero.
- Step 2 accepts only explicit `num_blocks`; memory-planner modes remain a
  Python-side concern.
- The selected analytical model values must match the implemented Rubin,
  Llama-2-7B, and TP8 surface.

### Output schema v2

Schema v2 retains the v1 fields and uses
`metrics_semantics=canonical`. It adds:

```text
requests[].first_scheduled_at_s
requests[].first_token_completed_at_s
requests[].scheduling_delay_ms
requests[].num_processed_tokens
requests[].preemption_count

batches[]
  batch_id
  iteration_id
  scheduled_at_s
  completed_at_s
  request_ids
  scheduled_tokens
  total_scheduled_tokens
  num_prefill_tokens
  num_decode_tokens

scheduler_trace[]
  iteration_id
  simulation_time_s
  decision sequence
  token budget before/after
  available blocks before/after
  waiting/running/preempted counts
  batch request order
  per-request scheduled tokens
```

IDs, counts, queue order, decision order, and event order compare exactly.
Simulation timestamps and analytical values use documented absolute and
relative tolerances.

## Target C++ Layout

Headers and implementations remain together in functional directories:

```text
cpp/frontier/
  config/
    config.h
    config.cc
  core/
    event.h
    event_queue.h
  entities/
    request.h
    request.cc
    batch.h
    batch.cc
  scheduler/
    scheduler_types.h
    global_scheduler/
      base_global_scheduler.h
      co_location_global_scheduler.h
      co_location_global_scheduler.cc
    cluster_scheduler/
      base_cluster_scheduler.h
      co_location_cluster_scheduler.h
      co_location_cluster_scheduler.cc
    replica_scheduler/
      base_replica_scheduler.h
      base_replica_scheduler.cc
      vllm_v1_engine_replica_scheduler.h
      vllm_v1_engine_replica_scheduler.cc
    replica_stage_scheduler/
      replica_stage_scheduler.h
      replica_stage_scheduler.cc
    kv_block_accounting.h
    kv_block_accounting.cc
  execution_time_predictor/
    analytical_roofline_execution_time_predictor.h
    analytical_roofline_execution_time_predictor.cc
  metrics/
    output_contract.h
    output_contract.cc
  simulator/
    simulator.h
    simulator.cc
    co_location_simulator.h
    co_location_simulator.cc

cpp/tests/
  entities/
  scheduler/
  simulator/
  parity/
  fixtures/step2/
```

The scheduler ownership hierarchy follows the production Python shape even
though Step 2 has only one target at each level:

```text
CoLocationGlobalScheduler
  -> CoLocationClusterScheduler
    -> VllmV1Scheduler : BaseReplicaScheduler
      -> ReplicaStageScheduler
        -> BatchExecutionModel
```

Global and cluster routing are deterministic pass-through operations for the
single monolithic cluster, replica zero, and DP target zero. The explicit
boundaries are retained so later multi-replica, DP, PP, and disaggregated
milestones extend ownership rather than splitting a flat simulator loop.

Do not create a GPU block-pool or prefix-cache map in Step 2. Ordinary
capacity accounting needs only per-request block counts and a total allocated
count. Physical block identities begin in the session prefix-cache milestone.

## Core Data Structures

### Request

The C++ Request mirrors the selected Python state transitions while using IDs:

```text
immutable input:
  request_id
  arrived_at
  num_prefill_tokens
  num_decode_tokens
  optional session_id / turn index

runtime:
  num_processed_tokens
  scheduler_num_computed_tokens
  is_prefill_complete
  completed
  preempted
  runtime_epoch
  execution_epoch
  preemption_count
  first_scheduled_at
  prefill_completed_at
  first_token_completed_at
  completed_at
  cumulative_waiting_time
```

Invariants:

- `0 <= num_processed_tokens <= prefill + decode`.
- Derived prefill progress is
  `min(num_processed_tokens, num_prefill_tokens)`.
- Derived decode progress is
  `max(num_processed_tokens - num_prefill_tokens, 0)`.
- `scheduler_num_computed_tokens` may lead request-visible progress while a
  scheduled batch is in flight.
- On monolithic prefill completion, the Python path grants the first decode
  token at the same boundary. If output length is one, the request completes
  at that boundary.
- Prefill and first-token timestamps are written once.
- A completed request cannot re-enter any scheduler queue.

### Batch

Batch is a stable arena value referenced by `BatchId`:

```text
batch_id
iteration_id
request_ids
scheduled_tokens per request
request runtime/execution epoch snapshots
request progress snapshots
scheduled_at
completed_at
schedule_epoch
```

The simulator retains batch slots until the run ends so event payloads never
hold pointers or dangling references. The event layer validates the batch
generation before dispatching completion. For a generation-valid batch, the
scheduler validates and applies each request snapshot independently: a stale
request is skipped while valid requests in the same batch continue to mutate.
The batch is then completed and its in-flight state is released, matching the
production Python scheduler contract.

### Scheduler

The selected scheduler state is:

```text
waiting queue:    deque<RequestId>
running order:    vector<RequestId>
preempted queue:  deque<RequestId>
allocation map:   RequestId -> allocated block count
total allocated blocks
iteration ID
in-flight batch count
```

FCFS queue rules:

- initial arrivals append to waiting;
- preempted requests re-enter at the front;
- running requests retain admission order;
- newly admitted/resumed requests appear before continuing running requests
  in the emitted batch, matching Python output order.

## Event Model

Add only the events required by the single-replica vertical slice:

```text
RequestArrival
SchedulerPoll
BatchCompletion
```

`FoundationCompletion` remains only for schema v1.

Normalized flow:

```text
RequestArrival
  -> enqueue request
  -> SchedulerPoll
  -> create and schedule Batch
  -> BatchCompletion
  -> mutate request progress and free/retain resources
  -> SchedulerPoll
```

All workload arrival events are inserted before the event loop starts.
Consequently, arrivals at a timestamp precede scheduler polls created while
handling those arrivals. This preserves the Step 1 `(time, sequence)` rule and
allows same-time arrivals to batch deterministically.

The C++ event trace does not need to reproduce every Python global/cluster/
replica forwarding event. The differential normalizer compares these semantic
milestones:

- request arrival;
- scheduler iteration;
- batch start; and
- batch completion.

## Scheduler Semantics to Preserve

Each scheduler iteration has two phases:

1. **RUNNING**
   - Visit running requests in order.
   - Determine their next schedulable tokens.
   - Apply max-model, long-prefill, token-budget, and block-capacity limits.
   - Preempt only in this phase.
2. **WAITING**
   - Run only when the RUNNING phase performed no preemption.
   - Enforce `batch_size_cap`.
   - Admit FCFS requests while budget and blocks allow.
   - With chunking disabled, skip an oversized prefill for the iteration and
     continue considering later waiting requests.
   - Stop at the first waiting allocation failure.

Batch output order is:

```text
new/resumed admissions, then continuing running requests
```

### Ordinary KV-block accounting

For prefix caching disabled:

- required blocks use `ceil(tokens / block_size)`;
- a new admission checks that the post-reservation free count stays at or
  above the configured watermark;
- the initial allocation materializes blocks only for tokens scheduled in the
  current iteration;
- a running request grows its allocation only when the accounted frontier
  crosses a block boundary;
- completion frees every block held by the request; and
- successful termination requires zero allocated blocks.

Preserve Python's distinction between:

- request-visible processed tokens;
- scheduler-visible computed tokens; and
- KV-accounted tokens at the monolithic first-decode boundary.

## Vertical Slices

### Step 2A: Canonical Single-request Lifecycle

#### Scope

- Add a Python scheduler probe that directly drives the production Request,
  Batch, and `VLLMv1EngineReplicaScheduler`.
- Freeze the first normalized config, workload, scheduler trace, and expected
  output fixture from that probe.
- Add schema v2 fixed execution configuration.
- Add minimal Request and Batch entities.
- Add SchedulerPoll and BatchCompletion.
- Support one request with `num_decode_tokens=1`.
- Use one fixed-latency batch.
- Produce output schema v2 with canonical metrics.

#### Acceptance

- The request transitions `waiting -> running -> completed` once.
- Exactly one batch is created.
- Prefill completion, first-token completion, and request completion occur at
  the same batch boundary.
- TTFT and E2E use canonical fields, not foundation-placeholder semantics.
- Python and C++ request IDs, event order, batch contents, and timestamps
  match.

### Step 2B: Multiple Arrivals and Continuous Batch Formation

#### Scope

- Admit multiple decode-length-one requests.
- Enforce FCFS and `batch_size_cap`.
- Handle arrivals while one batch is in flight.
- Trigger the next scheduler iteration at batch completion.
- Permit deterministic same-time arrival batching.

#### Fixtures

- two requests arriving at the same timestamp;
- an arrival during an active batch;
- more waiting requests than `batch_size_cap`; and
- repeated runs proving byte-stable output.

#### Acceptance

- No request appears in two in-flight batches.
- Batch request order matches Python.
- Each request completes exactly once.
- Waiting/running queue counts and batch IDs match at every iteration.

### Step 2C: Token Budget and Ordinary KV-block Admission

#### Scope

- Enforce `max_tokens_in_batch`.
- Add explicit block size, block count, watermark, and allocation map.
- Implement waiting admission without preemption.
- Record decision reasons and allocation snapshots.

#### Fixtures

- two prefills fitting one token budget;
- token budget splitting requests across batches;
- oversized unchunked head request skipped while a smaller follower is
  admitted;
- waiting allocation failure causing head-of-line stop;
- exact block-boundary and one-token-over-boundary growth; and
- watermark admission rejection.

#### Acceptance

- Sum of scheduled tokens never exceeds the iteration budget.
- Running request count never exceeds `batch_size_cap`.
- Allocated blocks stay in `[0, num_blocks]`.
- Successful runs finish with an empty allocation map.
- Unschedulable runs fail with a deterministic quiescence report rather than
  hanging.

### Step 2D: Monolithic Decode Iterations

#### Scope

- Support output lengths greater than one.
- Grant the first decode token at the prefill-complete boundary.
- Schedule one subsequent decode token per running request per iteration.
- Mix running decode work with new waiting prefill work.
- Free resources only at final completion.

#### Critical invariants

- The prefill boundary advances request-visible decode progress by one.
- The scheduler-visible frontier does not double-count that boundary token.
- RUNNING is scheduled before WAITING.
- Emitted batch order remains waiting admissions before running continuations.
- TTFT is arrival to prefill completion; E2E is arrival to final decode
  completion.

#### Fixtures

- decode length one;
- decode length two;
- long decode with a new request arriving mid-run;
- mixed prefill/decode batch under a tight token budget; and
- KV growth across a decode block boundary.

### Step 2E: Chunked Prefill

#### Scope

- Enable partial prefill admission.
- Keep partial-prefill requests in the running queue.
- Schedule running partial prefills before new waiting requests.
- Apply `long_prefill_token_threshold` before the remaining iteration budget.
- Record prefill completion only after the final chunk.

#### Fixtures

- prompt exactly equal to budget;
- prompt one token above budget;
- multiple chunks;
- long-prefill threshold smaller than batch budget;
- partial prefill mixed with decode; and
- incremental KV-block growth over chunks.

#### Acceptance

- Chunk sizes and per-iteration order match Python exactly.
- Sum of chunks equals the original prefill length.
- No decode progress occurs before the final prefill chunk completes.
- Prefix-cache fields remain unused.

### Step 2F: Memory-pressure Preemption and Recovery

#### Scope

- Enable preemption in RUNNING only.
- Select the FCFS tail victim, excluding the requesting request.
- If no other victim exists, preempt the requesting request.
- Free all victim blocks.
- Reset recompute progress and scheduler frontier.
- Prepend the victim to the preempted waiting queue.
- Increment epochs and preemption metrics.
- Skip WAITING phase for an iteration that preempted work.
- Roll back and refund any same-iteration scheduled victim.

#### Stale-event safety

Batch completion has two stale-event levels:

- If the batch schedule epoch changed, the event layer ignores the entire
  completion event.
- For a generation-valid batch, each request is checked independently. A
  request is skipped when its runtime/execution epoch changed, its scheduled
  progress snapshot no longer matches, or it already completed.
- Other requests with valid snapshots in the same batch are still updated.
  The batch completes and its in-flight state is released even when one or
  more request snapshots are stale.

#### Fixtures

- one tail victim;
- requesting-request self-preemption;
- same-iteration schedule rollback;
- recomputation and eventual completion;
- preemption disabled under the same pressure; and
- stale completion after preemption.

#### Acceptance

- Victim selection and queue re-entry match Python.
- Block accounting is conserved through preemption.
- A stale request snapshot never mutates that request.
- Valid request snapshots in a mixed stale/valid batch are still applied.
- Preemption counts, tokens-at-preemption, final completion count, TTFT, and
  E2E match the selected Python fixtures.

### Step 2G: Analytical Batch Timing Integration

#### Scope

Introduce a small execution-model interface:

```text
BatchExecutionModel
  FixedBatchExecutionModel
  AnalyticalRooflineExecutionTimePredictor
```

The analytical predictor converts each immutable batch snapshot into:

- prefill attention slices with scheduled query tokens and prior context;
- decode attention slices with one query token and current context;
- dense attention and MLP component times for every layer; and
- required TP communication costs.

The predicted batch duration is converted from milliseconds to event seconds
exactly once at the simulator boundary.

#### Validation

- Keep all Steps 2A-2F fixed-latency fixtures.
- Add Python-generated analytical batch fixtures for:
  - single prefill;
  - single decode at short context;
  - single decode at long context; and
  - mixed prefill/decode.
- Compare component diagnostics and final batch duration at `1e-12` absolute
  and relative tolerance unless a formula-specific tolerance is documented.
- Run one complete analytical co-location workload through the full Python
  Simulator and C++ simulator.

## Differential Harness

Extend `cpp/tests/parity/test_differential.py` with Step 2 cases. On failure,
preserve:

```text
normalized config
normalized workload
Python request metrics
C++ request metrics
Python scheduler decision log
C++ scheduler trace
Python batch/event trace
C++ batch/event trace
field-level difference report
```

Normalization rules:

- drop wall-clock timestamps and output-directory paths;
- compare simulation time, IDs, counts, tokens, decisions, and order;
- convert Python decision-log string IDs to integers;
- compare request and batch rows by stable ID;
- compare floats with explicit tolerances; and
- never sort queues, batch request lists, decisions, or event traces.

Every vertical slice gets at least one Python/C++ fixture before the next slice
starts.

## Tests

CTest remains responsible for C++ unit and contract tests:

```text
frontier_entities.request
frontier_entities.batch
frontier_scheduler.vllm_v1
frontier_scheduler.kv_block_accounting
frontier_simulator.colocation
frontier_contract.output_v2
```

Pytest remains responsible for Python/C++ differential tests.

Required invariant tests:

- token and block counts never underflow or overflow;
- request IDs occur in at most one queue/state;
- in-flight batches contain unique request IDs;
- scheduler frontier never trails completed KV-accounted work incorrectly;
- completion timestamps are monotonic;
- every successful request completes once;
- completed requests hold no blocks;
- successful simulator quiescence means all queues, batches, and allocations
  are empty; and
- unsupported features fail before the event loop starts.

## Step 2 Exit Criteria

Step 2 is complete only when:

- schema v1 foundation tests remain unchanged and passing;
- schema v2 config/output contracts are documented and tested;
- all Steps 2A-2G CTest targets pass from a clean WSL build;
- every Step 2 differential fixture passes against the Python oracle;
- fixed-latency and analytical co-location runs both complete deterministically;
- request completion count, batch order, token progression, TTFT, E2E,
  scheduler decisions, and block accounting match Python;
- all successful runs end with no queued work, in-flight batches, or allocated
  blocks;
- block-hash, prefix-cache, PDD, MoE, speculative decode, multi-replica, DP,
  and PP configurations still fail clearly; and
- no session affinity, physical cache-block identity, cache lookup, eviction,
  or CPU-tier behavior has leaked into Step 2.

## Suggested Change Sequence

Keep each change independently reviewable:

1. `test(cpp): add Step 2 production-scheduler oracle probe`
2. `feat(cpp): add scheduler schema v2 and canonical output contract`
3. `feat(cpp): add request and batch runtime entities`
4. `feat(cpp): add canonical single-request scheduler lifecycle`
5. `feat(cpp): add continuous co-location batch formation`
6. `feat(cpp): add token budget and ordinary KV admission`
7. `feat(cpp): add monolithic decode iterations`
8. `feat(cpp): add chunked prefill`
9. `feat(cpp): add recompute preemption`
10. `feat(cpp): connect analytical batch timing`
11. `test(cpp): complete Step 2 Python differential matrix`

## Recommended Starting Point

Begin with the Python scheduler probe and schema v2 fixtures before writing
C++ scheduler logic. The first executable target is deliberately narrow:

```text
one co-location request
prefill=32
decode=1
fixed batch latency=1 ms
prefix cache disabled
no chunking
no preemption
```

That case establishes the real Request/Batch/event/output boundary and removes
the Step 1 foundation placeholder without prematurely introducing token-budget
or KV-memory policy.
