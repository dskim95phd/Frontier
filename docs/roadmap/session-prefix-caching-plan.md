# Session-Scoped Prefix Caching Implementation Plan

## Status

- **State:** Implemented and correctness-hardened (MVP)
- **Target release:** To be determined
- **Primary scope:** `co-location` and sequential `pd-disaggregation`
- **Compatible schedulers:** `vllm_v1`, `sglang`, and the SJ2Q schedulers that
  inherit the vLLM V1 KV-cache path
- **PDD affinity scheduler:** `sticky_round_robin` (`sticky_lor` remains
  monolithic-only in this release)

## Summary

Frontier currently models prefix-cache reuse using explicit `block_hash_ids`
provided by the workload. This is suitable when a trace contains token-level
prefix equivalence metadata, but it makes simple multi-turn workloads difficult
to describe: a length-only trace must manufacture a block hash for every
cacheable prompt block.

This proposal adds an opt-in, session-scoped keying mode. In this mode, a trace
only needs:

- request arrival time,
- input sequence length (ISL),
- output sequence length (OSL), and
- `session_id`.

Frontier will generate internal block keys from `(session_id, block_index)`.
Requests can therefore reuse KV blocks only within the same session, without
requiring token IDs or explicit block hashes in the dataset. In this mode,
trace ISLs represent only the new tokens added by each turn or Thinking round;
Frontier materializes the effective full prompt before scheduling.

The existing explicit block-hash mode remains the default and retains its
current behavior.

## Motivation

The current prefix-cache implementation does not read or compare real token
IDs. It trusts `Request.block_hash_ids` as externally supplied prefix
equivalence metadata:

- `TraceReplayRequestGenerator` parses the `block_hash_ids` column.
- `KVCacheManager._get_request_block_hashes()` returns those identifiers.
- `BlockPool` looks up cached blocks by identifier.
- Prefix matching stops at the first missing block.

The current `prefix_caching_hash_algo` setting is validated and stored, but it
does not generate hashes from tokens. Consequently, even a simple linear
conversation trace must contain synthetic block identifiers.

For studies that only need session-local KV reuse, token-level equivalence is
unnecessary. A session-local approximation is simpler and sufficient if the
workload obeys an append-only conversation contract.

## Goals

1. Allow prefix caching from a trace with no `block_hash_ids` column.
2. Restrict reuse to requests with the same `session_id`.
3. Preserve block allocation, reference counting, eviction, and contiguous
   prefix-match behavior.
4. Preserve the existing explicit block-hash behavior by default.
5. Report correct per-request and aggregate prefix-cache metrics in both modes.
6. Model the different KV availability semantics of `co-location` and PDD
   without inventing unmodeled cross-cluster transfers.
7. Fail early when a trace violates assumptions that would make session-based
   matching unsafe.

## Non-goals

The first implementation will not model:

- token identity within a session,
- branching conversations under one `session_id`,
- prompt edits in the middle of a conversation,
- context-window truncation,
- migration of cached KV state between replicas,
- decode-to-prefill KV return in PDD,
- closed-loop user think time or turn dependencies,
- sharing common system prompts across different sessions.

Those features require additional workload metadata or new transfer and
scheduling models.

## Workload Contract

### CSV schema

The minimal session-prefix trace is:

```csv
arrived_at,num_prefill_tokens,num_decode_tokens,session_id
0.0,32,16,7
10.0,8,8,7
2.0,24,12,8
```

In session mode, `num_prefill_tokens` is the number of new input tokens added
by that turn. Before creating a `Request`, Frontier expands it into the full
prompt length required by the scheduler:

```text
effective_ISL = prior_session_context + new_ISL
next_session_context = effective_ISL + OSL
```

For session 7:

```text
turn 1: new ISL 32 -> effective ISL 32; response 16; context 48
turn 2: new ISL 8  -> effective ISL 56; response 8;  context 64
```

### Required invariants

For every session:

1. `session_id` is non-empty and identifies one linear conversation.
2. Raw and time-scaled `arrived_at` values are finite and nonnegative, and
   turns appear in nondecreasing scaled-arrival order.
3. Raw ISL/OSL values, their scale factors, and their scaled values are finite
   and positive. The shared conversion rule is
   `int(raw_tokens * scale_factor)`, and that integer result must remain at
   least one.
4. ISL means newly appended input; Frontier never interprets it as a retained
   range or truncation instruction.
5. A session ID is not reused for an unrelated conversation.
6. The expanded effective prompt plus OSL must not exceed `max_tokens`.

For Thinking Mode, every `thinking_round_plans_json` entry follows the same
incremental rule and the same prefill/decode scale factors and integerization
rule as the top-level CSV columns. Scaling occurs before the final round is
compared with the top-level row and before round contexts are accumulated.
Given `48/16` followed by `32/8`, Frontier materializes effective round prompts
of 48 and 96, not 48 and 32. This makes a decreasing per-round new-ISL value
valid without implying context truncation. Runtime and metrics retain the
materialized effective prompt lengths; the CSV remains the source of
incremental inputs.

The implementation fails fast with the trace row and field in the error when
these invariants are violated. It does not clip zero or negative session
lengths to one. Silently accepting invalid, truncated, or branched sessions
would report cache hits for blocks whose token identity is not known.

### Arrival-time semantics

`arrived_at` remains an absolute, open-loop arrival timestamp. Session prefix
caching does not delay a turn until the previous response completes.

Users must select one of the following when trace timing matters:

```bash
--simulation_mode online
```

or:

```bash
--simulation_mode offline
--offline_use_generated_request_arrivals
```

Legacy offline batch mode forces all requests to arrive at time zero. It is not
an appropriate default for causal multi-turn studies.

A future closed-loop workload mode may interpret a field such as `think_time`
as a delay after the prior turn completes. That is independent of the cache-key
change proposed here.

## Configuration Design

Add a mode field to `VllmV1SchedulerConfig`:

```python
prefix_caching_key_mode: str = "block_hash"
```

Allowed values:

| Value | Meaning |
| --- | --- |
| `block_hash` | Existing behavior. Read explicit `block_hash_ids` from each request. |
| `session` | Generate internal block keys from `session_id` and block position. |

Example:

```bash
--vllm_v1_scheduler_config_enable_prefix_caching
--vllm_v1_scheduler_config_prefix_caching_key_mode session
```

The field belongs in `VllmV1SchedulerConfig` so it is inherited by SGLang and
the SJ2Q scheduler configurations that reuse the vLLM V1 cache-management path.

Do not overload `prefix_caching_hash_algo`. Hash algorithm selection and
prefix-identity policy are different concepts.

## Cache-Key Design

### Internal session key

Session mode generates a namespaced, hashable key:

```python
("session", int(request.session_id), block_index)
```

The tuple namespace prevents collisions with integer `block_hash_ids` used by
the existing mode.

For a block size of 16, the first three blocks of session 7 are:

```text
("session", 7, 0)
("session", 7, 1)
("session", 7, 2)
```

### Separate lookup and storage key sequences

The current manager uses the same request hash list for lookup and storage.
Session mode needs two related but distinct sequences:

1. **Lookup keys**
   - Cover only full blocks in the current prompt.
   - Upper bound:

     ```text
     floor(num_prefill_tokens / block_size)
     ```

2. **Storage keys**
   - Provide keys for all full blocks that this request may compute.
   - In a monolithic cluster, the upper bound may grow through prefill and
     decode.
   - In a PDD prefill cluster, only blocks actually processed by the prefill
     scheduler are cached; merely creating a possible key must not mark a block
     as available.

Suggested manager helpers:

```python
def _get_request_lookup_block_keys(self, request: Request) -> list[Hashable]:
    ...

def _get_request_storage_block_keys(
    self,
    request: Request,
    *,
    start_block_index: int,
    end_block_index: int,
) -> list[Hashable]:
    ...
```

`get_computed_blocks()` uses lookup keys. Storage keys are published only after
the batch has completed the corresponding full blocks. `mark_blocks_computed()`
first compares the completed frontier with the already-published frontier. If
it has not advanced, it returns without generating any keys; otherwise it
generates keys only for `[published_frontier, ready_frontier)`. Decode-time
cost is therefore proportional to newly ready full blocks rather than all
blocks in a long request.

### Allocated versus ready blocks

Allocating a block does not make it a prefix-cache hit. This distinction is
required when multiple requests from one session are admitted in the same
scheduling pass: later admissions must not observe work that has merely been
scheduled for an earlier request.

The manager therefore tracks two states:

1. blocks reserved or attached to a request; and
2. the full-block frontier confirmed ready after batch completion.

`allocate_slots()` only performs the first transition. After
`Batch.on_batch_end()` advances the request's completed-token frontier, the
replica scheduler calls `mark_blocks_computed()` to publish newly ready keys.
Chunked prefill advances this frontier one completed chunk at a time.

### Prefix and partial-block semantics

The existing semantics remain:

- Only full blocks can be reused.
- Lookup stops at the first missing block.
- A cached block must remain subject to normal refcount and eviction behavior.
- A fully cached prompt still follows the existing vLLM V1 rule that forces at
  least one block of new work when the scheduler cannot admit a zero-token
  prefill.

No cache hit should be synthesized from session history alone. A matching
session-position key must exist in the local `BlockPool`.

## Architecture Semantics

### Co-location

Prefill and decode execute on the same replica. Full blocks produced while
decoding can remain in that replica's cache and may be reused by a later turn
in the same session.

Example with block size 16:

```text
turn 1: new ISL=32, OSL=16 -> effective ISL=32
turn 2: new ISL=8          -> effective ISL=56
```

If turn 1 has completed and its blocks have not been evicted, turn 2 may reuse
48 tokens: the two prompt blocks and one full decode block.

### Sequential PDD

Prefix caching is owned by the prefill cluster. Decode output KV is not
automatically returned to that cluster.

Using the same example, the first follow-up turn may reuse only the 32 prompt
tokens that the prefill cluster previously computed. It must recompute the
prior output tokens as part of the new 56-token prompt. Once that prompt is
processed, later turns can reuse the larger prefix from the prefill cache.

The implementation must preserve this difference. Reporting a 48-token PDD
hit would imply an unmodeled decode-to-prefill transfer.

Decode-to-prefill KV return, session-aware KV migration, or remote cache lookup
should be proposed as separate features with explicit transfer cost and cache
ownership.

### Multiple replica/DP cache targets

The sticky scheduler requirement is based on the total number of cache targets,
not only the physical replica count:

```text
num_cache_targets = num_replicas * data_parallel_size
```

Requests from one session must be routed to the same `(replica_id, dp_id)`
target because each DP replica scheduler owns an independent `BlockPool`.
Consequently, `num_replicas=1` with `data_parallel_size=2` still requires a
sticky cluster scheduler.

For this release, `sticky_lor` supports only the `MONOLITHIC` cluster path.
Sequential PDD Prefix Caching must use `sticky_round_robin`; selecting
`sticky_lor` for PDD fails during cluster validation instead of reaching the
guarded runtime path.

Session keying does not replace replica affinity.

## Detailed Code Changes

### 1. Configuration

File:

- `frontier/config/config.py`

Changes:

- Add `prefix_caching_key_mode` to `VllmV1SchedulerConfig`.
- Validate `{"block_hash", "session"}`.
- Keep `block_hash` as the default.
- Add CLI help explaining the required trace metadata for each mode.
- Add config unit tests for valid and invalid values.

### 2. Trace and cluster validation

Files:

- `frontier/scheduler/cluster_scheduler/base_cluster_scheduler.py`
- `frontier/request_generator/trace_replay_request_generator.py`, if shared
  parsing/validation helpers are extracted

Changes:

- Preserve the trace-replay requirement for prefix caching.
- In `block_hash` mode, require non-empty `session_id` and `block_hash_ids`
  under the existing public contract.
- In `session` mode, require non-empty `session_id` but not
  `block_hash_ids`.
- Accumulate per-session incremental ISLs into effective full-prompt lengths.
- Apply the same accumulation to explicit Thinking Mode round plans.
- Detect expanded contexts that exceed `max_tokens` and reject them in session
  mode.
- Require a sticky scheduler whenever
  `num_replicas * data_parallel_size > 1`.
- Produce mode-specific errors that identify the file and row number.

Where possible, parsing rules should be shared with trace replay so validation
and request construction do not interpret values differently.

### 3. Request cache metadata

File:

- `frontier/entities/request.py`

Changes:

- Add request-level lookup metadata:

  ```text
  prefix_cache_query_blocks
  prefix_cache_hit_blocks
  prefix_cache_key_mode
  ```

- Replace or extend `on_cache_hit()` with a method that records lookup results
  even when the hit count is zero, for example:

  ```python
  on_prefix_cache_lookup(
      *,
      query_blocks: int,
      hit_blocks: int,
      cached_tokens: int,
      key_mode: str,
  )
  ```

- Preserve `num_prefill_tokens_cached` and the existing processed-token
  transition.
- Separate first-lookup metrics from admission-time runtime restoration.
  Preemption may reset `num_processed_tokens` and require a later admission to
  restore a cached frontier, but it must not rewrite the request's original
  query/hit metrics.
- Include the new fields in request debug serialization.

### 4. KV-cache manager

Files:

- `frontier/kv_cache/base_kv_cache_manager.py`
- `frontier/kv_cache/replica_kv_cache_manager.py`, only if the mode-specific
  policy is kept in the replica subclass

Changes:

- Pass `prefix_caching_key_mode` into the manager.
- Introduce lookup-key and storage-key helpers.
- Generate namespaced tuple keys in session mode.
- Require `session_id` in session mode.
- Require `block_hash_ids` in block-hash mode.
- Return or expose the query-block count along with hit blocks/tokens.
- Keep newly allocated blocks private until batch completion calls
  `mark_blocks_computed()`.
- Return immediately when the completed full-block frontier has not advanced,
  and generate only the newly published storage-key range when it has.
- Keep `BlockPool` allocation, touch, free, and eviction semantics unchanged.

`BlockPool` already accepts `Hashable` keys, so it should not require a data
model change.

### 5. Scheduler admission

File:

- `frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py`

Changes:

- Pass the configured key mode when constructing
  `ReplicaKVCacheManager`.
- Make `_prepare_prefix_cache_admission()` validate metadata according to the
  selected mode.
- Carry query count, hit count, cached tokens, and computed blocks as one
  admission result.
- Commit request-level lookup metrics when the request is actually admitted.
- Preserve chunked-prefill, preemption, allocation, and the existing
  fully-cached-prompt adjustment.

Avoid recording request-level lookup results every time an unadmitted waiting
request is reconsidered. Scheduling-attempt statistics and request outcome
statistics should remain distinct.

Scheduler decision diagnostics use successful admissions as their commit
point. Their block-level fields are:

```text
prefix_cache_admissions
prefix_cache_queries
prefix_cache_hits
prefix_cache_metric_semantics=successful_admission_block_level
```

A lookup performed while checking token budget or allocation feasibility does
not change these counters. Re-admission after preemption is a new successful
admission and is counted once when allocation succeeds.

### 6. Metrics

Files:

- `frontier/metrics/metrics_store.py`
- `frontier/metrics/constants.py`, only if additional exported fields are
  needed

Changes:

- Stop deriving query blocks exclusively from
  `len(request.block_hash_ids)`.
- Export query and hit blocks from request-level cache lookup metadata.
- Include `key_mode` in `prefix_cache_statistics`.
- Preserve:
  - cached prefill tokens,
  - requests with hits,
  - total query blocks,
  - total hit blocks, and
  - hit ratio.
- Keep round-local lookup state separate from request-lifetime totals.
  Request CSV and `prefix_cache_statistics` sum cached tokens, query blocks,
  and hit blocks across all Thinking rounds. A preemption/re-admission within
  one round restores runtime state without adding the same round twice.
- Verify that block-hash-mode metrics remain byte-for-byte compatible where
  practical.

### 7. Examples and documentation

Files:

- `examples/fixtures/`
- `examples/architecture/co-location/`
- `examples/architecture/pdd/`
- `examples/README.md`
- `examples/architecture/README.md`
- `docs/cli/README.md`

Changes:

- Keep the existing explicit-block-hash fixture as a compatibility example.
- Add a length-only multi-turn fixture with multiple sessions.
- Add or adapt a co-location session-prefix example.
- Add a sequential PDD session-prefix example and document its smaller
  first-follow-up hit boundary.
- Ensure examples replay trace arrival times.
- Document the incremental-ISL materialization contract and unsupported
  truncation.

## Testing Plan

### Configuration tests

- Default mode is `block_hash`.
- `block_hash` and `session` are accepted.
- Unknown values fail during config construction.

### Trace validation tests

- Session mode accepts a trace without `block_hash_ids`.
- Session mode rejects an empty or missing `session_id`.
- Block-hash mode retains its existing metadata checks.
- Independent sessions accumulate context separately.
- Explicit Thinking Mode rounds accumulate incremental ISLs within the session.
- Explicit round values use the same scale factors and integerization rule as
  the top-level ISL/OSL columns.
- Nonfinite/negative arrivals and nonfinite/nonpositive raw or scaled token
  counts fail before scheduling instead of being clipped.
- An expanded context that exceeds `max_tokens` fails.
- Multiple physical replicas or DP lanes require a sticky scheduler.

### KV manager unit tests

- Same session and position produces a hit.
- Different sessions with identical lengths do not share blocks.
- Partial blocks do not hit.
- Matching stops at the first missing block.
- Evicted session blocks miss.
- Blocks scheduled in the same admission pass remain misses until their batch
  completes.
- A cached block with `ref_cnt == 0` is touched correctly on reuse.
- Monolithic decode can add reusable full session blocks.
- PDD prefill does not cache unprocessed decode output.
- Repeated `mark_blocks_computed()` calls at an unchanged frontier generate no
  session keys; an advancing frontier generates only its delta.

### Scheduler integration tests

- A second turn receives the expected cached-token count.
- Query metrics are correct without `block_hash_ids`.
- A fully cached prompt retains the one-block recomputation rule.
- Chunked prefill grows the cached full-block frontier correctly.
- Preemption and re-admission restore runtime cached tokens without
  double-counting or rewriting request-level metrics.
- Hidden and final Thinking-round cache lookups are accumulated in final
  request/system metrics, while re-admission within one round is not counted
  twice.
- SGLang and SJ2Q inherit the same key-mode behavior.
- Scheduler diagnostic counters remain unchanged across lookup retries and are
  committed once after successful allocation.
- Sequential PDD rejects `sticky_lor` during validation; monolithic
  Prefix Caching continues to allow it.

### End-to-end tests

Use at least two sessions so cross-session isolation is observable.

For block size 16:

```csv
arrived_at,num_prefill_tokens,num_decode_tokens,session_id
0.0,32,16,7
100.0,8,8,7
1.0,32,16,8
101.0,8,8,8
```

Expected behavior, assuming no eviction:

- First turn of each session: zero hit blocks.
- Second co-location turn: three hit blocks / 48 cached tokens.
- Second PDD turn: two hit blocks / 32 cached tokens.
- No block from session 7 satisfies a lookup for session 8.

Run:

- focused unit tests,
- co-location runtime smoke tests,
- sequential PDD runtime smoke tests, and
- existing block-hash prefix-cache example tests as regression coverage.

### Implemented regression verification

The correctness-hardened MVP includes focused coverage for:

- per-session incremental ISL materialization and independent session context;
- explicit Thinking Mode round materialization (`48 -> 96` effective ISL);
- Thinking-round scale factors below and above one, including shared integer
  truncation with the top-level CSV values;
- fail-fast rejection of nonfinite/negative arrivals, nonfinite/nonpositive
  token counts, and positive scaled values that integerize to zero;
- request/system cache totals spanning hidden and final Thinking rounds
  (`3 + 6 = 9` query blocks), without preemption double-counting;
- expanded-context overflow rejection without silent clipping;
- one-replica/two-DP affinity and equal follow-up cache hits on both lanes;
- PDD `sticky_lor` fail-fast validation with `sticky_round_robin` documented as
  the supported sequential PDD affinity scheduler;
- same-session requests arriving at the same time and entering one scheduling
  window;
- publication only at the completed chunk frontier;
- zero storage-key generation across repeated unchanged ready frontiers for a
  128K-token request, followed by one-key generation for a one-block advance;
- admission-scoped diagnostic counters that ignore speculative lookup retries;
- real vLLM V1 preemption state reset followed by cache-based re-admission;
- fully cached prompt one-block recomputation across vLLM V1, SGLang, and all
  three SJ2Q variants;
- reuse of a cached `ref_cnt == 0` block;
- co-location and sequential PDD hit boundaries; and
- the existing explicit block-hash runtime path.

## Implementation Sequence

1. Add the configuration field and validation.
2. Split trace validation by key mode.
3. Add the KV manager key provider and separate lookup/storage key sequences.
4. Update scheduler admission and request lookup metadata.
5. Update metrics to use recorded lookup results.
6. Add manager and scheduler unit tests.
7. Add co-location and PDD end-to-end fixtures.
8. Update examples and CLI documentation.
9. Run existing prefix-cache and release smoke suites.

Each step should leave `block_hash` mode operational.

## Acceptance Criteria

The feature is complete when:

1. A trace containing only `arrived_at`, ISL, OSL, and `session_id` can run
   with prefix caching enabled.
2. Two turns in the same append-only session reuse only locally available full
   prefix blocks.
3. Different sessions never share a session-keyed block.
4. Co-location and PDD report their architecture-appropriate hit boundaries.
5. Request CSV and system JSON metrics report nonzero query counts without
   `block_hash_ids`.
6. Incremental session ISLs, including Thinking Mode rounds, are materialized
   into effective prompts and expanded overflow fails before scheduling.
7. Existing explicit `block_hash_ids` traces and examples continue to pass
   unchanged.

## Follow-up Work

Potential follow-up proposals:

- closed-loop multi-turn generation using `think_time`,
- explicit turn IDs and parent-turn dependencies,
- context truncation metadata and retained-range modeling,
- branching-session identifiers,
- decode-to-prefill KV return in PDD,
- remote or hierarchical session KV caches,
- session cache TTL and explicit session termination,
- per-session cache residency and eviction metrics.
