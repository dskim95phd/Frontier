
## Probabilistic workload

Every JSONL record is one append-only session with
`reuse_previous_kv: true`. A later turn's input is derived from the previous
turn rather than sampled independently:

```text
input[t + 1] = input[t] + output[t] + new_context[t]
```

This makes the declared reuse opportunity internally consistent without token
IDs. The primary workload uses the following bounded distributions:

| Variable | Distribution |
| --- | --- |
| Session arrivals | Poisson process; exponential inter-arrival time |
| Turns per session | `min(12, 2 + Geometric(p=0.25))` |
| Initial input | log-normal, median 1,024, sigma 0.8, range 256-8,192 tokens |
| Output per turn | log-normal, median 128, sigma 0.7, range 16-1,024 tokens |
| New context per turn | log-normal, median 256, sigma 0.9, range 16-4,096 tokens |
| Maximum context | 32,768 tokens; stop the session before exceeding it |

The mixed gap profile samples each non-terminal pause from:

| Class | Probability | Log-normal median | Sigma | Bounds |
| --- | ---: | ---: | ---: | ---: |
| Fast tool | 65% | 0.2 s | 0.8 | 0.01-2 s |
| Slow tool | 25% | 5 s | 1.0 | 0.5-60 s |
| Human thinking | 10% | 120 s | 0.8 | 10-600 s |

The last turn always has `tool_duration_ns: 0`. The generator writes a summary
manifest containing its arguments and realized p50/p90/p99 values. The same
generated workload and seed must be reused across every policy and capacity in
a comparison.

Context truncation and summarization are excluded from the primary experiment.
A later sensitivity sweep may set `reused_prefix_toks` on 10% or 30% of turns.

The workload generator also exposes `--context-profile long` for a controlled
long-context sensitivity run. It changes the initial-context distribution to
a log-normal median of 8,192 tokens (range 4,096-16,384) and the per-turn new
context distribution to a median of 2,048 tokens (range 256-8,192). The same
32,768-token context ceiling, turn-count distribution, output distribution,
arrival process, and gap profile are retained. Use a separate experiment run
root because context profile is part of workload identity.

For a follow-up capacity refinement that reuses completed baselines, pass
`--no-include-baselines` to `confirm`. Only the requested Session-offload
capacities are scheduled; the existing summary still includes earlier records
from the same run root.
