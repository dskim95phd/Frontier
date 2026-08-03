# Frontier C++ examples

These examples are user-facing recipes. Unlike files under `cpp/tests`, each
one starts from a serving question and points to the metrics that answer it.

Run an example from the repository root:

```powershell
python cpp/examples/run_example.py hello `
  --binary .build-step4/frontier_sim.exe
```

Use `--output-mode summary`, `requests`, or `full`. The default `requests`
writes normalized inputs, `summary.json`, and `requests.csv` under
`outputs/cpp_examples/<example>/`. `full` additionally writes the complete
deterministic trace as `trace.json`.

## Recipes

| Name | Question | Important settings | Inspect |
| --- | --- | --- | --- |
| `hello` | Does the simulator and output contract work end to end? | Fixed timing, one replica, TP1/PP1/DP1 | `requests.csv`, batch histogram |
| `dense-analytical` | How does a dense online service behave on a modeled device? | Rubin analytical model, TP4/PP2/DP2 | throughput, TTFT/TPOT percentiles |
| `kv-pressure` | What happens when scheduler KV capacity is tight? | Two DP lanes, only two blocks per lane | preemption count, E2E tail |
| `pdd` | What does Prefill/Decode separation cost? | Separate clusters and analytical KV transfer | prefill latency versus TTFT, transfer latency |
| `moe` | How does expert routing interact with EP? | EP4, top-2 Zipf routing | MoE routing trace and batch execution |
| `prefix-cache` | How much work is reused by later session turns? | Session cache and sticky round robin | hit rate, cached tokens, target affinity |

## Latency definitions

- `prefill`: request arrival to Prefill completion.
- `ttft`: request arrival to completion of the first generated token.
- `tpot`: time after the first token divided by the remaining generated tokens.
- `e2e`: request arrival to completion of the full response.

For PDD, TTFT includes KV-cache transfer and the first Decode execution. This
is intentionally different from Prefill latency.
