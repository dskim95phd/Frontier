# Kimi K2/K3 warm-prefix PREFILL latency

This experiment compares one 1,024-token PREFILL after an exactly 128K-token
cached context. Both models use one hypothetical 8-GPU Rubin node with 500 GB
HBM per GPU, TP1/DP8/EP8, batch size cap 1, and a 1,024-token chunk/batch
budget. The first request only warms the session cache; the second request is
the measured point.

Build and run a Release simulator:

```bash
cmake -S cpp -B build-release -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
cmake --build build-release -j
python cpp/experiments/kimi_prefill_latency/run_compare.py \
  --binary build-release/frontier_sim
```

The script validates that the measured request has 131,072 cached tokens,
1,024 scheduled PREFILL tokens, and a 132,096-token materialized context. It
writes `comparison.json` beside the two simulator output directories.
