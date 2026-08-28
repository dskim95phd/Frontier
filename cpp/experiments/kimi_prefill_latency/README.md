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

## LMSYS Day-0 TEP8 comparison

The LMSYS Kimi K3 Day-0 prefill result uses TEP8: attention TP8 and MoE EP8
over the same eight GB300 GPUs. Reproduce that topology, compare its published
8K-prefill concurrency sweep, and also isolate TP8/EP8 versus the earlier
DP8/EP8 warm-prefix setup with:

```bash
python cpp/experiments/kimi_prefill_latency/run_lmsys_tep8_compare.py \
  --binary build-release/frontier_sim
```

The published curve saturates at concurrency 16. The comparison therefore
uses a 131,072-token simulator batch budget (16 requests × 8,192 tokens) and
records that inference explicitly in `comparison.json`; it is not a published
SGLang command-line value.

## TP8/EP8 precision-matched K2 versus K3

To isolate architecture from the released checkpoint precision policy, run
both models at TP8/DP1/EP8 with K2-equivalent FP8/FP4 operator precision. K3
uses the isolated FlashKDA prefill profile; the broader SGLang efficiency and
fusion profile is deliberately excluded:

```bash
python cpp/experiments/kimi_prefill_latency/run_tp8_precision_matched_compare.py \
  --binary build-release/frontier_sim
```

The runner emits full diagnostics and fails unless the effective element size
of every modeled operator family is identical between K2 and K3.

Pass `--batch-size 16` to measure sixteen independent warm-prefix sessions in
one batch. Each session contributes one 1,024-token chunk after its own 128K
cached prefix, so the measured batch contains 16 requests and 16,384 prefill
tokens in total.
