# Kimi K2 / Vera Rubin TraceLab baseline

This baseline mirrors the latest GB300 TraceLab experiment topology while
replacing the hardware contract with NVIDIA Vera Rubin.

## Frozen configuration

The runnable config is:

`cpp/experiments/kimi_k2_cpu_dram/configs/tracelab_vera_rubin_p8_d24_cache_aware.json`

| Item | Setting |
| --- | --- |
| Model | `moonshotai/Kimi-K2-Instruct` |
| Architecture | Sequential online PDD, one replica |
| PREFILL | 8 Rubin GPUs, TP1 / DP8 / EP8 |
| DECODE | 24 Rubin GPUs, TP4 / DCP4 / DP6 / EP24 |
| PREFILL scheduling | vLLM V1, FCFS, 1,024-token chunked PREFILL |
| Batch cap | 256 requests |
| PREFILL token cap | 131,072 tokens/batch |
| DECODE token cap | 8,192 tokens/batch |
| Routing | Cache-aware PREFILL and vLLM-like queue-aware DECODE |
| Compute precision | FP8 except FP4 MoE expert weights |
| CPU KV offload | Disabled in the baseline |

## Rubin hardware contract

The analytical `rubin` preset uses dense, non-sparse ceilings for training and
roofline calculations. The 50 PFLOPS NVFP4 inference number is not used because
NVIDIA labels it as sparse; the simulator uses the 35 PFLOPS dense NVFP4
training ceiling.

| Resource | Per Rubin GPU or Vera CPU |
| --- | ---: |
| Rubin HBM4 | 288 GB |
| Rubin HBM4 bandwidth | 22 TB/s |
| Rubin FP8/FP6 dense | 17.5 PFLOPS/GPU |
| Rubin NVFP4 dense | 35 PFLOPS/GPU |
| Rubin FP16/BF16 dense | 4 PFLOPS/GPU |
| Rubin FP32 | 130 TFLOPS/GPU |
| Rubin NVLink 6 | 3.6 TB/s/GPU = 28,800 Gbps |
| Vera LPDDR5X capacity | 1.5 TB/CPU |
| Vera LPDDR5X bandwidth | 1.2 TB/s/CPU |
| Vera NVLink-C2C | 1.8 TB/s/CPU |

One Vera CPU serves two Rubin GPUs in a Vera Rubin Superchip. Static CPU KV
slices therefore use 750 GB capacity, 600 GB/s DRAM bandwidth, and 900 GB/s
C2C bandwidth per GPU. CPU KV remains disabled until an offload experiment
explicitly enables it.

The config supplies 288 GB of physical HBM per GPU and reserves 10% for runtime
workspace. It does not set `scheduler.num_blocks`: the simulator derives the
maximum rank-local model-weight footprint from Kimi K2, the cluster
parallelism, and the operator weight precisions, then gives the remaining HBM
to KV blocks:

```text
KV budget = 288 GB - model weights/GPU - 28.8 GB runtime reserve

PREFILL TP1/EP8:
  weights = 73,818,731,520 B
  KV budget = 185,381,268,480 B
  block = 624,640 B
  blocks/GPU = floor(185,381,268,480 / 624,640) = 296,780

DECODE TP4/DCP4/EP24:
  weights = 25,547,535,360 B
  KV budget = 233,652,464,640 B
  rank-local block = 156,160 B
  blocks/GPU = floor(233,652,464,640 / 156,160) = 1,496,237
```

These resolved values are also written to `config.normalized.json`, so a
change to quantization, TP/EP/DCP, model shape, block size, or HBM capacity is
visible without maintaining a second hand-calculated block constant.

## Run

The existing TraceLab runner accepts the Rubin config without changing the
workload:

```powershell
python cpp/experiments/kimi_k2_cpu_dram/run_tracelab_arrival_sweep.py `
  --binary cpp/build/Release/frontier_sim.exe `
  --config-template cpp/experiments/kimi_k2_cpu_dram/configs/tracelab_vera_rubin_p8_d24_cache_aware.json `
  --rates 0.5 `
  --sample-sessions 1000 `
  --output-root outputs/tracelab_vera_rubin_r0p5
```

The NVIDIA values are preliminary and should be revisited if the published
Rubin platform specifications change.

## Sources

- [NVIDIA Vera Rubin NVL72 specifications](https://www.nvidia.com/en-us/data-center/vera-rubin-nvl72/)
- [NVIDIA Rubin GPU architecture](https://developer.nvidia.com/blog/inside-nvidia-rubin-gpu-architecture-powering-the-era-of-agentic-ai/)
- [NVIDIA Vera CPU specifications](https://www.nvidia.com/en-gb/data-center/vera-cpu/)
