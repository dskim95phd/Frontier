<div align="center">

<img src="figs/icon-frontier-transparent.png" alt="Frontier logo" width="300" />

# Frontier

<h4>A Discrete-Event Simulator for Modern LLM Serving</h4>

[![docs](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](./docs)
[![version](https://img.shields.io/badge/release-v0.2.0-green)](#latest-news-)
[![license](https://img.shields.io/badge/license-MIT-blue)](./LICENSE)
[![arxiv](https://img.shields.io/badge/arXiv-2605.21312-b31b1b.svg)](https://arxiv.org/abs/2605.21312)

<div align="left">

## Latest News 🎯
📍[2026/07] We refactored the operator registration module to improve support and integration for diverse models and attn algorithms. More examples will be provided, including how to use Frontier for end-to-end simulation of a new/customized model.<br />
📍[2026/06] Prefill-Decode Disaggregation (PDD) version released! Support for Attention-FFN Disaggregation (AFD) will be available soon.<br />
📍[2026/06] Initial version released, with support for co-located serving and modern optimizations.<br />

## Frontier Overview

Frontier is a discrete-event simulator for modern LLM serving. It is built for serving systems that combine complex parallelism, runtime optimizations, sparse model architectures (MoE), and stateful workloads (reasoning agents, RL rollouts). It currently simulates vLLM-style serving behavior, and we plan to include other serving engines soon.

Frontier helps researchers and engineers better understand serving system designs and tradeoffs without the time and financial costs of repeatedly deploying on GPU clusters.

<div align="left">
  <img src="figs/arch.png" alt="Frontier system architecture" width="760" />
</div>

### Key Features

- **Co-located & Disaggregated Serving**: The primary C++ core supports
  monolithic co-location and sequential PDD serving. AFD is not part of this
  release.
- **Modern Runtime Optimizations**: The C++ core models prefix caching,
  chunked prefill, recompute preemption, quantized analytical operators, and
  tiered CPU KV-cache behavior inside the scheduler-batch-engine loop. The
  retained Python path provides the experimental speculative decoding/MTP,
  learned predictors, and profiling workflows described below.
- **Fidelity**: Frontier combines operator, communication, transfer, and
  KV-cache memory models with deterministic event ordering. This supports
  repeatable SLA and design-space comparisons without reducing behavior to
  coarse average speedup factors.

> AFD serving architecture is intentionally not included in this release and will be available in a later public release.

## Minimum Hardware Requirements

- **C++ simulation:** CPU-only; fixed latency and analytical roofline modes do
  not require a GPU or profiling database.
- **Python profiling:** At least **1 GPU** is required only when collecting new
  operator-level performance data.

## Use Cases

Frontier is designed for what-if studies that would be expensive or slow to run directly on large GPU clusters. The current paper draft demonstrates four use cases.

<table align="center">
  <tr>
    <td width="50%" align="center" valign="top">
      <strong>SLA-Aware Pareto Frontier Search</strong><br />
      Find the best serving architecture and parallelism configuration under TTFT and generation-speed constraints.<br /><br />
      <img src="figs/use_case_pareto_frontier.png" alt="SLA-aware Pareto frontier search" height="220" style="object-fit: contain; max-width: 100%;" />
    </td>
    <td width="50%" align="center" valign="top">
      <strong>Heterogeneous GPU Allocation</strong><br />
      Study when disaggregated role placement can turn cheaper GPU types into real cost efficiency while still meeting SLA targets.<br /><br />
      <img src="figs/use_case_heterogeneous_gpu_allocation.png" alt="Heterogeneous GPU allocation for disaggregated serving" height="220" style="object-fit: contain; max-width: 100%;" />
    </td>
  </tr>
  <tr>
    <td width="50%" align="center" valign="top">
      <strong>Stateful Reasoning Scheduler Validation</strong><br />
      Validate scheduling policies for multi-round reasoning workloads with hidden planning, tool calls, and prefix-cache continuity.<br /><br />
      <img src="figs/use_case_stateful_reasoning_scheduler.png" alt="Stateful reasoning scheduler validation" height="220" style="object-fit: contain; max-width: 100%;" />
    </td>
    <td width="50%" align="center" valign="top">
      <strong>Dynamic Reconfiguration for RL Rollouts</strong><br />
      Test whether switching parallelism layouts during rollout execution can reduce long-tail makespan before implementing it in a production stack.<br /><br />
      <img src="figs/use_case_rl_rollout_reconfiguration.png" alt="Dynamic parallelism reconfiguration for RL rollouts" height="220" style="object-fit: contain; max-width: 100%;" />
    </td>
  </tr>
</table>

## Quick Start and Examples

The deterministic C++ core under `cpp/` is the primary implementation. It
requires CMake 3.24+, Ninja, and a C++17 compiler (GCC 11+, Clang 14+, or a
current MSVC toolchain):

```bash
cmake -S cpp -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build
ctest --test-dir build --output-on-failure

./build/frontier_sim \
  --config cpp/examples/configs/00_hello_colocation_fixed.json \
  --workload cpp/examples/workloads/00_tiny.csv \
  --output-dir outputs/hello \
  --output-mode requests
```

Preset users can run `cmake --preset dev`, `cmake --build --preset dev`, and
`ctest --preset dev` from `cpp/`. To create a relocatable release archive,
configure the `release` preset and run `cpack --config ../build/release/CPackConfig.cmake`
from `cpp/`.

The runnable C++ recipes cover dense analytical serving, KV pressure, MoE,
PDD, session prefix caching, and CPU KV-cache tiering. See
[`cpp/examples/README.md`](cpp/examples/README.md).

The Python simulator under `frontier/` is retained for capabilities not yet in
C++, including speculative decoding/MTP, learned execution-time predictors,
GPU profiling, topology-aware communication backends, SGLang scheduling, and
Thinking Mode. Its examples remain under `examples/`; see
[`docs/cli/README.md`](docs/cli/README.md) before using that path.

For contributor setup, supported test gates, and pull-request expectations,
see [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Communication Backend

The release example suites default to the lightweight formula-based backend for one-click smoke runs:

```bash
--cc_backend_config_type analytical
```

The ASTRA-Sim-inspired topology model remains available for direct CLI experiments:

```bash
--cc_backend_config_type astra_sim_analytical
```

`collective_sim` is optional and is used only when you explicitly select `--cc_backend_config_type collective_sim`.

To enable the optional target-runtime backend:

```bash
git submodule update --init --recursive frontier/cc_backend/backends/collective-sim
cd frontier/cc_backend/backends/collective-sim/sim
make -j"$(nproc)"
```

After building, verify that `frontier/cc_backend/backends/collective-sim/sim/datacenter/htsim_ndp` exists before selecting `--cc_backend_config_type collective_sim`.

If your host C++ runtime reports a GLIBC or GLIBCXX mismatch, rebuild the optional backend in the target runtime:

```bash
make -B -j"$(nproc)"
```

## Environment and Docker

For the standard simulator environment, install from `environment.yml` or use editable `pip` installation. Profiling wrappers use a dedicated environment because vllm and flashinfer can be sensitive to CUDA and Python versions:

```bash
conda env create -f environment_profiling.yml
conda activate frontier-profiling
```

Do not blindly install profiling dependencies into an existing environment unless you have confirmed version compatibility.

Production Docker users can start from the public image:

```bash
docker pull fengyicheng/frontier-env
docker run --rm --gpus all --shm-size 16g \
  --tmpfs /workspace/frontier/outputs \
  --tmpfs /workspace/frontier/cache \
  -v "$PWD":/workspace/frontier \
  -w /workspace/frontier \
  fengyicheng/frontier-env bash
```

## Plan

We will continuously release Frontier's core components to the community. We are actively developing new features and plan to support:

- **Serving Engines Integration**: Support for SGLang and TensorRT-LLM frameworks.
- **Advanced Caching**: Support for `tair-kvcache` as an advanced runtime backend for the Hierarchical Caching feature.
- **Advanced Model Support**: Expanded support for state-of-the-art models such as DeepSeek-V4, Kimi 2.5, and more.
- **Simulation Acceleration**: Introduce more simulation acceleration mechanisms.

## Acknowledgments

Frontier is mainly built on top of Vidur. The following great systems have been referenced or adapted as runtime backends. We sincerely thank all the developers for their contributions to the community!

- [**Vidur**](https://github.com/microsoft/vidur)
- [**ASTRA-Sim**](https://github.com/astra-sim/astra-sim)
- [**htsim**](https://github.com/Broadcom/csg-htsim)
- [**AIConfigurator**](https://github.com/ai-dynamo/aiconfigurator)

## Citation

If you use this repo in your research, please cite our arXiv paper:

```bibtex
@article{feng2026frontier,
  title={Frontier: Towards Comprehensive and Accurate LLM Inference Simulation},
  author={Feng, Yicheng and Tan, Xin and Deng, Yangtao and Jiang, Yimin and Zhu, Yibo and Xu, Hong},
  journal={arXiv preprint arXiv:2605.21312},
  year={2026}
}
```

## Contact

Email **Yicheng Feng** (<yichengfeng@link.cuhk.edu.hk>) or **Hong Xu** (<hongxu@cuhk.edu.hk>) if you have any questions.

Feedback, issues, and PRs are highly welcome. We welcome your contributions!

<div align="left">
  <img src="figs/lark.png" alt="Lark" width="100" />
</div>

## License

This project is licensed under the MIT license - see the [LICENSE](LICENSE) file for details.
