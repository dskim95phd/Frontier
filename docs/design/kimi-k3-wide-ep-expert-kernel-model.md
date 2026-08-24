# Kimi K3 wide-EP expert-kernel timing model

Status: implemented

Date: 2026-08-18

Retuned: 2026-08-22

Applies to: C++ analytical predictor, `device="gb300"`, Kimi K3,
`kernel_profile="k3_deepgemm_megamoe"`

## Decision

Frontier models the destination-side Kimi K3 wide-EP expert kernel with a
hybrid tile/grid-aware roofline. The old model multiplied the recomposed
expert path by a constant 1.75. The replacement derives its cost from the
actual destination-lane expert-token histogram:

1. select a DeepGEMM-style SM100 block-M from expected tokens per expert;
2. round every active expert's M dimension independently to that block-M;
3. charge padded tensor-core FLOPs and padded activation HBM traffic;
4. derive the up/down two-CTA cluster grids and final-wave utilization for
   diagnostics, without a separately fitted wave multiplier;
5. add an empirical cost proportional to cluster-grid size.

The precision-correct grid coefficient is `0 us` per routed two-CTA cluster
task. The former `0.06325 us` fit is superseded: it was obtained while FP32
router logits incorrectly selected the FP32 CUDA-core ceiling for the BF16
router GEMM and while one common communication dtype drove both dispatch and
combine. Router, latent projection, shared expert, normalization, and finalize
work are evaluated once per DP source with that source's local token count;
only routed destination-expert work is recomposed at the EP barrier.

This is deliberately a reduced RaMP-style model rather than a complete RaMP
implementation. Frontier's existing roofline already accounts for kernel
launch, arithmetic, and bulk HBM traffic. Reintroducing all four fitted RaMP
terms would double count some of that work and would require a GB300 kernel
profiling matrix that is not currently available.

## Why the aggregate roofline was insufficient

The portable MoE roofline starts from total routed-token work and the number
of active expert weights. That is useful for broad device comparisons, but it
cannot distinguish these equal-sum histograms:

```text
A = [4, 4, 4, 4, 4, 4, 4, 4]
B = [32, 0, 0, 0, 0, 0, 0, 0]
```

Both contain 32 expert-token routes. The aggregate roofline sees different
active weight bytes, but it does not explicitly see eight small M tiles versus
one larger expert GEMM. Real grouped kernels see:

- per-expert M padding;
- a discrete number of CTA/cluster tasks;
- partially occupied final SM waves;
- scheduler and pipeline-fill work that grows with grid size.

This omission is especially visible for K3: it has 896 routed experts and
top-k 16, so wide EP produces many very small per-expert M dimensions during
decode.

The prior `1.75` multiplier compensated for the missing cost at the serving
curve level, but it had three drawbacks:

- it could not respond to a different expert histogram with the same total;
- it scaled shuffling even though the missing phenomenon was expert-kernel
  geometry;
- it offered no useful extrapolation axis for batch size, routing skew, or EP.

## Public evidence used

The model combines three public observations:

- DeepGEMM's contiguous grouped-GEMM contract aligns each expert segment to
  the kernel M block, and its SM100 runtime selects an M alignment from the
  expected per-group M. This motivates exact per-expert padding and the
  block-M policy used here.
- SonicMoE shows that expert-frequency rounding to GEMM tile boundaries is a
  material performance effect. This supports treating the expert histogram,
  rather than only its sum, as a timing input.
- RaMP models fused-MoE performance from runtime CTA-grid geometry with
  physically distinct startup, wave, per-CTA, and sub-wave terms. This
  motivates testing cluster-grid terms after Frontier's roofline has already
  accounted for arithmetic and memory work; the factorial test ultimately
  rejected a separately identifiable wave term.

References:

- DeepGEMM: <https://github.com/deepseek-ai/DeepGEMM>
- DeepGEMM grouped API: <https://github.com/deepseek-ai/DeepGEMM/blob/main/csrc/apis/gemm.hpp>
- RaMP: <https://arxiv.org/abs/2604.26039>
- SonicMoE: <https://arxiv.org/abs/2512.14080>

These sources establish the functional form. They do not provide an
independent GB300 K3 value for every coefficient.

## Model definition

### Inputs and scope

For one DP-source-composed MoE group, let:

- `T` be the sum of input tokens whose routed work is recomposed in the group;
- `T_s` be the local input-token count of DP source `s`;
- `k` be router top-k;
- `E` be the model's total routed-expert count;
- `m_e` be the exact routed-token count for local expert `e` on a destination
  EP lane;
- `S` be GPU SM count;
- `C_s` be CTA cluster size;
- `B_N` be the output-column tile width.

For the implemented GB300 MegaMoE profile:

```text
S   = 160 SMs
C_s = 2 CTAs per cluster
B_N = 128 columns per CTA
```

### Rubin resource-ratio projection

Rubin is not assigned one blanket speedup. Each term follows the hardware
resource that dominates it:

| Model term | GB300 → Rubin factor | Projection rule |
| --- | ---: | --- |
| Tensor-core arithmetic | precision-specific device ceiling | handled once by the ordinary roofline |
| HBM traffic | `22 / 8 = 2.75x` bandwidth | handled once by the ordinary roofline |
| Wave capacity | `224 / 160 = 1.4x` SMs | set `mega_moe_sm_count=224` |
| A2A payload | `3.6 / 1.8 = 2x` NVLink bandwidth | double the public effective payload bandwidth |
| Per-cluster coefficient | precision-correct GB300 fit selected zero | retain `0 us`; nonzero values are explicit sensitivities |
| A2A startup | no public generational latency ratio | retain the GB300 public prior |
| Communication/compute overlap | producer-consumer dependency | retain the measured wide-EP `1.0` residual; report optimistic overlap only as sensitivity |

The 35-PFLOP/s Rubin value is Frontier's conservative NVFP4 training ceiling;
the simulator deliberately does not mix it with the public 50-PFLOP/s
inference maximum. Peak NVLink bandwidth scales only the payload term. It does
not reduce fixed dispatch/combine startup, and it does not change overlap by
itself. Rubin counted writes and tile-level dependent triggering motivate a
future sensitivity case but do not provide a numeric K3 MegaMoE overlap value.
The former throughput-per-SM projection is superseded. Explicit nonzero
cluster-task-latency overrides remain available for sensitivity studies, not
as the Rubin default. This separation prevents applying the same hardware gain
twice or silently treating throughput as a latency measurement.

The geometry model is enabled only in `predict_moe_group_layer()`, after all
active DP source rows have been composed into the destination EP lanes. That
prediction uses a routed-only model with zero shared experts. In the same
single predictor pass, every source/lane prediction also emits an explicit
`source_local_ms` made from gating, routing, shared-expert, normalization,
latent-projection, and attention-residual work evaluated at that source's
local token count, plus the shared-expert component used by the communication
overlap window. The scheduler carries that component into the group predictor;
the group barrier does not predict the full MoE layer again per source. The
scheduler forms the lane-aligned critical path:

```text
T_group = max_l (R_l + max_s N_s,l)
```

There is no longer a `max(full) - max(routed)` subtraction, launch-latency
cancellation, or non-negative clamp. This matters because evaluating
source-local work on `sum_s T_s` incorrectly multiplies shared/router/latent
work by DP size and lets the fitted grid coefficient absorb that topology
error. Preserving the lane index also avoids the looser
`max_l R_l + max_s,l N_s,l` upper bound.

### 1. Block-M selection

One block-M is selected for the grouped call from expected tokens per expert:

```text
mu = T * k / E

mu <=  8.5  -> B_M =  16
mu <= 16.5  -> B_M =  32
mu <= 32.5  -> B_M =  64
mu <= 64.5  -> B_M =  96
mu <= 96.5  -> B_M = 128
otherwise   -> B_M = 192
```

The expectation chooses a call-level kernel alignment; it does not replace
the runtime histogram. Every expert is still padded from its exact `m_e`.

### 2. Per-expert padding

For each active routed expert:

```text
q_e       = ceil(m_e / B_M)
m_e_pad   = q_e * B_M
Q_routed  = sum_e q_e
```

The ordinary roofline work is built twice: once with actual token counts and
once with padded counts. The MegaMoE work passed to the roofline is:

```text
F_effective = F_padded
B_effective = B_actual + rho_tail * (B_padded - B_actual)
rho_tail    = 1
```

Thus tensor-core work and activation IO include padded rows. Active expert
weights remain charged by the ordinary roofline rather than being multiplied
by the padding ratio. A four-corner factorial refit selected full tail IO over
the former masked-tail assumption.

### 3. Two-CTA cluster grid

A two-CTA cluster covers:

```text
N_cluster = C_s * B_N = 256 output columns
```

For projection `p` and expert family `f`:

```text
n_clusters(p, f) = ceil(N(p, f) / N_cluster)
g(p, f)          = Q_f * n_clusters(p, f)
```

The group up/down grid sizes contain routed cluster tasks only. For gated up
projections, `N_up` is twice the local intermediate size. Routed down
projection uses K3's routed expert hidden size when present. Replicated shared
experts stay in each source-local prediction and never contribute a
destination-group grid.

This makes timing sensitive to all of the intended axes:

- more active experts increase `Q_routed`;
- fragmented expert counts can add M blocks without changing total routes;
- larger batches eventually select a larger block-M;
- increasing DP changes the routed destination histogram without multiplying
  source-local shared/router/latent work.

### 4. Final-wave utilization

The number of two-CTA clusters that fit in one wave is:

```text
C_wave = floor(S / C_s) = 80
```

For a projection grid of `g` cluster tasks:

```text
waves = ceil(g / C_wave)
u     = g / (waves * C_wave)
phi   = 1 + lambda_wave * (1/u - 1)
```

The selected profile uses `lambda_wave = 0`, so `u` and `phi` remain geometry
diagnostics but do not add a second timing penalty:

```text
T_wave = T_launch + (T_roofline - T_launch) * phi
       = T_roofline
```

When an additive per-grid-task residual was independently refit, all four
`(rho_tail, lambda_wave)` corners selected statistically equivalent fits and
the precision-correct model selected the zero boundary. The term was therefore
not identifiable and has been removed rather than retained as a dormant knob.

### 5. No fitted grid-size residual

The final projection time is simply:

```text
T_projection = T_wave
```

The earlier empirical scheduler/pipeline coefficient was fitted while router
and communication precisions were wrong. The corrected fit did not identify a
positive value, so both the coefficient and its configuration/diagnostic fields
were deleted. Consequently:

- an empty or small grid does not inherit a blanket 1.75 penalty;
- a fragmented or larger grid costs more;
- shuffling remains governed by its streaming roofline;
- the cost can be replaced by future microbenchmark coefficients without
  changing the geometry model.

Shared-expert precision remains part of the source-local portable roofline;
the destination-group geometry always uses the routed-expert precision.

## Why this reduced model was selected

The alternatives considered were:

| Alternative | Advantage | Reason not selected |
| --- | --- | --- |
| Aggregate roofline only | Minimal and portable | Cannot represent expert-M fragmentation or wave steps |
| Constant 1.75 scale | Matched the original LMSYS wide-EP points | No histogram dependence; scaled unrelated shuffling |
| Tile padding only | No fitted parameter | Still omitted grid scheduler/pipeline cost and made wide EP much too fast |
| Full RaMP four-parameter fit per kernel config | Best-supported kernel-dispatch model | Requires a GB300 profiling matrix and a much larger configuration surface; overlaps existing roofline terms |
| Exact kernel simulator | Highest potential fidelity | Too complex and backend/version specific for the DES critical path |
| Chosen hybrid | Exact histogram geometry with one residual coefficient | Retains one calibrated GB300/K3 parameter |

The chosen model is the smallest one that passed both requirements:

1. two expert histograms with equal aggregate work can produce different
   predicted times for a physical reason;
2. the LMSYS wide-EP curves remain close without a full-path scale factor.

## Superseded 2026-08-18 calibration

The numbers in this section document the former mixed-precision-invalid fit
and must not be used as the current calibration. The 2026-08-22 retune uses
MXFP4/MXFP8 routed experts, BF16 non-expert operators/KV/combine, FP32 KDA
state and router logits, MXFP8 post-quant dispatch, and the BF16 operand
ceiling for the router GEMM. It selected no additive grid residual on DP4/EP32: user MAPE
0.19%, throughput MAPE 26.28% before the LatentMoE-front fidelity fix. After
modeling the published fused gate/down GEMM and its small-M TGV geometry, the
same points are 1.07% and 28.29%. The untouched DP2/EP16 holdout moves from
8.06%/27.50% to 8.11%/30.07%. The common throughput error is a separate
prefill-model error and is not identifiable from expert-grid timing or the source-local
front GEMM.

## Historical calibration and validation

The development sequence is important because it exposes what each component
contributes.

| Model | Mean log distance | User MAPE | Throughput MAPE | Log-user Pearson r |
| --- | ---: | ---: | ---: | ---: |
| Previous 1.75 constant | 0.15395 | 12.39% | 6.88% | 0.97606 |
| Exact padding + tail wave only | 0.16982 | 14.25% | 7.17% | 0.96273 |
| Factorial-selected, full 140-case rerun | **0.14990** | **11.95%** | **6.81%** | **0.97670** |

The tile-only failure is evidence that padded FLOPs and final-wave occupancy
were not enough: the aggregate roofline still lacked a grid-size cost.

The former additive grid coefficient was selected using only the three LMSYS
DP4/EP32 measurements. The two DP2/EP16 measurements were kept out as a
topology holdout:

| Wide-EP arm | Role | User MAPE | Throughput MAPE |
| --- | --- | ---: | ---: |
| DP4 / EP32 | calibration, 3 points | **0.23%** | **0.03%** |
| DP2 / EP16 | holdout, 2 points | **0.31%** | **0.03%** |

The holdout is still not independent hardware validation: both arms come
from the same LMSYS K3 serving study. The full 69-point Pearson correlations
remain high because only 5 wide-EP measurements move and the original global
curve ordering was already strong:

```text
log user speed:   0.97670
log throughput:   0.99703
joint log:        0.99832
```

Absolute log distance and MAPE are therefore the more informative metrics for
this change; Pearson correlation mainly confirms that the overall topology
ordering was preserved.

The two retained physical sensitivity coefficients are exposed as optional
analytical execution configuration fields:

```text
mega_moe_tail_io_fraction
mega_moe_wave_exposure
```

They are accepted only with `kernel_profile="k3_deepgemm_megamoe"`.
The former three-factor fit is retained only as historical evidence that the
additive coefficient was confounded: all four tail/wave corners produced
effectively equivalent DP4 errors. With corrected precision the optimum moved
to zero, leaving `rho_tail=1, lambda_wave=0` and no fitted latency constant.

The generated comparison artifact is:

`outputs/k3_lmsys_gb300_serving_frontier_20260816/report_model_v14_factorial_selected.html`

Under the former throughput-per-SM Rubin sensitivity, the four equally
calibrated GB300 corners remain close: their per-user decode spread is about
0.8% for DP4 and 2.3% for DP2 at c=256/c=1024. Those results now define an
optimistic boundary rather than the default projection. Applying the former
GB300 fitted residual reduced the wide-EP Rubin median per-user decode speed by
8.0% relative to that boundary; the reduction is about 10--11% for DP4 and
14% for DP2 at c=256/c=1024. The conservative full-sweep artifact is:

`outputs/k3_lmsys_rubin_serving_frontier_20260817/report_rubin_frontier_latency_conservative_v15.html`

Neither boundary replaces a Rubin measurement.

## Compatibility and invariants

- `generic` and `k3_sglang_mxfp4` profiles keep their previous behavior.
- The model is opt-in and K3-only. GB300 is the calibrated device; Rubin use
  is an unvalidated projection that maps each term to its dominant resource:
  arithmetic/HBM use Rubin roofline ceilings and A2A payload uses the 2x
  per-GPU NVLink ratio. The default cluster-task residual is zero on both
  devices. Geometry records 224 SMs, but
  `lambda_wave=0` means that count does not add a separate wave timing
  multiplier. Fixed startup, cluster-task latency, and wide-EP overlap are not
  scaled without a Rubin K3 measurement. Nonzero coefficients are explicit
  sensitivities only.
- Routed and source-local timing are emitted in one lane pass; enabling the
  breakdown does not double the number of MoE-layer roofline evaluations.
- The already-predicted shared-expert path is carried per source into the
  overlap window; group barriers do not repeat router/latent/norm work merely
  to recover that one component.
- Communication overlap retains the maximum source-local shared-expert path
  in addition to routed expert compute.
- The routed critical lane exports its live block-M, cluster-task, and wave
  geometry to production diagnostics.
- DP=1/EP>1 uses the same explicit decomposition. A DP1/EP4 integration test
  guards the former missing-breakdown crash.
- Expert shuffling remains unchanged.
- Routing remains deterministic for the same seed and batch.
- The exact histogram is reused across MoE layers when routing scope is
  `shared`; the model does not redraw routing per layer.
- Full CTest passed after implementation: 35/35 tests.

The focused unit test checks that equal-total/equal-active-expert histograms
with different M-block fragmentation receive different MegaMoE predictions,
while the portable aggregate roofline still treats them equally. It also
checks the block-M threshold, live diagnostic geometry, group-only scope, and
configuration overrides.

## Limitations and revisit conditions

The model should be revisited when any of the following becomes available:

1. GB300 K3 grouped-GEMM latency versus exact expert histograms;
2. public SM100 MegaMoE CTA/cluster grid and achieved-bandwidth traces;
3. backend-specific block-M/block-N dispatch rules that differ from the
   current DeepGEMM policy;
4. measurements covering routing skew, not only approximately random/balanced
   serving traffic;
5. enough data to fit and validate separate startup, per-wave, per-task, and
   sub-wave terms without using the same serving frontier for all parameters.
6. Rubin MegaMoE measurements that can replace the resource-ratio projection,
   especially A2A startup, achieved bandwidth, overlap, and tile policy.

The preferred next validation is a microbenchmark matrix indexed by
`(B_M, active experts, [m_e], up/down N, precision)`. At that point the zero
default can be replaced by hardware-derived coefficients while retaining the
current histogram-to-grid transformation.
