# Kimi K2 / GB300 CPU DRAM Capacity 실험 계획

> 실행 상태: 완료 (2026-08-03). 결과 요약은
> [`kimi-k2-gb300-cpu-dram-capacity-results.md`](kimi-k2-gb300-cpu-dram-capacity-results.md),
> 전체 CSV/그래프는 `outputs/kimi_k2_cpu_dram_v3/final/`에 있다.

## 1. 목적

순차 Prefill-Decode Disaggregation(PDD) 환경에서 PREFILL-side CPU KV cache의
DRAM 용량이 multi-turn workload의 KV 재사용, 반복 PREFILL 계산량, 그리고
successor-turn TTFT에 미치는 영향을 측정한다.

검증할 주 가설은 다음과 같다.

> CPU DRAM 용량이 증가하면 session prefix의 CPU KV 잔존율이 높아지고,
> CPU restore hit가 증가하며, 이전 context를 다시 계산하는 PREFILL 작업이
> 감소하여 successor-turn TTFT가 개선된다.

이 실험은 현재 C++ simulator가 지원하는 **PREFILL-side CPU KV tier**만
사용한다. DECODE에서 생성된 KV를 CPU 또는 PREFILL cluster로 반환하는 기능은
사용하거나 새로 구현하지 않는다.

## 2. 확정된 모델링 가정

### 2.1 Kimi K2 MLA KV 복제

- Model: `moonshotai/Kimi-K2-Instruct`
- Attention: MLA
- KV dtype: FP8 latent + BF16 RoPE
- TP rank마다 전체 MLA KV를 보관한다.
- Attention TP=4이므로 하나의 DP target에는 동일한 logical KV가 4개 존재한다.
- CPU offload도 네 TP rank의 복제본을 모두 저장하고 전송하는 것으로 모델링한다.

Kimi K2의 한 TP rank 기준 KV 크기는 다음과 같다.

```text
per-layer bytes/token = 512 * 1 byte + 64 * 2 bytes
                      = 640 bytes

one-copy bytes/token  = 640 * 61 layers
                      = 39,040 bytes

TP4 target bytes/token = 39,040 * 4
                       = 156,160 bytes
```

`block_size=16`일 때:

```text
one-copy bytes/block  = 39,040 * 16
                      = 624,640 bytes

TP4 target bytes/block = 624,640 * 4
                       = 2,498,560 bytes
```

평균 64 Ki-token context의 KV footprint는 다음과 같다.

```text
one TP rank = 65,536 * 39,040
            = 2,558,525,440 bytes
            = 2.383 GiB

one TP4 DP target = 10,234,101,760 bytes
                  = 9.531 GiB
```

### 2.2 DECODE KV 반환 없음

CPU tier는 PREFILL이 완료된 시점의 PREFILL KV snapshot만 보관한다. 이전
turn에서 DECODE가 생성한 KV는 다음 turn의 PREFILL cluster로 반환되지 않는다.

따라서 CPU hit가 완전하더라도 successor turn에서 계산해야 하는 최소 작업은
다음과 같다.

```text
implementation minimum PREFILL
  = current-turn new input
  + immediately preceding turn's decode output
  + at most one demoted 16-token block
```

실험 결과에서는 두 종류의 minimum을 구분한다.

- **Theoretical minimum:** 각 turn의 신규 입력 token만 계산하는 이상적 하한.
- **Implementation minimum:** 현재 PREFILL-only tier에서 피할 수 없는 신규 입력,
  직전 decode 출력 및 final-block demotion을 포함한 하한.

TTFT 개선 결과는 implementation minimum에 대한 수렴으로 해석한다. 이론적
minimum에 도달했다고 표현하지 않는다.

### 2.3 300% 추가 PREFILL의 정의

CPU offload가 없고 기존 GPU prefix도 모두 사라진 경우 다음을 목표로 한다.

```text
(scheduled PREFILL - theoretical minimum PREFILL)
------------------------------------------------- = 3.0
          theoretical minimum PREFILL
```

즉 총 scheduled PREFILL은 theoretical minimum의 약 4배가 된다.

## 3. 사전 구현 정합성 작업

현재 C++ `model_kv_cache_size_bytes()`는 Kimi MLA에 대해 한 logical copy의
크기를 반환한다. 반면 본 실험은 TP4의 각 rank에 전체 KV가 복제된다고
가정한다. 물리 DRAM 용량과 transfer time을 올바르게 모델링하려면 본 sweep
전에 다음 보정이 필요하다.

### 3.1 Target-physical KV byte contract

MLA에 대해 아래 두 값을 명시적으로 분리한다.

- `one_copy_kv_bytes`: GPU 한 TP rank와 scheduler block admission에 사용하는 값.
- `target_physical_kv_bytes`: CPU offload 및 PDD transfer traffic에 사용하는 값.

MLA TP replication이 활성화된 경우:

```text
target_physical_kv_bytes = one_copy_kv_bytes * attention_tensor_parallel_size
```

Dense/GQA/MQA의 기존 sharding contract는 변경하지 않는다.

### 3.2 적용 위치

보정된 target-physical byte 값은 다음 경로에서 동일하게 사용해야 한다.

1. CPU KV `bytes_per_block` 및 `capacity_blocks` 계산
2. D2H offload byte 수와 service time
3. H2D restore byte 수와 service time
4. PREFILL-to-DECODE KV transfer byte 수
5. request/system byte metrics

GPU scheduler의 `num_blocks`는 각 TP rank의 HBM에 들어가는 block 수이므로
one-copy bytes/block 기준으로 계산한다.

### 3.3 필수 단위 테스트

Kimi K2, FP8 KV, TP4, PP1, block size 16에 대해 다음을 고정값으로 검증한다.

| 항목 | 기대값 |
| --- | ---: |
| One-copy bytes/token | 39,040 |
| Target-physical bytes/token | 156,160 |
| One-copy bytes/block | 624,640 |
| Target-physical bytes/block | 2,498,560 |
| 64 Ki-token target footprint | 10,234,101,760 bytes |

TP1 결과와 기존 dense model 결과가 변하지 않는 회귀 테스트도 포함한다.

## 4. 시스템 구성

### 4.1 공통 PDD 구성

실제 turn 간 think time을 사용해야 하므로 `simulation_mode=online`을 사용한다.
현재 공개 C++ surface에 맞춰 sequential PDD만 사용한다.

```text
system_architecture       = pd-disaggregation
cluster_scheduler.type                 = sticky_round_robin  # fallback
cluster_scheduler.prefill_type         = cache_aware
cluster_scheduler.decode_type          = vllm_queue_aware
cluster_scheduler.cache_threshold      = 0.5
cluster_scheduler.balance_abs_threshold = 32
cluster_scheduler.balance_rel_threshold = 1.1
prefix_cache.enabled     = true
prefix_cache.key_mode    = session
```

PREFILL과 DECODE cluster는 동일하게 구성한다.

| 항목 | 값 |
| --- | ---: |
| GPUs per cluster | 16 |
| Replicas | 1 |
| Attention TP | 4 |
| PP | 1 |
| DP | 4 |
| MoE TP | 1 |
| EP | 16 |
| Total experts | 384 |
| Router top-k | 8 |

병렬 domain 검증식은 다음과 같다.

```text
Attention TP * DP = 4 * 4 = 16
MoE TP * EP       = 1 * 16 = 16
```

PREFILL 16 GPU와 DECODE 16 GPU를 합쳐 총 32 GPU를 모델링한다.

### 4.2 Execution model

```text
execution_model.type           = analytical
execution_model.device         = gb300
moe_layer_event_mode           = first_layer_scaled
moe_routing.distribution       = balanced
KV cache precision             = fp8
PDD KV transfer dtype size     = 1 byte
```

Kimi K2의 공개 FP8 checkpoint와 맞추기 위해 attention, dense, MoE expert,
router, LM head의 기본 weight/activation precision은 FP8로 둔다. FP4 expert
weight는 별도 sensitivity가 아니므로 사용하지 않는다. MLA RoPE component는
기존 analytical contract에 따라 BF16 2 bytes를 유지한다.

PREFILL-to-DECODE network는 기존 Kimi analytical benchmark의 고대역폭
baseline을 우선 재사용한다.

```text
network_bandwidth_gbps = 38,400
network_latency_ms     = 0.02
```

이 값은 모든 CPU DRAM case에서 고정하므로 주 sweep의 독립변수가 아니다.

### 4.3 GPU KV capacity

GB300 GPU의 288 GB HBM에서 다음을 제외한 약 190 GB/GPU를 KV budget으로
사용하는 것을 초기값으로 한다.

- Kimi K2 FP8 model weight의 추정 per-GPU footprint: 약 68 GB
- runtime workspace 및 안전 margin: 총 HBM의 약 10%

MLA KV가 TP rank마다 복제되므로 `num_blocks`는 한 rank의
`624,640 bytes/block`을 기준으로 계산한다.

```text
num_blocks = floor(190,000,000,000 / 624,640)
           = 304,175 blocks per DP target

token capacity = 304,175 * 16
               = 4,866,800 tokens per DP target
```

Pilot에서 다음 조건을 확인한 뒤 이 값을 전체 sweep에 고정한다.

- 최대 길이 request 한 개가 항상 admission 가능할 것
- DECODE preemption rate가 0.1% 미만일 것
- CPU-OFF control의 successor GPU prefix hit rate가 1% 이하일 것

마지막 조건을 만족하지 못하면 먼저 workload의 reuse distance를 조정한다.
GPU KV budget을 임의로 축소하는 것은 마지막 수단으로만 사용하고 변경 사실을
결과에 명시한다.

## 5. GB300 CPU topology와 DRAM 설정

NVIDIA GB300 NVL72는 72 GPU와 36 Grace CPU, compute tray당 4 GPU와 2 Grace
CPU 구성이다. 따라서 한 Grace CPU의 용량과 대역폭을 두 GPU가 공유한다고
가정한다.

공개 사양 기준 nominal 값:

```text
CPU DRAM capacity = 500 GB / Grace CPU
                  = 250 GB / GPU slice

CPU DRAM bandwidth = 약 388.89 GB/s / Grace CPU
                   = 약 194.44 GB/s / GPU slice
                   = 1,555.56 Gbps / GPU slice

NVLink-C2C          = 약 450 GB/s / GPU slice
                   = 3,600 Gbps / GPU slice
```

한 `(replica_id, dp_id)` CPU target은 TP4/PP1의 네 GPU slice를 포함한다.

```text
target capacity  = capacity_per_gpu * 4
target bandwidth = min(DRAM, C2C) * 4
                 = 6,222.22 Gbps
                 = 777.78 GB/s
```

용량 효과를 분리하기 위해 DRAM/C2C bandwidth와 latency는 모든 physical
capacity case에서 고정한다.

참고 사양:

- <https://www.nvidia.com/en-us/data-center/gb300-nvl72/>
- <https://docs.nvidia.com/enterprise-reference-architectures/nvl72-ai-factory/latest/components.html>
- <https://docs.nvidia.com/pdf/dgx-spod-gb300-ra.pdf>

## 6. CPU DRAM capacity sweep

### 6.1 Inclusive-cache 가설과 유효 CPU 용량

현재 PREFILL GPU cache와 CPU cache는 exclusive victim cache가 아니라
inclusive cache에 가까운 방식으로 동작한다. PREFILL 완료 시 CPU로 snapshot을
복사하더라도 같은 prefix가 GPU에 계속 남아 있을 수 있다. 따라서 CPU의
**logical prefix capacity가 GPU의 logical prefix capacity보다 작거나 같으면**
CPU에 저장된 대부분의 prefix가 GPU에도 존재하여 CPU restore가 GPU frontier를
실제로 연장하지 못할 수 있다.

TP4 복제에서는 물리 byte 총합이 아니라 각 계층이 보관할 수 있는 서로 다른
logical token frontier를 비교해야 한다.

```text
GPU logical capacity per DP target
  = num_blocks * one-copy bytes/block
  = 약 190 GB

CPU logical capacity per DP target
  = target physical capacity / TP replication factor
  = capacity_per_gpu_slice
```

따라서 GPU 대비 CPU logical capacity ratio를 다음과 같이 정의한다.

```text
rho = CPU capacity per GPU slice / GPU KV capacity per TP rank
    = CPU capacity per GPU slice / 190 GB
```

`rho <= 1`인 case에서 효과가 없거나 매우 작은 것은 실패가 아니라 inclusive
계층의 예상 가능한 결과다. 이 구간은 CPU 용량이 GPU cache보다 작을 때의
negative control로 유지한다.

CPU가 실제로 GPU frontier를 얼마나 연장했는지는 단순 CPU occupancy가 아니라
다음 지표로 판단한다.

- GPU hit 이후 CPU가 추가로 제공한 `cpu_prefix_hit_blocks`
- CPU restore 후 실제 소비된 `cpu_restore_consumed_blocks`
- `cached_prefill_tokens` 중 CPU extension이 차지하는 비율
- OFF 대비 actual scheduled PREFILL 감소량

CPU에 byte가 resident하더라도 위 extension/consumption이 0이면 유효한 계층
확장으로 간주하지 않는다.

### 6.2 Coarse sweep

주 sweep은 Grace CPU 한 대당 용량을 user-facing label로 사용하고, 실제
config에는 그 절반인 GPU slice당 용량을 넣는다.

| Case | Grace CPU당 DRAM | GPU slice당 DRAM | `rho` | PREFILL cluster 총 DRAM | 64 Ki session 등가 수 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `off` | 0 | 0 | 0.00 | 0 | 0 |
| `cpu32` | 32 GB | 16 GB | 0.08 | 256 GB | 약 25 |
| `cpu64` | 64 GB | 32 GB | 0.17 | 512 GB | 약 50 |
| `cpu128` | 128 GB | 64 GB | 0.34 | 1.024 TB | 약 100 |
| `cpu256` | 256 GB | 128 GB | 0.67 | 2.048 TB | 약 200 |
| `cpu500` | 500 GB | 250 GB | 1.32 | 4.000 TB | 약 390 |

`off`는 `cpu_kv_cache.enabled=false`로 실행한다. 나머지는 다음을 사용한다.

```text
cpu_kv_cache.enabled                  = true
cpu_kv_cache.static_slice_per_gpu     = true
cpu_kv_cache.dram_bandwidth_gbps_per_gpu = 1555.56
cpu_kv_cache.c2c_bandwidth_gbps_per_gpu  = 3600.0
cpu_kv_cache.eviction_policy          = session_lru_suffix
cpu_kv_cache.capacity_pressure_policy = prefix_fit
cpu_kv_cache.transfer_concurrency     = full_duplex_serialized
```

이 coarse grid는 작은 간격에서 정확한 knee를 찾기 위한 것이 아니라 다음을
한 번에 판별하기 위한 1차 탐색이다.

- `rho < 1` 구간이 실제로 무효 구간인지
- 효과가 처음 나타나는 인접 capacity pair
- physical GB300 최대 용량 안에 plateau가 존재하는지
- physical 범위 밖의 oracle에서만 효과가 나타나는지

### 6.3 Unbounded oracle

물리 sweep 외에 capacity eviction이 전혀 없는 reference를 한 개 추가한다.

- 이름: `oracle_unbounded_cpu`
- 각 DP target이 해당 seed의 모든 session snapshot을 수용할 수 있도록 direct
  capacity를 사전 계산한다.
- D2H/H2D bandwidth는 physical `cpu500` case와 동일하게 유지한다.

이 case는 제품 구성으로 해석하지 않고 현재 PREFILL-only implementation의
latency 및 recomputation lower bound를 확인하는 용도로만 사용한다.

### 6.4 Adaptive fine sweep

Coarse sweep 결과를 본 뒤 변화가 나타난 구간에만 더 작은 간격의 2차 sweep을
추가한다. 변화 지점은 seed 평균 기준으로 인접 두 capacity 사이에서 다음 중
하나 이상이 처음 발생하는 구간으로 정의한다.

- CPU prefix extension hit rate가 5 percentage points 이상 증가
- actual scheduled PREFILL이 5% 이상 감소
- successor-turn TTFT p90이 3% 이상 개선
- `evicted_blocks / offload_blocks` 또는
  `truncated_offloads / offload_operations`가 10 percentage points 이상 변화

변화가 나타난 마지막 무효점과 첫 유효점을 각각 `C_low`, `C_high`라고 한다.
두 끝점을 포함해 총 5~7개의 capacity point를 사용한다.

```text
fine step ~= (C_high - C_low) / 6
```

용량은 Grace CPU당 8 GB 단위로 반올림하며 `C_low`와 `C_high`도 다시 실행해
coarse 결과와 일치하는지 확인한다.

예를 들어 coarse sweep의 `cpu256`과 `cpu500` 사이에서 변화가 시작되면 다음
형태의 fine grid를 사용한다.

```text
Grace CPU당 DRAM = 256, 296, 336, 376, 416, 456, 500 GB
GPU slice당 DRAM = 128, 148, 168, 188, 208, 228, 250 GB
rho              = 0.67, 0.78, 0.88, 0.99, 1.09, 1.20, 1.32
```

Coarse physical sweep 전체에서 유의한 변화가 없고 unbounded oracle에서만
변화가 나타나면 physical GB300 용량이 inclusive GPU cache를 유효하게
확장하지 못한 것으로 일차 결론을 낸다. 이 경우 mechanism knee의 위치를 찾기
위해 GPU logical capacity의 2배와 4배에 해당하는 CPU capacity를
**non-physical analytical case**로 추가할 수 있다. 이 확장 case는 GB300 제품
결과와 분리해서 표시한다.

## 7. Multi-turn workload

### 7.1 Session 수와 turn 수

CPU-OFF calibration과 mechanism-pilot runtime gate를 통과한 frozen workload는
한 seed당 1,728 sessions를
생성한다.

| Session 종류 | 수 | Turn 수 |
| --- | ---: | ---: |
| A | 1,152 | 6 |
| B | 576 | 7 |

입력:출력 비율이 8:1이고 turn 크기가 평균적으로 균등하면 N-turn session의
추가 PREFILL 비율은 다음과 같다.

```text
redundant_ratio(N) = 9/16 * (N - 1)

N=6: 2.8125
N=7: 3.3750
```

6-turn과 7-turn session을 2:1로 혼합하면 기대 redundant ratio가 3.0이므로
총 PREFILL은 theoretical minimum의 4배가 된다.

### 7.2 최종 context 길이

- 평균 final context length: 65,536 tokens
- 분포: bounded log-normal
- 목표 CV: 약 0.15
- 범위: 49,152~98,304 tokens
- Kimi K2의 131,072-token context limit 이하로 제한

생성 후 sample 평균을 65,536 tokens에 맞도록 결정론적으로 rescale하고
block/ratio rounding을 다시 수행한다.

### 7.3 Turn별 token 분배

각 session에서 다음 순서로 token을 생성한다.

1. 최종 context 길이를 bounded log-normal에서 표본화한다.
2. 전체 output token을 final length의 1/9로 정한다.
3. 전체 input token을 output의 정확히 8배로 정한다.
4. 대화 turn별 share를 대칭 Dirichlet 분포에서 표본화한다.
5. output을 turn share에 따라 정수 배분한다.
6. 각 turn의 input을 해당 turn output의 정확히 8배로 둔다.
7. rounding residual은 마지막이 아닌 임의 turn들에 seed 기반으로 분산한다.

Dirichlet concentration은 `alpha=20`을 사용해 turn 길이가 지나치게 한쪽으로
치우치지 않게 하면서도 동일 길이의 합성 trace는 피한다.

CSV의 `num_prefill_tokens`에는 누적 context가 아니라 해당 turn의 **신규 input**
길이를 기록한다. C++ workload materialization이 이전 input/decode context를
successor의 effective PREFILL에 누적한다.

### 7.4 Session arrival과 think time

First-turn session arrival:

```text
process = Poisson
rate    = 100 new sessions/s
```

Successor think time:

```text
distribution = log-normal
median       = 1,000 s
log sigma    = 0.001
clip         = [990 s, 1,010 s]
```

초기값(768 sessions, 1 session/s, think-time median 120 s)에서는 CPU-OFF
GPU-prefix hit가 약 50.9% 발생했다. 위 frozen 설정은 확률적 turn 간격을
유지하면서 같은 session의 reuse distance가 190 GB/GPU prefix-cache budget보다
충분히 커지도록 calibration한 값이다.

### 7.5 Seed

```text
pilot seed = 20260803
final seeds = 20260803, 20260804, 20260805, 20260806, 20260807
```

모든 capacity case는 seed별로 완전히 동일한 workload CSV를 공유한다.

### 7.6 CPU-OFF workload calibration

6-turn/7-turn 혼합으로 계산한 300%는 workload 생성 분포의 기대값일 뿐이며,
실제 simulator 실행값을 보장하지 않는다. GPU prefix hit, turn-size rounding,
scheduler preemption 및 batch 재계산 때문에 실제 scheduled PREFILL은 달라질
수 있다. 따라서 CPU capacity sweep보다 먼저 CPU-OFF calibration을 수행한다.

Calibration에서 측정하는 값은 다음과 같다.

```text
R_actual =
  (sum(actual_scheduled_prefill_tokens) - sum(new_input_tokens))
  / sum(new_input_tokens)
```

허용 범위:

```text
2.80 <= R_actual <= 3.20
```

즉 추가 PREFILL은 minimum의 280~320%, 총 scheduled PREFILL은 minimum의
3.80~4.20배여야 한다.

조정 순서는 원인별로 고정한다.

1. **Generator 자체 비율 확인:** GPU hit와 preemption을 적용하지 않은
   materialized prompt의 이론값이 2.80~3.20인지 확인한다.
2. **R_actual이 너무 낮고 GPU hit가 존재:** session 수, think time 또는
   reuse distance를 늘려 GPU hit rate를 1% 이하로 낮춘다.
3. **R_actual이 너무 낮고 GPU hit가 없음:** 7-turn session 비중 또는
   turn-share 분포를 높여 누적 context 재계산량을 늘린다.
4. **R_actual이 너무 높고 preemption이 존재:** offered load를 낮추거나 GPU
   KV admission 여유를 늘려 preemption 재계산을 먼저 제거한다. 이 경우 turn
   수를 줄여 수치를 억지로 맞추지 않는다.
5. **R_actual이 너무 높고 preemption이 없음:** 6-turn session 비중을 높이거나
   turn-share 분포를 조정한다.

한 항목을 조정할 때 final context mean 64 Ki와 aggregate input:output 8:1은
고정한다. 조정 후에는 새 workload version과 조정 이유를 manifest에 기록하고
모든 capacity case가 동일한 최종 workload를 사용하도록 한다.

#### Frozen workload calibration 결과

`generator_version=kimi-k2-cpu-dram-v3-frozen`으로 다음 결과를 확인하고
workload 규칙을 동결했다.

| Seed | Request 수 | 실제 추가 PREFILL | GPU prefix hit | Preemption |
| --- | ---: | ---: | ---: | ---: |
| 20260803 | 10,944 | 296.276% | 0.815% | 0 |
| 20260804 | 10,944 | 297.222% | 0.767% | 0 |
| 20260805 | 10,944 | 297.014% | 0.774% | 0 |
| pooled | 32,832 | 296.837% | 0.785% | 0 |

따라서 pooled 280~320%, 개별 seed 270~330%, GPU hit 1% 이하 조건을 모두
통과했다. 이후 capacity sweep에서는 session/arrival/think-time/turn 분포를
변경하지 않는다.

초기 768-session 설정은 실제 추가 PREFILL 96.0%로 실패했다. reuse distance를
늘린 2,304-session 후보는 gate를 통과했지만 CPU 활성 pilot이 900초를 넘겨
runtime gate를 만족하지 못했다. hot-path invariant 검사를 실험 runner에서
비활성화하고(최종 diagnostics와 테스트의 전체 검사는 유지), 1,728-session
후보로 재검증해 위 v3 규칙을 최종 동결했다.

Pilot seed에서 범위를 맞춘 뒤 서로 다른 세 seed로 validation-only CPU-OFF
실행을 수행한다. 세 seed의 pooled 결과가 2.80~3.20이고 개별 seed가 모두
2.70~3.30 안에 들어올 때 workload 생성 규칙을 freeze한다. 이후 main sweep의
5개 seed에는 frozen 규칙만 적용하고 capacity 결과를 본 뒤 workload를 다시
조정하지 않는다.

### 7.7 Workload manifest

raw simulator CSV 외에 request 단위 분석 manifest를 저장한다.

```text
request_id
session_id
session_turn_index
new_input_tokens
decode_tokens
prior_context_tokens
prior_decode_tokens
theoretical_min_prefill_tokens
implementation_min_prefill_tokens
actual_scheduled_prefill_tokens
actual_extra_ratio
```

이 manifest를 사용해 simulator output의 materialized prompt와 actual scheduled
PREFILL을 비교한다.

## 8. 실험 matrix

### 8.1 Phase W: workload calibration

CPU-OFF만 사용해 Section 7.6의 280~320% gate를 먼저 통과시킨다.

1. Pilot seed 한 개로 generator/modeling 원인을 진단한다.
2. 필요하면 session 수, think time, 6/7-turn 혼합 및 turn share를 정해진
   순서로 조정한다.
3. 서로 다른 세 seed로 validation-only CPU-OFF 실행을 수행한다.
4. gate를 통과하면 workload generator version과 모든 파라미터를 freeze한다.

이 단계가 끝나기 전에는 CPU capacity 간 TTFT 비교를 시작하지 않는다.

### 8.2 Phase P: mechanism pilot

Frozen pilot workload로 다음 세 case를 실행한다.

1. `off`
2. `cpu128`
3. `oracle_unbounded_cpu`

Pilot 목적:

- TP-replicated KV byte accounting 확인
- 모든 session/request 완료 확인
- OFF의 GPU prefix hit rate 확인
- CPU capacity가 실제 pressure를 만드는지 확인
- DECODE preemption과 scheduler progress 확인
- summary/request output만으로 분석 가능한지 확인
- simulator 실행 시간과 host RAM 사용량 추정

### 8.3 Phase C: coarse capacity sweep

Pilot 통과 후 7 cases와 5 seeds를 paired execution한다.

```text
7 capacity cases * 5 seeds = 35 runs
```

실행 순서는 seed 안에서 case 순서를 섞어 host thermal/cache 영향이 특정
capacity에 편향되지 않도록 한다. simulator 자체는 deterministic해야 하므로
같은 config/workload 재실행 결과가 동일한지도 한 case에서 확인한다.

### 8.4 Phase F: adaptive fine capacity sweep

Coarse 결과에서 Section 6.4의 변화 구간을 선택한다. 선택 규칙은 결과를 보기
전에 고정하며, coarse 양 끝점을 포함한 총 5~7개 fine capacity를 5개 paired
seed로 실행한다.

Fine sweep의 run 수는 선택된 point 수를 `N_fine`이라고 할 때 다음과 같다.

```text
N_fine * 5 seeds
```

Fine 결과로 다음을 추정한다.

- 최초 유효 CPU extension capacity
- scheduled PREFILL 감소가 가장 가파른 knee
- TTFT 개선이 plateau에 도달하는 capacity
- `rho=1` 전후 inclusive-cache 전환 여부

### 8.5 Optional bandwidth sensitivity

주 capacity sweep이 끝난 뒤에만 `cpu500`을 대상으로 별도 실행한다.

```text
0.5x DRAM bandwidth
1.0x DRAM bandwidth
2.0x DRAM bandwidth
```

이 결과는 capacity 효과와 섞지 않고 restore bandwidth가 TTFT 개선을 제한하는
정도를 확인하는 보조 분석으로만 사용한다.

## 9. 측정 지표

### 9.1 Primary metric

첫 turn을 제외한 successor turn의 TTFT:

- mean
- p50
- p90
- p99
- turn index별 mean/p90

Session당 turn 수가 6/7로 다르므로 전체 request를 단순 pooling한 값과 함께
session-equal-weighted 통계도 보고한다.

### 9.2 PREFILL work

각 request와 전체 workload에 대해 다음을 계산한다.

```text
actual_scheduled_prefill_tokens

theoretical_extra_ratio =
  (actual_scheduled_prefill_tokens - new_input_tokens)
  / new_input_tokens

avoidable_extra_tokens =
  actual_scheduled_prefill_tokens
  - implementation_min_prefill_tokens
```

Preemption에 의한 재계산까지 포함하려면 PREFILL batch들의 실제 scheduled
token 합을 사용해야 한다. 대규모 run에서 `full` trace를 저장하지 않도록
compact summary/request metrics에 다음 필드를 추가한다.

- `scheduled_prefill_tokens`
- `preemption_recomputed_prefill_tokens`
- cluster aggregate `prefill_scheduled_tokens`

### 9.3 Cache/transfer metrics

- GPU prefix query/hit blocks
- CPU prefix query/hit blocks
- CPU hit rate
- cached PREFILL tokens
- CPU resident/peak-resident bytes
- CPU block/session evictions
- truncated/skipped offloads
- D2H/H2D operation count와 bytes
- D2H/H2D queue time와 service time
- PREFILL source-GPU hold time
- restore transferred/consumed/discarded blocks

### 9.4 System guard metrics

- request completion count
- incomplete request count
- PREFILL/DECODE preemptions
- request/token throughput
- TPOT
- PDD KV transfer latency
- simulation makespan
- wall-clock runtime와 simulator process peak RSS

## 10. 성공 조건과 실험 gate

### Gate A: 구성 정합성

- PREFILL/DECODE가 각각 정확히 16 GPU일 것.
- TP4/PP1/DP4/MoE-TP1/EP16 validation을 통과할 것.
- FP8 MLA target bytes/block이 2,498,560일 것.
- CPU target 수가 PREFILL DP 수와 같은 4개일 것.
- `cpu500` target capacity가 target당 1 TB일 것.

### Gate B: Workload 정합성

- final context mean이 65,536의 1% 이내일 것.
- aggregate input:output ratio가 8:1의 0.5% 이내일 것.
- generator/materialized-prompt 기준 theoretical extra ratio가
  `3.00 +/- 0.05`일 것.
- 모든 length가 Kimi K2 context limit 이하일 것.

### Gate C: OFF control

- CPU hit/offload/restore가 모두 0일 것.
- successor GPU prefix hit rate가 1% 이하일 것.
- actual extra ratio가 2.80~3.20일 것.
- 총 actual scheduled PREFILL이 theoretical minimum의 3.80~4.20배일 것.
- calibration 후 세 validation seed의 pooled ratio가 2.80~3.20이고 각 seed가
  2.70~3.30일 것.

GPU hit가 1%보다 높으면 think-time median 또는 동시 session 수를 늘린다.
이때 모든 capacity case의 workload를 다시 생성하여 paired comparison을
유지한다.

### Gate D: Unbounded oracle

- CPU eviction, truncation, skip이 모두 0일 것.
- 첫 turn, 신규 input, 직전 decode를 제외한 restorable prior-prefix 중 GPU에
  남지 않은 block을 분모로 한 CPU hit rate가 99% 이상일 것. 전체 raw
  `cpu_query_blocks`는 복원 불가능한 신규 token도 포함하므로 Gate D 분모로
  사용하지 않는다.
- 같은 restorable prior-prefix에 대한 GPU+CPU combined coverage가 99% 이상일 것.
- scheduled PREFILL이 implementation minimum과 block/preemption tolerance
  이내에서 일치할 것.

### Gate E: Coarse-to-fine 전환

- 모든 coarse case에 `rho`를 기록할 것.
- CPU occupancy와 CPU의 unique prefix extension을 구분할 것.
- Section 6.4의 사전 정의된 기준으로 `C_low`, `C_high`를 선택할 것.
- coarse 결과를 확인한 뒤 workload, GPU capacity 또는 transition threshold를
  변경하지 않을 것.
- physical range에 transition이 없으면 무효 결과를 숨기지 않고 먼저 보고한
  뒤 non-physical analytical extension 여부를 결정할 것.

### Gate F: Runtime 안정성

- incomplete requests가 0일 것.
- terminal CPU reservation/lease/pending/staged state가 모두 0일 것.
- DECODE preemption rate가 0.1% 미만일 것.
- 동일 seed/config 재실행 결과가 deterministic할 것.

## 11. 분석 방법

### 11.1 주 그래프

1. CPU DRAM capacity vs successor TTFT p50/p90/p99
2. CPU DRAM capacity vs scheduled PREFILL / theoretical minimum
3. CPU DRAM capacity vs CPU prefix hit rate
4. CPU DRAM capacity vs CPU eviction/truncation
5. CPU DRAM capacity vs H2D queue/service time
6. Turn index별 TTFT와 cached-token 비율

X축은 Grace CPU당 DRAM GB를 사용하고 보조 축 또는 point label에 `rho`를
표시한다. OFF는 별도 categorical point로 표시한다. Coarse point와 adaptive
fine point는 marker를 달리해 탐색 단계와 정밀 측정 단계를 구분한다.

### 11.2 인과 확인

다음 chain이 같은 capacity 순서에서 나타나는지 확인한다.

```text
capacity 증가
  -> eviction 감소
  -> CPU hit 증가
  -> scheduled PREFILL 감소
  -> PREFILL latency 감소
  -> successor TTFT 감소
```

TTFT가 단조 감소하지 않으면 다음 항목으로 원인을 분리한다.

- H2D restore queue/service 증가
- D2H offload로 인한 PREFILL source hold 증가
- scheduler batch composition 변화
- GPU/CPU hit 구성 변화
- PREFILL 또는 DECODE preemption

`rho <= 1`에서 CPU occupancy만 증가하고 CPU extension hit가 증가하지 않으면
inclusive duplication으로 판정한다. 반대로 `rho <= 1`에서도 extension이
나타나면 GPU/CPU LRU 순서, offload commit 시점 및 GPU eviction timing 차이를
추적해 단순 capacity 비교로 설명하지 않는다.

### 11.3 통계

- 각 seed의 session-equal-weighted 결과를 기본 관측치로 사용한다.
- 5개 paired seed의 차이를 보고한다.
- session bootstrap 95% confidence interval을 보조로 제공한다.
- 절대 TTFT뿐 아니라 OFF 대비 상대 변화율을 함께 보고한다.

```text
TTFT improvement = (TTFT_off - TTFT_capacity) / TTFT_off
PREFILL reduction = (scheduled_off - scheduled_capacity) / scheduled_off
```

## 12. 예상 결과

Workload calibration과 byte-contract 검증을 통과한 뒤 다음 가설을 평가한다.

- `off`에서 총 PREFILL은 theoretical minimum의 3.8~4.2배이다.
- `rho < 1`인 32~256 GB/CPU 구간은 CPU와 GPU의 inclusive overlap 때문에
  occupancy가 생겨도 unique CPU extension과 TTFT 개선이 거의 없을 수 있다.
- 효과가 존재한다면 `rho=1` 전후 또는 GPU eviction working-set boundary에서
  CPU hit와 scheduled PREFILL이 급격히 변할 가능성이 있다.
- 정확한 knee는 사전에 단정하지 않고 coarse sweep으로 인접 변화 구간을 찾은
  뒤 adaptive fine sweep으로 추정한다.
- 500 GB/CPU에서도 변화가 없고 oracle에서만 변화하면 physical GB300 CPU
  DRAM은 현재 inclusive 정책에서 GPU cache를 유효하게 확장하지 못한다는
  결과가 된다.
- 500 GB/CPU는 전체 1,728개 completed session snapshot을 동시에 보관하지는
  못한다.
- `oracle_unbounded_cpu`는 implementation minimum에 근접한다.
- DRAM이 커져도 직전 decode output과 신규 input은 계속 PREFILL되므로
  theoretical minimum에는 도달하지 않는다.

복제된 TP4 KV를 사용하면 평균 64 Ki session 하나가 CPU tier에서 약
9.53 GiB를 차지한다. 따라서 TP replication을 생략한 one-copy 모델보다
capacity knee가 약 4배 작은 session 수에서 나타나야 한다.

## 13. 산출물 구조

실험 구현 시 다음 구조를 사용한다.

```text
cpp/experiments/kimi_k2_cpu_dram/
  README.md
  generate_workload.py
  run_sweep.py
  analyze_results.py
  configs/
    base_pdd.json
  workloads/
    seed_<seed>.csv
    seed_<seed>_manifest.csv

outputs/kimi_k2_cpu_dram/
  manifest.json
  workload_calibration/
    iteration_<n>/
    frozen_workload_manifest.json
  mechanism_pilot/
  coarse/
    seed_<seed>/
      off/
      cpu32/
      cpu64/
      cpu128/
      cpu256/
      cpu500/
      oracle_unbounded_cpu/
  fine/
    selection.json
    seed_<seed>/
      cpu<selected_capacity>/
  summary.csv
  session_summary.csv
  plots/
```

대규모 run은 `summary` 또는 `requests` output mode를 사용한다. `full` trace는
pilot의 소규모 subset과 실패 재현 case에만 사용해 simulator host memory가
trace 크기에 의해 지배되지 않도록 한다.

## 14. 실행 순서

1. MLA TP-replicated physical byte contract 구현
2. CPU/PDD transfer와 metric byte 회귀 테스트 추가
3. Compact scheduled-PREFILL metric 추가
4. Multi-turn workload generator와 manifest 구현
5. 초기 workload 생성
6. CPU-OFF pilot로 actual extra ratio 측정
7. 280~320% 밖이면 원인별 순서에 따라 workload/load 조정
8. 세 CPU-OFF validation seed 통과 후 workload 생성 규칙 freeze
9. `off`, `cpu128`, `oracle_unbounded_cpu` mechanism pilot 실행
10. Gate A~D 및 Gate F 검증
11. 5-seed, 35-run coarse capacity sweep 실행
12. Gate E 기준으로 `C_low`, `C_high` 선택
13. 선택 구간의 adaptive fine sweep 실행
14. Coarse/fine 결과 집계 및 그래프 생성
15. Optional analytical-capacity 또는 bandwidth sensitivity 실행
16. 최종 결과 문서 작성

## 15. 범위 밖 항목

다음은 본 실험 결과의 해석 범위에 포함하지 않는다.

- DECODE KV의 CPU/PREFILL 반환
- shared Grace CPU/NUMA/memory-controller contention의 상세 모델
- CPU KV compression 또는 dtype conversion
- NVMe tier
- session termination/TTL-aware CPU cache reclamation
- 실측 GB300 profiling 기반 latency calibration
- expert routing skew sensitivity

본 결과는 GB300 공개 사양을 사용한 analytical what-if study이며, 실측
GB300 성능 예측치로 표현하지 않는다.
