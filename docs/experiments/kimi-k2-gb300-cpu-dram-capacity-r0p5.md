# Kimi K2 / GB300 CPU DRAM Capacity 실험 — 최신 r0.5 기준

> 상태: 실행 완료 및 결과 재집계 (2026-08-04)  
> 기준 seed: `20260803`  
> 기준 workload: `long_mixed_steady4096_r0p5`  
> 기준 코드: Git commit `f677654b350e21fc6d0cf878cc53b68512f146c3` (`cxx-port`)  
> 대화형 보고서: `outputs/kimi_k2_gap_sweep_r0p5/experiment_report.html`

이 문서는 마지막으로 수행한 `0.5 sessions/s` CPU KV capacity sweep의 실제 실행
조건, workload 생성 방법, 지표 정의, 산출물 위치 및 결과를 한 곳에 정리한다.
이전의 `kimi-k2-gb300-cpu-dram-capacity-study.md`와
`kimi-k2-gb300-cpu-dram-capacity-results.md`는 1,728-session v3 실험을 설명하며,
본 문서의 4,096-session r0.5 실험과는 별개의 결과다.

## 1. 실험 질문과 결론

### 1.1 질문

순차 Prefill-Decode Disaggregation(PDD) 환경에서 PREFILL-side CPU KV cache의
DRAM 용량을 늘렸을 때 다음 변화가 발생하는지 측정했다.

1. GPU에서 사라진 session prefix를 CPU에서 더 많이 복구할 수 있는가?
2. successor turn의 반복 PREFILL 계산량이 줄어드는가?
3. PREFILL 감소가 TTFT 개선으로 이어지는가?
4. TPOT, decode batch size 및 request throughput은 용량과 무관하게 유지되는가?

### 1.2 최종 결론

- CPU 용량 증가에 따라 CPU-added prefix hit와 PREFILL 절감이 단조 증가했다.
- Grace CPU당 `1.75 TB`에서 oracle 대비 CPU 복구량의 `95.91%`를 달성했다.
- `1.75 TB`에서 최대 PREFILL 절감의 `95.92%`, successor TTFT mean 개선의
  약 `95.7%`, p90 개선의 약 `96.5%`를 달성했다.
- 운영상 단순한 반올림 지점은 Grace CPU당 `2 TB`다.
- `2.75 TB`부터 PREFILL, CPU hit 및 TTFT가 oracle과 동일해졌다.
- `2.75 TB`보다 큰 용량은 이 workload에서는 추가 효과가 없었다.
- TPOT은 `26.815~26.824 ms`, decode average batch size는
  `15.155~15.185`로 사실상 일정했다.

단, Grace CPU당 500 GB를 넘는 모든 지점은 실제 GB300 제품 용량이 아니라
analytical extension이다. 따라서 1.75 TB knee는 물리 제품 추천값이 아니라
현재 cache 정책과 workload에 대한 용량 민감도 결과다.

## 2. 초기 계획 대비 변경 사항

초기 실험은 CPU-OFF에서 GPU prefix hit를 1% 이하로 만들고, 실제 scheduled
PREFILL을 theoretical minimum의 약 4배로 만드는 것을 목표로 했다. 이후 신규
세션이 한꺼번에 몰리지 않게 하고 decode batch를 현실적인 크기로 유지하기 위해
arrival과 turn gap을 다시 구성했다.

최종 r0.5 workload의 결과는 다음과 같다.

| 항목 | 초기 목표 | 최종 r0.5 실현값 |
| --- | ---: | ---: |
| CPU-OFF GPU prefix hit | 1% 이하 | 52.998% |
| CPU-OFF 추가 PREFILL / theoretical minimum | 280~320% | 88.116% |
| CPU-OFF scheduled / theoretical minimum | 약 4배 | 1.881배 |
| Materialized extra / theoretical minimum | 약 300% | 300.141% |
| 신규 세션 유입 | calibration 대상 | stratified 0.5 sessions/s |

`materialized extra`는 cache가 하나도 없다고 가정한 workload 자체의 누적
context 중복량이다. 실제 CPU-OFF 실행에서는 GPU cache가 약 53%를 복구하기
때문에 실제 scheduled PREFILL은 이 값보다 훨씬 작다.

따라서 이 결과는 다음 두 질문에는 답하지만, “실제 CPU-OFF 계산량이 최소치의
4배인 상태”를 검증한 결과로 사용하면 안 된다.

- 현실화한 turn gap과 0.5 sessions/s 유입에서 CPU capacity가 주는 효과
- GPU+CPU inclusive 계층에서 CPU가 GPU miss를 얼마나 추가 복구하는지

## 3. 시스템 및 모델 설정

### 3.1 전체 구조

| 항목 | 설정 |
| --- | --- |
| Simulation mode | `online` |
| System architecture | `pd-disaggregation` |
| Parallel cluster execution | 비활성화, sequential PDD |
| PREFILL cluster | 16 × GB300 |
| DECODE cluster | 16 × GB300 |
| Replica | cluster당 1개 |
| Attention parallelism | TP4 / PP1 / DP4 |
| MoE parallelism | MoE-TP1 / EP16 |
| Model | `moonshotai/Kimi-K2-Instruct` |
| Experts / top-k | 384 / 8 |
| Routing | balanced simulation, seed 42 |
| Scheduler | `vllm_v1`, FCFS |
| Cluster routing | `sticky_round_robin` |
| Prefix key | session |
| Execution model | analytical roofline |

Attention domain과 MoE domain은 각각 16 GPU를 사용한다.

```text
Attention TP × DP = 4 × 4  = 16
MoE TP × EP       = 1 × 16 = 16
```

PREFILL과 DECODE는 별도의 16-GPU cluster이므로 총 32 GPU를 모델링한다.
parallel cluster execution은 사용하지 않았으며 PREFILL cluster simulation 뒤에
DECODE cluster simulation을 수행하는 공개 C++ sequential PDD 경로를 사용했다.

### 3.2 Scheduler 설정

| 설정 | PREFILL | DECODE |
| --- | ---: | ---: |
| Batch size cap | 64 | 32 |
| Max tokens in batch | 131,072 | 8,192 |
| Chunked PREFILL | 사용 | 사용 안 함 |
| Preemption | 허용 | 허용 |
| Block size | 16 tokens | 16 tokens |
| KV blocks | 304,175 | 304,175 |
| Watermark | 0 | 0 |
| Preallocated tokens | 0 | 0 |

실제 전 capacity run에서 preemption은 0회였다.

### 3.3 Analytical GB300 성능값

GB300 수치는 실측 profiling 결과가 아니라 simulator의 dense, non-sparse
per-GPU roofline ceiling preset이다.

| 항목 | Per-GPU ceiling |
| --- | ---: |
| HBM bandwidth | 8.0 TB/s |
| FP32 | 83.333 TFLOP/s |
| FP16 | 2,500 TFLOP/s |
| FP8 | 5,000 TFLOP/s |
| FP4 | 15,000 TFLOP/s |

모델 연산, activation, weight, communication 및 KV cache precision은 FP8을
기본으로 사용했다. MLA RoPE component의 byte 계산에는 BF16 2 bytes를
유지했다. MoE layer event mode는 `first_layer_scaled`다.

### 3.4 GPU 및 PDD 통신

| 항목 | 값 |
| --- | ---: |
| Collective network bandwidth | 38,400 Gb/s |
| Intra-node bandwidth | 14,400 Gb/s |
| Collective latency | 1 μs |
| PREFILL→DECODE KV bandwidth | 38,400 Gb/s |
| PREFILL→DECODE base latency | 0.02 ms |
| KV transfer dtype | 1 byte |
| Compression | 사용 안 함 |

PREFILL→DECODE KV transfer는 25,941회 발생했고 총 149.098 TB를 전송했다.
평균 transfer latency는 1.217 ms, p90은 2.089 ms, p99는 2.652 ms였다.

## 4. KV cache byte contract

### 4.1 MLA TP 복제 가정

본 실험에서는 MLA KV를 TP shard로 나누지 않고 TP4의 각 rank가 전체 KV를
저장한다고 가정했다.

```text
per-layer bytes/token
  = latent KV 512 × 1 byte + RoPE 64 × 2 bytes
  = 640 bytes

one-copy bytes/token
  = 640 × 61 layers
  = 39,040 bytes

TP4 target physical bytes/token
  = 39,040 × 4
  = 156,160 bytes
```

Block size가 16 tokens이므로 다음 값을 사용했다.

```text
one-copy bytes/block       = 39,040 × 16 = 624,640 bytes
TP4 physical bytes/block   = 624,640 × 4 = 2,498,560 bytes
```

64 Ki-token context 하나의 KV 크기는 GPU 한 대에서 약 2.559 GB, TP4 전체에서
약 10.234 GB다.

### 4.2 GPU KV capacity

각 TP rank의 GPU HBM 중 190,000,000,000 bytes를 KV budget으로 사용했다.

```text
blocks per GPU/DP target
  = floor(190,000,000,000 / 624,640)
  = 304,175 blocks

token capacity
  = 304,175 × 16
  = 4,866,800 tokens
```

`num_blocks`는 logical block 수이며 TP4의 각 GPU가 같은 block을 저장한다.
CPU offload와 PREFILL→DECODE transfer byte에는 TP4 물리 복제량을 사용한다.

### 4.3 Decode KV 반환

CPU tier는 PREFILL 완료 시점의 PREFILL-side snapshot만 저장한다. DECODE에서
생성한 KV를 CPU나 PREFILL cluster로 반환하지 않는다. 따라서 CPU hit가
완전하더라도 successor turn에서 최소한 다음 token은 다시 PREFILL해야 한다.

```text
implementation minimum
  = current-turn new input
  + immediately preceding decode output
  + successor당 최대 한 개의 16-token demoted block
```

### 4.4 Inclusive cache 해석

GPU와 CPU는 exclusive victim cache가 아니라 inclusive cache에 가깝다. 같은
session prefix가 GPU와 CPU에 동시에 존재할 수 있다. 따라서 CPU occupancy가
증가했다고 해서 CPU가 GPU cache의 유효 범위를 늘렸다고 바로 해석하지 않았다.

결과에서는 다음을 별도로 계산했다.

- GPU direct hit: CPU 도움 없이 GPU에서 직접 복구된 block
- CPU added hit: GPU miss 중 CPU가 추가로 복구한 block
- Combined hit: GPU direct + CPU added
- CPU recovery/oracle: 해당 capacity의 CPU hit blocks / oracle CPU hit blocks

## 5. CPU topology, bandwidth 및 capacity 환산

### 5.1 Grace CPU 공유 가정

한 Grace CPU의 용량과 대역폭을 두 GPU가 공유한다고 가정했다. 사용자에게
표시한 capacity `C`는 Grace CPU 한 대 기준이며 simulator에는 GPU slice당
`C/2`를 입력했다.

PREFILL cluster에는 16 GPU, 즉 8개의 Grace CPU-equivalent가 있다.

```text
GPU slice capacity       = C / 2
TP4 DP-target capacity   = 4 × C/2 = 2C
PREFILL CPU tier total   = 4 targets × 2C = 8C
rho                      = (C/2) / 190 GB
```

### 5.2 CPU bandwidth

| 항목 | 값 |
| --- | ---: |
| DRAM bandwidth per GPU slice | 1,555.56 Gb/s = 194.445 GB/s |
| C2C bandwidth per GPU slice | 3,600 Gb/s = 450 GB/s |
| TP4 target effective bandwidth | 6,222.24 Gb/s = 777.78 GB/s |
| D2H latency | 0.01 ms |
| H2D latency | 0.01 ms |
| Eviction | `session_lru_suffix` |
| Capacity pressure | `prefix_fit` |
| Transfer concurrency | direction별 serialized, D2H/H2D full duplex |

DRAM이 C2C보다 느리므로 TP4 target의 resolved bandwidth는 DRAM 합계가 된다.
모든 capacity case에서 bandwidth와 latency는 고정했고 용량만 변경했다.

### 5.3 Capacity sweep

| Case | Grace CPU당 | GPU slice당 | TP4 target당 | 전체 PREFILL tier | rho | 물리성 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| OFF | 0 | 0 | 0 | 0 | 0 | control |
| cpu500 | 0.50 TB | 0.25 TB | 1.0 TB | 4 TB | 1.32 | 물리 상한 기준점 |
| cpu1000 | 1.00 TB | 0.50 TB | 2.0 TB | 8 TB | 2.63 | analytical |
| cpu1250 | 1.25 TB | 0.625 TB | 2.5 TB | 10 TB | 3.29 | analytical |
| cpu1500 | 1.50 TB | 0.75 TB | 3.0 TB | 12 TB | 3.95 | analytical |
| cpu1750 | 1.75 TB | 0.875 TB | 3.5 TB | 14 TB | 4.61 | analytical |
| cpu2000 | 2.00 TB | 1.00 TB | 4.0 TB | 16 TB | 5.26 | analytical |
| cpu2250 | 2.25 TB | 1.125 TB | 4.5 TB | 18 TB | 5.92 | analytical |
| cpu2500 | 2.50 TB | 1.25 TB | 5.0 TB | 20 TB | 6.58 | analytical |
| cpu2750 | 2.75 TB | 1.375 TB | 5.5 TB | 22 TB | 7.24 | analytical |
| cpu3000 | 3.00 TB | 1.50 TB | 6.0 TB | 24 TB | 7.89 | analytical |
| cpu3500 | 3.50 TB | 1.75 TB | 7.0 TB | 28 TB | 9.21 | analytical |
| cpu4000 | 4.00 TB | 2.00 TB | 8.0 TB | 32 TB | 10.53 | analytical |
| oracle | 해당 없음 | 해당 없음 | 41.929 TB | 167.716 TB | 해당 없음 | unbounded reference |

Oracle capacity는 모든 session snapshot을 eviction 없이 저장하도록 workload에서
계산한 값이다. Oracle의 실제 peak resident는 전체 tier 기준 약 41.178 TB였다.

## 6. Workload 생성

### 6.1 생성 파라미터

| 항목 | 값 |
| --- | ---: |
| Seed | 20260803 |
| Sessions | 4,096 |
| 6-turn sessions | 2,731 |
| 7-turn sessions | 1,365 |
| Total requests | 25,941 |
| Final context target mean | 65,536 tokens |
| Final context range | 49,152~98,304 tokens |
| Context CV | 0.15 |
| Turn allocation | symmetric Dirichlet, alpha 20 |
| Input:output | 정확히 8:1 |
| First-session arrival | stratified 0.5 sessions/s |
| Think profile | `long_mixed` |

재현 가능한 workload 생성 명령은 다음과 같다.

```powershell
python cpp/experiments/kimi_k2_cpu_dram/generate_workload.py `
  --seed 20260803 `
  --sessions 4096 `
  --six-turn-sessions 2731 `
  --arrival-rate 0.5 `
  --arrival-process stratified `
  --think-profile long_mixed `
  --output-dir cpp/experiments/kimi_k2_cpu_dram/workloads_profiles/long_mixed_steady4096_r0p5
```

생성 workload SHA-256은 다음과 같다.

```text
seed_20260803.csv
  2B4EA720CBB83753E7954A709389EE8C819376473FF372F6A926AC55A213551E

seed_20260803_manifest.csv
  E6A1AC06BAF6C931E6E349BBBACBC27CEF0ED524879C29F6F7525474737C0E76
```

### 6.2 First-turn session arrival

Poisson arrival 대신 stratified stochastic arrival을 사용했다. 세션 `i`의 시작
시각은 다음과 같이 생성한다.

```text
session_start_i = (i + Uniform(0, 1)) / 0.5
```

첫 세션 시작을 0초로 이동한다. 따라서 평균 2초마다 한 세션이 시작되지만 모든
세션이 한 번에 도착하지 않고 각 2초 구간 안에서 확률적으로 분산된다.

실현된 세션 시작 간격은 다음과 같다.

| 통계 | 값 |
| --- | ---: |
| Mean | 2.0000 s |
| p50 | 2.0036 s |
| p90 | 3.1032 s |
| p99 | 3.7073 s |
| Max | 3.9563 s |

마지막 신규 세션은 약 8,190.14초에 시작했다.

### 6.3 Same-session turn gap

Successor turn의 think time은 세 개의 clipped log-normal mixture로 생성했다.

| Weight | Median | Log sigma | Clip |
| ---: | ---: | ---: | ---: |
| 0.25 | 2 s | 0.8 | 0.1~20 s |
| 0.45 | 60 s | 0.9 | 5~600 s |
| 0.30 | 600 s | 0.8 | 60~3,600 s |

실현된 21,845개 successor gap은 다음과 같다.

| 통계 | 값 |
| --- | ---: |
| Mean | 284.390 s |
| p50 | 68.678 s |
| p90 | 840.507 s |
| p99 | 2,577.520 s |
| Max | 3,600 s |

Metadata에는 호환성을 위해 single-lognormal용 `think_median=1000`,
`think_log_sigma=0.001` 등의 필드도 남아 있지만, `think_profile=long_mixed`일
때 이 scalar 값은 사용되지 않는다.

### 6.4 Completion-relative successor arrival

Successor turn은 이전 요청의 arrival을 기준으로 하지 않는다. 이전 요청의
terminal DECODE가 완료된 시각에 think time을 더해 다음 요청을 도착시킨다.

```text
arrival(successor) = completion(predecessor) + sampled think_time
```

따라서 신규 세션 유입은 0.5 sessions/s지만 모든 session turn을 합친 실제
request arrival은 평균 약 1.498 requests/s가 된다. 마지막 request arrival은
약 17,315.89초, 마지막 완료는 약 17,328.96초였다.

### 6.5 Token 길이

| 분포 | Mean | p50 | p90 | p99 | Min~Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| New input / turn | 9,198.15 | 8,923 | 12,447 | 16,048.6 | 2,494~21,929 |
| Output / turn | 1,149.77 | 1,115 | 1,556 | 2,005.8 | 312~2,741 |
| Final context / session | 65,536.00 | 64,512 | 78,768 | 91,591.2 | 49,248~98,064 |

전체 input은 238,609,280 tokens, output은 29,826,160 tokens로 정확히 8:1이다.

### 6.6 PREFILL minimum과 materialization

```text
theoretical minimum = current-turn new input

implementation minimum
  = new input + preceding decode + successor당 최대 한 block

materialized PREFILL
  = new input + all prior context
```

| 집계 | Tokens | Theoretical minimum 대비 |
| --- | ---: | ---: |
| Theoretical minimum | 238,609,280 | 1.000× |
| Implementation minimum | 264,068,435 | 1.107× |
| Fully materialized | 954,774,498 | 4.001× |
| 실제 CPU-OFF scheduled | 448,862,722 | 1.881× |
| Oracle scheduled | 263,882,994 | 1.106× |

Oracle은 block 단위 오차 안에서 implementation minimum에 도달했다. 반면
CPU-OFF는 GPU direct hit 때문에 fully materialized 값보다 505.9M tokens 적게
계산했다.

## 7. 실행 절차와 산출물

### 7.1 실행 파일과 공통 설정

- Binary: `cpp/build/Release/frontier_sim.exe`
- Base config: `cpp/experiments/kimi_k2_cpu_dram/configs/base_pdd.json`
- Runner: `cpp/experiments/kimi_k2_cpu_dram/run_sweep.py`
- Output mode: `requests`
- Runtime validation: `false`
- 모든 case가 같은 workload CSV와 seed를 공유하는 paired comparison

각 run에서 runner가 사용한 명령은 해당 `run.json`의 `command` 배열에 그대로
기록되어 있다. 형태는 다음과 같다.

```powershell
cpp/build/Release/frontier_sim.exe `
  --config <case>/config.input.json `
  --workload cpp/experiments/kimi_k2_cpu_dram/workloads_profiles/long_mixed_steady4096_r0p5/seed_20260803.csv `
  --output-dir <case> `
  --output-mode requests `
  --runtime-validation false
```

`config.input.json`은 runner가 base config에 capacity를 적용한 파일이고,
`config.normalized.json`은 simulator가 실제 사용한 정규화 설정이다.

### 7.2 결과 provenance

최종 표는 동일 workload를 사용한 두 실행 위치를 합쳤다.

- OFF:
  `outputs/kimi_k2_gap_screen/long_mixed_steady4096_r0p5/`
- 4 TB:
  `outputs/kimi_k2_gap_screen/long_mixed_steady4096_r0p5_cpu4000/`
- 0.5~3.5 TB 및 oracle:
  `outputs/kimi_k2_gap_sweep_r0p5/`

OFF와 4 TB가 별도 디렉터리에 있는 이유는 gap screening 단계에서 먼저 실행한
동일 workload 결과를 재사용했기 때문이다. 나머지 capacity는 후속 sweep으로
추가했다.

각 case 디렉터리에는 다음 파일이 있다.

| 파일 | 내용 |
| --- | --- |
| `config.input.json` | runner가 만든 입력 설정 |
| `config.normalized.json` | simulator가 사용한 최종 설정 |
| `workload.normalized.csv` | simulator가 읽은 정규화 workload |
| `requests.csv` | 요청별 lifecycle, cache hit, routing, transfer 지표 |
| `summary.json` | 시스템 집계 결과 |
| `run.json` | 명령, 입력 경로, 상태, runner wall-clock |

### 7.3 요청 배치 기록

`requests.csv`에는 다음 target 정보가 요청별로 기록된다.

```text
prefill_replica_id, prefill_dp_id
decode_replica_id,  decode_dp_id
```

이번 실험은 replica 0과 DP 0~3을 사용했고 각 DP에 약 6,485개 요청이
배치되었다. 물리 `gpu_id`는 기록하지 않는다. 연속 rank 배치를 가정하면 DP 0,
1, 2, 3은 각각 GPU 0~3, 4~7, 8~11, 12~15의 TP4 group에 대응한다. MLA KV는
해당 TP4 group의 네 GPU에 복제된다.

## 8. 지표 정의

### 8.1 PREFILL

`summary.json / prefill_work.scheduled_prefill_tokens`를 사용했다.

```text
PREFILL reduction(case)
  = (scheduled_OFF - scheduled_case) / scheduled_OFF
```

### 8.2 Cache hit

```text
GPU direct blocks = prefix hit blocks - CPU hit blocks
CPU added blocks  = cpu_kv_cache.hit_blocks
Combined hit rate = prefix_cache.hit_rate
CPU recovery      = CPU added blocks / oracle CPU added blocks
```

CPU raw query에는 신규 token처럼 원래 복구할 수 없는 block도 포함되므로,
capacity progress를 비교할 때 raw CPU hit rate 대신 oracle recovery를 함께 봤다.

### 8.3 TTFT

Capacity 비교의 TTFT는 첫 turn을 제외한 21,845개 successor request만 집계했다.

```text
TTFT = first_token_completed_at - arrived_at
```

첫 turn은 CPU session reuse의 대상이 아니므로 capacity 효과를 희석하지 않도록
주 TTFT 표에서 제외했다.

### 8.4 TPOT와 decode queue proxy

TPOT은 모든 요청의 output token 간 평균 시간을 사용했다.

Decode queue wait 전용 timestamp는 현재 결과에 없다. 대신 다음 값을 proxy로
사용했다.

```text
decode-to-first = first_token_completed_at - decode_arrived_at
```

이 값에는 순수 queue wait뿐 아니라 첫 decode iteration service time도 포함된다.
대표적으로 p50은 약 40.4~40.5 ms, p90은 약 55.5 ms였고 capacity에 따른
유의미한 변화는 없었다.

### 8.5 Throughput

```text
request throughput = completed requests / simulation window
```

이 workload는 finite completion-relative trace다. 따라서 약 1.497 req/s라는
결과는 open-loop maximum serving capacity가 아니며 long think-time drain tail의
영향을 받는다.

### 8.6 Simulator wall-clock

표의 wall-clock은 `run.json / wall_clock_seconds`를 사용한다. 이는 runner가
측정한 simulator process 실행시간이며 CSV 후처리와 HTML 생성시간은 제외한다.

## 9. 전체 실험 결과

### 9.1 PREFILL, cache hit 및 successor TTFT

| Grace CPU DRAM | PREFILL (M) | 절감 vs OFF | GPU direct | CPU added | Combined | CPU recovery/oracle | TTFT mean | TTFT p90 | TTFT p99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OFF | 448.863 | 0.00% | 52.998% | 0.000% | 52.998% | 0.00% | 660.6 ms | 1,247.9 ms | 6,929.8 ms |
| 0.50 TB | 407.621 | 9.19% | 52.999% | 4.320% | 57.319% | 22.33% | 612.9 ms | 1,149.7 ms | 6,507.7 ms |
| 1.00 TB | 308.952 | 31.17% | 53.020% | 14.635% | 67.655% | 75.65% | 518.7 ms | 714.9 ms | 6,655.1 ms |
| 1.25 TB | 287.882 | 35.86% | 53.001% | 16.861% | 69.862% | 87.16% | 505.8 ms | 578.1 ms | 6,868.1 ms |
| 1.50 TB | 276.695 | 38.36% | 53.030% | 18.005% | 71.034% | 93.07% | 492.6 ms | 498.7 ms | 6,861.4 ms |
| 1.75 TB | 271.435 | 39.53% | 53.032% | 18.554% | 71.585% | 95.91% | 488.3 ms | 482.5 ms | 6,858.7 ms |
| 2.00 TB | 268.657 | 40.15% | 53.027% | 18.849% | 71.876% | 97.44% | 486.1 ms | 469.0 ms | 6,870.2 ms |
| 2.25 TB | 266.466 | 40.64% | 53.032% | 19.074% | 72.106% | 98.60% | 483.5 ms | 464.0 ms | 6,868.8 ms |
| 2.50 TB | 265.218 | 40.91% | 53.026% | 19.210% | 72.237% | 99.31% | 481.6 ms | 459.9 ms | 6,843.2 ms |
| 2.75 TB | 263.883 | 41.21% | 53.032% | 19.345% | 72.377% | 100.00% | 480.6 ms | 454.8 ms | 6,859.1 ms |
| 3.00 TB | 263.883 | 41.21% | 53.032% | 19.345% | 72.377% | 100.00% | 480.6 ms | 454.8 ms | 6,859.1 ms |
| 3.50 TB | 263.883 | 41.21% | 53.032% | 19.345% | 72.377% | 100.00% | 480.6 ms | 454.8 ms | 6,859.1 ms |
| 4.00 TB | 263.883 | 41.21% | 53.032% | 19.345% | 72.377% | 100.00% | 480.6 ms | 454.8 ms | 6,859.1 ms |
| Oracle | 263.883 | 41.21% | 53.032% | 19.345% | 72.377% | 100.00% | 480.6 ms | 454.8 ms | 6,859.1 ms |

### 9.2 Decode 및 throughput invariants

| Grace CPU DRAM | Mean TPOT | Decode avg batch | Requests/s | Simulator wall-clock |
| --- | ---: | ---: | ---: | ---: |
| OFF | 26.824 ms | 15.155 | 1.497166 | 61.582 s |
| 0.50 TB | 26.824 ms | 15.165 | 1.497005 | 78.642 s |
| 1.00 TB | 26.817 ms | 15.179 | 1.497118 | 77.132 s |
| 1.25 TB | 26.815 ms | 15.169 | 1.497012 | 77.895 s |
| 1.50 TB | 26.819 ms | 15.185 | 1.497013 | 75.443 s |
| 1.75 TB | 26.819 ms | 15.173 | 1.496975 | 76.725 s |
| 2.00 TB | 26.819 ms | 15.175 | 1.496974 | 79.194 s |
| 2.25 TB | 26.819 ms | 15.176 | 1.496968 | 80.161 s |
| 2.50 TB | 26.819 ms | 15.176 | 1.497003 | 86.003 s |
| 2.75 TB | 26.819 ms | 15.175 | 1.496967 | 79.150 s |
| 3.00 TB | 26.819 ms | 15.175 | 1.496967 | 84.791 s |
| 3.50 TB | 26.819 ms | 15.175 | 1.496967 | 87.699 s |
| 4.00 TB | 26.819 ms | 15.175 | 1.496967 | 73.129 s |
| Oracle | 26.819 ms | 15.175 | 1.496967 | 95.843 s |

Oracle을 제외한 13개 capacity process의 평균 wall-clock은 78.3초, 범위는
61.6~87.7초, 순차 합계는 약 16분 57.5초였다. Oracle은 95.8초였고 전체
순차 합계는 약 18분 33.4초였다. 호스트 부하의 영향을 받으므로 wall-clock을
capacity 성능지표로 해석하지 않는다.

### 9.3 CPU cache 동작

| Case | Restore ops | Restore blocks | Evicted blocks | Terminal resident | H2D queue 합 | H2D service 합 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.50 TB | 1,309 | 2,578,021 | 23,862,083 | 4.000 TB | 15.6 ms | 8,294.8 ms |
| 1.00 TB | 4,374 | 8,731,907 | 16,094,459 | 8.000 TB | 132.4 ms | 28,094.3 ms |
| 1.75 TB | 5,541 | 11,069,270 | 11,349,289 | 14.000 TB | 109.0 ms | 35,614.6 ms |
| 2.00 TB | 5,634 | 11,245,703 | 10,375,214 | 16.000 TB | 96.2 ms | 36,182.3 ms |
| 2.75 TB | 5,774 | 11,541,313 | 7,675,442 | 22.000 TB | 114.9 ms | 37,133.3 ms |
| 4.00 TB | 5,774 | 11,541,313 | 3,673,134 | 32.000 TB | 114.9 ms | 37,133.3 ms |
| Oracle | 5,774 | 11,541,313 | 0 | 41.178 TB | 114.9 ms | 37,133.3 ms |

2.75 TB에서도 eviction은 남아 있지만, eviction된 snapshot이 이후 다시 요청되지
않아 유효 CPU hit와 PREFILL 결과는 oracle과 동일했다. 따라서 “eviction 0”보다
“향후 재사용되는 prefix를 모두 보존했는가”가 saturation 판단에 더 중요하다.

## 10. 결과 해석

### 10.1 PREFILL 감소가 TTFT를 실제로 개선했는가

그렇다. OFF에서 2.75 TB로 이동하면 scheduled PREFILL은 448.863M에서
263.883M으로 41.21% 감소했고 successor TTFT mean은 660.6 ms에서 480.6 ms로
약 27.2% 개선되었다. p90은 1,247.9 ms에서 454.8 ms로 약 63.6% 개선되었다.

PREFILL token 감소율과 TTFT 개선율이 같지 않은 이유는 다음과 같다.

- TTFT에는 PREFILL 계산 외에 scheduler wait, KV transfer 및 첫 decode가 포함된다.
- 긴 PREFILL이 줄면서 queueing tail도 함께 감소해 p90 효과가 평균보다 컸다.
- p99는 약 6.86초로 남았으며 decode tail에 더 큰 영향을 받았다.

### 10.2 Throughput이 거의 변하지 않은 이유

Request throughput은 약 1.497 req/s로 평평하다. workload가 고정된 finite
completion-relative trace이고 simulation 종료시각이 long turn-gap drain 및
decode tail에 지배되기 때문이다. 이 값은 서버의 포화 처리량 측정이 아니다.

### 10.3 Capacity knee

| 판단 기준 | 결과 |
| --- | --- |
| 95% oracle recovery 최초 달성 | 1.75 TB |
| 단순 운영 반올림 지점 | 2.00 TB |
| PREFILL/TTFT 완전 포화 | 2.75 TB |
| 물리 GB300 범위 | 0.50 TB 이하 |

현재 workload와 inclusive 정책에서는 물리 상한 기준점인 0.5 TB에서도 PREFILL이
9.19% 감소하고 TTFT mean이 약 7.2% 개선되었다. 다만 oracle 효과의 22.33%만
복구한다. 95% knee는 물리 범위를 크게 넘어선다.

## 11. 검증 상태 및 한계

### 11.1 확인된 invariant

- 모든 case에서 25,941 requests가 완료됐다.
- 모든 case에서 preemption은 0회였다.
- CPU 활성 run 종료 시 pending restore, staged payload, active restore lease,
  active offload reservation은 모두 0이었다.
- Oracle에서 eviction, truncation 및 skipped offload는 0이었다.
- 2.75 TB 이상과 oracle의 PREFILL, cache hit 및 latency 결과가 동일했다.
- 모든 capacity가 동일 seed와 workload를 사용했다.

### 11.2 주요 한계

1. **단일 seed**: 현재 결과는 seed 20260803 한 개다. confidence interval이나
   seed 간 변동을 평가하지 않았다.
2. **300% actual-PREFILL gate 미충족**: CPU-OFF GPU hit가 53%여서 실제 추가
   PREFILL은 88.1%다.
3. **Analytical GPU model**: GB300 실측 kernel/profile 결과가 아니다.
4. **비물리 capacity**: 0.5 TB 초과는 제품 구성이 아닌 what-if extension이다.
5. **Runtime validation 비활성화**: 속도를 위해 `--runtime-validation false`로
   실행했다. 종료 시 accounting 지표는 확인했지만 hot-path invariant 검사는
   매 event마다 실행하지 않았다.
6. **Decode KV 반환 없음**: PREFILL snapshot만 CPU에 남는다.
7. **Shared CPU contention 단순화**: 두 GPU가 Grace CPU 하나를 공유하는 효과를
   용량과 대역폭 절반으로 환산했으며 NUMA/memory-controller contention을 별도로
   모델링하지 않았다.
8. **Finite trace throughput**: req/s는 open-loop saturation throughput이 아니다.
9. **GPU active-KV 시계열 없음**: 기존 산출물에는 request lifecycle과 DP target은
   있지만 scheduler iteration별 `active_blocks` 시계열은 없다. 기존 데이터로는
   근사만 가능하며 정확한 HBM occupancy 추적에는 추가 계측과 재실행이 필요하다.

## 12. 후속 실험 권장안

1. 동일한 frozen r0.5 workload를 최소 3개 seed로 반복해 knee의 안정성을 본다.
2. 초기 요구사항인 CPU-OFF 실제 추가 PREFILL 280~320%를 반드시 유지해야 한다면
   GPU hit를 낮추는 별도 calibrated workload를 만들고 본 결과와 분리한다.
3. GPU KV manager의 occupancy-change event마다 `active_blocks`,
   `resident_blocks`, `evictable_blocks`, `available_blocks`를 기록한다.
4. occupancy 원본은 event 단위로 저장하고 보고서에서는 1초 및 5초 bin으로
   downsample한다.
5. 물리 0~0.5 TB 구간의 실제 변화 모양이 필요하면 0.05~0.1 TB 간격의 fine
   sweep을 추가한다.
6. 실제 GB300 profiling을 확보하면 analytical roofline 결과와 절대 latency를
   다시 정렬한다.

## 13. 기준 파일

- 최신 HTML 보고서:
  `outputs/kimi_k2_gap_sweep_r0p5/experiment_report.html`
- Workload metadata:
  `cpp/experiments/kimi_k2_cpu_dram/workloads_profiles/long_mixed_steady4096_r0p5/seed_20260803_metadata.json`
- Workload manifest:
  `cpp/experiments/kimi_k2_cpu_dram/workloads_profiles/long_mixed_steady4096_r0p5/seed_20260803_manifest.csv`
- Base config:
  `cpp/experiments/kimi_k2_cpu_dram/configs/base_pdd.json`
- Sweep runner:
  `cpp/experiments/kimi_k2_cpu_dram/run_sweep.py`
- Current sweep:
  `outputs/kimi_k2_gap_sweep_r0p5/`
- OFF control:
  `outputs/kimi_k2_gap_screen/long_mixed_steady4096_r0p5/`
- 4 TB endpoint:
  `outputs/kimi_k2_gap_screen/long_mixed_steady4096_r0p5_cpu4000/`

