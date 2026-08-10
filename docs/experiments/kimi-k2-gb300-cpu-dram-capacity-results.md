# Kimi K2 / GB300 CPU DRAM Capacity 실험 결과

## 결론

GB300 physical 범위의 최대점인 500 GB/Grace CPU에서는 inclusive CPU KV
cache가 GPU cache를 유효하게 확장하지 못했다. CPU restorable-prefix hit는
1.89%, scheduled PREFILL 감소는 1.39%였고, session-equal successor TTFT
개선은 -0.07%(session bootstrap 95% CI -0.17%~0.04%)로 유의하지 않았다.

사전 정의 threshold를 처음 넘은 구간은 제품 범위 밖인 672→720 GB/CPU였다.
이는 `rho=1.77→1.89`에 해당한다. 720 GB/CPU에서 PREFILL은 11.85% 감소했지만
TTFT 개선은 0.99%였다. unbounded oracle에서도 PREFILL 감소 72.13%에 비해
TTFT 개선은 5.76%에 그쳐, 현재 analytical model에서는 PREFILL 외 지연이
TTFT 효과를 제한한다.

## 실행 구성

- Model/device: `moonshotai/Kimi-K2-Instruct`, GB300 analytical model
- Architecture: sequential online PDD, parallel clusters disabled
- PREFILL/DECODE: 각각 16 GPUs, 1 replica, TP4/PP1/DP4/MoE-TP1/EP16
- KV: FP8 MLA, 각 attention TP가 전체 KV 저장
- Decode KV 반환: 없음
- Workload: seed당 1,728 sessions, 10,944 requests, final context 평균 65,536
  tokens, aggregate input:output 8:1
- Frozen generator: `kimi-k2-cpu-dram-v3-frozen`
- Seeds: 20260803–20260807

## Workload calibration

초기 768-session workload의 CPU-OFF 실측 추가 PREFILL은 96.0%로 목표에
미달했다. GPU prefix hit가 원인이었으며 session reuse distance를 조정한 뒤
다음 validation 결과로 workload를 동결했다.

| Seed | 실제 추가 PREFILL | GPU prefix hit | Preemption |
| --- | ---: | ---: | ---: |
| 20260803 | 296.28% | 0.815% | 0 |
| 20260804 | 297.22% | 0.767% | 0 |
| 20260805 | 297.01% | 0.774% | 0 |
| pooled | 296.84% | 0.785% | 0 |

## 주요 결과

| Case | ρ | CPU restorable hit | PREFILL 감소 vs OFF | Session-equal TTFT 개선 | 95% bootstrap CI | Successor TTFT p90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| off | 0.00 | 0.00% | 0.00% | 0.00% | 0.00%~0.00% | 46,973.9 ms |
| cpu256 | 0.67 | 0.00% | 0.03% | 0.02% | -0.02%~0.07% | 46,979.0 ms |
| cpu500 | 1.32 | 1.89% | 1.39% | -0.07% | -0.17%~0.04% | 46,971.9 ms |
| cpu672 | 1.77 | 10.10% | 7.31% | 0.73% | 0.62%~0.83% | 46,848.1 ms |
| cpu720 | 1.89 | 16.40% | 11.85% | 0.99% | 0.88%~1.10% | 46,899.9 ms |
| cpu760 | 2.00 | 18.49% | 13.36% | 0.93% | 0.82%~1.05% | 46,829.4 ms |
| cpu1520 | 4.00 | 57.22% | 41.28% | 3.19% | 3.08%~3.30% | 46,431.1 ms |
| oracle | — | 100.00% | 72.13% | 5.76% | 5.64%~5.89% | 45,965.1 ms |

`cpu544` 이상은 GB300 제품 구성이 아닌 non-physical analytical capacity
extension이다. Fine sweep에서 최초의 사전 정의 변화 구간은 `cpu672`와
`cpu720` 사이였다.

## Gate 및 재현성

- Coarse 35/35, analytical extension 10/10, fine 35/35 run 성공
- 각 run 10,944/10,944 requests 완료, preemption 0
- 모든 CPU 활성 run의 terminal reservation/lease/pending/staged state 0
- Oracle eviction/truncation/skip 0, CPU restorable hit 및 GPU+CPU coverage 100%
- 같은 `cpu128` config/workload 재실행의 `requests.csv` SHA-256 일치
- Coarse/fine의 `cpu500`, extension/fine의 `cpu760` endpoint가 모든 seed에서
  동일한 `requests.csv` SHA-256 생성

## 산출물

- [최종 자동 생성 보고서](../../outputs/kimi_k2_cpu_dram_v3/final/final_report.md)
- [Capacity summary CSV](../../outputs/kimi_k2_cpu_dram_v3/final/final_capacity_summary.csv)
- [Turn summary CSV](../../outputs/kimi_k2_cpu_dram_v3/final/final_turn_summary.csv)
- [Session summary CSV](../../outputs/kimi_k2_cpu_dram_v3/final/final_session_summary.csv)
- [그래프 디렉터리](../../outputs/kimi_k2_cpu_dram_v3/final/plots/)

## 해석 범위와 남은 항목

결과는 TP4 전체-KV 복제, sequential PDD, decode KV 반환 없음, analytical
GB300 실행시간/전송 모델에 한정된다. Optional bandwidth sensitivity는 주
질문이 capacity 효과이고 physical 범위에서 유효한 cache extension이 관측되지
않았으므로 실행하지 않았다. 실제 GB300 profiling 기반 latency calibration은
별도 후속 검증이 필요하다.
