# 주요 오픈 모델의 context length와 KV 캐시 변화 조사

## 1. 조사 목적

이 문서는 보고서 제1장 1.2절의 모델 구조 변화 설명을 Kimi K2와 Kimi K3의 단일 비교에서 주요 open-weight 모델의 시계열 비교로 확장하기 위한 기초 자료다.

조사의 핵심 질문은 다음 두 가지다.

1. 주요 open-weight 모델의 출시 시점에 따라 모델 규모와 지원 context length가 어떻게 변했는가?
2. context length가 증가할 때 세션당 KV 캐시는 attention 구조에 따라 어떤 곡선을 그리는가?

여기서 `open model`은 엄격한 OSI 의미의 open source가 아니라, 가중치가 공개되어 구조 정보를 확인할 수 있는 **open-weight model**을 뜻한다. 모델별 라이선스 조건은 서로 다르다.

이 조사로 생성한 보고서용 그림은 다음과 같다.

- [`open-model-release-trends.svg`](../presentation_assets/open-model-release-trends.svg): 출시일별 모델 규모와 context length
- [`open-model-kv-cache-curves.svg`](../presentation_assets/open-model-kv-cache-curves.svg): context length별 세션당 논리 KV 캐시

## 2. 생성할 그림

### 2.1 그림 A — 출시일에 따른 모델 규모와 context length

한 그림에 모든 값을 억지로 넣기보다 두 패널을 권장한다.

- **상단 패널:** x축 출시일, y축 총 파라미터 수(log scale)
- **하단 패널:** x축 출시일, y축 advertised context length(log scale)
- Dense model은 원, MoE model은 마름모로 표시한다.
- MoE model은 총 파라미터를 큰 marker, 활성 파라미터를 같은 x 위치의 작은 marker로 표시하고 선으로 연결한다.
- `trained/native context`와 `advertised/extended context`가 다른 모델은 같은 x 위치에서 두 점을 점선으로 연결한다.

이 구성을 사용하면 다음 변화가 동시에 드러난다.

- 2023년의 수천~수만 token context가 2024~2025년에 128K급으로 확대됨
- 2025~2026년에 1M~10M을 표방하는 모델이 등장함
- Dense model의 단순 확대뿐 아니라 MoE를 통한 총 파라미터 확대가 진행됨
- context window의 확대가 항상 native training length의 동일한 확대를 뜻하지는 않음

원자료는 [`data/open-model-release-timeline.csv`](data/open-model-release-timeline.csv)에 정리하였다.

### 2.2 그림 B — context length에 따른 세션당 논리 KV 캐시

이 그림은 x축을 context length(tokens, log scale), y축을 세션당 KV cache(GiB, log scale)로 구성한다. 비교 기준은 다음과 같이 고정한다.

- KV precision: BF16, 2 bytes/element
- batch 또는 session 수: 1
- 물리 TP·PP·DP 복제 및 shard: 제외
- block allocator rounding, fragmentation 및 watermark: 제외
- 모델 가중치, activation 및 temporary workspace: 제외
- 각 모델의 공식 attention 구조와 효율적인 rolling/hybrid cache 정책: 반영

따라서 이 그림은 실제 GPU 한 장의 메모리 사용량이 아니라, **모델 구조가 세션 하나에 요구하는 논리 KV 상태량**을 비교한다. 실제 장비 용량으로 환산할 때는 병렬화와 cache implementation을 별도로 적용해야 한다.

계산식 원자료는 [`data/open-model-kv-cache-formulas.csv`](data/open-model-kv-cache-formulas.csv)에 정리하였다.

## 3. 기본 KV 계산식

### 3.1 MHA·GQA·MQA의 full attention

일반적인 full attention에서 세션 하나의 KV cache bytes는 다음과 같다.

```text
KV bytes(N)
= N
  × number_of_layers
  × 2                       # Key + Value
  × number_of_kv_heads
  × head_dim
  × bytes_per_element
```

GQA의 핵심은 Query head 수가 아니라 `number_of_kv_heads`가 cache 크기를 결정한다는 점이다. 예를 들어 Llama 3.1 405B는 128개의 Query head가 아니라 8개의 KV head를 사용하므로, MHA로 가정한 계산보다 KV가 크게 작다.

### 3.2 MLA

DeepSeek-V2/V3 및 Kimi K2 유형의 MLA는 head별 K와 V 전체를 저장하는 대신 compressed latent와 decoupled RoPE key를 저장한다. 본 조사에서는 다음 구조적 BF16 계산을 사용한다.

```text
MLA KV bytes(N)
= N
  × number_of_mla_layers
  × (kv_lora_rank + qk_rope_head_dim)
  × bytes_per_element
```

DeepSeek-V3와 Kimi K2는 `kv_lora_rank=512`, `qk_rope_head_dim=64`를 사용한다. 61개 MLA 계층과 128K context를 BF16으로 정규화하면 약 8.58 GiB다. 실제 serving에서는 latent와 RoPE cache에 서로 다른 정밀도를 사용할 수 있으므로, 이 값은 구조 비교용 기준이다.

Frontier의 Kimi K2 실험처럼 latent를 FP8, RoPE를 BF16으로 저장하면 128K의 논리 cache는 약 4.77 GiB가 된다. TP rank마다 cache를 복제하는 구성에서는 이 논리값에 물리 복제량이 추가된다.

### 3.3 Sliding-window attention

모든 계층이 window `W`만 보존하고 backend가 rolling cache를 구현하면 다음과 같이 포화된다.

```text
Sliding KV bytes(N)
= min(N, W)
  × per_token_full_KV_bytes
```

Mistral 7B v0.1은 32K sequence를 받을 수 있지만 각 계층의 sliding window는 4,096 tokens다. 효율적인 rolling cache에서는 BF16 세션 cache가 약 0.5 GiB에서 포화한다. 반면 backend가 과거 KV를 버리지 않고 모두 보존하면 32K에서 4.0 GiB까지 증가한다.

따라서 그래프에는 다음 두 선을 함께 표시하는 것이 좋다.

- 실선: rolling cache 적용값
- 옅은 점선: full-retention 상한

### 3.4 Local/global hybrid attention

일부 계층만 full attention이고 나머지가 local window를 사용하면 cache는 완전히 포화하지 않는다.

```text
Hybrid KV bytes(N)
= per_layer_KV_bytes
  × (global_layers × N
     + local_layers × min(N, W))
```

window 이후에는 local 계층의 cache가 고정되고 global 계층만 선형으로 증가한다. 따라서 곡선은 꺾이지만 계속 상승한다.

대표 사례는 다음과 같다.

- **Gemma 3 27B:** 62개 계층 중 52개 local, 10개 global, local window 1,024
- **Llama 4 Scout:** 48개 계층 중 36개 chunked, 12개 full, chunk size 8,192

Llama 4 Scout는 10M context를 지원하지만 4개 계층 중 1개는 여전히 full attention이므로, BF16 논리 KV가 완전히 고정되지는 않는다. 효율적인 chunk cache를 가정해도 10M에서 약 481.1 GiB이며, 모든 계층의 KV를 보존하는 상한은 약 1,920 GiB다.

### 3.5 Linear/recurrent attention과 KDA hybrid

순수 recurrent attention 계층은 최신 fixed-size state만 유지할 수 있다. 그러나 Kimi K3처럼 recurrent 계층과 full-attention 계층을 섞은 모델은 다음 형태다.

```text
Hybrid recurrent KV bytes(N)
= fixed_recurrent_state_bytes
  + N × full_attention_layer_slope
```

Frontier의 Kimi K3 모델에서는 69개 KDA 계층의 fixed snapshot이 232,316,928 bytes이고, 24개 MLA 계층만 context에 비례하여 증가한다. 1M context의 BF16 논리 상태는 약 27.22 GiB다. 즉 KDA가 증가 기울기를 줄이지만 전체 세션 cache를 완전한 상수로 만들지는 않는다.

## 4. Sparse attention을 그래프에 넣는 원칙

Sparse attention은 하나의 공통식으로 처리하면 안 된다. `attention 연산이 sparse하다`는 사실과 `resident KV storage가 줄어든다`는 사실은 서로 다르기 때문이다.

### 4.1 연산만 sparse하고 KV는 모두 보존하는 경우

Top-k retrieval이나 block-sparse kernel이 일부 KV만 읽더라도, 후보 선택을 위해 전체 과거 KV 또는 별도 index를 보존할 수 있다. 공식 문서에 cache 압축·축출 규칙이 없다면 KV 그래프에서는 full-retention 값을 유지하고 다음 주석만 붙이는 것이 안전하다.

> Attention compute/read traffic is sparse; resident KV reduction is not established.

### 4.2 window 밖의 KV를 폐기할 수 있는 경우

Sliding-window 또는 chunked attention처럼 오래된 KV가 다시 참조되지 않는 구조는 효율적인 rolling cache를 가정한 포화 곡선을 그릴 수 있다. 단, framework가 실제로 과거 KV를 폐기하는지에 따라 결과가 달라지므로 full-retention 상한을 함께 표시한다.

### 4.3 local과 global 계층이 섞인 경우

Gemma 3와 Llama 4처럼 local 계층 사이에 global 계층이 존재하면 `global linear + local plateau` 공식을 사용한다. 이 경우 sparse/local attention은 KV 증가율을 낮추지만 증가를 제거하지 않는다.

### 4.4 token compression이 명시된 경우

DeepSeek-V4는 compressed sparse attention(CSA)과 heavily compressed attention(HCA)을 혼합한다. 공식 자료는 CSA compression rate 4, HCA compression rate 128 및 1M context에서 DeepSeek-V3.2 대비 약 10%의 KV cache라는 결과를 제시한다. 그러나 실제 cache에는 compressed entries, local branch 및 indexer 상태가 함께 포함되므로 단순히 `context/4` 또는 `context/128`만 적용하면 안 된다.

따라서 DeepSeek-V4는 첫 번째 그래프 버전에서 다음 중 하나로 처리하는 것이 좋다.

1. 1M 지점에 공식 상대값을 별도 marker로 표시
2. 정확한 implementation-level cache formula를 확정할 때까지 정량 곡선에서 제외하고 구조 변화 주석으로만 표시

본 보고서의 1장에는 **두 번째 방법**을 우선 권장한다. 그래프의 나머지 선은 절대 GiB인데 DeepSeek-V4만 상대값을 절대값으로 환산하면 비교 기준이 불균일해질 수 있기 때문이다.

## 5. 1차 계산 결과

다음 값은 각 모델의 advertised context 끝점에서 계산한 단일 세션 BF16 논리 KV다.

| 모델 | Context | Attention/cache 가정 | BF16 논리 KV |
| --- | ---: | --- | ---: |
| Llama 2 70B | 4K | GQA full | 1.25 GiB |
| Mistral 7B v0.1 | 32K | 4K rolling window | 0.50 GiB |
| Mixtral 8x7B | 32K | GQA full | 4.00 GiB |
| Llama 3 70B | 8K | GQA full | 2.50 GiB |
| DeepSeek-V2 | 128K | MLA, BF16 normalized | 8.44 GiB |
| Llama 3.1 405B | 128K | GQA full | 63.00 GiB |
| Qwen2.5-72B | 128K | GQA full | 40.00 GiB |
| DeepSeek-V3 | 128K | MLA, BF16 normalized | 8.58 GiB |
| Gemma 3 27B | 128K | 52 local + 10 global | 10.41 GiB |
| Qwen3-235B-A22B | 128K | GQA full, extended context | 23.50 GiB |
| Kimi K2 | 128K | MLA, BF16 normalized | 8.58 GiB |
| Llama 4 Scout | 10M | 36 chunked + 12 full | 481.13 GiB |
| Kimi K3 | 1M | fixed KDA + 24 MLA | 27.22 GiB |

이 표에서 중요한 것은 최신 모델일수록 KV가 일관되게 커진다는 결론이 아니다. 오히려 context window는 수십~수천 배 확대되었지만 GQA, MLA, local/global hybrid 및 recurrent state가 증가 기울기를 서로 다른 방식으로 낮추고 있다는 점이다.

## 6. 보고서 1.2절에 권장하는 메시지

기존의 Kimi K2와 Kimi K3 중심 설명은 다음과 같은 업계 전체의 변화로 확장할 수 있다.

> 주요 open-weight 모델의 지원 context length는 2023년 수천~수만 tokens에서 2025~2026년 1M 이상으로 확대되었다. 그러나 세션당 KV cache는 context length만으로 결정되지 않는다. GQA는 KV head 수를 줄이고, MLA는 토큰당 저장 표현을 압축하며, sliding·chunked attention은 일부 계층의 resident history를 제한한다. Linear attention은 과거 문맥을 fixed-size state로 바꾸지만, full-attention 계층이 혼합된 모델에서는 KV가 더 낮은 기울기로 계속 증가한다. 따라서 같은 128K 또는 1M context를 지원하는 모델이라도 필요한 메모리 용량과 증가 곡선은 크게 다르다.

Kimi K2와 Kimi K3는 이 전체 그래프에서 MLA와 KDA hybrid를 보여주는 두 사례로 남기는 것이 좋다. 두 모델을 절의 중심으로 삼기보다, MHA/GQA → MLA → local/sparse hybrid → recurrent hybrid로 이어지는 구조 변화의 일부로 배치하면 보고서의 일반성과 수명이 길어진다.

## 7. 데이터 사용 시 주의사항

- `advertised context`, checkpoint의 `max_position_embeddings`, 실제 training length는 다를 수 있다.
- context 지원 여부와 해당 길이에서의 정확도·처리 가능성은 동일한 개념이 아니다.
- 모델 가중치 정밀도와 KV cache 정밀도는 별개다.
- MoE의 total/active parameter 차이는 weight memory와 연산량에 중요하지만, attention KV는 attention 계층 구조로 계산해야 한다.
- TP·PP·DCP와 cache replication/sharding을 적용하면 물리 장비별 KV는 본 자료와 달라진다.
- paged KV allocator의 block rounding, fragmentation, watermark 및 temporary workspace는 본 계산에서 제외했다.
- multimodal model은 image token 수가 실제 context와 KV 점유량에 포함될 수 있다.
- sparse attention은 공식적으로 cache 압축 또는 폐기가 확인된 경우에만 KV 절감으로 계산해야 한다.

## 8. 주요 공식 출처

- [Meta Llama 2 발표](https://ai.meta.com/blog/llama-2/)
- [Meta Llama 3 발표](https://ai.meta.com/blog/meta-llama-3/)
- [Meta Llama 3.1 발표](https://ai.meta.com/blog/meta-llama-3-1/)
- [Meta Llama 4 발표](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)
- [Mistral 7B 발표](https://mistral.ai/news/announcing-mistral-7b/)
- [Mixtral 8x7B 발표](https://mistral.ai/news/mixtral-of-experts/)
- [Mistral Large 2 발표](https://mistral.ai/news/mistral-large-2407/)
- [Qwen2 발표](https://qwenlm.github.io/blog/qwen2/)
- [Qwen2.5 발표](https://qwenlm.github.io/blog/qwen2.5/)
- [Qwen3 발표](https://qwenlm.github.io/blog/qwen3/)
- [DeepSeek-V2 저장소](https://github.com/deepseek-ai/DeepSeek-V2)
- [DeepSeek-V3 발표](https://deepseek.com/en/news/deepseek-v3/)
- [DeepSeek-V4 발표](https://deepseek.com/en/news/v4-preview/)
- [DeepSeek-V4 기술 보고서](https://arxiv.org/abs/2606.19348)
- [Gemma 3 발표](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-3/)
- [Kimi K2 저장소](https://github.com/MoonshotAI/Kimi-K2)
- [Kimi K3 발표](https://www.kimi.com/en/blog/kimi-k3)
