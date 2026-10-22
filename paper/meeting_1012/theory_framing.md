# Theory framing for self-speculative chunk generation (meeting item G, 10/13)

목적: 10/12 미팅의 "theory framing" 요청에 대한 초안. 우리가 측정한 숫자(facts §15, §21–24)를 하나의 틀로 묶고, 논문에 들어갈 명제(claim)와 그 근거를 정리한다. 수식은 명제 수준이고, 증명이 필요한 것은 "가정"으로 표시했다. 숫자 출처는 각 줄에 적었다.

## 1. Setup

- Target sampler T: 체인 길이 K = 4 step의 causal chunk 생성기 (Fast 14B). chunk c에서 noise z₀ 와 committed history h_c 로부터 x̂₁, …, x̂_K 를 차례로 만들고 x_K 를 commit 한다. step k 의 x0 추정 x̂_k 는 z₀, h_c, 그리고 step 2..k 에서 뽑힌 noise ε₂..ε_k 의 함수다.
- Token gap: d_k(u) = ‖x̂_k(u) − x_K(u)‖ / σ_T, u = 1×2×2 latent patch, σ_T = x_K 의 token norm 중앙값. ε = 0.24.
- Verifier V: 14B의 마지막 block hidden 위의 conv head. step k forward 의 feature 로부터 d_k(u) 를 예측한다 (AUROC 0.93–0.98, facts §5). 비용은 forward 안에서 공짜다.
- Acceptance rule (chunk-wise): step k 가 끝난 뒤 예측 gap 이 ε 이하인 token 비율이 f* 이상이면 chunk 를 x̂_k 로 commit 하고 다음 chunk 로 간다 (f* = 0.7 이 teacher-parity mode).

## 2. Cost: expected steps and the speed-up ceiling

a_k = P(chunk accepted at step k | not accepted before) 라 하면

E[steps] = 1 + (1 − a₁) + (1 − a₁)(1 − a₂) + (1 − a₁)(1 − a₂)(1 − a₃),   speed-up S = (K · t_step + t_commit) / (E[steps] · t_step + t_commit′).

- 측정 (facts §15): f* = 0.7 → E[steps] = 3.10, S = 1.67×; f* = 0.5 → 2.47, 1.97×. 상한은 E[steps] = 1, 즉 ≈ 4× (fused commit 포함 시 §15 의 "704 fused" 4.66× 가 실측 상한에 가장 가깝다).
- 거부된 chunk 에 낭비되는 계산은 0 이다: x̂_k 는 어차피 step k+1 의 입력이다. 외부 drafter 방식은 거부 시 draft 비용 전체를 버린다 (SDVG τ −0.7: teacher 보다 느림, §13). 이것이 self-draft 의 첫 번째 구조적 이점이다.
- 따라서 속도는 오직 a₁, a₂, a₃ 로 결정되고, 그것은 "T 의 중간 step 이 최종 출력과 얼마나 같은가" 라는 sampler 의 성질이다 (§21: 토큰 기준 34 / 56 / 81 %).

## 3. Quality: what the acceptance rule guarantees and how error propagates

- 수락된 chunk 는 정의상 토큰의 ≥ f* 가 K-step 출력에서 ε 이내다. 즉 teacher-parity mode 는 "분포가 같다"가 아니라 "같은 noise 에서 같은 샘플에 ε-가깝다" 는 paired guarantee 다. 측정: Δquality +0.03 ± 0.07, Δdrift +3 (§2, §17).
- 오류 전파: commit 된 chunk 의 오차는 KV 를 통해 다음 chunk 로 들어간다. §21 의 own-history 실험은 이 전파가 폭발하지 않음을 보여준다: 1.3B 를 자기 history 로 돌릴 때 anchored 대비 추가 gap 이 chunk 1 에서 +0.08, chunk 5 이후 +0.45 로 포화한다. 가정 A (empirical): committed history 에 대한 chunk 생성 map 은 평균적으로 수축적(contractive, λ < 1) 이어서 per-chunk 오차 e 가 O(e/(1 − λ)) 로 제한된다. §21 의 saw-tooth (commit 직후 frame 이 가장 가깝고 chunk 안에서 벌어짐) 는 clean commit pass 가 오차를 매 chunk 리셋하는 메커니즘이다.
- 따라서 verifier 가 보장하는 것은 궤적 거리이고, 품질은 간접적으로 따라온다. §21 의 "거리 ≠ 품질" (|ρ| ≤ 0.2) 은 이 보장이 필요 이상으로 보수적일 수 있음을 뜻한다 (lever 2: f* 완화 실험, 진행 중).

## 4. Why a separate drafter cannot do this (the main claim)

Drafter D 가 chunk 마다 x_D 를 제안하고 T 가 검증한다고 하자. 수락률 P(‖x_D − x_K‖ ≤ ε) 의 상한을 세 경우로 나눈다.

1. **D 가 T 의 noise 와 독립인 경우** (다른 모델, 또는 같은 모델의 다른 샘플). x_D 와 x_K 는 같은 조건에서 뽑힌 두 샘플이므로 수락률은 T 의 자기 일치도(self-agreement)로 상한된다. 측정: 같은 14B 를 다른 noise 로 돌리면 29 % (§21). 어떤 독립 drafter 도, 분포를 완벽히 맞춰도, 이 값을 넘을 수 없다.
2. **D 가 z₀ 에 coupled 되지만 모델이 다른 경우** (1.3B anchored, §21). 수락률은 모델 불일치(capacity) 로 결정된다: 0.68 σ_T, 6 %, chunk index 에 평평. distillation 으로 0.56 σ_T, 11 % (§22) 까지만 움직였다.
3. **D 가 T 자신의 앞 step 인 경우** (self-draft). 같은 z₀, 같은 history, 같은 noise path 를 공유하므로 noise 항과 capacity 항이 모두 0 이고, 남는 것은 "덜 돈 step" 의 오차뿐이다: 34 / 56 / 81 % (§21). step 1 하나만으로 이미 1번 경우의 상한(29 %)을 넘는다.

보조 명제 (회귀 distillation 의 한계, §24): D 를 L2 로 x_K 에 맞추면 최적해는 E[x_K | z₀, h] 이고, x̂₁ 은 ε₂..ε_K 를 모르므로 이 조건부 평균은 뒤 step 의 noise 에 대해 평균된 값이다. 평균은 어떤 샘플보다 texture 가 적다 (14B 자기 distillation 에서도 1-step 영상이 6.0 → 3.3 으로 붕괴). 그러므로 drafter 를 T 의 출력에 맞추려면 분포 매칭 목적함수가 필요한데, 그 경우 다시 1번 경우의 상한(29 %)에 걸린다. 결론: **독립 drafter 의 수락률 상한 < self-draft 의 step-1 수락률**, 그리고 coupled drafter 는 T 자신으로 수렴할 때만 그 상한을 넘는다.

이 주장은 LLM speculative decoding 과 다른 점을 분명히 한다. LLM 에서는 draft 가 exact sampling 으로 교정되므로 분포 일치만 있으면 되지만, 비디오에서는 (i) 보여준 프레임을 되돌릴 수 없고 (ii) 토큰이 연속값이라 "같은 샘플" 이어야 한다. 그래서 paired(coupled) 제안이 필요하고, paired 제안을 공짜로 주는 것은 T 의 자기 궤적뿐이다.

## 5. Latency decomposition and the sub-chunk lever

action→frame 지연 = (chunk 경계까지 대기, 평균 chunk/2) + chunk 계산 (E[steps] · t_step + t_commit) + decode. 3-frame chunk 에서 chunk 자체가 0.75 s 이므로 어떤 계산 최적화도 750 ms 를 못 맞춘다 (§13, p50 1.30 s). chunk 를 m latent frame 으로 줄이면 첫 항은 m/3 배, 둘째 항도 거의 m/3 배 (§23: video-second 당 계산 +8 % at m = 2, +21 % at m = 1). 즉 latency 는 chunk 길이에 거의 선형이고, 모델은 재학습 없이 m = 1, 2 에서 동작한다 (§23). 품질 쪽 비용은 history 가 더 잘게 commit 되는 데서 오는 drift 증가 (m = 1 에서 −0.33, 내용 상실) 이며, 실측은 진행 중이다.

## 6. Where the hybrid pays (deployment regimes)

Guandao 의 틀: T 가 device 에서 돌면 그냥 T 를 돌린다. 우리 수치로 regime 을 나누면
- device 가 T 를 담을 수 없음 (< 17 GiB): cloud T + device decode/SR (realtime mode; latency 는 RTT + chunk 에 묶임).
- device 24–32 GB: T 자체가 들어가고 (§18–19) self-speculation 이 device 에서 그대로 1.67–2.4× 를 준다. cloud 는 선택이다.
- 공통: self-speculation 은 어느 쪽에서 돌든 같은 비율로 이득을 주므로, "어디서 돌리나" 와 "몇 step 도나" 가 직교한다. 이것이 scalable recipe 주장의 형태다: 차세대 base 모델에도 head 만 다시 학습하면 되고 (몇 시간), drafter 는 없다.

## 7. Claims for the paper (one line each, with the supporting number)

1. Self-draft acceptance exceeds any independent drafter's ceiling: 34 % (step 1) vs 29 % (T self-agreement), 6 % (1.3B coupled), 11 % (distilled). [§21–22]
2. Rejection costs nothing: E[steps] 가 곧 비용; 1.67× at parity (Δ +0.03), 1.97× at f* = 0.5. [§15]
3. Regression-distilled first steps are conditional means and lose texture (14B 1-step 6.0 → 3.3). [§24]
4. Paired verification is cheap (head on existing features, AUROC 0.93–0.98) and scene-general (held-out scenes, §17).
5. Error through the KV saturates (+0.45 σ_T plateau) rather than compounding; the clean commit resets it. [§21]
6. Latency is linear in chunk length; the model tolerates 1–2-frame chunks without retraining. [§23]

열린 이론 질문: (a) 가정 A 의 수축성을 KV window 길이의 함수로 측정할 수 있다 (window 12 vs 18 의 품질 무차이 §19 가 힌트). (b) acceptance threshold f* 와 품질의 관계 (lever 2). (c) 독립 drafter 상한 29 % 는 ε 에 의존한다 — ε 를 바꾼 곡선으로 명제 1 을 그림 하나로 만들 수 있다 (§21 의 per-token gap 분포에서 계산 가능, GPU 불필요).
