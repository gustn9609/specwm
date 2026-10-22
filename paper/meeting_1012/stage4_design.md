# Stage 4 설계안 — 14B causal-fast의 4→2-step 재증류 (DMD, self-forcing 방식) — 2026-10-16

## 0. 왜 지금 이것인가
10/16 밤의 네 회귀(MSE) 학습 라인이 전부 같은 이유로 닫혔다 (facts §46, §46b, §48, §49): 결정적 타깃이라도 낮은 노이즈 노드에서 MSE 최적해는 조건부 평균(blur)이고, autoregressive 루프가 그 잔차를 증폭한다. 남은 유일한 학습 경로는 **분포 손실**로 생성기 자체를 고치는 것이다. 그 중 비디오 world model에서 실제로 작동이 입증된 것은 DMD 계열(CausVid, Self-Forcing, Self-Forcing++)뿐이고, LingBot causal-fast 자체가 그 방식으로 만들어진 4-step 모델이다.

목표: 지금 rt2(nodes 0,500 = σ 1.0 → 0.833 → 0; teacher −0.62 at 15.7 fps)가 하는 **그 2-step 스케줄 그대로**, 생성기 LoRA만 바꿔서 품질을 teacher 쪽으로 올린다. 성공하면 adapt25(§47, −0.05 at 15.3 fps)와 그대로 합성된다(step-1 verifier는 feature 기반이라 재학습만 하면 됨).

## 1. §34(1.3B DMD pilot) 실패의 재해석
§34는 "bid real score가 신호를 못 준다"로 닫혔지만, 다시 보면 그 결론을 내릴 만한 실험이 아니었다:

| item | §34 pilot | 왜 문제인가 |
|---|---|---|
| student | 1.3B causal-fast + (t,r) flow-map 임베딩 + semigroup co-train | 두 가지 새 요소(any-step 임베딩, §32에서 이미 collapse로 확인된 semigroup 항)를 DMD와 동시에 넣음 |
| steps / batch | 115 steps, batch 1 clip | Self-Forcing은 수천 step × batch 64. 100 step은 DMD 손실이 움직이기 전 |
| gradient | last step만, N∈{1,2,3,4} 랜덤 | 1-step 샘플까지 real score에 맞추려 함 → 1.3B 용량으로 불가 |
| lr | gen 1e-5 (LoRA r256) | DMD2 권장 생성기 lr은 full-param 2e-6 수준; LoRA r256 + 1e-5는 100 step 안에 collapse 가능한 크기 |
| 판정 | 6 clip, 100 step에서 품질 하락 | collapse는 "signal 없음"이 아니라 "step이 너무 큼"의 증거일 수도 있음 |

따라서 stage 4는 **student를 14B causal-fast(이미 4-step으로 증류된 모델)로, 스케줄을 고정 2-step으로, 새 임베딩·co-train 없이, 작은 lr로** 간다. 바뀌는 건 LoRA 하나뿐이다.

## 2. 구성

| component | choice | note |
|---|---|---|
| generator G | 14B causal-fast (FP8 weight, bf16 LoRA r64, q/k/v/o/ffn) | 2-step 스케줄 σ = (1.0, 0.833) 고정 = rt2. 추론 경로 `two_machine --student`로 그대로 로드 (lora_util.state 포맷) |
| real score | bid 14B (bid_as_v1_cam, 두 expert, CFG 5, plucker 카메라) | §34 코드 그대로 (dmd_any.real_x0); GPU 하나 전용 (85 GiB at 832 → 704에서 ~70) |
| fake score | 같은 14B causal-fast base + 두 번째 LoRA 'fake' | G와 같은 GPU에서 어댑터 토글 (pilot 방식); G의 rollout에 teacher-forced, flow matching |
| rollout | free-running, 7 chunks(81 frames) at 704×384, 자기 KV | exposure bias 제거 = §48이 죽은 이유를 정면으로 다룸 |
| DMD sample | chunk마다 k∈{1,2} 랜덤: step k의 x0 추정치를 샘플로 사용 | 두 step 모두 gradient를 받음 (Self-Forcing의 random-stop). step-1 x0가 real 분포로 당겨지면 §47 step-1 verifier/§40 early-send 쪽도 이득 |
| gradient | last step만 backprop, 3 chunk/step (Self-Forcing) | 메모리: chunk 단위 checkpoint, lora14_train 실측 81 GiB at 704 1-chunk grad → G GPU 96 GB 안 |
| DMD grad | (x0_fake − x0_real) / mean|x − x0_real|, σ ~ shift-5 U[0.02, 0.98] | DMD2 표준 |
| critic | ratio 2, lr 2e-5, fresh rollout | pilot 동일 |
| regression anchor | 없음 | DMD2가 다양성 저하로 버린 항이고, 우리 §22/§46/§48이 모두 blur로 죽은 항 |
| gen lr | R1 2e-6, R2 5e-6 (둘 다 r64, grad clip 1) | §34의 1e-5는 상한선으로 보고 아래에서 시작 |
| data | 47 train tasks × action streams (fm_data_train_scenes + tasks_camera.json camera_WM) | bench-43 scene-disjoint |
| eval | 200 step마다: 12-task a832 bench의 704 버전 → solo rt2(+LoRA) → judge vs rt2 / teacher | 기존 pipeline, 다른 arm과 직접 비교 가능 |

## 3. 컴퓨트

| item | estimate (704×384, 81 frames = 7 chunks) | basis |
|---|---|---|
| G rollout (no grad) | 7 × 2 × 0.38 s ≈ 5 s | rt2 0.765 s/chunk at 704 |
| real score (bid, CFG 2 fwd) | ≈ 10 s | §33: 19–20 s at 832×161f, 토큰 수 비례 |
| fake score fwd (7 chunks) | ≈ 3 s | |
| replay w/ grad (3 chunks) | ≈ 10 s | lora14_train 실측 ~5 s/step(1 chunk fwd+bwd) |
| critic ×2 (rollout + 7-chunk fwd/bwd) | ≈ 40 s | |
| **per generator step** | **≈ 70 s** | pilot(1.3B, 832) 61 s/step과 같은 자리수 |
| 300-step pilot | ≈ 6 h, 2 GPUs | go/no-go |
| 2000-step run | ≈ 40 h, 2 GPUs | |
| eval per checkpoint | 12 clips solo + judge ≈ 15 min, 1 GPU | |

GPU 배치: R1(GPU 0+1), R2(GPU 2+3), GPU 4 = eval chain, GPU 5–6 = exp 1/2 후속 또는 R3(아래). 6 GPU 1주 안에 pilot + 본 run + 평가가 들어간다.

## 4. Go / no-go (pilot 300 steps, 12 held-out tasks, judge paired vs rt2 base)

| step | pass | fail → action |
|---|---|---|
| 100 | quality ≥ rt2 − 0.15 (collapse 없음), DMD loss 추세 하락 | R1/R2 모두 collapse → lr 1e-6 + r32로 R1', 동시에 R3 준비 |
| 200 | quality ≥ rt2 ± 0.10, drift ≤ rt2 + 5 | 횡보면 300까지 계속 |
| 300 | quality > rt2 (≥ +0.15, 승>패) | 미달이면 DMD 라인 닫고 R3로 전환 |
| 2000 (본 run) | 목표 teacher −0.2 이내 at 15.7 fps (rt2 속도 그대로), adapt 재학습 후 teacher parity | |

R3 (DMD 실패 시 대안): feature-space adversarial 증류 — frozen 14B teacher의 layer-20 hidden(drift head와 같은 feature)을 입력으로 하는 작은 discriminator, real = teacher 4-step rollout 조각, fake = G의 free-running 조각; hinge loss, R1 생성기 설정 유지. 장점: bid 모델·real score 불필요(GPU 1개/run), 우리가 이미 그 feature로 verifier를 학습해 봤음(AUROC 0.88–0.93).

## 5. 구현 (specwm/dmd14.py, dmd_any.py에서 파생)
- 재사용: V1Binding bid real score(real_x0, bid_cond), rollout/commit 루프, critic 루프, 로깅.
- 제거: (t,r) 임베딩, flow-map co-train(fm_*), shift_sig 기반 any-N 샘플러 → V2Binding의 node 스케줄(timesteps_index=(0,500))과 target_x0 사용.
- 추가: random-stop k, lora_util 포맷 저장(두 어댑터: gen은 lora.pt, fake는 fake_last.pt), 200-step마다 `two_machine --role solo --student` 평가 체인 호출(별도 GPU), 12-task 704 bench 생성(a704_12 = a832 12 tasks의 704 버전; rt2/teacher 기준값은 bench-43 run에서 이미 있음 — 같은 12 task는 bench-43의 부분집합이므로 재생성 불필요).

## 6. 리스크
1. bid real score의 조건(plucker, i2v y)과 causal student의 조건이 다르게 해석되면 DMD gradient가 체계적으로 틀어진다 → pilot step 0에서 x0_real을 디코드해 눈으로 확인하는 sanity 체크를 넣는다(§34에서는 안 했음).
2. 2-step 고정 스케줄에서 σ 0.833 → 0 점프를 DMD만으로 배우는 것은 CausVid(4-step)보다 어렵다 → 안 되면 nodes (0,500,750) 3-step 학생으로 후퇴 = adapt50 자리(12.6 fps).
3. 1주 안에 판정이 안 날 수 있다 → pilot gate를 step 300(6 h)에 두어 하루 안에 1차 판정.
