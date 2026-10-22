# Action items — overnight report (9/28, 04:00–09:00 CDT)

상세 근거와 원문: `paper/overnight/` (15-agent workflow: 문헌 조사, 코드 감사, 설계 3안 + 적대적 검증, 완결성 비평), 실험 로그 `lingbot-world-v2/specwm_out/spec_v0/NIGHT3_NOTES.md`. 표는 영어, 설명은 한국어. 시간은 모두 CDT.

## 한눈에

| # | Action item | Status | One-line result |
|---|---|---|---|
| 1 | Test first on the two failure prompts | **Done** | Every method was judged on jar / castle first; tonight extended to 8 seeds × 2 prompts |
| 2 | Reproduce the speculative diffusion baseline | **Partly done** | De Bortoli et al. (ICML 2025) exact coupling reproduced in our setting and checked; no official code, email draft ready for you to send |
| 3 | Unbiased frame-level (AR) speculative sampling | **Designed, implemented, tested** | Exactness is achievable, but with Fast as the draft no speed-up is possible, and the chunk-level Base target is itself not Base-like at any CFG |
| 4 | Evaluation protocol (many seeds, binary drift) | **Done (exploratory); confirmatory run started** | DINO-based drift shown invalid, replaced by blind 3-rater labels; strict rubric cannot rank methods (Base itself passes 3/16); counting only failures Base avoids, clean clips: Base 12/16, Fast 0/16, exact spec 0/5, v2-safe 5/16, v2-fast-CFG 6/16 |
| 5 | Verifier & correction design space | **Mapped** | Detection is not the bottleneck, correction is; ranked next experiments |
| 6 | One-step Base check cost | **Profiled** | Cost is linear in tokens; CFG doubles it; plan resolution is 2.2–2.6× cheaper; batched checks fail, so parallel checks need separate GPUs |

---

## 1. 두 실패 프롬프트에서 먼저 (속도 최적화 없이)

- jar(`sand_filling_up_jar`)와 castle(`sandcastle_wave_impact`)에서 모든 방법을 먼저 검증했고, 속도 작업(CFG 6 + flow reuse, 1-GPU 렌더, 4-step 플래너)은 일관성이 확보된 뒤에 했습니다.
- 오늘 밤 이 두 프롬프트를 seed 8개(1234, 4321, 9999, 11, 22, 33, 44, 55)로 넓혀 item 4의 배치 비교에 썼습니다.
- 출력 길이 차이: Fast 기반 방법들은 153 프레임(13 chunks), Base-70은 161 프레임, 해상도는 모두 464×848.

## 2. Speculative diffusion baseline 재현

**논문.** De Bortoli, Galashov, Gretton, Doucet, *Accelerated Diffusion Models via Speculative Sampling*, ICML 2025, https://arxiv.org/abs/2501.05370. 공식 코드는 없습니다(arXiv v2 HTML, PMLR PDF 확인). 교신저자 주소 vdebortoli@google.com (논문에 인쇄된 유일한 주소).

**이메일.** 초안 `paper/overnight/item2_email.md` — 코드/하이퍼파라미터(CIFAR-10 Table 1의 window L, 체크포인트, FID 설정, NFE 계산 방식) 요청. 지도교수 성함이 파일 어디에도 확정돼 있지 않아 괄호로 비워 두었습니다. 발송은 직접 해 주세요.

**우리 설정에서 재현한 것.** 논문의 핵심인 reflection maximal coupling(Alg. 4)을 Fast의 node 0→1→2→3 전이마다, latent 위치(16채널)별로 적용했습니다(`spec_v0 --specstep`).

| check | result |
|---|---|
| Exactness on CPU (16-dim, 200k positions) | acceptance 1.000 / 0.865 / 0.575 / 0.093 vs theory (1 − TV) 1.000 / 0.867 / 0.576 / 0.094; output mean error ≤ 0.006, covariance error ≤ 0.010 |
| Runtime per-position acceptance (jar / castle, s1234, CFG-5 target) | node 0: 0.72 / 0.82, node 1: 0.86 / 0.91, node 2: 0.78 / 0.85 (range 0.58–0.94) |
| Whole-chunk acceptance (upper bound from logged ‖Δ‖) | ≤ 10^-25 (castle, node 1) to ≤ 10^-222 (jar, node 0) |
| Cost per clip (2 GPUs) | 341–349 s at CFG 5; 218–232 s at CFG 1 |

- 오늘 밤 수정한 버그: 처음 구현은 거부 위치를 최대 4회 rejection으로 재샘플하고 실패 시 target 샘플로 대체해서 편향이 있었습니다. reflection coupling으로 바꿔 한 번에 정확해졌습니다.
- 아직 안 한 것: 논문 설정(CIFAR-10, 100-step EM, frozen-target draft)에서 FID ≈ 2.35 @ ≈ 39 NFE를 재현해 우리 coupling 코드를 외부 기준으로 검증하는 단계(`paper/overnight/item2_repro_plan.md`, Stage A). GPU 몇 시간 규모로 추정.

## 3. Unbiased frame-level (AR) speculative sampling

**Target 정의가 핵심입니다.** Base는 161프레임을 한꺼번에 만드는 양방향 모델이라 chunk 단위 AR 조건부 분포가 없습니다. 그래서 target을 이렇게 이름 붙여 정의했습니다.

- **T_H^c9**: chunk 0과 각 chunk의 마지막 node는 Fast, node 0/1/2의 확률적 전이는 Base 커널 = Base A14B 한 번의 CFG forward로 얻은 x0를 다음 σ로 re-noise (앞 chunk 9 latent를 같은 σ로 re-noise해 조건으로 주는 RePaint 방식).
- `--specstep`은 이 T_H^c9에 대해 정확합니다(증명 스케치와 적대적 검토 통과: `paper/overnight/item3_design.md`). Base-70의 분포에 대해 정확한 것은 아닙니다.

**Draft가 분포를 바꾸지 않는다는 것의 의미.** 정확한 coupling이므로 Fast는 결과의 분포에 영향을 못 주고, 어떤 실현값이 나오는지만 정합니다(58–94% 위치를 재사용). 따라서 품질은 전적으로 target 커널이 결정합니다.

**속도: Fast draft로는 불가능.** 여러 스텝을 미리 draft하고 한 번에 검증해 이득을 보려면 chunk 전체(295k 차원)가 한 번에 수락돼야 하는데, 측정된 ‖Δ‖로 계산한 수락 확률이 ≤ 10^-25입니다. 예상 가속 S = E[L̂]/(1 + γ·C_D/C_T) ≈ 0.9(순차) ~ 1.0(Fast·Base 동시 실행). 문헌도 같은 방향입니다(block verification으로도 ImageNet LDM에서 1.22–1.55×, https://arxiv.org/abs/2606.13426).

**Target 커널 품질 (오늘 밤 새 결과, s1234).** 정확한 샘플러와 target-only null arm이 같은 방식으로 망가져서 문제의 원인이 coupling이 아니라 target임을 확인했습니다.

| target CFG | jar: luminance at frames 0 / 60 / 100 / 152 | jar camera pans like Base? | castle: luminance at 0 / 60 / 100 / 152 | castle look | time per clip ↓ |
|---|---|---|---|---|---|
| Base-70 (reference) | 119 / 107 / 127 / 128 | yes (table → white wall) | 124 / 194 / 200 / 186 | castle destroyed, beach and sky | 2,777–2,829 s |
| 5 | 125 / 186 / 189 / 157 (saturation → 198, green) | no | 130 / 246 / 244 / 224 | washes out to white/green by chunk 4 | 341–349 s |
| 2.5 | 125 / 153 / 188 / 199 | no (jar stays centred) | 130 / 210 / 222 / 222 | over-exposed, hazy | <u>342–345 s</u> |
| 1 | **125 / 126 / 139 / 137** | no (jar stays centred) | 130 / 179 / 137 / 177 | colour OK, hazy and grainy | **218–232 s** |

- CFG 5 target-only null arm(jar s1234)도 같은 흰색/초록 wash-out: luminance 125 / 168 / 211 / 212.
- 해석: 같은 σ로 prefix를 re-noise하는 한 스텝 Base 커널은 CFG가 크면 색이 누적 폭주하고, 작으면 카메라 명령을 무시하고 prefix에 붙어 있습니다. 이전에 chunk 단위 Base 교정이 "frame 0으로 snap back"하던 현상과 같은 원인입니다.

**결론과 다음 단계.**
1. 정확성 자체는 달성했지만, "무엇에 대해 정확한가"가 좋지 않습니다. 쓸모 있는 target이 먼저입니다.
2. 후보: (a) node 사이에서 Base를 여러 스텝 돌리는 target(De Bortoli식 frozen-draft를 sub-step에 적용), (b) Base의 whole-clip 저해상도 플랜에 조건을 거는 chunk target(plan→render와 결합), (c) block-causal Base(KV cache) 커널.
3. 속도 목표는 speculative 검증이 아니라 plan→render(114–206 s)로 달성하는 것이 현실적입니다.

## 4. 평가 프로토콜: 다수 seed, 이진 drift 성공률

**한 줄 요약.** 프로토콜(같은 seed 목록, blind 이진 판정, 정확 검정)은 완성해서 돌렸습니다. 사전에 정한 엄격한 rubric에서는 Base 자신이 3/16만 통과해 방법 간 순위를 못 매깁니다. Base가 거의 하지 않는 실패(teleport, 색/texture 붕괴, freeze)만 세는 탐색적 기준에서는 차이가 뚜렷합니다: Base 12/16 clean, Fast 0/16, exact spec 0/5, v2-safe 5/16, v2-fast-CFG 6/16.

**설정.**
- 프롬프트 2개(jar, castle) × seed 8개(1234, 4321, 9999, 11, 22, 33, 44, 55), 모든 arm이 같은 seed 목록을 씁니다.
- Arms: Base-70(target), Fast 단독, exact speculative sampler(CFG-5 target), target-only null arm(같은 target을 draft 없이), v2-safe, v2-fast-CFG, v2-fast. CFG 1과 2.5 target은 seed 1234에서 각 1–2개만 돌렸습니다.
- 이진 함수: 익명화한 contact sheet(START 프레임 + 이후 15프레임, 파일 이름은 해시)를 3명의 독립 평가자(모델 agent)가 방법을 모른 채 rubric으로 판정하고 다수결을 냅니다. drift 유형은 새 물체, teleport, 색 붕괴, 재질 변화, freeze, texture 붕괴, 조명 급변의 7가지입니다.
- 평가자 일치도: 85개 clip에서 Fleiss κ 0.69, 만장일치 89%.
- 검정: arm별 Wilson 95% CI, Base와의 Boschloo exact test(두 프롬프트 합산).

**DINO 기반 drift 함수를 버린 이유.**
1. 같은 프롬프트의 Base끼리도 seed가 다르면 DINO 거리가 0.29–0.75라서, 같은 seed의 Base와의 거리로 drift를 정의할 수 없습니다.
2. Base 8개의 95th percentile을 threshold로 쓰는 규칙은 Base 통과율이 구조적으로 7/8로 고정되고(측정이 아님), 같은 Base가 threshold와 기준을 겸해서 순환적입니다.
3. 궤적 단위 nearest-neighbour 규칙으로 고쳐도, 흰색으로 wash-out된 exact spec 영상을 가장 Base-like로 평가했습니다. DINO 특징은 색 붕괴를 못 봅니다.
4. 같은 16개 case에서 그 DINO 규칙은 v2-safe 14/16, v2-fast-CFG 16/16, Fast 4/16을 no-drift로 판정했습니다. blind 판정의 severe-clean은 각각 5/16, 6/16, 0/16입니다. DINO 규칙은 plan의 teleport를 거의 못 잡습니다.

**결과 (두 프롬프트 합산; 괄호 안은 프롬프트별 strict no-drift 수).**

| arm | clips (jar + castle) | no drift, strict rubric ↑ [95% CI] | p vs Base | no severe failure ↑ [95% CI] | p vs Base | most common failure types (majority votes) |
|---|---|---|---|---|---|---|
| Base-70 (target, bidirectional) | 16 (jar 2/8, castle 1/8) | <u>3/16 = 0.19</u> [0.07, 0.43] | – | <u>12/16 = 0.75</u> [0.51, 0.90] | – | new object 5, lighting change 5, material swap 4 |
| Fast alone (AR) | 16 (jar 0/8, castle 0/8) | 0/16 = 0.00 [0.00, 0.19] | 0.106 | 0/16 = 0.00 [0.00, 0.19] | < 0.001 | teleport 16, lighting change 12, new object 10 |
| v2-safe (cr plan + render) | 16 (jar 3/8, castle 0/8) | <u>3/16 = 0.19</u> [0.07, 0.43] | 1.000 | 5/16 = 0.31 [0.14, 0.56] | 0.016 | teleport 8, lighting change 8, new object 3 |
| v2-fast-CFG (l4c plan + render) | 16 (jar 3/8, castle 0/8) | <u>3/16 = 0.19</u> [0.07, 0.43] | 1.000 | 6/16 = 0.38 [0.18, 0.61] | 0.037 | lighting change 8, teleport 7, colour collapse 3 |
| v2-fast (l4 plan + render) | 6 (jar 1/3, castle 1/3) | **2/6 = 0.33** [0.10, 0.70] | 0.554 | **5/6 = 0.83** [0.44, 0.97] | 0.777 | new object 3, teleport 1 |
| Exact spec sampler, CFG-5 target | 5 (jar 0/3, castle 0/2) | 0/5 = 0.00 [0.00, 0.43] | 0.469 | 0/5 = 0.00 [0.00, 0.43] | 0.004 | texture collapse 5, colour collapse 5, material swap 3 |
| Target-only null arm, CFG 5 | 4 (jar 0/2, castle 0/2) | 0/4 = 0.00 [0.00, 0.49] | 0.562 | 0/4 = 0.00 [0.00, 0.49] | 0.009 | texture collapse 4, colour collapse 4, material swap 1 |

- CFG 1 / 2.5 target(seed 1234; exact spec 4개, null arm 2개)은 모두 drift: CFG 1은 jar에서 freeze, castle에서 흐릿한 texture, CFG 2.5는 과노출.
- v2-fast(l4 plan)는 seed 1234/4321/9999만 있어 n = 6이고 구간이 넓습니다. 사전 등록 배치에는 v2-fast-CFG만 넣었습니다.
- exact spec과 target-only null arm이 같은 실패 양상(색 + texture 붕괴)을 보입니다. 정확한 sampler라면 둘의 분포가 같아야 하므로 기대와 일치합니다. n이 작아 약한 확인입니다.

**해석.**
1. **사전에 정한 strict rubric은 판별력이 없습니다.** Base의 "drift"는 jar에서 새 물체(파란 화분)와 재질 변화, castle에서 흐림→맑음 조명 변화입니다. rubric이 Base의 정상 행동을 drift로 세므로, 이 기준에서는 모든 arm이 Base와 유의한 차이가 없습니다(최소 p = 0.11).
2. **Base가 하지 않는 실패만 세면 순위가 선명합니다.** Fast는 16개 전부 teleport(책상 → 야외, 해변 → 다른 풍경). exact spec과 null arm은 전부 색/texture 붕괴. v2 계열은 Fast보다 유의하게 낫습니다(Boschloo vs Fast: v2-safe p = 0.017, v2-fast-CFG p = 0.006). 하지만 v2-safe(5/16, p = 0.016)와 v2-fast-CFG(6/16, p = 0.037)는 여전히 Base(12/16)보다 나쁩니다. 프롬프트별 clean 수는 jar에서 Base 7/8, v2-safe 4/8, v2-fast-CFG 4/8이고, castle에서 Base 5/8, v2-safe 1/8, v2-fast-CFG 2/8입니다. 격차의 대부분은 castle이고, plan이 바다 대신 풀이나 숲이 있는 평지나 열대 해변을 만드는 teleport가 주 원인입니다. 맑은 하늘로 바뀌는 것은 Base도 castle 8개 중 5개에서 보이므로 이 기준에서는 세지 않습니다.
3. **주의: 이 두 번째 기준은 라벨을 본 뒤 고른 사후 기준입니다.** 결론으로 쓰려면 새 seed 세트에서 미리 등록하고 다시 재야 합니다.
4. **검정력.** Base 0.75 대 방법 0.40을 α = 0.05, power 0.8로 가르려면 arm당 약 30 clip이 필요합니다. 지금은 arm당 4–16개입니다.

**다음 단계.**
1. **시작함 (8:40 AM CDT):** severe-failure 기준을 `paper/overnight/prereg/PREREG_item4.md`에 미리 등록했습니다(평가 스크립트 해시 고정, Holm 보정 5개 검정). 새 seed 101–116 × 2 프롬프트 × 4 arm(Base-70, Fast, v2-safe, v2-fast-CFG) = 128 clip을 GPU 4–7에서 생성 중입니다. Fast와 v2는 약 2시간, Base-70 32개는 약 6–7시간 걸립니다. GPU 0–3도 쓰면 절반 정도로 줄어듭니다. 멈추려면 `exp/queue/qprereg.txt`를 비우면 됩니다.
2. 평가자 타당도: 이미 판정된 clip 40개를 직접 보시고 라벨과 비교(사람–평가자 κ).
3. 같은 난수를 공유하도록 seed 처리(ξ/u)를 고치면 McNemar 짝 검정으로 필요한 표본 수를 줄일 수 있습니다.

## 5. Verifier & correction design space

측정 결과 요약(자세한 표: `paper/overnight/item5_design_space.md`).

| verifier | professor's category | cost per check ↓ | measured result | verdict |
|---|---|---|---|---|
| DINO + colour gate (`sem`) | – | **≈ 0.5 s** | AUC **0.900** on 816 labelled chunks | best cheap detector; misses new/duplicate objects and material swaps |
| DINO nearest neighbour to Base frames | – | ≈ 0.5 s | AUC <u>0.888</u> | rates washed-out videos as Base-like (tonight) |
| One-step Base x0 distance (480p / 360p / 240p, ± CFG, KV) | target-likelihood proxy, low-res | 0.85–9.8 s | AUC 0.19–0.60, direction unstable | not a drift detector |
| Base diffusion loss at σ 0.24 | likelihood | ≈ 6 s | 0.75 Fast vs Base | degrades with lower resolution |
| VGGT camera-rotation jump | – (geometry) | ≈ 4 s | 0.81 Fast vs Base, 1.00 teleport | good for teleports, blind to freezing |
| Logistic Base-vs-Fast probe on DINO | learned | ≈ 0.5 s | AUC 0.832 | cheap learned baseline |
| LikePhys (Base loss as physics prior) | physics | – | PPE 0.51 (chance 0.50) | texture prior, not physics |
| VLM judge (Qwen3-VL-32B) | physics / semantics | seconds | prefers Fast 41:18 pairwise | unreliable pairwise |
| Exact per-position speculative test (tonight) | likelihood (exact) | 3 CFG calls / chunk | acceptance 0.58–0.94 per position, ≤ 10^-25 per chunk | exact but never saves a Base call |

- 교정 쪽: 16-step Base RePaint는 seam과 snap-back, Base가 Fast 상태에서 이어받으면 Fast의 drift가 통과, Base-guided node는 정지, x0-MSE LoRA는 흐림/정지, KV translator는 hint 채널과 동급, **plan→render만 작동**.
- 다음 3개(순위): (1) 이미 라벨된 데이터에서 decoded-frame 신호 bake-off(object persistence, 카메라 궤적 일치, render-vs-plan, VLM yes/no 루브릭), (2) speculative planning: 4-step 플랜을 draft로, 플랜 검증기로 통과/실패 → 실패 시 16-step 플랜, (3) Fast node-1 특징에 대한 학습형 probe → Fast 재샘플.

## 6. One-step Base 검증 비용과 병렬 체크

Base 한 번의 forward(batch 1, 한 GPU, 두 expert 상주). 비용은 토큰 수에 선형이고 expert 간 차이가 없으며 CFG는 두 배입니다.

| resolution | window (latent frames) | tokens | no CFG ↓ | CFG ↓ | peak GB ↓ |
|---|---|---|---|---|---|
| 480p (464×848) | 6 | 9,222 | 1.66 s | 3.32 s | 74 |
| 480p | 12 | 18,444 | 3.73 s | 7.47 s | 78 |
| 480p | 21 | 32,277 | 7.50 s | 14.99 s | 85 |
| 576×304 (plan) | 6 | 4,104 | **0.70 s** | **1.40 s** | **72** |
| 576×304 | 12 | 8,208 | <u>1.46 s</u> | <u>2.91 s</u> | <u>74</u> |
| 576×304 | 21 | 14,364 | 2.83 s | 5.65 s | 76 |

- 여러 노이즈 레벨을 한 번의 batched forward로 검사하는 것은 현재 Base 코드의 카메라(Plücker) 조건 경로가 batch 1만 지원해서 실패합니다. 병렬 체크는 GPU를 나눠서만 가능합니다(예: 4 GPU → σ 4개 동시).
- 조기 drift 검출 관점에서 문제는 비용보다 신호입니다: 한 스텝 Base x0 거리는 drift를 가르지 못했습니다(AUC 0.19–0.60). 같은 비용이면 decoded frame 위의 DINO+colour(0.5 s, AUC 0.90)가 낫습니다.
- 저해상도(576×304)에서의 검사는 2.2–2.6× 싸고, 플랜 단계에서 전체 클립(41 latent, 약 28k 토큰)을 한 번에 검사하면 CFG 없이 약 6 s로 추정됩니다(측정 아님, 선형 외삽).

## 부록: KV translator kv4 (어제 "그렇게 가자"로 시작한 학습 개선)

**무엇을 바꿨나.** kv1(5개 block, Base plan K/V 전체를 memory로)에서 세 가지를 키웠습니다.
- memory block 5 → 20개(짝수 block). 40개는 GPU 메모리(88–90 GB)에 안 들어갔습니다.
- 시간 국소 memory: 현재 chunk와 다음 plan chunk에 해당하는 6 latent frame(약 4.1k 토큰)만 attend.
- Translator2: head별 128×128 linear(K는 identity, V는 0으로 초기화) + rank-256 head 간 mixing, Base GPU에 올려 학습.
- 데이터: mixdata 124 clips, 2 epochs, step 1612에서 종료. 더 큰 데이터(mixdata2, 224 clips)로 돌린 kv5는 GPU를 action item에 쓰려고 3:47 AM에 step 약 690에서 멈췄습니다.

**결과 (jar/castle × seed 1234/4321/9999, cr plan, `--kv-low plain`).**

| render | DINO distance to same-seed Base, mean of 6 ↓ | jar s1234 / s4321 / s9999 ↓ | castle s1234 / s4321 / s9999 ↓ | seam: max boundary/inside ratio, mean ↓ (Base 1.88) |
|---|---|---|---|---|
| v2-safe (y-hint LoRA, reference) | <u>0.541</u> | <u>0.39</u> / **0.78** / **0.34** | 0.48 / **0.56** / **0.69** | <u>2.40</u> |
| kv1 ck875 (5 blocks, full memory) | 0.584 | 0.44 / 0.80 / 0.60 | <u>0.39</u> / <u>0.57</u> / 0.71 | **1.88** |
| kv4 ck800 | **0.536** | **0.33** / **0.78** / 0.50 | **0.36** / **0.56** / <u>0.70</u> | 2.73 |
| kv4 ck1612 | 0.565 | <u>0.39</u> / <u>0.79</u> / <u>0.45</u> | 0.50 / 0.58 / **0.69** | 2.62 |

- kv5 ck575(224 clips)은 jar s1234 / s4321만 평가됨: 0.39 / 0.79. jar s9999는 평가 중 다른 작업과 GPU가 겹쳐 OOM.
- 영상으로 확인: castle s1234에서 v2-safe에 나타나던 걷는 사람이 kv4에서는 사라졌습니다. 반대로 jar s9999에서는 kv4가 흰 벽 대신 호수와 먼 해안을 만들어 Base에서 더 멀어졌습니다.
- 결론: kv4는 kv1보다 낫지만 y-hint 경로(v2-safe)와 같은 수준입니다. step 1612가 800보다 나쁜 것은 plan6 LoRA 때와 같은 과적합 패턴입니다. castle에서 남는 소나무 숲 해안과 맑은 하늘은 plan 자체에 들어 있고 render는 plan을 충실히 따르므로, 더 좋은 plan 없이는 KV 경로로 개선할 여지가 작습니다.

---

## 오늘 밤 운영 메모

- Base-70 시간 정정: 공식 파이프라인은 464×848에서 한 GPU로 **2,737–2,845 s/clip**(8+ runs)입니다. 제가 앞서 쓴 "~600 s"는 360×512 + WorldCache(k=4, 손실 캐시) 조건의 ~620 s였습니다. v2-safe 206 s는 약 13.6× 빠릅니다.
- GPU 충돌 3건(오래된 평가 worker와 겹침, 늦은 chunk에서 Base 창 OOM, 번들 정의를 늦게 읽는 worker)을 잡았고, 다른 사용자(hl35496)의 Blender가 GPU 3을 4 GB 점유한 상태에서도 작업을 계속했습니다.
- 지금 GPU: 4–7에서 item 4 사전 등록 배치(192 jobs) 실행 중, 0–3은 비어 있음(8:40 AM CDT 기준). 다른 사용자 프로세스 없음.
- kv4 학습은 step 1612까지 완료, 평가는 위 부록 참고. kv5는 GPU 확보를 위해 3:47 AM에 중단(step 약 690).
