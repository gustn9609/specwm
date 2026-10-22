# Draft를 "super fast"로 만들어 cloud/teacher와 협업하기 — 조사 메모 (10/13)

배경: 10/12 미팅에서 Guandao는 "비디오는 보여준 프레임을 되돌릴 수 없으니, draft(또는 cloud)가 실시간보다 훨씬 빨라서 teacher가 아직 안 보여준 미래를 검증·재생성할 시간이 있어야 한다. 픽셀까지 다 만들 필요는 없고 중간 신호(latent, feature)면 된다"고 했다. 이 메모는 (1) 우리가 이미 가진 숫자로 시간 예산을 계산하고, (2) draft 가속 기법(pruning, quantization, caching, sparse attention, KV 압축, step distillation)을 조사해 우리 상황에서 기대 이득을 추정하고, (3) 협업 설계안 3개를 비교한다. 숫자 출처: `facts.md` §4, §12–13, §15, §18–19, `data/int4_bench*.json`.

## 1. 시간 예산 — 지금 어디가 느린가

청크 = 3 latent frames = 12 frames = 0.75 s. 모두 RTX PRO 6000 한 장, 최적화 스택(FP8, Sage, compile, ring KV) 기준.

| stage | 832×480 | 704×384 | note |
|---|---|---|---|
| draft 1.3B, 4-step, latent만 | 0.35 s/chunk (2.1× real time) | ≈0.25 s (est.) | SDVG profile, facts §4 |
| draft 1.3B, 4-step + full VAE decode | 0.35 + 0.67 | – | student clip: gen 8.2 s + decode 8.7 s / 13 chunks |
| per-frame streamed decode (device) | 0.63 s/chunk | 0.43 s/chunk | fit24; tiled 0.90 / 0.49 |
| teacher 14B one-step x0 (verify) | 0.69 s (SDVG verify, bf16 path) / ≈0.55 (FP8) | ≈0.35–0.40 | 2-step chunk = 0.80 s at 704 |
| teacher 14B commit to its KV | 0.54 s | ≈0.3 | fused commit로 0으로 만들 수 있음 (facts §10) |
| teacher 2-step chunk (realtime mode) | 1.20 s | 0.80 s (w12: 0.69 s) | fits 32 GB |
| latent 전송 | 0.4 MB fp16/chunk | 0.27 MB | 50 Mbps에서 ≈ 65 / 45 ms |

읽기:
- **Draft의 latent는 이미 "super fast"다** (실시간의 2배). 느린 것은 (a) 픽셀 디코딩(생성 시간과 같은 크기)과 (b) teacher의 검증·재생성이다. 그래서 "중간 신호까지만"은 정확히 맞는 방향이고, **검증 경로에서 디코딩을 빼는 것**이 첫 번째 설계 원칙이다.
- 그러나 10/6–10/12의 결과가 보여준 진짜 병목은 속도가 아니라 **draft의 수락률**이다: ImageReward τ −0.7에서 13청크 중 1.65 수락, physics verifier에서도 teacher보다 느려짐(facts §16), draft 토큰이 teacher 궤적에서 0.63 σ 떨어짐(§4). 거부된 청크마다 teacher가 다시 만들어야 하므로(0.8–1.2 s) draft가 아무리 빨라도 **수락률 × draft 속도**가 실효 속도를 정한다. 아래 가속 조사는 이 전제 위에서 읽어야 한다.

## 2. Draft 가속 기법 조사 — 우리 형상(1.3B, 3k–5k tokens/forward, 4-step causal)에서의 기대치

| technique | representative work (2025–26) | reported | expected on our draft | cost / risk |
|---|---|---|---|---|
| **Fewer steps (1–2)** | Self Forcing (2506.08009), Causal Forcing (ICML 2026, 2602.02214): AR teacher로 ODE init 후 DMD; Causal Forcing++ (2605.15141): frame-wise 1–2 step | 4-step 대비 VBench 동등, first-frame latency −50 % | **2–4× (DiT 부분)**. 우리 14B 2-step은 이미 −0.30 품질로 쓰고 있음; 1.3B 2-step은 미측정 | 재증류 필요(우리는 AR teacher 14B를 갖고 있어 Causal Forcing 레시피에 정확히 맞음). 이전 교훈: low-noise node MSE 증류는 blur (memory) |
| **Lower resolution** | – (우리 §9 곡선) | 14B: 832 7.53 → 576 6.83 quality | **576: ≈1.8×** (tokens 2.2× 감소) | draft 품질 하락; 하지만 draft는 어차피 teacher가 고침 |
| **KV window 12** | (우리 §19) | 14B: +15 % fps, 품질 변화 없음 | **≈1.15×** | 없음 |
| FP8 / Sage / compile | (우리 §5) | 14B 스택 2.6× vs 원본 | 이미 적용됨 | – |
| **Feature caching** | WorldCache content-aware (2603.22286, ECCV'26): Cosmos 2B 2.3×; WorldCache heterogeneous-token (2603.06331): 3.7×; TeaCache / MagCache / TaylorSeer / EasyCache | 50–70 step 모델에서 2–5× | **≤1.3×**: 4-step 모델은 건너뛸 step이 ≤1개. 우리 측정: Base 70-step에서 1.21× (memory) | few-step에는 사실상 무용 |
| **Sparse attention** | Sparse Forcing (2604.21221): Self Forcing +1.11–1.27×; Light Forcing (2602.04789): 1.3× e2e; DFSAttn (ICML'26) 1.19× on 4-step; SVG2, Radial, VSA | 1.1–1.3× on AR few-step | **≤1.1×**: 3k tokens에서 attention은 병목 아님 (memory: 10/8) | – |
| **Depth pruning** | TinyFusion (CVPR'25, 2412.01199): DiT-XL 50 % depth → 2×, FID 2.86, 사전학습 비용 7 %; OBS pruning (2607.20048) | 2× at 50 % depth with fine-tune | **1.5–2× 가능하나 재학습 필요** | 이미 약한 draft가 더 약해짐; 수락률이 핵심이므로 역효과 위험 |
| **W4A4 / NVFP4** | Nunchaku SVDQuant (W4A4): e2e 1.35× → 1.8× with compile (FLUX); NVFP4 on Blackwell 2.24× (Z-Image); Wan2.2 W4A4 PTQ 연구 3편 (2605.27003, 2605.26628, 2606.29337) | 1.3–2.2× GEMM | **1.3–1.6×** over bf16 (Blackwell NVFP4), Ada(4090) INT4 1.3×. 우리 microbench: FP8 W8A8는 bf16 대비 1.1–1.5×, weight-only INT4는 오히려 느림 (§19) | 활성화 outlier → 품질; draft엔 허용 가능 |
| **KV cache 압축** | Forcing-KV (2605.09681): head 역할별 pruning, −30 % memory, 1.35–1.5× (Self Forcing/LongLive); VideoMLA (2605.30351) low-rank latent KV; Future Forcing (2605.30083) training-free | 1.35–1.5× | 메모리에 유효, 속도는 ≈1.1× (우리는 window 12가 같은 효과) | – |
| **VAE 교체/제거** | tiny VAE (TAEHV; 우리 `taehv_ablation.py`), Guandao: Apple 팀 VAE·KV-adapter 제거 논문 예정 | – | **디코딩 0.43–0.67 s/chunk → ≈0.1 s** | 검증 경로에서 디코딩을 빼면 draft 쪽은 영향 없음; 표시 경로에만 해당 |

합산 추정(곱셈 가정, 보수적): 1.3B **2-step × 576 × window 12 ≈ 0.35 → ≈0.08 s/chunk (≈9× real time)**, 여기에 NVFP4까지 ≈0.06 s. 즉 draft는 **청크당 0.6–0.7 s의 여유**를 teacher 검증·재생성에 넘길 수 있다. 반대로 caching·sparse attention·pruning은 우리 형상에서 10–30 %짜리이고 재학습/품질 위험이 있어 후순위다.

## 3. 협업 설계안 비교

| design | device | cloud (14B) | per-chunk critical path | 기대 품질 | 전제 |
|---|---|---|---|---|---|
| **D1. Latent look-ahead + one-step departure verify** | draft 1.3B(2-step, 576, w12) latent를 k=2 청크 앞서 생성, x0 latent 전송(0.27 MB); 수락된 latent만 디코딩 | draft x0에 대해 one-step x0 (0.35 s at 704) + fused commit; departure score(§B: PCA 거리 / drift head AUROC 0.88)로 accept, 아니면 teacher x0 반환 | draft 0.08 + 전송 0.05 + verify 0.35 + 전송 0.05 ≈ 0.55 s < 0.75 s → 파이프라인 가능 | draft–teacher 사이; 수락률 p에서 cloud 부하 = verify + (1−p)·regenerate | **수락률**. 지금 p ≈ 0.1–0.3 → cloud가 teacher 단독(0.8 s)보다 더 바쁨. D3 없이는 성립 안 함 |
| **D2. Cloud가 fast model (현재 realtime 모드)** | VAE만 (0.5 GB): latent 수신·디코드·업샘플 | 14B 2-step 704 w12: 0.69 s/chunk (17.5 fps), 32 GB면 device 단독도 가능 (§19) | RTT + 0.69 + decode 0.43 (파이프라인) | teacher −0.62 ± 0.09, StEvo phys 0.68 (측정 완료) | 네트워크 4 Mbps; cloud 비용·에너지 (Diana의 지적) |
| **D3. Teacher-anchored draft** (D1의 전제 조건) | draft가 cloud가 보낸 **앵커**(마지막 수락 청크의 14B KV 요약 또는 x0)에서 출발해 앞서 생성 | 앵커 전송 + 이탈 청크만 재생성 | D1과 같음 | 앵커로 draft 궤적이 teacher에 가까워져 p ↑ | 우리 10/8 "anchor split works", KV translator ≈ y-route (memory). 학습 필요: Causal Forcing 레시피로 14B on-policy 증류 |
| D4. Jump-ahead + interpolation (Guandao) | draft가 키프레임(2청크마다)만 예측, 경량 보간기가 중간을 채움 | 키프레임만 검증 | 검증 부하 ½ | 보간은 물리보다 쉬움 | EcoVideo (2606.30557)가 cloud 키프레임 + edge 보간을 DiT에서 구현 (step-split 대비 VBench +0.08) — 참고 구현 |
| D5. Action-change fast path | 액션이 바뀌면 device draft가 즉시 반응한 청크를 표시, teacher가 결과를 사후 검증 | 검증 후 다음 청크부터 교정 | 액션→프레임 latency = draft latency (≈0.1 s) | 교정 시 보이는 점프 위험 | LLM의 "interruptible spec decoding" 유사물; 보여준 프레임 되돌림 불가 → 교정은 미래 청크에만 |

읽기:
- **D2가 현재 측정된 것 중 유일하게 전 축에서 성립**한다(품질·속도·메모리). D1/D3는 "수락률이 높아지면" 성립하는데, 우리 데이터는 아직 그 반대를 가리킨다. 그러므로 **D1을 하려면 D3(앵커/증류로 draft를 teacher 궤적에 붙이기)가 선행**되어야 하고, 그 전에 D1의 비용 모델을 수락률의 함수로 먼저 측정하는 게 옳다.
- D1의 verifier는 디코딩 없이 14B one-step x0 + drift head/PCA 거리로 충분하다 — 이것이 Guandao의 "중간 신호" 요구와 item B(departure point)를 동시에 만족한다.

## 4. 제안 실험 순서 (각각 하루 이내)

1. **Draft 속도 사다리(latent만)**: 1.3B {4,2}-step × {832, 704, 576} × window {18, 12}, FP8 — s/chunk와 VLM 품질. "draft가 몇 청크 앞설 수 있는가"를 숫자로.
2. **D1 비용 모델**: 기존 spec_ar_loop에 `latent-only verify`(14B one-step x0 + departure score, 디코딩 없음) 모드 추가 → 수락률 p별 cloud 부하와 최종 품질 곡선. 수락 기준은 drift head logit / PCA 거리 (item B와 공유).
3. **D3 앵커**: cloud가 마지막 수락 청크의 14B x0(또는 KV 요약)를 보내고 draft가 거기서 출발 → p가 얼마나 오르는지. 이것이 오르지 않으면 Causal Forcing 레시피로 14B on-policy 증류(학습, 수 일).
4. (병렬) **device 표시 경로**: tiny VAE / tiled decode로 디코딩 0.43 → 0.1 s — D1·D2 모두에 이득.

## 5. Sources
- WorldCache (content-aware): https://arxiv.org/abs/2603.22286 , code https://github.com/umair1221/WorldCache ; WorldCache (heterogeneous token caching): https://arxiv.org/html/2603.06331v1
- Survey, video generation models as world models (efficient paradigms): https://arxiv.org/pdf/2603.28489 ; cache survey: https://arxiv.org/pdf/2510.19755
- MagCache: https://arxiv.org/html/2506.09045v2 ; TaylorSeer: https://openaccess.thecvf.com/content/ICCV2025/papers/Liu_From_Reusing_to_Forecasting_Accelerating_Diffusion_Models_with_TaylorSeers_ICCV_2025_paper.pdf ; EasyCache: https://arxiv.org/html/2507.02860v1
- Nunchaku / SVDQuant in Diffusers: https://huggingface.co/blog/nunchaku-diffusers ; W4A4 for Wan2.2: https://arxiv.org/pdf/2605.27003 , https://arxiv.org/pdf/2605.26628 , https://arxiv.org/pdf/2606.29337 ; vLLM-Omni NVFP4 PR: https://github.com/vllm-project/vllm-omni/pull/3830
- TinyFusion: https://arxiv.org/abs/2412.01199v1 ; OBS pruning for diffusion: https://arxiv.org/pdf/2607.20048
- Self Forcing: https://arxiv.org/abs/2506.08009 ; Causal Forcing: https://arxiv.org/abs/2602.02214v1 ; Causal Forcing++: https://www.emergentmind.com/papers/2605.15141
- Forcing-KV: https://arxiv.org/abs/2605.09681 ; VideoMLA: https://arxiv.org/pdf/2605.30351 ; Future Forcing: https://arxiv.org/pdf/2605.30083 ; Relax Forcing: https://arxiv.org/pdf/2603.21366
- Sparse Forcing: https://arxiv.org/html/2604.21221v1 ; Light Forcing: https://arxiv.org/html/2602.04789v1 ; DFSAttn: https://arxiv.org/html/2605.23445v1 ; Sparse VideoGen2: https://arxiv.org/pdf/2505.18875 ; DSA (ICLR'26): https://proceedings.iclr.cc/paper_files/paper/2026/file/c3728248f3c627d1f16ca5726cdf83f5-Paper-Conference.pdf
- SDVG (speculative decoding for AR video): https://arxiv.org/pdf/2604.17397 ; DFlash: https://arxiv.org/pdf/2602.06036 ; multimodal SD survey: https://arxiv.org/pdf/2608.20743
- EcoVideo (cloud keyframes + edge interpolation): https://arxiv.org/abs/2606.30557 ; SANA-Streaming (RTX 5090 real-time): https://arxiv.org/pdf/2605.30409 ; StreamDiT: https://cvpr.thecvf.com/virtual/2026/poster/40044
- Waypoint-1.5: https://aivideobootcamp.com/blog/overworld-waypoint-1-5-ai-world-model-guide-2026/ ; Decart Oasis 3: https://techcrunch.com/2026/06/10/decarts-new-world-model-can-simulate-hours-of-photorealistic-driving-with-some-caveats/ , https://decart.ai/oasis

## 6. Result of step (1), the draft speed ladder (10/13) — see facts.md §20, data/draft_ladder.json
Measured (1.3B, FP8 weights, bf16 KV, fused + partial commit, 43 bench-43 prompts, PRO 6000, generation only): 4-step 832/w18 0.437 s/chunk (1.7x real time, 7.4 GiB) → 4-step 704/w12 0.220 s (3.4x, 4.7 GiB, quality = student arm −0.02 ± 0.17) → 2-step 576/w12 0.118 s (6.4x, 3.9 GiB) but 2 steps cost −1.5 to −1.9 quality (4.0–4.4) and are unusable; 576 4-step −0.37. KV window 12 is free. Drift 74–81 for every config (teacher 50): the ladder buys look-ahead budget, not a better proposal — the §3 estimate "2× real time, acceptance is the bottleneck" is confirmed and the usable budget is 3.4× (704/w12). Streamed VAE decode (0.63 / 0.43 / 0.30 s per chunk at 832 / 704 / 576) exceeds the draft generation time, so a shown-frame path is decode-bound; the look-ahead exists only in latent space. Next: D3 anchor (teacher-conditioned draft) before any D1 cost model is worth running.

## 7. Result of the distillation test (10/13) — see facts.md §22
LoRA(r64/128) distilled on 224 clips (47 unseen-to-bench scenes × 4 seeds) against the 14B's own 4-step x0 targets with teacher-forced context, 3000 steps. On 12 held-out bench scenes the anchored acceptance goes 5.3 % → 8–11 % (14B own step 1: 32 %), and the draft's own clips blur (VLM quality 6.0 → 3.2–4.3; same x0-MSE failure as 9/26, also when only the layout nodes carry the loss). Together with §6 (speed) and facts §21 (anchoring) this closes the separate-drafter line: D1/D2 are not worth a cost model, D4/D5 (jump-ahead, action-change) would inherit the same 1.05 σ_T own-history gap. The paper keeps the 14B self-speculation mechanism.
