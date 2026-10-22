# On-device interactive world models — baseline tables for the paper (draft 2026-10-16, 13:45 CDT)

작성 원칙: (1) **reported** = 저자/모델 카드 숫자(출처 명시), **measured** = 우리가 한 장의 RTX PRO 6000 Blackwell 96 GB에서 같은 프로토콜로 잰 숫자. 두 종류를 한 칸에 섞지 않는다. (2) 표마다 ↑/↓, **bold** = best, <u>underline</u> = second. (3) 비어 있는 칸은 "n/m"(not measured)으로 두고 §6의 TODO에 비용과 함께 적는다. (4) fps는 **pixel frames / s** (latent fps 아님); 생성만(디코드 제외)인지 디코드 포함인지 열에 명시. (5) 4090 실측은 없다. 모든 measured fps는 PRO 6000 숫자이고 4090은 별도 측정 전까지 "추정"으로도 적지 않는다.

출처: 사용자 조사 `specwm/4090_search.txt`(10/15), `paper/1012/sota_comparison.md`(10/9), facts §2/§9/§13/§18–20/§35/§42/§47, `paper/meeting_1012/research_mg2_waypoint_eval.md`(WBench).

---

## 1. Candidate pool: public interactive world models with a ≤ 24 GB (consumer GPU) claim — reported numbers only

Tier A = 저자가 데스크톱 4090 실행/벤치를 명시, B = 공식적으로 ≤ 24 GB 또는 다른 소비자 GPU 검증, C = 공개 코드는 있으나 24 GB 근거 없음. "Text" = 초기 장면 캡션만(런타임 이벤트 제어 아님).

| model (org, date) | params / base | inputs: image · text · action | native res, chunk, steps | reported device evidence (GPU · VRAM · fps · latency) | weights | tier | our status |
|---|---|---|---|---|---|---|---|
| **Waypoint-1.5-1B** (Overworld, 2026-04; arXiv 2609.37107) | 1.28 B causal DiT | image/video · no · keyboard + mouse, 4 frames per call | 360p–720p; 1024×512; 4 frames/call | 4090: web page "72.88 video FPS INT8 / 66.6 BF16 at 1024×512" but paper Table 2 calls the same number **latent** fps; HF card has 3090/5090 tables → unit inconsistent, re-measured by us | HF Overworld/Waypoint-1.5-1B (Apache-2.0) | A | **run** (bench-43 + cmp10) |
| **ABot-World-0-5B-LF** (AMAP, 2026-07; arXiv 2607.19191) | 5 B, Wan2.2-TI2V-5B causal student (LongForcing DMD) | image · caption · 8 keys WASD + IJKL (multi-hot) | 1280×704, 12-frame blocks, 4 steps, local KV 21 | 5090: ≤ 19.3 GiB, 12.4–13.3 fps FP8 (16 fps low-bit), 1.2 s action→first frame; FAQ "verified on RTX 4090" (no 4090 fps) | github amap-cvlab/ABot-World, HF acvlab (teacher not released) | A | **run** (12 prompts, 1280 + 704) |
| **Matrix-Game 2.0** (Skywork, 2025-08; arXiv 2508.13009) | 1.8 B causal (Wan-1.3B class), Self-Forcing | image · **no text** · keyboard + mouse | 640×352, 3 steps | official "GPU ≥ 24 GB"; **25 fps is one H100** | HF Skywork/Matrix-Game-2.0 | B | **run** (bench-43 + cmp10) |
| **Astra** (ICLR 2026; github EternalEvan/Astra) | Wan2.1-T2V-1.3B + AR denoising | image · prompt · **camera trajectory** (cam_type / custom) | 832×480 | "single 24 GB GPU" inference (training 60 GB); no fps given | github | B | not run — **fairest camera-trajectory baseline, next** |
| Yume-5B (stdstu12/YUME) | 5 B | image · text (per-2 s caption) · motion/camera keys | 720p, 4–50 steps | RTX 4090 **Laptop 16 GB** web demo tested; no real-time claim | HF stdstu123/Yume-5B-720P | B | not run |
| Hunyuan-GameCraft 1.0 (Tencent, 2025) | HunyuanVideo 13 B, 8-step distilled variant | image · prompt · keys → camera space | 1216×704, 33 frames per action | 24 GB single-GPU path = CPU offload + FP8, README "very slow"; "25 fps" is playback | github | B | not run (offline only) |
| ForgeWM (arXiv 2608.14022) | small causal DiT, 1/2/4-step ckpts + all distillation stages | image · – · keyboard + continuous mouse | 640×352, 12 frames/chunk, local_attn 6 | H20: 32.5 / 50.3 / 72.1 fps for 4/2/1 steps, **VAE excluded**; no 4090 numbers | HF ForgeWM/ForgeWM | C | not run |
| minWM (shengshu-ai) | Wan2.1-1.3B / HunyuanVideo-1.5 stagewise ckpts | Wan path: text + camera (no image); HY path: image + text + camera | 832×480, 77 frames | no 24 GB verification; "16 fps" is playback | HF MIN-Lab/minWM | C | not run |
| DIAMOND CSGO (eloialonso/diamond) | small UNet diffusion | game state · – · game actions | CSGO only | **3090: 10 fps** high-quality setting | github | B | not run (game-only) |
| Oasis-500M (Etched/Decart) | 500 M | video prompt · – · Minecraft keys | Minecraft | no 4090 memory/fps in repo | HF Etched/oasis-500m | C | not run (game-only) |
| Interactive World Simulator (Wang et al.) | latent consistency model | observation · – · robot/WASD end-effector | **128×128** | 4090: 15 fps, 10 min stable; min GPU 2080 | github | A | not run (task-specific, low-res) |
| CLAP (omni-CLAP) | – | observation · language/pose/latent action modes | 11 frames, 25 steps | ≥ 12 GB GPU, ~10 GB model; 2.9 s/chunk A100 | github | B | not run (not real-time) |
| LingBot-World v2 1.3B causal-fast (Robbyant; our draft) | 1.3 B causal, 4-step DMD | image · prompt · camera trajectory | 832×480, 12-frame chunks, 4 steps | community: 16 fps on RTX 5090 | HF Robbyant | B | **run** (bench-43, ladder §20) |
| LingBot-World v2 14B causal-fast (our base) | 14 B causal, 4-step | image · prompt · camera trajectory | 832×480, 4 steps | official fast inference = **8 GPUs**; NF4 = base-cam only | HF Robbyant | – | **ours** (single-GPU FP8 stack = our port) |

Excluded from the on-device set (reason): Matrix-Game 3.5 (5 B, same camera-path interface, measured 0.9 fps e2e / 63.8 GB on our GPU = cloud class); Matrix-Game 3.0 (5 B, no 24 GB evidence); SANA-WM (BF16 47 GB / FP8 35 GB / FP4 29 GB Blackwell); HY-WorldPlay 8B (72 GB); DreamForge-World 0.1 (4090 480p 14–15 fps claimed, **weights not released**); Zing-0.5 (H100 context); MineWorld (ckpt withdrawn); DreamX-World (no 24 GB range stated); InSpatio-WorldFM (4090 512² 7 fps but 3D-anchor view synthesis, different task); Matrix-Game 4.0 (game engine does the dynamics).

---

## 2. Measured head-to-head on ONE RTX PRO 6000 (our prompts, our protocol) — the §7 checklist columns

Protocol: same start image + caption + camera path per task; camera path converted to each model's native action encoding (lossy for keyboard models: Waypoint mouse ≈ 118 units/deg per 4-frame call, Matrix-Game 2.0 mouse 0.067×deg/frame clipped ±0.1, ABot 8 keys per 12-frame block); 161 frames (13 chunks) unless noted; seed 1234 (+4321 on bench-43). fps = steady pixel frames/s (median over chunks ≥ 5, generation only for ours / incl. own decoder for MG2 and Waypoint as noted). Quality/drift = Qwen3-VL-32B clip judge (quality 1–10 ↑, drift 0–100 ↓; drift prompt AUROC 0.82 vs human labels, quality prompt not human-validated). Δq = paired difference vs the named reference on the same clips.

| model · stack (ckpt / precision / attention / VAE) | res | frames per chunk · steps | steady fps ↑ (decode?) | Mpix/s ↑ | TTFF ↓ | peak VRAM ↓ (GiB) | KV window | quality ↑ / drift ↓ | Δq vs reference (paired) | prompt set | caveats |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ours 14B teacher 4-step · FP8 w, sage attn, full VAE | 832×480 | 12 · 4 | 4.08 (gen only) | 1.63 | 1.71 s (§13) | 48.4 (bf16 KV w18) | 18 | **7.51** / **51.0** | reference (832) | bench-43 ×2 seeds (86) | |
| ours 14B teacher 4-step | 704×384 | 12 · 4 | 8.5 | 2.30 | 1.41 s | ≈ 39 / 48.1† | 18 | 7.33 / 38 (12) | reference (704) | 12 prompts / bench-43 | |
| ours rt2 (2-step, fused + partial) | 704×384 | 12 · 2 | <u>15.7</u> | 4.24 | **0.76 s** | 38.9 / 48.1† | 18 | 6.90 / 56.9 | −0.62 ± 0.09 (832 ref) | bench-43 (86) | |
| ours adapt25 (rt2 + step-1 head, 3rd step on 21 % chunks) | 704×384 | 12 · 2–3 | 15.3 | 4.14 | ≈ 1.1 s (chunks 0–2 are 3-step) | ≈ 39 | 18 | – | **−0.05 ± 0.10** (704 ref, 6/28/9) | bench-43 (43) | §47 |
| ours adapt25c1hyb device stream (+ cloud 1-step + KV feedback) | 704×384 | 12 · 2–3 | 11.1 | 3.00 | ≈ 1.1 s | ≈ 39 + cloud GPU | 18 | – | +0.00 ± 0.12 (704 ref); cloud stream +0.07 ± 0.11 | bench-43 (43) | needs a second GPU, cloud lag 1.08 s (§47c) |
| ours 24 GB config: GPTQ-4 w, FP8 KV, window 12, tiled decode (cap-24 run) | 704×384 | 12 · 2 | 11.3 | 3.05 | n/m | **17.0** | 12 | – | −0.14 ± 0.09 vs rt2 (3/30/10) | bench-43 (43) | §19; INT4 kernels give no speed (dequant → bf16) |
| ours 32 GB config: FP8 w, FP8 KV, window 12 | 704×384 | 12 · 2 | **17.5** | <u>4.73</u> | n/m | 29.4 | 12 | – | −0.07 ± 0.12 vs rt2 | bench-43 (43) | §19 |
| ours 32 GB config at 832: FP8 w, FP8 KV, window 12, tiled | 832×480 | 12 · 2 | 11.3 | 4.51 | n/m | 28.8 | 12 | – | −0.05 ± 0.10 vs ours 832 | bench-43 (43) | §19 |
| LingBot 1.3B causal-fast student · FP8 | 832×480 | 12 · 4 | 27.5 (gen only; 17.2 under load) | 11.0 | 0.44 s (§13) | <u>7.4</u> | 18 | 5.83 / 75.1 | −1.69 ± 0.17 (832 ref) | bench-43 (86) | §20: 704/w12 54.5 fps in 4.7 GiB at the same quality; 2-step −1.5 more |
| SDVG τ −0.7 (Hu & Zhang) on the same base | 832×480 | 12 · draft+verify+4 | 3.1 | 1.24 | 1.79 s | n/m | 18 | <u>7.51</u> / 52.9 | +0.00 ± 0.07 (832 ref) | bench-43 (86) | method baseline; 42/86 clips take no draft chunk |
| SDVG τ −2.2 | 832×480 | 12 · mixed | 6.7 | 2.68 | 1.79 s | n/m | 18 | 6.67 / 70.6 | −0.84 ± 0.10 | bench-43 (86) | |
| **Matrix-Game 2.0** · official ckpt, bf16, flash-attn→SDPA shim, no compile, own VAE | 640×352 | n/m · 3 | 20.5 (incl. decode) | 4.62 | n/m | 11.6 | – | 6.91 / 70.0 | −0.60 ± 0.09 (832 ref); vs ours 704 −0.01 ± 0.10 | bench-43 (86) | game domain: photos degrade (lake → noise by frame 100 with zero actions); 2-step 448×256 = 34.3 fps, q 5.7 |
| **Waypoint-1.5-1B** · official ckpt, bf16, torch.compile max-autotune (eager = 1.1 fps), own decoder | 1024×512 (image stretched to 2:1) | 4 per call · n/m | 28–31 (incl. decode) | **14.8–16.3** | 0.52 s (first call) | **6.7** | rolling | 4.03 / 75.9 | −3.48 ± 0.15 (832 ref) | bench-43 (86) | no text input; under-rotates long sweeps; game-style rendering of photos |
| **ABot-World-0 5B-LF** · official ckpt, FP8 per-tensor, flash/sage, taew2_2 tiny VAE, T5 resident | **1280×704** | 12 · 4 | 13.3 (incl. tiny VAE) | 12.0 | 1.57 s | 32.0 (official 19.3 with offload) | 21 | 6.42 / 58 | −0.92 ± 0.26 (704 ref, 1/2/9); roaming −1.75, physics −0.50 | 12 prompts | keyboard conversion lossy (pitch keys over-rotate); ignores caption events (pour stops) |
| ABot-World-0 5B-LF | 704×384 (off-distribution) | 12 · 4 | 23.3 | 6.30 | 1.24 s | 22.5 | 21 | 5.00 / 76 | −2.33 ± 0.50 (704 ref) | 12 prompts | scene collapses → no matched comparison at our res |
| Matrix-Game 3.5 distilled (cloud class, for reference) · official infer_distilled.py | 1280×704 | – · 3 | 0.9 e2e (3.6 denoise only) | 0.81 | n/m | 63.8 | patch memory | – | loses the photo scene by frame 40–80 | cmp10 | depth + patch-memory reprojection ≈ 9 s/chunk |

† 38.9 GiB = fit24 device stack (T5 off GPU, streamed decode); 48.1 = two_machine stack with T5/CLIP resident. Mpix/s = res × fps (pixel throughput; normalizes the resolution mismatch). "n/m" = not measured.

Reading (for the text): 지금 공개된 ≤ 24 GB 모델 중 우리 프롬프트에서 품질이 우리 teacher에 근접하는 것은 없다(최선 MG2 −0.60, ABot −0.92). 속도·메모리는 1–5 B 모델이 이기고(Waypoint 16 Mpix/s in 6.7 GiB, ABot 12 Mpix/s in 32 GiB), 우리 14B edge-only는 4.1–4.7 Mpix/s in 17–39 GiB. 즉 "edge-only SOTA"와의 비교는 모델 크기 차이가 지배하므로, 논문은 (a) 같은 base에서 teacher 대비 속도 이득(apples-to-apples)과 (b) pixel-throughput 대 품질 곡선에서 외부 모델 위치, 두 가지로 나눠 보여야 한다.

---

## 3. Matched resolution at ABot's native 1280×704 (one PRO 6000, 12 prompts, FP8, KV window 12; facts §42 addendum)

| model at 1280×704 | s per chunk ↓ | fps ↑ | real-time factor ↑ | peak GiB ↓ | quality ↑ | drift ↓ | ABot − this (paired Δq) |
|---|---|---|---|---|---|---|---|
| ours 14B teacher 4-step | 5.969 | 2.0 | 0.13× (16 fps) | 83.8 | <u>7.83</u> | **40.8** | −1.42 ± 0.34 (1/1/10); roaming −2.25 |
| ours 14B solo3 (0,500,750) | 4.613 | 2.6 | 0.16× | 83.8 | **8.00** | 44.6 | −1.58 ± 0.26 |
| ours 14B rt2 2-step | 3.276 | <u>3.7</u> | 0.23× | 83.8 | 7.92 | <u>42.1</u> | −1.50 ± 0.31 (0/2/10); roaming −2.50 |
| ABot-World-0 5B (FP8, tiny VAE) | **0.902** | **13.3** | **1.11×** (12 fps) | **32.0** | 6.42 | 57.5 | 0 |

읽기: ABot의 해상도에서 우리 14B는 1.4–1.6 점 더 좋고 drift 13–17 낮지만 3.6배 느리고 메모리 2.6배. 이 차이는 5B 대 14B이지 방법 차이가 아니다. 같은 4-node schedule이므로 우리 step split / KV fit / feedback을 ABot에 이식하는 portability 실험이 "방법" 비교가 된다.

---

## 4. Device-budget rows for the 14B (emulated with a CUDA allocator cap on the PRO 6000; facts §18–19; 704×384 2-step, 43 prompts)

| device budget | config | peak reserved ↓ (GiB) | fps (PRO 6000) ↑ | Δ quality vs same mode unquantized ↑ | note |
|---|---|---|---|---|---|
| 24 GB (4090 / A5000) | GPTQ-4 w, FP8 KV, window 12, tiled decode | **17.0** | 11.3 | <u>−0.14 ± 0.09</u> | GPTQ (3 calib chunks, 13 min) ≫ RTN −0.70 / AWQ −0.60; at 832 GPTQ-4 is −0.58 |
| 24 GB | RTN-INT4 w, FP8 KV, window 18, tiled | <u>19.4</u> | 10.3 | −0.70 ± 0.14 | |
| 32 GB (5090) | FP8 w, FP8 KV, window 18, streamed | 31.9 | 15.1 | −0.12 ± 0.11 | |
| 32 GB | FP8 w, FP8 KV, window 12, streamed | 29.4 | **17.5** | **−0.07 ± 0.12** | |
| 32 GB | 832×480, FP8 w, FP8 KV, window 12, tiled | 28.8 | 11.3 | −0.05 ± 0.10 | quality-mode resolution fits 32 GB |
| 8 GB (draft only) | 1.3B 4-step 704 window 12 | 4.7 (gen peak) | 54.5 | −0.02 ± 0.17 vs 1.3B 832 | §20; quality −1.72 vs teacher |

주의: 4090에서의 fps는 실측이 없다(4090은 메모리 대역폭 1 TB/s, FP8 텐서 처리량이 PRO 6000의 1/2–1/3 수준이므로 더 낮을 것이나 숫자를 적지 않는다). adapt25의 step-1 head는 FP8 feature로 학습되었으므로 GPTQ-4 경로에서는 재검증 필요(§6).

---

## 5. Neutral third-party benchmark (reported by the WBench authors, arXiv 2605.25874; navigation split; not our measurement)

| model | Avg ↑ | Quality ↑ | Setting ↑ | Interaction ↑ | Consistency ↑ | Physical ↑ |
|---|---|---|---|---|---|---|
| LingBot-World (fast v2) = our base | **79.4** | **81.8** | <u>76.8</u> | <u>82.8</u> | <u>86.5</u> | <u>69.1</u> |
| LingBot-World (base-camera) | <u>78.5</u> | 78.9 | 72.6 | 80.1 | **89.9** | **71.2** |
| LingBot-World (fast) | 77.4 | <u>79.4</u> | **77.9** | 79.2 | 84.9 | 65.7 |
| YUME 1.5 | 73.3 | 77.6 | 72.4 | 71.4 | 80.1 | 65.2 |
| Matrix-Game 3 | 71.3 | 75.5 | 63.6 | **83.6** | 74.5 | 59.3 |
| Matrix-Game 2 | 68.7 | 73.8 | 67.1 | 80.3 | 65.1 | 57.2 |

용도: 우리 base가 외부 리더보드에서 MG2/MG3/YUME보다 위라는 독립 근거. WBench에는 LingBot-v2 어댑터(`examples/lingbot/`)가 있어 우리 realtime/adapt25 모드를 같은 리더보드에 올릴 수 있다(VLM judge API 키 필요한 지표 있음).

---

## 6. What is still missing for a fair table (cost estimate, our GPUs)

| item | why | cost |
|---|---|---|
| Astra (image + prompt + camera trajectory, 832×480, 24 GB) | the only ≤ 24 GB baseline with our exact interface | install ½ day + 43 clips ≈ 2 h on 1 GPU |
| Waypoint / MG2 / ABot on **their own** scenes with native inputs; ours on the same scenes via key→camera conversion | removes the "our scenes" bias (2×2 design); GameWorld Minecraft protocol (32 images, Bench76) exists for MG2 | 1 day, 2 GPUs |
| yaw/translation following error column for every model (teacher-calibrated optical flow) | makes the keyboard-conversion loss visible instead of hidden in quality | ½ day (script exists for easy_lake) |
| MG2 with its official flash-attn + compile (our run used an SDPA shim) | MG2 fps may be under-reported | ½ day (rebuild env; weights re-download 5 GB) |
| 4090 (or 5090) hardware run of: ours 24/32 GB configs, Waypoint, MG2, ABot | all fps here are PRO 6000 | needs a machine; 1 day once available |
| adapt25 on the 24 GB path (merge fit24 quantization into two_machine; re-validate the step-1 head on GPTQ-4 features) | the realtime recommendation must exist at 24 GB | 1 day |
| Yume-5B, ForgeWM, GameCraft (offline) | complete tier B/C coverage; ForgeWM gives a 1/2/4-step ladder like ours | 1 day each |
| small blind human rating on both scene sets | the VLM quality prompt is not human-validated | 1 h of rater time per 40 clips |
| WBench navigation run of our realtime / adapt25 modes | interface-neutral leaderboard | 1 day (adapter exists; judge API key) |
