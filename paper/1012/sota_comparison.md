# Is anything faster than ours (14B 2-step at 576x320 + SR, 6.9 s per 153 frames)? — survey and head-to-head, 2026-10-09

## 1. Survey (action-conditioned, interactive video world models; status as of 2026-10-09)

| model (org, date) | open weights | params | input / actions | resolution, speed (GPU as reported) | comparable to ours? |
|---|---|---|---|---|---|
| **Waypoint-1.5-1B** (Overworld, Apr 2026) | yes, Apache-2.0 (HF `Overworld/Waypoint-1.5-1B`) | 1.2 B AR diffusion transformer, 4 frames / call | start image or video, keyboard + mouse (no text) | 360p–720p, "up to 60 fps", laptop GPUs; **30 fps at 1024x512 on our GPU** | yes — run here |
| **Matrix-Game 2.0** (Skywork, Aug 2025) | yes (HF `Skywork/Matrix-Game-2.0`) | 1.8 B causal (Wan-1.3B class), 3 steps | image, keyboard (WASD) + mouse, no text | 640x352, 25 fps on H100 claimed; **20.5 fps on our GPU** | yes — run here |
| Matrix-Game 3.0 (Skywork, Mar 2026) | yes, Apache-2.0 | – | image + keyboard/mouse | 720p "40 fps" | not run (not fetched) |
| **Matrix-Game 3.5** (Riemann Dynamics / Kunlun, Jul 2026) | yes, Apache-2.0 | 5 B (Wan2.2-TI2V-5B), distilled 3-step causal + patch memory | **image + camera trajectory (.npz c2w + K) + text** — same interface as ours | 704x1280, 20 fps on one ≥40 GB GPU | **best match; needs ~40 GB of weights (Wan2.2 DiT 20 GB + distilled 10 + VAE 2.8 + DA3 6.8) — not runnable with 9 GB free on /home; commands below** |
| ABot-World-0 (AMAP, Jul 2026) | yes (github amap-cvlab/ABot-World) | – | raw keyboard actions | 720p, 16 fps on RTX 5090, 1.2 s first-frame latency, 19 GiB | plausible, not run |
| DreamForge-World 0.1 (Jun 2026) | yes | – | keyboard + mouse, re-prompting | 480p, 14–15 fps on RTX 4090 | plausible, not run |
| InSpatio-WorldFM (Mar 2026) | yes, Apache-2.0 | – | camera exploration | 512x512, 10 fps A100 / 7 fps 4090, single step | not run |
| Hunyuan-GameCraft-2 (Tencent, Nov 2025) | partial | 14 B MoE, self-forcing distilled, block-sparse attn | text + keyboard/mouse | "1080p 60 fps real-time streaming" claimed | not run |
| LingBot-World v2 (Robbyant, 2026) — our base | yes | 14 B causal-fast / 1.3 B causal-fast | image + text + camera trajectory | official "≈16 fps, <1 s latency" (multi-GPU); community port: 1.3 B at 16 fps on RTX 5090; **our 1.3 B draft: 10 fps at 832x480 on one GPU** | this is us |
| Oasis 500M (Decart/Etched), MineWorld (MS), Genie 3 (DeepMind, closed), Project Genie | Oasis/MineWorld open (Minecraft only); Genie closed | – | keyboard | Oasis 20 fps on H100; Genie 3 720p 24 fps | domain-specific / closed |

Answer to the question: **yes — Waypoint-1.5-1B (30 fps) and Matrix-Game 2.0 (20 fps) are at or above our generated frame rate (17.8–22 fps), and Matrix-Game 3.5 / GameCraft-2 claim more.** They are 1–2 B-parameter models trained on game footage with keyboard/mouse actions; none of them accepts our camera-trajectory + text + photo interface except Matrix-Game 3.5.

## 2. Head-to-head on our 10 prompts (same start image, same camera path, 153 frames, seed 1234, one RTX PRO 6000 Blackwell 95 GB each)
Camera trajectories (`poses.npy`) were converted per model: Waypoint mouse = 118 × degrees per 4-frame call (calibrated by optical flow, `tools/waypoint/poses_to_actions.py`), W/S/A/D from translation; Matrix-Game 2.0 mouse = 0.067 × degrees per frame clipped to ±0.1 (its bench magnitude), keyboard one-hot from translation (`tools/mg2/poses_to_cond.py`). Both are approximations of a camera path in an FPS-controller action space.

| model | resolution | gen s / 153 frames ↓ (median) | generated fps ↑ | gen s per prompt (lake, greatwall, stonehenge, rain, dragon, pouring, domino, hourglass, cradle, billiard) |
|---|---|---|---|---|
| 14B 4-step teacher (ours' reference) | 832x480 | 31.2 | 4.9 | 29.8 30.1 29.9 29.9 29.9 32.3 32.3 32.3 32.3 32.3 |
| **ours: 14B 2-step 576x320 + SR** | 576x320 → 832x480 | **6.9** (8.6 for the 5 stevo tasks) | 17.8–22 | 6.8 6.9* 6.8 6.8 6.8 6.9* 8.6 8.6 8.6 8.6 |
| **Waypoint-1.5-1B** | 1024x512 | **5.4** | **28.3** | 5.4 ×10 |
| Matrix-Game 2.0 | 640x352 | 7.5 | 20.5 | 7.5 7.4 7.5 7.5 7.5 7.4 7.5 7.5 7.5 7.5 |

\* first run of a new input shape recompiles (32.8 / 19.4 s); the steady values were re-measured in a separate process (6.9 / 6.9 s), see verification.

Video: `paper/1012/videos/sota_cmp10.mp4` (1x4 per prompt: teacher | ours | Waypoint | Matrix-Game 2.0). What the frames show (my reading):
- Waypoint and Matrix-Game 2.0 render every photo as a game world (Matrix-Game turns the lake into a game shoreline, domino into a game room; Waypoint keeps the lake but smears the domino room into mush). Ours and the teacher stay photographic. Waypoint under-rotates on long camera sweeps; Matrix-Game under-rotates because its mouse range is clipped.
- On the physics prompts (pouring, domino, cradle, billiard) neither game model shows the described action; ours/teacher show the hand but also hallucinate people (domino) as before.
- Action following (easy_lake, camera yaw +129° over the clip; cumulative horizontal flow, negative = turning right, 5.1 px per degree calibrated on the teacher): teacher −143°, ours −183°, Waypoint −112°, Matrix-Game 2.0 −47°. All four turn the right way; Matrix-Game turns a third as far.

## 3. Verification
- Re-timed in fresh processes: ours greatwall 6.9 s and pouring 6.9 s (vs 32.8 / 19.4 s first-shape runs → recompile, not model cost); Waypoint lake 5.4 s again (p50 121 ms per 4-frame call, identical); Matrix-Game 2.0 lake 7.57 s (vs 7.5). Frame counts: 153 (ours, teacher, MG2), 156 (Waypoint: 39 calls × 4). All on the same GPU type; nvidia-smi was down (NVML driver mismatch) so GPU sharing could not be checked except via `torch.cuda.mem_get_info` (0–3 free).
- Waypoint eager mode is 3.5 s per call (1.1 fps); the 30 fps number requires `--compile` (both measured). Matrix-Game 2.0 runs with a torch-SDPA shim for flash-attn (`tools/mg2/shim`); its official `inference.py` reproduces a coherent 153-frame rollout with the shim; its VAE decoder is run eager in bf16 (fp16 + compile gave the same results, just a 5-minute compile).
- Matrix-Game 2.0 with zero actions degrades into texture noise by frame ~100 on the lake photo (out-of-distribution for a game-trained model), and a constant 0.1 yaw produces a black wedge — so its outputs on our photos are not representative of its game-domain quality.

## 4. To run Matrix-Game 3.5 (the only one with our exact camera-trajectory interface) — needs ~40 GB freed
```
cd /tmp/mg35/Matrix-Game-3.5   # already cloned
hf download RiemannDynamics/Matrix-Game-3.5-Distilled --local-dir checkpoints/Matrix-Game-3.5-Distilled   # 10 GB
hf download Wan-AI/Wan2.2-TI2V-5B --exclude "assets/*" "examples/*" --local-dir checkpoints/Wan2.2-TI2V-5B   # 34 GB (T5 11 GB could be symlinked from lingbot-world-v2-14b-causal-fast/models_t5_umt5-xxl-enc-bf16.pth if identical)
hf download depth-anything/DA3NESTED-GIANT-LARGE-1.1 --local-dir checkpoints/DA3NESTED-GIANT-LARGE-1.1   # 6.8 GB
python infer_distilled.py --config configs/infer_distilled_6blocks.yaml --checkpoint checkpoints/Matrix-Game-3.5-Distilled/first-person.safetensors --image IMG --camera cam.npz --prompt "..." --output out.mp4
```
cam.npz = {extrinsics_c2w: our poses.npy, intrinsics: our intrinsics.npy} (same convention as our binding). Candidates to delete for space (need approval): specwm_out/gap_dump_v3 13 GB and gap_dump_cross 5.3 GB (head training dumps, regenerable), specwm_out/stevo 15 GB (judged StEvo videos).

Sources: [Waypoint-1.5-1B](https://huggingface.co/Overworld/Waypoint-1.5-1B-360P), [world_engine](https://github.com/Wayfarer-Labs/world_engine), [Matrix-Game 2.0](https://arxiv.org/html/2508.13009v3), [Matrix-Game 3.5](https://github.com/Riemann-Dynamics/Matrix-Game-3.5), [ABot-World-0](https://arxiv.org/html/2607.19191v1), [DreamForge-World 0.1](https://huggingface.co/papers/2606.30292), [InSpatio-WorldFM](https://inspatio.github.io/worldfm/), [Hunyuan-GameCraft-2](https://arxiv.org/abs/2511.23429), [LingBot-World](https://github.com/robbyant/lingbot-world-v2), [lingbot-world-v2-realtime](https://github.com/kaarelkaarelson/lingbot-world-v2-realtime), [Robbyant press](https://www.businesswire.com/news/home/20260128459962/en/Robbyant-Open-Sources-LingBot-World-a-World-Model-for-Millisecond-Level-Real-Time-Interaction), [cinevva guide](https://app.cinevva.com/guides/frontier-gen-ai-models).

## 5. Matrix-Game 3.5 (added after freeing Tier-1 disk, 2026-10-09 PM)
Setup: `tools/mg35/Matrix-Game-3.5` (repo copy), checkpoints 48 GB (Distilled 9.4 + Wan2.2-TI2V-5B 22 + DA3 6.3; the 11.4 GB umt5-xxl is byte-identical to ours and symlinked), our venv + OpenEXR/pycolmap/decord/moviepy/modelscope. Inputs exactly ours: start image, `camera.npz` = {extrinsics_c2w: our poses.npy (168 poses for 2 blocks), intrinsics: our [fx,fy,cx,cy]}, prompt = our text. Config `configs/infer_distilled_2blocks.yaml` (3-step causal student, CFG 3 → 2 forwards per step, 704×1280, patch memory on). Timing = wall time of the whole inference call after model load (rollout + DA3 depth + mosaic memory + decode), `[MG35 TIMING]` patch in `tools/run_distilled_inference.py`.

| model | resolution | s per clip ↓ | frames | generated fps ↑ | repeat |
|---|---|---|---|---|---|
| Matrix-Game 3.5 distilled, 2 blocks | 1280×704 | **171.8** (median of 10; 167–186) | 169 | **0.9** | lake 177.9 → 176.8 |
| same, denoise only (progress bars: 14 chunks × 3 steps × 1.1 s, first chunks 0.6 s) | | ≈ 45 | | ≈ 3.6 | |
| same, 1 block | | 92.5 | 89 | | → fixed cost ≈ 8 s, ≈ 85 s per 80-frame block |
| ours 576×320 + SR | | 6.9 / 8.6 | 153 | 17.8–22 | |

So on our GPU and with the released inference stack, Matrix-Game 3.5 is **25× slower than ours end-to-end and ~5× slower in pure denoising**, far from the paper's 20 fps (which presumably needs their serving stack / a different resolution; the released `infer_distilled.py` spends ~9 s per chunk in depth estimation + patch-memory reprojection on top of ~3.3 s of denoising). Peak VRAM 63.8 GB.

Quality (`videos/sota_cmp10.mp4`, 5th column; frames at 0/40/80/120/160): urban rain → a coherent, sharp game-style street that keeps the graffiti wall for ~1 s then becomes a different city; lake → a park with trees by frame 80; dragon → the dragon and castle vanish into a dark forest by frame 40; pouring / domino (tabletop, orbit camera) → white corridors from frame 40 on. It is a first-person navigation model: it follows the camera path (lake: −105° equivalent cumulative rotation vs ours −183°, teacher −143°) but does not keep the photo's scene or any tabletop physics content. Caveat: our camera translations are in arbitrary units, while its patch memory uses metric depth — a scale mismatch may push its memory into "new territory" faster than intended; its own samples use ~mm-scale per-frame translations.
