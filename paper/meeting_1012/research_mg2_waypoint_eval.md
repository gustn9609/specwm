# Evaluating LingBot-World v2 on the home turf of Matrix-Game 2.0 and Waypoint-1.5

Research date: 2026-10-12. Read-only web and code research; no GPU runs. All sources below were opened or cloned during this research. Code was cloned into the session scratchpad (`/tmp/mg2/Matrix-Game` was gone, so the repo was re-cloned at HEAD).

Conventions: ↑ = higher is better, ↓ = lower is better. In each results table, **bold** = best and <u>underline</u> = second among the rows shown. Two-row tables only bold the best.

---

## 0. TL;DR

| Question | Matrix-Game 2.0 (MG2) | Waypoint-1.5-1B (WP) |
|---|---|---|
| Official quality benchmark | **GameWorld Score** (from Matrix-Game 1.0): 8 metrics, Minecraft split (32 images) and "wild" split (16 images) | **None.** The paper (Sec. 8) evaluates only throughput (latent FPS across 10 consumer GPUs). The model card has an FPS table and demos only. |
| Action-control metric | Keyboard and mouse accuracy from **OpenAI VPT inverse-dynamics model (IDM)**, Minecraft only. Not reported on the wild split. | none |
| Eval code public? | **Yes:** `Matrix-Game-1/GameWorldScore/` (MIT). Needs GPU, VPT IDM, DROID-SLAM, CLIP, MUSIQ, AMT, LAION-aesthetic weights. No API keys. | No quality eval code. `world_engine/examples/benchmark.py` is a throughput benchmark. |
| Test images public? | Minecraft: **yes**, 8 biomes × 4 = 32 PNGs in `GameWorldScore/asset/init_image/`. Wild: **no**, the 16 images are not released (closest proxy: 17 images in `Matrix-Game-2/demo_images/universal/`). | Starter images in the HF Space `Overworld/waypoint-1-small` (e.g. `starter_18.png`), used in the model-card example. |
| Action sequences | MG1 protocol: 76 fixed constant actions × 65 frames (`Bench76/*.jsonl`) plus 8 "mirror" actions. MG2 paper: one **597-frame composite sequence** (= 150 latents; very likely `Bench_actions_universal`: random 12-frame action blocks). | Demos only (random `CtrlInput` choices in the examples). |
| Neutral third-party benchmark | **WBench** (Meituan LongCat, arXiv 2605.25874) navigation split: MG2 **68.7** avg; LingBot-World (fast v2) **79.4**. It already bridges 6-DoF pose and keyboard actions. | Not on WBench. |

**Recommended protocol (Section 5):**
- (A) Run the GameWorld Minecraft protocol with our model: 32 images, Bench76 + mirror actions converted to camera trajectories, scored with their code unchanged.
- (B) Replace or augment the IDM-based accuracy with a **pose-based accuracy** (DROID-SLAM is already a GameWorld dependency). Apply it to MG2, WP and us alike.
- (C) Report the **WBench navigation split** with its public LingBot-v2 adapter. That puts us, MG2 and MG3 on a published, interface-neutral leaderboard.
- (D) For Waypoint, its home turf is speed. Report its own speed quantities: latent FPS, rendered FPS and control rate, on the same GPU. Quality comes from protocol A/B on game starter images.

---

## 1. Matrix-Game 2.0: what they report and how

### 1.1 Sources opened
- Paper: arXiv 2508.13009, "Matrix-Game 2.0: An Open-Source, Real-Time, and Streaming Interactive World Model". Read the HTML of the current version (v5, 29 Sep 2026; v1 was 18 Aug 2025): https://arxiv.org/html/2508.13009 and https://arxiv.org/abs/2508.13009
- Repo: https://github.com/SkyworkAI/Matrix-Game (cloned; `Matrix-Game-2/`, `Matrix-Game-1/GameWorldScore/`)
- HF: https://huggingface.co/Skywork/Matrix-Game-2.0 (file list via `https://huggingface.co/api/models/Skywork/Matrix-Game-2.0`)
- Project page: https://matrix-game-v2.github.io/
- Benchmark origin, Matrix-Game 1.0: arXiv 2506.18701 (https://arxiv.org/html/2506.18701), Sec. 5 (metrics) and 6 (results)

### 1.2 Benchmark: GameWorld Score (defined in MG1 Sec. 5, used by MG2 Sec. 5.1)

Four dimensions: visual quality, temporal quality, action controllability, and physical rule understanding. Implementation: `Matrix-Game-1/GameWorldScore/GameWorld/*.py` (VBench-derived).

| Metric (dir.) | Dimension | How it is computed (code file) | External model / weights |
|---|---|---|---|
| Image Quality ↑ | visual | Per-frame MUSIQ no-reference IQA, averaged (`imaging_quality.py`) | `musiq_spaq_ckpt-358bb6af.pth` (pyiqa) |
| Aesthetic Quality ↑ | visual | Per-frame LAION aesthetic predictor on CLIP ViT-L/14 features, score/10 (`aesthetic_quality.py`) | `sa_0_4_vit_l_14_linear.pth`, CLIP ViT-L-14 |
| Temporal Consistency ↑ | temporal | CLIP ViT-B/32 at 224 px: mean over frames of ½(cos(f_t, f_{t−1}) + cos(f_t, f_0)) (`temporal_consistency.py`). The MG1 paper text says "adjacent frames" only, but the code also uses the first frame. | CLIP ViT-B-32 |
| Motion Smoothness ↑ | temporal | AMT-S frame interpolation: reconstruction error of dropped frames, (255 − err)/255 (`motion_smoothness.py`) | `amt-s.pth` |
| Keyboard Acc. ↑ | action | VPT IDM (`4x_idm.model/.weights`, frames resized to **128×128**) predicts per-frame actions. Precision (micro) for 4 exclusive tasks: {forward, back, none}, {left, right, none}, attack {0, 1}, jump {0, 1}; then the mean over tasks (`action_control.py`, `third_party/IDM/IDM_bench.py`) | OpenAI VPT IDM (Minecraft-only, trained on 1,962 h; 90.6 % keyboard acc., R² 0.97 mouse, per MG1 Sec. 5.3) |
| Mouse Acc. ↑ | action | The IDM camera prediction (pitch, yaw) is mapped to 9 direction classes (none, up, down, left, right and 4 diagonals) with a dead zone of **1e-2**. The score is per-video frame accuracy vs. the scripted label, averaged over videos (`compute_camera_precision`). Only the **direction** is scored, not the magnitude. | same IDM |
| Object Consistency ↑ | physical / 3D | DROID-SLAM (fixed intrinsics fx = fy = 500, cx = cy = 256; frames resized to ~512² area) gives the mean reprojection error. Score = 1 − err/2.5 (`object_consistency.py`) | `droid.pth` (gdown id 1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh) |
| Scenario Consistency ↑ | physical / 3D | For a time-symmetric "mirror" action video of 2n+1 frames, compare frame i with frame −(i+1) at 256×256, allowing up to ±4 px shift. Score = 1 − mean\|Δ\|/255. The code uses mean absolute difference, although the paper says MSE (`scenario_consistency.py`) | none |

Notes and gaps:
- MG1 Sec. 5.4 says "lower error is better" for Object and Scenario consistency, but the tables mark them ↑. The code resolves this: both are converted to similarity, so ↑ is correct.
- The MG1 paper does not give the mouse threshold. The code uses 1e-2 on the IDM camera output.
- Requirements: CUDA GPU, PyTorch 2.4.1 / cu124, `torch_scatter`, DROID-SLAM compiled from `third_party/DROID-SLAM`, plus the weights listed above. **No API keys.** Multi-GPU via `torch.distributed` is optional.
- File-naming contract: `{prefix}_{action_name}.mp4`. The action name selects `Bench76/test_{action}.jsonl` for the IDM labels. `n_frames = video_frames − 1`, and labels start at jsonl line 1 (`infer_demo_num = 1`). Videos must therefore have at least 65 frames for the 65-line jsonl. The IDM camera unit is VPT degrees (jsonl uses ±5), but only the sign matters for scoring.

### 1.3 Test sets

| Split | Images | Public? | Actions | Clip length | Used by |
|---|---|---|---|---|---|
| GameWorld Minecraft | 32 = 8 biomes (beach, desert, forest, hills, icy, mushroom, plain, river) × 4 | **Yes**, `GameWorldScore/asset/init_image/<biome>/0{0..3}.png` | MG1: 76 constant actions (`Bench76`: 6 single keys, 8 camera directions, 13 key pairs, 48 key × camera) plus 8 mirror actions → 2,432 + 256 videos | 65 frames at 16 fps, 720p (MG1 `inference_bench.py` defaults) | MG1 Table 2 / 3 |
| MG2 Minecraft | "32 Minecraft scenes" (MG2 Sec. 5.1). Very likely the same 32, but not stated. | (likely) yes | "a 597-frame composite action sequence" (Sec. 5.1). 597 = (150 − 1)·4 + 1, which matches `inference.py --num_output_frames 150` with `Bench_actions_universal`: random 12-frame blocks drawn from {forward, left, right, forward ± left/right, camera l/r, and combinations}, seeded with `--seed`. The exact sequence is not published (**inferred**). | 597 frames at 352×640 | MG2 Table 1, Table 3 |
| MG2 wild | "16 diverse wild scene images" | **No** (proxy: 17 images in `Matrix-Game-2/demo_images/universal/`) | same composite sequence; keyboard/mouse accuracy **not** computed ("designed for Minecraft") | 597 frames | MG2 Table 2 |
| GTA driving, Temple Run | separate checkpoints (Temple Run: `templerun_distilled_model`, keyboard-only, 7 keys) | demo images in repo (`demo_images/temple_run`, 6) | — | — | qualitative only (Figs. 12–13) |

MG2 universal-checkpoint action space (`Matrix-Game-2/utils/conditions.py`):
- keyboard: 4-dim one-hot [forward, back, left, right]. No jump or attack, unlike MG1.
- mouse: [pitch, yaw] per pixel frame. The benchmark magnitude is `CAM_VALUE = 0.1`; camera_up = [+0.1, 0], camera_l = [0, −0.1].
- Input image: center-crop and resize to **352×640**.
- The README warns that upward camera motion can produce black frames.

### 1.4 Reported numbers

MG1 Table 2 (GameWorld Score, Minecraft; MG1 arXiv 2506.18701 Sec. 6.1). The same table appears in the `GameWorldScore/README.md`.

| Model | Image ↑ | Aesthetic ↑ | Temporal ↑ | Motion ↑ | Keyboard ↑ | Mouse ↑ | Object ↑ | Scenario ↑ |
|---|---|---|---|---|---|---|---|---|
| Oasis | 0.65 | <u>0.48</u> | 0.94 | **0.98** | 0.77 | 0.56 | <u>0.56</u> | 0.86 |
| MineWorld | <u>0.69</u> | 0.47 | <u>0.95</u> | **0.98** | <u>0.86</u> | <u>0.64</u> | 0.51 | <u>0.92</u> |
| Matrix-Game 1.0 (17B) | **0.72** | **0.49** | **0.97** | **0.98** | **0.95** | **0.95** | **0.76** | **0.93** |

MG2 Table 1 (Minecraft, 32 scenes, 597 frames; arXiv 2508.13009 Sec. 5.2). The same numbers are on the project page and HF card.

| Model | Image ↑ | Aesthetic ↑ | Temporal ↑ | Motion ↑ | Keyboard ↑ | Mouse ↑ | Object ↑ | Scenario ↑ |
|---|---|---|---|---|---|---|---|---|
| Oasis | 0.27 | 0.27 | 0.82 | **0.99** | 0.73 | 0.56 | 0.18 | **0.84** |
| Matrix-Game 2.0 | **0.61** | **0.50** | **0.94** | 0.98 | **0.91** | **0.95** | **0.64** | 0.80 |

The paper notes that Oasis collapses into static frames, which inflates its Motion and Scenario scores. Scenario consistency rewards freezing; keep this in mind for us too.

MG2 Table 2 (wild, 16 images, 597 frames):

| Model | Image ↑ | Aesthetic ↑ | Temporal ↑ | Motion ↑ | Object ↑ | Scenario ↑ |
|---|---|---|---|---|---|---|
| YUME | 0.65 | 0.48 | 0.85 | **0.99** | **0.77** | **0.80** |
| Matrix-Game 2.0 | **0.67** | **0.51** | **0.86** | 0.98 | 0.71 | 0.76 |

MG2 Table 3 (acceleration ablation, Minecraft; the GPU is not stated in the table, the Abstract and Introduction say 25 FPS on a single H100):

| Config | Image ↑ | Aesthetic ↑ | Temporal ↑ | Motion ↑ | Keyboard ↑ | Mouse ↑ | Object ↑ | Scenario ↑ | FPS ↑ |
|---|---|---|---|---|---|---|---|---|---|
| +VAE cache | **0.61** | **0.51** | <u>0.93</u> | <u>0.97</u> | <u>0.91</u> | **0.95** | **0.68** | **0.81** | 15.49 |
| +half action modules | **0.61** | **0.51** | **0.94** | <u>0.97</u> | **0.92** | **0.95** | <u>0.63</u> | **0.81** | <u>21.03</u> |
| +3 steps (final) | **0.61** | <u>0.50</u> | **0.94** | **0.98** | <u>0.91</u> | **0.95** | 0.64 | <u>0.80</u> | **25.15** |

Speed and resolution claims:
- 1.8B parameters (SkyReels-V2-I2V-1.3B / Wan 2.1 base plus action modules).
- 352×640, 3 denoising steps, 3 latents per chunk, local attention 6 latents.
- **25 FPS on one H100**. The repo README says it needs ≥ 24 GB and was tested on A100/H100.
- Demo videos are written at 12 fps (`utils/visualize.py`). This is the playback fps, not the generation speed.
- Our own measurement (paper/meeting_1012 notes s6) on an RTX PRO 6000 without torch.compile and with an SDPA shim: 19.7–20.0 fps.

---

## 2. Waypoint (Overworld)

### 2.1 Sources opened
- Model card: https://huggingface.co/Overworld/Waypoint-1.5-1B (raw README downloaded)
- Paper PDF linked from the card: https://raw.githack.com/Overworldai/Biome/feat/paper/research/WP1_5_Paper.pdf ("Waypoint-1.5: A Real-Time Video World Model for Consumer Hardware", 15 pp.)
- Waypoint-1 card: https://huggingface.co/Overworld/Waypoint-1-Small; blog: https://huggingface.co/blog/waypoint-1
- Inference library: https://github.com/Wayfarer-Labs/world_engine (cloned)
- Local config: `/home/hk28965/specwm/tools/waypoint/Waypoint-1.5-1B/config.yaml`, `modular_blocks.py`

### 2.2 Official evaluation: speed only
- The paper's Sec. 8 "Evaluation" says: "We therefore evaluate the system primarily on throughput, measured in latent frames per second." There is **no FVD, no VBench, no controllability metric, no human study and no test set**.
- The abstract distinguishes rendered FPS, latent FPS and control rate, and says "We evaluate interactivity through latency and throughput". No separate latency table was found in the text.
- `world_engine/examples/benchmark.py` is a pytest throughput benchmark with random controls.
- Waypoint-1 (Jan 2026; 2.3B; ~10k h of game footage with control inputs and text captions) also has no quality benchmark. The blog and card give only FPS: 30 FPS at 4 steps or 60 FPS at 2 steps on an RTX 5090 per the blog; 20–30 FPS on a 5090 per the card.

Paper Table 2 (latent FPS ↑; one latent = 4 rendered frames at 15 latent FPS ↔ 60 FPS video):

| GPU | 720P INT8 | 720P BF16 | 360P INT8 | 360P BF16 |
|---|---|---|---|---|
| RTX PRO 6000 Blackwell | **132.40** | **87.76** | **371.08** | **303.72** |
| RTX 5090 | <u>111.40</u> | 66.28 | <u>346.64</u> | <u>277.52</u> |
| RTX 4090 | 72.88 | <u>66.60</u> | 275.84 | 207.52 |
| RTX 3090 | 37.36 | 21.60 | 188.44 | 116.12 |
| RTX 3060 | 15.76 | 8.80 | 81.88 | 36.76 |

(Subset of the 10 GPUs. The paper treats ≥ 30 latent FPS as the real-time threshold.)

Model-card table (FPS ↑; units not stated):

| Config | Waypoint 1 | Waypoint 1.5 |
|---|---|---|
| Resolution | 360P | 720P |
| Context | 2 s | 10 s |
| 4-step BF16, RTX 5090 | 20 | <u>56</u> |
| 4-step w8a8, RTX 5090 | n/a | **72** |
| 4-step w8a8, RTX 3090 | n/a | 30 |

**Unclear:** the card says 56–72 FPS on a 5090, while the paper's Table 2 gives 66–111 *latent* FPS on the same GPU. If one latent decodes to 4 frames, the paper would imply 260–445 rendered FPS. The two documents use FPS in different senses. Our measurement on an RTX PRO 6000 (BF16, compiled, 1024×512) was **31.2 rendered fps** (bench-43 quiet pass). That is far below 4 × 87.76, so the paper's "latent FPS" probably excludes something (decode, Python loop, or per-call overhead), or uses a different definition. Report our own measured rendered FPS, and label any quoted paper number as latent FPS.

### 2.3 Model facts relevant to running it

| Item | Value | Source |
|---|---|---|
| Params | 1,281,959,992 (1.28B). The card says "1.2B" in one place and "2B" in the metadata. | paper Table 1 |
| Input resolution | 512×1024 ("720P" tier) / 256×512 ("360P" tier), i.e. **2:1** aspect | paper Table 1. Card example: `load_image(...starter_18.png).resize((1024, 512))` |
| AE | TAEHV 1.5, 4× temporal, 16× spatial (paper) / 8× (card), 32 channels | paper Sec. 2.1; card |
| Steps | 4 (σ = 1.0, 0.9, 0.75, 0.3, 0.0); CFG distilled in | paper Sec. 5.3 |
| Context | 16-frame local ring plus 128-frame dilated global (every 4th layer) | paper Sec. 5.1 / config |
| Control rate | 15 controller snapshots per second, one per latent = one pipeline call = 4 output frames | paper Sec. 7.2 |
| Keys | `button`: set of int keycodes (256-way multi-hot), Owl-Control keycodes. These look like Windows virtual-key codes: W = 87 (card: "W key (walk forward)"), A = 65, S = 83, D = 68, Space = 32. The full table (`owl-control/src/system/keycode.rs`) returned 404, so it was **not verified**. | card, world_engine README |
| Mouse | `mouse = (dx, dy)`: **raw mouse displacement accumulated since the previous snapshot** (paper Sec. 7.2), passed unnormalized to the model (`modular_blocks.py`). No unit, DPI or sensitivity is documented. Our optical-flow calibration: **≈ 118 units per degree per call** (`tools/waypoint/poses_to_actions.py`). | paper; local code |
| Scroll | ternary −1 / 0 / +1 | world_engine README |
| Text prompt | **Waypoint-1.5: no.** `prompt_conditioning: null` in config.yaml, and the paper Sec. 3.3 conditions on controls "rather than free-form text". **Waypoint-1-Small: yes** (UMT5; `engine.set_prompt`). | local config; paper; WP-1 card |
| Seed input | one image (or several seed frames / a video), encoded and cached with a σ = 0 pass | paper Sec. 5.2 |
| Training data | Owl-Control, proprietary, > 100k h, hundreds of games, 720p60 | paper Sec. 7.1 |
| Domain | Video games (first person). The card lists known failures: drift or collapse over long rollouts, and faces, text, HUD and entities. | card "Limitations" |

---

## 3. Third-party neutral benchmark: WBench (strongly relevant)

- Paper: arXiv 2605.25874 (25 May 2026), https://arxiv.org/abs/2605.25874. Repo: https://github.com/meituan-longcat/WBench (cloned; MIT). Data on HF: `meituan-longcat/WBench`; weights: `meituan-longcat/WBench-weights`.
- 289 cases and 1,058 turns. The **navigation split has 158 cases**. There are 5 dimensions and 19–22 automatic metrics.
- It has a **unified navigation protocol** across text, 6-DoF pose and keyboard/mouse:
  - `src/models/camera/poses.py`: per latent step, forward 0.08 units, yaw 3°, pitch 3°; OpenCV c2w.
  - `src/models/action/actions.py`: maps to MG3-style keyboard plus mouse ±0.1.
- Navigation metric `navigation_trajectory.py`:
  - MegaSAM poses are aligned to an adaptive ground truth: straight lines for translation, an orbit for rotation.
  - NavScore = ½(Accuracy + Consistency), where Accuracy = 1 − ½(nATE_t + nATE_r) and Consistency comes from repeated or symmetric action pairs.
  - The ATE is scale-normalized, so absolute speed does not matter.
- Requirements: CUDA GPU(s), SAM2, Depth-Anything-V3, MegaSAM precompute, and a **VLM judge API key (Doubao-Seed-2.0-lite via Volcengine ARK, `VLM_API_KEY`)** for some metrics.
- A ready **LingBot-World-v2 adapter** exists: `examples/lingbot/` (`wbench_to_json.py`, `generate_from_json_fast.py`, `run_wbench_fullseq_v2.py`; 14B causal-fast). It feeds one continuous c2w trajectory per case.

Navigation-split leaderboard rows. The source is `README.before_leaderboard_removal.md` (latest news entry 2026/08/16); the current README no longer shows the leaderboard. The MG2 HF card quotes "WBench Navi 68.5", which differs from 68.7.

| Model | Avg ↑ | Quality ↑ | Setting ↑ | Interaction (= Nav. trajectory) ↑ | Consistency ↑ | Physical ↑ |
|---|---|---|---|---|---|---|
| LingBot-World (fast v2) | **79.4** | **81.8** | <u>76.8</u> | <u>82.8</u> | <u>86.5</u> | <u>69.1</u> |
| LingBot-World (base-camera) | <u>78.5</u> | 78.9 | 72.6 | 80.1 | **89.9** | **71.2** |
| LingBot-World (fast) | 77.4 | <u>79.4</u> | **77.9** | 79.2 | 84.9 | 65.7 |
| YUME 1.5 | 73.3 | 77.6 | 72.4 | 71.4 | 80.1 | 65.2 |
| Matrix-Game 3 | 71.3 | 75.5 | 63.6 | **83.6** | 74.5 | 59.3 |
| Matrix-Game 2 | 68.7 | 73.8 | 67.1 | 80.3 | 65.1 | 57.2 |

Selected MG2 sub-metrics vs LingBot fast v2:
- aesthetic 54.0 vs 64.4
- imaging 60.3 vs 67.5
- HPSv3 41.0 vs 74.6
- scene adherence 49.4 vs 66.7

Waypoint is **not** on WBench (no Overworld entry in the README).

Takeaway: on a neutral interface, MG2 loses mostly on quality, consistency and physics, and only slightly on trajectory following (80.3 vs 82.8).

---

## 4. Converting their actions into our camera trajectory

Our model takes per-frame c2w poses (OpenCV: x right, y down, z forward) plus intrinsics. Their actions are WASD (+ jump/attack in MG1) plus mouse (pitch, yaw) per frame.

### 4.1 Integration (FPS-controller kinematics)

For frame k = 0…N−1 at our output rate f (16 fps), keep yaw ψ_k and pitch θ_k in degrees and position p_k:

```
ψ_{k+1} = ψ_k + s_yaw  · m_yaw[k]                 # mouse yaw   → degrees/frame
θ_{k+1} = clip(θ_k + s_pitch · m_pitch[k], −80°, +80°)
R_k     = R_y(ψ_k) · R_x(−θ_k)                     # yaw about world up, then pitch; OpenCV y-down ⇒ "look up" = negative x-rotation
v_k     = speed · [ (D−A), 0, (W−S) ]             # local, heading frame only (no pitch: walking stays horizontal)
           (normalize diagonals by 1/√2 if their engine does; MG1/2 do not document this; WBench uses ~1.41× for W+D)
p_{k+1} = p_k + R_y(ψ_k) · v_k
jump    : optional vertical offset −h·4t(1−t) over ~0.6 s (y-down ⇒ negative = up); h ≈ 1.25 blocks
attack  : not representable as camera motion → mark n/a
c2w_k   = [R_k | p_k]
```

Sign conventions to respect:
- **MG1/MG2 model input:** `mouse = [pitch, yaw]`; camera_up = [+v, 0], camera_l = [0, −v].
- **VPT IDM labels** (`Bench76` jsonl): `camera = [dy, dx]` with **negative dy = up**; camera_up = [−5, 0], camera_l = [0, −5].
- **WBench:** "right" = +yaw about local Y; "up" = +pitch about local X.

Write a sign unit test before running anything. Render one "camera_up" and one "camera_l" clip with our model and check the optical-flow direction.

### 4.2 Magnitudes

Speed matching is the fairness-critical step. The IDM accuracy only scores direction, but too small a rotation falls into the IDM's "none" class, and too large a rotation hurts quality. Options, in order of preference:
1. **Speed-match to the opponent's own output.** Run MG2 on the 32 images with each Bench action, estimate its realized per-frame yaw/pitch and translation with DROID-SLAM or MegaSAM, and use the **median realized rate** as s_yaw, s_pitch and speed for our trajectories. Both models then make the same apparent motion. This is the same calibration we did for Waypoint, in reverse.
2. Use our existing calibrations: MG2 0.1 mouse ≈ 1.5°/frame (0.067 per degree, `tools/mg2/poses_to_cond.py`); Waypoint ≈ 118 units per degree per call (`tools/waypoint/poses_to_actions.py`).
3. WBench reference: 3° yaw/pitch and 0.08 units per latent step at 6 latents/s.

Translation scale is ambiguous for a monocular model. Choose `speed` so that our model's realized forward motion (from SLAM depth-normalized flow) matches MG2's. Report the chosen constants.

Also:
- **Frame rate:** MG1 benchmark clips are 65 frames at 16 fps; MG2 uses 597 frames; ours are 16 fps. Generate at least 65 frames for Bench76 (one chunk group: our 153-frame clips are fine, then truncate to 65). Our model's long-horizon drift is a known weakness. Treat 597 frames (~37 s at 16 fps) as a stress test, not the headline.
- **Intrinsics:** Minecraft's default vertical FOV is 70°. Use fy = (H/2)/tan(35°) for our trajectory, so that a yaw of x° moves pixels the way the game would.

### 4.3 Which of their metrics stay meaningful for a camera-pose model

| Metric | Meaningful for us? | Note |
|---|---|---|
| Image / Aesthetic / Temporal / Motion smoothness | Yes, as is | Per-frame and temporal no-reference metrics. They are domain-sensitive: Minecraft pixels score low on MUSIQ for everyone. Temporal and Motion reward static video (Oasis caveat). Pair them with a dynamic-degree check. |
| Object consistency (DROID-SLAM reprojection) | Yes | Interface-agnostic. Fixed intrinsics (500, 256) are assumed. |
| Scenario consistency (mirror) | Yes, **and favorable to a pose model** | We can generate exactly time-symmetric trajectories. The 8 mirror actions are **not in the repo**; reconstruct them from MG1 Sec. 5.4 (up→down, down→up, left→right, right→left and 4 diagonals, n out + n back + 1 center frame). It rewards freezing, so report it with a motion check. |
| Mouse acc. (VPT IDM) | Partly | The IDM infers camera rotation from pixels, so it works on any Minecraft-looking video. It is invalid if our model "de-Minecrafts" the scene (photo-trained). Report it on the Minecraft split only, as MG2 does. |
| Keyboard acc. (VPT IDM) | Partly | forward/back and left/right yes, since translation is in our trajectory. **Jump** only if we add a vertical bump. **Attack** n/a. MG2's universal checkpoint also has no jump or attack keys, so MG2's 0.91 was presumably computed on a reduced key set or with those always 0. **Unclear; flag.** Report the 2-task (fwd/back, left/right) keyboard precision for every model to compare like with like. |
| **Pose-based action accuracy (new, recommended)** | Yes, for all models | Run DROID-SLAM (already installed for Object consistency) on every video. Classify per-frame Δyaw/Δpitch into the same 9 classes and translation direction into {fwd, back, none} × {left, right, none}. Then compute the same precision as the IDM metric, and also rotation error (°/frame) and normalized ATE vs the commanded trajectory. Domain-agnostic, so it also works on the wild split. |
| WBench NavScore | Yes | Already designed for this cross-interface comparison. |

---

## 5. Recommended protocol (concrete)

**Arms:**
- ours: the meeting_1012 modes (teacher-parity / quality / realtime), plus the 14B teacher and the 1.3B student;
- MG2: official checkpoint, ideally **with** torch.compile as in the official repo, since its 25 fps claim depends on it;
- Waypoint-1.5-1B: 1024×512, compiled.

All arms run on one GPU type, and the GPU is reported.

**A. GameWorld-Minecraft (MG2's turf).**
1. Images: the 32 public PNGs. MG2 crops them to 352×640; we feed the same image at our resolution (center-crop to our aspect); Waypoint gets a resize to 1024×512.
2. Actions: (i) the 76 Bench76 actions × 65 frames, giving 2,432 videos per arm (or a stratified subset, e.g. 32 images × 20 actions, if compute-bound); (ii) 8 reconstructed mirror actions → 256 videos; (iii) optionally, 32 × one 597-frame composite sequence generated by `Bench_actions_universal(597)` with a fixed seed, saved as JSON so every arm gets the identical sequence.
3. Conversion: Section 4 kinematics, with constants speed-matched to MG2's realized motion.
4. Scoring: GameWorldScore code unchanged for the 8 metrics, plus the pose-based accuracy and the 2-task keyboard precision.
5. Report MG2's published numbers next to our re-run of MG2 as a sanity check: our MG2 re-run should land near Image 0.61 and Mouse 0.95.

**B. Wild split.** The 16 MG2 wild images are not public. Use the 17 `Matrix-Game-2/demo_images/universal/` images as the MG2-wild proxy (their own demo images), plus our bench-43 photos for our home turf. Use the same 597-frame sequence. Metrics: everything except the IDM accuracy, plus the pose-based accuracy.

**C. WBench navigation (neutral, published).** Run our modes through `examples/lingbot/` (it already targets LingBot-World-v2 causal-fast; swap in our speculative pipeline as the `generate()` backend). This yields directly comparable numbers vs MG2 (68.7), MG3 (71.3) and LingBot fast v2 (79.4). Needs the Volcengine ARK `VLM_API_KEY` for the VLM sub-metrics (**ask the user**). Without the key, report the non-VLM metrics only. Adding Waypoint requires writing an `ActionConditionedModel` adapter with our 118 units/deg calibration.

**D. Waypoint's turf (speed and interactivity).** On one GPU, report for every arm:
- rendered FPS (decoded frames/s, decode included);
- "latent FPS" as Waypoint defines it (latents/s, denoiser only);
- control rate (how often a new action can take effect: WP 15 Hz = every 4 frames; MG2 every 3 latents = 12 frames; ours every chunk);
- action-to-photon latency (time from an action change to the first frame reflecting it);
- time to first frame (TTFF).

This matches the latency metrics from the 10/5 meeting. For quality on Waypoint's domain, use the HF Space starter images (`huggingface.co/spaces/Overworld/waypoint-1-small/.../starter_*.png`) at 2:1, with protocol A/B metrics and pose-based accuracy.

---

## 6. Open issues and things that are unclear

1. **The MG2 597-frame sequence is not published.** It is only inferred to come from `Bench_actions_universal` (random 12-frame blocks; seed-dependent). The mirror sequences used for Scenario consistency are also not in the repo.
2. **MG2 keyboard accuracy with a 4-key checkpoint:** how jump and attack were handled is unclear.
3. **IDM limitations:** the VPT IDM is Minecraft-trained at 128×128 and assumes Minecraft dynamics (20 Hz). Its accuracy on photo-like or non-Minecraft output is meaningless, and its frame-rate sensitivity is unknown. That is why the pose-based accuracy is recommended.
4. **Waypoint units:** mouse units are undocumented (raw device counts). The keycode table could not be fetched (404). FPS definitions conflict between the card and the paper. The 16× vs 8× spatial compression differs between paper and card.
5. **Domain:** all three opponents are game-trained; ours is photo/real-video-trained. Minecraft inputs are their home turf by design. Expect our IDM scores to suffer if we render Minecraft images photographically. Show qualitative frames.
6. **WBench:** the leaderboard is taken from a pre-removal README snapshot (news up to 2026/08/16). The MG2 card says 68.5 vs the README's 68.7. The VLM judge needs a paid API key.
7. Paper versions: MG2 numbers are from arXiv v5 (29 Sep 2026). They match the project page and HF card, so they are probably unchanged since v1, but older versions were not checked.

## 7. File and URL index
- GameWorldScore code: `github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-1/GameWorldScore` (`evaluate.py`, `scripts/evaluate_Matrix*.sh`, `GameWorld/{action_control,object_consistency,scenario_consistency,temporal_consistency,imaging_quality,aesthetic_quality,motion_smoothness}.py`, `GameWorld/third_party/IDM/{IDM_bench.py,Bench76/*.jsonl}`, `asset/init_image/*`)
- MG2 inference and actions: `Matrix-Game-2/inference.py`, `inference_streaming.py`, `utils/conditions.py`, `configs/inference_yaml/inference_{universal,templerun}.yaml`
- MG1 bench generator: `Matrix-Game-1/inference_bench.py`, `condtions.py` (`Bench_actions_76`, CAM_VALUE 0.05, 65 frames)
- VPT IDM weights: `openaipublic.blob.core.windows.net/minecraft-rl/idm/4x_idm.{model,weights}`
- Waypoint: HF card, paper PDF (above), `github.com/Wayfarer-Labs/world_engine` (`README.md` CtrlInput, `examples/benchmark.py`)
- WBench: `github.com/meituan-longcat/WBench` (`src/models/camera/poses.py`, `src/models/action/actions.py`, `src/metrics/interaction/navigation_trajectory.py`, `examples/lingbot/`)
- Ours (existing converters): `/home/hk28965/specwm/tools/mg2/poses_to_cond.py`, `/home/hk28965/specwm/tools/waypoint/poses_to_actions.py` (camera → actions; the inverse direction is Section 4 here)
