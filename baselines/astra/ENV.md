# Astra baseline — environment, checkpoints, run command (PROTOCOL §4)

Astra: General Interactive World Model with Autoregressive Denoising (ICLR 2026; arXiv 2512.08931). Image + text prompt + camera trajectory
→ 832x480 video, Wan2.1-T2V-1.3B based autoregressive DiT with FramePack history, noise-augmented memory, action adapter + MoE. Official
claim: inference on a single 24 GB GPU. This is the only external baseline that takes a camera PATH (per-latent relative pose), so the
protocol programs are passed exactly (no key/mouse quantisation).

## Code
* repo `https://github.com/EternalEvan/Astra`, clone `/home/hk28965/specwm/tools/astra/Astra`, pinned commit `3fb82e1939bfcf87d6e27a7da57d8b55ff9cb393`
  ("fix", 2026-02-02). Nothing in the clone is modified; the only additions are symlinks `models/Wan-AI/Wan2.1-T2V-1.3B` and
  `models/Astra/checkpoints/diffusion_pytorch_model.ckpt` (both git-ignored).
* inference logic = `scripts/infer_demo.py::inference_moe_framepack_sliding_window` (README "Step 3", `infer_demo.sh`). Our driver
  `baselines/astra/astra_gen.py` imports that module and re-runs the same steps (model assembly, FramePack sliding window, 50-step FlowMatch
  shift 5, no CFG, MoE experts 3 / top_k 1, modality `sekai`, tiled VAE, imageio 20 fps writer) and adds only: fixed seed, the protocol camera
  program, per-block timing, TTFF decode, peak VRAM, timing.json. The README's python `infer_demo.py --cam_type N` call is reproduced by
  `run.sh IMAGE cam:N SEED OUT`.

## Checkpoints (all sha256-verified against the HF LFS metadata)
| file | source | size | sha256 (first 8) | where |
|---|---|---|---|---|
| Astra DiT `diffusion_pytorch_model.ckpt` | HF `EvanEternal/Astra` models/Astra/checkpoints/ | 3.15 GB | 5ca9c5e0 | `/home/hk28965/specwm/hf/Astra/models/Astra/checkpoints/` (downloaded 10/16) |
| Wan2.1 DiT `diffusion_pytorch_model.safetensors` | HF `Wan-AI/Wan2.1-T2V-1.3B` | 5.68 GB | 96b6b242 | `/home/hk28965/specwm/hf/Wan2.1-T2V-1.3B/` (downloaded 10/16; `~/.cache/huggingface/hub/models--Wan-AI--Wan2.1-T2V-1.3B` was an EMPTY stub: refs only) |
| `config.json` | same | 249 B | – | same |
| T5 `models_t5_umt5-xxl-enc-bf16.pth` | same | 11.36 GB | 7cace0da | symlink → `lingbot-world-v2/lingbot-world-v2-14b-causal-fast/` (byte-identical, hash checked; no download) |
| `Wan2.1_VAE.pth` | same | 0.51 GB | 38071ab5 | symlink → same LingBot folder (hash checked) |
| `google/umt5-xxl/` tokenizer | same | 21 MB | spiece e3909a67, tokenizer.json 6e197b4d | symlink → same LingBot folder (hash checked) |

New downloads: **8.83 GB** (Astra ckpt 3.15 + Wan DiT 5.68). The Astra checkpoint loads with `load_state_dict(strict=False)` as in the
official script (the assembled DiT has the added cam_encoder/projector/FramePack/MoE modules).

## Environment (`/home/hk28965/specwm/tools/astra/venv`, uv 0.12.15)
```
uv venv --python 3.11 /home/hk28965/specwm/tools/astra/venv          # CPython 3.11.16 (the README gives no python version; 3.11 was already installed for uv)
uv pip install --python venv/bin/python --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0 torchvision==0.23.0
uv pip install --python venv/bin/python -r Astra/requirements.txt scipy "numpy<2.3" pandas
```
Resolved: torch 2.8.0+cu128, torchvision 0.23.0+cu128, triton 3.4.0, transformers 4.46.2 (their pin), tokenizers 0.20.3, controlnet-aux 0.0.7
(their pin), cupy-cuda12x 14.2.0, modelscope 1.40.2, imageio 2.38.1 + imageio-ffmpeg 0.6.0, scipy 1.17.1 (needed by infer_demo.py, missing from
their requirements), numpy 2.2.6, pillow 12.3.0, einops 0.8.2, safetensors 0.8.0, sentencepiece 0.2.2, protobuf 7.36.2, ftfy 6.3.1, pandas 3.0.6.
No pin conflicts. `pip install -e .` (their setup.py imports the deprecated `pkg_resources`) was NOT used: the driver puts the clone on
`sys.path` exactly as `scripts/infer_demo.py` does itself (`sys.path.append(ROOT_DIR)`). Rust/cargo (README step 1, for DiffSynth extensions)
is not needed for inference and was not installed. venv size 7.3 GB (uv cache hard-links).

* precision: bf16 everywhere (`ModelManager(torch_dtype=torch.bfloat16)`; DiT, T5, VAE).
* attention: `torch.nn.functional.scaled_dot_product_attention` — `diffsynth/models/wan_video_dit_moe.py::flash_attention` prefers
  flash_attn 3 → flash_attn 2 → SageAttention → SDPA; none of the three is listed in their requirements, so the official stack = SDPA
  (PyTorch's own flash/cuDNN SDPA kernels on Blackwell). The cp312 flash-attn wheel in tools/abot does not fit this 3.11 venv.
* text encoder: umT5-xxl bf16 RESIDENT on the GPU (official `pipe.to(device)` moves text encoder, DiT and VAE; ~11.4 GB of the footprint).
  `T5_OFFLOAD=1 run.sh …` moves it to CPU after prompt encoding (reported separately if ever used).
* VAE: Wan2.1 VAE, tiled encode of the 10x-repeated start image (tile 34/18) and ONE tiled decode of the whole clip at the end (official:
  the script does not stream decode); the TTFF number below decodes the first block separately (not counted in e2e).
* compile: none (official script has none).  CFG: none (official defaults).  Steps: 50 FlowMatch (shift 5, sigma_min 0).
* chunking: 8 latents (= 32 frames = 1.6 s at 20 fps) per generation block (infer_demo.sh `--frames_per_generation 8`); FramePack context
  per block = 1 start + 16 (4x-compressed) + 2 (2x) + 1 (1x) history latents; history capped at `max_history_frames` 100 latents (never hit).

## Official stack, frozen (what every Astra number in this benchmark is produced with)
* **Attention = PyTorch SDPA.** `Astra/requirements.txt` lists neither flash-attn nor SageAttention, and the README installs nothing of the kind;
  `diffsynth/models/wan_video_dit_moe.py::flash_attention` therefore falls through to `torch.nn.functional.scaled_dot_product_attention`
  (PyTorch 2.8 picks its own flash / cuDNN kernel on Blackwell). We did not add flash-attn (it would be a non-official speed-up); the ABot cp312
  wheel would not fit this 3.11 venv anyway.
* **Sampling = 50 denoising steps per block**, hard-coded in the official script (`infer_demo.py` L1140 `pipe.scheduler.set_timesteps(50)`,
  no CLI flag; FlowMatch scheduler of `WanVideoAstraPipeline` = `FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)`), **no CFG** (camera CFG off, text guidance 1.0 = the README/`infer_demo.sh` defaults), MoE
  3 experts / top-1, modality `sekai`, **8 latents (32 frames) per block**, FramePack context 1 + 16 + 2 + 1 history latents, bf16, tiled Wan2.1 VAE.
  50 steps x 7 blocks = 350 DiT forwards per 10 s clip (0.50 s each at steady state) is the reason Astra runs at ~1.3 fps generation.

## Run
```
baselines/astra/run.sh IMAGE PROGRAM_ID SEED OUTDIR            # -> OUTDIR/video.mp4 (832x480, 20 fps, 201 frames = 10.05 s), timing.json, log.txt, camera.npz, sheet.jpg
flock /tmp/gpu6.lock env CUDA_VISIBLE_DEVICES=6 /home/hk28965/specwm/baselines/astra/run.sh /home/hk28965/specwm/baselines/scenes/astra/garden_1.png P2 1234 /home/hk28965/specwm/baselines/runs/astra/astra__garden_1__P2
```
### Batch mode (10/17): one model load, many clips
```
python3 baselines/astra/make_jobs.py astra                     # -> baselines/runs/astra/jobs_astra.json (3 scenes x P1..P5, prompt = scene .txt)
python3 baselines/astra/make_jobs.py neutral --no-caption --suffix _nocap   # empty-prompt arm -> runs/astra_nocap/jobs_neutral_nocap.json
flock /tmp/gpu6.lock env CUDA_VISIBLE_DEVICES=6 baselines/astra/run.sh batch baselines/runs/astra/jobs_astra.json [BATCH_LOG]
```
`run.sh batch JOBS.json` = `astra_gen.py --jobs`: the pipeline (DiT + T5 + VAE, ~11 s, 13.8 GiB) is loaded ONCE and the discarded warm-up block runs
once; every job then does exactly what a single run does (image VAE encode, prompt T5 encode, 7 blocks, one tiled decode, timing.json with
the same fields + `batch: {jobs_file, index, n_jobs}`, log.txt, camera.npz, sheet.jpg). Jobs whose video.mp4 + timing.json exist are skipped
(`REDO=1` regenerates), a failing job writes OUTDIR/ERROR.txt and the batch continues; exit status 1 if any job failed. Per clip ~3.3 min
(vs ~3.5 min single, the saving is the load), so 75 clips ≈ 4.1 h; keep each lock hold ≤ the orchestrator's limit by splitting the jobs file
(`make_jobs.py --scenes a,b --programs P1,P2`). JOBS.json entries: `{"image", "program", "seed", "out", "prompt"}` (prompt omitted = the image's .txt).
PROGRAM_ID = `P1..P5` (protocol program → `program_to_astra.py` → Astra's real-pose input) or `cam:N` (Astra's own preset cam_type 1..7,
their README demo). Prompt = `$PROMPT` if set, else the image's `.txt` sibling (scenes/astra/*.txt; scenes/neutral captions), else empty.
Env knobs: `LATENTS` (56), `FRAMES` (201), `BLOCK` (8), `STEPS` (50), `WARM` (1 = one discarded 8-latent block before timing), `T5_OFFLOAD` (0).
10 s clip = 1 condition latent + 56 generated latents in 7 FULL blocks of 8 (225 frames), cut to the first 201 frames (10.05 s). The official loop
allows a partial last block, but 50 latents = 6x8 + 2 collapsed in the 2-latent block (smoke 2 below), so run.sh always generates whole blocks.
timing.json therefore reports `frames` = 201 kept and `frames_generated` = 225 (gen_s / e2e_s cover all 7 blocks; the steady fps is per block).
The start image is STRETCHED to 832x480 by Astra's own loader (no crop; the example images are 3:2).

### Camera input (program_to_astra.py)
c2w from `baselines/convert.py::program_to_c2w` at 20 fps (the shared definition; convert.py fixes +yaw = pan RIGHT), then per latent j:
`T_rel = inv(c2w[4j]) @ c2w[4j+4]`, `M_j = S·T_rel[:3,:]·S`, `S = diag(1,-1,-1)` — so the protocol's right pan becomes Astra's "Rotate Right"
matrix (M[0,2] < 0, their cam_type 3) and forward becomes M[2,3] < 0 (their cam_type 1). Rotation is passed exactly (P2: 3.6° per latent, 2.1x
Astra's preset 1.72°); translation 1 u/s = Astra's preset forward speed 0.03 per latent = u_scale 0.15 (calib_astra.json; VERIFIED 10/17 with
DA3: ≈ 1.1 m/s walking speed, 1 unit ≈ 7.3 m; see Calibration below).
Closest presets (not used): P1→1, P2→3, P3→5, P4→6 (opposite order, one reversal), P5→none.
Embedding length (10/17 fix): one row per generated latent + 1 for the condition latent = 57 rows for LATENTS=56, so `astra_gen.py` builds the
program for all 225 generated frames (past 10 s the program continues its last segment; `camera.npz` keeps the c2w of the 201 kept frames as
the RPE reference). Before the fix the 201-frame program gave 51 rows and the official code zero-padded the rest on the CPU
(`prepare_framepack_sliding_window_with_camera_moe` L934 `torch.cat` with a cuda tensor → crash at block 7; a zero 3x4 is not "no motion").
NOTE: `baselines/convert.py` CALIB['astra'] was moved to 20 fps / 0.15 units per u by its owner on 10/17 (its `embedding_rows 51` and
`translation_status UNVERIFIED` strings are stale, see calib_astra.json `*_da3` keys, which survive a `convert.py build` under the
calib.json `_merge` policy). run.sh never reads calib.json or programs/*.npz: program_to_astra.py builds the 20 fps embedding itself.

## Smoke (GPU 6 under `flock /tmp/gpu6.lock`, 2026-10-16; RTX PRO 6000 Blackwell Server Edition, driver 595.99.02)

### 1. Their README example — `run.sh scenes/astra/garden_1.png cam:4 1234 baselines/astra/smoke/example_cam4` (LATENTS=24, official prompt)
= `infer_demo.sh` (cam_type 4 Forward + Rotate Left, 1 condition + 24 latents = 97 frames, block 8, 50 steps, no CFG) plus seed 1234 and one
warm-up block. GPU 6 was EMPTY during this run (nvidia-smi 0 MiB / 0 % before start), so these numbers are clean. Wall time under the lock
~2 min (22:33:20 → 22:35:18 incl. 10.6 s model load).
| item | value |
|---|---|
| load (safetensors + T5 + VAE → GPU) | 10.6 s; 13.76 GiB allocated after load (T5 resident) |
| input | image VAE encode 0.84 s, prompt T5 encode 0.12 s (870 chars) |
| warm-up block (discarded) | 22.0 s |
| blocks (8 latents = 32 frames each) | 21.85 s, 25.26 s, 25.22 s → 0.437 / 0.505 / 0.504 s per DiT step (context grows to the full FramePack window after block 1) |
| gen_s (3 blocks, 96 new frames) | 72.3 s → 1.33 fps generation; **steady_fps_gen 1.27** (blocks 2-3) |
| decode (one tiled decode, 97 frames) | 8.83 s |
| e2e_s | 82.1 s → 1.18 fps; **steady_fps_e2e 1.14** (block + its share of the decode) |
| TTFF (input → first 33 decoded frames) | **26.0 s** (0.84 + 0.12 + 21.85 + 3.18 s decode of the first 9 latents; that decode is extra work not in e2e_s) |
| peak VRAM | **14.85 GiB allocated / 16.13 GiB reserved** (nvidia-smi 17.2 GB at the end) — fits their "24 GB" claim with the T5 resident |
| output | video.mp4 h264 832x480 20 fps 97 frames (4.85 s) |
Frames (sheet.jpg, looked at): sharp, stable, prompt-consistent Amsterdam street; forward motion down the brick square with a LEFT pan — the
lamp post drifts from centre-right to the right edge and leaves the frame by ~1.6 s, the left-hand townhouses come into view, the potted
plants pass on the right; sun flare on the left persists; no collapse or flicker over 97 frames. So Astra's "left" preset moves scene
content to the RIGHT of the frame (geometric left pan), confirming the sign reading in program_to_astra.py.

### 2. Protocol P2 on their scene — `run.sh scenes/astra/garden_1.png P2 1234 …` with LATENTS=50 (6x8 + 2 latents = 201 frames), official prompt
Kept at `baselines/astra/smoke/garden_1_P2_lat50/` (video.mp4, timing.json, sheet.jpg, frames_0_36.png, frames_176_200.png, log.txt);
NOT a frozen benchmark clip because of the partial-block collapse described below. GPU 6 was again empty (0 MiB / 0 % before). Wall time under
the lock 3 min 25 s (23:22:18 → 23:25:43).
| item | value |
|---|---|
| load / input | 10.8 s; image 0.84 s, prompt 0.14 s; 13.76 GiB after load |
| warm-up | 22.0 s |
| blocks | 21.84, 25.16, 25.11, 25.12, 25.11, 25.11 s (8 latents each) + 8.21 s (2 latents) → steady **0.502 s per step** |
| gen_s (50 latents, 200 new frames) | 155.6 s → 1.29 fps; **steady_fps_gen 1.27** |
| decode (201 frames, one tiled decode) | 18.4 s |
| e2e_s | 175.0 s → 1.15 fps; **steady_fps_e2e 1.14** |
| TTFF | **26.0 s** (first 33 decoded frames; extra decode 3.14 s) |
| peak VRAM | **14.86 GiB allocated / 16.13 GiB reserved**; nvidia-smi 17.2 GB |
| output | h264 832x480 20 fps 201 frames |
Frames (looked at, sheet.jpg + frames_0_36.png + frames_176_200.png):
* **Direction correct.** Scene content moves steadily to the LEFT of the frame from the first generated frames (the lamp post goes from
  centre-right at f0 to the centre by f36 and the left townhouse leaves the frame) = the camera pans RIGHT, exactly what the protocol's
  +18 deg/s (convert.py: +yaw = pan right) commands, and the opposite of Astra's own cam_type 4 clip above. The sign mapping in
  program_to_astra.py is therefore verified for yaw; the pan rate (should be 180 deg over 10 s) is NOT measured (needs DA3 pose recovery,
  scripts/rpe_da3.py, PROTOCOL rule 3).
* **Astra drifts on this program.** New content entering from the right is hallucinated from ~1.5 s (a large odd townhouse by 2-3 s), by
  4.5 s the street has become a different building + lawn, by 5.5-9 s a park with bare trees; textures get smeary (per-frame |diff| rises
  from ~11 to ~15 grey levels per block). The clip never looks back at the start view. This is Astra's genuine behaviour at 18 deg/s (2.1x its
  preset speed) over 10 s and is what the judge/VBench/RPE metrics must score; a slower P2' (rule 2 of PROTOCOL §2) would also be fair to test.
* **Partial-block collapse.** Frames 193-200 (= exactly the final 2-latent block, latents 49-50) are saturated noise/garbage while frame 192 is
  still the coherent park (frames_176_200.png: top row f176/184/190/192 fine, bottom row f193/195/198/200 destroyed). The official loop permits
  `min(frames_per_generation, remaining)`, but the model clearly cannot generate a 2-latent block. Hence run.sh now generates 56 latents
  (7 full blocks) and cuts the video to 201 frames; that configuration has not been run end-to-end as a single clip yet (every component of
  it — 7 consecutive full blocks, trimming — was exercised in these two runs).

### 3. Frozen configuration, one batch (10/17, GPU 5 under `flock /tmp/gpu5.lock`): `run.sh batch calib/jobs_calib_P1P2.json` = P1 (u_scale 0.15) + P2, garden_1, official prompt, seed 1234, LATENTS=56 (default), FRAMES=201
One model load (11.3 s) for both clips, one warm-up block (22.0 s); batch wall time 424 s = 7.1 min for 2 clips (≈ 3.3 min per clip + load).
GPU 5 had 14.9 GB of another job's memory resident but 0 % utilisation during the batch (gpu_smi_before), block times equal the clean 10/16 numbers.
| item | P1 (`astra/calib/garden_1_P1_u0.15`) | P2 (`runs/astra/astra__garden_1__P2`, frozen clip) |
|---|---|---|
| blocks (7 x 8 latents) | 21.83, 25.36, 25.32, 25.27, 25.26, 25.26, 25.21 s → 0.506 s/step | 21.86, 25.15, 25.15, 25.16, 25.15, 25.15, 25.17 s → 0.503 s/step |
| gen_s (56 latents = 224 new frames) | 173.5 s → **steady_fps_gen 1.27** | 172.8 s → **1.27** |
| decode (one tiled decode, 225 frames) | 21.1 s | 21.1 s |
| e2e_s (encode + 7 blocks + decode, 225 frames generated, 201 kept) | 195.5 s → **steady_fps_e2e 1.12** | 194.6 s → **1.12** |
| TTFF (input → first 33 decoded frames) | 26.0 s | 25.7 s |
| peak VRAM | **14.86 GiB alloc / 16.13 GiB reserved** | same (one process) |
| output | h264 832x480 20 fps 201 frames (ffprobe) | same |
Frames (looked at: `calib/strips/P1_u0.15_{sec,end,q}.png`, `P2_lat56_{sec,end,q}.png`, made by `frame_strip.sh`):
* **P2, trimmed 201-frame video is CLEAN.** Frames 160–200 every 4th (`_end.png`): one continuous park/garden view, the final block's frames
  193–200 look exactly like 192 (per-frame |diff| 20.7 vs clip median 20.1, 0.8 % saturated pixels — the 10/16 2-latent block had destroyed
  frames there). Content = the same right pan and the same drift as the 10/16 lat50 clip (identical seed → identical first 6 blocks): the
  Amsterdam square until ~1.5 s, a hallucinated large townhouse 2–3 s, a different building + lawn 4–5 s, a park with bare trees 5–9 s, a
  hedge at 9–10 s; the camera pans RIGHT throughout (content flows left). Pan rate: DA3 below.
* **P1 (u_scale 0.15 = Astra's preset forward speed): slow, coherent forward motion for 8.45 s, then a scene cut.** Frames 0–168: the lamp post
  grows and slides from centre-right to the right edge, the foreground plants pass on the right, the townhouses approach slowly (per-frame
  |diff| 7–11 grey levels, blocks 1–5). At frame 169→170 (8.45 s, INSIDE block 6 = latents 41–48, not at a block boundary; |diff| 50) the
  image jumps to a dense hedge/bush scene that stays until the end (blocks 6–7 |diff| 21–22). This is Astra's own failure on a 10 s forward
  program (the 4.85 s README example never gets there); the DA3 forward-speed calibration therefore uses the first 7.5 s (segments 0–2).

### 4. P3 verification clip (10/17): `run.sh batch calib/jobs_calib_P3.json` → `runs/astra/astra__garden_1__P3` (u_scale 0.15, LATENTS=56, official prompt, seed 1234)
223 s batch (load 11.2 s + 1 clip); blocks 21.9 / 25.1 x 6 s, gen 172.6 s, e2e 194.7 s, TTFF 25.9 s, 1.27 / 1.12 fps, 14.86 / 16.13 GiB — identical
to smoke 3. Frames (`calib/strips/P3_u0.15_{sec,q}.png`): the camera pans right from the square; the planters on the right pass in the
first ~2 s, then a hallucinated half-timbered house (2–5 s), a park lawn with a statue (5–8 s), bushes and a red flower bed (8–10 s); no
scene cut, coherent to the end. DA3 (`calib/da3_calib.json: astra_P3_u0.15`): yaw 11.3 / 13.0 / 22.9 / 24.3 deg/s per segment (commanded 12:
exact for the first 5 s, then 1.9x once the content is hallucinated, total 178.7 deg vs 120 commanded), forward **≈ 0** (−0.06 / −0.14 /
+0.21 / 0.00 m/s; flow divergence −0.02 = no zoom), lateral +0.2–0.4 m/s. So at 0.03 units/latent forward + 2.4 deg/latent yaw Astra
executes the rotation and drops the translation — its P3 forward/turn ratio is ~0 m/deg, not the 0.092 predicted from P1 and P2
separately. Sweep (`calib/run_chain_P3_sweep.sh`, `U_SCALE=0.30` and `0.45`, 4 min per clip, DA3 `astra_P3_u0.30/u0.45`): forward 0.11 and
0.17 m/s, yaw 18.1 and 16.3 deg/s, both clips coherent (the 0.45 one pans through a hallucinated street with cherry blossoms and a car),
no visible dolly in either — see the Calibration table and the u_scale decision above.

### Sizes written to disk (10/16)
Astra ckpt 3.0 GiB + Wan2.1 DiT 5.3 GiB (new downloads, 8.83 GB); venv 7.3 GB (uv hard-links from ~/.cache/uv); clone 0.1 GB; symlinked T5/VAE/
tokenizer 11.9 GB (not duplicated). /home had 53 GB free at start, 22 GB at the end (other agents' downloads ran in parallel).

## Calibration (10/17, PROTOCOL rule 3: magnitudes measured on Astra's OWN outputs) — `calib_astra.json`, merged into `baselines/calib.json['astra']` by `calib_merge.py`
Tool: `calib_da3.py` = Depth Anything 3 pose recovery (DA3NESTED-GIANT-LARGE-1.1, the nested model is METRIC: its metric branch scales the
recovered translations to ~metres per 16-frame window, `is_metric 1`), per-step camera-frame yaw/forward with the converters' formulas,
cross-checked with DIS optical flow (median horizontal flow for the pan, radial divergence for forward). Results `calib/da3_calib.json`;
reference clips of OURS (14B teacher, same scene stretched to 832x480, area 399360) by `calib/run_chain_all.sh` → `calib/ours_teacher/`.
GPU 5 holds: Astra batch 7.1 min, ours 6.5 min, DA3 1.5 min (peak 9.4 GiB).

| clip | commanded | realised (DA3 metric) | flow cross-check | reading |
|---|---|---|---|---|
| Astra P2 (frozen clip) | yaw 18 deg/s, 180 deg | **27.3 deg/s, 272.6 deg** (per 2.5 s segment 18.2 / 35.8 / 27.5 / 27.5); pitch −14.8 deg; walks 0.38 m/s while panning | median −15.1 px/frame at DA3's fx ≈ 650 px → 26.6 deg/s (16.7 / 31.5 / 27.0 / 29.0) | radians-per-latent mapping EXACT in the first 2.5 s (while the view is still the real square); the 1.5x over-rotation starts with the hallucinated content (~2 s) = drift behaviour at 2.1x its preset rate, scored by RPE-Rot, NOT re-calibrated (rotation_scale 1.0). Pan direction right = correct. |
| Astra P1 (u_scale 0.15) | fwd 0.03 units/latent = 0.15 units/s | **1.10 m/s** (segments 1.04 / 1.15 / 1.13; windows 0.94–1.31) = 0.22 m/latent, lateral −0.47 m/s, yaw 1.4 deg/s; segment 3 after the scene cut excluded | divergence +0.63 px/frame (forward) | Astra's unit is not a metre: **1 Astra unit ≈ 7.3 m**; its preset forward = a walking speed (~1.1 m/s), as in its Sekai/SpatialVID training videos |
| ours P3 (teacher) | fwd 1 u/s, yaw 12 deg/s (0.0833 u/deg) | fwd 0.32 m/s (0.45 / 0.32 / 0.26 / 0.25), yaw 4.0 deg/s (6.0 / 4.0 / 3.5 / 2.4) → **0.080 m/deg** (0.075 / 0.081 / 0.074 / 0.106) | −2.2 px/frame at fx 416 → 4.9 deg/s | ours realises only ~1/3 of the commanded yaw and ~0.3 m per u, i.e. the commanded arc SHAPE (ratio 0.0833 u/deg with 1 u ≈ 0.96 m, radius 4.6 m) at one third of the rate |
| ours P1 (teacher) | fwd 1 u/s | 0.27 m/s (0.32 / 0.23 / 0.25 / 0.26), yaw −0.2 deg/s | divergence +0.45 | ours' forward speed is 4x slower than Astra's preset (per-video normalised translation → the model's own prior) |

| Astra P3, u_scale 0.15 (frozen clip) | fwd 0.03 units/latent + yaw 2.4 deg/latent (12 deg/s) | fwd **0.00 m/s** (−0.06 / −0.14 / +0.21 / 0.00), yaw 17.9 deg/s (11.3 / 13.0 / 22.9 / 24.3), lateral +0.23 → **0.000 m/deg** | divergence −0.02 (no zoom), pan −9.8 px/frame right | yaw exact for 5 s then 1.9x (drift); the translation is DROPPED when combined with rotation |
| Astra P3, u_scale 0.30 (2x preset, `calib/garden_1_P3_u0.30`) | fwd 0.06 + yaw 2.4 | fwd 0.11 m/s (0.11 / 0.05 / 0.33 / −0.05; windows median 0.006), yaw 18.1 → 0.006 m/deg | divergence −0.02 | same |
| Astra P3, u_scale 0.45 (3x preset, `calib/garden_1_P3_u0.45`) | fwd 0.09 + yaw 2.4 | fwd 0.17 m/s (0.17 / 0.01 / 0.29 / 0.19; windows median 0.18), yaw 16.3 (12.0 / 11.9 / 17.9 / 23.5), lateral +0.49 → 0.010 m/deg | divergence +0.10 | coherent clip (hallucinated street with blossoms), still no visible dolly |

**u_scale decision (1 u/s → Astra units/s): 0.15 kept; the task's ratio match is NOT reachable by the unit.** Target = Astra's realised P3
forward/turn ratio equal to ours' 0.080 m/deg. Astra moves at 1.10 m/s on P1 (pure forward, 1 unit ≈ 7.3 m), but when the same forward is
combined with the P3 rotation it executes the pan and suppresses the translation: 0.00 / 0.11 / 0.17 m/s at 1x / 2x / 3x its preset speed
(≤ 0.010 m/deg, 8x below ours); a linear extrapolation would need ~16x the preset (0.48 units/latent), far outside its demo/training range.
What CAN be matched is the input ratio: at 0.15 Astra is asked for 0.22 m per 2.4 deg = 0.092 m/deg vs ours' realised 0.080 (ratio-matched
input u_scale 0.131; 0.20 with the drifted yaw gain 1.5) — 0.15 lies inside that band and inside DA3's ±15 % metric-scale spread. 0.15 is also
Astra's own preset forward speed (its "held W key", PROTOCOL §2 keyboard rule / rule 3: the baseline at its native operating point), larger
values buy no translation on P3, and it keeps the frozen P1/P3 clips valid. Consequence: Astra's RPE-Trans on P3/P4/P5 scores the dropped
translation and RPE-Rot the 1.5x drifted pan — the model's behaviour at the protocol's rates. Override for experiments: `U_SCALE=x run.sh …`
/ `astra_gen.py --u-scale` (camera.u_scale in timing.json records it).

### Known limits / to do
1. (done 10/17) translation unit verified with DA3 (above); (done) frozen P2 clip regenerated with LATENTS=56, tail clean (smoke 3).
2. The other programs x 3 astra scenes + 12 neutral scenes (= 75 clips, ~3.3 min each in batch mode = ~4.1 h of GPU time; `make_jobs.py`).
3. `baselines/convert.py` CALIB['astra'] still carries the 16-fps draft (161 frames / 0.12 units per u); `convert.py build` regenerates
   calib.json from it, so re-run `baselines/astra/calib_merge.py` after any build. run.sh never reads calib.json or the programs/*.npz
   astra arrays (program_to_astra.py builds the 20 fps embedding itself).
4. Astra's P2 over-rotation (273 deg realised) and its P1 scene cut at 8.45 s are the model's own behaviour at the protocol's rates; a slower
   P2' (PROTOCOL rule 2) is the fair complement, and if the protocol later wants the realised P2 pan at 180 deg, scale the commanded yaw by
   0.66 (not frozen here, would invalidate the frozen P2 clip).
5. ours under-rotates on P3 (0.34x) and moves 0.27–0.32 m/s — a property of OUR arm, outside this folder's scope but relevant to RPE-Rot.
