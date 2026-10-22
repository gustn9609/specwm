# Waypoint-1.5-1B baseline — ITS setting (frozen 2026-10-16)

## Checkpoint / stack (verified)
| item | value |
|---|---|
| checkpoint | HF `Overworld/Waypoint-1.5-1B`, revision `391f92827075edcf4a8b3c8a2ddae010698f8636` (= `refs/main` on 2026-10-16; full snapshot in `~/.cache/huggingface/hub/models--Overworld--Waypoint-1.5-1B`, 7.0 GB: `transformer/diffusion_pytorch_model.safetensors` 7.44 GB fp32 + `vae/diffusion_pytorch_model.safetensors` 23 MB). The 720p variant; the 360P repo (`Waypoint-1.5-1B-360P`) is NOT used. |
| params | transformer `WorldModel` 1.236 B after `apply_inference_patches()` (1.861 B in the raw checkpoint incl. unused text-conditioning weights; the card says "1.2B"), VAE `ChunkedStreamingTAEHV` 11.4 M |
| inference stack | official Modular-Diffusers path of the model card: `ModularPipeline.from_pretrained(..., trust_remote_code=True)` (the repo's `modular_blocks.py`), `apply_inference_patches()`, `transformer.compile(fullgraph=True, mode="max-autotune", dynamic=False)`; diffusers 0.39.0, torch 2.11.0+cu128, python 3.12 (main env `scripts/envrun.sh`) |
| precision | bf16 weights + activations (model card `torch_dtype=torch.bfloat16`), no quantization (world_engine's intw8a8/fp8/nvfp4 options not used) |
| attention | torch `flex_attention` with a block mask: local window 16 latent frames, global window 128 (dilation 8, period 4); no flash-attn / SageAttention |
| VAE | taehv1_5 (`ChunkedStreamingTAEHV`, Tiny Hunyuan AE), 32 latent channels, 4x temporal / 16x spatial compression (8x conv + 2x2 pixel-unshuffle), streaming decode with internal temporal memory (reset per clip) |
| text | `prompt_conditioning = null` in this checkpoint: the UMT5-XL text encoder is never loaded, zero prompt embeddings are passed (`text_encoder_resident = false`) |
| resolution | 1024x512 (patch grid 32x16 x patch 2 x VAE 16); the loader STRETCHES any input to 2:1 — no letterbox. `run_waypoint_program.py` does `PIL resize((1024, 512), Image.BILINEAR)` like the official Space (since 10/17; the model card's bare `resize((1024, 512))` is PIL-default bicubic, which the 10/16 smoke clip used) |
| frames per call / fps | **native 60 fps, 4 frames per call**: 1 latent = 4 decoded frames per pipeline call (`config.yaml` `inference_fps 60`, `base_fps 15` -> ts_mult 1, temporal compression 4), i.e. 15 calls/s. 10 s clip = 150 calls = 600 frames; the videos are written at 60 fps |
| denoising | rectified flow, sigmas [1.0, 0.9, 0.75, 0.3, 0.0] -> 4 Euler steps + 1 KV-cache pass per call (5 transformer forwards per 4 frames), no CFG |
| KV cache | `StaticKVCache`, context `n_frames 512` latent frames; local 16 / global 128 windows |
| seed | `torch.manual_seed(seed)` before the clip (the blocks draw noise with `torch.randn` from the global RNG; the `generator` input is unused) |
| warm-up | one discarded rollout of 12 calls on the same image (compile + allocator), then a fresh world (new KV cache, `vae.reset()`) for the timed clip |

## Inputs
* Scenes: `baselines/scenes/wp/` (20 of Waypoint's own starter images, provenance in `scenes/wp/SOURCES.md`), no captions.
* Control: PROTOCOL Section 2 programs -> `encode_program.py`, which uses the shared `baselines/convert.py` (`program_to_c2w` at 60 fps + `encode_waypoint`,
  4 frames summed per call) when it is importable and otherwise its own inline encoder (same rules, sampled at the 15 Hz call rate; the two agree on all
  150 calls of P1/P2/P3 and differ only at the boundary calls 37/112 of P4/P5, which convert.py averages). K_yaw comes from `baselines/calib.json`
  (`waypoint.K_yaw`, same value as `K_yaw_units_per_deg_per_call`), overridable with `--k-yaw`. Per call `{mouse: [K_yaw*dyaw_deg, K_yaw*dpitch_deg],
  buttons: [W 87 | S 83 | A 65 | D 68]}`; the commanded c2w is saved as `program_c2w.npy`, the metadata (encoder, K, totals) as `program.json`.
  Call 0 seeds the world with the image AND applies action[0]; calls 1..149 pass `image=None`.
* **The `image=None` fix (bug in the old runner, fixed here):** `tools/waypoint/run_waypoint_actions.py` (10/9 head-to-head and the 10/9 K_yaw = 118
  calibration) never passed `image=None` after the first call; `ModularPipeline.__call__` deep-copies the state and keeps every previously passed input,
  so the start image was re-encoded and re-appended to the KV cache at EVERY call (the model card sets `state.values["image"] = None`, the official Space
  passes `image=None`). **Every 10/9 Waypoint number was therefore measured with the start image re-appended on every call**: the 28–31 fps (each call
  paid an extra VAE encode + cache pass), the −112 deg on easy_lake and the "118 units/deg/call" are not the model's behaviour and are superseded by the
  10/16–10/17 numbers below. `frame_timestamp_end` in timing.json (= calls + 1 = 151) verifies that the image is appended exactly once here.

## Run
```
flock /tmp/gpu6.lock env CUDA_VISIBLE_DEVICES=6 /home/hk28965/specwm/baselines/waypoint/run.sh IMAGE PROGRAM_ID SEED OUTDIR
   -> OUTDIR/{actions.json, program_c2w.npy, program.json, video.mp4 (60 fps, libx264 crf 14), timing.json, run.log}
```
`run.sh` is offline (`HF_HUB_OFFLINE=1`, components loaded from the frozen snapshot with `AutoModel.from_pretrained(snapshot, subfolder=...)` +
`pipe.update_components`, because `pipe.load_components` resolves the hub id and fails offline). Extra args go to `run_waypoint_program.py`
(`--no-compile`, `--warm N`, `--sweep 60,120 --sweep-calls 30` for calibration rollouts). Peak VRAM ~7 GiB.

**Batch mode (10/17)** — the torch.compile warm-up (~146 s) is paid once per process, so the frozen generation runs many clips per process:
```
flock /tmp/gpu4.lock env CUDA_VISIBLE_DEVICES=4 /home/hk28965/specwm/baselines/waypoint/run.sh --batch LIST.tsv OUTROOT [extra args]
   LIST.tsv: IMAGE<TAB>PROGRAM_ID<TAB>SEED[<TAB>OUTNAME]  per line (# comments ok); OUTNAME defaults to <set>__<scene>__<program> (PROTOCOL Section 4 layout)
   -> OUTROOT/<name>/{actions.json, program_c2w.npy, program.json, video.mp4, timing.json, run.log} per clip + OUTROOT/{batch_jobs.json, batch.log, batch_summary.json}
```
`batch_jobs.py` encodes every program (K_yaw from calib.json) and `run_waypoint_program.py --batch` loads + compiles the model once, runs ONE warm-up rollout
(12 calls on clip 0, discarded), then runs each clip exactly like the single-clip path: fresh world (new `StaticKVCache`, `vae.reset()`),
`torch.manual_seed(seed)`, peak-memory stats reset per clip, same timing fields (plus `batch_mode/batch_index/batch_size`, and `warmup_*` of the shared
warm-up). Existing clips (video.mp4 + timing.json) are skipped unless `--overwrite`; a failing clip writes `<out>/FAILED` and the batch continues (exit 3).
Budget: ~150 s compile + ~20 s per clip, ~7 GiB -> ~20 clips per 10-min lock hold (split LIST.tsv accordingly; skipped clips make re-runs cheap).
Verified 10/17 on 2 clips (P2 + P3, `smoke/batch_k79`, see the calibration section): the batch P2 clip vs the single-clip P2 run with the same seed is
compared frame-by-frame there.

## timing.json (PROTOCOL Section 4)
gen_s = sum of the pipeline calls with `output_type="latent"` (controller encode + 4 denoise steps + cache pass; call 0 also VAE-encodes the image);
e2e_s = gen_s + `vae.decode` + copy of the 4 uint8 frames to CPU (exactly the pipeline's own "pil" path); ttff_s = call 0 end-to-end (image in -> 4 frames on CPU);
steady_fps_gen / steady_fps_e2e over calls 1..149; peak_alloc/reserved from `torch.cuda.max_memory_*` reset at the start of the clip (warm-up excluded).
`gpu_smi_before/after` record the sharing state of the GPU during the measurement.

## Smoke (GPU 6 under the lock 22:35:18–22:39:01 CDT, 2026-10-16; GPU otherwise idle: 0 % util, 9 GB held by an idle process) — `smoke/`
`smoke/run_smoke.sh` = `run.sh scenes/wp/highland_castle_loch.png P2 1234 smoke/wp__highland_castle_loch__P2 --sweep 0,60,120,240,480 --sweep-calls 30`, then
`frame_strip.sh` and `calib_yaw.py`. Whole session 223 s wall.

| quantity (timing.json) | value |
|---|---|
| frames / calls | 600 / 150 (10.0 s at 60 fps native) |
| load_s / warm-up | 3.4 s / 146.5 s (12 warm calls; the first call = 144 s torch.compile max-autotune; the inductor autotuner logs many benign "No valid triton configs ... out of resource" lines while skipping over-sized candidates) |
| ttff_s (image in -> 4 frames on CPU) | **0.464 s** |
| gen_s / e2e_s (150 calls, DiT only / incl. decode+CPU copy) | **15.02 s / 15.94 s** |
| steady_fps_gen / steady_fps_e2e (calls 1..149) | **40.9 / 38.5 fps** (p50 94.9 ms per 4-frame call, p99 99.5 ms; decode 6.9 ms per call) -> 0.64x real time at 60 fps native (the card quotes 56 fps on an RTX 5090) |
| peak_alloc_GiB / peak_reserved_GiB | **6.63 / 7.39** (`torch.cuda.max_memory_*`, clip only) |
| frame_timestamp_end | 151 = calls + 1 (image appended exactly once) |

Frames (`frames_sec.png`, one tile per second; `frames_strip6.png` = PROTOCOL strip): the start image is a game-style loch with a castle at the left-centre; the
camera **pans RIGHT** (content flows left): the castle leaves through the left edge within ~1.5 s, then hills/trees (2–5 s), grass bank and sky (6–7 s), a shore
with a boat (8 s) and finally a lake with a castle-like building at the right edge (9–10 s). The first-person rifle/hands stay fixed (static overlay). Scene
stays coherent, no collapse. It does NOT stop at 180 deg: DA3 pose recovery gives **269.7 deg over the clip (1.80 deg/call = 27 deg/s vs 18 commanded)**,
i.e. 1.5x over-rotation with K_yaw = 118; the per-second rate ramps from 1.0 deg/call (second 1) to 1.5–2.3 deg/call later.

Note on the PROTOCOL label: P2 is "+18 deg/s (left)" there, but by the Section-2 formula (`yaw = atan2(Rr[0,2], Rr[2,2])`, OpenCV c2w) positive yaw turns the
camera to its RIGHT, and Waypoint's +mouse x is also a right turn — every converter and the RPE reference use the same formula, so all models get the same
(right) turn; only the word "left" is wrong (calib.json `yaw_pitch_convention.geometric_note` says the same).

## K_yaw calibration (PROTOCOL rule 3: on the baseline's own outputs) — `calib_yaw.py`, results `smoke/calib_da3.json`
DA3 (DA3NESTED-GIANT-LARGE-1.1, windows of 16 frames sampled every 4 = one per call, relative camera-frame rotation with the encoder's formulas):

| rollout | mouse x / call | calls | realised yaw total | deg / call | deg / s | K = units/deg |
|---|---|---|---|---|---|---|
| P2 clip (K 118) | 141.6 | 150 | 269.7 | 1.798 | 27.0 | **78.8** |
| sweep m0 | 0 | 30 | -0.02 | -0.001 | 0 | – (no drift without input) |
| sweep m60 | 60 | 30 | 24.1 | 0.804 | 12.1 | 74.6 |
| sweep m120 | 120 | 30 | 34.0 | 1.134 | 17.0 | 105.8 |
| sweep m240 | 240 | 30 | 71.0 | 2.368 | 35.5 | 101.4 |
| sweep m480 | 480 | 30 | 139.0 | 4.633 | 69.5 | 103.6 |

All turns are to the right (positive). The response is sublinear in the magnitude and ramps up over the first ~1 s (the 2 s sweeps are mostly ramp: last-fifth
rates 1.1 / 1.6 / 3.0 / 5.6 deg/call), so one linear K is only valid near the operating point; the 10 s P2 clip is the measurement that matters.
The 10/16 one-clip proposal was K_yaw = 79 (= 141.6 / 1.798). The whole-frame phase-correlation shift (−1.6 to −2.7 px/frame for every magnitude) is NOT
usable here because the static rifle overlay dominates it; only the direction (right) is taken from it.

### Verification 10/17 (GPU 4 under `flock /tmp/gpu4.lock`, idle GPU, 4 lock sessions of 86–221 s; logs `smoke/verify.log`, `smoke/batch_test.log`)
Round 1 = `smoke/run_verify.sh` (single-clip `run.sh`, K 79) -> `smoke/wp__highland_castle_loch__P2_k79`; rounds 2–4 = `smoke/run_batch_test.sh` with
`run.sh --batch` (`smoke/batch_test.tsv`, `batch_k69.tsv`, `batch_k66.tsv` -> `smoke/batch_k`, `batch_k69`, `batch_k66`; per-clip K in the 5th TSV column).
All clips: highland_castle_loch, 150 calls = 10 s, BILINEAR stretch, DA3 yaw recovery (`calib_yaw.py`, labels in `smoke/calib_da3.json`; `b_` = batch).

| round | clip (label) | K | mouse x / call | seed | realised yaw over 10 s (↓ distance to 180) | deg / call | deg / s |
|---|---|---|---|---|---|---|---|
| 10/16 | P2_k118 | 118 | 141.6 | 1234 | 269.7 | 1.798 | 27.0 |
| 1 | P2_k79 (single run.sh) | 79 | 94.8 | 1234 | 219.6 | 1.464 | 22.0 |
| 2 | b_P2_k79 (batch) | 79 | 94.8 | 1234 | 206.6 | 1.377 | 20.7 |
| 2 | b_P2_k65 | 64.7 | 77.6 | 1234 | 165.4 | 1.103 | 16.5 |
| 2 | b_P2_k54 | 53.6 | 64.3 | 1234 | 150.8 | 1.005 | 15.1 |
| 3 | b_P2_k69 (= b_P2_k69_rep, same process: identical) | 69 | 82.8 | 1234 | 203.1 | 1.354 | 20.3 |
| 3 | b_P2_k69_s1235 | 69 | 82.8 | 1235 | **175.9** | 1.173 | 17.6 |
| 4 | b_P2_k66 | 66 | 79.2 | 1234 | 161.1 | 1.074 | 16.1 |
| 4 | b_P2_k66_s1235 | 66 | 79.2 | 1235 | 168.2 | 1.121 | 16.8 |

* K = 79 over-rotates (207–220 deg, i.e. 15–22 % too much; the 10/16 proposal from the single K 118 clip assumed a linear response). The task's update
  rule "K = 79 × realised/180" is written inverted (it would raise K to 96); the proportional correction is K = 79 × 180/219.6 = 64.7, which gave 165 deg.
* The **turn response at a fixed input is noisy**: the same 82.8 units/call give 203 deg (seed 1234) and 176 deg (seed 1235); the same seed in two
  processes gives 220 vs 207 deg (K 79). Over the eight 10-s clips with 64–95 units/call the proportional fit total_yaw = c·units has c = 2.214 deg per
  unit with residual rms 10.7 deg (5.9 %), so one clip cannot verify K to the ±5 % band (170–190) — two of the four clips at K 66/69 land in it, two do not.
* **Final `calib.json` waypoint.K_yaw = 68** (= 150 / c; affine fit 67.8, power-law fit 67.9, exponent 0.97 ≈ linear in this range; `smoke/kyaw_fit.json`).
  Expected realised P2 pan at K 68: 181 ± 11 deg (rms over seeds/processes). It is bracketed by direct clips: K 66 -> 161/168 deg, K 69 -> 176/203 deg.
* Within one process the rollout is deterministic (b_P2_k69 vs b_P2_k69_rep: 600 frames bit-identical, MAD 0.000); across processes the same seed
  diverges slowly (single P2_k79 vs batch b_P2_k79: MAD 1.3 at frame 0 -> 16 at frame 600, PSNR 26.9 dB mean) because torch.compile max-autotune picks
  kernels per process. Batch mode therefore reproduces the single-clip path exactly up to this compile nondeterminism (`frame_diff.py`).
* **P3 (12 deg/s + W, commanded 120 deg)**: realised 81.8 (K 53.6), 102.8 (64.7), 108.5 (66), 105.8 (69), 149.6 (79) deg -> at the final K ≈ 0.9× of the
  command, i.e. slightly sublinear relative to P2 (1.0×), but the measurement is noisy on this starter: the forward walk runs into the grass bank and the
  frames become a close-up ground texture at ~6–9 s (DA3 yaw per second jumps 3 -> 16 deg there; `batch_k*/wp__*P3*/frames_sec.png`), after which a
  different lake/ruin scene appears (9–10 s). No extra P3 correction is applied; the single K_yaw holds for every program.
* Per-second realised yaw ramps over the first ~2 s (10–12 deg in second 1, 16–20 deg in seconds 2–3 at K 66–69), as on 10/16.

Timing of the 14 clips of 10/17 (GPU 4 idle, `timing.json` of each): ttff 0.44–0.59 s (median 0.51), steady_fps_gen 40.2–53.2 (median 42.1), steady_fps_e2e
37.8–48.6 (median 39.5; one clip, b_P3_k54, ran at 62.9 ms/call instead of the usual 94 ms), peak_alloc 6.63 GiB / peak_reserved 7.13 GiB in every clip,
frame_timestamp_end 151 in every clip. Compile warm-up: 146 s on 10/16 with a cold inductor cache, 18–34 s per process on 10/17 with the cache
(`/tmp/torchinductor_<user>`) warm. Batch sessions: 6 clips in 165 s, 4 in 113 s, 3 in 97 s wall (load 3.4 s + warm-up + ~20.5 s per clip incl. ffmpeg).

## Known caveats
* The 10/16 fps numbers are from a single clip on GPU 6 (idle but shared, another process held 9 GB); the 14 clips of 10/17 on an idle GPU 4 agree
  (median 42.1 / 39.5 fps gen / e2e, see the calibration section). The compile warm-up is per process: 146 s cold, 18–34 s with a warm inductor cache.
* K_yaw = 68 is a fit with ~6 % rms seed-to-seed / process-to-process spread of the realised turn; it was calibrated on ONE starter (highland_castle_loch).
* Pitch sign (+mouse y) is untested (no program uses pitch). Mouse magnitudes beyond ~500 units/call untested.
* Waypoint renders every input as a first-person game view with a fixed rifle overlay; photo scenes (neutral set) are out of its domain.
* `tools/waypoint/run_waypoint_actions.py` is left untouched (it has the image re-append bug); use `baselines/waypoint/run.sh` only.
* The 10/16 smoke clip (`smoke/wp__highland_castle_loch__P2`, K 118) was generated with the PIL-default bicubic stretch; everything from 10/17 on
  (K 79 verification, batch mode, frozen runs) uses the BILINEAR stretch of the official Space (`timing.json` `image_handling` says which).
