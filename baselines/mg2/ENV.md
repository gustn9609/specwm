# Matrix-Game 2.0 baseline — ITS setting (frozen 2026-10-16, PROTOCOL v1)

| item | value |
|---|---|
| model | Matrix-Game 2.0, universal checkpoint (Skywork), 1.8 B causal DiT (SkyReels-V2 / Wan2.1-1.3B class + action modules), few-step distilled |
| checkpoint | HF `Skywork/Matrix-Game-2.0` revision `08c42d1a99e57e79df60a964bb815618cdf460f8`, file `base_distilled_model/base_distill.safetensors` (6.03 GiB) + `Wan2.1_VAE.pth` (0.47 GiB) + `models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth` (4.44 GiB) + `xlm-roberta-large/` tokenizer; local copy `/home/hk28965/specwm/hf/Matrix-Game-2.0` (11 GiB; `base_model/` and `templerun_distilled_model/` were NOT downloaded). The HF cache entry `~/.cache/huggingface/hub/models--Skywork--Matrix-Game-2.0` was an empty stub (12 KB, refs only). |
| code | `https://github.com/SkyworkAI/Matrix-Game` commit `c7a88bfc039cd97695082484986e95b962390676` (2026-10-16), clone `/home/hk28965/specwm/tools/mg2/Matrix-Game`, sub-dir `Matrix-Game-2/`; the repo is used unmodified (imported via `sys.path`, cwd = `Matrix-Game-2/` because its config paths are relative) |
| config | `configs/inference_yaml/inference_universal.yaml`: denoising_step_list [1000, 666, 333] (warp_denoising_step: true -> their shift-5 FlowMatch timesteps), num_frame_per_block 3 latents, context_noise 0, timestep_shift 5.0, model config `configs/distilled_model/universal` (local_attn_size 6, 30 blocks, action modules in blocks 0-14) |
| resolution / fps | 352x640 native (centre-crop to 20:11 then resize, their `_resizecrop` + `v2.Resize`), latent 44x80 = 880 tokens per latent frame (their hard-coded `frame_seq_length`); **12 fps native playback** (their `utils/visualize.py` writes every video at 12 fps; the paper's "25 FPS" is throughput on an H100). Content fps is not stated in the paper; 12 fps is the official export rate. |
| clip | 33 latents = 129 frames generated (constraints: frames % 4 == 1 and latents % 3 == 0; 121 frames would be 31 latents, not a multiple of 3), the first 120 frames = 10.0 s written to `video.mp4`; the program's last segment simply continues for the 9 extra generated frames |
| precision | DiT bf16 (their `weight_dtype`); VAE decoder fp16 + `torch.compile(mode="max-autotune-no-cudagraphs")` (their `inference.py`); image VAE encoder weights fp16 moved to bf16 (their `get_wanx_vae_wrapper(..., torch.float16).to(device, bf16)`); CLIP image encoder bf16 |
| attention | flash-attn 2.8.1 (`flash_attn_func` in the action modules, `flash_attn_varlen_func` in the DiT self-attention via `wan/modules/attention.py`); FA3 (`flash_attn_interface`) not installed and not applicable (Blackwell); the action modules' `flex_attention` is `torch.compile`d at import (their module-level default `max-autotune-no-cudagraphs`) — this is the official stack, no SDPA shim (the 10/9 run used `tools/mg2/shim`) |
| VAE | Wan2.1 VAE; decoder = their streaming `demo_utils.vae_block3.VAEDecoderWrapper` with per-chunk feature cache (first chunk -> 9 frames, then 12 per chunk); encoder = their `wan/vae/wanx_vae` tiled encode (tile 44x80, stride 23x38) of the start image + zero padding |
| text encoder | none (MG2 takes no text). Resident on GPU: CLIP ViT-H/14 + xlm-roberta-large image/text tower (used once per clip for `visual_context`). |
| KV window | local attention 6 latent frames (= 24 pixel frames) for the DiT KV cache, 6 for the keyboard cache, 6 x 880 for the mouse cache; cross-attn cache 257 tokens |
| controls | per-frame keyboard one-hot [fwd, back, left, right] + mouse [pitch, yaw] in their units, from `baselines/convert.py` (`encode_mg2(program_to_c2w(P, 12, 129))`, rules of `tools/mg2/poses_to_cond.py`); mouse = K x deg/frame clipped to their benchmark magnitude CAM_VALUE = 0.1 (`utils/conditions.py`); the saturation (max unclipped |mouse|, fraction of frames clipped) is written to `timing.json["control"]` |
| seed | `utils.misc.set_seed(seed)` (python/numpy/torch) right before the noise draw; default 1234 |
| warm-up | one full clip with the same shapes (compile + autotune) run first and discarded (`OUT/warmup/`); the measured clip is the second; `--warm 0` skips it (batch mode warms once) |
| GPU | one RTX PRO 6000 Blackwell 96 GB; GPU 6 on spi0 is SHARED (other jobs resident) — timings below were taken with another process on the same GPU and must be re-measured on a quiet GPU for the table |

## Environment (`/home/hk28965/specwm/tools/mg2/venv`, built by `baselines/mg2/build_venv.sh`, log `build_venv.log`)
The ABot venv (`tools/abot/venv`) could not be reused: its torchvision 0.26.0+cu128 does not match its torch 2.8.0 (`operator torchvision::nms does not exist`), and MG2's loader needs `torchvision.transforms.v2`.
```
uv venv tools/mg2/venv --python /usr/bin/python3.12 --seed
uv pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install tools/abot/flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
uv pip install diffusers==0.35.1 transformers==4.57.1 "tokenizers>=0.20.3" "accelerate>=1.1.1" tqdm imageio imageio-ffmpeg easydict ftfy "numpy<2.3" omegaconf einops av safetensors opencv-python open_clip_torch sentencepiece pydantic scikit-image matplotlib "huggingface_hub[cli]" triton==3.4.0
```
(their `requirements.txt` minus the serving / TensorRT / ONNX / wandb / dashscope / pycuda extras that `inference.py` never imports; `python setup.py develop` replaced by `sys.path`). Verified: `torch 2.8.0+cu128, torchvision 0.23.0+cu128, flash_attn 2.8.1, triton 3.4.0, diffusers 0.35.1, transformers 4.57.1`, Python 3.12.3.
`run.sh` exports the same JIT environment as `tools/abot/run.sh` (python headers from `tools/pydev`, CUDA toolkit `tools/cuda128_tk`) and keeps the inductor/triton caches in `tools/mg2/{inductor_cache,triton_cache}`.

## Run
```
# one clip (GPU chosen by the caller; on spi0 GPU 6 is shared -> flock)
flock /tmp/gpu6.lock env CUDA_VISIBLE_DEVICES=6 /home/hk28965/specwm/baselines/mg2/run.sh \
    /home/hk28965/specwm/baselines/scenes/mg2/universal_0000.png P2 1234 /home/hk28965/specwm/baselines/runs/mg2/mg2__universal_0000__P2
# -> OUT/video.mp4 (120 frames, 12 fps, 640x352, libx264 q8) + OUT/timing.json (+ OUT/warmup/ discarded)
# many clips with one model load: JOBS.json = [{"image":..., "program":"P1", "seed":1234, "out":...}, ...]
env <same exports as run.sh> /home/hk28965/specwm/tools/mg2/venv/bin/python /home/hk28965/specwm/baselines/mg2/run_mg2.py --jobs JOBS.json
# calibration of the mouse unit on MG2's own output (DA3 pose recovery, main env):
CUDA_VISIBLE_DEVICES=6 /home/hk28965/specwm/scripts/envrun.sh python /home/hk28965/specwm/baselines/mg2/calib_da3.py OUT/video.mp4 OUT/timing.json --out calib.json
```
Flags: `--frames 129 --keep-frames 120 --fps 12 --vae-dtype fp16|bf16 --no-compile --mouse-scale K --max-mouse 0.1 --warm 0|1`.

## timing.json
PROTOCOL Section 4 fields (model, ckpt, precision, attention, vae, resolution_out, frames, fps_native, frames_per_call_or_chunk, steps, gen_s, e2e_s, ttff_s, steady_fps_gen, steady_fps_e2e, peak_alloc_GiB, peak_reserved_GiB, text_encoder_resident, kv_window, warm) plus: frames_generated, latents_per_chunk, steps_warped, decode_s, prep_s (image -> conditions incl. VAE/CLIP encode), fps_gen_all / fps_e2e_all (all chunks incl. the first), per-chunk `chunk_dit_s` / `chunk_decode_s` / `chunk_frames`, `control` (K, saturation), load_s, gpu, gpu_mem_used_by_others_at_start_GiB.
Definitions: gen_s = sum over chunks of (3 denoising forwards + 1 clean-context KV forward) with cuda synchronisation; decode_s = VAE decoder per chunk; e2e_s = from the loaded PIL image to the last decoded frame on GPU (image preprocessing + VAE/CLIP encode + all chunks); ttff_s = image -> first decoded chunk (9 frames) available; steady fps = frames / time over chunks 2..11 (the first chunk is excluded).

## Smoke (2026-10-16, GPU 6 under `flock /tmp/gpu6.lock`, GPU otherwise empty: 3 MiB used before the run)
`run.sh scenes/mg2/universal_0000.png P2 1234 baselines/mg2/smoke/universal_0000__P2` (lock 23:02:46 -> exit 23:09:03, 6.3 min incl. 24 s load and the compile):

| clip | gen_s (DiT, 11 chunks) | steady fps gen ↑ | decode_s | e2e_s | ttff_s ↓ | steady fps e2e ↑ | peak alloc / reserved GiB |
|---|---|---|---|---|---|---|---|
| warm-up (compile + autotune) | 3.92 | 33.9 | 291.8 | 297.5 | 291.7 | 20.8 | 11.53 / 12.80 |
| measured (2nd clip) | **3.83** | **34.4** | 38.2 | 43.7 | 38.0 | **21.0** | 11.53 / 12.80 |

Per chunk (measured clip): DiT 0.345-0.357 s for 3 latents = 12 frames (3 denoising forwards + 1 context forward), decode 0.22 s per chunk — except the FIRST chunk's decode: 289.5 s in the warm-up (torch.compile + max-autotune of the decoder and of the action modules' flex_attention) and **36.0 s again in the 2nd clip** (torch 2.8 automatic dynamic shapes recompile the decoder's first-chunk graph, whose feature cache is `None`, once more; `inference.py` never sees this because it runs one clip per process). Hence `run_mg2.py --warm 2` (default from the verification below) and ttff must be read from a clip after two warm-ups. Steady-state numbers are unaffected: 34.4 fps DiT-only, 21.0 fps incl. decode (= the "25 FPS on H100" claim's quantity; the 10/9 SDPA-shim run gave 20.5 fps gen-only).
Inductor printed 6 x "No valid triton configs ... triton_mm Required: 147456 Hardware limit: 101376" during autotune (candidate tiles that exceed Blackwell's shared memory are skipped, other configs are used; benign).
Frames (`smoke/universal_0000__P2/sheet.png`, f0/20/40/60/80/100/119): the painting-like snowy field with a haystack is rendered as a game-style snowy valley; the haystack exits on the LEFT while pine trees and cabins enter from the RIGHT (= content moves left = the camera pans RIGHT, the +yaw direction of convert.py), the pan continues steadily to a mountain at f119, textures stay sharp (fp16 compiled decoder: no overflow/texture collapse), no black wedge.
Control: P2 (18 deg/s) at K = 0.067 and 12 fps = 1.5 deg/frame -> mouse yaw 0.1005 -> clipped to their cap 0.1 on 99.2 % of frames (0.5 % over the cap). Optical-flow proxy (cv2 DIS, median horizontal flow): -9.63 px/frame, flat over the 4 segments (-9.78/-9.78/-9.45/-9.37) -> 11.5 deg/s if fx = 0.9 W, 20.7 deg/s if 90 deg HFOV, 26.4 deg/s if Minecraft's 70 deg VFOV: the focal length decides, which is why the K calibration uses DA3 pose recovery (below).

## Mouse calibration on MG2's own output (PROTOCOL 1.3; `calib_da3.py`, result `smoke/calib_P2_universal_0000.json`, merged into `baselines/calib.json["mg2"]` as `K_per_deg_per_frame_da3` + `measurement_da3`)
DA3 (DA3NESTED-GIANT-LARGE-1.1, process_res 504, all 120 frames of the P2 smoke clip, peak 16.5 GiB, 20 s) recovers the camera path; per-frame yaw by the converters' formula:

| quantity | value |
|---|---|
| commanded mouse yaw (their units, per frame) | 0.100 (P2's 1.5 deg/frame x 0.067 = 0.1005, clipped to the 0.1 cap on 99 % of frames) |
| realized yaw per frame (DA3, median) | **1.420 deg** (per 2.5 s segment 1.43 / 1.415 / 1.399 / 1.565); pitch median -0.004 deg; sign = right turn as commanded |
| realized yaw rate / total | **17.0 deg/s**, 169.5 deg over the 10 s clip (commanded 18 deg/s, 180 deg) |
| K (mouse units per deg per frame) | **0.0704** (10/9 flow estimate: 0.067, 5 % lower) |
| largest unsaturated yaw rate at 12 fps | **17.0 deg/s** (= cap 0.1 / K x 12 fps) — P2 (18 deg/s) and the P2/P4 18 deg/s segments ARE saturated (by 5-6 %); P3's 12 deg/s is not (0.070 < 0.1) |
| DA3 intrinsics of the clip | fx 251 px at DA3's 504 px width -> HFOV ~ 90 deg (fx ~ 320 px at 640 px width), not the protocol's 0.9 W (58 deg) default |

`calib.json["mg2"]["K_per_deg_per_frame"]` is left at 0.067 (the shared constant the programs/*.npz encodings and the other models' entries were built with); `calib_merge.py --apply` switches it to 0.0704. Either way a P2' at <= 17 deg/s (16 deg/s leaves a 6 % margin for the +-0.07 deg/frame segment spread) is needed for an unsaturated turn program, to be run for all models per PROTOCOL Section 2.

## Verification of the warm-up count (same lock session, fresh process, inductor cache warm): `run.sh scenes/mg2/universal_0005.png P2 1234 smoke/verify_warm2 --warm 2`
| clip | gen_s | decode_s | e2e_s | ttff_s | steady fps gen / e2e | first-chunk decode |
|---|---|---|---|---|---|---|
| warm-up 1 | 3.82 | — | 45.7 | 40.0 | 34.7 / 21.2 | ~38 s (compile-cache load + first-chunk graph) |
| warm-up 2 | 3.81 | — | 12.4 | 6.7 | 34.6 / 21.2 | 4.8 s (dynamic-shape recompile) |
| **measured (3rd clip)** | **3.79** | **2.37** | **7.76** | **2.11** | **34.8 / 21.2** | 0.17 s |

So with two warm-ups the measured clip is clean: ttff 2.11 s = 1.59 s prep (their tiled VAE encode of the start image padded to 129 frames + CLIP) + 0.34 s first-chunk DiT + 0.17 s decode; e2e 7.76 s per 129-frame clip (16.6 fps incl. prep, 21.2 fps steady), DiT alone 34.8 fps. `--warm 2` is the default now. Frames (`smoke/verify_warm2/sheet.png`): road scene with mountains, pans right coherently (car and road exit left, lawn/trees enter right), slight downward drift in the last 2 s.

## Notes / caveats
- Both GPU sessions had the GPU to themselves (0-3 MiB used by others at lock time), so the steady-state numbers are clean single-GPU measurements; the queue wait for the shared lock was 14 + 37 min.
- Their universal checkpoint renders photos/paintings as a game world (as noted on 10/9); the 8 Minecraft scenes are its training domain.
- `baselines/calib.json["mg2"]["n_frames_10s"] = 121` (written by convert.py's CALIB) is not a generable length (31 latents); the driver generates 129 and keeps 120 (`n_frames_generated` / `n_frames_written` added to calib.json).
- Disk: venv 7.3 GB + weights 11 GB + caches 0.6 GB; /home was at 22 GB free after all agents' downloads (23:50).
