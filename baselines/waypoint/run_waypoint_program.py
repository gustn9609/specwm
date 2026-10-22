"""[10/16 frozen baselines] Waypoint-1.5-1B in ITS setting: official checkpoint (HF Overworld/Waypoint-1.5-1B @ 391f928), official Modular-Diffusers
stack (trust_remote_code modular_blocks.py), bf16, torch.compile(fullgraph=True, mode="max-autotune", dynamic=False) as in the model card, 1024x512
(720p variant), 4 frames per call at 60 fps native (15 calls/s), 4 rectified-flow steps (sigmas 1, .9, .75, .3, 0) + 1 KV-cache pass per call.
One scripted action per call (actions.json from encode_program.py); call 0 seeds the world with the image (stretched to 1024x512 with a BILINEAR
resize like the official Space; the model card's PIL default would be bicubic) and ALSO applies action[0]; later calls pass image=None (REQUIRED: the
modular pipeline keeps a passed input in the state, so without image=None the start image is re-encoded and re-appended to the KV cache at every
call — this is what tools/waypoint/run_waypoint_actions.py did).
Timing per call (cuda-synced): gen = the pipeline call with output_type="latent" (controller encode + denoise + cache pass; call 0 also encodes the
image), dec = vae.decode of that latent to 4 uint8 frames + copy to CPU (identical to the pipeline's own "pil" path). ttff = call 0 (image in -> 4 frames
on CPU). A warm-up rollout (compile + allocator) runs first on the same image and is discarded; the clip itself starts from a fresh KV cache and a reset
streaming VAE. Seed: torch.manual_seed (the pipeline draws its noise from the global RNG; its `generator` input is unused by this checkpoint's blocks).
usage (one clip):  CUDA_VISIBLE_DEVICES=g envrun.sh python run_waypoint_program.py --image IMG --actions A.json --out DIR [--seed 1234] [--no-compile]
                   [--warm 12] [--sweep 60,120,240 --sweep-calls 30]   (extra constant-mouse rollouts for the K_yaw calibration -> DIR/sweep_m<M>.mp4)
usage (batch):     ... run_waypoint_program.py --batch JOBS.json [--warm 12]   JOBS.json = [{"image": IMG, "actions": A.json, "out": DIR, "seed": 1234}, ...]
                   one model load + one compile + ONE warm-up rollout (on job 0), then every job is a fresh world (new KV cache, vae.reset(),
                   torch.manual_seed(job seed)) measured exactly like the single-clip path; a job whose DIR/video.mp4 + timing.json exist is skipped
                   unless --overwrite; a failing job is logged (DIR/FAILED) and the batch continues; --sweep is ignored in batch mode.
writes DIR/video.mp4 (60 fps, libx264 crf 14) + DIR/timing.json (PROTOCOL Section 4 fields + extras) + DIR/run.log per clip"""
import argparse, os, sys, json, time, subprocess, datetime, traceback, gc, numpy as np, torch
from PIL import Image
from diffusers.modular_pipelines import ModularPipeline
from diffusers.utils import load_image
SNAP = "/home/hk28965/.cache/huggingface/hub/models--Overworld--Waypoint-1.5-1B/snapshots/391f92827075edcf4a8b3c8a2ddae010698f8636"
FFMPEG = "/home/hk28965/specwm/tools/ffmpeg-7.0.2-amd64-static/ffmpeg"
ap = argparse.ArgumentParser(); ap.add_argument("--image"); ap.add_argument("--actions"); ap.add_argument("--out")
ap.add_argument("--batch", default=None, help="JSON list of jobs {image, actions, out, seed}; one load/compile/warm-up for all of them")
ap.add_argument("--overwrite", action="store_true", help="batch: redo jobs whose video.mp4 + timing.json already exist")
ap.add_argument("--seed", type=int, default=1234); ap.add_argument("--model", default=SNAP); ap.add_argument("--no-compile", action="store_true"); ap.add_argument("--warm", type=int, default=12)
ap.add_argument("--size", default="1024x512"); ap.add_argument("--fps-out", type=int, default=60); ap.add_argument("--sweep", default="", help="comma list of constant mouse-x magnitudes (units/call) for calibration rollouts")
ap.add_argument("--sweep-calls", type=int, default=30); a = ap.parse_args()
if a.batch:
    JOBS = json.load(open(a.batch)); assert isinstance(JOBS, list) and JOBS, "--batch: a non-empty JSON list is expected"
    for j in JOBS: j.setdefault("seed", a.seed); assert all(k in j for k in ("image", "actions", "out")), "job needs image, actions, out: %s" % j
    LOGDIR = os.path.dirname(os.path.abspath(a.batch))
else:
    assert a.image and a.actions and a.out, "either --batch JOBS.json or --image/--actions/--out"
    JOBS = [dict(image=a.image, actions=a.actions, out=a.out, seed=a.seed)]; LOGDIR = a.out
os.makedirs(LOGDIR, exist_ok=True); sync = torch.cuda.synchronize; W, H = [int(v) for v in a.size.split("x")]
LOG = open(os.path.join(LOGDIR, "batch.log" if a.batch else "run.log"), "a"); CLIPLOG = [None]
def log(*s):
    m = "[%s] " % datetime.datetime.now().strftime("%H:%M:%S") + " ".join(str(x) for x in s); print(m, flush=True); LOG.write(m + "\n"); LOG.flush()
    if CLIPLOG[0] is not None and CLIPLOG[0] is not LOG: CLIPLOG[0].write(m + "\n"); CLIPLOG[0].flush()
def smi():
    g = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    try: return subprocess.run(["nvidia-smi", "-i", g, "--query-gpu=index,name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as e: return "nvidia-smi failed: %s" % e
log("start", vars(a), "jobs %d" % len(JOBS)); log("gpu before:", smi())
t0 = time.time(); pipe = ModularPipeline.from_pretrained(a.model, trust_remote_code=True)
# components loaded explicitly from the frozen snapshot (pipe.load_components resolves the hub id and fails offline); same classes/weights/dtype as
# the model card's load_components(device_map="cuda", torch_dtype=bf16): WorldModel (transformer) + ChunkedStreamingTAEHV (vae), cast to bf16
from diffusers import AutoModel
tr = AutoModel.from_pretrained(a.model, subfolder="transformer", torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda", torch.bfloat16).eval()
vae = AutoModel.from_pretrained(a.model, subfolder="vae", torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda", torch.bfloat16).eval()
pipe.update_components(transformer=tr, vae=vae); pipe.transformer.apply_inference_patches()
log("transformer %.3f B params, vae %.1f M, dtype %s, device %s" % (sum(p.numel() for p in tr.parameters()) / 1e9, sum(p.numel() for p in vae.parameters()) / 1e6, next(tr.parameters()).dtype, next(tr.parameters()).device))
if not a.no_compile: pipe.transformer.compile(fullgraph=True, mode="max-autotune", dynamic=False)
load_s = time.time() - t0; log("loaded in %.1fs (compile is lazy: happens in the warm-up)" % load_s)
emb = torch.zeros(1, 8, 2048, device="cuda", dtype=torch.bfloat16); mask = torch.zeros(1, 8, dtype=torch.bool, device="cuda")   # prompt_conditioning = null: no text encoder, dummy embeds
IMAGE_HANDLING = "PIL resize to %dx%d (stretch to 2:1, BILINEAR, like the official Space; no letterbox) before the pipeline" % (W, H)
def load_start_image(path): return load_image(path).convert("RGB").resize((W, H), Image.BILINEAR)

def rollout(image, actions):
    """fresh world from `image`; returns (frames list of [4,H,W,3] uint8 arrays, per-call (gen_s, dec_s, n_frames), final frame_timestamp)"""
    pipe.vae.reset(); state = None; frames, rec = [], []
    for i, act in enumerate(actions):
        btn = set(int(b) for b in act.get("buttons", [])); mouse = (float(act["mouse"][0]), float(act["mouse"][1]))
        sync(); t1 = time.perf_counter()
        if i == 0: state = pipe(image=image, prompt_embeds=emb, prompt_pad_mask=mask, button=btn, mouse=mouse, output_type="latent")
        else: state = pipe(state, image=None, button=btn, mouse=mouse, output_type="latent")
        sync(); t2 = time.perf_counter()
        lat = state.values["images"]; fr = pipe.vae.decode(lat.squeeze(1)).cpu().numpy(); sync(); t3 = time.perf_counter()
        frames.append(fr); rec.append((t2 - t1, t3 - t2, int(fr.shape[0])))
    ts = state.values.get("frame_timestamp", None) if hasattr(state, "values") else None
    try: ts = int(ts.reshape(-1)[0].item()) if torch.is_tensor(ts) else (int(ts) if ts is not None else -1)
    except Exception: ts = -1
    del state; return frames, rec, ts

def write_video(frames, path, fps):
    arr = np.concatenate(frames, 0); h, w = arr.shape[1:3]
    p = subprocess.run([FFMPEG if os.path.exists(FFMPEG) else "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (w, h), "-r", str(fps), "-i", "-",
                        "-c:v", "libx264", "-preset", "medium", "-crf", "14", "-pix_fmt", "yuv420p", path], input=arr.tobytes(), capture_output=True)
    if p.returncode: log("ffmpeg failed:", p.stderr.decode()[-500:]); np.save(path + ".npy", arr)
    return len(arr)

# warm-up (compile + allocator) on a separate rollout of job 0, discarded; in batch mode this is the ONLY warm-up (all later jobs run on the warm, compiled model)
j0 = JOBS[0]; image0 = load_start_image(j0["image"]); ACTS0 = json.load(open(j0["actions"]))
tw = time.time(); torch.manual_seed(int(j0["seed"])); _, recw, _ = rollout(image0, ACTS0[:max(a.warm, 2)]); warm_s = time.time() - tw
log("warm-up: %d calls in %.1fs (first call %.1fs incl. compile); steady call %.1f ms" % (len(recw), warm_s, recw[0][0] + recw[0][1], 1000 * np.median([r[0] + r[1] for r in recw[1:]])))
WARM = dict(warmup_s=round(warm_s, 1), warmup_calls=len(recw), warmup_first_call_s=round(recw[0][0] + recw[0][1], 1), warmup_image=os.path.abspath(j0["image"]))

def run_clip(job, idx):
    """one clip = PROTOCOL run: fresh world, seeded, timed; writes video.mp4 + timing.json + run.log in job['out']"""
    out = job["out"]; seed = int(job["seed"]); os.makedirs(out, exist_ok=True); CLIPLOG[0] = open(os.path.join(out, "run.log"), "a") if a.batch else None   # single mode: LOG already is out/run.log
    log("clip %d/%d: image %s actions %s seed %d -> %s" % (idx + 1, len(JOBS), job["image"], job["actions"], seed, out))
    image = image0 if (idx == 0 or os.path.abspath(job["image"]) == os.path.abspath(j0["image"])) else load_start_image(job["image"]); ACTS = json.load(open(job["actions"]))
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); sync(); tc = time.time()
    frames, rec, ts_end = rollout(image, ACTS); clip_wall = time.time() - tc
    gen = np.array([r[0] for r in rec]); dec = np.array([r[1] for r in rec]); nf = np.array([r[2] for r in rec]); n_frames = write_video(frames, out + "/video.mp4", a.fps_out)
    meta = {}
    pj = os.path.join(os.path.dirname(os.path.abspath(job["actions"])), "program.json")
    if os.path.exists(pj): meta = json.load(open(pj))
    cfg = pipe.transformer.config
    R = dict(model="Waypoint-1.5-1B (720p variant)", ckpt="Overworld/Waypoint-1.5-1B @ 391f92827075edcf4a8b3c8a2ddae010698f8636 (transformer/diffusion_pytorch_model.safetensors + vae)",
             precision="bf16 weights + activations, no quantization", attention="torch flex_attention with block mask (local_window %d, global_window %d latent frames, period %d), torch.compile(fullgraph, max-autotune, dynamic=False)=%s" % (cfg.local_window, cfg.global_window, cfg.global_attn_period, not a.no_compile),
             vae="taehv1_5 ChunkedStreamingTAEHV bf16 (32 latent ch, 4x temporal / 16x spatial incl. 2x2 pixel-unshuffle), streaming decode", resolution_out="%dx%d" % (W, H), frames=int(n_frames),
             fps_native=60, frames_per_call_or_chunk=4, steps=4, steps_note="4 Euler steps on sigmas %s + 1 KV-cache pass per call (5 transformer forwards)" % list(cfg.scheduler_sigmas),
             gen_s=round(float(gen.sum()), 3), e2e_s=round(float((gen + dec).sum()), 3), ttff_s=round(float(gen[0] + dec[0]), 3),
             steady_fps_gen=round(float(nf[1:].sum() / gen[1:].sum()), 2), steady_fps_e2e=round(float(nf[1:].sum() / (gen[1:] + dec[1:]).sum()), 2),
             peak_alloc_GiB=round(torch.cuda.max_memory_allocated() / 2 ** 30, 3), peak_reserved_GiB=round(torch.cuda.max_memory_reserved() / 2 ** 30, 3),
             text_encoder_resident=False, text_encoder_note="prompt_conditioning = null in this checkpoint: UMT5-XL never loaded, zero prompt embeds passed",
             kv_window="local 16 / global 128 latent frames (4 frames each), max context n_frames %d" % cfg.n_frames, warm=True,
             # extras
             calls=len(rec), call_ms_gen_p50=round(1000 * float(np.median(gen[1:])), 1), call_ms_gen_p99=round(1000 * float(np.percentile(gen[1:], 99)), 1), call_ms_dec_p50=round(1000 * float(np.median(dec[1:])), 1),
             clip_wall_s=round(clip_wall, 2), load_s=round(load_s, 1), compiled=not a.no_compile, seed=seed, image=os.path.abspath(job["image"]), actions=os.path.abspath(job["actions"]),
             program=meta.get("program"), k_yaw=meta.get("k_yaw"), k_yaw_source=meta.get("k_yaw_source"), frame_timestamp_end=ts_end, frame_timestamp_expected=len(rec) + 1,
             image_handling=IMAGE_HANDLING, first_call_action="action[0] applied on the seeding call", first_call_note="call 0 passes the image (VAE-encode + cache pass), calls 1.. pass image=None",
             batch_mode=bool(a.batch), batch_index=idx, batch_size=len(JOBS), batch_note="one model load + compile + warm-up rollout shared by all clips of the batch; each clip is a fresh world (new KV cache, vae.reset, re-seeded)" if a.batch else None,
             gpu=torch.cuda.get_device_name(0), gpu_smi_before=smi(), torch=torch.__version__, diffusers=__import__("diffusers").__version__, date=datetime.datetime.now().isoformat(timespec="seconds"),
             timing_caveat="GPU shared with other jobs during the measurement (see gpu_smi_before/after): fps numbers are a lower bound", **WARM)
    R["gpu_smi_after"] = smi()
    json.dump(R, open(out + "/timing.json", "w"), indent=1); log("[WAYPOINT]", json.dumps({k: R[k] for k in ("frames", "gen_s", "e2e_s", "ttff_s", "steady_fps_gen", "steady_fps_e2e", "peak_alloc_GiB", "peak_reserved_GiB", "frame_timestamp_end", "frame_timestamp_expected")}))
    if ts_end != len(rec) + 1: log("WARNING: frame_timestamp %d != calls+1 %d (image appended more than once?)" % (ts_end, len(rec) + 1))
    if CLIPLOG[0] is not None: CLIPLOG[0].close()
    CLIPLOG[0] = None; del frames; return R

SUMMARY = []; n_ok = n_skip = n_fail = 0
for idx, job in enumerate(JOBS):
    if a.batch and not a.overwrite and os.path.exists(job["out"] + "/video.mp4") and os.path.exists(job["out"] + "/timing.json"):
        log("clip %d/%d: %s exists -> skip (use --overwrite)" % (idx + 1, len(JOBS), job["out"])); n_skip += 1; SUMMARY.append(dict(out=job["out"], status="skipped")); continue
    try:
        R = run_clip(job, idx); n_ok += 1; SUMMARY.append(dict(out=job["out"], status="ok", **{k: R[k] for k in ("frames", "ttff_s", "steady_fps_e2e", "peak_alloc_GiB", "clip_wall_s")}))
    except Exception as e:
        if not a.batch: raise
        n_fail += 1; tb = traceback.format_exc(); log("clip %d/%d FAILED: %s" % (idx + 1, len(JOBS), e)); open(job["out"] + "/FAILED", "w").write(tb); SUMMARY.append(dict(out=job["out"], status="failed", error=str(e)))
        if CLIPLOG[0] is not None: CLIPLOG[0].write(tb); CLIPLOG[0].close(); CLIPLOG[0] = None
        gc.collect(); torch.cuda.empty_cache()
if a.batch:
    json.dump(dict(jobs=len(JOBS), ok=n_ok, skipped=n_skip, failed=n_fail, load_s=round(load_s, 1), **WARM, clips=SUMMARY, date=datetime.datetime.now().isoformat(timespec="seconds")), open(os.path.join(LOGDIR, "batch_summary.json"), "w"), indent=1)
    log("batch done: %d ok, %d skipped, %d failed of %d (load %.1f s + warm-up %.1f s paid once)" % (n_ok, n_skip, n_fail, len(JOBS), load_s, warm_s))

# calibration sweep: constant mouse x for N calls from the same image (not part of the clip); single-clip mode only
if a.sweep and not a.batch:
    sw = {}
    for m in [float(x) for x in a.sweep.split(",")]:
        acts = [dict(mouse=[m, 0.0], buttons=[]) for _ in range(a.sweep_calls)]; torch.manual_seed(a.seed); fr, rc, _ = rollout(image0, acts)
        name = "sweep_m%g" % m; write_video(fr, "%s/%s.mp4" % (a.out, name), a.fps_out); sw[name] = dict(mouse_x_per_call=m, calls=len(rc), frames=int(sum(r[2] for r in rc)), gen_s=round(float(sum(r[0] for r in rc)), 3))
        log("sweep", name, sw[name])
    json.dump(sw, open(a.out + "/sweep.json", "w"), indent=1)
log("done; gpu after:", smi())
if a.batch and n_fail: sys.exit(3)
