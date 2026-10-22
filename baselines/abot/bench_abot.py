"""[10/16 frozen baselines; 10/17 fixes] ABot-World-0-5B-LF timed streaming run in ITS setting: start image + (optional) caption + per-block 8-key
actions [W,A,S,D,I,J,K,L] from a PROTOCOL control program, native 1280x704 @ 12 fps, 4-step DMD schedule, FP8 GEMM + taew2_2 streaming VAE.
Copy of tools/abot/bench_ours.py (untouched) extended with: --program (keys from baselines/abot/encode_abot.py), --offload-t5 (official DynamicSwap
T5 offload = the 19 GiB footprint; default off = T5 resident), an untimed warm-up stream (warm=true; the cold first-block time is kept as
cold_ttff_s), a fixed seed, a 10 s trim (11 blocks = 9 + 10x12 = 129 frames -> first 120 frames = 10.0 s written), and every timing.json field of
PROTOCOL.md Section 4. Runs inside the ABot venv (tools/abot/run.sh) and chdir's into tools/abot itself.
10/17 fixes: (1) every stream resets the streaming VAE exactly as the web client does after a stream (web_client/inference.py L358-363:
pipeline.vae.model.clear_cache() -> StreamingTAEHV.reset()); without it the warm-up's taew2_2 MemBlock memory / n_frames_decoded leaked into the
measured clip (frames 0-3 ghosted, block 0 decoded 12 frames instead of 9, 132 generated). (2) The frame layout is asserted: block 0 = 9 frames,
9 + 12 x (blocks-1) generated (129 for 11 blocks), --keep-frames written; a failing clip is written as video_failed_check.mp4 and the run exits 1.
(3) The quant arm is recorded (timing.json quant + quant_decision). (4) --batch JOBS.json runs many (image, program) pairs with ONE model load.
usage (single): CUDA_VISIBLE_DEVICES=g /home/hk28965/specwm/tools/abot/run.sh python /home/hk28965/specwm/baselines/abot/bench_abot.py --image IMG --program P2
       --seed 1234 --out DIR [--prompt "caption"] [--offload-t5 off|on|auto] [--quant fp8-per-token|fp8-per-tensor|none] [--blocks 11 --keep-frames 120]
       [--warmup-blocks 2] [--actions '[[1,0,0,0,0,0,0,0],...]' (overrides --program)]
usage (batch): ... bench_abot.py --batch JOBS.json [same options as defaults]   JOBS.json = [{"image": IMG, "program": "P2", "seed": 1234, "out": DIR,
       "prompt": "caption" (optional; default = IMG's sibling .txt, empty if none), "actions": [[...]] (optional, overrides program)}, ...]
       (baselines/abot/make_jobs.py writes it). One warm-up stream precedes the first job; later jobs are warm from the previous one. Each job gets
       its own video.mp4 / timing.json / log.txt; the process exits 1 if any job failed (the others still run).
"""
import argparse, json, os, sys, time, statistics, subprocess, traceback
from pathlib import Path
ABOT = Path("/home/hk28965/specwm/tools/abot"); HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ABOT)); sys.path.insert(0, str(HERE)); os.chdir(ABOT); os.environ.setdefault("PROJECT_ROOT", str(ABOT))
import imageio, torch, numpy as np
from web_client.config import KEY_ORDER, VIDEO_FPS, FIRST_BLOCK_FRAMES, FRAMES_PER_BLOCK
from web_client import pipeline_loader as PL
from web_client.pipeline_loader import decode_block_to_frames
from encode_abot import program_to_c2w, encode_abot, frames_for_blocks, KEYS

CAPTION_PREFIX = "| unknown |"   # web_client/app.py prepends this to every caption (and sends it alone when the caption is empty)
QUANT_DECISION = {"fp8-per-token": "FROZEN ARM: the official default quant_type (tools/abot/configs/default_config.yaml, what the web client loads); it runs on the RTX PRO 6000 (10/16 GPU-6 smoke, 10/17 GPU-4 runs), so the fallback fp8-per-tensor is not used",
                  "fp8-per-tensor": "fallback arm (the 10/15 setting); only to be used if fp8-per-token fails to run", "none": "bf16, not a frozen arm"}
ap = argparse.ArgumentParser()
ap.add_argument("--image", default=None, help="start image (required unless --batch)"); ap.add_argument("--prompt", default="", help="caption (may be empty); the web client's '| unknown |' prefix is added")
ap.add_argument("--out", default=None, help="output dir (required unless --batch)"); ap.add_argument("--program", default=None, help="P1..P5 -> per-block keys (PROTOCOL Section 2)")
ap.add_argument("--actions", default=None, help="JSON list of per-block 8-dim lists [W,A,S,D,I,J,K,L]; overrides --program; default = W every block")
ap.add_argument("--batch", default=None, help="JSON list of jobs [{image, program|actions, seed, out, prompt?}]: all run with ONE model load (see module docstring)")
ap.add_argument("--blocks", type=int, default=11, help="11 blocks = 9 + 10 x 12 = 129 decoded frames (10.75 s), trimmed to --keep-frames")
ap.add_argument("--keep-frames", type=int, default=120, help="frames written to video.mp4 (120 = 10 s at 12 fps); 0 = keep all")
ap.add_argument("--height", type=int, default=704); ap.add_argument("--width", type=int, default=1280)
ap.add_argument("--quant", default="fp8-per-token", help="official default (configs/default_config.yaml quant_type) = the frozen arm; fp8-per-tensor = the 10/15 setting (fallback); none = bf16")
ap.add_argument("--vae", default=None, help="default = config vae_type (taew2_2)")
ap.add_argument("--offload-t5", default="off", choices=["off", "on", "auto"], help="off = umT5-XXL resident on GPU; on = DynamicSwap offload (official low-memory path, ~19 GiB); auto = official rule (offload iff free VRAM < 40 GB at load)")
ap.add_argument("--warmup-blocks", type=int, default=2, help="untimed warm-up stream on the first image before the measured clip (0 = cold measurement)")
ap.add_argument("--seed", type=int, default=1234); ap.add_argument("--fps", type=int, default=VIDEO_FPS, help="fps written into the mp4 (native 12)")
ap.add_argument("--ref-dir", default=str(ABOT / "outputs" / "ref_image_cache" / "none"), help="character reference dir (5 views); empty dir = no character memory (as the official bench)")
ap.add_argument("--no-prefix", action="store_true", help="send --prompt verbatim without the '| unknown |' prefix")
ap.add_argument("--no-vae-reset", action="store_true", help="debug only: reproduce the 10/16 leak (no streaming-VAE reset between streams)")
a = ap.parse_args(); os.makedirs(a.ref_dir, exist_ok=True)

def sibling_caption(img):
    t = os.path.splitext(img)[0] + ".txt"
    return open(t).read().strip() if os.path.exists(t) else ""
if a.batch:
    jobs = json.load(open(a.batch)); assert isinstance(jobs, list) and jobs, "--batch: a non-empty JSON list of jobs"
    for j in jobs:
        assert j.get("image") and j.get("out"), "every job needs image and out: %r" % j
        j.setdefault("program", a.program); j.setdefault("actions", a.actions); j.setdefault("seed", a.seed)
        if "prompt" not in j or j["prompt"] is None: j["prompt"] = sibling_caption(j["image"])
else:
    assert a.image and a.out, "--image and --out are required (or --batch)"
    jobs = [dict(image=a.image, out=a.out, program=a.program, actions=a.actions, seed=a.seed, prompt=a.prompt)]
for j in jobs: os.makedirs(j["out"], exist_ok=True)

def full_prompt(cap): return (cap or "").strip() if a.no_prefix else (CAPTION_PREFIX + (" " + cap.strip() if (cap or "").strip() else ""))
def job_actions(j):
    if j.get("actions"): acts = json.loads(j["actions"]) if isinstance(j["actions"], str) else list(j["actions"])
    elif j.get("program"): acts = encode_abot(program_to_c2w(j["program"], VIDEO_FPS, n_gen), FIRST_BLOCK_FRAMES, FRAMES_PER_BLOCK, n_blocks=a.blocks)
    else: acts = [[1, 0, 0, 0, 0, 0, 0, 0]] * a.blocks
    return (acts + [acts[-1]] * a.blocks)[:a.blocks]
def keys_str(acts): return " ".join("".join(k for k, v in zip(KEYS, b) if v) or "-" for b in acts)
n_gen = frames_for_blocks(a.blocks, FIRST_BLOCK_FRAMES, FRAMES_PER_BLOCK)
for i, j in enumerate(jobs):
    j["_prompt"] = full_prompt(j["prompt"]); j["_acts"] = job_actions(j)
    print("[bench] job %d/%d image=%s prompt=%r program=%s seed=%s blocks=%d gen_frames=%d keep=%d keys=%s -> %s" % (i + 1, len(jobs), j["image"], j["_prompt"], j["program"], j["seed"], a.blocks, n_gen, a.keep_frames, keys_str(j["_acts"]), j["out"]), flush=True)

# T5 placement: pipeline_loader decides by get_cuda_free_memory_gb() < LOW_MEMORY_THRESHOLD_GB (40). Patch the module-level name (not the file).
if a.offload_t5 == "off": PL.get_cuda_free_memory_gb = lambda *x, **k: 1e9
elif a.offload_t5 == "on": PL.get_cuda_free_memory_gb = lambda *x, **k: 0.0
from wan.modules import attention as ATTN

def sync(): torch.cuda.synchronize()
t_load0 = time.perf_counter()
pipeline, cfg, device = PL.get_pipeline(vae_type=a.vae, use_fp8_gemm=a.quant != "none", quant_type=a.quant if a.quant != "none" else None)
sync(); load_s = time.perf_counter() - t_load0
dtype = torch.bfloat16; nfpb = int(pipeline.num_frame_per_block); fpb = nfpb * 4
vae_s = pipeline.encoder if pipeline.encoder is not None else pipeline.vae; up = getattr(vae_s, "upsampling_factor", 8)
lat_shape = (1, nfpb, vae_s.z_dim, a.height // up, a.width // up); ps = pipeline.generator.model.patch_size
pipeline.frame_seq_length = (lat_shape[3] * lat_shape[4]) // (ps[1] * ps[2])   # tokens per latent frame at this resolution (bench_ours.py)
t5_devs = sorted({p.device.type for p in pipeline.text_encoder.parameters()}); t5_resident = t5_devs == ["cuda"]
print("[bench] loaded in %.1f s; attention backend %s; T5 param devices %s (resident=%s); frame_seq_length %d; latent %s" % (load_s, ATTN.DEFAULT_ATTN_BACKEND, t5_devs, t5_resident, pipeline.frame_seq_length, lat_shape), flush=True)

def reset_vae_stream():
    """The web client's own per-stream VAE reset (tools/abot/web_client/inference.py L358-363, run after every stream). taew2_2's StreamingTAEHV keeps
    its MemBlock memory, work queue and n_frames_decoded across streams; pipeline.reset_stream() only resets the DiT KV cache. Without this reset a
    second stream in the same process (our measured clip after the warm-up) starts with the previous stream's decoder memory and skips the 3
    start-up frames it must trim (block 0 -> 12 frames, ghosted frames 0-3)."""
    if a.no_vae_reset: return "none (--no-vae-reset)"
    v = pipeline.vae
    if hasattr(v, "model") and hasattr(v.model, "clear_cache"): v.model.clear_cache(); return "pipeline.vae.model.clear_cache()"
    if hasattr(v, "taehv") and hasattr(v.taehv, "reset"): v.taehv.reset(); return "pipeline.vae.taehv.reset()"
    return "none (no clear_cache/reset on %s)" % type(v).__name__

def run_stream(n_blocks, acts, seed, tag, image, prompt):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats(); sync(); t0 = time.perf_counter()
    pipeline.set_prompts([prompt], device=device); sync(); t5_s = time.perf_counter() - t0
    pipeline.set_ref_latent_mask_from_exists_paths(ref_dir=a.ref_dir, device=device)
    pipeline.reset_stream(1, dtype=dtype, device=device, initial_latent=None); vae_reset = reset_vae_stream(); sync(); setup_s = time.perf_counter() - t0
    frames, rows = [], []; t_start = time.perf_counter()
    for b in range(n_blocks):
        noise = torch.randn(lat_shape, device=device, dtype=dtype)
        if b == 0: pipeline.set_first_frame_latent(image, height=a.height, width=a.width, device=device)
        pipeline.set_act({k: bool(v) for k, v in zip(KEY_ORDER, acts[b])}, height=a.height, width=a.width, num_frames=nfpb, device=device)
        sync(); t1 = time.perf_counter(); pred = pipeline.generate_next_block(noise); sync(); t2 = time.perf_counter()
        fr = decode_block_to_frames(pipeline, pred) if pred is not None else []; sync(); t3 = time.perf_counter()
        frames.extend(fr); rows.append(dict(block=b, keys="".join(k for k, v in zip(KEYS, acts[b]) if v), dit_s=t2 - t1, vae_s=t3 - t2, total_s=t3 - t1, pre_s=t1 - (t_start if b == 0 else rows[-1]["t_end"]), n_frames=len(fr), t_end=t3, ttff_s=(t3 - t_start) if b == 0 else None))
        print("[%s block %2d %-4s] dit %.3f s  vae %.3f s  frames %d" % (tag, b, rows[-1]["keys"] or "-", t2 - t1, t3 - t2, len(fr)), flush=True)
    e2e = time.perf_counter() - t_start
    for r in rows: r.pop("t_end")
    return dict(frames=frames, rows=rows, t5_s=t5_s, setup_s=setup_s, e2e_s=e2e, vae_reset=vae_reset, peak_alloc_GiB=torch.cuda.max_memory_allocated() / 2**30, peak_reserved_GiB=torch.cuda.max_memory_reserved() / 2**30)

def frame_problems(rows, frames, keep_n):
    """the asserted frame layout: block 0 = FIRST_BLOCK_FRAMES (9), every later block FRAMES_PER_BLOCK (12), 9 + 12 x (blocks-1) generated, keep_n written"""
    p = []
    if rows[0]["n_frames"] != FIRST_BLOCK_FRAMES: p.append("block 0 decoded %d frames, expected %d (streaming-VAE state leaked from a previous stream?)" % (rows[0]["n_frames"], FIRST_BLOCK_FRAMES))
    bad = [r["block"] for r in rows[1:] if r["n_frames"] != FRAMES_PER_BLOCK]
    if bad: p.append("blocks %s decoded != %d frames" % (bad, FRAMES_PER_BLOCK))
    exp = frames_for_blocks(len(rows), FIRST_BLOCK_FRAMES, FRAMES_PER_BLOCK)
    if len(frames) != exp: p.append("%d frames generated, expected %d for %d blocks" % (len(frames), exp, len(rows)))
    if keep_n and len(frames) < keep_n: p.append("only %d frames for --keep-frames %d" % (len(frames), keep_n))
    return p

try: hf_rev = open(next(Path("/home/hk28965/specwm/hf/ABot-World-0-5B-LF/.cache/huggingface/download").glob("config.json.metadata"))).read().split()[0]
except Exception: hf_rev = "unknown"
try: commit = subprocess.run(["git", "-C", str(ABOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
except Exception: commit = "unknown"

class Tee:
    """batch mode: mirror everything printed during a job into that job's log.txt (single mode: run.sh tees the whole stdout itself)"""
    def __init__(self, path, base): self.f = open(path, "a"); self.base = base
    def write(self, s): self.base.write(s); self.f.write(s)
    def flush(self): self.base.flush(); self.f.flush()
    def close(self): self.f.close()

warm = None; failures = []; summaries = []
for ji, j in enumerate(jobs):
    tee = None
    if a.batch: tee = Tee(os.path.join(j["out"], "log.txt"), sys.stdout); sys.stdout = tee; print("[bench] batch job %d/%d %s" % (ji + 1, len(jobs), j["out"]), flush=True)
    try:
        if ji == 0 and a.warmup_blocks > 0:
            warm = run_stream(a.warmup_blocks, j["_acts"], int(j["seed"]) + 1, "warm", j["image"], j["_prompt"]); warm.pop("frames")
            print("[bench] warm-up done: cold TTFF %.2f s, peak %.1f GiB, block-0 frames %d" % (warm["rows"][0]["ttff_s"], warm["peak_alloc_GiB"], warm["rows"][0]["n_frames"]), flush=True)
        m = run_stream(a.blocks, j["_acts"], int(j["seed"]), "run", j["image"], j["_prompt"])
        frames = m["frames"]; rows = m["rows"]; keep = frames[:a.keep_frames] if a.keep_frames > 0 else frames; problems = frame_problems(rows, frames, a.keep_frames)
        st = [r for r in rows if r["block"] >= 3] or rows[1:] or rows
        med_tot = statistics.median(r["total_s"] for r in st); med_dit = statistics.median(r["dit_s"] for r in st)
        res = dict(model="ABot-World-0-5B-LF", ckpt="hf acvlab/ABot-World-0-5B-LF rev %s (diffusion_pytorch_model.safetensors, Wan2.2-TI2V-5B based) via tools/abot commit %s" % (hf_rev, commit),
                   precision="bf16 weights/activations; FP8 GEMM (%s) on the DiT linear layers (quantizer/apply_fp8_quantization, lightx2v_kernel); umT5-XXL bf16; VAE bf16" % a.quant,
                   attention="%s (priority %s; sage=%s fa3=%s fa2=%s); causal local attention local_attn_size=%d latent frames" % (ATTN.DEFAULT_ATTN_BACKEND, ATTN.ATTN_BACKEND_PRIORITY, ATTN.SAGE_ATTN_AVAILABLE, ATTN.FLASH_ATTN_3_AVAILABLE, ATTN.FLASH_ATTN_2_AVAILABLE, int(pipeline.local_attn_size)),
                   vae="%s (decode: %s; first-frame encode: %s); streaming-decoder reset per stream: %s" % (a.vae or str(cfg.vae_type), type(pipeline.vae).__name__, type(pipeline.encoder).__name__ if pipeline.encoder is not None else type(pipeline.vae).__name__, m["vae_reset"]),
                   resolution_out="%dx%d" % (a.width, a.height), frames=len(keep), frames_generated=len(frames), fps_native=VIDEO_FPS, clip_s=len(keep) / VIDEO_FPS,
                   frames_per_call_or_chunk="%d (block = %d latent frames; first block %d frames incl. the start image)" % (fpb, nfpb, FIRST_BLOCK_FRAMES), blocks=a.blocks, frames_per_block=[r["n_frames"] for r in rows],
                   frame_check_ok=not problems, frame_check="block 0 = %d frames, %d generated, %d written (expected %d / %d / %d)" % (rows[0]["n_frames"], len(frames), len(keep), FIRST_BLOCK_FRAMES, n_gen, a.keep_frames or n_gen) + ("" if not problems else " FAILED: " + "; ".join(problems)),
                   steps=len(list(cfg.denoising_step_list)), denoising_step_list=list(map(int, cfg.denoising_step_list)), timestep_shift=float(cfg.model_kwargs.timestep_shift),
                   gen_s=sum(r["dit_s"] for r in rows), decode_s=sum(r["vae_s"] for r in rows), e2e_s=m["e2e_s"], ttff_s=rows[0]["ttff_s"], cold_ttff_s=warm["rows"][0]["ttff_s"] if warm else None,
                   steady_fps_gen=fpb / med_dit, steady_fps_e2e=fpb / med_tot, median_block_s=med_tot, median_dit_s=med_dit, realtime_factor=(fpb / VIDEO_FPS) / med_tot,
                   peak_alloc_GiB=m["peak_alloc_GiB"], peak_reserved_GiB=m["peak_reserved_GiB"], warm_peak_alloc_GiB=warm["peak_alloc_GiB"] if warm else None,
                   text_encoder_resident=t5_resident, text_encoder="umT5-XXL (models_t5_umt5-xxl-enc-bf16.pth), offload mode %s, encode %.2f s" % (a.offload_t5, m["t5_s"]),
                   kv_window="local_attn_size %d latent frames = %d blocks = %d decoded frames (%.1f s); KV cache sized %d tokens" % (int(pipeline.local_attn_size), int(pipeline.local_attn_size) // nfpb, int(pipeline.local_attn_size) * 4, int(pipeline.local_attn_size) * 4 / VIDEO_FPS, int(pipeline.local_attn_size) * pipeline.frame_seq_length),
                   warm=(a.warmup_blocks > 0) or ji > 0, warmup_blocks=a.warmup_blocks, warmed_by=("warm-up stream (%d blocks, same image)" % a.warmup_blocks if ji == 0 and a.warmup_blocks > 0 else ("previous batch job %s" % jobs[ji - 1]["out"] if ji > 0 else "nothing (cold)")),
                   load_s=load_s, model_loads_in_process=1, batch=dict(file=a.batch, index=ji, size=len(jobs)) if a.batch else None,
                   setup_s=m["setup_s"], t5_s=m["t5_s"], seed=int(j["seed"]), gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                   image=os.path.abspath(j["image"]), caption=j["prompt"], prompt_sent=j["_prompt"], program=j["program"], actions=j["_acts"], quant=a.quant, quant_decision=QUANT_DECISION.get(a.quant, a.quant),
                   ref_dir=a.ref_dir, rows=rows, warm_rows=warm["rows"] if (warm and ji == 0) else None)
        if keep:
            name = "video.mp4" if not problems else "video_failed_check.mp4"
            t_w = time.perf_counter(); w = imageio.get_writer(os.path.join(j["out"], name), fps=a.fps, format="FFMPEG", codec="libx264", ffmpeg_params=["-crf", "18", "-preset", "fast", "-pix_fmt", "yuv420p"])
            for f in keep: w.append_data(f)
            w.close(); res["write_s"] = time.perf_counter() - t_w; res["video"] = name
            if problems and os.path.exists(os.path.join(j["out"], "video.mp4")): os.remove(os.path.join(j["out"], "video.mp4"))   # never leave a stale valid-looking clip next to a failed one
        json.dump(res, open(os.path.join(j["out"], "timing.json"), "w"), indent=1)
        line = "[ABOT_BENCH_DONE] %dx%d %s T5-resident=%s: TTFF %.2f s (cold %s), median block %.3f s (dit %.3f) = %.1f fps e2e / %.1f fps gen (%.2fx real time), %d frames kept of %d (block 0 = %d), peak %.1f GiB alloc / %.1f reserved -> %s" % (
            a.width, a.height, a.quant, t5_resident, res["ttff_s"], "%.2f s" % res["cold_ttff_s"] if warm else "n/a", med_tot, med_dit, res["steady_fps_e2e"], res["steady_fps_gen"], res["realtime_factor"], len(keep), len(frames), rows[0]["n_frames"], res["peak_alloc_GiB"], res["peak_reserved_GiB"], j["out"])
        print(line, flush=True); summaries.append(line)
        if problems: raise AssertionError("frame layout check FAILED for %s: %s" % (j["out"], "; ".join(problems)))
    except Exception as e:
        failures.append((j["out"], repr(e))); print("[ABOT_BENCH_FAILED] %s: %r" % (j["out"], e), flush=True); traceback.print_exc()
        if not a.batch: raise
    finally:
        if tee is not None: sys.stdout = tee.base; tee.close()
if a.batch:
    print("[bench] batch done: %d/%d jobs ok, 1 model load (%.1f s)" % (len(jobs) - len(failures), len(jobs), load_s), flush=True)
    for s in summaries: print(s)
    for o, e in failures: print("[bench] FAILED %s: %s" % (o, e))
    sys.exit(1 if failures else 0)
