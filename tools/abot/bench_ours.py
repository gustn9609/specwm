"""[10/15 baseline] ABot-World-0-5B-LF timed streaming run with our inputs: reference image + text prompt + per-block keyboard actions
(8-dim multi-hot W,A,S,D,I,J,K,L), chunk = 3 latent frames = 12 decoded frames (ABot's native 12 fps → 1.0 s of video per block).
Records per block: DiT time (generate_next_block, CUDA-synchronised), VAE decode time, and action-to-first-decoded-frame latency (TTFF);
steady fps = 12 / median(DiT + VAE) over blocks 3..; peak memory. Output: <out>/video.mp4 + timing.json.
usage (inside tools/abot via run.sh): CUDA_VISIBLE_DEVICES=g tools/abot/run.sh python bench_ours.py --image IMG --prompt "..." --out DIR
       [--blocks 13 --height 704 --width 1280 --quant fp8-per-tensor|none --vae taew2_2|wan2.2 --actions '[[1,0,0,0,0,0,0,0],...]' --ref-dir DIR]
"""
import argparse, json, os, sys, time
from pathlib import Path
_ROOT = Path(__file__).resolve().parent; sys.path.insert(0, str(_ROOT)); os.chdir(_ROOT); os.environ.setdefault("PROJECT_ROOT", str(_ROOT))
import imageio, torch
from web_client.config import KEY_ORDER, VIDEO_FPS
from web_client.pipeline_loader import decode_block_to_frames, get_pipeline
ap = argparse.ArgumentParser(); ap.add_argument("--image", required=True); ap.add_argument("--prompt", default="| unknown |"); ap.add_argument("--out", required=True)
ap.add_argument("--blocks", type=int, default=13); ap.add_argument("--height", type=int, default=704); ap.add_argument("--width", type=int, default=1280)
ap.add_argument("--quant", default="fp8-per-tensor"); ap.add_argument("--vae", default=None); ap.add_argument("--actions", default=None, help="JSON list of per-block 8-dim lists; default = W (forward) every block")
ap.add_argument("--ref-dir", default=str(_ROOT / "outputs" / "ref_image_cache" / "none"), help="character reference dir (5 views); an empty dir = no character memory")
ap.add_argument("--fps", type=int, default=VIDEO_FPS, help="fps written into the mp4 (ABot native 12)")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); os.makedirs(a.ref_dir, exist_ok=True)
pipeline, cfg, device = get_pipeline(vae_type=a.vae, use_fp8_gemm=a.quant != "none", quant_type=a.quant if a.quant != "none" else None)
dtype = torch.bfloat16; nfpb = int(pipeline.num_frame_per_block); fpb = nfpb * 4
vae_s = pipeline.encoder if pipeline.encoder is not None else pipeline.vae; up = getattr(vae_s, "upsampling_factor", 8)
lat_shape = (1, nfpb, vae_s.z_dim, a.height // up, a.width // up)
ps = pipeline.generator.model.patch_size   # the pipeline sizes its KV cache from the config's 1280x704 grid; re-derive tokens per frame for our resolution
pipeline.frame_seq_length = (lat_shape[3] * lat_shape[4]) // (ps[1] * ps[2]); print("[bench] frame_seq_length", pipeline.frame_seq_length, "latent", lat_shape, flush=True)
acts = json.loads(a.actions) if a.actions else [[1, 0, 0, 0, 0, 0, 0, 0]] * a.blocks
acts = (acts + [acts[-1]] * a.blocks)[:a.blocks]
def sync(): torch.cuda.synchronize()
sync(); t_setup0 = time.perf_counter()
pipeline.set_prompts([a.prompt], device=device); pipeline.set_ref_latent_mask_from_exists_paths(ref_dir=a.ref_dir, device=device)
pipeline.reset_stream(1, dtype=dtype, device=device, initial_latent=None); torch.cuda.reset_peak_memory_stats(); sync(); t_setup = time.perf_counter() - t_setup0
frames, rows = [], []; t_start = time.perf_counter()
for b in range(a.blocks):
    noise = torch.randn(lat_shape, device=device, dtype=dtype)
    if b == 0: pipeline.set_first_frame_latent(a.image, height=a.height, width=a.width, device=device)
    pipeline.set_act({k: bool(v) for k, v in zip(KEY_ORDER, acts[b])}, height=a.height, width=a.width, num_frames=nfpb, device=device)
    sync(); t0 = time.perf_counter(); pred = pipeline.generate_next_block(noise); sync(); t1 = time.perf_counter()
    fr = decode_block_to_frames(pipeline, pred) if pred is not None else []; sync(); t2 = time.perf_counter()
    frames.extend(fr); rows.append(dict(block=b, dit_s=t1 - t0, vae_s=t2 - t1, total_s=t2 - t0, n_frames=len(fr), ttff_s=(t2 - t_start) if b == 0 else None))
    print("[block %2d] dit %.3f s  vae %.3f s  frames %d" % (b, t1 - t0, t2 - t1, len(fr)), flush=True)
import statistics
st = [r for r in rows if r["block"] >= 3] or rows; med = statistics.median(r["total_s"] for r in st); med_dit = statistics.median(r["dit_s"] for r in st)
res = dict(image=a.image, prompt=a.prompt, height=a.height, width=a.width, quant=a.quant, vae=a.vae or str(cfg.vae_type), blocks=a.blocks, frames=len(frames), frames_per_block=fpb,
           video_fps_native=VIDEO_FPS, setup_s=t_setup, ttff_s=rows[0]["ttff_s"], median_block_s=med, median_dit_s=med_dit, steady_fps=fpb / med, dit_only_fps=fpb / med_dit,
           realtime_factor=(fpb / VIDEO_FPS) / med, peak_alloc_gib=torch.cuda.max_memory_allocated() / 2**30, peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30, rows=rows)
json.dump(res, open(os.path.join(a.out, "timing.json"), "w"), indent=1)
if frames:
    w = imageio.get_writer(os.path.join(a.out, "video.mp4"), fps=a.fps, format="FFMPEG", codec="libx264", ffmpeg_params=["-crf", "18", "-preset", "fast", "-pix_fmt", "yuv420p"])
    for f in frames: w.append_data(f)
    w.close()
print("[ABOT_BENCH_DONE] %dx%d %s: TTFF %.2f s, median block %.3f s (dit %.3f) = %.1f fps shown (%.2fx real time at %d fps), peak %.1f GiB" % (
    a.width, a.height, a.quant, res["ttff_s"], med, med_dit, res["steady_fps"], res["realtime_factor"], VIDEO_FPS, res["peak_alloc_gib"]), flush=True)
