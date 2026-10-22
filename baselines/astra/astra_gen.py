#!/usr/bin/env python
"""[10/16 frozen baselines] One Astra clip with the OFFICIAL inference logic of tools/astra/Astra/scripts/infer_demo.py
(inference_moe_framepack_sliding_window, commit 3fb82e1): same model assembly, same FramePack sliding window, same 50-step FlowMatch schedule,
no CFG (the README / infer_demo.sh defaults: use_camera_cfg False, text_guidance_scale 1.0, moe_num_experts 3, top_k 1, modality sekai),
same tiled VAE encode/decode, same imageio 20 fps writer.  Added on top (the official script has none of these): a fixed seed, the PROTOCOL
§2 camera program through program_to_astra.py (the real-pose path of the official code, entry j = motion latent j -> j+1), per-block timing,
TTFF (first block decoded separately, NOT counted in e2e), peak VRAM, and timing.json (PROTOCOL §4 fields).

usage: venv/bin/python astra_gen.py --image IMG --program P2|cam:4 --seed 1234 --out DIR [--prompt "..."] [--latents 56 --block 8 --steps 50 --warm 1]
batch (10/17, one model load for many clips):
       venv/bin/python astra_gen.py --jobs JOBS.json [--latents 56 ...]      JOBS.json = [{"image":..., "program":..., "seed":1234, "out":..., "prompt":"..."}, ...]
       per job the SAME steps as a single run (image encode, prompt encode, generation, one decode, timing.json/log); the warm-up block runs
       once before the first job (its timing.json carries warm_s, later ones warm_s null, warm true); a job whose out/video.mp4 and timing.json
       exist is skipped unless --redo; a failing job is logged (out/ERROR.txt) and the batch continues.
"""
import os, sys, time, json, argparse, random, subprocess, traceback
A = "/home/hk28965/specwm/tools/astra/Astra"
sys.path.insert(0, A); sys.path.insert(0, A + "/scripts"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
import numpy as np, torch, torch.nn as nn, imageio
from PIL import Image, ImageDraw
import infer_demo as ID
from diffsynth import WanVideoAstraPipeline, ModelManager
import program_to_astra as P2A

ASTRA_COMMIT = "3fb82e1939bfcf87d6e27a7da57d8b55ff9cb393"


def sync():
    torch.cuda.synchronize()
    return time.perf_counter()


def gpu_used_mib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu,name", "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        rows = [r for r in out.strip().splitlines() if not vis or r.split(",")[0].strip() in vis.split(",")]
        return rows
    except Exception as e:
        return [str(e)]


def load_pipeline(wan, dit, device, steps):
    """Steps 1-5 of inference_moe_framepack_sliding_window, verbatim semantics (argparse defaults: experts 3, top_k 1)."""
    ID.replace_dit_model_in_manager()
    mm = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    mm.load_models([os.path.join(wan, "diffusion_pytorch_model.safetensors"), os.path.join(wan, "models_t5_umt5-xxl-enc-bf16.pth"), os.path.join(wan, "Wan2.1_VAE.pth")])
    pipe = WanVideoAstraPipeline.from_model_manager(mm, device=device)
    dim = pipe.dit.blocks[0].self_attn.q.weight.shape[0]
    for block in pipe.dit.blocks:
        block.cam_encoder = nn.Linear(13, dim); block.projector = nn.Linear(dim, dim)
        block.cam_encoder.weight.data.zero_(); block.cam_encoder.bias.data.zero_()
        block.projector.weight = nn.Parameter(torch.eye(dim)); block.projector.bias = nn.Parameter(torch.zeros(dim))
    ID.add_framepack_components(pipe.dit)
    moe_config = {"num_experts": 3, "top_k": 1, "hidden_dim": dim * 2, "sekai_input_dim": 13, "nuscenes_input_dim": 8, "openx_input_dim": 13}
    ID.add_moe_components(pipe.dit, moe_config)
    try:
        sd = torch.load(dit, map_location="cpu")                       # torch >= 2.6 default weights_only=True
    except Exception as e:
        print("torch.load weights_only failed (%s) -> weights_only=False" % str(e)[:120], flush=True)
        sd = torch.load(dit, map_location="cpu", weights_only=False)
    if "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    res = pipe.dit.load_state_dict(sd, strict=False)
    print("Astra ckpt: %d tensors; missing %d, unexpected %d" % (len(sd), len(res.missing_keys), len(res.unexpected_keys)), flush=True)
    if res.missing_keys: print("  missing e.g.", res.missing_keys[:5])
    if res.unexpected_keys: print("  unexpected e.g.", res.unexpected_keys[:5])
    pipe = pipe.to(device)                                            # nn.Module.to: text encoder, DiT and VAE all resident on the GPU
    model_dtype = next(pipe.dit.parameters()).dtype
    if hasattr(pipe.dit, "clean_x_embedder"):
        pipe.dit.clean_x_embedder = pipe.dit.clean_x_embedder.to(dtype=model_dtype)
    pipe.scheduler.set_timesteps(steps)
    return pipe, model_dtype, dict(missing=len(res.missing_keys), unexpected=len(res.unexpected_keys), n_tensors=len(sd))


def denoise_block(pipe, history, cur, cam_full, prompt_emb, C, H, W, device, dtype, gen):
    """one FramePack block = the official per-block denoising loop (no CFG branch)"""
    fp = ID.prepare_framepack_sliding_window_with_camera_moe(history, cur, cam_full, "sekai")
    clean_latents = fp["clean_latents"].unsqueeze(0); clean_2x = fp["clean_latents_2x"].unsqueeze(0); clean_4x = fp["clean_latents_4x"].unsqueeze(0)
    cam = fp["camera_embedding"].unsqueeze(0); modality_inputs = {"sekai": cam}
    li = fp["latent_indices"].unsqueeze(0).cpu(); ci = fp["clean_latent_indices"].unsqueeze(0).cpu(); c2i = fp["clean_latent_2x_indices"].unsqueeze(0).cpu(); c4i = fp["clean_latent_4x_indices"].unsqueeze(0).cpu()
    new = torch.randn(1, C, cur, H, W, generator=gen, device=device, dtype=dtype)
    extra = pipe.prepare_extra_input(new)
    with torch.no_grad():
        for timestep in pipe.scheduler.timesteps:
            tt = timestep.unsqueeze(0).to(device, dtype=dtype)
            noise_pred, _ = pipe.dit(new, timestep=tt, cam_emb=cam, modality_inputs=modality_inputs, latent_indices=li, clean_latents=clean_latents,
                                     clean_latent_indices=ci, clean_latents_2x=clean_2x, clean_latent_2x_indices=c2i, clean_latents_4x=clean_4x,
                                     clean_latent_4x_indices=c4i, **prompt_emb, **extra)
            new = pipe.scheduler.step(noise_pred, timestep, new)
    return new.squeeze(0)


def contact_sheet(frames, path, n=10, w=416, h=240, label=""):
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int); cols = 5; rows = (n + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * h), "black"); d = ImageDraw.Draw(sheet)
    for k, i in enumerate(idx):
        im = Image.fromarray(frames[i]).resize((w, h)); sheet.paste(im, ((k % cols) * w, (k // cols) * h))
        d.text(((k % cols) * w + 6, (k // cols) * h + 4), "f%d %.2fs %s" % (i, i / 20.0, label), fill="yellow")
    sheet.save(path, quality=90)


def run_clip(pipe, dtype, ck, load_s, mem_after_load, a, job, warm_state, batch_meta=None):
    """One clip (image, program, seed, out, prompt) on an already loaded pipeline = the body of the original single-clip main().
    warm_state = {'done': bool, 'warm_s': float|None}: the discarded warm-up block runs once per process (before the first clip)."""
    image, program, seed, out, prompt = job["image"], job["program"], int(job["seed"]), job["out"], job.get("prompt", "")
    os.makedirs(out, exist_ok=True); device = "cuda"; fps = P2A.ASTRA_FPS
    n_gen = 1 + P2A.ASTRA_T * a.latents                                 # frames generated (225 for 56 latents)
    n_frames = min(a.frames or n_gen, n_gen)                            # frames kept in video.mp4 (201 = 10.05 s)
    smi_before = gpu_used_mib(); print("GPU before:", smi_before, flush=True)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed); random.seed(seed)

    # ---- camera input: one embedding row per GENERATED latent (+1 for the condition latent). The program is built for all n_gen frames
    # (past 10 s it continues its last segment, convert.program_to_c2w), otherwise the official code zero-pads the missing rows (10/17: with
    # 56 latents and a 201-frame program the pad was 6 rows -> their torch.cat pads on the CPU and crashes at block 7; and a zero 3x4 pose
    # is not 'no motion' anyway).  camera.npz keeps the commanded c2w of the KEPT frames only (the RPE reference).
    if program.startswith("cam:"):
        cam_type = int(program.split(":")[1]); prog = P2A.preset_embedding(cam_type, a.latents); cam_meta = dict(kind="astra_preset", cam_type=cam_type)
        c2w = None
    else:
        d = P2A.build(program, fps, n_gen, u_scale=a.u_scale); prog = d["astra"]; c2w = d["c2w"][:n_frames]
        assert len(prog) == 1 + a.latents, "embedding rows %d != 1 + %d latents" % (len(prog), a.latents)
        cam_meta = dict(kind="protocol_program", program=program, u_scale=float(d["u_scale"]), n_latents=int(d["n_latents"]), fps=fps, frames_generated=n_gen, frames_kept=n_frames,
                        convention="M_j = S [inv(c2w[4j]) c2w[4j+4]] S, S = diag(1,-1,-1); +yaw(protocol) -> M[0,2] < 0 = Astra 'right'; forward -> M[2,3] < 0")
    cam_full = torch.as_tensor(np.asarray(prog, dtype=np.float32)).to(device, dtype=dtype)
    np.savez(os.path.join(out, "camera.npz"), astra=np.asarray(prog, dtype=np.float32), **({} if c2w is None else dict(c2w=c2w, intrinsics=d["intrinsics"])))

    # ---- input: image -> latent (official InlineVideoEncoder: stretch to 832x480, 10 repeats, tiled VAE encode, first latent), prompt -> T5
    torch.cuda.reset_peak_memory_stats()
    t_in0 = sync()
    enc = ID.InlineVideoEncoder(pipe=pipe, device=device)
    frames0 = ID.image_to_frame_stack(Path(image), enc, repeat_count=10)
    latents = enc.encode_frames_to_latents(frames0)                     # [C, T, H, W] cpu
    cond = latents[:, 0:1]
    C, T0, H, W = cond.shape
    if H > 60 or W > 104:
        hs, ws = (H - 60) // 2, (W - 104) // 2; cond = cond[:, :, hs:hs + 60, ws:ws + 104]; H, W = 60, 104
    t_img = sync() - t_in0
    t1 = sync()
    with torch.no_grad():
        prompt_emb = pipe.encode_prompt(prompt)
    t_prompt = sync() - t1
    if a.t5_offload:
        pipe.text_encoder.cpu(); torch.cuda.empty_cache()
    history = cond.to(device, dtype=dtype)
    print("input: image %.2f s, prompt %.2f s (%d chars); latent %s" % (t_img, t_prompt, len(prompt), tuple(history.shape)), flush=True)

    # ---- warm-up (discarded): one full block on the same input, once per process
    warm_s = None
    if a.warm and not warm_state["done"]:
        tw = sync(); g = torch.Generator(device).manual_seed(seed + 1)
        _ = denoise_block(pipe, history.clone(), a.block, cam_full, prompt_emb, C, H, W, device, dtype, g); warm_s = sync() - tw
        warm_state["done"] = True; warm_state["warm_s"] = warm_s
        print("warm-up block %.1f s" % warm_s, flush=True)

    # ---- generation (official sliding window), timed per block
    gen = torch.Generator(device).manual_seed(seed)
    block_s, block_n, out_latents, ttff_dec_s = [], [], [], None
    done = 0
    while done < a.latents:
        cur = min(a.block, a.latents - done)
        tb = sync(); new = denoise_block(pipe, history, cur, cam_full, prompt_emb, C, H, W, device, dtype, gen); tbe = sync()
        block_s.append(tbe - tb); block_n.append(cur)
        history = torch.cat([history, new], dim=1)
        if history.shape[1] > a.max_history:
            history = torch.cat([history[:, 0:1], history[:, -(a.max_history - 1):]], dim=1)
        out_latents.append(new); done += cur
        print("block %d: %d latents in %.2f s (%.3f s/step)" % (len(block_s), cur, block_s[-1], block_s[-1] / a.steps), flush=True)
        if len(block_s) == 1:                                            # TTFF: decode what the user could see after the first block (not in e2e)
            td = sync()
            with torch.no_grad():
                first = pipe.decode_video(torch.cat([cond.to(device, dtype=dtype), new], dim=1).unsqueeze(0), tiled=True, tile_size=(34, 34), tile_stride=(18, 16))
            ttff_dec_s = sync() - td; del first
    gen_s = float(sum(block_s))

    # ---- decode (official: one tiled decode of condition + all generated latents) and write
    td = sync()
    final = torch.cat([cond.to(device, dtype=dtype)] + out_latents, dim=1).unsqueeze(0)
    with torch.no_grad():
        video = pipe.decode_video(final, tiled=True, tile_size=(34, 34), tile_stride=(18, 16))
    dec_s = sync() - td
    v = video[0].to(torch.float32).permute(1, 2, 3, 0).cpu().numpy(); v = ((v * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8)
    v = v[:n_frames]
    with imageio.get_writer(os.path.join(out, "video.mp4"), fps=fps) as wr:
        for f in v: wr.append_data(f)
    contact_sheet(v, os.path.join(out, "sheet.jpg"), label=program)
    del video, final, out_latents, history

    peak_alloc = torch.cuda.max_memory_allocated() / 2**30; peak_res = torch.cuda.max_memory_reserved() / 2**30
    full = [s for s, n in zip(block_s, block_n) if n == a.block]
    steady_gen = full[1:] if len(full) > 1 else full                     # exclude the first block (also the warm-up target) when possible
    fpb = P2A.ASTRA_T * a.block
    steady_fps_gen = fpb / float(np.mean(steady_gen)) if steady_gen else None
    dec_per_block = dec_s * fpb / len(v)
    steady_fps_e2e = fpb / (float(np.mean(steady_gen)) + dec_per_block) if steady_gen else None
    e2e_s = t_img + t_prompt + gen_s + dec_s
    timing = dict(
        model="Astra (Wan2.1-T2V-1.3B + Astra MoE/FramePack DiT)", ckpt="EvanEternal/Astra models/Astra/checkpoints/diffusion_pytorch_model.ckpt sha256 5ca9c5e0..., Wan2.1-T2V-1.3B DiT 96b6b242..., T5 umt5-xxl bf16 7cace0da..., Wan2.1_VAE 38071ab5...",
        commit=ASTRA_COMMIT, precision="bf16 (DiT, T5, VAE; ModelManager torch_dtype bfloat16)", attention="torch F.scaled_dot_product_attention (flash_attn / sageattention not installed; wan_video_dit_moe.flash_attention fallback)",
        vae="Wan2.1_VAE.pth, tiled encode/decode tile (34,34) stride (18,16), one decode of the whole clip at the end (official)",
        resolution_out=[832, 480], frames=int(len(v)), frames_generated=1 + P2A.ASTRA_T * a.latents, latents_generated=a.latents, fps_native=fps, frames_per_call_or_chunk=fpb, latents_per_block=a.block, steps=a.steps, cfg="none (official default)",
        gen_s=gen_s, e2e_s=e2e_s, ttff_s=(t_img + t_prompt + block_s[0] + (ttff_dec_s or 0.0)), ttff_decode_s=ttff_dec_s, ttff_first_block_frames=1 + P2A.ASTRA_T * block_n[0],
        steady_fps_gen=steady_fps_gen, steady_fps_e2e=steady_fps_e2e, decode_s=dec_s, decode_s_per_block_est=dec_per_block,
        image_encode_s=t_img, prompt_encode_s=t_prompt, load_s=load_s, warm_s=warm_s, block_s=block_s, block_latents=block_n, s_per_step_steady=(float(np.mean(steady_gen)) / a.steps if steady_gen else None),
        peak_alloc_GiB=peak_alloc, peak_reserved_GiB=peak_res, mem_after_load_GiB=mem_after_load, text_encoder_resident=(not a.t5_offload),
        kv_window="FramePack context: 1 start + 16 (4x) + 2 (2x) + 1 (1x) history latents per block; history kept up to max_history %d latents" % a.max_history,
        warm=bool(a.warm and warm_state["done"]), seed=seed, program=program, camera=cam_meta, prompt_chars=len(prompt), image=os.path.abspath(image),
        ckpt_load=ck, gpu=torch.cuda.get_device_name(0), gpu_smi_before=smi_before, gpu_smi_after=gpu_used_mib(), torch=torch.__version__,
        notes="timings on a SHARED GPU (see gpu_smi_before: another job resident) are upper bounds; e2e_s = image+prompt encode + all blocks + final decode; ttff_s adds a separate decode of the first block (excluded from e2e_s); steady fps from the full 8-latent blocks after the first",
    )
    if batch_meta is not None:
        timing["batch"] = batch_meta                                      # {jobs_file, index, n_jobs}: one model load shared by the batch (load_s is that load)
    json.dump(timing, open(os.path.join(out, "timing.json"), "w"), indent=1)
    print(json.dumps({k: timing[k] for k in ("frames", "gen_s", "e2e_s", "ttff_s", "steady_fps_gen", "steady_fps_e2e", "peak_alloc_GiB", "peak_reserved_GiB", "load_s")}), flush=True)
    print("GPU after:", gpu_used_mib(), flush=True)
    return timing


class Tee:
    """batch mode: mirror stdout/stderr into the job's log.txt (run.sh tees the whole batch into one log as well)."""
    def __init__(self, path, stream): self.f = open(path, "a"); self.s = stream
    def write(self, x): self.s.write(x); self.f.write(x)
    def flush(self): self.s.flush(); self.f.flush()
    def close(self): self.f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None); ap.add_argument("--prompt", default=""); ap.add_argument("--program", default=None, help="P1..P5 | cam:N")
    ap.add_argument("--seed", type=int, default=1234); ap.add_argument("--out", default=None)
    ap.add_argument("--jobs", default=None, help="batch mode: JSON list of {image, program, seed, out, prompt}; one model load for all")
    ap.add_argument("--redo", action="store_true", help="batch mode: re-generate jobs whose video.mp4 + timing.json already exist")
    ap.add_argument("--latents", type=int, default=56, help="latent frames to generate; keep it a multiple of --block (a partial last block collapses, ENV.md)")
    ap.add_argument("--block", type=int, default=8); ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--frames", type=int, default=None, help="frames kept in video.mp4 (default all = 1 + 4*latents; run.sh passes 201 = 10 s)")
    ap.add_argument("--warm", type=int, default=1); ap.add_argument("--t5-offload", action="store_true"); ap.add_argument("--max-history", type=int, default=100)
    ap.add_argument("--u-scale", type=float, default=None, help="Astra units per program unit per second (default None = Astra's preset forward speed 0.15 = the frozen calib_astra.json value); re-calibration runs only")
    ap.add_argument("--wan", default=A + "/models/Wan-AI/Wan2.1-T2V-1.3B"); ap.add_argument("--dit", default=A + "/models/Astra/checkpoints/diffusion_pytorch_model.ckpt")
    a = ap.parse_args()
    if a.jobs:
        jobs = json.load(open(a.jobs)); assert isinstance(jobs, list) and jobs, "--jobs must be a non-empty JSON list"
        for j in jobs:
            for k in ("image", "program", "out"): assert k in j, "job without %r: %s" % (k, j)
            j.setdefault("seed", a.seed); j.setdefault("prompt", "")
    else:
        assert a.image and a.program and a.out, "--image, --program and --out are required (or --jobs)"
        jobs = [dict(image=a.image, program=a.program, seed=a.seed, out=a.out, prompt=a.prompt)]
    if a.t5_offload and a.jobs:
        print("NOTE: --t5-offload in batch mode moves the T5 to the CPU after the first prompt; later prompts are encoded on the CPU (slow) — the text_encoder_resident flag is per run", flush=True)
    torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    t = time.perf_counter()
    pipe, dtype, ck = load_pipeline(a.wan, a.dit, "cuda", a.steps)
    load_s = sync() - t
    mem_after_load = torch.cuda.memory_allocated() / 2**30
    print("loaded in %.1f s, %.2f GiB allocated (T5 + DiT + VAE resident)" % (load_s, mem_after_load), flush=True)
    warm_state = dict(done=False, warm_s=None); n_ok = n_skip = n_fail = 0; t_batch = time.perf_counter()
    for i, job in enumerate(jobs):
        if a.jobs:
            if not a.redo and os.path.exists(os.path.join(job["out"], "video.mp4")) and os.path.exists(os.path.join(job["out"], "timing.json")):
                print("[batch %d/%d] skip (exists) %s" % (i + 1, len(jobs), job["out"]), flush=True); n_skip += 1; continue
            print("[batch %d/%d] %s %s seed %s -> %s" % (i + 1, len(jobs), job["image"], job["program"], job["seed"], job["out"]), flush=True)
            os.makedirs(job["out"], exist_ok=True); tee = Tee(os.path.join(job["out"], "log.txt"), sys.stdout); sys.stdout = tee
            if a.t5_offload and i > 0: pipe.text_encoder.to("cuda")      # keep per-job behaviour identical to a single run (encode on the GPU, then offload)
            try:
                run_clip(pipe, dtype, ck, load_s, mem_after_load, a, job, warm_state, batch_meta=dict(jobs_file=os.path.abspath(a.jobs), index=i, n_jobs=len(jobs))); n_ok += 1
            except Exception as e:
                n_fail += 1; msg = "JOB FAILED %s: %s\n%s" % (job["out"], e, traceback.format_exc()); print(msg, flush=True)
                open(os.path.join(job["out"], "ERROR.txt"), "w").write(msg); torch.cuda.empty_cache()
            finally:
                sys.stdout = tee.s; tee.close()
            print("[batch %d/%d] done, %.0f s since batch start" % (i + 1, len(jobs), time.perf_counter() - t_batch), flush=True)
        else:
            run_clip(pipe, dtype, ck, load_s, mem_after_load, a, job, warm_state)
    if a.jobs:
        print("BATCH DONE: %d ok, %d skipped, %d failed, %d jobs in %.0f s (load %.1f s, peak reserved %.2f GiB)" % (n_ok, n_skip, n_fail, len(jobs), time.perf_counter() - t_batch, load_s, torch.cuda.max_memory_reserved() / 2**30), flush=True)
        sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
