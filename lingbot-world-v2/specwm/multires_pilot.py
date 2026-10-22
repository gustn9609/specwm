"""[10/20] Multi-resolution denoising steps — training-free pilot (advisor's idea; go/no-go before a fine-tune).
Idea: in the 14B 4-step schedule (sigma 1 -> 0.9375 -> 0.833 -> 0.625 = nodes 0/250/500/750) the first step(s) only fix the coarse layout, so run
them at HALF resolution (latent grid ~h/2 x w/2 = 1/4 of the tokens, ~1/4 of a forward) and only the last step(s) at full resolution.
  x0 route : the low-res step predicts x0_lo = z_lo - sigma * flow; x0_lo is bilinear-upsampled to the full grid and re-noised to the next node's
             sigma with the SAME full-res noise draw the 4-step reference uses (act_spec2 / inject_probe convention: initial noise
             Generator(seed*1009 + c*7), re-noise draws Generator(seed*1009 + c*7 + 1)). Low-res noise = block sum of the full-res draw divided by
             sqrt(block size): unit variance, independent across low-res pixels, exactly 2*avgpool when the grid is even (the reference grids are
             58x106 / 56x108 / 58x104, so the half grids are 28x52 / 28x54 / 28x52 with 2- or 3-row blocks).
  context  : the low-res forward attends to a LOW-RES K/V cache (window 18, sink 6, ring) rebuilt from the committed chunks: every committed full-res
             x0 is bilinear(antialias)-downsampled and committed through a clean t=0 pass into the low-res cache (one extra low-res commit per
             chunk = the overhead). The full-res steps and the full-res commit use the normal cache. Two complete binding states (y / plucker /
             ctx / seq_len / caches / cross-attn flags) are kept and swapped before each forward (V2Binding attributes, see STATE_KEYS).
Arms (node indices into (0,250,500,750), number of leading half-res steps): full4 = [0,1,2,3] 0 (4-step full-res regenerated here = same-config
floor vs the archived reference T), full2 = [0,2] 0 (nodes 0,500), mr4 = [0,1,2,3] 2, mr4b = [0,1,2,3] 1, mr2 = [0,2] 1.
Outputs <out>/<arm>/<scene>_s<seed>/{lat.pt (fp16 CPU [16, 39, h, w]), run.json (per-chunk CUDA-synced seconds: lo_steps / hi_steps / hi_commit /
lo_commit / total, rel-L2 to T per chunk, Laplacian-variance sharpness per frame)}, decoded mp4 (VAE, crf 23, 16 fps) in
<out>/judge/<arm>_<scene>_s<seed>.mp4 (T decoded too), progress in <out>/PROGRESS.md.
usage: flock /tmp/gpu5.lock env CUDA_VISIBLE_DEVICES=5 envrun python /home/hk28965/specwm/lingbot-world-v2/specwm/multires_pilot.py --arms full4,full2,mr4,mr4b,mr2
"""
import argparse, os, sys, json, time, gc, subprocess, numpy as np, torch, torch.nn.functional as F
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--arms", default="full4,full2,mr4,mr4b,mr2")
ap.add_argument("--tasks", default="sand_filling_up_jar,sandcastle_wave_impact,easy_lake,easy_stonehenge_sunset,pouring_water_into_cup"); ap.add_argument("--seeds", default="1234,4321,9999")
ap.add_argument("--out", default=S + "/lingbot-world-v2/specwm_out/multires"); ap.add_argument("--tdir", default=S + "/lingbot-world-v2/specwm_out/paths2", help="reference T: <tdir>/<task>/tau2_s0/K0_uniform4P_fx_s<seed>/lat.pt")
ap.add_argument("--stream", default=S + "/data/actstream/tau2_s0"); ap.add_argument("--frames", type=int, default=161); ap.add_argument("--window", type=int, default=18); ap.add_argument("--warm", type=int, default=1); ap.add_argument("--decode", type=int, default=1)
a = ap.parse_args()
sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2"); os.environ["SPECWM_NO_FLASH"] = "1"
from specwm.binding_v2 import V2Binding
from specwm import fast_patch as FP
ALL = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}; FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"; dev = "cuda:0"; sync = torch.cuda.synchronize
FF = S + "/tools/ffmpeg-7.0.2-amd64-static/ffmpeg"; FPS = 16
ARMS = {"full4": ([0, 1, 2, 3], 0), "full2": ([0, 2], 0), "mr4": ([0, 1, 2, 3], 2), "mr4b": ([0, 1, 2, 3], 1), "mr2": ([0, 2], 1)}
B = V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=a.window, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750))
FP.apply(); FP.FLAGS.update(dedup_t=True, no_kv_cat=True, rope_cache=True, rope_fp32=True, attn="sage", ring_kv=True); FP.fp8_swap(B.target_model, True, keep=False)
import torch._dynamo as _dyn; _dyn.config.recompile_limit = 256; _dyn.config.accumulated_recompile_limit = 8192
for blk in B.target_model.blocks: blk.forward = torch.compile(blk.forward, dynamic=True)
B.pipe.text_encoder.model.to(dev); _te = B.pipe.text_encoder
class _TE:
    def __init__(s, te): s.te = te; s.model = te.model
    def __call__(s, texts, device): return s.te(texts, torch.device(dev))
B.pipe.text_encoder = _TE(_te); SIG = [float(s) for s in B.node_sigmas()]; os.makedirs(a.out + "/judge", exist_ok=True); PROG = a.out + "/PROGRESS.md"
def log(*x):
    s = " ".join(str(v) for v in x); print("[multires] " + s, flush=True)
    with open(PROG, "a") as f: f.write("- %s %s\n" % (time.strftime("%m/%d %H:%M:%S"), s))
STATE_KEYS = ["_y_chunks", "_emb_chunks", "_ctx", "_seq_len", "_chunk_shape", "_frame_seqlen", "_chunk_size", "_kv_size", "_mk_cache", "_caches", "_cross_init", "_committed_frames"]
def snapshot(): return {k: getattr(B, k) for k in STATE_KEYS}
def use(st):
    for k, v in st.items(): setattr(B, k, v)
def area_for(image, lh, lw):
    """the max_area that makes V2Binding.prepare pick the latent grid (lh, lw) for this image's aspect (prepare: round(sqrt(A*aspect)//8//2*2))"""
    from PIL import Image; im = Image.open(image); asp = im.height / im.width; nom = 64 * lh * lw
    for A in range(int(nom * 0.6), int(nom * 1.7), 32):
        if (round(np.sqrt(A * asp) // 8 // 2 * 2), round(np.sqrt(A / asp) // 8 // 2 * 2)) == (lh, lw): return A
    raise RuntimeError("no max_area gives grid %dx%d for aspect %.4f" % (lh, lw, asp))
def prepare_res(T, lh, lw):
    B._caches = {}; gc.collect(); torch.cuda.empty_cache()
    cond = B.prepare(T["image"], T.get("camera_WM", ""), a.stream, frame_num=a.frames, max_area=area_for(T["image"], lh, lw), chunk_size=3)
    assert tuple(cond["shape"][2:]) == (lh, lw), (cond["shape"], lh, lw); B._cross_init = {"draft": False, "target": False}
    return snapshot(), cond["num_chunks"]
def reset(st):
    st["_caches"] = None; use(st); B.reset_caches(); st.update(snapshot())   # drop the state's reference first so reset_caches frees the old cache before it allocates the new one (two 14B caches = 2 x 23 GiB)
def pool_mat(Hf, hl):
    """[hl, Hf]: non-overlapping blocks of 2 or 3 rows, entries 1/sqrt(n) -> block sum / sqrt(n) keeps iid noise at unit variance"""
    b = np.round(np.linspace(0, Hf, hl + 1)).astype(int); P = torch.zeros(hl, Hf)
    for i in range(hl): P[i, b[i]:b[i + 1]] = 1.0 / np.sqrt(b[i + 1] - b[i])
    return P.to(dev)
def pool_noise(e, Ph, Pw): return torch.einsum("ah,cthw,bw->ctab", Ph, e, Pw)
def up(x, hw): return F.interpolate(x.permute(1, 0, 2, 3), size=tuple(hw), mode="bilinear", align_corners=False).permute(1, 0, 2, 3)
def down(x, hw): return F.interpolate(x.permute(1, 0, 2, 3), size=tuple(hw), mode="bilinear", align_corners=False, antialias=True).permute(1, 0, 2, 3)
def gen(seed, c, k=0): return torch.Generator().manual_seed(seed * 1009 + c * 7 + k)
HI = LO = None; NCH = 0; HW = LW = None; PH = PW = None
@torch.no_grad()
def denoise(c, seed, nodes, lo_steps, tm):
    g1 = gen(seed, c, 1); e0 = torch.randn(HI["_chunk_shape"], generator=gen(seed, c, 0)).to(dev); x_lo = x_full = None
    for j, ni in enumerate(nodes):
        s = SIG[ni]; eps = None if j == 0 else torch.randn(HI["_chunk_shape"], generator=g1).to(dev)   # same draw sequence as the full-res reference
        if j < lo_steps:
            use(LO); e_lo = pool_noise(e0 if j == 0 else eps, PH, PW); z = e_lo if j == 0 else (1 - s) * x_lo + s * e_lo
            sync(); t = time.time(); x_lo = B.target_x0(z, ni, c).float(); sync(); tm["lo_steps"] += time.time() - t; x_full = up(x_lo, HW)
        else:
            use(HI); z = e0 if j == 0 else (1 - s) * x_full + s * eps
            sync(); t = time.time(); x_full = B.target_x0(z, ni, c).float(); sync(); tm["hi_steps"] += time.time() - t
    return x_full
@torch.no_grad()
def roll(seed, nodes, lo_steps):
    reset(HI)
    if lo_steps: reset(LO)
    xs = []; rows = []
    for c in range(NCH):
        tm = dict(lo_steps=0.0, hi_steps=0.0, hi_commit=0.0, lo_commit=0.0); sync(); t0 = time.time(); x = denoise(c, seed, nodes, lo_steps, tm)
        use(HI); sync(); t = time.time(); B.commit(x, c); sync(); tm["hi_commit"] = time.time() - t
        if lo_steps: use(LO); t = time.time(); B.commit(down(x, LW), c); sync(); tm["lo_commit"] = time.time() - t
        sync(); tm["total"] = time.time() - t0; xs.append(x); rows.append({k: round(v, 4) for k, v in tm.items()})
    return xs, rows
@torch.no_grad()
def decode(lat):
    with torch.amp.autocast("cuda", dtype=torch.bfloat16): im = B.pipe.vae.decode([lat.float().to(dev)])[0].float().clamp(-1, 1)
    return ((im + 1) / 2 * 255).permute(1, 2, 3, 0).round().to(torch.uint8)   # (F, H, W, 3) uint8 on the GPU
LAP = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=dev)[None, None]
@torch.no_grad()
def lapvar(fr):
    g = (fr.float() @ torch.tensor([0.299, 0.587, 0.114], device=fr.device))[:, None]; return F.conv2d(g, LAP).var(dim=(1, 2, 3)).cpu().numpy()
def write_mp4(fr, out):
    Fn, H, W, _ = fr.shape
    p = subprocess.Popen([FF, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H), "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "fast", "-threads", "4",
                          "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out + ".tmp.mp4"], stdin=subprocess.PIPE)
    p.stdin.write(fr.cpu().numpy().tobytes()); p.stdin.close(); p.wait(); os.replace(out + ".tmp.mp4", out)
def rel_l2(lat, T): return [round(float((lat[:, 3 * c:3 * c + 3].float() - T[:, 3 * c:3 * c + 3].float()).norm() / T[:, 3 * c:3 * c + 3].float().norm()), 4) for c in range(NCH)]
def finish(name, task, seed, lat, meta):
    od = "%s/%s/%s_s%d" % (a.out, name, task, seed); os.makedirs(od, exist_ok=True)
    if name != "T": torch.save(lat.half().cpu(), od + "/lat.pt")
    if a.decode:
        fr = decode(lat); lv = lapvar(fr); meta["lapvar_mean"] = float(lv.mean()); meta["lapvar_frames"] = [round(float(v), 2) for v in lv]
        write_mp4(fr, "%s/judge/%s_%s_s%d.mp4" % (a.out, name, task, seed)); del fr; torch.cuda.empty_cache()
    json.dump(meta, open(od + "/run.json", "w"), indent=1)
def tpath(task, seed): return "%s/%s/tau2_s0/K0_uniform4P_fx_s%d/lat.pt" % (a.tdir, task, seed)
ARM_LIST = a.arms.split(","); SEEDS = [int(x) for x in a.seeds.split(",")]; TASKS = a.tasks.split(",")
for ti, task in enumerate(TASKS):
    T = ALL[task]; Tl0 = torch.load(tpath(task, SEEDS[0]), map_location="cpu"); lh, lw = Tl0.shape[2:]; ll = (round(lh / 2 / 2) * 2, round(lw / 2 / 2) * 2)
    HI = LO = None; B._caches = {}; gc.collect(); torch.cuda.empty_cache(); sync(); t0 = time.time()
    HI, NCH = prepare_res(T, lh, lw); HW = (lh, lw); LO, n2 = prepare_res(T, *ll); LW = ll; assert n2 == NCH; PH, PW = pool_mat(lh, ll[0]), pool_mat(lw, ll[1]); sync()
    log("prepare %s: %d chunks, full grid %s (frame_seqlen %d, kv %d), half grid %s (frame_seqlen %d, kv %d), %.1f s" % (task, NCH, HW, HI["_frame_seqlen"], HI["_kv_size"], LW, LO["_frame_seqlen"], LO["_kv_size"], time.time() - t0))
    if ti == 0 and a.warm: t0 = time.time(); roll(SEEDS[0], [0, 1, 2, 3], 2); sync(); log("warm-up (compile, mr4 full clip) %.0f s" % (time.time() - t0))
    for seed in SEEDS:
        Tl = torch.load(tpath(task, seed), map_location="cpu"); assert Tl.shape[1] == 3 * NCH and tuple(Tl.shape[2:]) == HW, (Tl.shape, NCH, HW)
        if not os.path.exists("%s/T/%s_s%d/run.json" % (a.out, task, seed)): finish("T", task, seed, Tl, dict(task=task, seed=seed, arm="T", src=tpath(task, seed)))
        for name in ARM_LIST:
            if os.path.exists("%s/%s/%s_s%d/run.json" % (a.out, name, task, seed)): continue
            nodes, lo = ARMS[name]; xs, rows = roll(seed, nodes, lo); lat = torch.cat(xs, 1).cpu(); r = rel_l2(lat, Tl)
            meta = dict(task=task, seed=seed, arm=name, nodes=nodes, sigmas=[SIG[i] for i in nodes], lo_steps=lo, hi_grid=list(HW), lo_grid=list(LW), chunks=rows, rel_l2_T=r)
            finish(name, task, seed, lat, meta); tot = [c["total"] for c in rows]
            log("%s s%d %s: %.1f s (%.2f s/chunk; steady c>=6 %.2f: lo %.2f hi %.2f commit %.2f + lo-commit %.2f) relL2->T c0 %.3f c6 %.3f c12 %.3f lapvar %.1f" % (
                task, seed, name, sum(tot), float(np.mean(tot)), float(np.mean(tot[6:])), float(np.mean([c["lo_steps"] for c in rows[6:]])), float(np.mean([c["hi_steps"] for c in rows[6:]])),
                float(np.mean([c["hi_commit"] for c in rows[6:]])), float(np.mean([c["lo_commit"] for c in rows[6:]])), r[0], r[6], r[12], meta.get("lapvar_mean", float("nan"))))
            del xs, lat; torch.cuda.empty_cache()
log("ALL_DONE arms=%s" % a.arms)
