"""[10/20 §52] Divergence-injection probe: does a generator G (1.3B 4-step, or the 14B 2-step as the same-model control) FOLLOW an injected
14B 4-step context (T) or RETURN to its own path?  Solo AR loop built from the pieces of spec_early.py / act_spec2.py / two_machine.py
(binding_v2 model loading, ring KV, per-chunk uniform K-step denoise, dense clean commit).  Arms per (task, seed), all in ONE code path:
  alone      : G on its own (reference)
  self@k     : KV reset, G's OWN chunks 0..k re-committed through the clean pass, G continues k+1..N-1   (re-commit cost / numerical floor)
  once@k     : KV reset, T's chunks 0..k (same seed's archived 14B 4-step clip) re-committed, G continues k+1..N-1   (the hybrid's cloud re-anchor)
  every      : teacher forcing on the context: chunk c is denoised by G in T's context 0..c-1, recorded, then T's chunk c is committed instead
Noise convention = act_spec2.py (the T arm): initial noise Generator(seed*1009 + c*7), re-noise draws Generator(seed*1009 + c*7 + 1), so the
injected 14B-2step control shares T's noise for its two nodes.  Re-commit = reset_caches + sequential B.commit(x_T[c], c) for c <= k: the KV
the ring would hold had G generated T's latents (two_machine.recommit rewrites one slot in place; identical when every chunk is corrected in order).
Outputs: <out>/<model><K>/<task>/<arm>_s<seed>/lat.pt (fp16 CPU, [16, 3N, h, w]) + run.json; optional contact sheet / mp4 for --sheet task:seed.
usage: CUDA_VISIBLE_DEVICES=5 python -m specwm.inject_probe --model 1p3b --steps 4 --ks 1,3,6,9 --self-k 3 --tasks ... --seeds ... --out DIR"""
import argparse, os, sys, json, time, numpy as np, torch
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--model", default="1p3b", choices=["1p3b", "14b"]); ap.add_argument("--steps", type=int, default=4, help="uniform K-step schedule over the first K of the 4 nodes (act_spec2 uniform:K)")
ap.add_argument("--tasks", default="sand_filling_up_jar,sandcastle_wave_impact,easy_lake,easy_stonehenge_sunset,pouring_water_into_cup"); ap.add_argument("--seeds", default="1234,4321,9999,11,22,33,44,55,66,77")
ap.add_argument("--out", default=S + "/lingbot-world-v2/specwm_out/inject"); ap.add_argument("--tdir", default=S + "/lingbot-world-v2/specwm_out/paths2", help="archived T clips: <tdir>/<task>/tau2_s0/K0_uniform4P_fx_s<seed>/lat.pt")
ap.add_argument("--stream", default=S + "/data/actstream/tau2_s0"); ap.add_argument("--ks", default="1,3,6,9"); ap.add_argument("--self-k", type=int, default=3); ap.add_argument("--arms", default="alone,self,once,every"); ap.add_argument("--frames", type=int, default=161); ap.add_argument("--area", type=int, default=480 * 832); ap.add_argument("--window", type=int, default=18)
ap.add_argument("--sheet", default="sandcastle_wave_impact:1234", help="task:seed decoded to a contact sheet (1 fps) + mp4s: T, alone, once@k (every k), every; '' = none"); ap.add_argument("--warm", type=int, default=1); a = ap.parse_args()
sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2"); os.environ["SPECWM_NO_FLASH"] = "1"
from specwm.binding_v2 import V2Binding
from specwm import fast_patch as FP
ALL = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}; FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"; dev = "cuda:0"; sync = torch.cuda.synchronize
if a.model == "14b": B = V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=a.window, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750))
else: B = V2Binding(S + "/lingbot-world-v2/lingbot-world-v2-1.3b-causal-fast", target_mode="fast_bf16", quant=None, sink=6, local_attn=a.window, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750), cfg_name="i2v-1.3B", assets_dir=FAST)
FP.apply(); FP.FLAGS.update(dedup_t=True, no_kv_cat=True, rope_cache=True, rope_fp32=True, attn="sage", ring_kv=True)
if a.model == "14b": FP.fp8_swap(B.target_model, True, keep=False)
import torch._dynamo as _dyn; _dyn.config.recompile_limit = 64; _dyn.config.accumulated_recompile_limit = 4096
for blk in B.target_model.blocks: blk.forward = torch.compile(blk.forward, dynamic=True)
B.pipe.text_encoder.model.to(dev); _te = B.pipe.text_encoder
class _TE:
    def __init__(s, te): s.te = te; s.model = te.model
    def __call__(s, texts, device): return s.te(texts, torch.device(dev))
B.pipe.text_encoder = _TE(_te); K = a.steps; TAG = "%s_%dstep" % (a.model, K); OUT = "%s/%s" % (a.out, TAG); os.makedirs(OUT, exist_ok=True); PROG = a.out + "/PROGRESS.md"
def log(*x):
    s = " ".join(str(v) for v in x); print("[inject] " + s, flush=True)
    with open(PROG, "a") as f: f.write("- %s %s %s\n" % (time.strftime("%m/%d %H:%M:%S"), TAG, s))
def gen(seed, c, k=0): return torch.Generator().manual_seed(seed * 1009 + c * 7 + k)   # = act_spec2.gen for fixed 3-frame units (k = 1: the re-noise draws)
SIG = None
@torch.no_grad()
def denoise(c, seed):
    g1 = gen(seed, c, 1); z = B.chunk_noise(c, gen(seed, c, 0))
    for i in range(K):
        x = B.target_x0(z, i, c)
        if i < K - 1: z = (1 - SIG[i + 1]) * x + SIG[i + 1] * B.randn_like(x, g1)
    return x
@torch.no_grad()
def roll(seed, ctx=None, k=-1, force=None):
    """ctx: [16, 3N, h, w] clip whose chunks 0..k are re-committed first (KV reset); force: clip whose chunk c replaces G's chunk c in the KV after every chunk (teacher forcing).
    returns (chunks list of [16, 3, h, w] fp32 on the GPU, per-chunk seconds, re-commit seconds)"""
    N = NCH; B.reset_caches(); xs = []; tc = []; sync(); t0 = time.time()
    for c in range(k + 1): xc = ctx[:, 3 * c:3 * c + 3].float().to(dev); B.commit(xc, c); xs.append(xc)
    sync(); t_re = time.time() - t0
    for c in range(k + 1, N):
        sync(); t1 = time.time(); x = denoise(c, seed)
        if force is not None: B.commit(force[:, 3 * c:3 * c + 3].float().to(dev), c)
        else: B.commit(x, c)
        sync(); tc.append(round(time.time() - t1, 4)); xs.append(x)
    return xs, tc, t_re
def save(xs, od, meta):
    os.makedirs(od, exist_ok=True); torch.save(torch.cat(xs, 1).half().cpu(), od + "/lat.pt"); json.dump(meta, open(od + "/run.json", "w"), indent=1)
def tpath(task, seed): return "%s/%s/tau2_s0/K0_uniform4P_fx_s%d/lat.pt" % (a.tdir, task, seed)
@torch.no_grad()
def decode(lat):
    with torch.amp.autocast("cuda", dtype=torch.bfloat16): im = B.pipe.vae.decode([lat.float().to(dev)])[0].float().clamp(-1, 1).cpu()
    return ((im + 1) / 2 * 255).permute(1, 2, 3, 0).numpy().astype("uint8")
def sheet(task, seed, rows, od):
    """rows: [(label, lat)] -> 1 fps contact sheet png + small mp4 per row"""
    from PIL import Image, ImageDraw; import imageio
    tiles = []; W = 208
    for lab, lat in rows:
        v = decode(lat); imageio.mimwrite("%s/%s.mp4" % (od, lab.replace("@", "").replace(" ", "_")), list(v), fps=16, quality=6)
        fr = [Image.fromarray(v[i]).resize((W, int(W * v.shape[1] / v.shape[2]))) for i in range(0, len(v), 16)]; H = fr[0].size[1]
        row = Image.new("RGB", (W * len(fr), H + 14), "black"); d = ImageDraw.Draw(row); d.text((3, 1), "%s  %s s%d  (1 fps; chunk = 0.75 s)" % (lab, task, seed), fill="white")
        for i, f in enumerate(fr): row.paste(f, (i * W, 14))
        tiles.append(row)
    sh = Image.new("RGB", (tiles[0].size[0], sum(t.size[1] for t in tiles)), "black"); y = 0
    for t in tiles: sh.paste(t, (0, y)); y += t.size[1]
    sh.save(od + "/sheet.png"); log("sheet", od + "/sheet.png")
ARMS = a.arms.split(","); KS = [int(x) for x in a.ks.split(",")]; SEEDS = [int(x) for x in a.seeds.split(",")]; TASKS = a.tasks.split(",")
SHEET = tuple(a.sheet.split(":")) if a.sheet else None
for ti, task in enumerate(TASKS):
    T = ALL[task]; sync(); t0 = time.time(); B._caches = {}; import gc; gc.collect(); torch.cuda.empty_cache()   # drop the previous task's KV before prepare() builds the new one (14B: 21 GiB each)
    cond = B.prepare(T["image"], T.get("camera_WM", ""), a.stream, frame_num=a.frames, max_area=a.area, chunk_size=3); NCH = cond["num_chunks"]; SIG = [float(s) for s in B.node_sigmas()]; sync(); log("prepare %s: %d chunks, shape %s, %.1f s" % (task, NCH, tuple(cond["shape"]), time.time() - t0))
    if ti == 0 and a.warm: t0 = time.time(); roll(SEEDS[0], ctx=torch.load(tpath(task, SEEDS[0])), k=1); sync(); log("warm-up (compile) %.0f s" % (time.time() - t0))
    for seed in SEEDS:
        od = "%s/%s" % (OUT, task); Tl = torch.load(tpath(task, seed)); assert Tl.shape[1] == 3 * NCH, (Tl.shape, NCH); runs = {}
        def need(name): return name.split("@")[0] in ARMS and not os.path.exists("%s/%s_s%d/lat.pt" % (od, name, seed))
        def have(arm): return torch.load("%s/%s_s%d/lat.pt" % (od, arm, seed))
        if need("alone"): xs, tc, _ = roll(seed); save(xs, "%s/alone_s%d" % (od, seed), dict(task=task, seed=seed, arm="alone", chunk_s=tc)); log("%s s%d alone: %.1f s (%.2f s/chunk)" % (task, seed, sum(tc), float(np.mean(tc))))
        if need("self@%d" % a.self_k):
            G = have("alone"); xs, tc, tr = roll(seed, ctx=G, k=a.self_k); save(xs, "%s/self@%d_s%d" % (od, a.self_k, seed), dict(task=task, seed=seed, arm="self@%d" % a.self_k, k=a.self_k, chunk_s=tc, recommit_s=tr))
            Gc = torch.cat(xs, 1).half().cpu(); r = [float((Gc[:, 3 * c:3 * c + 3].float() - G[:, 3 * c:3 * c + 3].float()).norm() / G[:, 3 * c:3 * c + 3].float().norm()) for c in range(NCH)]
            log("%s s%d self@%d: %.1f s (re-commit %.2f s); rel L2 to alone per chunk: %s" % (task, seed, a.self_k, sum(tc), tr, " ".join("%.3f" % v for v in r)))
        for k in KS:
            if need("once@%d" % k): xs, tc, tr = roll(seed, ctx=Tl, k=k); save(xs, "%s/once@%d_s%d" % (od, k, seed), dict(task=task, seed=seed, arm="once@%d" % k, k=k, chunk_s=tc, recommit_s=tr)); log("%s s%d once@%d: %.1f s (re-commit %.2f s)" % (task, seed, k, sum(tc), tr))
        if need("every"): xs, tc, _ = roll(seed, force=Tl); save(xs, "%s/every_s%d" % (od, seed), dict(task=task, seed=seed, arm="every", chunk_s=tc)); log("%s s%d every: %.1f s" % (task, seed, sum(tc)))
        if SHEET and SHEET[0] == task and int(SHEET[1]) == seed:
            sd = "%s/sheet_%s_s%d" % (OUT, task, seed); os.makedirs(sd, exist_ok=True)
            if not os.path.exists(sd + "/sheet.png"):
                rows = [("T 14B 4-step", Tl), ("G alone (%s)" % TAG, have("alone"))] + [("once@%d" % k, have("once@%d" % k)) for k in KS if os.path.exists("%s/once@%d_s%d/lat.pt" % (od, k, seed))] + [("every", have("every"))]
                sheet(task, seed, rows, sd)
        torch.cuda.empty_cache()
log("ALL_DONE %s" % TAG)
