"""[10/20] SOFT re-anchor sweep: copy of specwm/inject_probe.py (same model loading, noise convention, ring KV, re-commit) with two new arms for the
same-model pair (device = 14B 2-step, cloud T = 14B 4-step archived latents, specwm_out/paths2/<task>/tau2_s0/K0_uniform4P_fx_s<seed>/lat.pt):
  soft@k_a<alpha> : KV reset; context chunks 0..k = x_dev + alpha * (x_T - x_dev) (x_dev = the device's OWN `alone` latents of the same seed, i.e. what the
                    device had shown), re-committed through the device's own clean forward; device continues k+1..N-1.  alpha = 1 is the existing once@k,
                    alpha = 0 is `alone` (both NOT re-run here; the analysis reads them from the once@k / alone dirs).
  prog@k          : PROGRESSIVE blend: at chunk j in {k, k+1, k+2} with alpha_j in {0.25, 0.5, 1.0}: cur[0..j] <- cur + alpha_j * (T - cur) over ALL context chunks
                    0..j (cur = the device's current latents: alone's for chunks <= k, its freshly generated ones for j > k), KV reset + re-commit, generate
                    chunk j+1.  After the last step the context equals T's chunks 0..k+2 (once@k+2's KV), but the device showed chunks k+1, k+2 generated
                    under the partial blends.  Saved lat.pt = chunks 0..k: the first blend (alpha_0); chunks > k: the device-generated chunks AS SHOWN.
Everything else (arms alone / self / once / every, outputs, run.json) is identical to inject_probe.py; the `alone` arm is generated here if missing.
Outputs: <out>/<model><K>/<task>/soft@<k>_a<alpha>_s<seed>/lat.pt and .../prog@<k>_s<seed>/lat.pt (fp16 CPU, [16, 3N, h, w]) + run.json.
usage: CUDA_VISIBLE_DEVICES=5 envrun python /abs/.../specwm/inject_soft.py --model 14b --steps 2 --k 6 --alphas 0.25,0.5,0.75 --tasks ... --seeds ... [--prog]"""
import argparse, os, sys, json, time, numpy as np, torch
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--model", default="14b", choices=["1p3b", "14b"]); ap.add_argument("--steps", type=int, default=2, help="uniform K-step schedule over the first K of the 4 nodes (act_spec2 uniform:K)")
ap.add_argument("--tasks", default="sandcastle_wave_impact,easy_stonehenge_sunset,sand_filling_up_jar"); ap.add_argument("--seeds", default="1234,4321,9999,11,22,33,44,55,66,77")
ap.add_argument("--out", default=S + "/lingbot-world-v2/specwm_out/inject"); ap.add_argument("--tdir", default=S + "/lingbot-world-v2/specwm_out/paths2", help="archived T clips: <tdir>/<task>/tau2_s0/K0_uniform4P_fx_s<seed>/lat.pt")
ap.add_argument("--stream", default=S + "/data/actstream/tau2_s0"); ap.add_argument("--k", type=int, default=6); ap.add_argument("--alphas", default="0.25,0.5,0.75"); ap.add_argument("--prog", action="store_true", help="also run the progressive arm prog@k (alpha 0.25 / 0.5 / 1.0 at chunks k / k+1 / k+2)")
ap.add_argument("--prog-alphas", default="0.25,0.5,1.0"); ap.add_argument("--frames", type=int, default=161); ap.add_argument("--area", type=int, default=480 * 832); ap.add_argument("--window", type=int, default=18); ap.add_argument("--warm", type=int, default=1); a = ap.parse_args()
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
B.pipe.text_encoder = _TE(_te); K = a.steps; TAG = "%s_%dstep" % (a.model, K); OUT = "%s/%s" % (a.out, TAG); os.makedirs(OUT, exist_ok=True); PROG = a.out + "/PROGRESS_soft.md"
def log(*x):
    s = " ".join(str(v) for v in x); print("[inject_soft] " + s, flush=True)
    with open(PROG, "a") as f: f.write("- %s %s %s\n" % (time.strftime("%m/%d %H:%M:%S"), TAG, s))
def gen(seed, c, k=0): return torch.Generator().manual_seed(seed * 1009 + c * 7 + k)   # = act_spec2.gen / inject_probe.gen (same noise per chunk in every arm)
SIG = None
@torch.no_grad()
def denoise(c, seed):
    g1 = gen(seed, c, 1); z = B.chunk_noise(c, gen(seed, c, 0))
    for i in range(K):
        x = B.target_x0(z, i, c)
        if i < K - 1: z = (1 - SIG[i + 1]) * x + SIG[i + 1] * B.randn_like(x, g1)
    return x
def blend(cur, T, alpha, upto):
    """cur, T: [16, 3N, h, w] fp16/fp32 CPU; returns a copy of cur whose chunks 0..upto are cur + alpha * (T - cur)"""
    out = cur.float().clone(); n = 3 * (upto + 1); out[:, :n] = out[:, :n] + alpha * (T[:, :n].float() - out[:, :n]); return out
@torch.no_grad()
def roll(seed, ctx=None, k=-1, force=None):
    """identical to inject_probe.roll: ctx chunks 0..k re-committed first (KV reset), then the device continues k+1..N-1"""
    N = NCH; B.reset_caches(); xs = []; tc = []; sync(); t0 = time.time()
    for c in range(k + 1): xc = ctx[:, 3 * c:3 * c + 3].float().to(dev); B.commit(xc, c); xs.append(xc)
    sync(); t_re = time.time() - t0
    for c in range(k + 1, N):
        sync(); t1 = time.time(); x = denoise(c, seed)
        if force is not None: B.commit(force[:, 3 * c:3 * c + 3].float().to(dev), c)
        else: B.commit(x, c)
        sync(); tc.append(round(time.time() - t1, 4)); xs.append(x)
    return xs, tc, t_re
@torch.no_grad()
def roll_prog(seed, dev_lat, T, k, alphas):
    """progressive re-anchor: steps (k + i, alphas[i]); step j fires AFTER chunk j is shown (alone's for j = k, the device's fresh one for j > k): the whole
    context 0..j is blended toward T with alpha_j, KV reset + re-commit, then chunk j+1 is generated in that KV.
    returns (shown chunks list: 0..k = first blend, > k = generated), per-chunk seconds, re-commit seconds list, final context [16, 3N, h, w] (CPU fp16)"""
    N = NCH; cur = dev_lat.float().clone(); steps = {k + i: float(al) for i, al in enumerate(alphas)}; xs = []; tc = []; tre = []
    for c in range(N):
        if c > k:
            sync(); t1 = time.time(); x = denoise(c, seed); B.commit(x, c); sync(); tc.append(round(time.time() - t1, 4)); xs.append(x.cpu()); cur[:, 3 * c:3 * c + 3] = x.cpu()
        if c in steps:
            cur = blend(cur, T, steps[c], c); B.reset_caches(); sync(); t0 = time.time()
            for cc in range(c + 1): B.commit(cur[:, 3 * cc:3 * cc + 3].to(dev), cc)
            sync(); tre.append(round(time.time() - t0, 3))
            if c == k: xs = [cur[:, 3 * cc:3 * cc + 3].clone() for cc in range(k + 1)]   # shown context = the first blend (the once@k convention: lat.pt holds the KV content at the re-anchor)
    return xs, tc, tre, cur.half()
def save(xs, od, meta):
    os.makedirs(od, exist_ok=True); torch.save(torch.cat([x.cpu() for x in xs], 1).half(), od + "/lat.pt"); json.dump(meta, open(od + "/run.json", "w"), indent=1)
def tpath(task, seed): return "%s/%s/tau2_s0/K0_uniform4P_fx_s%d/lat.pt" % (a.tdir, task, seed)
SEEDS = [int(x) for x in a.seeds.split(",")]; TASKS = a.tasks.split(","); ALPHAS = [float(x) for x in a.alphas.split(",")]; PALPHAS = [float(x) for x in a.prog_alphas.split(",")]; k = a.k
for ti, task in enumerate(TASKS):
    T = ALL[task]; sync(); t0 = time.time(); B._caches = {}; import gc; gc.collect(); torch.cuda.empty_cache()
    cond = B.prepare(T["image"], T.get("camera_WM", ""), a.stream, frame_num=a.frames, max_area=a.area, chunk_size=3); NCH = cond["num_chunks"]; SIG = [float(s) for s in B.node_sigmas()]; sync(); log("prepare %s: %d chunks, shape %s, %.1f s" % (task, NCH, tuple(cond["shape"]), time.time() - t0))
    if ti == 0 and a.warm: t0 = time.time(); roll(SEEDS[0], ctx=torch.load(tpath(task, SEEDS[0])), k=1); sync(); log("warm-up (compile) %.0f s" % (time.time() - t0))
    for seed in SEEDS:
        od = "%s/%s" % (OUT, task); Tl = torch.load(tpath(task, seed)); assert Tl.shape[1] == 3 * NCH, (Tl.shape, NCH)
        def need(name): return not os.path.exists("%s/%s_s%d/lat.pt" % (od, name, seed))
        def have(arm): return torch.load("%s/%s_s%d/lat.pt" % (od, arm, seed))
        if need("alone"): xs, tc, _ = roll(seed); save(xs, "%s/alone_s%d" % (od, seed), dict(task=task, seed=seed, arm="alone", chunk_s=tc)); log("%s s%d alone: %.1f s (%.2f s/chunk)" % (task, seed, sum(tc), float(np.mean(tc))))
        G = have("alone")
        for al in ALPHAS:
            name = "soft@%d_a%g" % (k, al)
            if not need(name): continue
            ctx = blend(G, Tl, al, k); xs, tc, tr = roll(seed, ctx=ctx, k=k)
            Gc = torch.cat([x.cpu() for x in xs], 1); dT = [float((Gc[:, 3 * c:3 * c + 3].float() - Tl[:, 3 * c:3 * c + 3].float()).norm() / Tl[:, 3 * c:3 * c + 3].float().norm()) for c in range(k + 1, NCH)]
            save(xs, "%s/%s_s%d" % (od, name, seed), dict(task=task, seed=seed, arm=name, k=k, alpha=al, chunk_s=tc, recommit_s=tr, ctx="alone + alpha*(T - alone) for chunks <= k")); log("%s s%d %s: %.1f s (re-commit %.2f s); rel L2 to T after k: %s" % (task, seed, name, sum(tc), tr, " ".join("%.3f" % v for v in dT)))
        if a.prog and need("prog@%d" % k):
            xs, tc, tre, cur = roll_prog(seed, G, Tl, k, PALPHAS); name = "prog@%d" % k; od2 = "%s/%s_s%d" % (od, name, seed)
            save(xs, od2, dict(task=task, seed=seed, arm=name, k=k, alphas=PALPHAS, steps=[k + i for i in range(len(PALPHAS))], chunk_s=tc, recommit_s=tre, note="lat.pt chunks <= k = first blend; > k = generated chunks as shown; ctx_final.pt = the KV content after the last blend"))
            torch.save(cur, od2 + "/ctx_final.pt"); log("%s s%d %s: %.1f s (re-commits %s)" % (task, seed, name, sum(tc), tre))
        torch.cuda.empty_cache()
log("ALL_DONE %s soft k=%d alphas=%s prog=%s" % (TAG, k, ALPHAS, a.prog))
