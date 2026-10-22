"""Can draft/target velocities or denoising scores verify a chunk? Paired probe on the action-conditioned LingBot-World v2.
Rollout = 14B target (2 steps, FP8, SageAttention, ring KV) whose chunks are committed to BOTH models' caches. At every chunk c >= 1, from the same
noise and context: x_T = target chunk (2 steps), x_D = draft chunk (1.3B, 4 steps). Signals, for each candidate x in {x_T, x_D} and sigma in SIGMAS,
with one fixed eps per chunk:  z = (1 - s) x + s eps,  v = model velocity at z,  x0_hat = z - s v
  vdis   = ||v_T - v_D||^2 / ||v_T||^2                (where the two models point differently)
  eT, eD = ||x0_hat_T - x||^2 / ||x||^2, same with the draft   (denoising error = score-matching / ELBO term: low eT -> x likely under the target)
  ctr    = log(eT / eD)                               (density-ratio proxy p_T / p_D)
plus the node-0 agreement d0 = ||x0_T(z0) - x0_D(z0)||^2 / ||x0_T||^2 at sigma = 1 (candidate-independent; the current agree verifier) and
local_err = ||x_D - x_T||^2 / ||x_T||^2 (how far the draft lands from what the target would have produced from the same noise).
Saves <out>/records.json (per chunk) and <out>/maps_<task>_s<seed>_c<c>.pt (per-pixel |v_T - v_D|, eT maps) for the first chunks of the first seed.
usage: CUDA_VISIBLE_DEVICES=g python -m specwm.velocity_probe --tasks a --seeds 1,2 --out DIR"""
import argparse, os, sys, json, time, math, numpy as np, torch
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--tasks", required=True); ap.add_argument("--seeds", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--sigmas", default="0.83,0.6,0.4,0.2"); ap.add_argument("--maps", type=int, default=4, help="save spatial maps for the first N chunks of the first seed"); a = ap.parse_args()
sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2"); os.environ["SPECWM_NO_FLASH"] = "1"
from specwm.binding_v2 import V2Binding
from specwm import fast_patch as FP
ALL = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}; FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"
Tg = V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 500))
D = V2Binding(S + "/lingbot-world-v2/lingbot-world-v2-1.3b-causal-fast", target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750), cfg_name="i2v-1.3B", assets_dir=FAST)
FP.apply(); FP.FLAGS.update(dedup_t=True, no_kv_cat=True, rope_cache=True, rope_fp32=True, attn="sage", ring_kv=True); FP.fp8_swap(Tg.target_model, True, keep=False)
SIG = [float(x) for x in a.sigmas.split(",")]; os.makedirs(a.out, exist_ok=True)
def gen(seed, c, k): return torch.Generator().manual_seed(seed * 1009 + c * 7 + k)
@torch.no_grad()
def x0_at(B, z, s, c):   # x0 prediction and velocity at an arbitrary sigma s (nearest scheduler timestep)
    v = B.target_flow(z, s, c); return z - s * v, v
@torch.no_grad()
def sample(B, z, sig, c, g):
    x = None
    for i in range(len(sig)):
        x = B.target_x0(z, i, c)
        if i < len(sig) - 1: s_ = float(sig[i + 1]); z = (1 - s_) * x + s_ * B.randn_like(x, g)
    return x.float()
rel = lambda A, B_: float(((A - B_) ** 2).mean() / (B_ ** 2).mean().clamp_min(1e-8))
recs = []; tasks = a.tasks.split(","); seeds = [int(s) for s in a.seeds.split(",")]
for task in tasks:
    T = ALL[task]
    for si, seed in enumerate(seeds):
        t0 = time.time()
        with torch.no_grad():
            cond = Tg.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=161, max_area=480 * 832); Tg.reset_caches()
            D.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=161, max_area=480 * 832); D.reset_caches()
            N = cond["num_chunks"]; st = [float(s) for s in Tg.node_sigmas()]; sd = [float(s) for s in D.node_sigmas()]
            for c in range(N):
                z0 = Tg.chunk_noise(c, gen(seed, c, 0))
                xT = sample(Tg, z0, st, c, gen(seed, c, 2))
                if c > 0:
                    xD = sample(D, z0, sd, c, gen(seed, c, 1))
                    x0T0 = Tg.target_x0(z0, 0, c); x0D0 = D.target_x0(z0, 0, c); r = dict(task=task, seed=seed, chunk=c, d0=rel(x0D0, x0T0), local_err=rel(xD, xT))
                    eps = Tg.randn_like(xT, gen(seed, c, 5))
                    for s in SIG:
                        for name, x in (("T", xT), ("D", xD)):
                            z = (1 - s) * x + s * eps; xhT, vT = x0_at(Tg, z, s, c); xhD, vD = x0_at(D, z, s, c)
                            eT, eD = rel(xhT, x), rel(xhD, x)
                            r["%s_s%.2f" % (name, s)] = dict(vdis=rel(vD, vT), eT=eT, eD=eD, ctr=math.log(eT / max(eD, 1e-12)))
                            if si == 0 and c <= a.maps and s in (SIG[0], SIG[2]):
                                torch.save(dict(vdis=((vT - vD) ** 2).mean(0).half().cpu(), eT=((xhT - x) ** 2).mean(0).half().cpu(), x=x.half().cpu()),
                                           "%s/maps_%s_s%d_c%d_%s_s%.2f.pt" % (a.out, task, seed, c, name, s))
                    recs.append(r)
                Tg.commit(xT, c); D.commit(xT, c)   # context = the target's own rollout
        print("[PROBE] %s s%d %.0fs  chunks %d" % (task, seed, time.time() - t0, N), flush=True)
        json.dump(recs, open("%s/records_%s.json" % (a.out, "_".join(tasks)), "w"))
print("[PROBE_DONE]", flush=True)
