"""Can the KV commit of chunk c be hidden behind step 1 of chunk c+1? (lever 1: commit overlap on 2 cloud GPUs)
At chunk c+1 (same noise z, same re-noise generator) compare to the normal rollout x_ref (clean committed K/V of chunk c for every step):
  stale1   : step 1 of chunk c+1 reads chunk c's K/V from its LAST DENOISE STEP (sigma 0.63 input, no commit yet); steps 2-4 use the clean K/V
             -> what the overlapped pipeline would produce
  stale_all: all 4 steps on the stale K/V (= SPECWM_SKIP_COMMIT, the known hallucination case)
  other    : clean K/V, different re-noise (the 14B's own noise floor)
Token gap in sigma units (median token norm of x_ref) and share within eps 0.24.
usage: CUDA_VISIBLE_DEVICES=g python -m specwm.overlap_probe --tasks a,b --seeds 4321,9999 --c 2,5,8 --out DIR"""
import argparse, os, sys, json, numpy as np, torch
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--tasks", required=True); ap.add_argument("--seeds", default="4321,9999"); ap.add_argument("--c", default="2,5,8"); ap.add_argument("--out", required=True); a = ap.parse_args()
sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2"); os.environ["SPECWM_NO_FLASH"] = "1"
from specwm.binding_v2 import V2Binding
from specwm import fast_patch as FP
ALL = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}; FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"
B = V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750))
FP.apply(); FP.FLAGS.update(dedup_t=True, no_kv_cat=True, rope_cache=True, rope_fp32=True, attn="sage", ring_kv=True); FP.fp8_swap(B.target_model, True, keep=False)
import torch._dynamo as _dyn; _dyn.config.recompile_limit = 64
for blk in B.target_model.blocks: blk.forward = torch.compile(blk.forward, dynamic=False)
os.makedirs(a.out, exist_ok=True); SIG = [float(s) for s in B.node_sigmas()]
def patch(x):
    c, f, h, w = x.shape; return x.view(c, f, 1, h // 2, 2, w // 2, 2).permute(1, 3, 5, 2, 4, 6, 0).reshape(f * (h // 2) * (w // 2), 64)
def gen(seed, c, k=0): return torch.Generator().manual_seed(seed * 1009 + c * 7 + k)
@torch.no_grad()
def steps(c, z, g, start=0, x=None):   # run nodes start..3 of chunk c; if start > 0, x is the clean estimate at node start-1
    for i in range(start, len(SIG)):
        if i > 0: s_ = SIG[i]; z = (1 - s_) * x + s_ * B.randn_like(x, g)
        x = B.target_x0(z, i, c).float()
    return x
def snapshot(): return {k: dict(kv=[{kk: vv.to("cpu", copy=True) for kk, vv in l.items()} for l in v["kv"]], cross=[{kk: (vv.to("cpu", copy=True) if torch.is_tensor(vv) else vv) for kk, vv in l.items()} for l in v["cross"]]) for k, v in B._caches.items()}, dict(B._cross_init), B._committed_frames
def restore(snap):
    caches, ci, cf = snap
    for k, v in caches.items():
        for l, l2 in zip(B._caches[k]["kv"], v["kv"]):
            for kk in l: l[kk].copy_(l2[kk])
        for l, l2 in zip(B._caches[k]["cross"], v["cross"]):
            for kk in l:
                if torch.is_tensor(l[kk]): l[kk].copy_(l2[kk])
    B._cross_init = dict(ci); B._committed_frames = cf
recs = []
for task in a.tasks.split(","):
    T = ALL[task]
    for seed in [int(s) for s in a.seeds.split(",")]:
        for c0 in [int(x) for x in a.c.split(",")]:
            with torch.no_grad():
                B.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=161, max_area=480 * 832); B.reset_caches()
                for c in range(c0): B.commit(steps(c, B.chunk_noise(c, gen(seed, c)), gen(seed, c, 1)), c)   # dense prefix, committed
                xc = steps(c0, B.chunk_noise(c0, gen(seed, c0)), gen(seed, c0, 1))                         # chunk c0 denoised, NOT committed yet
                S_stale = snapshot(); B.commit(xc, c0); S_clean = snapshot()
                c1 = c0 + 1; z1 = B.chunk_noise(c1, gen(seed, c1)); g = lambda k: gen(seed, c1, k)
                x_ref = steps(c1, z1, g(1))
                restore(S_clean); x_other = steps(c1, z1, g(21))
                restore(S_stale); B._committed_frames += B._chunk_size; x_all = steps(c1, z1, g(1))
                restore(S_stale); B._committed_frames += B._chunk_size; x1 = B.target_x0(z1, 0, c1).float()   # step 1 on stale K/V
                restore(S_clean); gg = g(1); x_st1 = steps(c1, None, gg, start=1, x=x1)
                t = patch(x_ref); sg = float(t.norm(dim=1).median()); G = lambda x: ((patch(x) - t).norm(dim=1) / sg)
                r = dict(task=task, seed=seed, c=c1)
                for k, x in (("stale1", x_st1), ("stale_all", x_all), ("other", x_other)): gp = G(x); r[k] = dict(gap=float(gp.mean()), within=float((gp <= 0.24).float().mean()))
                recs.append(r); print("[OVERLAP] %s s%d c%d " % (task, seed, c1) + " ".join("%s %.3f" % (k, r[k]["gap"]) for k in ("stale1", "stale_all", "other")), flush=True)
                json.dump(recs, open(a.out + "/records.json", "w"))
print("| chunk c+1 variant (vs normal rollout, same noise) | mean token gap ↓ (σ) | within ε 0.24 ↑ | n |"); print("|---|---|---|---|")
for k, lab in (("stale1", "step 1 on chunk c's last-denoise K/V, steps 2-4 clean (overlapped commit)"), ("stale_all", "all steps on stale K/V (skip commit)"), ("other", "clean K/V, different re-noise (14B noise floor)")):
    print("| %s | %.3f | %.0f %% | %d |" % (lab, np.mean([r[k]["gap"] for r in recs]), 100 * np.mean([r[k]["within"] for r in recs]), len(recs)))
print("[OVERLAP_DONE]", flush=True)
