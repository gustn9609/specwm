"""Anchor split + commit offload: the only way to take the commit off the critical path is to run the step-1 anchor of chunk c+1 on the K/V the
cloud already has for chunk c -- which, in the anchor pipeline, is the K/V written by the sigma=1 anchor forward itself (not a last-denoise step).
Measures, at chunk c+1 with the same noise: anchor on clean K/V vs anchor on the sigma=1-stale K/V of chunk c, each finished by the 1.3B (last 2
steps), gap to the 14B 4-step output and to each other. Also the anchor-only gaps. usage: CUDA_VISIBLE_DEVICES=g python -m specwm.anchor_stale_probe --tasks a,b --seeds 4321,9999 --c 2,5,8 --out DIR"""
import argparse, os, sys, json, numpy as np, torch
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--tasks", required=True); ap.add_argument("--seeds", default="4321,9999"); ap.add_argument("--c", default="2,5,8"); ap.add_argument("--out", required=True); a = ap.parse_args()
sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2"); os.environ["SPECWM_NO_FLASH"] = "1"
from specwm.binding_v2 import V2Binding
from specwm import fast_patch as FP
ALL = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}; FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"
Tg = V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750))
D = V2Binding(S + "/lingbot-world-v2/lingbot-world-v2-1.3b-causal-fast", target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750), cfg_name="i2v-1.3B", assets_dir=FAST)
FP.apply(); FP.FLAGS.update(dedup_t=True, no_kv_cat=True, rope_cache=True, rope_fp32=True, attn="sage", ring_kv=True); FP.fp8_swap(Tg.target_model, True, keep=False)
import torch._dynamo as _dyn; _dyn.config.recompile_limit = 64
for blk in Tg.target_model.blocks: blk.forward = torch.compile(blk.forward, dynamic=False)
for blk in D.target_model.blocks: blk.forward = torch.compile(blk.forward, dynamic=False)
os.makedirs(a.out, exist_ok=True); SIG = [float(s) for s in Tg.node_sigmas()]
def patch(x):
    c, f, h, w = x.shape; return x.view(c, f, 1, h // 2, 2, w // 2, 2).permute(1, 3, 5, 2, 4, 6, 0).reshape(f * (h // 2) * (w // 2), 64)
def gen(seed, c, k=0): return torch.Generator().manual_seed(seed * 1009 + c * 7 + k)
@torch.no_grad()
def finish(B, x, c, g, start):
    for i in range(start + 1, len(SIG)): s_ = SIG[i]; z = (1 - s_) * x + s_ * B.randn_like(x, g); x = B.target_x0(z, i, c).float()
    return x
def snapshot(B): return {k: dict(kv=[{kk: vv.to("cpu", copy=True) for kk, vv in l.items()} for l in v["kv"]], cross=[{kk: (vv.to("cpu", copy=True) if torch.is_tensor(vv) else vv) for kk, vv in l.items()} for l in v["cross"]]) for k, v in B._caches.items()}, dict(B._cross_init), B._committed_frames
def restore(B, snap):
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
                Tg.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=161, max_area=480 * 832); Tg.reset_caches()
                D.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=161, max_area=480 * 832); D.reset_caches()
                for c in range(c0):   # dense 14B prefix committed to both
                    x = finish(Tg, Tg.target_x0(Tg.chunk_noise(c, gen(seed, c)), 0, c).float(), c, gen(seed, c, 1), 0); Tg.commit(x, c); D.commit(x, c)
                z0 = Tg.chunk_noise(c0, gen(seed, c0)); A1 = Tg.target_x0(z0, 0, c0).float()   # the anchor forward of chunk c0: its sigma=1 K/V now sits in chunk c0's slots
                S_stale = snapshot(Tg); x4 = finish(Tg, A1, c0, gen(seed, c0, 1), 0); Tg.commit(x4, c0); D.commit(x4, c0); S_clean = snapshot(Tg)
                c1 = c0 + 1; z1 = Tg.chunk_noise(c1, gen(seed, c1))
                ref = finish(Tg, Tg.target_x0(z1, 0, c1).float(), c1, gen(seed, c1, 1), 0)                      # 14B 4-step reference
                restore(Tg, S_clean); A_clean = Tg.target_x0(z1, 0, c1).float(); h_clean = finish(D, A_clean, c1, gen(seed, c1, 4), 1)
                restore(Tg, S_stale); Tg._committed_frames += Tg._chunk_size; A_stale = Tg.target_x0(z1, 0, c1).float(); h_stale = finish(D, A_stale, c1, gen(seed, c1, 4), 1)
                restore(Tg, S_clean); zB = Tg.chunk_noise(c1, gen(seed, c1, 11)); other = finish(Tg, Tg.target_x0(zB, 0, c1).float(), c1, gen(seed, c1, 12), 0)
                t = patch(ref); sg = float(t.norm(dim=1).median()); G = lambda x, y=t: ((patch(x) - y).norm(dim=1) / sg)
                r = dict(task=task, seed=seed, c=c1)
                for k, x in (("anchor_clean", A_clean), ("anchor_stale", A_stale), ("h2_clean", h_clean), ("h2_stale", h_stale), ("other", other)): gp = G(x); r[k] = dict(gap=float(gp.mean()), within=float((gp <= 0.24).float().mean()))
                r["stale_vs_clean_anchor"] = float(G(A_stale, patch(A_clean)).mean()); r["stale_vs_clean_h2"] = float(G(h_stale, patch(h_clean)).mean())
                recs.append(r); print("[ASTALE] %s s%d c%d " % (task, seed, c1) + " ".join("%s %.3f" % (k, r[k]["gap"]) for k in ("anchor_clean", "anchor_stale", "h2_clean", "h2_stale", "other")) + " | stale-vs-clean anchor %.3f h2 %.3f" % (r["stale_vs_clean_anchor"], r["stale_vs_clean_h2"]), flush=True)
                json.dump(recs, open(a.out + "/records.json", "w"))
print("| chunk c+1 variant | gap to 14B 4-step ↓ (σ) | within ε ↑ | gap to the clean-K/V version ↓ |"); print("|---|---|---|---|")
for k, lab, kk in (("anchor_clean", "14B anchor on clean K/V (today's pipeline)", None), ("anchor_stale", "14B anchor on the σ=1-stale K/V of chunk c (commit hidden)", "stale_vs_clean_anchor"), ("h2_clean", "anchor clean + 1.3B 2 steps", None), ("h2_stale", "anchor stale + 1.3B 2 steps", "stale_vs_clean_h2"), ("other", "14B 4-step, different noise (floor)", None)):
    print("| %s | %.3f | %.0f %% | %s |" % (lab, np.mean([r[k]["gap"] for r in recs]), 100 * np.mean([r[k]["within"] for r in recs]), ("%.3f" % np.mean([r[kk] for r in recs])) if kk else "–"))
print("[ASTALE_DONE]", flush=True)
