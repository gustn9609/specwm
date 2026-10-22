"""Calibration for the cross-model gap (feedback 10/7): how far apart are two 14B samples of the SAME chunk that differ only in noise?
Rollout = 14B 4-step target-only (chunk committed = sample A). At each chunk c >= 1, besides A (noise z), also sample B (14B, an independent
noise z') and D (1.3B draft from A's initial noise z0, its own re-noising) and record per-token gaps to A, normalized by A's median token norm:
  gap_TT  = ||B - A|| / sigma   (14B vs 14B, different noise: the irreducible same-context variability)
  gap_DT  = ||D - A|| / sigma   (draft vs 14B, same z0: what gap_dump_cross measured)
  gap_T1  = ||A^(1) - A|| / sigma   (14B's own step-1 estimate: the early-exit reference)
usage: CUDA_VISIBLE_DEVICES=g python -m specwm.noise_calib --tasks a,b --seeds 7 --out DIR"""
import argparse, os, sys, json, time, numpy as np, torch
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--tasks", required=True); ap.add_argument("--seeds", default="7"); ap.add_argument("--out", required=True); ap.add_argument("--d-lora", default=""); a = ap.parse_args()
sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2"); os.environ["SPECWM_NO_FLASH"] = "1"
from specwm.binding_v2 import V2Binding
from specwm import fast_patch as FP
ALL = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}; FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"
Tg = V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750))
D = V2Binding(S + "/lingbot-world-v2/lingbot-world-v2-1.3b-causal-fast", target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750), cfg_name="i2v-1.3B", assets_dir=FAST)
if a.d_lora:   # merge W += B A (block order q,k,v,o[,ffn0,ffn2]), as spec_pipe.py
    st = torch.load(a.d_lora, map_location="cpu"); it = iter(zip(st["A"], st["B"]))
    with torch.no_grad():
        for blk in D.target_model.blocks:
            for lin in [getattr(blk.self_attn, n) for n in ("q", "k", "v", "o")] + ([blk.ffn[0], blk.ffn[2]] if st["ffn"] else []): A_, B_ = next(it); lin.weight += (B_.float() @ A_.float()).to(lin.weight.device, lin.weight.dtype)
    print("[calib] merged draft LoRA", a.d_lora, "step", st.get("step"), flush=True)
FP.apply(); FP.FLAGS.update(dedup_t=True, no_kv_cat=True, rope_cache=True, rope_fp32=True, attn="sage", ring_kv=True); FP.fp8_swap(Tg.target_model, True, keep=False)
import torch._dynamo as _dyn; _dyn.config.recompile_limit = 64
for blk in Tg.target_model.blocks: blk.forward = torch.compile(blk.forward, dynamic=False)
os.makedirs(a.out, exist_ok=True)
def patch(x):
    c, f, h, w = x.shape; return x.view(c, f, 1, h // 2, 2, w // 2, 2).permute(1, 3, 5, 2, 4, 6, 0).reshape(f * (h // 2) * (w // 2), 64)
def gen(seed, c, k=0): return torch.Generator().manual_seed(seed * 1009 + c * 7 + k)
@torch.no_grad()
def sample(B, z, c, g):
    sig = [float(s) for s in B.node_sigmas()]; xs = []
    for i in range(len(sig)):
        x = B.target_x0(z, i, c).float(); xs.append(x)
        if i < len(sig) - 1: s_ = sig[i + 1]; z = (1 - s_) * x + s_ * B.randn_like(x, g)
    return xs
recs = []
for task in a.tasks.split(","):
    T = ALL[task]
    for seed in [int(s) for s in a.seeds.split(",")]:
        t0 = time.time()
        with torch.no_grad():
            cond = Tg.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=161, max_area=480 * 832); Tg.reset_caches()
            D.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=161, max_area=480 * 832); D.reset_caches(); N = cond["num_chunks"]
            for c in range(N):
                z0 = Tg.chunk_noise(c, gen(seed, c)); xsA = sample(Tg, z0, c, gen(seed, c, 0)); A = xsA[-1]
                if c >= 1:
                    z1 = Tg.chunk_noise(c, gen(seed, c, 11)); B_ = sample(Tg, z1, c, gen(seed, c, 12))[-1]          # independent noise, same context
                    Dx = sample(D, z0, c, gen(seed, c, 1))[-1]                                                      # draft from A's z0
                    tA = patch(A); sg = float(tA.norm(dim=1).median())
                    g = lambda X: ((patch(X) - tA).norm(dim=1) / sg)
                    gTT, gDT, gT1 = g(B_), g(Dx), g(xsA[0])
                    recs.append(dict(task=task, seed=seed, chunk=c, frac_TT=[float((gTT <= e).float().mean()) for e in (0.24, 0.32, 0.5)], frac_DT=[float((gDT <= e).float().mean()) for e in (0.24, 0.32, 0.5)],
                                     frac_T1=[float((gT1 <= e).float().mean()) for e in (0.24, 0.32, 0.5)], mean_TT=float(gTT.mean()), mean_DT=float(gDT.mean()), mean_T1=float(gT1.mean()),
                                     easy_TT=float((gTT[gT1 <= 0.24] <= 0.24).float().mean()), hard_TT=float((gTT[gT1 > 0.24] <= 0.24).float().mean()),
                                     easy_DT=float((gDT[gT1 <= 0.24] <= 0.24).float().mean()), hard_DT=float((gDT[gT1 > 0.24] <= 0.24).float().mean())))
                Tg.commit(A, c); D.commit(A, c)
        print("[CALIB] %s s%d %.0fs" % (task, seed, time.time() - t0), flush=True); json.dump(recs, open(a.out + "/records.json", "w"))
import collections
print("| proposal | mean gap ↓ | within 0.24 | 0.32 | 0.5 | within 0.24 on 14B-easy tokens | on 14B-hard tokens |"); print("|---|---|---|---|---|---|---|")
for n, lab in (("TT", "14B, different noise"), ("DT", "1.3B draft, same z0"), ("T1", "14B own step 1")):
    fr = np.mean([r["frac_" + n] for r in recs], 0); print("| %s | %.3f | %.0f %% | %.0f %% | %.0f %% | %s | %s |" % (lab, np.mean([r["mean_" + n] for r in recs]), *(100 * fr),
          ("%.0f %%" % (100 * np.mean([r["easy_" + n] for r in recs]))) if n != "T1" else "-", ("%.1f %%" % (100 * np.mean([r["hard_" + n] for r in recs]))) if n != "T1" else "-"))
print("[CALIB_DONE]", flush=True)
