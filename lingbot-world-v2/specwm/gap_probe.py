"""Speculative-Forcing oracle-gap probe on the 14B causal-fast (4-step schedule). Target-only rollout; at every chunk and step k in 1..3 record the
token-wise convergence gap to the step-4 output,  rel_k(i) = ||x0^(k)(i) - x0^(4)(i)||_2 / sigma_t,  sigma_t = median token norm of x0^(4) of the chunk
(token = 1x2x2 latent patch = 64 values, as in the paper). Reports, per k, the fraction of tokens with rel_k <= eps for eps in {0.16, 0.24, 0.32} =
the share of the chunk that could stop at step k, i.e. the achievable mean denoising depth. Saves per-chunk histograms + a few gap maps.
usage: CUDA_VISIBLE_DEVICES=g python -m specwm.gap_probe --tasks a,b --seeds 1,2 --out DIR [--model 14b|1.3b]"""
import argparse, os, sys, json, time, numpy as np, torch
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--tasks", required=True); ap.add_argument("--seeds", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--model", default="14b"); ap.add_argument("--fast", type=int, default=1, help="FP8 + Sage + ring KV patches (14b only)"); a = ap.parse_args()
sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2"); os.environ["SPECWM_NO_FLASH"] = "1"
from specwm.binding_v2 import V2Binding
from specwm import fast_patch as FP
ALL = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}; FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"
if a.model == "14b": B = V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750))
else: B = V2Binding(S + "/lingbot-world-v2/lingbot-world-v2-1.3b-causal-fast", target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750), cfg_name="i2v-1.3B", assets_dir=FAST)
if a.fast and a.model == "14b": FP.apply(); FP.FLAGS.update(dedup_t=True, no_kv_cat=True, rope_cache=True, rope_fp32=True, attn="sage", ring_kv=True); FP.fp8_swap(B.target_model, True, keep=False)
os.makedirs(a.out, exist_ok=True); EPS = (0.16, 0.24, 0.32)
def patch(x):   # [16, F, H, W] -> [F*(H/2)*(W/2), 64]
    c, f, h, w = x.shape; return x.view(c, f, 1, h // 2, 2, w // 2, 2).permute(1, 3, 5, 2, 4, 6, 0).reshape(f * (h // 2) * (w // 2), 64)
def gen(seed, c): return torch.Generator().manual_seed(seed * 1009 + c * 7)
recs = []
for task in a.tasks.split(","):
    T = ALL[task]
    for seed in [int(s) for s in a.seeds.split(",")]:
        t0 = time.time()
        with torch.no_grad():
            cond = B.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=161, max_area=480 * 832); B.reset_caches(); N = cond["num_chunks"]; sig = [float(s) for s in B.node_sigmas()]
            for c in range(N):
                g = gen(seed, c); z = B.chunk_noise(c, g); xs = []
                for i in range(len(sig)):
                    x = B.target_x0(z, i, c).float(); xs.append(x)
                    if i < len(sig) - 1: s_ = sig[i + 1]; z = (1 - s_) * x + s_ * B.randn_like(x, g)
                xK = patch(xs[-1]); sg = float(xK.norm(dim=1).median()); r = dict(task=task, seed=seed, chunk=c, sigma=sg, frac={}, mean={})
                for k in range(1, len(sig)):
                    rel = (patch(xs[k - 1]) - xK).norm(dim=1) / sg; r["frac"][k] = {str(e): float((rel <= e).float().mean()) for e in EPS}; r["mean"][k] = float(rel.mean())
                    if seed == int(a.seeds.split(",")[0]) and c in (1, 4, 8): torch.save(rel.view(xs[-1].shape[1], xs[-1].shape[2] // 2, xs[-1].shape[3] // 2).half().cpu(), "%s/gap_%s_c%d_k%d.pt" % (a.out, task, c, k))
                recs.append(r); B.commit(xs[-1], c)
        print("[GAP] %s s%d %.0fs" % (task, seed, time.time() - t0), flush=True)
        json.dump(recs, open("%s/records_%s.json" % (a.out, a.model), "w"))
print("| k (stop after step) | mean rel_k ↓ | tokens with rel ≤ 0.16 | ≤ 0.24 | ≤ 0.32 |"); print("|---|---|---|---|---|")
for k in (1, 2, 3):
    print("| %d | %.3f | %.0f %% | %.0f %% | %.0f %% |" % (k, np.mean([r["mean"][k] for r in recs]), *[100 * np.mean([r["frac"][k][str(e)] for r in recs]) for e in EPS]))
# achievable mean depth at eps = 0.24 with greedy per-token stopping (a token stops at the first k whose gap <= eps; monotone assumption)
for e in EPS:
    f1 = np.mean([r["frac"][1][str(e)] for r in recs]); f2 = np.mean([r["frac"][2][str(e)] for r in recs]); f3 = np.mean([r["frac"][3][str(e)] for r in recs])
    depth = 1 * f1 + 2 * max(f2 - f1, 0) + 3 * max(f3 - max(f2, f1), 0) + 4 * (1 - max(f1, f2, f3)); print("eps %.2f: oracle mean depth ≈ %.2f steps (uniform = 4)" % (e, depth))
print("[GAP_DONE]", flush=True)
