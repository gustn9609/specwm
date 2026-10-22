"""Cheaper target-based verifiers. The gold signal is the anchor distance d0 = ||x0_T^(step1)(z0) - x0_D|| (rho 0.95 with the final gap, one full
14B forward). Here: the same with the 14B truncated to its first L layers (logit lens: the real head on layer-L hidden), L in --layers, plus a ridge
head fitted on layer-L hidden -> 14B step-1 output tokens (fit on the first half of the chunks, evaluated on the rest). Also the draft's own free
signals. Reports Spearman with local_err = gap(draft final, 14B final) and the cost fraction L/40.
usage: CUDA_VISIBLE_DEVICES=g python -m specwm.cheap_anchor_probe --tasks a,b --seeds 4321,9999 --out DIR"""
import argparse, os, sys, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import spearmanr
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--tasks", required=True); ap.add_argument("--seeds", default="4321,9999"); ap.add_argument("--layers", default="5,10,20,30"); ap.add_argument("--out", required=True); a = ap.parse_args()
sys.path.insert(0, S + "/lingbot-world-v2"); os.chdir(S + "/lingbot-world-v2"); os.environ["SPECWM_NO_FLASH"] = "1"
from specwm.binding_v2 import V2Binding
from specwm import fast_patch as FP
ALL = {t["id"]: t for t in json.load(open(S + "/data/stevo/tasks_camera.json"))}; FAST = S + "/lingbot-world-v2/lingbot-world-v2-14b-causal-fast"
Tg = V2Binding(FAST, target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750))
D = V2Binding(S + "/lingbot-world-v2/lingbot-world-v2-1.3b-causal-fast", target_mode="fast_bf16", quant=None, sink=6, local_attn=18, swap_models=False, build_draft=False, timesteps_index=(0, 250, 500, 750), cfg_name="i2v-1.3B", assets_dir=FAST)
FP.apply(); FP.FLAGS.update(dedup_t=True, no_kv_cat=True, rope_cache=True, rope_fp32=True, attn="sage", ring_kv=True); FP.fp8_swap(Tg.target_model, True, keep=False)
os.makedirs(a.out, exist_ok=True); dev = "cuda:0"; LAY = [int(x) for x in a.layers.split(",")]; M = Tg.target_model; ALLB = M.blocks
H = {}
def mk(k):
    def f(m, i, o): H[k] = (o[0] if isinstance(o, tuple) else o)[0]
    return f
for L in LAY: ALLB[L - 1].register_forward_hook(mk(L))
def hook_head(m, args, out): H["e"] = args[1]
M.head.register_forward_hook(hook_head)
def patch(x):
    c, f, h, w = x.shape; return x.view(c, f, 1, h // 2, 2, w // 2, 2).permute(1, 3, 5, 2, 4, 6, 0).reshape(f * (h // 2) * (w // 2), 64)
def gen(seed, c, k): return torch.Generator().manual_seed(seed * 1009 + c * 7 + k)
@torch.no_grad()
def sample(B, z, c, g):
    sig = [float(s) for s in B.node_sigmas()]; xs = []
    for i in range(len(sig)):
        x = B.target_x0(z, i, c).float(); xs.append(x)
        if i < len(sig) - 1: s_ = sig[i + 1]; z = (1 - s_) * x + s_ * B.randn_like(x, g)
    return xs
recs = []; feats = {L: [] for L in LAY}; targets = []
for task in a.tasks.split(","):
    T = ALL[task]
    for seed in [int(s) for s in a.seeds.split(",")]:
        with torch.no_grad():
            cond = Tg.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=161, max_area=480 * 832); Tg.reset_caches()
            D.prepare(T["image"], T.get("camera_WM", ""), T["action"], frame_num=161, max_area=480 * 832); D.reset_caches(); N = cond["num_chunks"]
            for c in range(N):
                z0 = Tg.chunk_noise(c, gen(seed, c, 0)); xsT = sample(Tg, z0, c, gen(seed, c, 1)); A4 = xsT[-1]
                if c >= 1:
                    A1 = xsT[0]; tD1 = None; xsD = sample(D, z0, c, gen(seed, c, 2)); xD = xsD[-1]; t4 = patch(A4); sg = float(t4.norm(dim=1).median())
                    tD = patch(xD); tA1 = patch(A1); local_err = float(((tD - t4).norm(dim=1) / sg).mean()); d0 = float(((patch(xsD[0]) - tA1).norm(dim=1) / sg).mean())
                    r = dict(task=task, seed=seed, chunk=c, local_err=local_err, d0=d0, eD_stab=float(((patch(xsD[1]) - patch(xsD[0])).norm(dim=1) / sg).mean()))
                    # re-run the 14B step-1 forward to capture layer hiddens (the earlier call was inside sample)
                    Tg.target_x0(z0, 0, c); e = H["e"]; sig0 = float(Tg.node_sigmas()[0]); zt = patch(z0.float())
                    for L in LAY:
                        h = H[L]; v = M.head(h.unsqueeze(0), e)[0].float(); x0h = zt - sig0 * v   # logit lens at layer L
                        r["lens%d" % L] = float(((patch(xsD[0]) - x0h).norm(dim=1) / sg).mean())
                        feats[L].append(F.layer_norm(h.float(), (h.shape[-1],)).half().cpu());
                    targets.append((tA1.half().cpu(), patch(xsD[0]).half().cpu(), sg, len(recs))); recs.append(r)
                Tg.commit(A4, c); D.commit(A4, c)
        print("[CHEAP] %s s%d" % (task, seed), flush=True)
# ridge heads per layer: fit on the first half of chunks, evaluate ridge-anchor distance on the second half
n = len(recs); half = n // 2; y = np.array([r["local_err"] for r in recs])
print("| signal | cost (14B layers) | Spearman with gap(draft final, 14B final) ↑ |"); print("|---|---|---|")
print("| anchor distance d0 (full 14B step 1) | 40 | %.2f |" % spearmanr([r["d0"] for r in recs], y)[0])
print("| draft stability ‖x̂0^(2)−x̂0^(1)‖ (free) | 0 | %.2f |" % spearmanr([r["eD_stab"] for r in recs], y)[0])
for L in LAY:
    print("| logit lens at layer %d (untrained head) | %d | %.2f |" % (L, L, spearmanr([r["lens%d" % L] for r in recs], y)[0]))
    X = torch.cat([f for f in feats[L][:half]]).float().cuda(); Y = torch.cat([t[0] for t in targets[:half]]).float().cuda(); Xa = torch.cat([X, torch.ones(len(X), 1, device="cuda")], 1)
    W = torch.linalg.solve(Xa.T @ Xa + 1e2 * torch.eye(Xa.shape[1], device="cuda"), Xa.T @ Y); rd = []
    for f, t in zip(feats[L][half:], targets[half:]):
        Xe = torch.cat([f.float().cuda(), torch.ones(len(f), 1, device="cuda")], 1); pred = Xe @ W; rd.append(float(((t[1].float().cuda() - pred).norm(dim=1) / t[2]).mean()))
    print("| ridge anchor at layer %d (fit on %d chunks, eval on %d) | %d | %.2f |" % (L, half, n - half, L, spearmanr(rd, y[half:])[0]))
json.dump(recs, open(a.out + "/records.json", "w")); print("[CHEAP_DONE]", flush=True)
