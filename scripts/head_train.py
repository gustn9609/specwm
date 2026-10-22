"""Learned convergence verifier for the 14B (Speculative Forcing head) on gap_dump.py data.
Input per (chunk, step k): projected last-block hidden [F, H/4, W/4, 1536]; target: oracle gap rel_k pooled to the same grid. Head (paper Appendix A):
LayerNorm -> Conv3d 1536->96 (3x3x3, GELU) -> Conv3d 96->48 (3x3x3, GELU) -> Conv3d 48->1 (1x1x1), shared over steps; smooth-L1 (Huber) loss; Adam 1e-3.
Splits: test = sand_filling_up_jar, sandcastle_wave_impact + every 10th task (sorted); train = the rest. Reports per k: AUROC of 'needs more
refinement' (rel_k > eps) for eps 0.16/0.24/0.32, Spearman(pred, oracle), and the mean denoising depth under oracle vs predicted acceptance at eps.
usage: CUDA_VISIBLE_DEVICES=g python scripts/head_train.py --data DIR --out DIR [--epochs 20]"""
import argparse, os, glob, json, time, math, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True); ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--bs", type=int, default=4); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True); torch.manual_seed(a.seed); random.seed(a.seed); dev = "cuda:0"; EPS = (0.16, 0.24, 0.32)
files = sorted(glob.glob(a.data + "/*_c*.pt")); tasks = sorted({os.path.basename(f).rsplit("_s", 1)[0] for f in files})
test_tasks = {"sand_filling_up_jar", "sandcastle_wave_impact"} | {t for i, t in enumerate(tasks) if i % 10 == 0}
print("files %d tasks %d test tasks %d" % (len(files), len(tasks), len(test_tasks)), flush=True)
D = {"train": [], "test": []}
for f in files:
    d = torch.load(f, map_location="cpu"); split = "test" if d["task"] in test_tasks else "train"
    D[split].append(dict(h=d["h"].to(dev), y=d["gap_pool"].float().to(dev), task=d["task"], chunk=d["chunk"]))   # h [3,F,h,w,C] fp16, y [3,F,h,w]
KS = D["train"][0]["y"].shape[0]; print("train chunks %d test chunks %d steps %d" % (len(D["train"]), len(D["test"]), KS), flush=True)
class Head(nn.Module):
    def __init__(s, cin=1536):
        super().__init__(); s.ln = nn.LayerNorm(cin); s.c1 = nn.Conv3d(cin, 96, 3, padding=1); s.c2 = nn.Conv3d(96, 48, 3, padding=1); s.c3 = nn.Conv3d(48, 1, 1)
    def forward(s, h):   # h [B, F, H, W, C] -> [B, F, H, W]
        x = s.ln(h.float()).permute(0, 4, 1, 2, 3); x = F.gelu(s.c1(x)); x = F.gelu(s.c2(x)); return s.c3(x).squeeze(1)
m = Head(cin=D["train"][0]["h"].shape[-1]).to(dev); print("head params %.2fM" % (sum(p.numel() for p in m.parameters()) / 1e6), flush=True)
opt = torch.optim.Adam(m.parameters(), lr=a.lr); steps = a.epochs * math.ceil(len(D["train"]) / a.bs); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
@torch.no_grad()
def evaluate(split):
    m.eval(); P = {k: [] for k in range(KS)}; Y = {k: [] for k in range(KS)}
    for d in D[split]:
        p = m(d["h"])   # [3,F,h,w]
        for k in range(KS): P[k].append(p[k].flatten().cpu()); Y[k].append(d["y"][k].flatten().cpu())
    m.train(); R = {}
    for k in range(KS):
        p = torch.cat(P[k]).numpy(); y = torch.cat(Y[k]).numpy(); r = dict(spearman=float(spearmanr(p, y)[0]), mae=float(np.abs(p - y).mean()))
        for e in EPS: r["auroc_%.2f" % e] = float(roc_auc_score(y > e, p)) if 0 < (y > e).mean() < 1 else None
        R["k%d" % (k + 1)] = r
    # mean denoising depth at eps: a token stops at the first k where (oracle | predicted) gap <= eps, else 4
    for e in EPS:
        for name, G in (("oracle", Y), ("pred", P)):
            g = torch.stack([torch.cat(G[k]) for k in range(KS)]).numpy(); stop = np.full(g.shape[1], KS + 1)
            for k in range(KS - 1, -1, -1): stop = np.where(g[k] <= e, k + 1, stop)
            R["depth_%s_%.2f" % (name, e)] = float(stop.mean())
        # quality proxy: mean oracle gap of the tokens committed by the predicted policy at their commit step
        g = torch.stack([torch.cat(Y[k]) for k in range(KS)]).numpy(); pg = torch.stack([torch.cat(P[k]) for k in range(KS)]).numpy(); stop = np.full(g.shape[1], KS)
        for k in range(KS - 1, -1, -1): stop = np.where(pg[k] <= e, k, stop)
        comm = np.where(stop < KS, g[np.minimum(stop, KS - 1), np.arange(g.shape[1])], 0.0)
        R["committed_gap_pred_%.2f" % e] = float(comm[stop < KS].mean()) if (stop < KS).any() else None
    return R
t0 = time.time()
for ep in range(a.epochs):
    random.shuffle(D["train"]); tl = 0.0
    for i in range(0, len(D["train"]), a.bs):
        b = D["train"][i:i + a.bs]; loss = sum(F.smooth_l1_loss(m(d["h"]), d["y"]) for d in b) / len(b)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tl += loss.item() * len(b)
    if ep % 5 == 4 or ep == a.epochs - 1:
        R = evaluate("test"); print("epoch %d loss %.4f (%.0fs) test: " % (ep + 1, tl / len(D["train"]), time.time() - t0) + " | ".join("k%d AUROC@0.24 %s ρ %.3f" % (k + 1, "%.3f" % R["k%d" % (k + 1)]["auroc_0.24"] if R["k%d" % (k + 1)]["auroc_0.24"] is not None else "-", R["k%d" % (k + 1)]["spearman"]) for k in range(KS)), flush=True)
res = dict(test=evaluate("test"), train=evaluate("train"), test_tasks=sorted(test_tasks), args=vars(a))
torch.save(dict(state=m.state_dict(), proj="proj.pt"), a.out + "/head.pt"); json.dump(res, open(a.out + "/result.json", "w"), indent=1)
print("[HEAD_DONE]", json.dumps(res["test"]), flush=True)
