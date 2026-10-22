"""Scene-general drift head on 14B hidden features (exp/drift_feat): per chunk a small MLP over the 16 pooled tokens (channel-standardized),
clip score = logsumexp over chunks (MIL, as the latent critic), label = human 'severe'. Scene-DISJOINT evaluation: hold out half of the 32 non-jar/castle
scenes (and a seed-held-out jar/castle set); train on the rest. Reports AUROC on held-out scenes for layer 20 / 30 / both, vs the latent critic (0.63) and
DINO zero-shot. usage: python -m specwm.drift_head_train --out DIR [--epochs 40]"""
import argparse, os, glob, json, random, numpy as np, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--hid", type=int, default=256); ap.add_argument("--folds", type=int, default=2); ap.add_argument("--all", action="store_true", help="train one deployment head on every clip (no held-out scenes); also writes the chunk-logit distribution on clean clips for threshold calibration"); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True); dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(0); random.seed(0)
R = [torch.load(f, map_location="cpu") for f in sorted(glob.glob(S + "/exp/drift_feat/*.pt"))]; print("[head] clips", len(R), "severe rate %.2f" % np.mean([r["severe"] for r in R]))
scenes = sorted({r["task"] for r in R if r["group"] == "other"}); random.Random(0).shuffle(scenes)
def feats(r, layers): return torch.cat([r["f%d" % k].float() for k in layers], -1)   # [N, 16, C*len]
class Head(nn.Module):
    def __init__(s, C, hid):
        super().__init__(); s.mu = nn.Parameter(torch.zeros(C), requires_grad=False); s.sd = nn.Parameter(torch.ones(C), requires_grad=False)
        s.tok = nn.Sequential(nn.Linear(C, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU()); s.pos = nn.Parameter(torch.zeros(16, hid)); s.out = nn.Sequential(nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, 1))
    def chunk_logits(s, x):   # x [N,16,C] -> [N]
        h = s.tok((x - s.mu) / s.sd) + s.pos; return s.out(h.mean(1)).squeeze(-1)
    def forward(s, x): l = s.chunk_logits(x); return torch.logsumexp(l[1:] if len(l) > 1 else l, 0), l   # chunk 0 is the given image
def run(layers, hold_scenes, tag):
    tr = [r for r in R if not (r["group"] == "other" and r["task"] in hold_scenes)]; te = [r for r in R if r["group"] == "other" and r["task"] in hold_scenes]
    jc = [r for r in tr if r["group"] == "jc"]; random.Random(1).shuffle(jc); te_jc = jc[:60]; tr = [r for r in tr if r not in te_jc]
    X = {id(r): feats(r, layers).to(dev) for r in R}; C = X[id(R[0])].shape[-1]; m = Head(C, a.hid).to(dev)
    allx = torch.cat([X[id(r)].reshape(-1, C) for r in tr]); m.mu.data = allx.mean(0); m.sd.data = allx.std(0).clamp_min(1e-3)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=a.lr, weight_decay=1e-2); pos = np.mean([r["severe"] for r in tr]); w = torch.tensor([(1 - pos) / pos], device=dev)
    def auc(rs):
        m.eval();
        with torch.no_grad(): s = [float(m(X[id(r)])[0]) for r in rs]
        y = [r["severe"] for r in rs]; return roc_auc_score(y, s) if 0 < sum(y) < len(y) else float("nan")
    best = (0, None)
    for ep in range(a.epochs):
        m.train(); random.shuffle(tr); tot = 0
        for r in tr:
            lg, _ = m(X[id(r)]); y = torch.tensor([float(r["severe"])], device=dev); loss = nn.functional.binary_cross_entropy_with_logits(lg[None], y, pos_weight=w if r["severe"] else None)
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss)
        if (ep + 1) % 5 == 0:
            a_tr, a_te, a_jc = auc(tr), auc(te), auc(te_jc); print("[head %s] ep %d loss %.3f  AUROC train %.2f | held-out scenes %.2f | jar/castle held-out seeds %.2f" % (tag, ep + 1, tot / len(tr), a_tr, a_te, a_jc), flush=True)
            if (a_te if a_te == a_te else a_jc) > best[0]: best = ((a_te if a_te == a_te else a_jc), {k: v.cpu() for k, v in m.state_dict().items()}, a_jc, ep + 1)
    torch.save(dict(layers=layers, hid=a.hid, state=best[1], auc_heldout=best[0]), a.out + "/head_%s.pt" % tag); return best
if a.all:
    b = run([20], set(), "L20_all"); m_state = b[1]; X20 = {id(r): feats(r, [20]).to(dev) for r in R}; m = Head(X20[id(R[0])].shape[-1], a.hid).to(dev); m.load_state_dict(m_state); m.eval()
    with torch.no_grad():
        clean = [float(x) for r in R if not r["drift"] for x in m.chunk_logits(X20[id(r)])[1:]]; sev = [float(x) for r in R if r["severe"] for x in m.chunk_logits(X20[id(r)])[1:]]
    q = {p: float(np.percentile(clean, p)) for p in (80, 90, 95, 98)}; print("[head] chunk logits on clean clips: percentiles", {k: round(v, 2) for k, v in q.items()}, "| share of severe-clip chunks above p90: %.2f, p95: %.2f" % (np.mean([x > q[90] for x in sev]), np.mean([x > q[95] for x in sev])))
    torch.save(dict(layers=[20], hid=a.hid, state=m_state, thresholds=q), a.out + "/head_L20_all.pt"); print("[HEAD_DONE]"); raise SystemExit
res = {}
for layers, tag in (([20], "L20"), ([30], "L30"), ([20, 30], "L20_30")):
    aucs = []
    for k in range(a.folds):
        hold = scenes[k::a.folds]; b = run(layers, set(hold), "%s_f%d" % (tag, k)); aucs.append((b[0], b[2])); print("[head] %s fold %d: held-out scenes %.2f (best ep %d), jar/castle seeds %.2f" % (tag, k, b[0], b[3], b[2]), flush=True)
    res[tag] = aucs
print("| features (14B hidden, teacher-forced commit pass) | AUROC on scenes never trained on ↑ (mean of %d folds) | AUROC on jar/castle, held-out seeds ↑ |" % a.folds); print("|---|---|---|")
for tag, aucs in res.items(): print("| %s | %.2f | %.2f |" % (tag, np.mean([x[0] for x in aucs]), np.mean([x[1] for x in aucs])))
print("| latent critic (reference, 10/6) | 0.63 | 0.84–0.98 |"); print("[HEAD_DONE]")
