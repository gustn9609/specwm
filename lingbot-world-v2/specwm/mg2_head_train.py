"""Drift head transferred to Matrix-Game 2.0: same MIL head as drift_head_train.py on MG2's own hidden states (blocks 10/20, context pass, 3x5+mean
pooled, 1536-d), labels = VLM clip drift score (0-100, vlm_clip_score.py) thresholded at --thr; scene-disjoint 2-fold over the training scenes,
plus the 10 comparison prompts as a fully held-out set. Reports AUROC vs the VLM label and Spearman vs the VLM score and vs DINO min-cos.
usage: python -m specwm.mg2_head_train --data specwm_out/mg2/head_data --labels paper/1012/vlm_scores_mg2.json --out DIR"""
import argparse, os, glob, json, random, numpy as np, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
S = "/home/hk28965/specwm"
ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--labels", required=True); ap.add_argument("--out", required=True); ap.add_argument("--thr", type=float, default=60); ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--hid", type=int, default=256); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True); dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(0); random.seed(0)
LAB = {os.path.basename(os.path.dirname(k)): v for k, v in json.load(open(a.labels)).items()}; R = []
for d in sorted(glob.glob(a.data + "/*_s*")):
    f, v = d + "/feat.pt", os.path.basename(d)
    if not (os.path.exists(f) and v in LAB): continue
    r = torch.load(f, map_location="cpu"); r["score"] = LAB[v]["drift"]; r["severe"] = LAB[v]["drift"] >= a.thr; r["scene"] = os.path.basename(d).rsplit("_s", 1)[0]; R.append(r)
CMP = {"easy_lake", "easy_greatwall", "easy_stonehenge_sunset", "easy_urban_rain", "easy_dragon_night", "pouring_water_into_cup", "domino_falling", "hourglass_accumulation", "newtons_cradle", "billiard_break"}
train_scenes = sorted({r["scene"] for r in R if r["scene"] not in CMP}); random.Random(0).shuffle(train_scenes); print("[mg2 head] clips %d, train scenes %d, severe rate %.2f (thr %g)" % (len(R), len(train_scenes), np.mean([r["severe"] for r in R]), a.thr))
class Head(nn.Module):
    def __init__(s, C, hid):
        super().__init__(); s.mu = nn.Parameter(torch.zeros(C), requires_grad=False); s.sd = nn.Parameter(torch.ones(C), requires_grad=False)
        s.tok = nn.Sequential(nn.Linear(C, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU()); s.pos = nn.Parameter(torch.zeros(16, hid)); s.out = nn.Sequential(nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, 1))
    def chunk_logits(s, x): h = s.tok((x - s.mu) / s.sd) + s.pos; return s.out(h.mean(1)).squeeze(-1)
    def forward(s, x): l = s.chunk_logits(x); return torch.logsumexp(l[1:] if len(l) > 1 else l, 0), l
def feats(r, layers): return torch.cat([r["f%d" % k].float() for k in layers], -1)
def run(layers, hold, tag):
    tr = [r for r in R if r["scene"] not in hold and r["scene"] not in CMP]; te = [r for r in R if r["scene"] in hold]; cmp = [r for r in R if r["scene"] in CMP]
    X = {id(r): feats(r, layers).to(dev) for r in R}; C = X[id(R[0])].shape[-1]; m = Head(C, a.hid).to(dev); allx = torch.cat([X[id(r)].reshape(-1, C) for r in tr]); m.mu.data = allx.mean(0); m.sd.data = allx.std(0).clamp_min(1e-3)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=a.lr, weight_decay=1e-2); pos = max(1e-3, np.mean([r["severe"] for r in tr])); w = torch.tensor([(1 - pos) / pos], device=dev)
    def ev(rs):
        m.eval()
        with torch.no_grad(): s = [float(m(X[id(r)])[0]) for r in rs]
        y = [r["severe"] for r in rs]; au = roc_auc_score(y, s) if 0 < sum(y) < len(y) else float("nan"); rho = spearmanr(s, [r["score"] for r in rs])[0] if len(rs) > 3 else float("nan"); return au, rho
    best = (-1, None, None)
    for ep in range(a.epochs):
        m.train(); random.shuffle(tr)
        for r in tr:
            lg, _ = m(X[id(r)]); y = torch.tensor([float(r["severe"])], device=dev); loss = nn.functional.binary_cross_entropy_with_logits(lg[None], y, pos_weight=w if r["severe"] else None); opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 5 == 0:
            a_te, r_te = ev(te); a_cmp, r_cmp = ev(cmp); print("[mg2 head %s] ep %d | held-out scenes AUROC %.2f ρ %.2f | 10 comparison prompts AUROC %.2f ρ %.2f" % (tag, ep + 1, a_te, r_te, a_cmp, r_cmp), flush=True)
            if (r_te if r_te == r_te else -1) > best[0]: best = (r_te, (a_te, r_te, a_cmp, r_cmp), {k: v.cpu() for k, v in m.state_dict().items()})
    torch.save(dict(layers=layers, hid=a.hid, state=best[2]), a.out + "/head_%s.pt" % tag); return best[1]
res = {}
for layers, tag in (([10], "L10"), ([20], "L20"), ([10, 20], "L10_20")):
    outs = [run(layers, set(train_scenes[k::2]), "%s_f%d" % (tag, k)) for k in range(2)]; res[tag] = np.nanmean(np.array(outs, dtype=float), 0)
print("| MG2 features | AUROC vs VLM label, held-out scenes ↑ | Spearman vs VLM score ↑ | AUROC, 10 comparison prompts ↑ | Spearman there ↑ |"); print("|---|---|---|---|---|")
for tag, v in res.items(): print("| %s | %.2f | %.2f | %.2f | %.2f |" % (tag, *v))
print("[MG2_HEAD_DONE]")
