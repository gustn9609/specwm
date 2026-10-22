"""Latent critic V_theta for the 1.3B-draft / 14B-target speculative loop.
Input: candidate chunk latent x [16,3,H,W], committed previous chunk x_prev [16,3,H,W] (zeros + flag for chunk 0), camera Plucker emb [C,3,H,W].
Heads: (a) logit 'target-generated' (density-ratio estimate p/q in context), (b) regression of the ③ contrast log(e_target/e_draft) (teacher: one-step errors).
Data: lingbot-world-v2/specwm_out/shadow/<task>_s<seed>/shadow.pt (spec_ar_shadow.py). Held-out TASKS for eval (sand jar, sandcastle + --holdout-extra random tasks).
usage: critic_train.py OUT_DIR [--epochs 30] [--dim 256] [--layers 6] [--holdout-extra 8] [--seed 0]"""
import argparse, os, sys, glob, json, random, time, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
S = "/home/hk28965/specwm"; SH = S + "/lingbot-world-v2/specwm_out/shadow"
sys.path.insert(0, S + "/lingbot-world-v2"); from specwm.critic_model import Critic, canon, pool_cam, CH, CW
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--dim", type=int, default=256); ap.add_argument("--layers", type=int, default=6)
ap.add_argument("--holdout-extra", type=int, default=8); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--lam", type=float, default=0.3, help="weight of the ③ regression loss"); ap.add_argument("--no-prev", action="store_true"); ap.add_argument("--no-cam", action="store_true"); ap.add_argument("--max-clips", type=int, default=0)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); random.seed(a.seed); torch.manual_seed(a.seed); dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# ---------------- data
files = sorted(glob.glob(SH + "/*/shadow.pt"))
if a.max_clips: files = files[:a.max_clips]
tasks = sorted({os.path.basename(os.path.dirname(f)).rsplit("_s", 1)[0] for f in files})
hold = {"sand_filling_up_jar", "sandcastle_wave_impact"}; rest = [t for t in tasks if t not in hold]; random.Random(a.seed).shuffle(rest); hold |= set(rest[:a.holdout_extra])
print("clips %d, tasks %d, held-out tasks %d: %s" % (len(files), len(tasks), len(hold), sorted(hold)), flush=True)
def load(f):
    d = torch.load(f, map_location="cpu"); task = d["task"]; out = []; prev = None
    for r in d["rows"]:
        flag = 0.0 if prev is None else 1.0; prev_t = torch.zeros_like(r["x_draft"]) if prev is None else prev; cam = pool_cam(r["cam"]).half()
        for lab, key in ((0, "x_draft"), (1, "x_target")):
            e = r["e3"].get("draft" if lab == 0 else "target", {})
            c3 = float(np.log(np.mean(e["e_target"]) / np.mean(e["e_draft"]))) if e else 0.0
            out.append(dict(task=task, seed=d["seed"], chunk=r["chunk"], y=lab, x=r[key], prev=prev_t, flag=flag, cam=cam, c3=c3, committed=r["committed"]))
        prev = r["x_target"] if r["committed"] == "target" else r["x_draft"]   # the chunk actually written into both KV caches
    return out
rows = [s for f in files for s in load(f)]
tr = [s for s in rows if s["task"] not in hold]; te = [s for s in rows if s["task"] in hold]
print("samples train %d  test %d" % (len(tr), len(te)), flush=True)
# latent normalisation from the training set (per channel)
# pre-canonicalise everything ONCE and keep it on the GPU as fp16 (the per-step CPU resize made an epoch take 12 min)
def prep(samples):
    for s_ in samples: s_["x"] = canon(s_["x"].float()).half().to(dev); s_["prev"] = canon(s_["prev"].float()).half().to(dev); s_["cam"] = s_["cam"].half().to(dev)
prep(tr); prep(te); print("tensors on %s: %.1f GB" % (dev, (sum(s_["x"].numel() + s_["prev"].numel() + s_["cam"].numel() for s_ in tr + te) * 2) / 1e9), flush=True)
xs = torch.stack([s_["x"].float() for s_ in tr[:1500]]); MEAN = xs.mean(dim=(0, 2, 3, 4)).view(1, 16, 1, 1, 1); STD = xs.std(dim=(0, 2, 3, 4)).view(1, 16, 1, 1, 1) + 1e-6; del xs
CAMC = tr[0]["cam"].shape[0]; H, W = CH, CW
def batch(samples):
    x = torch.stack([s["x"] for s in samples]).float(); p = torch.stack([s["prev"] for s in samples]).float(); cam = torch.stack([s["cam"] for s in samples]).float()
    x = (x - MEAN) / STD; p = (p - MEAN) / STD
    if a.no_prev: p = torch.zeros_like(p)
    if a.no_cam: cam = torch.zeros_like(cam)
    flag = torch.tensor([s["flag"] for s in samples], device=dev).view(-1, 1, 1, 1, 1).expand(-1, 1, *x.shape[2:])
    inp = torch.cat([x, p, flag], dim=1)
    return (inp.to(dev), cam.to(dev)), torch.tensor([s["y"] for s in samples], dtype=torch.float32, device=dev), torch.tensor([s["c3"] for s in samples], dtype=torch.float32, device=dev)
# ---------------- model
model = Critic(16 + 16 + 1, a.dim, a.layers, 3, H, W, camc=CAMC).to(dev); nparam = sum(p.numel() for p in model.parameters()); print("critic params %.2fM, input %s + cam %s" % (nparam / 1e6, (16 + 16 + 1, 3, H, W), tuple(tr[0]["cam"].shape)), flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05); steps = a.epochs * math.ceil(len(tr) / a.bs); sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.1)
@torch.no_grad()
def evaluate(samples, tag):
    model.eval(); lg, c3p, ys, c3s, meta = [], [], [], [], []
    for i in range(0, len(samples), 64):
        b = samples[i:i + 64]; inp, y, c3 = batch(b); l, r = model(inp); lg += l.cpu().tolist(); c3p += r.cpu().tolist(); ys += y.cpu().tolist(); c3s += c3.cpu().tolist(); meta += [(s["task"], s["seed"], s["chunk"]) for s in b]
    auc = roc_auc_score(ys, lg); auc3 = roc_auc_score(ys, [-c for c in c3s]) if len(set(c3s)) > 1 else float("nan"); corr = float(np.corrcoef(c3p, c3s)[0, 1])
    # paired accuracy: for each (task, seed, chunk) does the target candidate score higher than the draft candidate?
    d = {}
    for m, y, l in zip(meta, ys, lg): d.setdefault(m, {})[int(y)] = l
    pairs = [v for v in d.values() if 0 in v and 1 in v]; pacc = float(np.mean([v[1] > v[0] for v in pairs])) if pairs else float("nan")
    print("[%s] AUC critic %.3f | AUC ③ -log(eT/eD) %.3f | paired acc %.3f (n=%d) | ③-regression corr %.3f" % (tag, auc, auc3, pacc, len(pairs), corr), flush=True); model.train()
    return dict(auc=auc, auc_e3=auc3, paired_acc=pacc, corr_c3=corr, n=len(samples))
t0 = time.time(); hist = []
for ep in range(a.epochs):
    random.shuffle(tr); tl = 0.0; n = 0
    for i in range(0, len(tr), a.bs):
        inp, y, c3 = batch(tr[i:i + a.bs]); l, r = model(inp); loss = F.binary_cross_entropy_with_logits(l, y) + a.lam * F.mse_loss(r, c3)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); tl += loss.item() * len(y); n += len(y)
    if ep % 5 == 4 or ep == a.epochs - 1:
        print("epoch %d loss %.4f (%.0fs)" % (ep + 1, tl / n, time.time() - t0), flush=True); hist.append(dict(epoch=ep + 1, loss=tl / n, test=evaluate(te, "test")))
res = dict(train=evaluate(tr[:3000], "train-subset"), test=evaluate(te, "test"), holdout=sorted(hold), n_train=len(tr), n_test=len(te), params=nparam, args=vars(a), hist=hist)
torch.save(dict(state=model.state_dict(), mean=MEAN.cpu(), std=STD.cpu(), dim=a.dim, layers=a.layers, camc=CAMC, h=H, w=W, no_prev=a.no_prev, no_cam=a.no_cam), os.path.join(a.out, "critic.pt"))
json.dump(res, open(os.path.join(a.out, "result.json"), "w"), indent=1); print("[CRITIC_DONE]", json.dumps({k: res[k] for k in ("train", "test")}), flush=True)
