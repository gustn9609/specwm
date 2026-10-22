"""Trajectory critic vs final-latent critic on human labels (1.3B draft rollouts, blind_rl + blind_dpo_big; trajectories from specwm/draft_traj.py).
Input per chunk = x0-hat of the chosen draft nodes concatenated on channels (+ flag), camera Plucker emb; same Critic / MIL loss as drift_train.py.
Task-grouped K-fold: writes out-of-fold clip scores. usage: CUDA_VISIBLE_DEVICES=g traj_critic_train.py OUT.json --nodes 0,1,2,3 [--label severe] [--seed 0]"""
import argparse, os, sys, glob, json, random, math, time, collections, numpy as np, torch, torch.nn.functional as F
S = "/home/hk28965/specwm"; TD = S + "/lingbot-world-v2/specwm_out/draft_traj"
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--nodes", default="3"); ap.add_argument("--label", default="severe"); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--folds", type=int, default=5); ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--dim", type=int, default=192); ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--bs", type=int, default=16); ap.add_argument("--temp", type=float, default=2.0); ap.add_argument("--lam", type=float, default=0.3); ap.add_argument("--delta", action="store_true", help="feed node3 + (node_k - node3) instead of raw nodes")
a = ap.parse_args(); random.seed(a.seed); torch.manual_seed(a.seed); np.random.seed(a.seed); dev = torch.device("cuda:0"); NODES = [int(x) for x in a.nodes.split(",")]
sys.path.insert(0, S + "/lingbot-world-v2"); from specwm.critic_model import Critic, canon, pool_cam, CH, CW
LAB = json.load(open(S + "/exp/kveval/traj_labels.json")); clips = []; cams = {}
for f in sorted(glob.glob(TD + "/*.pt")):
    d = torch.load(f, map_location="cpu"); k = "%s|%s|%d" % (d["arm"], d["task"], d["seed"])
    if k not in LAB: continue
    tr = d["traj"].to(dev).float()                                   # [4,16,39,h,w]
    if a.delta: X = torch.cat([tr[3]] + [tr[n] - tr[3] for n in NODES if n != 3], 0)
    else: X = torch.cat([tr[n] for n in NODES], 0)                   # [16K,39,h,w]
    C = X.shape[0]; X = torch.stack([X[:, 3 * c:3 * c + 3] for c in range(13)])   # [13,C,3,h,w]
    X = canon(X).half()
    if d["task"] not in cams: cams[d["task"]] = torch.stack([pool_cam(c.to(dev)) for c in torch.load(S + "/exp/drift_lat/cam_%s.pt" % d["task"], map_location="cpu")["cam"][:13]]).half()
    clips.append(dict(key=k, task=d["task"], arm=d["arm"], y=float(LAB[k][a.label]), y_drift=float(LAB[k]["drift"]), x=X, cam=cams[d["task"]]))
CIN = clips[0]["x"].shape[1]; print("clips %d, in-channels %d, pos %.2f" % (len(clips), CIN, np.mean([c["y"] for c in clips])), flush=True)
tasks = sorted({c["task"] for c in clips}); rng = random.Random(a.seed); rng.shuffle(tasks)
# balance folds: jar / castle (many clips) go to different folds
fold_of = {t: i % a.folds for i, t in enumerate(sorted(tasks, key=lambda t: -sum(c["task"] == t for c in clips)))}
oof = {}
def clip_logit(l): return a.temp * torch.logsumexp(l / a.temp, dim=0) - a.temp * math.log(l.shape[0])
for fo in range(a.folds):
    tr = [c for c in clips if fold_of[c["task"]] != fo]; te = [c for c in clips if fold_of[c["task"]] == fo]
    xs = torch.cat([c["x"].float() for c in tr[:150]]); MEAN = xs.mean(dim=(0, 2, 3, 4)).view(1, CIN, 1, 1, 1); STD = xs.std(dim=(0, 2, 3, 4)).view(1, CIN, 1, 1, 1) + 1e-6; del xs
    model = Critic(CIN + 1, a.dim, a.layers, 3, CH, CW, camc=384).to(dev)
    def logits(c):
        x = (c["x"].float() - MEAN) / STD; flag = torch.ones(x.shape[0], 1, *x.shape[2:], device=dev); flag[0] = 0
        l, _ = model((torch.cat([x, flag], 1), c["cam"].float())); return l
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05); steps = a.epochs * math.ceil(len(tr) / a.bs); sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.1)
    pw = (len(tr) - sum(c["y"] for c in tr)) / max(sum(c["y"] for c in tr), 1)   # class balance: weight positives down
    for ep in range(a.epochs):
        random.shuffle(tr)
        for i in range(0, len(tr), a.bs):
            loss = 0.0
            for c in tr[i:i + a.bs]:
                l = logits(c); w = pw if c["y"] == 1 else 1.0
                loss = loss + w * F.binary_cross_entropy_with_logits(clip_logit(l), torch.tensor(c["y"], device=dev))
                if c["y"] == 0: loss = loss + a.lam * F.binary_cross_entropy_with_logits(l, torch.zeros_like(l))
            opt.zero_grad(); (loss / len(tr[i:i + a.bs])).backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
    model.eval()
    with torch.no_grad():
        for c in te: l = logits(c); oof[c["key"]] = dict(score=float(clip_logit(l)), chunk=[round(float(v), 3) for v in l], y=c["y"], y_drift=c["y_drift"], task=c["task"], arm=c["arm"])
    print("[fold %d] train %d test %d done" % (fo, len(tr), len(te)), flush=True)
from sklearn.metrics import roc_auc_score
ys = [v["y"] for v in oof.values()]; sc = [v["score"] for v in oof.values()]; yd = [v["y_drift"] for v in oof.values()]
res = dict(nodes=a.nodes, delta=a.delta, seed=a.seed, label=a.label, auc=roc_auc_score(ys, sc), auc_drift=roc_auc_score(yd, sc), oof=oof)
json.dump(res, open(a.out, "w")); print("[TRAJ_CRITIC_DONE] nodes %s delta %s seed %d AUC(%s) %.3f AUC(drift) %.3f" % (a.nodes, a.delta, a.seed, a.label, res["auc"], res["auc_drift"]), flush=True)
