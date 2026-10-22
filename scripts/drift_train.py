"""Drift critic trained on HUMAN labels (plan A). Data: exp/drift_lat/*.pt (drift_dataset.py). Each clip = 13 chunk latents + clip-level label (severe / drift).
Model: specwm.critic_model.Critic on (x_chunk, x_prev, flag) + cam -> chunk logit. Multiple-instance loss: clip logit = logsumexp(chunk logits / T) * T
(smooth max), BCE vs the clip label; clean clips additionally push every chunk logit down. Splits: TEST = seeds in --test-seeds (any arm) ∪ arms in --test-arms (any seed).
Reports clip-level AUC (max over chunks) on test seeds, within-arm AUC on held-out arms, and the HSV baseline on the same clips.
usage: CUDA_VISIBLE_DEVICES=g drift_train.py OUT_DIR [--label severe|drift] [--epochs 40] [--dim 256 --layers 6] [--test-seeds 109,...] [--test-arms a,b]"""
import argparse, os, sys, glob, json, random, time, math, collections, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
S = "/home/hk28965/specwm"; DL = S + "/exp/drift_lat"
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--label", default="severe"); ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--dim", type=int, default=256); ap.add_argument("--layers", type=int, default=6)
ap.add_argument("--test-seeds", default="109,110,111,112,113,114,115,116,9999"); ap.add_argument("--test-arms", default="none,target,periodic_2,e3_0.4541,e3_0.5387,critic_main_-8.6183,critic_main_-8.7456,ir_-0.7,ir_-2.2,never1p3b")
ap.add_argument("--init", default="", help="load this critic.pt (weights + mean/std); with --epochs 0 = evaluation only"); ap.add_argument("--test-tasks", default="", help="clips of these tasks (any arm/seed) go to the test-task split"); ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--bs", type=int, default=16, help="clips per batch (each = 13 chunks)"); ap.add_argument("--temp", type=float, default=2.0); ap.add_argument("--lam", type=float, default=0.3)
ap.add_argument("--train-only-batch", default="blind_more,blind", help="clips from these batches are never used for testing (single-rater / recovered labels)"); ap.add_argument("--exclude-batch", default="", help="comma list of blind batches whose clips are dropped from TRAINING (leakage checks)"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--no-prev", action="store_true"); ap.add_argument("--no-cam", action="store_true"); ap.add_argument("--max-clips", type=int, default=0)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); random.seed(a.seed); torch.manual_seed(a.seed); dev = torch.device("cuda:0")
sys.path.insert(0, S + "/lingbot-world-v2"); from specwm.critic_model import Critic, canon, pool_cam, CH, CW
files = sorted(glob.glob(DL + "/*__s*.pt"))
if a.max_clips: files = files[:a.max_clips]
cams = {os.path.basename(f)[4:-3]: torch.load(f, map_location="cpu")["cam"] for f in glob.glob(DL + "/cam_*.pt")}
test_seeds = {int(x) for x in a.test_seeds.split(",")}; test_arms = set(a.test_arms.split(","))
clips = []
for f in files:
    d = torch.load(f, map_location="cpu"); y = float(d[a.label])
    lat = canon(torch.stack([d["lat"][:, 3 * k:3 * k + 3] for k in range(d["lat"].shape[1] // 3)]).to(dev).float()).half()   # [13,16,3,CH,CW], resized on the GPU
    cam = torch.stack([pool_cam(c.to(dev)) for c in cams[d["task"]][:lat.shape[0]]]).half()
    split = "test_task" if d["task"] in set(a.test_tasks.split(",")) else ("test_seed" if d["seed"] in test_seeds else ("test_arm" if d["arm"] in test_arms else "train"))
    if d.get("batch") in set(a.train_only_batch.split(",")):
        if split != "train": continue          # keep the 3-rater test sets unchanged
    if split == "train" and d.get("batch") in set(a.exclude_batch.split(",")): continue
    clips.append(dict(task=d["task"], arm=d["arm"], seed=d["seed"], y=y, lat=lat.to(dev), cam=cam.to(dev), hsv=np.array(d["hsv"]), split=split, cats=d.get("cats", [])))
tr = [c for c in clips if c["split"] == "train"]; te_s = [c for c in clips if c["split"] == "test_seed"]; te_a = [c for c in clips if c["split"] == "test_arm"]; te_t = [c for c in clips if c["split"] == "test_task"]
print("clips %d: train %d (pos %.2f) | test-seed %d (pos %.2f) | test-arm %d (pos %.2f)" % (len(clips), len(tr), np.mean([c["y"] for c in tr]), len(te_s), np.mean([c["y"] for c in te_s]), len(te_a), np.mean([c["y"] for c in te_a])), flush=True)
xs = torch.cat([c["lat"].float() for c in tr[:200]]); MEAN = xs.mean(dim=(0, 2, 3, 4)).view(1, 16, 1, 1, 1).to(dev); STD = (xs.std(dim=(0, 2, 3, 4)).view(1, 16, 1, 1, 1) + 1e-6).to(dev); del xs
if a.init:
    _ck = torch.load(a.init, map_location="cpu"); a.dim, a.layers = _ck["dim"], _ck["layers"]; MEAN, STD = _ck["mean"].to(dev), _ck["std"].to(dev)
model = Critic(16 + 16 + 1, a.dim, a.layers, 3, CH, CW, camc=384).to(dev); print("params %.2fM" % (sum(p.numel() for p in model.parameters()) / 1e6), flush=True)
def chunk_logits(c):
    x = (c["lat"].float() - MEAN) / STD; p = torch.cat([torch.zeros_like(x[:1]), x[:-1]]); flag = torch.ones(x.shape[0], 1, *x.shape[2:], device=dev); flag[0] = 0
    if a.no_prev: p = torch.zeros_like(p)
    cam = torch.zeros_like(c["cam"].float()) if a.no_cam else c["cam"].float()
    l, _ = model((torch.cat([x, p, flag], dim=1), cam)); return l   # [13]
def clip_logit(l): return a.temp * torch.logsumexp(l / a.temp, dim=0) - a.temp * math.log(l.shape[0])
if a.init: model.load_state_dict(_ck["state"]); print("[drift] init from", a.init, flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05); steps = max(1, a.epochs * math.ceil(len(tr) / a.bs)); sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.1)
@torch.no_grad()
def evaluate(cs, tag):
    model.eval(); ys = [c["y"] for c in cs]; mx = []; cl = []; hv = []; per = collections.defaultdict(lambda: ([], []))
    for c in cs:
        l = chunk_logits(c); mx.append(float(l.max())); cl.append(float(clip_logit(l))); hv.append(float(max(abs(v - 1) for s, v in c["hsv"]))); per[c["arm"]][0].append(c["y"]); per[c["arm"]][1].append(float(l.max()))
    model.train()
    if len(set(ys)) < 2: print("[%s] n %d all labels %s" % (tag, len(cs), ys[:1]), flush=True); return {}
    r = dict(n=len(cs), pos=float(np.mean(ys)), auc_max=roc_auc_score(ys, mx), auc_lse=roc_auc_score(ys, cl), auc_hsv=roc_auc_score(ys, hv))
    print("[%s] n %d pos %.2f | AUC critic max %.3f lse %.3f | HSV-value baseline %.3f" % (tag, r["n"], r["pos"], r["auc_max"], r["auc_lse"], r["auc_hsv"]), flush=True)
    for arm, (yy, ss) in sorted(per.items()):
        if len(set(yy)) > 1 and len(yy) >= 8: print("    within-arm %-32s n %3d pos %.2f AUC %.3f" % (arm, len(yy), np.mean(yy), roc_auc_score(yy, ss)), flush=True)
    return r
t0 = time.time()
for ep in range(a.epochs):
    random.shuffle(tr); tl = 0.0
    for i in range(0, len(tr), a.bs):
        b = tr[i:i + a.bs]; loss = 0.0
        for c in b:
            l = chunk_logits(c); loss = loss + F.binary_cross_entropy_with_logits(clip_logit(l), torch.tensor(c["y"], device=dev))
            if c["y"] == 0: loss = loss + a.lam * F.binary_cross_entropy_with_logits(l, torch.zeros_like(l))
        loss = loss / len(b); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); tl += loss.item() * len(b)
    if ep % 5 == 4 or ep == a.epochs - 1:
        print("epoch %d loss %.4f (%.0fs)" % (ep + 1, tl / len(tr), time.time() - t0), flush=True); evaluate(te_s, "test-seed"); evaluate(te_a, "test-arm")
@torch.no_grad()
def pair_acc(cs, tag):
    """within (task, arm-family) groups: fraction of (clean, severe) pairs where the critic scores the severe clip higher (= the DPO ranking task)"""
    model.eval(); sc = {}
    for c in cs: l = chunk_logits(c); sc[id(c)] = float(clip_logit(l))
    model.train(); G = collections.defaultdict(list)
    for c in cs: G[(c["task"], c["arm"].split("_")[0])].append(c)
    ok = n = 0
    for g in G.values():
        for x in g:
            for y in g:
                if x["y"] > y["y"]: n += 1; ok += sc[id(x)] > sc[id(y)]
    print("[%s] within-task pair accuracy %.3f over %d pairs" % (tag, ok / max(n, 1), n), flush=True); return ok / max(n, 1)
res = dict(train=evaluate(tr, "train"), test_seed=evaluate(te_s, "test-seed"), test_arm=evaluate(te_a, "test-arm"), test_task=evaluate(te_t, "test-task") if te_t else {}, pair_task=pair_acc(te_t, "test-task") if te_t else None, args=vars(a))
torch.save(dict(state=model.state_dict(), mean=MEAN.cpu(), std=STD.cpu(), dim=a.dim, layers=a.layers, camc=384, h=CH, w=CW, no_prev=a.no_prev, no_cam=a.no_cam, kind="drift", temp=a.temp), os.path.join(a.out, "critic.pt"))
json.dump(res, open(os.path.join(a.out, "result.json"), "w"), indent=1); print("[DRIFT_DONE]", json.dumps({k: res[k] for k in ("test_seed", "test_arm", "test_task", "pair_task")}), flush=True)
