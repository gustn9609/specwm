"""[10/16 trained device refiner, step 2] train RefinerNet on the two_machine --dump-pairs data.
Data: <pairs>/s<seed>/cloud/<task>/pairs.pt = {pairs: [{c, xd, eps, x}], ...}; held-out = --val-tasks (comma list) at all seeds.
Loss = MSE((xd + net) - x) / Var(x - xd) per batch, i.e. 1.0 = the untouched draft, 0 = perfect; reports val rel-MSE and the per-chunk
'residual energy explained'. usage: python -m specwm.refiner_train --pairs DIR --out DIR [--steps 20000 --bs 16 --lr 2e-4 --val-tasks a,b,c]"""
import argparse, glob, json, os, time, math, random, torch
from specwm.refiner_net import RefinerNet
ap = argparse.ArgumentParser(); ap.add_argument("--pairs", required=True); ap.add_argument("--out", required=True); ap.add_argument("--steps", type=int, default=20000)
ap.add_argument("--bs", type=int, default=16); ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--val-tasks", default=""); ap.add_argument("--val-every", type=int, default=1000)
ap.add_argument("--c0", type=int, default=128); ap.add_argument("--c1", type=int, default=256); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--no-eps", type=int, default=0, help="ablation: zero the eps input")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); torch.manual_seed(a.seed); random.seed(a.seed); dev = "cuda:0"
def log(*m): print("[refiner %s]" % time.strftime("%H:%M:%S"), *m, flush=True); open(a.out + "/log.txt", "a").write(" ".join(str(x) for x in m) + "\n")
VAL = set(t for t in a.val_tasks.split(",") if t); tr, va = [], []
for f in sorted(glob.glob(a.pairs + "/s*/cloud/*/pairs.pt")):
    d = torch.load(f, map_location="cpu"); task = d["task"]
    for p in d["pairs"]: (va if task in VAL else tr).append((p["xd"], p["eps"], p["x"], task, d["seed"], p["c"]))
log("train chunks", len(tr), "val chunks", len(va), "val tasks", sorted(VAL), "grid", tuple(tr[0][0].shape))
_s = torch.stack([t[2].float().mean((1, 2, 3)) for t in tr[:3000]]); _q = torch.stack([(t[2].float() ** 2).mean((1, 2, 3)) for t in tr[:3000]]); mu = _s.mean(0); sd = (_q.mean(0) - mu ** 2).clamp_min(1e-6).sqrt()
GROUPS = {}   # tasks have different latent grids (landscape / portrait): batches are drawn within one grid shape
for t in tr: GROUPS.setdefault(tuple(t[0].shape), []).append(t)
GW = [len(v) for v in GROUPS.values()]; GK = list(GROUPS.keys())
net = RefinerNet(c0=a.c0, c1=a.c1).to(dev); net.mu.copy_(mu.to(dev)); net.sd.copy_(sd.to(dev)); log("params M", sum(p.numel() for p in net.parameters()) / 1e6, "mu", [round(v, 3) for v in mu.tolist()[:4]], "sd", [round(v, 3) for v in sd.tolist()[:4]])
opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.99)); sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 500) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
def batch(items):
    xd = torch.stack([i[0] for i in items]).to(dev).float(); eps = torch.stack([i[1] for i in items]).to(dev).float(); x = torch.stack([i[2] for i in items]).to(dev).float()
    if a.no_eps: eps = torch.zeros_like(eps)
    return xd, eps, x
def rel_loss(pred, xd, x):
    r = x - xd; return ((pred - r) ** 2).mean() / (r ** 2).mean().clamp_min(1e-8)
@torch.no_grad()
def validate():
    net.eval(); tot, n, per = 0.0, 0, {}
    for it in [[v] for v in va]:
        xd, eps, x = batch(it); pred = net(xd, eps)
        for j in range(len(it)):
            l = float(rel_loss(pred[j:j + 1], xd[j:j + 1], x[j:j + 1])); tot += l; n += 1; per.setdefault(it[j][3], []).append(l)
    net.train(); return tot / max(n, 1), {k: round(sum(v) / len(v), 3) for k, v in per.items()}
best = 9.0; t0 = time.time(); ema = None
for step in range(1, a.steps + 1):
    gk = random.choices(GK, weights=GW)[0]; xd, eps, x = batch(random.sample(GROUPS[gk], min(a.bs, len(GROUPS[gk]))))
    with torch.autocast("cuda", dtype=torch.bfloat16): pred = net(xd, eps)
    loss = rel_loss(pred.float(), xd, x); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
    ema = float(loss) if ema is None else 0.98 * ema + 0.02 * float(loss)
    if step % 100 == 0: log("step", step, "train rel-mse ema %.4f" % ema, "lr %.2e" % sched.get_last_lr()[0], "%.1f min" % ((time.time() - t0) / 60))
    if step % a.val_every == 0 or step == a.steps:
        v, per = validate(); log("VAL step", step, "rel-mse %.4f" % v, "(1.0 = raw draft)", per)
        torch.save(dict(state=net.state_dict(), c0=a.c0, c1=a.c1, step=step, val=v, args=vars(a)), a.out + "/refiner_s%d.pt" % step)
        if v < best: best = v; torch.save(dict(state=net.state_dict(), c0=a.c0, c1=a.c1, step=step, val=v, args=vars(a)), a.out + "/refiner_best.pt")
json.dump(dict(best_val=best, train=len(tr), val=len(va)), open(a.out + "/result.json", "w")); log("DONE best val rel-mse %.4f" % best)
