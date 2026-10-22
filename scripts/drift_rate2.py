"""Binary drift protocol v2 (action item 4), batch-vs-batch: for a task, the TARGET batch = Base-70 rollouts over seeds. A clip is 'no drift'
if its per-chunk DINO features stay within the Base batch: for each chunk take the min distance to the same chunk of any Base seed
(nearest-neighbour), average over chunks; the threshold tau is the 95th percentile of the leave-one-out Base self-distance (each Base clip
vs the other seeds). Reports the no-drift rate per method with Wilson 95% CIs.
usage: drift_rate2.py TASK METHOD1,METHOD2,... SEEDS   (METHOD = run prefix, e.g. never, plh_plan6_ck400_cr_st0_never)"""
import sys, os, json, math, numpy as np, torch, imageio
S = "/home/hk28965/specwm"; V0 = S + "/lingbot-world-v2/specwm_out/spec_v0"; dev = torch.device("cuda:0")
torch.hub.set_dir(S + "/.torch_hub"); dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True).eval().to(dev)
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
@torch.no_grad()
def feats(p):
    V = [np.asarray(f) for f in imageio.get_reader(p)]; out = []
    for i in range(0, len(V), 16):
        x = torch.from_numpy(np.stack(V[i:i + 16])).to(dev).permute(0, 3, 1, 2).float() / 255; x = torch.nn.functional.interpolate(x, size=(224, 392), mode="bilinear", antialias=True)
        out.append(torch.nn.functional.normalize(dino((x - MEAN) / STD).float(), dim=-1).cpu())
    F = torch.cat(out); n = (len(F) // 12) * 12; return torch.stack([F[i:i + 12].mean(0) for i in range(0, n, 12)])   # chunk-mean features [C, D]
def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"), 0, 0)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d; return p, max(0, c - h), min(1, c + h)
task, methods, seeds = sys.argv[1], sys.argv[2].split(","), [int(s) for s in sys.argv[3].split(",")]
B = {sd: feats("%s/%s/base70_s%d/video.mp4" % (V0, task, sd)) for sd in seeds if os.path.exists("%s/%s/base70_s%d/video.mp4" % (V0, task, sd))}
def nn_dist(F, ref):   # mean over chunks of min over ref clips of (1 - cos) at the same chunk index
    C = min([len(F)] + [len(r) for r in ref]); return float(np.mean([min(1 - float((F[c] * r[c]).sum()) for r in ref) for c in range(C)]))
loo = [nn_dist(B[sd], [B[o] for o in B if o != sd]) for sd in B]; tau = float(np.percentile(loo, 95)) if len(loo) > 2 else max(loo)
print("%s: Base batch %d seeds; leave-one-out self-distance %s -> tau (95th pct) = %.3f" % (task, len(B), [round(x, 3) for x in sorted(loo)], tau), flush=True)
out = {"task": task, "tau": tau, "base_loo": loo, "methods": {}}
for m in methods:
    ds = {}
    for sd in seeds:
        p = "%s/%s/%s_s%d/video.mp4" % (V0, task, m, sd)
        if os.path.exists(p): ds[sd] = nn_dist(feats(p), list(B.values()))
    k = sum(1 for d in ds.values() if d <= tau); p_, lo, hi = wilson(k, len(ds))
    print("  %-42s n=%2d  no-drift %.2f [%.2f, %.2f]   dist: %s" % (m, len(ds), p_, lo, hi, " ".join("%d:%.2f%s" % (sd, d, "" if d <= tau else "*") for sd, d in sorted(ds.items()))), flush=True)
    out["methods"][m] = dict(n=len(ds), ok=k, rate=p_, ci=[lo, hi], dist=ds)
json.dump(out, open("%s/%s/drift_rate2.json" % (V0, task), "w"), indent=1)
