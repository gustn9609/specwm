"""Binary drift protocol v3 (action item 4; fixes the review of drift_rate2):
- features: DINOv2-B/14 per frame -> mean per 12-frame chunk -> RE-NORMALISED; clips compared over their common chunks (Fast-based arms: 153 frames, Base: 161).
- score(clip, refs) = min over reference clips of the TRAJECTORY distance (mean over chunks of 1 - cos), i.e. one whole Base clip per match (no chimeras).
- Base calibration: leave-one-out scores s_i = score(base_i, other Base clips). A method clip is scored against every leave-one-out subset of the
  Base batch (n-1 refs, same as calibration) and the subset scores are averaged -> no circularity advantage.
- conformal p-value = (1 + #{s_i >= score}) / (n + 1); drift := p <= 1/(n+1) (score above every Base LOO score). With n Base seeds the level is 1/(n+1)
  (n = 8 -> 0.11; 19 seeds are needed for 0.05). Also reports a separate collapse/freeze flag: mean luminance > 230 or HSV saturation < 12 in >= 3 chunks
  (wash-out), or median inside-chunk motion < 0.5 x the Base minimum (freeze).
- per prompt: no-drift rate per arm with Wilson 95% CI, and Boschloo exact test vs Base (scipy); pooled Cochran-Mantel-Haenszel style is left to the reader.
usage: drift_rate3.py TASK SEEDS ARM1,ARM2,...    (ARM = run prefix; 'base70' is the reference batch)"""
import sys, os, json, math, numpy as np, torch, imageio, cv2
from scipy import stats
S = "/home/hk28965/specwm"; V0 = S + "/lingbot-world-v2/specwm_out/spec_v0"; dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.hub.set_dir(S + "/.torch_hub"); dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True).eval().to(dev)
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
@torch.no_grad()
def load(p):
    V = [np.asarray(f) for f in imageio.get_reader(p)]; out = []
    for i in range(0, len(V), 16):
        x = torch.from_numpy(np.stack(V[i:i + 16])).to(dev).permute(0, 3, 1, 2).float() / 255; x = torch.nn.functional.interpolate(x, size=(224, 392), mode="bilinear", antialias=True)
        out.append(torch.nn.functional.normalize(dino((x - MEAN) / STD).float(), dim=-1).cpu())
    F = torch.cat(out); n = (len(F) // 12) * 12
    C = torch.nn.functional.normalize(torch.stack([F[i:i + 12].mean(0) for i in range(0, n, 12)]), dim=-1)
    lum = [float(np.mean(V[i])) for i in range(0, n, 12)]; sat = [float(cv2.cvtColor(cv2.resize(V[i], (64, 36)), cv2.COLOR_RGB2HSV)[..., 1].mean()) for i in range(0, n, 12)]
    g = [cv2.cvtColor(cv2.resize(f, (212, 116)), cv2.COLOR_RGB2GRAY).astype(np.float32) for f in V]
    mot = [float(np.median([np.abs(g[j + 1] - g[j]).mean() for j in range(i, min(i + 11, len(g) - 1))])) for i in range(0, n, 12)]
    return dict(C=C, lum=lum, sat=sat, mot=mot)
def traj(a, b):
    k = min(len(a), len(b)); return float((1 - (a[:k] * b[:k]).sum(-1)).mean())
def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"), float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d; return p, max(0, c - h), min(1, c + h)
task, seeds, arms = sys.argv[1], [int(s) for s in sys.argv[2].split(",")], sys.argv[3].split(",")
B = {sd: load("%s/%s/base70_s%d/video.mp4" % (V0, task, sd)) for sd in seeds if os.path.exists("%s/%s/base70_s%d/video.mp4" % (V0, task, sd))}
bs = sorted(B); n = len(B)
loo = {sd: min(traj(B[sd]["C"], B[o]["C"]) for o in bs if o != sd) for sd in bs}; tau = max(loo.values())
mot_min = min(float(np.median(B[sd]["mot"])) for sd in bs)
def score(Cm):
    return float(np.mean([min(traj(Cm, B[o]["C"]) for o in bs if o != drop) for drop in bs]))
def flags(d):
    wash = sum(1 for l, s in zip(d["lum"], d["sat"]) if l > 230 or s < 12) >= 3; frz = float(np.median(d["mot"])) < 0.5 * mot_min
    return wash, frz
res = {"task": task, "n_base": n, "base_seeds": bs, "base_loo": loo, "tau_max_loo": tau, "level": 1 / (n + 1), "base_motion_min": mot_min, "arms": {}}
print("%s: Base n=%d seeds %s | LOO scores %s | tau = max = %.3f | test level 1/(n+1) = %.3f | Base min median motion %.2f" % (task, n, bs, " ".join("%.3f" % loo[s] for s in bs), tau, 1 / (n + 1), mot_min), flush=True)
bw = [flags(B[s]) for s in bs]; base_ok = sum(1 for s, (w, f) in zip(bs, bw) if not (w or f))   # Base clips: DINO criterion passes by construction (LOO <= tau); report the detector flags
res["arms"]["base70"] = dict(n=n, nodrift=base_ok, note="DINO criterion passes by construction; flags only")
print("  %-40s n=%2d  no-drift %d/%d (flags only)  wash/freeze flags: %s" % ("base70", n, base_ok, n, [(s, w, f) for s, (w, f) in zip(bs, bw) if w or f]), flush=True)
for arm in arms:
    rows = {}
    for sd in seeds:
        p = "%s/%s/%s_s%d/video.mp4" % (V0, task, arm, sd)
        if not os.path.exists(p): continue
        d = load(p); sc = score(d["C"]); pval = (1 + sum(1 for s in loo.values() if s >= sc)) / (n + 1); w, f = flags(d)
        rows[sd] = dict(score=round(sc, 4), p=round(pval, 3), dino_drift=bool(sc > tau), wash=w, freeze=f, drift=bool(sc > tau or w or f))
    k = sum(1 for r in rows.values() if not r["drift"]); m = len(rows); pr, lo, hi = wilson(k, m)
    try: pb = stats.boschloo_exact([[k, m - k], [base_ok, n - base_ok]]).pvalue if m else float("nan")
    except Exception: pb = float("nan")
    res["arms"][arm] = dict(n=m, nodrift=k, rate=pr, ci=[lo, hi], boschloo_vs_base=pb, clips=rows)
    print("  %-40s n=%2d  no-drift %d/%d = %.2f [%.2f, %.2f]  Boschloo p vs Base %.3f  | %s" % (arm, m, k, m, pr, lo, hi, pb, " ".join("%d:%.3f%s%s%s" % (sd, r["score"], "D" if r["dino_drift"] else "", "W" if r["wash"] else "", "F" if r["freeze"] else "") for sd, r in sorted(rows.items()))), flush=True)
json.dump(res, open("%s/%s/drift_rate3.json" % (V0, task), "w"), indent=1)
