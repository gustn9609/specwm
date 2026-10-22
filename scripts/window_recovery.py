#!/usr/bin/env python3
"""Exp 2 + Exp 6 scorer. recovery = (fast_alone - arm)/(fast_alone - resync_k6) on DINO chunks 6-12 to Base70 (matching seed).
Exp 2: e1repro 480p, arms resync_k{1,2,3,6} / kvrw_laundered{_k1,_k2,_k3, k6}. Exp 6: draft360_reground (320x560 2-step) arms fast_alone/resync_k6/kvrw_laundered/kvrw_laundered480."""
import os, json, numpy as np, torch
S = "/home/hk28965/specwm"; R = S + "/lingbot-world-v2/specwm_out"; torch.hub.set_dir(S + "/.torch_hub")
m = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True).eval().cuda()
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).cuda(); std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).cuda()
def feats(V):
    X = torch.from_numpy(np.ascontiguousarray(V)).permute(0, 3, 1, 2).float().cuda() / 255; X = torch.nn.functional.interpolate(X, size=(518, 518), mode="bilinear"); X = (X - mean) / std; o = []
    with torch.no_grad():
        for i in range(0, len(X), 8): o.append(torch.nn.functional.normalize(m(X[i:i + 8]).flatten(1), dim=-1).cpu())
    return torch.cat(o)
def dist(FA, FB): n = min(len(FA), len(FB)); return (1 - (FA[:n] * FB[:n]).sum(-1)).numpy()
def pc(d): return [float(d[c * 3:(c + 1) * 3].mean()) for c in range(len(d) // 3)]
def c612(V, FB): return float(np.mean(pc(dist(feats(V), FB))[6:]))
def npy(p): return np.load(p) if os.path.exists(p) else None
out = {}
for exp, root, arms in [("e1repro_480p", R + "/e1repro", ["fast_alone", "resync_k6", "resync_k3", "resync_k2", "resync_k1", "kvrw_laundered", "kvrw_laundered_k3", "kvrw_laundered_k2", "kvrw_laundered_k1"]),
                        ("draft360_2step", R + "/draft360_reground", ["fast_alone", "resync_k6", "kvrw_laundered", "kvrw_laundered480"])]:
    rows = {}
    for sc in ["05", "01", "02"]:
        for sd in [1234, 4321, 9999]:
            FB = feats(np.load(R + "/floor/base/S%s_s%d/frames_all.npy" % (sc, sd))); d = {}
            for arm in arms:
                V = npy("%s/S%s/W18_sink6/%s_s%d/frames_all.npy" % (root, sc, arm, sd))
                if V is not None: d[arm] = c612(V, FB)
            rows["S%s_s%d" % (sc, sd)] = d; print(exp, sc, sd, {k: round(v, 3) for k, v in d.items()}, flush=True)
    out[exp] = rows
json.dump(out, open(R + "/window_recovery.json", "w"), indent=1)
md = "# Exp 2: KV-rewrite window sweep (480p targets from E1'; DINO chunks 6-12 to Base70; recovery = (fast-arm)/(fast-ideal_k6))\n\n| window k (s) | ideal: resync_k (DINO / recovery vs k6 ideal) | ours: kvrw_laundered_k (DINO / recovery) |\n|---|---|---|\n"
rows = out["e1repro_480p"]
def agg(arm, ideal="resync_k6"):
    rec = []; dd = []
    for r in rows.values():
        if arm in r and "fast_alone" in r and ideal in r and r["fast_alone"] - r[ideal] > 1e-6: rec.append((r["fast_alone"] - r[arm]) / (r["fast_alone"] - r[ideal])); dd.append(r[arm])
    return (np.mean(dd), np.mean(rec) * 100, np.std(rec) * 100, len(rec)) if rec else (np.nan, np.nan, np.nan, 0)
md += "| fast_alone | %.3f | – |\n" % np.mean([r["fast_alone"] for r in rows.values() if "fast_alone" in r])
for k, s in [(6, 4.5), (3, 2.25), (2, 1.5), (1, 0.75)]:
    i = agg("resync_k%d" % k); o = agg("kvrw_laundered" if k == 6 else "kvrw_laundered_k%d" % k)
    md += "| %d (%.2f s) | %.3f / %.0f%% ± %.0f%% (n=%d) | %.3f / %.0f%% ± %.0f%% (n=%d) |\n" % (k, s, *i, *o)
md += "\n# Exp 6: 360p (320x560) 2-step draft + regrounding (same metric; Base ctx / providers resized to the draft grid)\n\n| arm | DINO 6-12 to Base70 | recovery vs own ideal | n |\n|---|---|---|---|\n"
rows = out["draft360_2step"]
md += "| fast_alone (360p 2-step) | %.3f | – | %d |\n" % (np.mean([r["fast_alone"] for r in rows.values() if "fast_alone" in r]), sum("fast_alone" in r for r in rows.values()))
for arm, lab in [("resync_k6", "ideal: Base70 ctx 0-5 (resized to 320x560)"), ("kvrw_laundered", "ours: 360p target (r1_360, native res)"), ("kvrw_laundered480", "ours: 480p target (E1') downsampled")]:
    v = agg(arm); md += "| %s | %.3f | %.0f%% ± %.0f%% | %d |\n" % (lab, *v)
md += "\nReference (480p 4-step draft, E1'): fast_alone %.3f, ideal recovery 100%%, ours 65%% ± 29%%.\n" % np.mean([r["fast_alone"] for r in out["e1repro_480p"].values()])
open(R + "/window_recovery.md", "w").write(md); print(md)
