#!/usr/bin/env python3
"""R1/R2 vs E1': KV-rewrite recovery with a 360p-class (320x560) corrected target vs the 480p target.
recovery = (fast_alone - arm) / (fast_alone - resync_k6[ideal]) on DINO chunks 6-12 to Base70 (same seed). Also target-video DINO to Base and correction time."""
import os, json, glob, numpy as np, torch, imageio
S = os.environ.get("SCRATCH", "/home/hk28965/specwm"); R = S + "/lingbot-world-v2/specwm_out"; torch.hub.set_dir(S + "/.torch_hub")
m = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True).eval().cuda()
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).cuda(); std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).cuda()
def feats(V):
    X = torch.from_numpy(np.ascontiguousarray(V)).permute(0, 3, 1, 2).float().cuda() / 255; X = torch.nn.functional.interpolate(X, size=(518, 518), mode="bilinear"); X = (X - mean) / std; o = []
    with torch.no_grad():
        for i in range(0, len(X), 8): o.append(torch.nn.functional.normalize(m(X[i:i + 8]).flatten(1), dim=-1).cpu())
    return torch.cat(o)
def dist(FA, FB): n = min(len(FA), len(FB)); return (1 - (FA[:n] * FB[:n]).sum(-1)).numpy()
def pc(d): return [float(d[c * 3:(c + 1) * 3].mean()) for c in range(len(d) // 3)]
def npy(p): return np.load(p) if os.path.exists(p) else None
def vid4(p): return np.stack([np.asarray(f) for f in imageio.get_reader(p)])[::4] if os.path.exists(p) else None
rows = []; out = {}
for sc in ["05", "01", "02"]:
    for sd in [1234, 4321, 9999]:
        FB = feats(np.load(R + "/floor/base/S%s_s%d/frames_all.npy" % (sc, sd)))
        e = R + "/e1repro/S%s/W18_sink6" % sc; r = R + "/r1_360/S%s/W18_sink6" % sc
        fa = npy(e + "/fast_alone_s%d/frames_all.npy" % sd); k6 = npy(e + "/resync_k6_s%d/frames_all.npy" % sd); kl = npy(e + "/kvrw_laundered_s%d/frames_all.npy" % sd); kr = npy(r + "/kvrw_laundered_s%d/frames_all.npy" % sd)
        if fa is None or k6 is None: continue
        d_fa = np.mean(pc(dist(feats(fa), FB))[6:]); d_k6 = np.mean(pc(dist(feats(k6), FB))[6:]); rec = lambda x: None if x is None else float((d_fa - np.mean(pc(dist(feats(x), FB))[6:])) / (d_fa - d_k6))
        t480 = vid4(R + "/e1repro/S%s/target_s%d/video.mp4" % (sc, sd)); t360 = vid4(R + "/r1_360/S%s/target_s%d/video.mp4" % (sc, sd))
        tq = lambda v: None if v is None else float(np.mean(dist(feats(v), FB)))
        ct = R + "/r1_360/S%s/target_s%d/correction_time.json" % (sc, sd); ct = json.load(open(ct)) if os.path.exists(ct) else {}
        row = dict(scene=sc, seed=sd, fast_alone=round(d_fa, 3), ideal_k6=round(d_k6, 3), rec_480=rec(kl), rec_360=rec(kr), target_dino_480=tq(t480), target_dino_360=tq(t360), launder_s=ct.get("launder_s"), repaint_s=ct.get("repaint_s"))
        rows.append(row); out["S%s_s%d" % (sc, sd)] = row; print(row, flush=True)
json.dump(out, open(R + "/r1_360/r1_recovery.json", "w"), indent=1)
def ms(k): v = [r[k] for r in rows if r.get(k) is not None]; return ("%.0f%% ± %.0f%% (n=%d)" % (100 * np.mean(v), 100 * np.std(v), len(v))) if v else "-"
def mq(k): v = [r[k] for r in rows if r.get(k) is not None]; return ("%.3f" % np.mean(v)) if v else "-"
md = "| | recovery (mean ± std) | target DINO to Base70 | launder+RePaint time |\n|---|---|---|---|\n"
md += "| E1' 480p target | %s | %s | 225 s + 766 s (wall, incl. ~70 s model load each; from q94 logs) |\n" % (ms("rec_480"), mq("target_dino_480"))
md += "| R1/R2 320x560 target, pixel-upsampled provider | %s | %s | %s s + %s s |\n" % (ms("rec_360"), mq("target_dino_360"), mq("launder_s"), mq("repaint_s"))
md += "\n| scene | seed | fast_alone | ideal(k6) | rec 480p | rec 360p | tgt DINO 480 | tgt DINO 360 |\n|---|---|---|---|---|---|---|---|\n"
for r in rows: md += "| S%s | %d | %.3f | %.3f | %s | %s | %s | %s |\n" % (r["scene"], r["seed"], r["fast_alone"], r["ideal_k6"], *["-" if r[k] is None else ("%.0f%%" % (100 * r[k]) if k.startswith("rec") else "%.3f" % r[k]) for k in ("rec_480", "rec_360", "target_dino_480", "target_dino_360")])
open(R + "/r1_360/r1_recovery.md", "w").write(md); print(md)
