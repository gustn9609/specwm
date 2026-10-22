"""[baselines/sf1p3b] cheap objective frame statistics for the "what do the frames show" check of an early-exit clip against its uniform:4 reference:
per 24-frame window: sharpness = variance of the 3x3 Laplacian of the grey frame (drops when a clip smears / collapses), inter-frame change = mean
|frame_t - frame_{t-1}| (grey, 0-255), and the pixel distance to the reference clip at the same frame id (mean |a - b|, 0-255; the early-exit arms
diverge from the 4-step trajectory as soon as tokens are accepted, so this is a divergence measure, not a quality one). Frames are downscaled to
width 416. usage: python frame_stats.py REF.mp4 CLIP.mp4 [CLIP.mp4 ...] [--win 24] [--json OUT]"""
import sys, json, argparse, numpy as np, imageio.v2 as imageio
from PIL import Image
ap = argparse.ArgumentParser(); ap.add_argument("ref"); ap.add_argument("clips", nargs="*"); ap.add_argument("--win", type=int, default=24); ap.add_argument("--json", default=""); a = ap.parse_args()
def load(v, w=416):
    fr = imageio.mimread(v, memtest=False); out = []
    for f in fr: im = Image.fromarray(f); im = im.resize((w, int(w * im.height / im.width))); out.append(np.asarray(im).astype(np.float32))
    return np.stack(out)
def lap_var(g):   # g [H, W] grey
    l = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]; return float(l.var())
def stats(V, R=None):
    G = V.mean(-1); n = len(G); rows = []
    for s in range(0, n, a.win):
        e = min(n, s + a.win); sh = float(np.mean([lap_var(G[i]) for i in range(s, e)])); ch = float(np.mean([np.abs(G[i] - G[i - 1]).mean() for i in range(max(1, s), e)])) if e > 1 else 0.0
        d = float(np.mean([np.abs(V[i] - R[i]).mean() for i in range(s, min(e, len(R)))])) if R is not None else None
        rows.append(dict(f0=s, f1=e - 1, sharp=round(sh, 1), change=round(ch, 2), dist_ref=None if d is None else round(d, 1)))
    return rows
R = load(a.ref); OUT = {a.ref: stats(R)}
for c in a.clips: OUT[c] = stats(load(c), R)
for v, rows in OUT.items():
    print(v.split("/")[-2] + "/" + v.split("/")[-1]); print("   window      " + "  ".join("f%3d-%3d" % (r["f0"], r["f1"]) for r in rows)); print("   sharpness   " + "  ".join("%8.0f" % r["sharp"] for r in rows)); print("   d/frame     " + "  ".join("%8.2f" % r["change"] for r in rows))
    if rows[0]["dist_ref"] is not None: print("   |x - ref|   " + "  ".join("%8.1f" % r["dist_ref"] for r in rows))
if a.json: json.dump(OUT, open(a.json, "w"), indent=1)
