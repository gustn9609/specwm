#!/usr/bin/env python3
"""per-chunk motion / flicker / boundary jump for a list of run dirs (no GPU). usage: motion_eval.py RUN_DIR..."""
import sys, os, json, numpy as np, imageio
def chunks(n): return [list(range(0, 9))] + [list(range(12 * c - 3, min(12 * c + 9, n))) for c in range(1, 13)]
for run in sys.argv[1:]:
    F = [np.asarray(f).astype(np.float32) for f in imageio.get_reader(os.path.join(run, "video.mp4"))]; n = len(F)
    d1 = [np.abs(F[i + 1] - F[i]).mean() for i in range(n - 1)]; d2 = [np.abs(F[i + 1] - 2 * F[i] + F[i - 1]).mean() if 0 < i < n - 1 else np.nan for i in range(n)]
    C = chunks(n); mo = [np.nanmean([d1[i] for i in c[:-1]]) for c in C]; fl = [np.nanmean([d2[i] for i in c[1:-1]]) for c in C]; b = [12 * c - 4 for c in range(1, 13) if 12 * c - 4 < n - 1]
    inside = np.median([d1[i] for i in range(n - 1) if i not in b]); pj = os.path.join(run, "spec_v0.json"); j = json.load(open(pj)) if os.path.exists(pj) else {}
    print("%-44s gen %5s  motion(median) %5.1f  boundary/inside max %4.1fx  flicker(mean) %5.1f  plan %s" % (run.split("/")[-2][:6] + "/" + run.split("/")[-1], round(j["gen_s"]) if j else "-", inside, max(d1[i] for i in b) / max(inside, 1e-3), np.nanmean(fl), j.get("plan", "")))
    print("      motion/chunk " + " ".join("%4.0f" % x for x in mo) + "\n      boundary     " + " ".join("%4.0f" % d1[i] for i in b))
