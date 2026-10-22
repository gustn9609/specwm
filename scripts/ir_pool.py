"""Offline ImageReward pool: per-chunk (12 frames) min/avg reward for existing videos. usage: ir_pool.py OUT.json ARM1,ARM2 SEEDS(csv)"""
import sys, json, numpy as np, imageio, ImageReward as RM, torch
from PIL import Image
S = "/home/hk28965/specwm"; V0 = S + "/lingbot-world-v2/specwm_out/spec_v0"
out, arms, seeds = sys.argv[1], sys.argv[2].split(","), [int(x) for x in sys.argv[3].split(",")]
m = RM.load("ImageReward-v1.0", device="cuda"); m.eval(); R = {}
for t in ("sand_filling_up_jar", "sandcastle_wave_impact"):
    p = open("%s/exp/irprompt_%s.txt" % (S, t)).read().strip()
    for arm in arms:
        for sd in seeds:
            f = "%s/%s/%s_s%d/video.mp4" % (V0, t, arm, sd)
            try: fr = [np.asarray(x) for x in imageio.get_reader(f)]
            except Exception: continue
            ch = []
            for c in range(1, (len(fr) - 1) // 12 + 1):   # chunk c = frames 12c-11 .. 12c (chunk 0 = frame 0 + first 12? keep it simple: 12-frame windows after frame 0)
                ims = [Image.fromarray(fr[i]) for i in range(12 * (c - 1) + 1, min(12 * c + 1, len(fr)))]
                with torch.no_grad(): _, rw = m.inference_rank(p, ims)
                ch.append(dict(min=float(min(rw)), avg=float(np.mean(rw))))
            R["%s|%s|%d" % (t, arm, sd)] = ch; print(t[:6], arm[:12], sd, "min per chunk", " ".join("%.2f" % x["min"] for x in ch), flush=True)
json.dump(R, open(out, "w"), indent=1)
for t in ("sand_filling_up_jar", "sandcastle_wave_impact"):
    for arm in arms:
        mins = [x["min"] for k, v in R.items() if k.startswith(t + "|" + arm + "|") for x in v[1:]]
        if mins: print("%s %-30s chunks %3d  min-score pct 10/19/50/90: %s" % (t[:6], arm, len(mins), " ".join("%.2f" % np.percentile(mins, q) for q in (10, 19, 50, 90))))
