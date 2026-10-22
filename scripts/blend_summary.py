"""Guandao's frame-jump fixes: seam size at the boundaries where the video switches between an accepted Fast chunk and a Base-corrected chunk.
boundary_ratio[b] = frame-to-frame change entering chunk b+1 / median change inside chunks (temporal_eval.py). Switch boundaries: plan[c] != plan[c-1]."""
import json, numpy as np
S = "/home/hk28965/specwm"; V = S + "/lingbot-world-v2/specwm_out/spec_v0"
RUNS = [("cfg6reuse_semc100", "replace (baseline)"), ("cfg6reuse_semc100_blsoft05", "soft update 0.5"), ("cfg6reuse_semc100_bllow4", "low-freq only, 4x4"), ("cfg6reuse_semc100_bllow8", "low-freq only, 8x8")]
rows = {}
for r, nm in RUNS:
    E = json.load(open("%s/exp/kveval/te6_%s.json" % (S, r))); sw, ns, tb, fl, mo, tm, nc = [], [], [], [], [], [], []
    for t in ("sand_filling_up_jar", "sandcastle_wave_impact"):
        for s in (1234, 4321, 9999):
            e = E["%s/%s_s%d" % (t, r, s)]; j = json.load(open("%s/%s/%s_s%d/spec_v0.json" % (V, t, r, s))); plan = j["plan"]
            for b, v in enumerate(e["boundary_ratio"]):
                c = b + 1
                if c < len(plan): (sw if plan[c] != plan[c - 1] else ns).append(v)
            tb.append(np.mean(e["to_base"])); fl.append(e["flicker"]); mo.append(e["motion"]); tm.append(j["gen_s"]); nc.append(j["corrected"])
    rows[r] = dict(name=nm, switch=float(np.mean(sw)), switch_max=float(np.max(sw)), same=float(np.mean(ns)), to_base=float(np.mean(tb)), flicker=float(np.mean(fl)), motion=float(np.mean(mo)), time=float(np.mean(tm)), corrected=float(np.mean(nc)), n_switch=len(sw))
    print("%-22s seam at Fast<->Base switches: mean %.2fx max %.2fx (n=%d) | other boundaries %.2fx | DINO to Base %.3f | flicker %.1f | motion %.1f | %.0f s, %.1f corrections" % (nm, rows[r]["switch"], rows[r]["switch_max"], len(sw), rows[r]["same"], rows[r]["to_base"], rows[r]["flicker"], rows[r]["motion"], rows[r]["time"], rows[r]["corrected"]))
json.dump(rows, open(S + "/exp/kveval/blend_summary.json", "w"), indent=1)
