"""Aggregate specwm/action_tol.py --warp 1 (proposal 1, warp reconciliation): per hypothesis variant, within-eps vs the true-action chunk of
(a) the branch as is, (b) warped only, (c) warped + last step / last two steps with the true action, (d) 2-step: branch, warped + last step;
both warp signs reported (the better one is the right convention). usage: python scripts/action_warp_report.py DIR [--json OUT]"""
import sys, json, glob, numpy as np
d = sys.argv[1]; R = [json.load(open(p)) for p in sorted(glob.glob(d + "/*/stats.json"))]; ch = [c for r in R for c in r["chunks"]]; print("runs", len(R), "chunks", len(ch))
f = lambda k: 100 * np.mean([c[k]["frac"] for c in ch]); vis = lambda k: 100 * np.mean([c[k]["visible"] for c in ch]); out = {}
print("4-step, floor (other noise) %.1f %% | 2-step floor %.1f %%" % (f("noise"), f("noise_2s")))
print("| hypothesis | branch as is ↑ | late-3 (no warp, last step true) ↑ | warp sign | visible area | warp only ↑ | warp + last step (J3) ↑ | warp + last 2 steps (J2) ↑ | 2-step branch ↑ | 2-step warp + last step ↑ |")
print("|---|---|---|---|---|---|---|---|---|---|")
for v in ("stop", "yaw10", "yaw30"):
    for sg in ("p", "n"):
        t = "%s_w%s" % (v, sg); out[t] = dict(asis=f(v), late3=f(v + "_late3"), warp=f(t), J3=f(t + "_J3"), J2=f(t + "_J2"), s2=f(v + "_2s"), s2w=f(t + "_2s"), visible=vis(t))
        print("| %s | %.1f %% | %.1f %% | %s | %.0f %% | %.1f %% | %.1f %% | %.1f %% | %.1f %% | %.1f %% |" % (v, f(v), f(v + "_late3"), "+" if sg == "p" else "−", vis(t), f(t), f(t + "_J3"), f(t + "_J2"), f(v + "_2s"), f(t + "_2s")))
out["noise"] = f("noise"); out["noise_2s"] = f("noise_2s")
if "--json" in sys.argv: json.dump(out, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
