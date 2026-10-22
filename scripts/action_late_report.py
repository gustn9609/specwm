"""Aggregate specwm/action_tol.py --late 1 runs (proposal A, late branching): for each hypothesis-action variant, within-eps of the chunk
generated (a) entirely with the hypothesis (the speculative branch, = a miss if shown), (b) with the first j steps under the hypothesis and
the rest under the true action (late branching: what a miss would cost if only the last steps are redone), against the true-action chunk
from the same noise; 4-step (j = 1..3) and 2-step (j = 1). usage: python scripts/action_late_report.py DIR [--json OUT]"""
import sys, json, glob, numpy as np
d = sys.argv[1]; R = [json.load(open(p)) for p in sorted(glob.glob(d + "/*/stats.json"))]; ch = [c for r in R for c in r["chunks"]]; print("runs", len(R), "chunks", len(ch))
V = ["extrap", "stop", "yaw10", "yaw30", "transfwd"]; out = {}
f = lambda k: 100 * np.mean([c[k]["frac"] for c in ch]); hit = lambda k, fl: 100 * np.mean([c[k]["frac"] >= c[fl]["frac"] - 0.05 for c in ch])
print("4-step (floor = other noise: %.1f %% within eps)" % f("noise"))
print("| hypothesis action | branch as is (all 4 steps hyp.) ↑ | late-1: steps 2-4 redone with the true action ↑ | late-2: steps 3-4 redone ↑ | late-3: step 4 redone ↑ | chunks at/above floor: as-is / late-1 / late-2 |")
print("|---|---|---|---|---|---|")
for v in V:
    out[v] = {k: float(f(v + k)) for k in ("", "_late1", "_late2", "_late3")}; print("| %s | %.1f %% | %.1f %% | %.1f %% | %.1f %% | %.0f / %.0f / %.0f %% |" % (v, f(v), f(v + "_late1"), f(v + "_late2"), f(v + "_late3"), hit(v, "noise"), hit(v + "_late1", "noise"), hit(v + "_late2", "noise")))
print("2-step (floor %.1f %%)" % f("noise_2s")); print("| hypothesis action | branch as is (2 steps hyp.) ↑ | late-1: step 2 redone with the true action ↑ | at/above floor: as-is / late-1 |"); print("|---|---|---|---|")
for v in V:
    out[v].update({"_2s": float(f(v + "_2s")), "_2s_late1": float(f(v + "_2s_late1"))}); print("| %s | %.1f %% | %.1f %% | %.0f / %.0f %% |" % (v, f(v + "_2s"), f(v + "_2s_late1"), hit(v + "_2s", "noise_2s"), hit(v + "_2s_late1", "noise_2s")))
out["noise"] = float(f("noise")); out["noise_2s"] = float(f("noise_2s"))
if "--json" in sys.argv: json.dump(out, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
