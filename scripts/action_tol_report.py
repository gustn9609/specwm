"""Aggregate specwm/action_tol.py runs: per perturbation, mean token gap / within-eps vs the true-action chunk, next to the other-noise floor.
usage: python scripts/action_tol_report.py DIR [--json OUT]"""
import sys, json, glob, numpy as np
d = sys.argv[1]; R = [json.load(open(p)) for p in sorted(glob.glob(d + "/*/stats.json"))]; ch = [c for r in R for c in r["chunks"]]
keys = ["noise", "extrap", "stop", "speed0.5", "speed2", "yaw10", "yaw30"]; print("runs", len(R), "chunks", len(ch))
print("| perturbation of the chunk's action | gap σ_T ↓ | within ε ↑ | chunks at/above the other-noise floor (within ε ≥ noise − 5 pts) |")
print("|---|---|---|---|")
out = {}
for k in keys:
    m = np.mean([c[k]["mean"] for c in ch]); f = np.mean([c[k]["frac"] for c in ch]); hit = np.mean([c[k]["frac"] >= c["noise"]["frac"] - 0.05 for c in ch])
    out[k] = dict(mean=float(m), frac=float(f), hit=float(hit)); print("| %s | %.3f | %.1f %% | %.0f %% |" % (k, m, 100 * f, 100 * hit))
mo = np.array([c["motion"] for c in ch]); print("chunk motion (mean relative translation norm): median %.3f, min %.3f, max %.3f" % (np.median(mo), mo.min(), mo.max()))
for k in ("speed0.5", "speed2", "stop"):
    lo = [c[k]["frac"] for c in ch if c["motion"] <= np.median(mo)]; hi = [c[k]["frac"] for c in ch if c["motion"] > np.median(mo)]
    print("  %s: within ε on low-motion chunks %.0f %% vs high-motion %.0f %%" % (k, 100 * np.mean(lo), 100 * np.mean(hi)))
if "--json" in sys.argv: json.dump(out, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
