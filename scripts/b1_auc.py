#!/usr/bin/env python3
"""B1 verifier AUC (positives = Fast/corrupted, negatives = Base; 4 scenes). usage: b1_auc.py [suffix]  (suffix e.g. _r256 -> *_vggt_r256.json / *_diffloss_r256.json)"""
import json, os, sys, numpy as np
S = "/home/hk28965/specwm"; B = S + "/lingbot-world-v2/specwm_out/b1"; suf = sys.argv[1] if len(sys.argv) > 1 else ""
SC = ["05", "01", "00", "02"]; ST = ["falling_glass", "pouring_water_into_cup", "rocket_launch", "inflating_balloon"]
def load(n, kind):
    p = "%s/%s_%s%s.json" % (B, n, kind, suf); return json.load(open(p)) if os.path.exists(p) else None
def auc(pos, neg):
    pos = [p for p in pos if p is not None]; neg = [n for n in neg if n is not None]
    if not pos or not neg: return None
    return float(np.mean([1.0 if p > n else (0.5 if p == n else 0.0) for p in pos for n in neg]))
axes = {"VGGT rot-jump max": ("vggt", lambda r: max(r["rot_jump_deg"])), "VGGT rot-jump p95": ("vggt", lambda r: float(np.percentile(r["rot_jump_deg"], 95))),
        "VGGT pos err vs cmd": ("vggt", lambda r: r.get("mean_pos_err_rel"))}
for s in ["0.24", "0.39", "0.60"]: axes["Base diffloss σ%s" % s] = ("diffloss", (lambda s: lambda r: r["mean"].get("sigma_%s" % s))(s))
rows = []
for name, (kind, fn) in axes.items():
    g = lambda n: (lambda r: fn(r) if r else None)(load(n, kind))
    neg = [g("base_S%s" % s) for s in SC]; cols = [auc([g("fast_S%s" % s) for s in SC], neg)]
    for c in ["teleport", "shuffle", "freeze"]: cols.append(auc([g("ctrl_S%s_%s" % (s, c)) for s in SC], neg))
    cols.append(auc([g("stevo_fast_%s" % t) for t in ST], [g("stevo_base_%s" % t) for t in ST]) if kind == "vggt" else None)
    rows.append((name, cols))
print("| axis | Fast vs Base | teleport | shuffle | freeze | StEvo Fast vs Base |\n|---|---|---|---|---|---|")
for name, cols in rows: print("| %s | %s |" % (name, " | ".join("-" if c is None else "%.2f" % c for c in cols)))
