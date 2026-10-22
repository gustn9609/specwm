#!/usr/bin/env python
"""[10/19 audit] Rewrite the node labels in the "model" and "steps" strings of the FROZEN timing.json files of adapt25 and every *c1hyb* arm
(runs/ours_adapt25_<set>, runs/ours_{adapt25c1hyb,rt2c1hyb,sf24c1hyb,rt2c1hyb32}_{dev,cloud}_<set>) in place, with ours_to_timing.NODE_LABEL_FIX:
"ONE step at t=750 (node index 3 of [1000, 750, 500, 250])" / "3rd step (t=250)" -> "one step at the last node, σ 0.625 (timesteps_index 750 of the
4-node schedule)" (the cloud's --use 3 of timesteps_index 0,250,500,750 = σ [1.0, 0.937, 0.833, 0.625] and adapt25's --adapt-node 2 of nodes 0,500,750
are the same node).  Only those two string fields change; the file is rewritten with indent=1.  Afterwards every file's "model" must equal
ours_to_timing.ARMS[arm][0] (checked; mismatches are listed).  Idempotent.  --dry-run: no writes.
usage: /home/hk28965/specwm/scripts/envrun.sh python /home/hk28965/specwm/baselines/gen/fix_node_labels_1019.py [--dry-run] [--runs DIR]"""
import os, sys, json, glob, argparse, collections
B = "/home/hk28965/specwm/baselines"; sys.path.insert(0, B)
from ours_to_timing import fix_node_labels, ARMS
TARGET = ("adapt25", "adapt25c1hyb_dev", "adapt25c1hyb_cloud", "rt2c1hyb_dev", "rt2c1hyb_cloud", "sf24c1hyb_dev", "sf24c1hyb_cloud", "rt2c1hyb32_dev", "rt2c1hyb32_cloud")
ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--runs", default=B + "/runs"); a = ap.parse_args()
n_upd = collections.Counter(); n_same = collections.Counter(); mism = []; n_files = collections.Counter()
for d in sorted(glob.glob(a.runs + "/ours_*")):
    name = os.path.basename(d); arm = next((x for x in sorted(TARGET, key=len, reverse=True) if name.startswith("ours_%s_" % x)), None)
    if arm is None: continue
    for tj in sorted(glob.glob(d + "/*/timing.json")):
        T = json.load(open(tj)); n_files[arm] += 1; changed = False
        for k in ("model", "steps"):
            v = fix_node_labels(T.get(k))
            if v != T.get(k): T[k] = v; changed = True
        if T.get("model") != ARMS[arm][0]: mism.append((tj, T.get("model")))
        if "t=750" in str(T.get("steps")) or "t=250" in str(T.get("steps")) or "node index 3" in str(T.get("model")): mism.append((tj, "old label left in steps/model: %s" % T.get("steps")))
        if not changed: n_same[arm] += 1; continue
        if not a.dry_run:
            tmp = tj + ".tmp"; json.dump(T, open(tmp, "w"), indent=1); os.replace(tmp, tj)
        n_upd[arm] += 1
print("[fix_node_labels] files seen per arm:", dict(n_files)); print("[fix_node_labels] rewritten per arm:", dict(n_upd), "| unchanged:", dict(n_same), "| dry-run" if a.dry_run else "")
print("[fix_node_labels] model-string mismatches vs ours_to_timing.ARMS after the fix: %d" % len(mism)); [print("  MISMATCH", m[0], "::", str(m[1])[:160]) for m in mism[:20]]
sys.exit(1 if mism else 0)
