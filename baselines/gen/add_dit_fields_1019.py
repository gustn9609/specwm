#!/usr/bin/env python
"""[10/19 audit] Patch the FROZEN timing.json files of the spec_early arms (sf24, student1p3b, sf1p3b_pub, sf1p3b_cal; every runs/ours_<arm>_<set>* dir that
is not a hybrid stream) IN PLACE with the three additive fields of ours_to_timing.spec_early_dit_fields: gen_s_dit, steady_fps_gen_dit, fps_correction
(the raw loop.json next to timing.json "source" supplies times = {denoise, commit, verify}). gen_s / steady_fps_gen / chunk_s are not touched; the file
is rewritten with indent=1 (same as ours_to_timing).  Prints per-arm counts, the files whose loop.json is missing, and the arm x set table of median
steady_fps_gen vs steady_fps_gen_dit.  Idempotent (re-computes the same values).  --dry-run: no writes.
usage: /home/hk28965/specwm/scripts/envrun.sh python /home/hk28965/specwm/baselines/gen/add_dit_fields_1019.py [--dry-run] [--runs DIR]"""
import os, sys, json, glob, argparse, collections
B = "/home/hk28965/specwm/baselines"; sys.path.insert(0, B)
from ours_to_timing import spec_early_dit_fields
ARMS = ("sf24", "student1p3b", "sf1p3b_pub", "sf1p3b_cal")
def median(v): v = sorted(v); return v[len(v) // 2] if v else None
ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--runs", default=B + "/runs"); a = ap.parse_args()
n_upd = collections.Counter(); n_skip = collections.Counter(); missing = []; rows = collections.defaultdict(lambda: ([], []))
for d in sorted(glob.glob(a.runs + "/ours_*")):
    name = os.path.basename(d); arm = next((x for x in ARMS if name.startswith("ours_%s_" % x)), None)
    if arm is None or "c1hyb" in name: continue
    set_ = name[len("ours_%s_" % arm):]
    for tj in sorted(glob.glob(d + "/*/timing.json")):
        T = json.load(open(tj)); lp = os.path.join(T.get("source", ""), "loop.json")
        if not os.path.isfile(lp): missing.append(tj); continue
        L = json.load(open(lp)); F = spec_early_dit_fields(L, T.get("gen_s"), T.get("steady_fps_gen"))
        if not F: missing.append(tj + " (no times in loop.json)"); continue
        rows[(arm, set_)][0].append(T["steady_fps_gen"]); rows[(arm, set_)][1].append(F["steady_fps_gen_dit"])
        if all(T.get(k) == v for k, v in F.items()): n_skip[arm] += 1; continue
        T.update(F)
        if not a.dry_run:
            tmp = tj + ".tmp"; json.dump(T, open(tmp, "w"), indent=1); os.replace(tmp, tj)
        n_upd[arm] += 1
print("[add_dit_fields] updated per arm:", dict(n_upd), "| already up to date:", dict(n_skip), "| dry-run" if a.dry_run else "")
print("[add_dit_fields] loop.json missing / unusable: %d" % len(missing)); [print("  MISSING", m) for m in missing]
print("\n| arm | set | n | median steady_fps_gen (frozen) ↑ | median steady_fps_gen_dit (corrected) ↑ | ratio |\n|---|---|---|---|---|---|")
for (arm, set_), (f0, f1) in sorted(rows.items()):
    m0, m1 = median(f0), median(f1); print("| %s | %s | %d | %.2f | %.2f | %.3f |" % (arm, set_, len(f0), m0, m1, m1 / m0))
