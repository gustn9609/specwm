"""Summarise the speculative-loop sweep (specwm_out/loop/<task>/<arm>_s<seed>/loop.json) per arm: accept rate, generation time and its breakdown,
and (if paper/overnight/blind_loop has r{1,2,3}*.json) blind severe-clean / strict-clean per arm. Also the per-arm verify cost per chunk.
usage: loop_summary.py [BLIND_DIR]"""
import os, sys, json, glob, collections, numpy as np
S = "/home/hk28965/specwm"; R = S + "/lingbot-world-v2/specwm_out/loop"; B = sys.argv[1] if len(sys.argv) > 1 else S + "/paper/overnight/blind_loop"
SEV = {"teleport", "color_collapse", "texture_collapse", "freeze"}
runs = collections.defaultdict(list)
for f in glob.glob(R + "/*/*/loop.json"):
    d = json.load(open(f)); arm = os.path.basename(os.path.dirname(f)).rsplit("_s", 1)[0]; runs[arm].append(d)
lab = {}
if os.path.exists(B + "/mapping.json") and glob.glob(B + "/r1*.json"):
    mp = json.load(open(B + "/mapping.json")); RR = {r: {x["id"]: x for f in sorted(glob.glob("%s/r%d*.json" % (B, r))) for x in json.load(open(f))} for r in (1, 2, 3)}
    sev = lambda x: x["drift"] and bool(set(x.get("categories", [])) & SEV)
    for i, v in mp.items():
        if all(i in RR[r] for r in (1, 2, 3)):
            lab[(v["task"], v["seed"], v["arm"])] = (sum(sev(RR[r][i]) for r in (1, 2, 3)) < 2, sum(RR[r][i]["drift"] for r in (1, 2, 3)) < 2)
print("%-26s %4s %7s %8s %6s %6s %6s %6s %7s | %s" % ("arm", "n", "accept", "gen s", "draft", "verify", "target", "commit", "ver/ch", "blind severe-clean / strict (n)"))
order = sorted(runs, key=lambda a: (0 if a == "none" else 9 if a == "target" else 5, a))
for arm in order:
    rs = runs[arm]; n = len(rs); acc = np.mean([r["accepted"] / r["chunks"] for r in rs]); t = {k: np.mean([r["times"][k] for r in rs]) for k in ("draft", "verify", "target", "commit")}
    nver = np.mean([sum(1 for s in r["scores"] if s is not None) for r in rs]); vpc = t["verify"] / nver if nver else 0.0
    bl = [lab[(r["task"], r["seed"], arm)] for r in rs if (r["task"], r["seed"], arm) in lab]
    bs = "%d / %d (%d)" % (sum(b[0] for b in bl), sum(b[1] for b in bl), len(bl)) if bl else "-"
    print("%-26s %4d %6.0f%% %7.1f %6.1f %6.1f %6.1f %6.1f %6.2fs | %s" % (arm, n, 100 * acc, np.mean([r["gen_s"] for r in rs]), t["draft"], t["verify"], t["target"], t["commit"], vpc, bs))
