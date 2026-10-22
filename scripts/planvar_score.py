"""Score a blind batch whose arms are planner variants: severe-clean / strict per arm, paired flips vs a reference arm. usage: planvar_score.py DIR REF_ARM_SUBSTR"""
import sys, json, collections
B, ref = sys.argv[1], sys.argv[2]; SEV = {"teleport", "color_collapse", "texture_collapse", "freeze"}
mp = json.load(open(B + "/mapping.json")); import glob
R = {r: {x["id"]: x for f in sorted(glob.glob("%s/r%d.json" % (B, r)) + glob.glob("%s/r%d_p*.json" % (B, r))) for x in json.load(open(f))} for r in (1, 2, 3)}
sev = lambda x: x["drift"] and bool(set(x.get("categories", [])) & SEV)
per = {}; un = 0
for i, v in mp.items():
    votes = [sev(R[r][i]) for r in (1, 2, 3)]; dv = [R[r][i]["drift"] for r in (1, 2, 3)]; un += len(set(votes)) == 1
    cats = collections.Counter(c for r in (1, 2, 3) for c in R[r][i].get("categories", []))
    per[(v["task"], v["seed"], v["arm"])] = (sum(votes) < 2, sum(dv) < 2, [c for c, m in cats.items() if m >= 2])
arms = sorted({a for (_, _, a) in per}); print("unanimous %d/%d" % (un, len(mp)))
print("%-42s %6s %8s %7s %7s" % ("arm", "jar", "castle", "clean", "strict"))
for a in arms:
    j = [v for (t, s, aa), v in per.items() if aa == a and "jar" in t]; c = [v for (t, s, aa), v in per.items() if aa == a and "castle" in t]
    print("%-42s %3d/%-2d %4d/%-3d %4d/%-2d %4d/%-2d" % (a, sum(x[0] for x in j), len(j), sum(x[0] for x in c), len(c), sum(x[0] for x in j + c), len(j) + len(c), sum(x[1] for x in j + c), len(j) + len(c)))
ra = [a for a in arms if ref in a][0]
for a in arms:
    if a == ra: continue
    b = o = r_ = n = 0
    for (t, s, aa), v in per.items():
        if aa != a or (t, s, ra) not in per: continue
        x, y = v[0], per[(t, s, ra)][0]; b += x and y; o += x and not y; r_ += y and not x; n += not x and not y
    print("paired %s vs ref: both %d / only this %d / only ref %d / both bad %d" % (a[:30], b, o, r_, n))
for (t, s, a), v in sorted(per.items()):
    if not v[0]: print("%-6s s%-4d %-42s %s" % (t[:6], s, a, ",".join(v[2])))
