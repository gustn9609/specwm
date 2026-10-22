"""Score blind_fast: per arm drift rate (majority of 3 raters), severe rate (teleport/new_object/color/texture/freeze), per-category majority counts,
mean quality; paired by (task, seed) is not possible (chaotic rollouts), so arms are compared as groups with a permutation test.
usage: python scripts/blind_fast_score.py DIR"""
import sys, json, glob, collections, numpy as np
D = sys.argv[1]; mp = json.load(open(D + "/mapping.json"))
R = [{x["id"]: x for f in sorted(glob.glob("%s/r%d*.json" % (D, r))) if not f.endswith("mapping.json") for x in json.load(open(f))} for r in (1, 2, 3)]
SEV = {"teleport", "new_object", "color_collapse", "texture_collapse", "freeze"}
rows = collections.defaultdict(list); cats = collections.defaultdict(collections.Counter); miss = 0
for i, v in mp.items():
    rs = [r[i] for r in R if i in r]
    if len(rs) < 3: miss += 1; continue
    drift = sum(x["drift"] for x in rs) >= 2; sev = sum(x["drift"] and bool(set(x.get("categories", [])) & SEV) for x in rs) >= 2
    q = np.mean([x.get("quality", np.nan) for x in rs]); rows[v["arm"]].append((drift, sev, q))
    for c, m in collections.Counter(c for x in rs for c in set(x.get("categories", []))).items():
        if m >= 2: cats[v["arm"]][c] += 1
print("missing ratings: %d" % miss)
print("| arm | clips | clean (no drift) ↑ | severe-clean ↑ | quality ↑ (1–5) | majority categories |"); print("|---|---|---|---|---|---|")
for arm, v in sorted(rows.items()):
    a = np.array(v, dtype=float); print("| %s | %d | %d/%d | %d/%d | %.2f | %s |" % (arm, len(a), (a[:, 0] == 0).sum(), len(a), (a[:, 1] == 0).sum(), len(a), np.nanmean(a[:, 2]), dict(cats[arm])))
arms = sorted(rows)
if len(arms) == 2:
    rng = np.random.default_rng(0); A, B = (np.array(rows[x], dtype=float) for x in arms); allv = np.concatenate([A, B])
    for k, n in ((0, "drift"), (1, "severe"), (2, "quality")):
        obs = abs(np.nanmean(A[:, k]) - np.nanmean(B[:, k])); c = 0
        for _ in range(10000):
            p = rng.permutation(len(allv)); c += abs(np.nanmean(allv[p[:len(A)], k]) - np.nanmean(allv[p[len(A):], k])) >= obs - 1e-12
        print("permutation p (%s difference): %.3f" % (n, c / 10000))
