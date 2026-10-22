"""Score blind drift labels: python blind_score.py MAPPING.json RESULTS1.json [RESULTS2.json ...] -> per (task, arm) no-drift rate, Wilson 95% CI,
Boschloo exact p vs the Base-70 batch of the same task, pooled over tasks, Fleiss' kappa of the 3 raters, category counts. Writes blind_scores.json next to MAPPING."""
import sys, json, math, collections, os
from scipy import stats
mp = json.load(open(sys.argv[1])); R = {}
for f in sys.argv[2:]:
    raw = json.load(open(f)); raw = raw.get("result", raw) if isinstance(raw, dict) else raw
    for r in raw: R[r["id"]] = r
def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 3
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d; return p, max(0, c - h), min(1, c + h)
G = collections.defaultdict(list)
for hid, r in R.items():
    m = mp[hid]; G[(m["task"], m["arm"])].append((m["seed"], r))
# Fleiss' kappa (binary, 3 raters)
N = len(R); P_i = []; pj = [0, 0]
for r in R.values():
    k1 = r["drift_votes"]; k0 = r["n"] - k1; n = r["n"]
    if n < 2: continue
    P_i.append((k0 * (k0 - 1) + k1 * (k1 - 1)) / (n * (n - 1))); pj[0] += k0; pj[1] += k1
tot = sum(pj); Pe = sum((x / tot) ** 2 for x in pj) if tot else float("nan"); Pb = sum(P_i) / len(P_i) if P_i else float("nan")
kappa = (Pb - Pe) / (1 - Pe) if tot and Pe < 1 else float("nan")
unan = sum(1 for r in R.values() if r["drift_votes"] in (0, r["n"]))
print("clips rated %d | unanimous %d (%.0f%%) | Fleiss kappa %.2f" % (N, unan, 100 * unan / max(N, 1), kappa))
out = {"n_clips": N, "fleiss_kappa": kappa, "unanimous": unan, "groups": {}}
ARM_ORDER = ["base70", "never", "specstep_n012c9_never", "tgtonly_n012c9_never", "specstep_n012c9g1_never", "tgtonly_n012c9g1_never", "specstep_n012c9g2p5_never", "plh_plan6_ck400_cr_st0_never", "plh_plan6_ck400_l4c_st0_never", "plh_plan6_ck400_l4_st0_never", "sdvg_t-0p7_ir", "sdvg_t-2p2_ir", "sdvg_t-2p5_ir"]
tasks = sorted({t for t, _ in G})
pooled = collections.defaultdict(lambda: [0, 0])
for t in tasks:
    b = G.get((t, "base70"), []); bk = sum(1 for _, r in b if not r["drift"]); bn = len(b)
    print("\n%s" % t)
    for arm in ARM_ORDER + sorted({a for tt, a in G if tt == t and a not in ARM_ORDER}):
        L = G.get((t, arm));
        if not L: continue
        k = sum(1 for _, r in L if not r["drift"]); n = len(L); p, lo, hi = wilson(k, n)
        try: pb = stats.boschloo_exact([[k, n - k], [bk, bn - bk]]).pvalue if arm != "base70" and bn else float("nan")
        except Exception: pb = float("nan")
        cats = collections.Counter(c for _, r in L if r["drift"] for c, v in r["categories"].items() if v >= 2)
        seeds = " ".join("%d:%s%d" % (sd, "D" if r["drift"] else "ok", r["drift_votes"]) for sd, r in sorted(L))
        print("  %-34s no-drift %2d/%-2d = %.2f [%.2f, %.2f]  Boschloo vs Base p=%s  cats %s | %s" % (arm, k, n, p, lo, hi, "%.3f" % pb if pb == pb else "  -  ", dict(cats), seeds))
        out["groups"]["%s|%s" % (t, arm)] = dict(k=k, n=n, rate=p, ci=[lo, hi], boschloo_vs_base=pb, categories=dict(cats), seeds={sd: dict(drift=r["drift"], votes=r["drift_votes"]) for sd, r in L})
        pooled[arm][0] += k; pooled[arm][1] += n
print("\npooled over tasks")
for arm in ARM_ORDER:
    if arm in pooled: k, n = pooled[arm]; p, lo, hi = wilson(k, n); print("  %-34s no-drift %2d/%-2d = %.2f [%.2f, %.2f]" % (arm, k, n, p, lo, hi))
out["pooled"] = {a: dict(k=v[0], n=v[1]) for a, v in pooled.items()}
json.dump(out, open(os.path.join(os.path.dirname(sys.argv[1]), "blind_scores.json"), "w"), indent=1)
