"""Per-category majority counts per arm (pooled over jar+castle) + exploratory 'severe' binary = teleport | color_collapse | texture_collapse | freeze
usage: blind_cats.py MAPPING.json RESULTS...  (prints a table; writes blind_cats.json next to MAPPING)"""
import sys, json, collections, os, math
from scipy import stats
mp = json.load(open(sys.argv[1])); R = {}
for f in sys.argv[2:]:
    raw = json.load(open(f)); raw = raw.get("result", raw) if isinstance(raw, dict) else raw
    for r in raw: R[r["id"]] = r
CATS = ["teleport", "color_collapse", "texture_collapse", "freeze", "material_swap", "new_object", "lighting_change"]
SEV = {"teleport", "color_collapse", "texture_collapse", "freeze"}
ARMS = ["base70", "never", "specstep_n012c9_never", "tgtonly_n012c9_never", "plh_plan6_ck400_cr_st0_never", "plh_plan6_ck400_l4c_st0_never", "plh_plan6_ck400_l4_st0_never", "sdvg_t-0p7_ir", "sdvg_t-2p2_ir", "sdvg_t-2p5_ir"]
G = collections.defaultdict(list)
for hid, r in R.items(): G[mp[hid]["arm"]].append((mp[hid]["task"], r))
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d; return p, max(0, c - h), min(1, c + h)
out = {}
bk = None
for arm in ARMS:
    L = G.get(arm, []); n = len(L)
    if not n: continue
    cc = {c: sum(1 for _, r in L if r["drift"] and r["categories"].get(c, 0) >= 2) for c in CATS}
    sev_ok = sum(1 for _, r in L if not (r["drift"] and any(r["categories"].get(c, 0) >= 2 for c in SEV)))
    any_ok = sum(1 for _, r in L if not r["drift"])
    if arm == "base70": bk = (sev_ok, n, any_ok)
    pb = stats.boschloo_exact([[sev_ok, n - sev_ok], [bk[0], bk[1] - bk[0]]]).pvalue if arm != "base70" else float("nan")
    pa = stats.boschloo_exact([[any_ok, n - any_ok], [bk[2], bk[1] - bk[2]]]).pvalue if arm != "base70" else float("nan")
    p, lo, hi = wilson(sev_ok, n)
    out[arm] = dict(n=n, cats=cc, no_drift=any_ok, p_any=pa, no_severe=sev_ok, severe_ci=[lo, hi], p_severe=pb)
    print("%-32s n=%2d  no-drift %2d (p=%s)  no-severe %2d = %.2f [%.2f,%.2f] (p=%s) | %s" % (arm, n, any_ok, "%.3f" % pa if pa == pa else "-", sev_ok, p, lo, hi, "%.3f" % pb if pb == pb else "-", " ".join("%s:%d" % (c[:8], v) for c, v in cc.items())))
json.dump(out, open(os.path.join(os.path.dirname(sys.argv[1]), "blind_cats.json"), "w"), indent=1)
