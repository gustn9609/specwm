"""Pre-registered item-4 analysis (paper/overnight/prereg/PREREG_item4.md) on the fresh seeds 101-116.
Merges 3 blind raters (r{1,2,3}_p*.json), majority vote; primary endpoint severe-clean; Boschloo exact tests, Holm over the 5 registered tests;
CMH (by prompt) as a check; amendment arm F (SDVG port tau -2.2) reported separately; Fleiss kappa; out-of-sample AUC of the p/q verifier."""
import json, glob, collections, math, numpy as np
from scipy import stats
D = "/home/hk28965/specwm/paper/overnight/blind_prereg"; mp = json.load(open(D + "/mapping.json"))
votes = collections.defaultdict(list)
for f in sorted(glob.glob(D + "/r[123]_p[0-3].json")):
    for x in json.load(open(f)): votes[x["id"]].append(x)
SEV = {"teleport", "color_collapse", "texture_collapse", "freeze"}
lab = {}
for hid, v in votes.items():
    nd = sum(x["drift"] for x in v); cats = collections.Counter(c for x in v if x["drift"] for c in x["categories"])
    drift = nd * 2 > len(v); lab[hid] = dict(n=len(v), nd=nd, drift=drift, severe=bool(drift and any(cats[c] >= 2 for c in SEV)), cats=dict(cats))
print("rated clips %d (missing %d), raters per clip %s" % (len(lab), len(mp) - len(lab), collections.Counter(l["n"] for l in lab.values())))
P = [l for l in lab.values() if l["n"] == 3]; pj = [sum(3 - l["nd"] for l in P), sum(l["nd"] for l in P)]; Pi = [((3 - l["nd"]) * (2 - l["nd"]) + l["nd"] * (l["nd"] - 1)) / 6 for l in P]
Pe = sum((x / sum(pj)) ** 2 for x in pj); kappa = (np.mean(Pi) - Pe) / (1 - Pe); print("Fleiss kappa %.2f, unanimous %d/%d" % (kappa, sum(l["nd"] in (0, 3) for l in P), len(P)))
ARM = {"base70": "A Base-70", "never": "D Fast alone", "plh_plan6_ck400_cr_st0_never": "E1 v2-safe", "plh_plan6_ck400_l4c_st0_never": "E2 v2-fast-CFG", "sdvg_t-2p2_ir": "F SDVG port tau -2.2 (amendment)"}
cnt = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))   # arm -> task -> [clean, n]
for hid, m in mp.items():
    if hid in lab: c = cnt[m["arm"]][m["task"]]; c[0] += not lab[hid]["severe"]; c[1] += 1
def tot(a): return [sum(v[0] for v in cnt[a].values()), sum(v[1] for v in cnt[a].values())]
def wil(k, n, z=1.96): p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d; return max(0, c - h), min(1, c + h)
print("\n| arm | severe-clean (jar + castle) | rate [95% CI] | strict no-drift |")
for a, nm in ARM.items():
    k, n = tot(a); lo, hi = wil(k, n); sn = sum(1 for h, m in mp.items() if m["arm"] == a and h in lab and not lab[h]["drift"])
    print("| %s | %d/%d (jar %d/%d, castle %d/%d) | %.2f [%.2f, %.2f] | %d/%d |" % (nm, k, n, *cnt[a]["sand_filling_up_jar"], *cnt[a]["sandcastle_wave_impact"], k / n, lo, hi, sn, n))
def bos(a, b): (k1, n1), (k2, n2) = tot(a), tot(b); return stats.boschloo_exact([[k1, n1 - k1], [k2, n2 - k2]]).pvalue
def cmh(a, b):
    tabs = [np.array([[cnt[a][t][0], cnt[a][t][1] - cnt[a][t][0]], [cnt[b][t][0], cnt[b][t][1] - cnt[b][t][0]]]) for t in ("sand_filling_up_jar", "sandcastle_wave_impact")]
    try: return stats.contingency.StratifiedTable(tabs).test_null_odds().pvalue
    except Exception:
        try:
            from statsmodels.stats.contingency_tables import StratifiedTable; return StratifiedTable(tabs).test_null_odds().pvalue
        except Exception: return float("nan")
REG = [("E1 vs A", "plh_plan6_ck400_cr_st0_never", "base70"), ("E2 vs A", "plh_plan6_ck400_l4c_st0_never", "base70"), ("E1 vs D", "plh_plan6_ck400_cr_st0_never", "never"),
       ("E2 vs D", "plh_plan6_ck400_l4c_st0_never", "never"), ("D vs A", "never", "base70")]
ps = [bos(x, y) for _, x, y in REG]; order = np.argsort(ps); holm = [None] * 5; run = 0
for r, i in enumerate(order): run = max(run, min(1, (5 - r) * ps[i])); holm[i] = run
print("\n| registered test | p (Boschloo) | Holm-adjusted p | CMH by prompt p |")
for (nm, x, y), p, h in zip(REG, ps, holm): print("| %s | %.4f | %.4f | %.4f |" % (nm, p, h, cmh(x, y)))
print("\n| amendment (not in Holm family) | p (Boschloo) |")
for nm, x, y in (("F vs A", "sdvg_t-2p2_ir", "base70"), ("F vs E2", "sdvg_t-2p2_ir", "plh_plan6_ck400_l4c_st0_never")): print("| %s | %.4f |" % (nm, bos(x, y)))
# out-of-sample p/q verifier AUC
try:
    from sklearn.metrics import roc_auc_score
    S = "/home/hk28965/specwm/exp/kveval"; F_ = json.load(open(S + "/fast_verify_prereg.json")); B_ = {}
    for i in (0, 1): B_.update(json.load(open(S + "/base_verify_prereg_s%d.json" % i)))
    key = {("%s|%s|%d" % (m["task"], m["arm"], m["seed"])): h for h, m in mp.items()}
    print("\n| sigma | score | n | AUC severe | AUC strict drift |")
    for s in ("0.9373", "0.8331", "0.6247"):
        ks = [k for k in B_ if k in F_ and k in key and key[k] in lab]
        for nm, f in (("Base only (mean)", lambda k: np.mean(B_[k][s]["err"][1:])), ("p/q Base - Fast (mean)", lambda k: np.mean(np.array(B_[k][s]["err"][1:]) - np.array(F_[k][s]["err"][1:]))),
                      ("p/q Base / Fast (mean ratio)", lambda k: np.mean(np.array(B_[k][s]["err"][1:]) / np.array(F_[k][s]["err"][1:])))):
            sc = [f(k) for k in ks]; print("| %s | %s | %d | %.3f | %.3f |" % (s, nm, len(ks), roc_auc_score([lab[key[k]]["severe"] for k in ks], sc), roc_auc_score([lab[key[k]]["drift"] for k in ks], sc)))
except Exception as e: print("p/q AUC skipped:", e)
json.dump(dict(labels=lab, kappa=kappa), open(D + "/prereg_labels.json", "w"), indent=1)
