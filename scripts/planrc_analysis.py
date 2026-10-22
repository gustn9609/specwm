"""Plan-level V/R analysis: for each case, which plan candidate does the one-step verifier (③) pick, and does its render pass the blind test?
Compares: k=0 (current), ③-selected (lowest mean err at a sigma), oracle (any candidate clean), random expectation (mean over candidates).
usage: planrc_analysis.py  (reads paper/overnight/blind_planrc/r{1,2,3}_p{0,1}.json, mapping.json, exp/kveval/planrc_scores.json)"""
import json, glob, collections, numpy as np
S = "/home/hk28965/specwm"; B = S + "/paper/overnight/blind_planrc"; SEV = {"teleport", "color_collapse", "texture_collapse", "freeze"}
mp = json.load(open(B + "/mapping.json")); R = collections.defaultdict(dict)
for f in glob.glob(B + "/r[123]_p[01].json"):
    r = f.split("/")[-1][1]
    for x in json.load(open(f)): R[r][x["id"]] = x
ids = [i for i in mp if all(i in R[r] for r in R)]
sev = lambda x: x["drift"] and bool(set(x.get("categories", [])) & SEV)
lab = {}   # (task, seed, k) -> clean (severe majority == False), drift majority
for i in ids:
    v = mp[i]; k = "0" if v["arm"].startswith("plhv2_") else v["arm"][6]
    votes = [sev(R[r][i]) for r in R]; dv = [R[r][i]["drift"] for r in R]
    lab[(v["task"], v["seed"], k)] = dict(clean=sum(votes) < len(votes) / 2, strict=sum(dv) < len(dv) / 2, unan=len(set(votes)) == 1, cats=collections.Counter(c for r in R for c in R[r][i].get("categories", [])))
sc = {(x["task"], x["seed"], x["k"]): x for x in json.load(open(S + "/exp/kveval/planrc_scores.json"))}
cases = sorted({(t, s) for (t, s, k) in lab})
print("rated clips %d, cases %d, unanimous %d" % (len(ids), len(cases), sum(l["unan"] for l in lab.values())))
def pick(t, s, key):
    ks = [k for k in "0123" if (t, s, k) in sc and (t, s, k) in lab]
    return min(ks, key=lambda k: sc[(t, s, k)][key])
res = collections.defaultdict(lambda: collections.Counter())
for (t, s) in cases:
    ks = [k for k in "0123" if (t, s, k) in lab]
    res["k=0 (current)"][t] += lab[(t, s, "0")]["clean"] if (t, s, "0") in lab else 0
    res["oracle (any candidate)"][t] += any(lab[(t, s, k)]["clean"] for k in ks)
    res["random (mean over candidates)"][t] += np.mean([lab[(t, s, k)]["clean"] for k in ks])
    for key in ("m0.3", "m0.5", "m0.7", "m0.95", "x0.95", "x0.7"):
        res["③ pick by %s" % key][t] += lab[(t, s, pick(t, s, key))]["clean"]
n = collections.Counter(t for (t, s) in cases)
print("%-32s %8s %8s %8s" % ("selector", "jar", "castle", "all"))
for name, c in res.items():
    print("%-32s %5.1f/%-2d %5.1f/%-2d %5.1f/%-2d" % (name, c["sand_filling_up_jar"], n["sand_filling_up_jar"], c["sandcastle_wave_impact"], n["sandcastle_wave_impact"], sum(c.values()), len(cases)))
# AUC of each score for predicting the render's severe drift, over all 88 candidates
try:
    from sklearn.metrics import roc_auc_score
    y = [not lab[k]["clean"] for k in lab if k in sc]
    for key in ("m0.3", "m0.5", "m0.7", "m0.95", "x0.7", "x0.95"):
        print("AUC(%s -> render severe) %.3f" % (key, roc_auc_score(y, [sc[k][key] for k in lab if k in sc])))
except Exception as e: print("auc skipped", e)
print("per case (k: clean/drift, ③ m0.7 rank):")
for (t, s) in cases:
    print("%-6s s%-4d " % (t[:6], s) + "  ".join("k%s:%s%s" % (k, "C" if lab[(t, s, k)]["clean"] else "X", "*" if pick(t, s, "m0.7") == k else "") for k in "0123" if (t, s, k) in lab) + "   cats: " + "; ".join("k%s %s" % (k, ",".join(c for c, m in lab[(t, s, k)]["cats"].items() if m >= 2)) for k in "0123" if (t, s, k) in lab and not lab[(t, s, k)]["clean"]))
json.dump({"%s|%d|%s" % k: v | {"cats": dict(v["cats"])} for k, v in lab.items()}, open(B + "/planrc_labels.json", "w"))
