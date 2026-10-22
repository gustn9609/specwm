"""Summarise base_verify shards: does Base's one-step whole-clip reconstruction error separate clips the blind raters marked as failed?
Clip score per sigma = max / mean / late-mean of the per-latent-frame error (latents 1..end; latent 0 = the start image).
Reports AUC vs the 'severe' label and vs strict drift, per-arm medians, and the latent frame where the error peaks."""
import json, glob, numpy as np
from sklearn.metrics import roc_auc_score
S = "/home/hk28965/specwm"; R = {}
for f in sorted(glob.glob(S + "/exp/kveval/base_verify_shard*.json")): R.update(json.load(open(f)))
L = {}
B = S + "/paper/overnight/blind"; mp = json.load(open(B + "/mapping.json")); RES = {}
for f in sorted(glob.glob(B + "/results_batch*.json")):
    for r in json.load(open(f))["result"]: RES[r["id"]] = r
SEV = {"teleport", "color_collapse", "texture_collapse", "freeze"}
for hid, m in mp.items():
    r = RES.get(hid)
    if r: L["%s|%s|%d" % (m["task"], m["arm"], m["seed"])] = dict(severe=bool(r["drift"] and any(v >= 2 and c in SEV for c, v in r["categories"].items())), drift=r["drift"])
keys = [k for k in R if k in L]; sig = sorted({s for k in keys for s in R[k]}, key=float)
print("clips with scores and labels: %d (severe %d)" % (len(keys), sum(L[k]["severe"] for k in keys)))
out = {}
for s in sig:
    ks = [k for k in keys if s in R[k]]
    for agg, f in (("max", lambda e: max(e[1:])), ("mean", lambda e: float(np.mean(e[1:]))), ("late mean (latents 20+)", lambda e: float(np.mean(e[20:])))):
        sc = [f(R[k][s]["err"]) for k in ks]
        a1 = roc_auc_score([L[k]["severe"] for k in ks], sc); a2 = roc_auc_score([L[k]["drift"] for k in ks], sc)
        out["%s|%s" % (s, agg)] = dict(auc_severe=a1, auc_drift=a2); print("sigma %-5s %-24s AUC severe %.3f | strict drift %.3f" % (s, agg, a1, a2))
    arms = {}
    for k in ks: arms.setdefault(k.split("|")[1], []).append(float(np.mean(R[k][s]["err"][1:])))
    print("   per-arm median mean-error: " + ", ".join("%s %.4f" % (a[:22], np.median(v)) for a, v in sorted(arms.items(), key=lambda x: np.median(x[1]))))
json.dump(out, open(S + "/exp/kveval/base_verify_summary.json", "w"), indent=1)
