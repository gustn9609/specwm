"""p/q-style verifier (Guandao's whiteboard): per latent frame, d = err_Base(sigma) - err_Fast(sigma) on the same clip, same noise
(one-step denoising errors ~ negative log-likelihood proxies: d > 0 means the target finds the frame less likely than the draft does).
Clip scores: max / mean / late mean of d (latents 1..); AUC vs blind 'severe' and strict drift; also Base-only and Fast-only for reference."""
import json, glob, numpy as np
from sklearn.metrics import roc_auc_score
S = "/home/hk28965/specwm"; RB = json.load(open(S + "/exp/kveval/base_verify_fastsig.json")); RF = json.load(open(S + "/exp/kveval/fast_verify.json"))
B = S + "/paper/overnight/blind"; mp = json.load(open(B + "/mapping.json")); RES = {}
for f in sorted(glob.glob(B + "/results_batch*.json")):
    for r in json.load(open(f))["result"]: RES[r["id"]] = r
SEV = {"teleport", "color_collapse", "texture_collapse", "freeze"}; L = {}
for h, m in mp.items():
    r = RES.get(h)
    if r: L["%s|%s|%d" % (m["task"], m["arm"], m["seed"])] = (bool(r["drift"] and any(v >= 2 and c in SEV for c, v in r["categories"].items())), r["drift"])
out = {}
for subset, keep in (("all arms", lambda k: True), ("without exact-spec / null arms", lambda k: not k.split("|")[1].startswith(("specstep", "tgtonly")))):
    print("\n== %s" % subset)
    for s in sorted({s for k in RB for s in RB[k]}, key=float):
        ks = [k for k in RB if k in RF and k in L and s in RB[k] and s in RF[k] and keep(k)]
        y1 = [L[k][0] for k in ks]; y2 = [L[k][1] for k in ks]
        for name, f in (("Base only (mean)", lambda k: np.mean(RB[k][s]["err"][1:])), ("Fast only (mean)", lambda k: np.mean(RF[k][s]["err"][1:])),
                        ("p/q: Base - Fast (mean)", lambda k: np.mean(np.array(RB[k][s]["err"][1:]) - np.array(RF[k][s]["err"][1:]))),
                        ("p/q: Base - Fast (max)", lambda k: np.max(np.array(RB[k][s]["err"][1:]) - np.array(RF[k][s]["err"][1:]))),
                        ("p/q: Base / Fast (mean ratio)", lambda k: np.mean(np.array(RB[k][s]["err"][1:]) / np.array(RF[k][s]["err"][1:])))):
            sc = [f(k) for k in ks]; a1, a2 = roc_auc_score(y1, sc), roc_auc_score(y2, sc)
            out["%s|%s|%s" % (subset, s, name)] = dict(n=len(ks), auc_severe=a1, auc_drift=a2)
            print("sigma %s  %-30s n=%3d  AUC severe %.3f | strict drift %.3f" % (s, name, len(ks), a1, a2))
json.dump(out, open(S + "/exp/kveval/pq_verify_summary.json", "w"), indent=1)
