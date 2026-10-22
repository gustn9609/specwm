"""pq calibration: per-chunk d = err_Base - err_Fast on teacher-forced Base-70 (good) and Fast (mostly failed) clips of fresh seeds.
Prints per-chunk distributions, clip-level AUC (max over chunks), and candidate thresholds (Base-70 chunk percentiles) with the Fast chunk rejection rate."""
import json, glob, sys, numpy as np
PFX = sys.argv[1] if len(sys.argv) > 1 else "pqcal"
from sklearn.metrics import roc_auc_score
V = "/home/hk28965/specwm/lingbot-world-v2/specwm_out/spec_v0"; D = {"base70": [], "never": []}; clip = {"base70": [], "never": []}
for arm in D:
    for f in sorted(glob.glob("%s/*/%s_%s_s*/spec_v0.json" % (V, PFX, arm))):
        ds = [l["pq_d"] for l in json.load(open(f))["log"] if l.get("pq_d") is not None]; D[arm].append(ds); clip[arm].append(max(ds))
for arm in D: print("%-7s clips %d | per-chunk d mean by chunk: %s" % (arm, len(D[arm]), " ".join("%.4f" % np.mean([d[c] for d in D[arm] if len(d) > c]) for c in range(12))))
b = np.concatenate(D["base70"]); f = np.concatenate(D["never"])
print("chunk-level AUC (Fast vs Base-70 chunks) %.3f | clip-level AUC (max over chunks) %.3f" % (roc_auc_score([0] * len(b) + [1] * len(f), np.r_[b, f]), roc_auc_score([0] * len(clip["base70"]) + [1] * len(clip["never"]), clip["base70"] + clip["never"])))
for q in (50, 75, 90, 95):
    tau = np.percentile(b, q); print("tau = Base-70 chunk p%d = %.4f -> rejects %.0f%% of Base-70 chunks, %.0f%% of Fast chunks" % (q, tau, 100 * (b > tau).mean(), 100 * (f > tau).mean()))
VD = {"base70": [], "never": []}
for arm in VD:
    for f in sorted(glob.glob("%s/*/%s_%s_s*/spec_v0.json" % (V, PFX, arm))): VD[arm] += [l["pq_vdiff"][0] for l in json.load(open(f))["log"] if l.get("pq_vdiff")]
if VD["base70"] and VD["never"]: print("velocity-difference score |v_T - v_D|^2: chunk-level AUC %.3f (n %d / %d)" % (roc_auc_score([0] * len(VD["base70"]) + [1] * len(VD["never"]), VD["base70"] + VD["never"]), len(VD["base70"]), len(VD["never"])))
json.dump({k: v for k, v in D.items()}, open("/home/hk28965/specwm/exp/kveval/%s.json" % PFX, "w"))
