"""Calibration of the target-agreement verifier: per chunk d_full / d_lf at node 0 vs local_err (rel. L2 between the draft-finished and the target-finished
chunk from the same node-0 state). Prints Spearman, AUC for 'local_err above median', and tau quantiles. usage: python agree_calib.py ROOT"""
import sys, json, glob, numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
R = {}
for f in glob.glob(sys.argv[1] + "/*/*/loop.json"):
    j = json.load(open(f)); ctx = "target-ctx" if j["verifier"].endswith("-1") else "draft-ctx"
    for r in j["early"]:
        if "local_err" in r: R.setdefault(ctx, []).append((r["d_full"], r["d_lf"], r["local_err"]))
for ctx, v in sorted(R.items()):
    v = np.array(v); y = v[:, 2] > np.median(v[:, 2])
    print("%s: %d chunks | local_err median %.4f p90 %.4f" % (ctx, len(v), np.median(v[:, 2]), np.percentile(v[:, 2], 90)))
    for i, n in ((0, "d_full"), (1, "d_lf")):
        print("   %-6s spearman vs local_err %.3f | AUC(local_err > median) %.3f | quantiles p25 %.4f p50 %.4f p75 %.4f" % (n, spearmanr(v[:, i], v[:, 2])[0], roc_auc_score(y, v[:, i]), *np.percentile(v[:, i], [25, 50, 75])))
json.dump({k: np.array(v).tolist() for k, v in R.items()}, open("/home/hk28965/specwm/exp/kveval/agree_calib.json", "w"))
