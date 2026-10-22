"""Early-rejection probe analysis: for every drafted chunk, the intermediate signals at draft nodes 0..3 (target one-step x0 vs draft x0 difference 'diff',
drift critic on the target's x0 'critT', drift critic on the draft's x0 'critD') and the final chunk's drift-critic logit 'final_critD'.
Question: how early can we predict that the finished draft chunk will be flagged (final_critD > tau)? Prints AUC per node and signal, and the draft steps saved.
usage: early_probe_summary.py [ROOT=specwm_out/loop_probe] [--tau 0.05]"""
import sys, json, glob, argparse, numpy as np
from sklearn.metrics import roc_auc_score
S = "/home/hk28965/specwm"; ap = argparse.ArgumentParser(); ap.add_argument("root", nargs="?", default=S + "/lingbot-world-v2/specwm_out/loop_probe"); ap.add_argument("--tau", type=float, default=0.05); a = ap.parse_args()
rows = []
for f in glob.glob(a.root + "/*/*/loop.json"):
    d = json.load(open(f))
    for r in d.get("early", []):
        if "nodes" in r and r.get("final_critD") is not None: rows.append(r)
print("chunks %d (clips %d)" % (len(rows), len(glob.glob(a.root + "/*/*/loop.json"))))
y = np.array([r["final_critD"] > a.tau for r in rows]); print("final flag rate at tau %+.2f: %.2f" % (a.tau, y.mean()))
nodes = sorted({k for r in rows for k in r["nodes"]}, key=int)
print("%-6s %7s | %9s %9s %9s | corr(diff, final) corr(critT, final) corr(critD, final)" % ("node", "sigma", "AUC diff", "AUC critT", "AUC critD"))
for k in nodes:
    sig = rows[0]["nodes"][k]["sigma"]; g = lambda key: np.array([r["nodes"][k][key] for r in rows], dtype=float); fin = np.array([r["final_critD"] for r in rows])
    out = []
    for key in ("diff", "critT", "critD"):
        v = g(key); out.append(roc_auc_score(y, v) if len(set(y)) > 1 and not np.isnan(v).any() else float("nan"))
    cors = [float(np.corrcoef(g(key), fin)[0, 1]) if not np.isnan(g(key)).any() else float("nan") for key in ("diff", "critT", "critD")]
    print("%-6s %7.3f | %9.3f %9.3f %9.3f | %9.3f %18.3f %18.3f" % (k, sig, *out, *cors))
# what early rejection at node k buys: fraction of chunks flagged (= draft steps saved on them) at thresholds matching the final flag rate
for k in nodes:
    for key in ("critT", "critD", "diff"):
        v = np.array([r["nodes"][k][key] for r in rows], dtype=float)
        if np.isnan(v).any(): continue
        t = np.quantile(v, 1 - y.mean()); flag = v > t; tp = (flag & y).sum(); print("  node %s %-5s: flag %.2f of chunks at matched rate -> precision %.2f recall %.2f, draft steps saved %.1f%%" % (k, key, flag.mean(), tp / max(1, flag.sum()), tp / max(1, y.sum()), 100 * flag.mean() * (3 - int(k)) / 4))
