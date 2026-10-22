"""Thresholds for the verifiers from the shadow data of the HELD-OUT tasks: for the critic logit and for the ③ contrast log(eT/eD),
find tau such that a given fraction of DRAFT chunks is accepted, and report the fraction of TARGET chunks that would pass the same tau.
usage: CUDA_VISIBLE_DEVICES=g critic_calibrate.py CRITIC_DIR OUT.json [--accept 0.3,0.5,0.7]"""
import os, sys, json, glob, argparse, numpy as np, torch
S = "/home/hk28965/specwm"; SH = S + "/lingbot-world-v2/specwm_out/shadow"
ap = argparse.ArgumentParser(); ap.add_argument("critic_dir"); ap.add_argument("out"); ap.add_argument("--accept", default="0.3,0.5,0.7"); a = ap.parse_args()
sys.path.insert(0, S + "/lingbot-world-v2"); from specwm.critic_model import CriticScorer
res = json.load(open(os.path.join(a.critic_dir, "result.json"))); hold = set(res["holdout"]); sc = CriticScorer(os.path.join(a.critic_dir, "critic.pt"), device="cuda:0")
crit = {0: [], 1: []}; e3 = {0: [], 1: []}; per_task = {}
for f in sorted(glob.glob(SH + "/*/shadow.pt")):
    d = torch.load(f, map_location="cpu")
    if d["task"] not in hold: continue
    prev = None
    for r in d["rows"]:
        for lab, key in ((0, "x_draft"), (1, "x_target")):
            l, _ = sc(r[key], prev, r["cam"]); crit[lab].append(l); e = r["e3"]["draft" if lab == 0 else "target"]; e3[lab].append(float(np.log(np.mean(e["e_target"]) / np.mean(e["e_draft"]))))
            per_task.setdefault(d["task"], {0: [], 1: []})[lab].append(l)
        prev = r["x_target"] if r["committed"] == "target" else r["x_draft"]
out = dict(holdout=sorted(hold), n_draft=len(crit[0]), n_target=len(crit[1]), critic={}, e3={})
for acc in [float(x) for x in a.accept.split(",")]:
    t_c = float(np.quantile(crit[0], 1 - acc)); t_e = float(np.quantile(e3[0], acc))          # critic: accept logit >= tau ; e3: accept log(eT/eD) <= tau
    out["critic"]["%.2f" % acc] = dict(tau=round(t_c, 4), draft_pass=float(np.mean(np.array(crit[0]) >= t_c)), target_pass=float(np.mean(np.array(crit[1]) >= t_c)))
    out["e3"]["%.2f" % acc] = dict(tau=round(t_e, 4), draft_pass=float(np.mean(np.array(e3[0]) <= t_e)), target_pass=float(np.mean(np.array(e3[1]) <= t_e)))
    print("accept %.2f of draft chunks: critic tau %+.3f (target chunks pass %.2f) | e3 tau %+.3f (target chunks pass %.2f)" % (acc, t_c, out["critic"]["%.2f" % acc]["target_pass"], t_e, out["e3"]["%.2f" % acc]["target_pass"]))
from sklearn.metrics import roc_auc_score
y = [0] * len(crit[0]) + [1] * len(crit[1]); print("held-out AUC: critic %.3f | ③ %.3f" % (roc_auc_score(y, crit[0] + crit[1]), roc_auc_score(y, [-v for v in e3[0] + e3[1]])))
for t, v in sorted(per_task.items()): print("  %-28s draft mean %+.2f  target mean %+.2f" % (t, np.mean(v[0]), np.mean(v[1])))
json.dump(out, open(a.out, "w"), indent=1)
