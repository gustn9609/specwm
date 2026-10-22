"""[baselines/sf1p3b] Threshold calibration of the training-free verifier from --tf-calib dumps (calib.pt = per-chunk dicts written by
specwm/early_exit_tf.py calib_scores: gap [K-1, L] = oracle rel_k vs the chunk's own 4-step output, red_draft [L], stab_draft [K-2, L] = the
smoothed training-free scores on the dense drafts, draft-sigma normalisation; *_none = raw latent units).
Rule (task 10/17, arm B): tau = the score quantile that matches the ORACLE acceptance rate P(rel_k <= eps) per verification step; tau_red from
k = 1, tau_stab from the pooled k = 2 and k = 3 tokens (one tau_stab is applied at both steps, as in the paper). Per scene the programs (P1, P2)
are pooled; the frozen value is the MEDIAN over scenes. Also reported: AUROC, acceptance rate / precision at the published and at the frozen
thresholds, and a sequential mean_steps estimate (step 1 accept -> else step 2 -> else step 3 -> else 4) on the dense scores (an approximation:
in a real run frozen tokens change the later drafts).
usage: python calib_analyze.py --runs DIR [DIR ...] (each = <out>/<task>/<arm>_s<seed> with calib.pt) [--eps 0.24] [--pub 0.375,0.300] [--out calib_summary.json]"""
import os, re, json, argparse, numpy as np, torch
ap = argparse.ArgumentParser(); ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--eps", type=float, default=0.24); ap.add_argument("--pub", default="0.375,0.300")
ap.add_argument("--out", default=""); ap.add_argument("--norm", default="draft", choices=["draft", "none"]); a = ap.parse_args(); EPS = a.eps; PUB = tuple(float(x) for x in a.pub.split(","))

def load(run):
    CAL = torch.load(run + "/calib.pt", map_location="cpu"); L = json.load(open(run + "/loop.json"))
    gap = torch.cat([c["gap"] for c in CAL], 1).float().numpy()                                 # [3, N]
    red = torch.cat([c["red_" + a.norm] for c in CAL]).float().numpy()                          # [N]
    stab = torch.cat([c["stab_" + a.norm] for c in CAL], 1).float().numpy()                     # [2, N]
    m = re.match(r"sf1p3b__(.+)__(P\d|.+)$", L["task"]); scene, prog = (m.group(1), m.group(2)) if m else (L["task"], "?")
    return dict(task=L["task"], scene=scene, prog=prog, gap=gap, red=red, stab=stab, chunks=len(CAL), mean_steps_run=L["mean_steps"], tau_run=(L.get("tau_red"), L.get("tau_stab")),
                err_run=[c[-1] for c in L["per_chunk_commit_hist"]])

def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if 0 < y.mean() < 1 else None

def stats(gap, red, stab):
    """per-step oracle rate + matched tau; pooled stab; AUROCs"""
    r = {}; y1 = gap[0] > EPS; ok = np.isfinite(red)
    r["k1"] = dict(n=int(ok.sum()), oracle_acc=float((~y1[ok]).mean()), tau_match=float(np.quantile(red[ok], (~y1[ok]).mean())), auroc=auroc(y1[ok], red[ok]), score_median=float(np.median(red[ok])))
    for k in (2, 3):
        y = gap[k - 1] > EPS; s = stab[k - 2]; r["k%d" % k] = dict(n=int(len(s)), oracle_acc=float((~y).mean()), tau_match=float(np.quantile(s, (~y).mean())), auroc=auroc(y, s), score_median=float(np.median(s)))
    yp = np.concatenate([gap[1] > EPS, gap[2] > EPS]); sp = np.concatenate([stab[0], stab[1]])
    r["k23_pooled"] = dict(n=int(len(sp)), oracle_acc=float((~yp).mean()), tau_match=float(np.quantile(sp, (~yp).mean())), auroc=auroc(yp, sp))
    return r

def at_tau(gap, red, stab, tr, ts):
    """acceptance rate / precision per step at (tau_red, tau_stab) + sequential mean_steps estimate and the mean oracle gap of the accepted tokens"""
    r = {}; N = gap.shape[1]; step = np.full(N, 4); ok = np.isfinite(red)
    a1 = ok & (red <= tr); step[a1] = 1
    a2 = (step == 4) & (stab[0] <= ts); step[a2] = 2
    a3 = (step == 4) & (stab[1] <= ts); step[a3] = 3
    for k, acc in ((1, a1), (2, a2), (3, a3)):
        y = gap[k - 1] > EPS; marg = ok & (red <= tr) if k == 1 else (stab[k - 2] <= ts)
        r["k%d" % k] = dict(acc_rate_marginal=float(marg.mean()), precision_marginal=float((~y[marg]).mean()) if marg.any() else None, acc_rate_sequential=float(acc.mean()), precision_sequential=float((~y[acc]).mean()) if acc.any() else None)
    r["mean_steps_est"] = float(step.mean()); r["hist_est"] = [int((step == k).sum()) for k in (1, 2, 3, 4)]
    g = np.where(step == 4, 0.0, gap[np.minimum(step, 3) - 1, np.arange(N)]); r["mean_gap_accepted"] = float(g[step < 4].mean()) if (step < 4).any() else None
    r["frac_accepted_bad"] = float(((step < 4) & (g > EPS)).mean())
    return r

runs = [load(r) for r in a.runs]; print("loaded %d runs: %s" % (len(runs), [r["task"] for r in runs]))
OUT = dict(eps=EPS, norm=a.norm, published=dict(tau_red=PUB[0], tau_stab=PUB[1]), per_run={}, per_scene={}, frozen={})
for r in runs:
    OUT["per_run"][r["task"]] = dict(scene=r["scene"], prog=r["prog"], chunks=r["chunks"], tokens=int(r["gap"].shape[1]), mean_steps_run=r["mean_steps_run"], tau_run=r["tau_run"], err_run_mean=float(np.nanmean(r["err_run"])), stats=stats(r["gap"], r["red"], r["stab"]), at_published=at_tau(r["gap"], r["red"], r["stab"], *PUB))
scenes = sorted(set(r["scene"] for r in runs))
for sc in scenes:
    rs = [r for r in runs if r["scene"] == sc]; gap = np.concatenate([r["gap"] for r in rs], 1); red = np.concatenate([r["red"] for r in rs]); stab = np.concatenate([r["stab"] for r in rs], 1)
    st = stats(gap, red, stab); OUT["per_scene"][sc] = dict(progs=[r["prog"] for r in rs], tokens=int(gap.shape[1]), stats=st, tau_red=st["k1"]["tau_match"], tau_stab=st["k23_pooled"]["tau_match"], tau_stab_k2=st["k2"]["tau_match"], tau_stab_k3=st["k3"]["tau_match"])
tr = float(np.median([OUT["per_scene"][s]["tau_red"] for s in scenes])); ts = float(np.median([OUT["per_scene"][s]["tau_stab"] for s in scenes]))
OUT["frozen"] = dict(tau_red=round(tr, 4), tau_stab=round(ts, 4), rule="median over scenes of the per-scene (P1+P2 pooled) score quantile at the oracle acceptance rate P(rel_k <= eps); tau_stab from the pooled k=2,3 tokens", scenes=scenes)
gap = np.concatenate([r["gap"] for r in runs], 1); red = np.concatenate([r["red"] for r in runs]); stab = np.concatenate([r["stab"] for r in runs], 1)
OUT["pooled_all"] = dict(tokens=int(gap.shape[1]), stats=stats(gap, red, stab), at_published=at_tau(gap, red, stab, *PUB), at_frozen=at_tau(gap, red, stab, tr, ts))
for r in runs: OUT["per_run"][r["task"]]["at_frozen"] = at_tau(r["gap"], r["red"], r["stab"], tr, ts)
# ---- print
print("\nper run (oracle acceptance rate | matched tau | AUROC) for k=1 / k=2 / k=3, and k=2+3 pooled tau:")
for t, v in OUT["per_run"].items():
    s = v["stats"]; print("  %-40s chunks %2d  k1 %.3f|%.3f|%s  k2 %.3f|%.3f|%s  k3 %.3f|%.3f|%s  k23 %.3f  err_run %.3f" % (t, v["chunks"], s["k1"]["oracle_acc"], s["k1"]["tau_match"], "%.2f" % s["k1"]["auroc"] if s["k1"]["auroc"] else "-", s["k2"]["oracle_acc"], s["k2"]["tau_match"], "%.2f" % s["k2"]["auroc"] if s["k2"]["auroc"] else "-", s["k3"]["oracle_acc"], s["k3"]["tau_match"], "%.2f" % s["k3"]["auroc"] if s["k3"]["auroc"] else "-", s["k23_pooled"]["tau_match"], v["err_run_mean"]))
print("per scene (P1+P2 pooled): tau_red | tau_stab (k2+3) [k2, k3]")
for sc, v in OUT["per_scene"].items(): print("  %-28s tau_red %.4f  tau_stab %.4f  [k2 %.4f, k3 %.4f]  oracle acc k1 %.3f k2 %.3f k3 %.3f" % (sc, v["tau_red"], v["tau_stab"], v["tau_stab_k2"], v["tau_stab_k3"], v["stats"]["k1"]["oracle_acc"], v["stats"]["k2"]["oracle_acc"], v["stats"]["k3"]["oracle_acc"]))
print("FROZEN (median over scenes): tau_red %.4f  tau_stab %.4f" % (tr, ts))
for nm, key in (("published", "at_published"), ("frozen", "at_frozen")):
    v = OUT["pooled_all"][key]; print("pooled all tokens @ %s: step-1 acc (marginal) %.3f prec %.3f | seq acc k1/k2/k3 %.3f/%.3f/%.3f | mean_steps_est %.3f | frac accepted with gap>eps %.3f" % (nm, v["k1"]["acc_rate_marginal"], v["k1"]["precision_marginal"] or 0, v["k1"]["acc_rate_sequential"], v["k2"]["acc_rate_sequential"], v["k3"]["acc_rate_sequential"], v["mean_steps_est"], v["frac_accepted_bad"]))
if a.out: json.dump(OUT, open(a.out, "w"), indent=1); print("wrote", a.out)
