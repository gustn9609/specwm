"""Step 1a analysis: do VGGT geometry signals predict human-judged drift? Joins exp/geo/*.json with the blind labels stored in exp/drift_lat/*.pt
(severe, drift, cats). Reports AUC of clip-level features for: severe, teleport-only (any clip whose majority categories contain 'teleport'), and
new_object; overall, per task, within the arms with enough labels; compares with the drift critic (exp/kveval/critic scores if present) and HSV.
usage: geo_auc.py [--min-arm 12]"""
import os, glob, json, argparse, collections, numpy as np, torch
from sklearn.metrics import roc_auc_score
S = "/home/hk28965/specwm"; ap = argparse.ArgumentParser(); ap.add_argument("--min-arm", type=int, default=12); a = ap.parse_args()
rows = []
for f in glob.glob(S + "/exp/geo/*.json"):
    g = json.load(open(f)); lp = "%s/exp/drift_lat/%s__%s__s%d.pt" % (S, g["task"], g["arm"], g["seed"])
    if not os.path.exists(lp): continue
    d = torch.load(lp, map_location="cpu"); rj = np.array(g["rot_jump"]); rp = np.array(g["reproj"]); va = np.array(g["valid"]); ce = np.array(g["cam_err"]); cf = np.array(g["conf"])
    hsv = max(abs(v - 1) for s_, v in d["hsv"])
    feat = dict(rotjump_max=rj.max(), rotjump_p90=np.percentile(rj, 90), rotjump_jumpiness=(rj.max() / (np.median(rj) + 1e-3)),
                reproj_max=rp.max(), reproj_p90=np.percentile(rp, 90), reproj_mean=rp.mean(), reproj_step=np.max(np.diff(rp)),
                valid_min=-va.min(), valid_mean=-va.mean(), conf_min=-cf.min(), conf_drop=-(cf.min() / (np.median(cf) + 1e-6)),
                camerr_mean=ce.mean(), camerr_half=ce[: len(ce) // 2].mean(), loop_dino=g["loop_dino"], hsv_value=hsv)
    rows.append(dict(task=g["task"], arm=g["arm"], seed=g["seed"], severe=bool(d["severe"]), drift=bool(d["drift"]), teleport="teleport" in d.get("cats", []), newobj="new_object" in d.get("cats", []), batch=d.get("batch"), n_raters=d.get("n_raters", 3), **feat))
print("clips with labels:", len(rows), "| severe %.2f teleport %.2f new_object %.2f" % (np.mean([r["severe"] for r in rows]), np.mean([r["teleport"] for r in rows]), np.mean([r["newobj"] for r in rows])))
FEATS = [k for k in rows[0] if k not in ("task", "arm", "seed", "severe", "drift", "teleport", "newobj", "batch", "n_raters")]
def auc(rs, lab, f):
    y = [r[lab] for r in rs]
    return roc_auc_score(y, [r[f] for r in rs]) if 0 < sum(y) < len(y) else float("nan")
for lab in ("severe", "teleport", "newobj"):
    print("\n== label: %s" % lab)
    print("%-18s %7s %7s %7s | %s" % ("feature", "all", "jar", "castle", "within-arm mean AUC (arms with >= %d clips, both classes)" % a.min_arm))
    by_arm = collections.defaultdict(list)
    for r in rows: by_arm[r["arm"]].append(r)
    arms = [k for k, v in by_arm.items() if len(v) >= a.min_arm and 0 < sum(x[lab] for x in v) < len(v)]
    for f in FEATS:
        wa = [auc(by_arm[k], lab, f) for k in arms]
        print("%-18s %7.3f %7.3f %7.3f | %.3f over %d arms" % (f, auc(rows, lab, f), auc([r for r in rows if "jar" in r["task"]], lab, f), auc([r for r in rows if "castle" in r["task"]], lab, f), np.nanmean(wa) if wa else float("nan"), len(arms)))
json.dump(rows, open(S + "/exp/kveval/geo_features.json", "w"), default=float)
