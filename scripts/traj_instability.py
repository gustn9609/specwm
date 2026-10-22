"""Training-free trajectory signals of the 4-step draft vs human drift labels: per chunk, r_k = ||x0hat_k - x0hat_3|| / ||x0hat_3|| (k = 0,1,2) and
successive change d_k = ||x0hat_{k+1} - x0hat_k|| / ||x0hat_3||; clip score = max or mean over chunks. Prints AUC (severe / drift labels)."""
import glob, json, numpy as np, torch
from sklearn.metrics import roc_auc_score
S = "/home/hk28965/specwm"; LAB = json.load(open(S + "/exp/kveval/traj_labels.json")); F = {}; Y = {}
for f in sorted(glob.glob(S + "/lingbot-world-v2/specwm_out/draft_traj/*.pt")):
    d = torch.load(f, map_location="cpu"); k = "%s|%s|%d" % (d["arm"], d["task"], d["seed"])
    if k not in LAB: continue
    t = d["traj"].float(); feats = {}
    ch = [t[:, :, 3 * c:3 * c + 3] for c in range(13)]                                  # [4,16,3,h,w] per chunk
    nrm = [x[3].norm() for x in ch]
    for j in range(3):
        r = np.array([float((x[j] - x[3]).norm() / n) for x, n in zip(ch, nrm)]); feats["r%d_max" % j] = r.max(); feats["r%d_mean" % j] = r.mean()
        dd = np.array([float((x[j + 1] - x[j]).norm() / n) for x, n in zip(ch, nrm)]); feats["d%d_max" % j] = dd.max(); feats["d%d_mean" % j] = dd.mean()
    # temporal: change of the step-0 layout between consecutive chunks' last frames vs the final latent's
    F[k] = feats; Y[k] = LAB[k]
keys = sorted(F); print("clips", len(keys))
for name in F[keys[0]]:
    s = [F[k][name] for k in keys]
    print("%-8s AUC severe %.3f  drift %.3f" % (name, roc_auc_score([Y[k]["severe"] for k in keys], s), roc_auc_score([Y[k]["drift"] for k in keys], s)))
import collections
G = collections.defaultdict(list)
for k in keys: G[k.split("|")[0]].append(k)
print("within-arm AUC of -r0_mean (higher = more drift):")
for arm, ks in sorted(G.items()):
    ys = [Y[k]["severe"] for k in ks]
    if 0 < sum(ys) < len(ys): print("  %-11s n %3d neg %2d AUC %.3f" % (arm, len(ks), len(ys) - sum(ys), roc_auc_score(ys, [-F[k]["r0_mean"] for k in ks])))
json.dump(F, open(S + "/exp/kveval/traj_instability.json", "w"))
