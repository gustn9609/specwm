"""Thresholds for the drift critic as a CHUNK verifier: score every chunk of the labelled clips (exp/drift_lat) on the held-out seeds; tau such that
a given fraction of chunks of CLEAN clips pass (logit <= tau); report the fraction of chunks of SEVERE clips that would be flagged, per arm family.
usage: CUDA_VISIBLE_DEVICES=g drift_calibrate.py CRITIC.pt OUT.json [--pass 0.8,0.9,0.95] [--seeds 109,...]"""
import os, sys, json, glob, argparse, collections, numpy as np, torch
S = "/home/hk28965/specwm"; DL = S + "/exp/drift_lat"
ap = argparse.ArgumentParser(); ap.add_argument("critic"); ap.add_argument("out"); ap.add_argument("--pass", dest="pas", default="0.8,0.9,0.95"); ap.add_argument("--seeds", default="109,110,111,112,113,114,115,116,9999"); a = ap.parse_args()
sys.path.insert(0, S + "/lingbot-world-v2"); from specwm.critic_model import CriticScorer
sc = CriticScorer(os.path.abspath(a.critic), device="cuda:0"); seeds = {int(x) for x in a.seeds.split(",")}
cams = {os.path.basename(f)[4:-3]: torch.load(f, map_location="cpu")["cam"] for f in glob.glob(DL + "/cam_*.pt")}
clean, severe, per = [], [], collections.defaultdict(lambda: [[], []])
for f in sorted(glob.glob(DL + "/*__s*.pt")):
    d = torch.load(f, map_location="cpu")
    if d["seed"] not in seeds: continue
    lat = d["lat"]; prev = None; ls = []
    for k in range(lat.shape[1] // 3):
        x = lat[:, 3 * k:3 * k + 3].float(); l, _ = sc(x, prev, cams[d["task"]][k]); ls.append(l); prev = x
    (severe if d["severe"] else clean).extend(ls); per[d["arm"]][int(d["severe"])].extend(ls)
clean, severe = np.array(clean), np.array(severe); out = dict(n_clean_chunks=len(clean), n_severe_chunks=len(severe), tau={})
print("chunks: clean %d (logit mean %+.2f) severe %d (mean %+.2f)" % (len(clean), clean.mean(), len(severe), severe.mean()))
for p in [float(x) for x in a.pas.split(",")]:
    t = float(np.quantile(clean, p)); out["tau"]["%.2f" % p] = dict(tau=round(t, 3), clean_pass=float(np.mean(clean <= t)), severe_flagged=float(np.mean(severe > t)))
    print("clean-chunk pass %.2f: tau %+.3f -> chunks of severe clips flagged %.2f" % (p, t, np.mean(severe > t)))
for arm, (c, s) in sorted(per.items()):
    if c and s: print("  %-34s clean chunks %4d mean %+.2f | severe chunks %4d mean %+.2f" % (arm, len(c), np.mean(c), len(s), np.mean(s)))
json.dump(out, open(os.path.abspath(a.out), "w"), indent=1)
